import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from tabulate import tabulate
import os

# Add these imports for pybullet
import pybullet as p
import pybullet_data
import tempfile
from rcl_interfaces.srv import GetParameters
from torque_controlled_arm_interfaces.srv import GetNextTarget

import xml.etree.ElementTree as ET
from io import StringIO

def clean_urdf_for_pybullet(urdf_xml_str: str) -> str:
    """
    Remove unsupported ROS2/Gazebo tags from URDF XML string for PyBullet compatibility.
    """

    # Parse URDF XML
    it = ET.iterparse(StringIO(urdf_xml_str))
    for _, el in it:
        # Remove namespaces in tag names (if any)
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]

    root = it.root

    # Tags to remove completely
    remove_tags = ['transmission', 'gazebo', 'plugin', 'actuator', 'ros2_control']

    def recursive_clean(element):
        to_remove = []
        for child in element:
            if child.tag in remove_tags:
                to_remove.append(child)
            else:
                recursive_clean(child)
        for child in to_remove:
            element.remove(child)

    recursive_clean(root)

    # Convert cleaned XML tree back to string
    cleaned_urdf = ET.tostring(root, encoding='unicode')

    return cleaned_urdf


class SixDOFPIDController(Node):
    def __init__(self):
        super().__init__('six_dof_pid_controller')

        self.robot_cli = self.create_client(GetParameters, '/controller_manager/get_parameters')

        while not self.robot_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /controller_manager/get_parameters service...')

        req = GetParameters.Request()
        req.names = ['robot_description']

        future = self.robot_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        urdf_xml = None
        if result is not None and result.values:
            # result.values is a list of ParameterValue messages, access string_value
            urdf_xml = clean_urdf_for_pybullet(result.values[0].string_value)

        if not urdf_xml:
            self.get_logger().error("Failed to get robot_description parameter from /controller_manager")
            raise RuntimeError("Missing robot_description")

        # Save URDF XML string to a temporary file because pybullet needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as urdf_file:
            urdf_file.write(urdf_xml.encode('utf-8'))
            urdf_path = urdf_file.name
        
        p.connect(p.DIRECT) 
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = [i for i in range(self.num_joints) if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            '/torque_arm_controller/commands',
            1
        )  

        self.cli = self.create_client(GetNextTarget, '/get_next_target')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /get_next_target not available, waiting...')

        self.kpNG = [500.0, 2400.0, 2400.0, 500.0, 300.0, 15.0]
        self.kpWG = [500.0, 1600.0, 1600.0, 350.0, 200.0, 15.0]

        self.ki = [40.0, 1000.0, 1000.0, 80.0, 100.0, 1.5]
        self.kd = [40.0, 150, 150, 10, 10, 1]

        self.target_velocity = [0.0 for _ in range(6)]
        self.target_index = 0
        self.interpolation_steps = 30
        self.joint_steps = [0] * 6
        self.step_targets = [[0.0] * self.interpolation_steps for _ in range(6)]
        self.target_angles = [0.0] * 6  

        self.integral = [0.0] * 6
        self.prev_time = time.time()

        self.reach_counter = 0
        self.done = False
        self.global_start_time = time.time()

        self.pos_tol_rough_val = 0.125
        self.pos_tol_final_val = 0.2
        self.vel_tol_final_val = 0.05

        self.min_joint_step = 0
        self.latency_threshold = self.interpolation_steps // 2

        self.all_times = []
        self.all_positions = [[] for _ in range(6)]
        self.all_velocities = [[] for _ in range(6)]
        self.all_targets = [[] for _ in range(6)]
        self.all_index = [[] for _ in range(6)]

        self.current_main_target = [0.0] * 6  # holds the actual target from the service

        self.prepare_interpolation()

    def call_get_next_target(self):
        req = GetNextTarget.Request()
        future = self.cli.call_async(req)
        future.add_done_callback(self.handle_target_response)

    def handle_target_response(self, future):
        try:
            response = future.result()
            next_target = list(response.joint_positions)

            if self.is_target_nan(next_target):
                self.get_logger().info("No more targets. Stopping interpolation.")
                self.done = True

                self.get_logger().info("All targets reached. Plotting results...")
                self.plot_all()
                return

            self.current_main_target = next_target

            self.joint_steps = [0] * 6
            current = self.target_angles

            for joint in range(6):
                self.step_targets[joint] = [
                    (1 - ((step + 1) / self.interpolation_steps)) * current[joint] +
                    ((step + 1) / self.interpolation_steps) * next_target[joint]
                    for step in range(self.interpolation_steps)
                ]
                self.target_angles[joint] = self.step_targets[joint][0]

            self.get_logger().info(f"Starting interpolation to target {self.target_index}: {next_target}")

        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")


    def is_target_nan(self, target):
        return all(math.isnan(val) for val in target)

    def prepare_interpolation(self):
        self.call_get_next_target()

    def joint_state_callback(self, msg):
        if self.done:
            return

        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0:
            return

        positions = np.array(msg.position[:6])
        velocities = msg.velocity[:6]
        torques = []

        final_target_reached = True
        self.all_times.append(current_time - self.global_start_time)
        errors = []
        
        # Compute Jacobian for current joint positions:
        joint_positions_list = positions.tolist()
        for i, q in zip(self.joint_indices, joint_positions_list):
            p.resetJointState(self.robot, i, q)

        zero_vec = [0.0] * len(self.joint_indices)
        jac_t, _ = p.calculateJacobian(
            self.robot,
            self.joint_indices[-1],
            localPosition=[0, 0, 0],
            objPositions=joint_positions_list,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jac_t = np.array(jac_t)
        dz_dj = jac_t[2, :]  # z-axis row of translational Jacobian

        link_states = [(p.getLinkState(self.robot, self.joint_indices[-1], computeForwardKinematics=True)) for j in self.joint_indices] # the fixed link is skipped self.joint_indices begin with 1

        self.min_joint_step = min(self.joint_steps)

        for i in range(6):
            error = self.target_angles[i] - positions[i]
            self.integral[i] += error * dt
            d_error = self.target_velocity[i] - velocities[i]

            kp = self.kpNG[i] if dz_dj[i] > 0 else self.kpWG[i]
            pi = kp * error + self.ki[i] * self.integral[i]
            limit = 2 * math.pi
            d = self.kd[i] * max(-limit, min(d_error, limit))
            torque = pi + d
            torques.append(torque)

            self.all_positions[i].append(positions[i])
            self.all_velocities[i].append(velocities[i])
            self.all_targets[i].append(self.current_main_target[i])
            self.all_index[i].append(self.target_index)

            pos_tol_rough = abs(error) < self.pos_tol_rough_val
            pos_tol_final = abs(error) < self.pos_tol_final_val
            vel_tol_final = abs(velocities[i]) < self.vel_tol_final_val

            if self.joint_steps[i] < self.interpolation_steps - 1:
                if pos_tol_rough and self.joint_steps[i] - self.min_joint_step < self.latency_threshold:
                    self.joint_steps[i] += 1
                    self.target_angles[i] = self.step_targets[i][self.joint_steps[i]]
                    self.get_logger().info(f"Joint {i} moving to step {self.joint_steps[i]} for target {self.target_index}")


                final_target_reached = False
            else:
                if not (pos_tol_final and vel_tol_final):
                    final_target_reached = False

            errors.append(error)

        self.prev_time = current_time

        cmd = Float64MultiArray()
        cmd.data = torques

        table_data = []
        for i in range(6):
            step_info = "TR" if self.joint_steps[i] == self.interpolation_steps - 1 else str(self.joint_steps[i])
            table_data.append([
                i,
                f"{errors[i]:+.3f}",
                f"{velocities[i]:+.3f}",
                f"{torques[i]:+.3f}",
                f"{self.target_angles[i]:+.3f}",
                step_info,
                f"{dz_dj[i]:+.4f}"   # Add dz/dj for joint i here
            ])

        headers = ["Joint", "Error (rad)", "Velocity (rad/s)", "Torque (Nm)", "Target (rad)", "Step", "dz/dj"]
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Target Joint Angles: {[f'{ta:+.2f}' for ta in self.target_angles]}")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print("update rate", 1 / dt)

        self.command_pub.publish(cmd)

        if final_target_reached:
            self.reach_counter += 1
        else:
            self.reach_counter = 0

        if self.reach_counter >= 10:
            self.reach_counter = 0
            self.target_index += 1

            self.prepare_interpolation()


    def reset_pid_state(self):
        self.integral = [0.0] * 6
        self.prev_time = time.time()

    def plot_all(self):
        for i_plt in range(6):
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (rad)', color='blue')
            ax1.plot(self.all_times, self.all_positions[i_plt], label='Position (rad)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            for i, t in enumerate(self.all_times):
                if i == 0 or self.all_targets[i_plt][i] != self.all_targets[i_plt][i - 1]:
                    ax1.axhline(y=self.all_targets[i_plt][i], linestyle='--', color='green', alpha=0.5)
                    ax1.text(t, self.all_targets[i_plt][i] + 0.05, f'target({self.all_index[i_plt][i]}):{self.all_targets[i_plt][i]:.2f}', color='green')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Velocity (rad/s)', color='orange')
            ax2.plot(self.all_times, self.all_velocities[i_plt], label='Velocity (rad/s)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            plt.title('Full Trajectory Over All Targets')
            fig.tight_layout()
            plt.grid(True)
            plt.savefig(f'joint{i_plt}_pid_full_trajectory.png')
            plt.close()
            self.get_logger().info("Full trajectory saved as pid_full_trajectory.png")


def main(args=None):
    rclpy.init(args=args)
    node = SixDOFPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
