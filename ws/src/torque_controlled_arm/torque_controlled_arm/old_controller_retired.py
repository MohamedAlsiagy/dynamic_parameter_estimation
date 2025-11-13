import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from tabulate import tabulate
import os
import pybullet as p
import pybullet_data
import tempfile
from rcl_interfaces.srv import GetParameters
import xml.etree.ElementTree as ET
from io import StringIO
import random
import sys
from math import pi, nan
import csv
import yaml
import json
from pathlib import Path

# Utility function from pid.py
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
    
# Utility function from target_gen.py
def is_successive_or_same(link_a, link_b):
    return abs(link_a - link_b) <= 1

class CombinedControllerNode(Node):
    def __init__(self):
        super().__init__('combined_controller_node')

        self.declare_parameter("robot_name", "robot_default")
        self.declare_parameter("plot", False)

        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.plot = self.get_parameter("plot").get_parameter_value().bool_value

        # Load settings
        base_dir = os.path.dirname(os.path.dirname(__file__))
        settings_file_path = os.path.join(base_dir, "config", "settings.yaml")

        with open(settings_file_path, 'r') as f:
            self.settings = yaml.safe_load(f)

        self.num_trajectory_points_per_robot = self.settings['num_trajectory_points_per_robot']
        self.timeout_trajectory_point = self.settings['timeout_trajectory_point'] #TODO: also exist when update rate is slow , the robot is stuck
        self.dataset_output_dir = self.settings['dataset_output_dir']

        # PID Controller parameters (from pid.py)
        self.kpNG = [5.0, 100.0, 100.0, 3.0, 3.0, 1.0]
        self.kpWG = [5.0, 75.0, 75.0, 1.0, 1.0, 1.0]
        self.ki = [0.4, 10.0, 10.0, 0.10, 1.00, 0.10]
        self.kd = [0.4, 5.00, 5.00, 0.25, 0.25, 0.05]

        self.target_velocity = [0.0 for _ in range(6)]
        self.interpolation_steps = self.settings['interpolation_steps']
        self.integral = [0.0] * 6
        self.prev_time = time.time()
        self.reach_counter = 0
        self.done = False
        self.global_start_time = time.time()
        self.pos_tol_rough_val = self.settings['intermediate_position_tolerance']
        self.pos_tol_final_val = self.settings['target_position_tolerance']
        self.vel_tol_final_val = self.settings['target_velocity_tolerance']
        self.min_joint_step = 0
        self.latency_threshold = self.interpolation_steps // 2

        # Data logging (from pid.py, adapted for CSV)
        self.all_times = []
        self.all_positions = [[] for _ in range(6)]
        self.all_velocities = [[] for _ in range(6)]
        self.all_targets = [[] for _ in range(6)]
        self.all_index = [[] for _ in range(6)]
        self.all_torques = [[] for _ in range(6)]
        self.all_link_positions = [[] for _ in range(6)]
        self.all_link_velocities = [[] for _ in range(6)]
        self.all_link_accelerations = [[] for _ in range(6)]
        self.all_target_points_followed = [] # To store the actual target points followed

        # Target Generation parameters (from target_gen.py)
        self.valid_configs = []
        self.current_target_index = 0
        self.joint_steps = [0] * 6
        self.step_targets = [[0.0] * self.interpolation_steps for _ in range(6)]
        self.target_angles = [0.0] * 6
        self.current_main_target = [0.0] * 6

        self.failed_count = 0

        # PyBullet setup (from pid.py and target_gen.py)
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
            urdf_xml = clean_urdf_for_pybullet(result.values[0].string_value)

        if not urdf_xml:
            self.get_logger().error("Failed to get robot_description parameter from /controller_manager")
            raise RuntimeError("Missing robot_description")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as urdf_file:
            urdf_file.write(urdf_xml.encode('utf-8'))
            self.urdf_path = urdf_file.name
        
        p.connect(p.DIRECT) 
        self.robot_pb = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.num_joints_pb = p.getNumJoints(self.robot_pb)
        self.joint_indices_pb = [i for i in range(self.num_joints_pb) if p.getJointInfo(self.robot_pb, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

        if len(self.joint_indices_pb) < 6:
            self.get_logger().error("Robot must have at least 6 movable joints.")
            sys.exit(1)

        # ROS 2 Subscriptions and Publishers
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

        self.trajectory_termination_pub = self.create_publisher(Int32, '/generation_state', 10)

        # Start target generation
        self.get_logger().info(f"starting generating {self.num_trajectory_points_per_robot} random joint configurations...")
        self.check_random_configs()
        self.prepare_interpolation()

        self.update_rate = 500
        self.print_every = 25
        self.update_count = 0

    def check_random_configs(self):
        base_pose = self.compute_tip_position([0.0] * len(self.joint_indices_pb))
        ref_pose = self.compute_tip_position([0.0, pi/2, 0.0, 0.0, 0.0, 0.0])
        ref_dist = np.linalg.norm(np.array(ref_pose) - np.array(base_pose))
        threshold_distance = 0.25 * ref_dist

        attempts = 0
        last_valid_config = [0.0] * len(self.joint_indices_pb)

        while len(self.valid_configs) < self.num_trajectory_points_per_robot:
            attempts += 1
            joint_positions = [random.uniform(-pi, pi) for _ in self.joint_indices_pb]

            for i in [1,2]: # Specific to the original robot, might need generalization
                joint_positions[i] = last_valid_config[i] + random.uniform(-pi, pi) / 2 

            all_interp_z_valid = True
            for step in range(self.interpolation_steps):
                interp_joint_positions = [
                    (1 - (step + 1) / self.interpolation_steps) * last_valid_config[j] +
                    ((step + 1) / self.interpolation_steps) * joint_positions[j]
                    for j in range(len(joint_positions))
                ]
                for i, q in zip(self.joint_indices_pb, interp_joint_positions):
                    p.resetJointState(self.robot_pb, i, q)

                for i in range(3, p.getNumJoints(self.robot_pb)):
                    link_pos = p.getLinkState(self.robot_pb, i)[4]
                    if link_pos[2] < 0.6:
                        all_interp_z_valid = False
                        break
                if not all_interp_z_valid:
                    break

            if not all_interp_z_valid:
                continue

            for i, q in zip(self.joint_indices_pb, joint_positions):
                p.resetJointState(self.robot_pb, i, q)

            contacts = p.getClosestPoints(bodyA=self.robot_pb, bodyB=self.robot_pb, distance=0.001)
            real_collisions = [
                c for c in contacts
                if c[3] != c[4] and not is_successive_or_same(c[3], c[4])
            ]
            if real_collisions:
                continue

            tip_pos = self.compute_tip_position(joint_positions)
            if np.linalg.norm(np.array(tip_pos) - np.array(base_pose)) < threshold_distance:
                continue

            self.valid_configs.append(joint_positions)
            last_valid_config = joint_positions

        self.get_logger().info(f"Generated {len(self.valid_configs)} valid configuration out of {attempts} attempts")


    def compute_tip_position(self, joint_values):
        for i, q in zip(self.joint_indices_pb, joint_values):
            p.resetJointState(self.robot_pb, i, q)
        link_state = p.getLinkState(self.robot_pb, self.joint_indices_pb[-1])
        return link_state[4]

    def prepare_interpolation(self):
        if self.current_target_index >= len(self.valid_configs):
            self.get_logger().info("No more targets. Stopping simulation.")
            self.done = True
            self.save_robot_dataset()

            if self.plot:
                self.plot_all()

            msg = Int32()
            msg.data = 0 # Indicate completion
            self.trajectory_termination_pub.publish(msg)
            return

        next_target = self.valid_configs[self.current_target_index]
        self.all_target_points_followed.append(next_target) # Store the target point
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

        self.get_logger().info(f"Starting interpolation to target {self.current_target_index}: {next_target}")
        self.reset_pid_state()

    def joint_state_callback(self, msg):
        
        if self.done:
            return
        
        # Timeout check first
        if (time.time() - self.global_start_time) > self.timeout_trajectory_point:
            self.get_logger().error(f"Timeout reached for target {self.current_target_index}")
            self.current_target_index += 1
            self.failed_count += 1
            
            # Print progress update for timeout
            progress_data = {
                'type': 'progress',
                'current': self.current_target_index,
                'total': len(self.valid_configs),
                'completion': 0.0,  # Mark as failed
                'failed': self.failed_count
            }
            print(f"PROGRESS_JSON:{json.dumps(progress_data)}")
            
            self.prepare_interpolation()
            return

        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0:
            return

        self.update_rate += 0.2/dt
        self.update_rate /= (1 + 0.2)

        positions = np.array(msg.position[:6])
        velocities = msg.velocity[:6]
        torques = []
        errors = []

        final_target_reached = True
        
        # Compute Jacobian for current joint positions:
        joint_positions_list = positions.tolist()
        for i, q in zip(self.joint_indices_pb, joint_positions_list):
            p.resetJointState(self.robot_pb, i, q)

        zero_vec = [0.0] * len(self.joint_indices_pb)
        jac_t, _ = p.calculateJacobian(
            self.robot_pb,
            self.joint_indices_pb[-1],
            localPosition=[0, 0, 0],
            objPositions=joint_positions_list,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jac_t = np.array(jac_t)
        dz_dj = jac_t[2, :]  # z-axis row of translational Jacobian

        # Get link states for position, velocity, acceleration
        link_positions = []
        link_velocities = []
        link_accelerations = [] # PyBullet doesn't directly provide link accelerations, will be zeros

        for i in range(len(self.joint_indices_pb)):
            link_state = p.getLinkState(self.robot_pb, self.joint_indices_pb[i], computeLinkVelocity=1)
            link_positions.append(link_state[0]) # World position of the link frame
            link_velocities.append(link_state[6]) # World linear velocity of the link frame
            link_accelerations.append([0.0, 0.0, 0.0]) # Placeholder, PyBullet doesn't provide this directly

        self.min_joint_step = min(self.joint_steps)

        self.all_times.append(current_time - self.global_start_time)
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

            # Store data for CSV
            self.all_positions[i].append(positions[i])
            self.all_velocities[i].append(velocities[i])
            self.all_targets[i].append(self.current_main_target[i])
            self.all_index[i].append(self.current_target_index)
            self.all_torques[i].append(torque)
            self.all_link_positions[i].append(link_positions[i])
            self.all_link_velocities[i].append(link_velocities[i])
            self.all_link_accelerations[i].append(link_accelerations[i])

            pos_tol_rough = abs(error) < self.pos_tol_rough_val
            pos_tol_final = abs(error) < self.pos_tol_final_val
            vel_tol_final = abs(velocities[i]) < self.vel_tol_final_val

            if self.joint_steps[i] < self.interpolation_steps - 1:
                if pos_tol_rough and self.joint_steps[i] - self.min_joint_step < self.latency_threshold:
                    self.joint_steps[i] += 1
                    self.target_angles[i] = self.step_targets[i][self.joint_steps[i]]

                final_target_reached = False
            else:
                if not (pos_tol_final and vel_tol_final):
                    final_target_reached = False

            errors.append(error)

        self.prev_time = current_time

        cmd = Float64MultiArray()
        cmd.data = torques

        
        
        if self.update_count % self.print_every == 0:

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

            headers = ["Joint", "Εrror (rad)", "Velocity (rad/s)", "Torque (Nm)", "Target (rad)", "Step", "dz/dj"]
            # In joint_state_callback, modify the printed output:
            printed = "'''\n"
            printed += f"Robot name: {self.robot_name} , Target index: {self.current_target_index}/{len(self.valid_configs)}"
            printed += (f"\nTarget Joint Angles: {[f'{ta:+.2f}' for ta in self.target_angles]}\n")
            printed += (tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
            printed += (f"\nCompletion: {min(self.joint_steps)/self.interpolation_steps*100:.1f}%")
            printed += ("\nupdate rate " + str(round(self.update_rate)))
            printed += "\n'''"

            print(printed, end="")

            # In CombinedControllerNode.joint_state_callback() add:
            progress_data = {
                'type': 'progress',
                'current': self.current_target_index,
                'total': len(self.valid_configs),
                'completion': min(self.joint_steps) / self.interpolation_steps,
                'failed': self.failed_count  # Add self.failed_count to __init__
            }
            print(f"PROGRESS_JSON:{json.dumps(progress_data)}")  # Special marker for parsing

        self.command_pub.publish(cmd)

        if final_target_reached:
            self.reach_counter += 1
        else:
            self.reach_counter = 0

        if self.reach_counter >= 10: # If target reached for 10 consecutive steps
            self.reach_counter = 0
            self.current_target_index += 1
            self.prepare_interpolation()
        
        self.update_count += 1

    def reset_pid_state(self):
        self.integral = [0.0] * 6
        self.prev_time = time.time()

    def save_robot_dataset(self):
        robot_name = self.robot_name
        robot_output_dir = Path(self.dataset_output_dir) / robot_name
        robot_output_dir.mkdir(parents=True, exist_ok=True)

        self._save_trajectory_csv(robot_output_dir, robot_name)
        self._save_target_points_yaml(robot_output_dir, robot_name)
        self._save_dynamic_parameters_yaml(robot_output_dir, robot_name)

        failure_rate = self.failed_count / len(self.valid_configs) if len(self.valid_configs) > 0 else 0
        self.get_logger().info(f"Trajectory completion rate: {(1-failure_rate)*100:.1f}%")

    def _save_trajectory_csv(self, output_dir, robot_name):
        csv_file_path = output_dir / f'{robot_name}_trajectory_data.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['time', 'index']
            for i in range(6):
                header.extend([f'Joint_{i}_Position', f'Joint_{i}_Velocity', f'Joint_{i}_Acceleration', f'Joint_{i}_Torque'])
                header.extend([f'Link_{i}_Position_x', f'Link_{i}_Position_y', f'Link_{i}_Position_z',
                               f'Link_{i}_Velocity_x', f'Link_{i}_Velocity_y', f'Link_{i}_Velocity_z',
                               f'Link_{i}_Acceleration_x', f'Link_{i}_Acceleration_y', f'Link_{i}_Acceleration_z'])
            writer.writerow(header)

            num_samples = len(self.all_times)
            for i in range(num_samples):
                row = [self.all_times[i], self.all_index[0][i]]
                for j in range(6):
                    row.extend([
                        self.all_positions[j][i],
                        self.all_velocities[j][i],
                        0.0,
                        self.all_torques[j][i]
                    ])
                    row.extend(self.all_link_positions[j][i])
                    row.extend(self.all_link_velocities[j][i])
                    row.extend(self.all_link_accelerations[j][i])
                writer.writerow(row)
        self.get_logger().info(f'Trajectory data saved to {csv_file_path}')

    def _save_target_points_yaml(self, output_dir, robot_name):
        target_points_file_path = output_dir / f'{robot_name}_target_points.yaml'
        with open(target_points_file_path, 'w') as yamlfile:
            yaml.dump({f"T{idx}" : {f"J{j_idx}" : joint for j_idx , joint in enumerate(target)} for idx , target in enumerate(self.all_target_points_followed)}, yamlfile, indent=4)
        self.get_logger().info(f'Target points saved to {target_points_file_path}')

    def _save_dynamic_parameters_yaml(self, output_dir, robot_name):
        def extract_dynamic_parameters(urdf_path):
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            dynamic_parameters = {"joints": {}, "links": {}}

            for j_idx, joint in enumerate(root.findall("joint")):
                joint_name = joint.get("name")
                dynamics = joint.find("dynamics")
                if dynamics is not None:
                    damping = float(dynamics.get("damping", 0.0))
                    friction = float(dynamics.get("friction", 0.0))
                    dynamic_parameters["joints"][joint_name] = {
                        "idx": j_idx - 1,
                        "coulomb_friction_coeff": friction,
                        "viscous_friction_coeff": damping
                    }

            for idx, link in enumerate(root.findall("link")):
                link_name = link.get("name")
                inertial = link.find("inertial")
                if inertial is not None:
                    mass = float(inertial.find("mass").get("value", 0.0))
                    inertia = inertial.find("inertia")
                    ixx = float(inertia.get("ixx", 0.0))
                    ixy = float(inertia.get("ixy", 0.0))
                    ixz = float(inertia.get("ixz", 0.0))
                    iyy = float(inertia.get("iyy", 0.0))
                    iyz = float(inertia.get("iyz", 0.0))
                    izz = float(inertia.get("izz", 0.0))
                    com_xyz = {"xyz"[i]: float(x) for i, x in enumerate(inertial.find("origin").get("xyz", "0 0 0").split())}

                    dynamic_parameters["links"][link_name] = {
                        "idx": idx,
                        "mass": mass,
                        "ixx": ixx,
                        "ixy": ixy,
                        "ixz": ixz,
                        "iyy": iyy,
                        "iyz": iyz,
                        "izz": izz,
                        "com": com_xyz
                    }

            return dynamic_parameters

        if os.path.exists(self.urdf_path):
            params = extract_dynamic_parameters(self.urdf_path)
            dyn_param_file = output_dir / f"{robot_name}_dynamic_parameters.yaml"
            with open(dyn_param_file, "w") as f:
                yaml.dump(params, f, indent=4)
            self.get_logger().info(f"Dynamic parameters saved to {dyn_param_file}")

    def plot_all(self):
        output_dir = Path(self.dataset_output_dir) / self.robot_name / "plotted_trajectories" 
        output_dir.mkdir(parents=True, exist_ok=True)
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
            plt.savefig(output_dir / f'joint{i_plt}_pid_full_trajectory.png')
            plt.close()
        self.get_logger().info(f"Trajectory plots saved to {output_dir}")

def main(args=None):
    rclpy.init(args=args)
    node = CombinedControllerNode()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


