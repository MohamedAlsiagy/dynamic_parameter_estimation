import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from torque_controlled_arm_interfaces.srv import GetNextTarget  # Adjust to your actual package
from std_msgs.msg import Int32
import pybullet as p
import pybullet_data
import numpy as np
import random
import sys
from math import pi , nan
import os

n_targets = 10

def is_successive_or_same(link_a, link_b):
    return abs(link_a - link_b) <= 1

class RandomConfigurationChecker(Node):
    def __init__(self):
        super().__init__('random_configuration_checker')

        self.trajectory_termination_pub = self.create_publisher(Int32, '/generation_state', 10)
        self.declare_parameter('urdf_path', '')
        urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        print(urdf_path)
        if not os.path.isfile(urdf_path):
            self.get_logger().error(f"URDF file '{urdf_path}' not found.")
            sys.exit(1)

        print("Received URDF path:", urdf_path)

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print("recieved" , urdf_path)
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = [i for i in range(self.num_joints)
                              if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

        if len(self.joint_indices) < 6:
            self.get_logger().error("Robot must have at least 6 movable joints.")
            sys.exit(1)

        self.valid_configs = []
        self.current_index = 0

        # Interpolation parameters
        self.interpolating_steps = 30
        self.interpolation_step_index = 0
        self.interpolated_targets = []

        self.get_logger().info("Checking 10 random joint configurations...")
        self.check_random_configs()

        # Create service
        self.srv = self.create_service(GetNextTarget, 'get_next_target', self.get_next_target_callback)

    def check_random_configs(self):
        base_pose = self.compute_tip_position([0.0] * len(self.joint_indices))
        ref_pose = self.compute_tip_position([0.0, pi/2, 0.0, 0.0, 0.0, 0.0])
        ref_dist = np.linalg.norm(np.array(ref_pose) - np.array(base_pose))
        threshold_distance = 0.15 * ref_dist

        attempts = 0
        last_valid_config = [0.0] * len(self.joint_indices)  # initialize with zeros or some default

        while len(self.valid_configs) < n_targets:
            attempts += 1
            print(attempts)

            joint_positions = [random.uniform(-pi, pi) for _ in self.joint_indices]

            for i in [1,2]:
                joint_positions[i] = last_valid_config[i] + random.uniform(-pi, pi) / 2 

            # Interpolate between last_valid_config and joint_positions in 30 steps and check z-condition for all
            interpolation_steps = 30
            all_interp_z_valid = True
            for step in range(interpolation_steps):
                interp_joint_positions = [
                    (1 - (step + 1) / interpolation_steps) * last_valid_config[j] +
                    ((step + 1) / interpolation_steps) * joint_positions[j]
                    for j in range(len(joint_positions))
                ]
                for i, q in zip(self.joint_indices, interp_joint_positions):
                    p.resetJointState(self.robot, i, q)

                # Check z condition on all links from 2 to end
                for i in range(3, p.getNumJoints(self.robot)):
                    link_pos = p.getLinkState(self.robot, i)[4]
                    if link_pos[2] < 0.6:
                        print(step)
                        all_interp_z_valid = False
                        break
                if not all_interp_z_valid:
                    break

            if not all_interp_z_valid:
                continue

            # Now check for collisions at the final joint_positions
            for i, q in zip(self.joint_indices, joint_positions):
                p.resetJointState(self.robot, i, q)

            contacts = p.getClosestPoints(bodyA=self.robot, bodyB=self.robot, distance=0.001)
            real_collisions = [
                c for c in contacts
                if c[3] != c[4] and not is_successive_or_same(c[3], c[4])
            ]
            if real_collisions:
                continue

            # Check tip position distance from base pose
            tip_pos = self.compute_tip_position(joint_positions)
            if np.linalg.norm(np.array(tip_pos) - np.array(base_pose)) < threshold_distance:
                continue

            self.get_logger().info(f"Valid config {len(self.valid_configs)+1}: {np.round(joint_positions, 3)}")
            self.valid_configs.append(joint_positions)
            last_valid_config = joint_positions

    def compute_tip_position(self, joint_values):
        for i, q in zip(self.joint_indices, joint_values):
            p.resetJointState(self.robot, i, q)
        link_state = p.getLinkState(self.robot, self.joint_indices[-1])
        return link_state[4]

    def get_next_target_callback(self, request, response):
        if not self.valid_configs:
            self.get_logger().error("No valid configurations available.")
            response.joint_positions = []
            return response

        if self.current_index >= len(self.valid_configs):
            response.joint_positions = [nan] * 6
            self.get_logger().info("All targets have been returned. Publishing trajectory termination.")

            msg = Int32()
            msg.data = 0
            self.trajectory_termination_pub.publish(msg)

        else:
            response.joint_positions = self.valid_configs[self.current_index]
            self.get_logger().info(f"Returning config {self.current_index + 1}")
            self.current_index += 1

        return response

def main(args=None):
    rclpy.init(args=args)
    node = RandomConfigurationChecker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
