import os
import random
import time
from tqdm import tqdm
import cadquery as cq
from cadquery import exporters
from robot_generator.utils.colors import generate_color_gradient_hsv
from robot_generator.utils.mesh_utils import generate_dae_dir
from robot_generator.robot.robot_arm_base import make_robot_arm

def generate_n_random_robots(
    n,  # Number of robots to generate
    parent_dir,  # Parent directory to save robot files
    initial_diameter_range=(1, 3),  # Range for initial diameter
    link_length_range=(5, 15),  # Range for link length
    com_bias_range=(-0.5, 0.5),  # Range for center of mass bias
    joint_rotations_range=(0, 0),  # Range for joint rotations
    offset_range=(-0, 0),  # Range for offset
    include_gripper=(False, "x"),  # Whether to include a gripper
    joints=["z" , "y" , "y" , "z" , "y" , "z"],  # Number of joints per robot
    unit="cm",
):

    """
    Generate `n` robots with random parameters within user-defined ranges.
    Each robot is saved in a separate folder under `parent_dir`.
    """
    for i in range(n):
        initial_color = [random.random() for i in range(3)]
        final_color = [random.random() for i in range(3)]

        # Create a subdirectory for each robot
        robot_dir = os.path.join(parent_dir, f"robot_{i}")
        os.makedirs(robot_dir, exist_ok=True)

        num_joints = len(joints)
        # Randomize parameters within the specified ranges
        initial_diameter = random.uniform(*initial_diameter_range)
        link_length = [random.uniform(*link_length_range) for _ in range(num_joints + 1)]
        com_bias = [random.uniform(*com_bias_range) for _ in range(num_joints + 1)]
        joint_rotations = [random.uniform(*joint_rotations_range) for _ in range(num_joints + 2)]
        offset = [[random.uniform(*offset_range), random.uniform(*offset_range)] for _ in range(num_joints + 1)]

        # Generate the robot arm
        make_robot_arm(
            parent_dir=robot_dir,
            joints=joints,  # Default joint types
            initial_diameter=initial_diameter,
            com_bias=com_bias,
            joint_rotations=joint_rotations,
            link_length=link_length,
            offset=offset,
            include_gripper=include_gripper,
            overide_diameter_negatives = True,
            cylendrify_base = True,

            initial_color = initial_color,
            final_color = final_color,
            unit = unit,

            robot_name = f"robot_{i}",
        )

        print(f"Robot {i} generated and saved in {robot_dir}")


def generate_n_kinematically_constant_robots(
    n,  # Number of robots to generate
    parent_dir,  # Parent directory to save robot files
    link_length,
    initial_diameter_range=(1, 3),  # Range for initial diameter
    com_bias_range=(-0.5, 0.5),  # Range for center of mass bias
    joints=["z" , "y" , "y" , "z" , "y" , "z"],  # Number of joints per robot
    unit="cm",
):

    """
    Generate `n` robots with random parameters within user-defined ranges.
    Each robot is saved in a separate folder under `parent_dir`.
    """
    for i in range(n):
        initial_color = [random.random() for i in range(3)]
        final_color = [random.random() for i in range(3)]
        # Create a subdirectory for each robot
        robot_dir = os.path.join(parent_dir, f"robot_{i}")
        os.makedirs(robot_dir, exist_ok=True)

        num_joints = len(joints)
        # Randomize parameters within the specified ranges
        initial_diameter = random.uniform(*initial_diameter_range)
        com_bias = [random.uniform(*[initial_diameter * percent for percent in com_bias_range]) for _ in range(num_joints + 1)]
        joint_rotations = [0 for _ in range(num_joints + 2)]
        offset = [[0 , 0] for _ in range(num_joints + 1)]
        cylinder = random.choice([True , False])

        # Generate the robot arm
        make_robot_arm(
            parent_dir=robot_dir,
            joints=joints,  # Default joint types
            initial_diameter=initial_diameter,
            com_bias=com_bias,
            joint_rotations=joint_rotations,
            link_length=link_length,
            offset=offset,
            include_gripper=(False, "x"),
            overide_diameter_negatives=False,
            cylendrify_base=cylinder,

            initial_color=initial_color,
            final_color=final_color,
            unit=unit,

            robot_name=f"robot_{i}",
        )

        print(f"Robot {i} generated and saved in {robot_dir}")