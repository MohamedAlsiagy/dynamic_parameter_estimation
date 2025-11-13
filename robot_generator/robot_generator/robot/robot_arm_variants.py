import cadquery as cq
from cadquery import exporters
import os
import json
import random
import time
from tqdm import tqdm

from robot_generator.robot.robot_arm_base import make_robot_arm_components
from robot_generator.core.link import create_link
from robot_generator.utils.colors import generate_color_gradient_hsv
from robot_generator.utils.mesh_utils import generate_dae_dir

def make_random_robot_arm_components(
    parent_dir, 
    num_joints=6, 
    max_link_length=14, 
    max_link_diameter_factor=2, 
    min_last_link_diameter=1,
    max_com_bias=0.44, 
    min_com_bias=0.22,
    max_joint_rotation=90,
    max_offset = (3,3),
    default_link_length=5,
    axis_biasis = [1,1,1],
    include_gripper = (True , "x")
    unit = "cm",
):
    # Generate random joint types (e.g., 'x', 'y', 'z')
    axes = ["x"]*axis_biasis[0] + ["y"]*axis_biasis[1] + ["z"]*axis_biasis[2]
    joints = [random.choice(axes) for _ in range(num_joints)]

    # Generate random joint rotations (in degrees)
    joint_rotations = [random.uniform(0, max_joint_rotation) for _ in range(num_joints + 2)]

    # Randomize link properties
    link_length = []
    previous_length = max_link_length  # Start with the maximum possible length for the first link
    for _ in range(num_joints + 1):
        current_length = random.uniform(max_link_length / 2, max_link_length)  # random link length <= previous length
        link_length.append(current_length)
        previous_length = current_length  # Update previous_length for the next link
    
    offset = [(random.uniform(-max_offset[0], max_offset[0]),random.uniform(-max_offset[1], max_offset[1])) for _ in range(num_joints + 1)]  # random offsets for each link
    com_bias = [random.uniform(-max_com_bias,-min_com_bias) for _ in range(num_joints + 1)]  # random center of mass biases for each joint
    com_bias = [-max_com_bias]*(num_joints + 1)
    min_link_diameter = -2 * sum(com_bias) + com_bias[0] + min_last_link_diameter
    max_link_diameter = min_link_diameter * max_link_diameter_factor
    link_diameter = random.uniform(min_link_diameter, max_link_diameter)  # random link diameter
    
    cylendrify_base = random.choice([True, False])
    # Call the existing function to make the robot arm

    make_robot_arm_components(
        parent_dir=parent_dir,
        joints=joints,
        joint_rotations=joint_rotations,
        link_length=link_length,
        initial_diameter=link_diameter,
        com_bias=com_bias,
        match_end_radius=True,
        cylendrify_base=cylendrify_base,
        support_height_percent=0.2,
        equalize_mass=True,
        offset=offset,
        default_link_length=default_link_length,
        include_gripper = include_gripper,
        unit = unit,
    )
    
def make_robot_arm_from_json(
    parent_dir,
    json_file,
    unit = "cm",
):
    json_data = json.load(open(json_file))
    joints = json_data["joints"]
    joint_rotations = json_data["joint_rotations"]
    link_length = json_data["link_length"]
    initial_diameter = json_data["initial_diameter"]
    com_bias = json_data["com_bias"]
    cylendrify_base = json_data["cylendrify_base"]
    offset = json_data["offset"]
    include_gripper = json_data["include_gripper"]
    
    make_robot_arm(
        parent_dir,
        joints=joints,
        initial_diameter=initial_diameter,
        com_bias=com_bias,
        joint_rotations=joint_rotations,
        link_length=link_length,
        match_end_radius=True,
        cylendrify_base=cylendrify_base,
        support_height_percent=0,
        equalize_mass=True,
        offset=offset,
        default_link_length=5,
        include_gripper=(include_gripper, "x"),

        overide_diameter_negatives=True,
        override_percentage=0.5,
        unit=unit,

        initial_color = (1,1,1),
        final_color = (0,0,0),
    )

def make_ulite6_mock_robot_arm(
    parent_dir,
):
    initial_color = [random.random() for i in range(3)]
    final_color = [random.random() for i in range(3)]
    
    num_joints = 6
    
    # Generate random joint types (e.g., 'x', 'y', 'z')
    joints = ["z" , "y" , "y" , "z" , "y" , "z"]
    link_length = [e for e in [2 , 2 , 4 , 2 ,1.5 , 2.5 , 1.5]]

    com_bias = [0 for L in link_length]
    offset = [(0 , 0) for L in link_length]
    link_diameter = 1  # random link diameter

    make_robot_arm(
        parent_dir=parent_dir,
        offset=offset,
        joints=joints,
        com_bias=com_bias,
        link_length=link_length,
        initial_diameter=link_diameter,
        match_end_radius=True,
        cylendrify_base=True,
        support_height_percent=0.2,
        equalize_mass=True,
        include_gripper = (False , "x"),

        initial_color = initial_color,
        final_color = final_color,
        unit="cm",
    )