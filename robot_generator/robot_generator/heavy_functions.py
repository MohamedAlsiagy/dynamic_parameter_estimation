import random
from robot_generator.robot.robot_arm_pack import generate_n_robots
from robot_generator.gazebo.gazebo_adapter import process_robot_pack

generate_robot_batch(
    n,  # Number of robots to generate
    parent_dir,  # Parent directory to save robot files
    initial_diameter_range=(1, 3),  # Range for initial diameter
    link_length_range=(5, 15),  # Range for link length
    com_bias_range=(-0.5, 0.5),  # Range for center of mass bias
    joint_rotations_range=(0, 90),  # Range for joint rotations
    offset_range=(-2, 2),  # Range for offset
    include_gripper=(True, "x"),  # Whether to include a gripper
    joints=["z" , "y" , "y" , "z" , "y" , "z"],  # Number of joints per robot
):
    generate_n_robots(
        n,  # Number of robots to generate
        parent_dir,  # Parent directory to save robot files
        initial_diameter_range,  # Range for initial diameter
        link_length_range,  # Range for link length
        com_bias_range,  # Range for center of mass bias
        joint_rotations_range,  # Range for joint rotations
        offset_range,  # Range for offset
        include_gripper,  # Whether to include a gripper
        joints,  # Number of joints per robot
    )
    process_robot_pack(parent_dir)
