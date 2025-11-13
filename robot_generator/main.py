from robot_generator.robot.robot_arm_pack import generate_n_kinematically_constant_robots

def main():
    generate_n_kinematically_constant_robots(
        512,  # Number of robots to generate
        "/root/ws/src/torque_controlled_arm/robots_p7",  # Parent directory to save robot files
        [15 , 7.5 , 30 , 25 , 10 , 7.5 , 10],
        initial_diameter_range=(6, 12),  # Range for initial diameter
        com_bias_range=(-0.35/6, 0),  # Range for center of mass bias
        joints=["z" , "y" , "y" , "z" , "y" , "z"],  # Number of joints per robot
    )

main()