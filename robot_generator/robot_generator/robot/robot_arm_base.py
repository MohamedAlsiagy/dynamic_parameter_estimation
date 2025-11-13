import cadquery as cq
from cadquery import exporters
from math import pi, sqrt
import os
from tqdm import tqdm

from robot_generator.core.link import create_link

from robot_generator.utils.colors import generate_color_gradient_hsv
from robot_generator.utils.mesh_utils import generate_dae_dir
from robot_generator.urdf.urdf_generator import generate_urdf

def make_robot_arm_components(
    parent_dir,
    joints=["x", "y", "z"],
    initial_diameter=3,
    com_bias=None,
    joint_rotations=None,
    link_length=None,
    match_end_radius=True,
    cylendrify_base=False,
    support_height_percent=0,
    equalize_mass=True,
    offset=None,
    default_link_length=5,
    include_gripper=(False, "x"),
    overide_diameter_negatives=False,
    override_percentage=0.5,
    unit="cm",
):
    # Convert inputs from meters to centimeters if needed
    convert_factor = 1.0
    if unit == "cm":
        convert_factor = 100.0

    # Default joint_rotations to an empty list if None is passed
    if joint_rotations is None:
        joint_rotations = []
        
    if link_length is None:
        link_length = [default_link_length] * (len(joints) + 1)

    # Apply unit conversion to link_length
    link_length = [l / convert_factor for l in link_length]

    # Default com_bias and offset to empty lists if None is passed
    if com_bias is None:
        com_bias = [0] * (len(joints) + 1)
    # Apply unit conversion to com_bias
    com_bias = [cb / convert_factor for cb in com_bias]

    if offset is None:
        offset = [0 , 0] * (len(joints) + 1)
    # Apply unit conversion to offset
    offset = [[o[j] / convert_factor for j in range(2)] for o in offset]

    # Convert initial_diameter
    initial_diameter = initial_diameter / convert_factor

    # Compute the link diameters
    diameters = [initial_diameter]
    for i in range(1, len(joints) + 1):
        diameters.append(diameters[-1] + com_bias[i-1] + com_bias[i])

    if overide_diameter_negatives:
        if min(diameters) < 0:
            diameters = [diameter - min(diameters) + 2 * abs(com_bias[-1]) + initial_diameter * override_percentage for diameter in diameters]
    print(diameters)

    joints = ["PH"] + joints + ["PH"]
    # Create links
    for i in tqdm(range(len(joints) - 1), desc="Processing joints"):
        link_name = f"link_{i}"

        links, axes = create_link(
            length=link_length[i],
            diameter=diameters[i],
            com_bias=com_bias[i],
            start_joint_type=joints[i],
            end_joint_type=joints[i + 1],
            match_end_radius=match_end_radius,
            cylendrify_base=cylendrify_base,
            support_height_percent=support_height_percent,
            equalize_mass=equalize_mass,
            offset=offset[i],
            s_xy_angle=joint_rotations[i] if i < len(joint_rotations) else 0,
            e_xy_angle=joint_rotations[i + 1] if i + 1 < len(joint_rotations) else 0,
            is_base_link=(i == 0),
            is_last_link=(i == len(joints) - 2),
            name=link_name,
            include_gripper_if_last_link=include_gripper
        )

        try:
            show_object(links[-1])
        except:
            pass

        # Export JSON for each axis
        json_parent_dir = os.path.join(parent_dir, 'json')
        os.makedirs(json_parent_dir, exist_ok=True)
        
        for axis in axes:
            axis.export_json(json_parent_dir)
            axis.show()
        
        subscripts = ["", "_g1", "_g2"]
        for i, link in enumerate(links):
            # Export STL file
            stl_dir = os.path.join(parent_dir, 'stl')
            os.makedirs(stl_dir, exist_ok=True)
            exporters.export(link, os.path.join(stl_dir, f"{link_name}{subscripts[i]}.stl"), tolerance=0.01)
    
            # Export STEP file
            stp_dir = os.path.join(parent_dir, 'stp')
            os.makedirs(stp_dir, exist_ok=True)
            exporters.export(link, os.path.join(stp_dir, f"{link_name}{subscripts[i]}.step"))

def make_robot_arm_from_components(
    parent_dir,
    num_joints,

    initial_color = (1,1,1),
    final_color = (0,0,0),
    robot_name = "robot",
):
    # Paths to the folders
    json_folder = f'{parent_dir}/json'
    stl_folder = f'{parent_dir}/stl'
    stp_folder = f'{parent_dir}/stp'  # You can incorporate STEP files if needed for visualization or other tasks
    urdf_filename = f'{parent_dir}/robot.urdf'
    dae_folder = f'{parent_dir}/dae'
    
    colors = generate_color_gradient_hsv(initial_color, final_color, num_joints + 1 + 2)

    generate_dae_dir(stl_folder, dae_folder , colors)
    
    urdf_dae_folder = f'dae'

    ## Generate the URDF
    generate_urdf(num_joints , json_folder, urdf_dae_folder , stp_folder, urdf_filename , robot_name)

def make_robot_arm(
    parent_dir,
    joints=["x", "y", "z"],
    initial_diameter=3,
    com_bias=None,
    joint_rotations=None,
    link_length=None,
    match_end_radius=True,
    cylendrify_base=False,
    support_height_percent=0,
    equalize_mass=True,
    offset=None,
    default_link_length=5,
    include_gripper=(True, "x"),
    overide_diameter_negatives=False,
    override_percentage=0.5,
    unit="cm",

    initial_color = (1,1,1),
    final_color = (0,0,0),
    robot_name = "robot",
):

    make_robot_arm_components(
        parent_dir,
        joints = joints,
        initial_diameter = initial_diameter,
        com_bias = com_bias,
        joint_rotations = joint_rotations,
        link_length = link_length,
        match_end_radius = match_end_radius,
        cylendrify_base = cylendrify_base,
        support_height_percent = support_height_percent,
        equalize_mass = equalize_mass,
        offset = offset,
        default_link_length = default_link_length,
        include_gripper = include_gripper,
        overide_diameter_negatives = overide_diameter_negatives,
        override_percentage = override_percentage,
        unit = unit,
    )

    make_robot_arm_from_components(
        parent_dir,
        len(joints),

        initial_color = initial_color,
        final_color = final_color,
        robot_name = robot_name,
    )

    

