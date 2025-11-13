import numpy as np
import json
import os
from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom
from tqdm import tqdm
from robot_generator.utils.json_utils import load_json , prettify
from robot_generator.utils.inertial_utils import get_mass_inertia_from_step

# Generate the URDF with mass and inertia properties
def generate_urdf(num_joints, json_folder, dae_folder, stp_folder, urdf_filename , robot_name = 'robot'):
    robot = Element('robot', name=robot_name)
    num_links = num_joints + 1

    # Loop through all links to generate link and joint elements
    for i in tqdm(range(num_links), desc="generating urdf"):
        json_path = os.path.join(json_folder, f"link_{i}.json")
        json_data = load_json(json_path)

        if i != num_links - 1:
            child_json_path = os.path.join(json_folder, f"link_{i+1}.json") 
            child_json_data = load_json(child_json_path)

        if json_data is None:
            print(f"Skipping link_{i} due to error in loading JSON data.")
            continue

        link = SubElement(robot, 'link', name=f"link_{i}")

        # Inertial properties
        stp_file = os.path.join(stp_folder, f"link_{i}.step")
        inertia_data = get_mass_inertia_from_step(stp_file)
        
        if inertia_data is None:
            print("No inertia data found. Exiting.")
            return

        inertial = SubElement(link, 'inertial')
        SubElement(inertial, 'origin', xyz=" ".join(map(str, inertia_data["center_of_mass"])), rpy="0 0 0")
        SubElement(inertial, 'mass', value=str(inertia_data["mass"]))
        inertia = SubElement(inertial, 'inertia', ixx=str(inertia_data["ixx"]), ixy=str(inertia_data["ixy"]), 
                             ixz=str(inertia_data["ixz"]), iyy=str(inertia_data["iyy"]), iyz=str(inertia_data["iyz"]), 
                             izz=str(inertia_data["izz"]))

        # Visual and collision elements
        dae_file = os.path.join(dae_folder, f"link_{i}.dae")
        for element in ["visual", "collision"]:
            elem = SubElement(link, element)
            SubElement(elem, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = SubElement(elem, 'geometry')
            SubElement(geometry, 'mesh', filename=dae_file)

        # Joint element if not the last link
        if i < num_links - 1:
            joint_name = f"joint_{i}"
            joint = SubElement(robot, 'joint', name=joint_name, type="revolute")
            parent_link = f"link_{i}"
            child_link = f"link_{i+1}"

            SubElement(joint, 'parent', link=parent_link)
            SubElement(joint, 'child', link=child_link)

            origin_start = np.array(json_data['start_joint_axis']['origin'])
            origin_end = np.array(json_data['end_joint_axis']['origin'])
            origin = origin_end - origin_start

            direction = np.array(child_json_data['start_joint_axis']['direction'])
            SubElement(joint, 'origin', xyz=" ".join(map(str, origin)), rpy="0 0 0")
            SubElement(joint, 'axis', xyz=" ".join(map(str, direction)))

            # Joint limit and dynamics
            SubElement(joint, 'limit', lower="-1.57", upper="1.57", effort="10", velocity="1")
            SubElement(joint, 'dynamics', damping="0.1", friction="0.2")

            # Transmission for each joint
            transmission = SubElement(robot, 'transmission', name=f"transmission_{i}")
            SubElement(transmission, 'type', value="transmission_interface/SimpleTransmission")
            SubElement(transmission, 'joint', name=joint_name)
            SubElement(transmission, 'hardwareInterface', value="hardware_interface/JointPositionInterface")
            actuator = SubElement(transmission, 'actuator', name=f"actuator_{i}")
            SubElement(actuator, 'mechanicalReduction', value="1.0")

    # Adding gripper fingers if it's the last link
    last_link_json_path = os.path.join(json_folder, f"link_{num_links - 1}.json")
    json_data = load_json(last_link_json_path)
    fingers_origin = np.array(json_data['end_joint_axis']['origin']) - np.array(json_data['start_joint_axis']['origin'])

    gripper_json_path = os.path.join(json_folder, "gripper_fingers.json")
    gripper_data = load_json(gripper_json_path)

    if gripper_data:
        for finger_id, finger_info in gripper_data.items():
            if "finger" not in finger_id:
                continue
            finger_name = f"link_{num_links - 1}_g{finger_id[-1:]}"
            finger_link = SubElement(robot, 'link', name=finger_name)

            # Inertial properties for the finger
            finger_stp_file = os.path.join(stp_folder, f"{finger_name}.step")
            finger_inertia_data = get_mass_inertia_from_step(finger_stp_file)
            if finger_inertia_data:
                finger_inertial = SubElement(finger_link, 'inertial')
                SubElement(finger_inertial, 'origin', xyz=" ".join(map(str, finger_inertia_data["center_of_mass"])), rpy="0 0 0")
                SubElement(finger_inertial, 'mass', value=str(finger_inertia_data["mass"]))
                SubElement(finger_inertial, 'inertia', 
                        ixx=str(finger_inertia_data["ixx"]), ixy=str(finger_inertia_data["ixy"]), ixz=str(finger_inertia_data["ixz"]),
                        iyy=str(finger_inertia_data["iyy"]), iyz=str(finger_inertia_data["iyz"]), izz=str(finger_inertia_data["izz"]))
            
            # Visual and collision elements for the finger
            finger_dae_file = os.path.join(dae_folder, f"{finger_name}.dae")
            for element in ["visual", "collision"]:
                elem = SubElement(finger_link, element)
                SubElement(elem, 'origin', xyz=" ".join(map(str, fingers_origin)), rpy="0 0 0")
                geometry = SubElement(elem, 'geometry')
                SubElement(geometry, 'mesh', filename=finger_dae_file)

            # Prismatic joint for the finger
            finger_joint = SubElement(robot, 'joint', name=f"joint_{finger_name}", type="prismatic")
            SubElement(finger_joint, 'parent', link=f"link_{num_links - 1}")
            SubElement(finger_joint, 'child', link=finger_name)

            origin = finger_info["origin"]
            direction = finger_info["direction"]

            SubElement(finger_joint, 'origin', xyz=" ".join(map(str, origin)), rpy="0 0 0")
            SubElement(finger_joint, 'axis', xyz=" ".join(map(str, direction)))

            # Prismatic joint limit and dynamics
            SubElement(finger_joint, 'limit', lower="0.0", upper=finger_info["travel"], effort="5", velocity="0.01")
            SubElement(finger_joint, 'dynamics', damping="0.1", friction="0.2")

    # Write the URDF to a file
    with open(urdf_filename, 'w') as urdf_file:
        urdf_file.write(prettify(robot))