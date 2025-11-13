import os
import yaml
from pathlib import Path
from xml.etree import ElementTree as ET
from tqdm import tqdm

def extract_dynamic_parameters(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    dynamic_parameters = {"joints": {}, "links": {}}

    # Extract joint parameters
    for j_idx , joint in enumerate(root.findall("joint")):
        joint_name = joint.get("name")
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            damping = float(dynamics.get("damping", 0.0))
            friction = float(dynamics.get("friction", 0.0))

            dynamic_parameters["joints"][joint_name] = ({
                "idx" : j_idx - 1,
                "coulomb_friction_coeff": friction,
                "viscous_friction_coeff": damping,
            })
            

    # Extract link parameters
    for idx , link in enumerate(root.findall("link")):
        link_name = link.get("name")
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value", 0.0))
            ixx = float(inertial.find("inertia").get("ixx", 0.0))
            ixy = float(inertial.find("inertia").get("ixy", 0.0))
            ixz = float(inertial.find("inertia").get("ixz", 0.0))
            iyy = float(inertial.find("inertia").get("iyy", 0.0))
            iyz = float(inertial.find("inertia").get("iyz", 0.0))
            izz = float(inertial.find("inertia").get("izz", 0.0))
            com_xyz = {"xyz"[i] : float(x) for i , x in enumerate(inertial.find("origin").get("xyz", "0 0 0").split())}

            dynamic_parameters["links"][link_name] = ({
                "idx": idx,
                "mass": mass,
                "ixx": ixx,
                "ixy": ixy,
                "ixz": ixz,
                "iyy": iyy,
                "iyz": iyz,
                "izz": izz,
                "com": com_xyz
            })

    return dynamic_parameters

def process_robots_and_save_parameters(robots_dir, settings):
    robots_path = Path(robots_dir)
    dataset_output_dir = Path(settings["dataset_output_dir"])

    for robot_folder in tqdm(robots_path.iterdir() ,"robots processed"):
        if robot_folder.is_dir():
            robot_name = robot_folder.name
            urdf_path = robot_folder / "robotGA.urdf"
            
            robot_dataset_subfolder = dataset_output_dir / robot_name
            robot_dataset_subfolder.mkdir(parents=True, exist_ok=True)

            if urdf_path.exists():
                params = extract_dynamic_parameters(urdf_path)
                output_file = robot_dataset_subfolder / f"{robot_name}_dynamic_parameters.yaml"
                with open(output_file, "w") as f:
                    yaml.dump(params, f, indent=4)
            else:
                print(f"Skipped {robot_name}: robotGA.urdf not found.")

if __name__ == "__main__":
    settings_file_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(settings_file_path, "r") as f:
        settings = yaml.safe_load(f)

    robots_directory = Path(__file__).parent.parent / "robots"
    process_robots_and_save_parameters(robots_directory, settings)


