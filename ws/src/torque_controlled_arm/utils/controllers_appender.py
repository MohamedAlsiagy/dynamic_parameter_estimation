import yaml
from pathlib import Path
from tqdm import tqdm  # progress bar

# Default path to the base controller template
DEFAULT_CONTROLLER_PATH = "/root/ws/src/torque_controlled_arm/config/controllers.yaml"

def generate_namespaced_controller_yaml(base_controller_config, robot_name):
    namespaced_config = {}
    for key, value in base_controller_config.items():
        namespaced_key = f"{robot_name}/{key}"
        namespaced_config[namespaced_key] = value
    return namespaced_config

def process_robot_folder(robot_folder: Path, base_controller_config: dict):
    robot_name = robot_folder.name
    output_config_dir = robot_folder / "config"
    output_config_dir.mkdir(parents=True, exist_ok=True)
    output_config_path = output_config_dir / "controllers.yaml"

    namespaced_config = generate_namespaced_controller_yaml(base_controller_config, robot_name)
    
    with open(output_config_path, 'w') as f:
        yaml.dump(namespaced_config, f, sort_keys=False)

def process_all_robot_folders(base_dir):
    base_path = Path(base_dir)

    # Load the shared base controller.yaml
    base_controller_path = Path(DEFAULT_CONTROLLER_PATH)
    if not base_controller_path.exists():
        raise FileNotFoundError(f"Base controller file not found: {base_controller_path}")

    with open(base_controller_path, 'r') as f:
        base_controller_config = yaml.safe_load(f)

    robot_folders = [folder for folder in base_path.iterdir() if folder.is_dir()]

    for folder in tqdm(robot_folders, desc="Processing robots", unit="robot"):
        process_robot_folder(folder, base_controller_config)

if __name__ == "__main__":
    robot_dir = Path(__file__).parent.parent / "robots"
    process_all_robot_folders(robot_dir)
    print("✅ Namespaced controllers.yaml files have been generated.")
