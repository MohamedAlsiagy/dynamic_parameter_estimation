import yaml
import sys
from pathlib import Path

# Predefined path to base_controller.yaml
BASE_CONTROLLER_PATH = Path("/root/ws/src/torque_controlled_arm/config/base_controllers.yaml")  # <-- Change this if needed

def namespace_controller_yaml(namespace: str):
    if not BASE_CONTROLLER_PATH.exists():
        print(f"❌ Error: {BASE_CONTROLLER_PATH} not found.")
        sys.exit(1)

    with open(BASE_CONTROLLER_PATH, 'r') as f:
        base_data = yaml.safe_load(f) or {}

    # Keys to namespace
    keys_to_namespace = ["controller_manager", "torque_arm_controller"]

    # Create new namespaced version
    namespaced_data = {}
    for key in keys_to_namespace:
        if key not in base_data:
            print(f"⚠️ Warning: Base key '{key}' not found in {BASE_CONTROLLER_PATH}.")
            continue
        namespaced_key = f"{namespace}/{key}"
        namespaced_data[namespaced_key] = base_data[key]

    # Output path: same directory as base file
    output_path = BASE_CONTROLLER_PATH.parent / "controllers.yaml"

    # Save result
    with open(output_path, 'w') as f:
        yaml.dump(namespaced_data, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Wrote '{output_path}' with namespace '{namespace}'.")

# --- CLI usage ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python controller_adapter.py <namespace>")
        sys.exit(1)

    namespace = sys.argv[1]

    namespace_controller_yaml(namespace)
