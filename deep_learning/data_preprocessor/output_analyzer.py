import yaml
from collections import defaultdict
import os

# -----------------------------
# CONFIG: set the main folder path
# -----------------------------
CONFIG_PATH = "/root/ws/deep_learning/config"  # <-- change this to your folder

DYNAMIC_PARAMS_FILE = os.path.join(CONFIG_PATH, "dynamic_parameters.yaml")
RAW_FEATURE_FILE = os.path.join(CONFIG_PATH, "data_preprocessor", "raw_feature_analysis.yaml")
OUTPUT_FILE = os.path.join(CONFIG_PATH, "data_preprocessor", "dynamic_parameters_analysis.yaml")

# -----------------------------
# Helper functions
# -----------------------------
def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data, filename):
    with open(filename, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def extract_derivatives(dynamic_cols, constant_cols):
    link_derivatives = {str(i): {
        "x": False, "y": False, "z": False,
        "wx": False, "wy": False, "wz": False
    } for i in range(1,6+1)}

    def process_col(col):
        parts = col.split("_")
        if "link" not in parts or parts[-1] != "dj":
            return
        link_index = parts.index("link") + 1  # the number after 'link'
        link_id = parts[link_index]           # e.g., '3'
        deriv = parts[link_index + 1]         # e.g., 'dx', 'dwy', etc.
        if deriv.startswith("d"):
            deriv = deriv[1:]  # remove leading 'd'
        link_derivatives[link_id][deriv] = True

    for col in dynamic_cols:
        process_col(col)

    return link_derivatives

def map_to_dynamic_parameters(link_id, derivs):
    dynamic = []
    constant = []

    # Mass
    if link_id != "1":
        dynamic.append(f"link_{link_id}_mass")
        dynamic.append(f"link_{link_id}_com_z")
    else:
        constant.append(f"link_{link_id}_mass")
        constant.append(f"link_{link_id}_com_z")

    # COM parameters
    constant.append(f"link_{link_id}_com_x")
    constant.append(f"link_{link_id}_com_y")

    # Inertia parameters
    if derivs["wx"]:
        dynamic.append(f"link_{link_id}_ixx")
    else:
        constant.append(f"link_{link_id}_ixx")

    if derivs["wy"]:
        dynamic.append(f"link_{link_id}_iyy")
    else:
        constant.append(f"link_{link_id}_iyy")

    # if derivs["wz"]:
    #     dynamic.append(f"link_{link_id}_izz")
    # else:
    #     constant.append(f"link_{link_id}_izz")

    dynamic.append(f"link_{link_id}_izz")

    # if (derivs["wx"]) and (derivs["wy"]):
    #     dynamic.append(f"link_{link_id}_ixy")
    # else:
    #     constant.append(f"link_{link_id}_ixy")

    # if (derivs["wx"]) and (derivs["wz"]):
    #     dynamic.append(f"link_{link_id}_ixz")
    # else:
    #     constant.append(f"link_{link_id}_ixz")

    # if (derivs["wy"]) and (derivs["wz"]):
    #     dynamic.append(f"link_{link_id}_iyz")
    # else:
    #     constant.append(f"link_{link_id}_iyz")
    
    constant.append(f"link_{link_id}_ixy")
    constant.append(f"link_{link_id}_ixz")
    constant.append(f"link_{link_id}_iyz")
    
    return dynamic, constant

# -----------------------------
# Main script
# -----------------------------
def main():
    dyn_params_yaml = load_yaml(DYNAMIC_PARAMS_FILE)
    raw_feature_yaml = load_yaml(RAW_FEATURE_FILE)

    all_dyn_params = dyn_params_yaml["dynamic_parameters"]
    dynamic_cols = raw_feature_yaml["dynamic_columns"]
    constant_cols = raw_feature_yaml["constant_columns"]

    link_derivatives = extract_derivatives(dynamic_cols, constant_cols)

    dynamic_params = []
    constant_params = []

    for link_id, derivs in link_derivatives.items():
        d, c = map_to_dynamic_parameters(link_id, derivs)
        dynamic_params.extend(d)
        constant_params.extend(c)

    # Include joint friction parameters as dynamic
    joint_params = [p for p in all_dyn_params if "joint" in p]
    dynamic_params.extend(joint_params)

    result = {
        "dynamic_columns": sorted(dynamic_params),
        "constant_columns": sorted(constant_params)
    }

    save_yaml(result, OUTPUT_FILE)
    print(f"Filtered dynamic and constant parameters saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
