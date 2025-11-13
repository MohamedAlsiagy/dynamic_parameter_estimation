import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# === CONFIGURATION ===
config_path = "/root/ws/deep_learning/config/"

# Load shared.yaml
with open(Path(config_path) / "shared.yaml", "r") as f:
    shared_config = yaml.safe_load(f)

PREPROCESSED_DATASET_DIR = Path(shared_config["preprocessed_dataset_dir"])

# === GATHER ALL DATA FOR NORMALIZATION ===
all_dfs = []
robot_paths = []

for robot_folder in sorted(PREPROCESSED_DATASET_DIR.iterdir()):
    if not robot_folder.is_dir():
        continue

    csv_path = robot_folder / "intermediate" / "dynamic_parameters.csv"
    if not csv_path.exists():
        print(f"⚠ Skipping {robot_folder.name}: No dynamic_parameters.csv")
        continue

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("idx", case=False)]

    all_dfs.append(df)
    robot_paths.append((robot_folder, df))

# Concatenate all data
combined_df = pd.concat(all_dfs, ignore_index=True)

# === Identify Link Indices ===
link_indices = sorted({int(col.split('_')[1]) for col in combined_df.columns if col.startswith("link_")})

# === Group Columns ===
inertia_suffixes = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
com_suffixes = ["com_x", "com_y", "com_z"]

groups = {}
for link_id in link_indices:
    # Inertia group
    inertia_cols = [f"link_{link_id}_{sfx}" for sfx in inertia_suffixes]
    if all(col in combined_df.columns for col in inertia_cols):
        vals = combined_df[inertia_cols].values.flatten()
        groups[f"link_{link_id}_inertia"] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "columns": inertia_cols
        }
    # COM group
    com_cols = [f"link_{link_id}_{sfx}" for sfx in com_suffixes]
    if all(col in combined_df.columns for col in com_cols):
        vals = combined_df[com_cols].values.flatten()
        groups[f"link_{link_id}_com"] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "columns": com_cols
        }

# === Prepare individuals: all columns excluding grouped columns ===
grouped_cols = set(col for g in groups.values() for col in g["columns"])
individuals = {}
for col in combined_df.columns:
    if col in grouped_cols:
        continue
    col_min = float(combined_df[col].min())
    col_max = float(combined_df[col].max())
    individuals[col] = {"min": col_min, "max": col_max}

# === Normalize and Save Per Robot ===
for robot_folder, df in tqdm(robot_paths, desc="Normalizing per robot"):
    norm_df = df.copy()

    # Normalize grouped columns jointly
    for group_name, criteria in groups.items():
        min_val = criteria["min"]
        max_val = criteria["max"]
        cols = criteria["columns"]
        sub_df = norm_df[cols]
        if max_val == min_val:
            norm_vals = 0.0
        else:
            norm_vals = (sub_df - min_val) / (max_val - min_val)
        norm_df[cols] = norm_vals

    # Normalize individuals column-wise
    for col, stats in individuals.items():
        min_val = stats["min"]
        max_val = stats["max"]
        if max_val == min_val:
            norm_df[col] = 0.0
        else:
            norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)

    out_path = robot_folder / "dynamic_parameters_normalized.csv"
    norm_df.to_csv(out_path, index=False)

# === Save Normalization Criteria ===
normalization_criteria = {
    "groups": groups,
    "individuals": individuals
}

dp_config_dir_path = Path(config_path) / "data_preprocessor"
os.makedirs(dp_config_dir_path, exist_ok=True)
with open(dp_config_dir_path / "dynamic_param_normalization.yml", "w") as f:
    yaml.dump(normalization_criteria, f, sort_keys=False)

print("✅ Per-robot normalized files saved.")
print(f"📄 Normalization criteria saved to: {dp_config_dir_path / 'dynamic_param_normalization.yml'}")
