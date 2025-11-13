import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
import re

def analyze_csv_dynamics(csv_path, std_threshold=1e-6, save_dir="/root/ws/deep_learning/config/data_preprocessor" , update_raw_yaml=True):
    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"\n[INFO] Analyzing: {csv_path}")
    print(f"[INFO] Total columns: {len(df.columns)}\n")

    dynamic_cols = {}
    constant_cols = {}

    for col in df.columns:
        std = df[col].std()
        if np.isnan(std) or std < std_threshold:
            constant_cols[col] = float(std) if not np.isnan(std) else None
        else:
            dynamic_cols[col] = float(std)

    # Summary
    print(f"[RESULTS]")
    print(f"- Total columns           : {len(df.columns)}")
    print(f"- Dynamic columns         : {len(dynamic_cols)}")
    print(f"- Nearly constant columns : {len(constant_cols)}\n")

    def extract_group_name(column_name):
        match = re.match(r"(Joint_\d+|Link_\d+)", column_name, re.IGNORECASE)
        return match.group(0) if match else "Other"

    def print_grouped_columns(col_dict, title):
        print(f"{title}")
        prev_group = None
        for k in sorted(col_dict):
            group = extract_group_name(k)
            if prev_group and group != prev_group:
                print("")
            std_value = col_dict[k]
            formatted_std = f"{std_value:.5e}" if std_value is not None else "NaN"
            print(f"- {k}: {formatted_std}")
            prev_group = group

    if dynamic_cols:
        print_grouped_columns(dynamic_cols, "[Changing columns]:")
    if constant_cols:
        print("")
        print_grouped_columns(constant_cols, "[Nearly constant columns]:")

    def group_columns_three_classes(col_dict):
        grouped = {
            "jacobians": {},
            "joints": {},
            "links": {},
        }

        for k, v in sorted(col_dict.items()):
            k_lower = k.lower()

            # Jacobian keys: contains 'dj'
            if "dj" in k_lower:
                # Extract link_i
                link_match = re.search(r"(link_\d+)", k_lower)
                link_key = link_match.group(0) if link_match else "unknown_link"

                # Extract joint_i
                joint_match = re.search(r"(joint_\d+)", k_lower)
                joint_key = joint_match.group(0) if joint_match else "unknown_joint"

                if link_key not in grouped["jacobians"]:
                    grouped["jacobians"][link_key] = {}
                if joint_key not in grouped["jacobians"][link_key]:
                    grouped["jacobians"][link_key][joint_key] = {}

                grouped["jacobians"][link_key][joint_key][k] = round(v, 6) if v is not None else None

            # Joints keys: start with joint_i (not jacobian)
            elif re.match(r"joint_\d+", k_lower):
                joint_match = re.match(r"(joint_\d+)", k_lower)
                joint_key = joint_match.group(0)
                if joint_key not in grouped["joints"]:
                    grouped["joints"][joint_key] = {}
                grouped["joints"][joint_key][k] = round(v, 6) if v is not None else None

            # Links keys: start with link_i but not jacobian
            elif re.match(r"link_\d+", k_lower):
                link_match = re.match(r"(link_\d+)", k_lower)
                link_key = link_match.group(0)
                if link_key not in grouped["links"]:
                    grouped["links"][link_key] = {}
                grouped["links"][link_key][k] = round(v, 6) if v is not None else None

        # Sort inner dicts alphabetically
        grouped["jacobians"] = dict(sorted(
            (link, dict(sorted(
                (joint, dict(sorted(keys.items())))
                for joint, keys in link_dict.items()
            ))) for link, link_dict in grouped["jacobians"].items()
        ))
        grouped["joints"] = dict(sorted(
            (joint, dict(sorted(keys.items())))
            for joint, keys in grouped["joints"].items()
        ))
        grouped["links"] = dict(sorted(
            (link, dict(sorted(keys.items())))
            for link, keys in grouped["links"].items()
        ))
        return grouped

    # Build YAML dict
    summary = {
        "std_threshold": std_threshold,
        "total_columns": len(df.columns),
        "dynamic_columns": len(dynamic_cols),
        "constant_columns": len(constant_cols),
        "dynamic_column_stats": group_columns_three_classes(dynamic_cols),
        "constant_column_stats": group_columns_three_classes(constant_cols),
    }

    save_path = Path(save_dir) / "humanized_feature_analysis.yaml"
    # Save to YAML
    with open(save_path, 'w') as f:
        yaml.dump(summary, f, sort_keys=False)

    print(f"\n✅ YAML summary saved to: {save_path}")

    if update_raw_yaml:
        # Create save directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        raw_save_path = Path(save_dir) / "raw_feature_analysis.yaml"
        # Save to YAML
        with open(raw_save_path, 'w') as f:
            yaml.dump({"dynamic_columns": list(dynamic_cols.keys()) , "constant_columns": list(constant_cols.keys())}, f, sort_keys=False)
        print(f"✅ Raw feature analysis saved to: {raw_save_path}")

if __name__ == "__main__":
    csv_path = "/root/ws/deep_learning/cached_dataset/robot_2/trajectory_data_segment_0.csv"
    std_threshold = 1e-6
    update_raw_yaml = False
    analyze_csv_dynamics(csv_path, std_threshold, update_raw_yaml=update_raw_yaml)
