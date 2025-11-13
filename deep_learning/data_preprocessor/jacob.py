import os
import yaml
import json
import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline

config_path = "/root/ws/deep_learning/config/"

# === Load shared.yaml ===
with open(config_path + "shared.yaml", 'r') as f:
    shared_config = yaml.safe_load(f)

RAW_DATASET_DIR = Path(shared_config["raw_dataset_dir"])
PREPROCESSED_DATASET_DIR = Path(shared_config["preprocessed_dataset_dir"])
ROBOTICS_DIR = Path(shared_config["robotics_dir"])

# === logic to read constant columns ===
with open(config_path + "data_preprocessor/" + "raw_feature_analysis.yaml", 'r') as f:
    feature_analysis = yaml.safe_load(f)

constant_columns = feature_analysis.get("constant_columns", [])

# === Load data_preprocessing.yaml ===
with open(config_path + "data_preprocessing.yaml", 'r') as f:
    dp_config = yaml.safe_load(f)

robot_range = range(dp_config["robot_range"]["start"], dp_config["robot_range"]["end"])
initial_naming_index = dp_config["initial_naming_index"]

include_config = dp_config.get("include", {})

sampling_frequency = dp_config["sampling_frequency"]  # Get sampling frequency from config

debug_sampling = dp_config.get("debug_sampling", {})
generate_samples, random_pick, sample_size = (
    debug_sampling["generate_samples"],
    debug_sampling["random_pick"],
    debug_sampling["sample_size"]
)

constant_indices = []
p.connect(p.DIRECT)

def resample_segment(segment_df, zeroed_time, sampling_frequency, robot_joint_indices, link_indices):
    """Resample segment data to constant sampling rate using cubic interpolation"""
    # Calculate new time vector
    duration = zeroed_time[-1] - zeroed_time[0]
    num_samples = int(duration * sampling_frequency) + 1
    new_time = np.linspace(0, duration, num_samples)
    
    # Prepare columns to interpolate
    columns_to_interpolate = ["index"]
    for j in robot_joint_indices:
        columns_to_interpolate += [
            f"joint_{j}_position", f"joint_{j}_velocity",f"joint_{j}_torque"
        ]
        if include_config.get("jacobians", True):
            for l in link_indices:
                columns_to_interpolate += [
                    f"joint_{j}_link_{l}_dx_dj", f"joint_{j}_link_{l}_dy_dj", f"joint_{j}_link_{l}_dz_dj",
                    f"joint_{j}_link_{l}_dwx_dj", f"joint_{j}_link_{l}_dwy_dj", f"joint_{j}_link_{l}_dwz_dj"
                ]

    for l in link_indices:
        if include_config.get("link_positions", True):
            columns_to_interpolate += [f"link_{l}_pos_x", f"link_{l}_pos_y", f"link_{l}_pos_z"]
        if include_config.get("link_orientations", True):
            columns_to_interpolate += [f"link_{l}_quat_x", f"link_{l}_quat_y", f"link_{l}_quat_z", f"link_{l}_quat_w"]
        if include_config.get("link_velocities", True):
            columns_to_interpolate += [f"link_{l}_vel_x", f"link_{l}_vel_y", f"link_{l}_vel_z"]
        if include_config.get("link_angular_velocities", True):
            columns_to_interpolate += [f"link_{l}_ang_vel_x", f"link_{l}_ang_vel_y", f"link_{l}_ang_vel_z"]
    
    # Filter the columns_to_interpolate to only include dynamic columns
    columns_to_interpolate = [col for col in columns_to_interpolate if col not in constant_columns]

    # Create new dataframe for resampled data
    resampled_data = {"time": new_time}
    
    # Interpolate each column
    for col in columns_to_interpolate:
        cs = CubicSpline(zeroed_time, segment_df[col].values, bc_type='natural')
        interpolated = cs(new_time)
        if col == "index":
            resampled_data[col] = np.round(interpolated).astype(int)
        else:
            resampled_data[col] = interpolated

    return pd.DataFrame(resampled_data)

for rid in robot_range:
    robot_name = f"robot_{rid}"
    print(f"\nProcessing: {robot_name}")
    
    robot_dataset_path = RAW_DATASET_DIR / robot_name
    robot_robotics_path = ROBOTICS_DIR / robot_name

    robot_name = f"robot_{rid + initial_naming_index}"
    output_robot_dataset_path = PREPROCESSED_DATASET_DIR / robot_name
    intermediate_output_path = output_robot_dataset_path / "intermediate"
    output_robot_dataset_path.mkdir(parents=True, exist_ok=True)
    intermediate_output_path.mkdir(parents=True, exist_ok=True)

    if not robot_dataset_path.exists():
        print(f"Dataset path {robot_dataset_path} does not exist. Skipping.")
        continue

    traj_csv_path = robot_dataset_path / "trajectory_data.csv"
    dyn_param_path = robot_dataset_path / "dynamic_parameters.yaml"
    urdf_path = robot_robotics_path / "robotGA.urdf"
    
    df = pd.read_csv(traj_csv_path, dtype={"index": int})
    with open(dyn_param_path, 'r') as f:
        dyn_params = yaml.safe_load(f)
    
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)

    urdf_joint_indices = [i for i in range(p.getNumJoints(robot_id))
                     if p.getJointInfo(robot_id, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

    robot_joint_indices = list(range(len(urdf_joint_indices)))

    link_indices = list(urdf_joint_indices)

    # === Detect and remove pre-failure corrupted data ===
    df["time_diff"] = df["time"].diff()
    raw_failure_indices = df[df["time_diff"] > 4].index.tolist()
    
    # make a copy to avoid modifying while iterating
    filtered_failures = []

    for i, fail_idx in enumerate(raw_failure_indices):
        if i > 0:  # not the first element
            prev_idx = raw_failure_indices[i - 1]
            # check if consecutive failures point to the same "index"
            if df.at[fail_idx, "index"] == df.at[prev_idx, "index"]:
                # skip adding the previous one (effectively removing it)
                filtered_failures.pop()  
        filtered_failures.append(fail_idx)

    # overwrite the old list if you want
    raw_failure_indices = filtered_failures
    
    for fail_idx in sorted(raw_failure_indices, reverse=True):
        fail_index_val = df.at[fail_idx, "index"]
        fail_time_val = df.at[fail_idx, "time"]
        before = len(df)
        df = df[~((df["index"] == fail_index_val) & (df["time"] < fail_time_val))]
        after = len(df)
        print(f"Removed {before - after} rows before failure at index={fail_index_val}, time={fail_time_val}.")
        print(len(df), "rows remaining after removal.")

    df = df.drop(columns=["time_diff"])
    df.reset_index(drop=True, inplace=True)

    # === Detect post-cleaning failures ===
    df["time_diff"] = df["time"].diff()
    cleaned_failure_indices = df[df["time_diff"] > 0.1].index.tolist()
    df = df.drop(columns=["time_diff"])

    # === Segment the dataframe ===
    segment_start_indices = [0] + cleaned_failure_indices + [len(df)]
    segmented_dfs = [
        (i, df.iloc[segment_start_indices[i]:segment_start_indices[i + 1]].copy())
        for i in range(len(segment_start_indices) - 1)
        if not df.iloc[segment_start_indices[i]:segment_start_indices[i + 1]].empty
    ]

    for seg_id, segment_df in segmented_dfs:
        print(f"Processing segment {seg_id} with {len(segment_df)} rows")
        result_rows = []
        columns = ["time", "index"]
        for j in robot_joint_indices:
            columns += [
                f"joint_{j}_position", f"joint_{j}_velocity",
                f"joint_{j}_torque"
            ]
            if include_config.get("jacobians", True):
                for l in link_indices:
                    columns += [
                        f"joint_{j}_link_{l}_dx_dj", f"joint_{j}_link_{l}_dy_dj", f"joint_{j}_link_{l}_dz_dj",
                        f"joint_{j}_link_{l}_dwx_dj", f"joint_{j}_link_{l}_dwy_dj", f"joint_{j}_link_{l}_dwz_dj"
                    ]

        for l in link_indices:
            if include_config.get("link_positions", True):
                columns += [f"link_{l}_pos_x", f"link_{l}_pos_y", f"link_{l}_pos_z"]
            if include_config.get("link_orientations", True):
                columns += [f"link_{l}_quat_x", f"link_{l}_quat_y", f"link_{l}_quat_z", f"link_{l}_quat_w"]
            if include_config.get("link_velocities", True):
                columns += [f"link_{l}_vel_x", f"link_{l}_vel_y", f"link_{l}_vel_z"]
            if include_config.get("link_angular_velocities", True):
                columns += [f"link_{l}_ang_vel_x", f"link_{l}_ang_vel_y", f"link_{l}_ang_vel_z"]

        constant_indices = [i for i, col in enumerate(columns) if col in constant_columns]
        columns = [col for col in columns if col not in constant_columns]

        for _, row in tqdm(segment_df.iterrows(), total=len(segment_df)):
            joint_positions = [row[f"Joint_{j}_Position"] for j in robot_joint_indices]
            joint_velocities = [row[f"Joint_{j}_Velocity"] for j in robot_joint_indices]
            joint_torques = [row[f"Joint_{j}_Torque"] for j in robot_joint_indices]

            for j, pos in zip(urdf_joint_indices, joint_positions):
                p.resetJointState(robot_id, j, pos)

            jac_ts = []
            jac_rs = []
            if include_config.get("jacobians", True) or include_config.get("link_velocities", True) or include_config.get("link_angular_velocities", True):
                for l in link_indices:
                    jac_t, jac_r = p.calculateJacobian(
                        robot_id,
                        linkIndex=l,
                        localPosition=[0, 0, 0],
                        objPositions=joint_positions,
                        objVelocities=joint_velocities,
                        objAccelerations=[0.0] * len(urdf_joint_indices)
                    )

                    jac_t = np.array(jac_t)
                    jac_r = np.array(jac_r)

                    jac_ts.append(jac_t)
                    jac_rs.append(jac_r)

            row_data = [row["time"], row["index"]]
            
            for j in range(len(robot_joint_indices)):
                row_data += [
                    joint_positions[j],
                    joint_velocities[j],
                    joint_torques[j]
                ]
                if include_config.get("jacobians", True):
                    for i, l in enumerate(link_indices):
                        # Calculate the Jacobian derivatives for the joint
                        jac_t = jac_ts[i]
                        jac_r = jac_rs[i]
                        row_data += [
                            jac_t[0, j], jac_t[1, j], jac_t[2, j],
                            jac_r[0, j], jac_r[1, j], jac_r[2, j]
                        ]

            joint_velocities_np = np.array(joint_velocities).reshape(-1, 1)

            for i, l in enumerate(link_indices):
                if include_config.get("link_positions", True) or include_config.get("link_orientations", True):
                    link_state = p.getLinkState(robot_id, l)
                    _, _, _, _, pos, quat = link_state
                if include_config.get("link_positions", True):
                    row_data += [pos[0], pos[1], pos[2]]
                if include_config.get("link_orientations", True):
                    row_data += [quat[0], quat[1], quat[2], quat[3]]
                    
                if include_config.get("link_velocities", True): 
                    lin_vel = (jac_ts[i] @ joint_velocities_np).flatten()
                    row_data += [lin_vel[0], lin_vel[1], lin_vel[2]]
                if include_config.get("link_angular_velocities", True):
                    ang_vel = (jac_rs[i] @ joint_velocities_np).flatten()
                    row_data += [ang_vel[0], ang_vel[1], ang_vel[2]]
            
            filtered_row_data = [row for i, row in enumerate(row_data) if i not in constant_indices]
            result_rows.append(filtered_row_data)

        # Create initial result dataframe with original timestamps
        result_df = pd.DataFrame(result_rows, columns=columns)
        result_df["index"] = result_df["index"].astype(int)
        
        # Make time start from zero and resample
        original_time = result_df["time"].values
        zeroed_time = original_time - original_time[0]
        result_df["time"] = zeroed_time  # Make time start from zero

        # Resample to constant sampling rate
        resampled_df = resample_segment(result_df, zeroed_time, sampling_frequency, 
                                      robot_joint_indices, link_indices)
        
        segment_path = output_robot_dataset_path / f"trajectory_data_segment_{seg_id}.csv"
        resampled_df.to_csv(segment_path, index=False)
        print(f"Saved resampled segment {seg_id} to {segment_path}")

        if generate_samples and seg_id == 0:
            sample_out_path = intermediate_output_path / "sample_data.csv"
            if random_pick:
                sample_df = resampled_df.sample(n=sample_size, random_state=42)
            else:
                sample_df = resampled_df.iloc[:sample_size]
            sample_df.to_csv(sample_out_path, index=False)

    dyn_params_df = pd.concat([
        pd.json_normalize(dyn_params["joints"], sep='_'),
        pd.json_normalize(dyn_params["links"], sep='_')
    ], axis=1)
    dyn_params_df = dyn_params_df.loc[:, ~dyn_params_df.columns.str.contains("idx", case=False)]
    dyn_params_df.to_csv(intermediate_output_path / "dynamic_parameters.csv", index=False)

    # Build summary dictionary
    summary = {
        "robot": robot_name,
        "num_failures": len(raw_failure_indices),
        "failure_indices": [int(i) for i in raw_failure_indices],
        "num_segments_saved": len(segmented_dfs),
        "segments": []
    }

    for seg_id, segment_df in segmented_dfs:
        segment_info = {
            "segment_id": seg_id,
            "start_time": float(segment_df["time"].iloc[0]),
            "end_time": float(segment_df["time"].iloc[-1]),
            "start_index": int(segment_df["index"].iloc[0]),
            "end_index": int(segment_df["index"].iloc[-1]),
            "num_rows": int(len(segment_df)),
            "resampled_frequency": sampling_frequency,
            "resampled_duration": float((segment_df["time"].iloc[-1] - segment_df["time"].iloc[0])),
            "resampled_samples": int((segment_df["time"].iloc[-1] - segment_df["time"].iloc[0]) * sampling_frequency) + 1
        }
        summary["segments"].append(segment_info)

    failure_gaps = df["time"].diff().iloc[cleaned_failure_indices].tolist()
    summary["failure_time_gaps"] = [float(gap) for gap in failure_gaps]

    # Write summary to YAML file
    summary_path = output_robot_dataset_path / "segmentation_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, sort_keys=False)

    print(f"Saved YAML summary to {summary_path}")

p.disconnect()