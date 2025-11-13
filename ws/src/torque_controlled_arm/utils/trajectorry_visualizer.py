import pandas as pd
import matplotlib.pyplot as plt

for i in range(1343 ,  1344):
    # === Config ===
    csv_path = f"/root/ws/src/torque_controlled_arm/dataset/robot_{i}/trajectory_data.csv"
    sampling_rate_hz = 20.0  # Desired rate
    resample_interval = f"{int(1000 / sampling_rate_hz)}ms"  # 50ms for 20 Hz

    # === Load the CSV ===
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found. Skipping.")
        continue
    df = df.sort_values(by="time")
    df = df.drop_duplicates(subset="time")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # === Convert to TimedeltaIndex (required for resample) ===
    df["time"] = pd.to_timedelta(df["time"], unit="s")
    df = df.set_index("time")

    # === Resample and interpolate ===
    df_resampled = df.resample(resample_interval).mean().interpolate()

    # === Plot ===
    plt.figure(figsize=(14, 7))
    for joint_id in range(6):
        joint_col = f"Joint_{joint_id}_Position" #f"joint_{joint_id}_position"
        if joint_col in df_resampled.columns:
            plt.plot(df_resampled.index.total_seconds(), df_resampled[joint_col], label=joint_col)

    plt.title("Joint Positions Over Time (20 Hz)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad or unit]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
