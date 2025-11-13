import os
import re
import csv
import matplotlib.pyplot as plt
import statistics

min_size_mb = float('inf')
min_size_robot = None
max_size_mb = 0
max_size_robot = None

def get_folder_size_mb(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def count_csv_rows(csv_path):
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            return max(len(rows) - 1, 0)  # exclude header
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return 0

def plot_folder_sizes(dataset_dir="dataset"):
    global min_size_mb, min_size_robot, max_size_mb, max_size_robot

    folder_nums = []
    folder_sizes = []

    if not os.path.isdir(dataset_dir):
        print(f"Directory '{dataset_dir}' does not exist.")
        return

    # First pass: find min and max folder size and track their robot numbers
    for folder_name in os.listdir(dataset_dir):
        if "robot" not in folder_name:
            continue
        full_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(full_path):
            match = re.search(r'\d+', folder_name)
            if not match:
                continue
            folder_num = int(match.group())
            size_mb = get_folder_size_mb(full_path)

            if size_mb < min_size_mb:
                min_size_mb = size_mb
                min_size_robot = folder_num
                min_size_folder_path = full_path
            if size_mb > max_size_mb:
                max_size_mb = size_mb
                max_size_robot = folder_num
                max_size_folder_path = full_path

            folder_nums.append(folder_num)
            folder_sizes.append(size_mb)

    if not folder_sizes:
        print("No robot folders found or no data collected.")
        return

    # Count rows only for min and max size folders
    min_rows = None
    max_rows = None

    if min_size_robot is not None:
        csv_path = os.path.join(min_size_folder_path, "trajectory_data.csv")
        min_rows = count_csv_rows(csv_path)

    if max_size_robot is not None:
        csv_path = os.path.join(max_size_folder_path, "trajectory_data.csv")
        max_rows = count_csv_rows(csv_path)

    # Sort for plotting
    zipped = sorted(zip(folder_nums, folder_sizes))
    folder_nums, folder_sizes = zip(*zipped)

    avg_size = sum(folder_sizes) / len(folder_sizes)
    median_size = statistics.median(folder_sizes)
    count = len(folder_sizes)

    print(f"Robot folders processed: {count}")
    print(f"Minimum folder size: {min_size_mb:.2f} MB (Robot #{min_size_robot})")
    if min_rows is not None:
        print(f"Rows in trajectory_data.csv for min size robot #{min_size_robot}: {min_rows}")

    print(f"Maximum folder size: {max_size_mb:.2f} MB (Robot #{max_size_robot})")
    if max_rows is not None:
        print(f"Rows in trajectory_data.csv for max size robot #{max_size_robot}: {max_rows}")

    print(f"Average folder size: {avg_size:.2f} MB")
    print(f"Median folder size: {median_size:.2f} MB")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(folder_nums, folder_sizes, color='skyblue')
    plt.xlabel('Folder Number')
    plt.ylabel('Folder Size (MB)')
    plt.title('Folder Size vs Folder Number in "dataset" Directory')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_folder_sizes("/root/ws/src/torque_controlled_arm/dataset")
