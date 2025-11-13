import pandas as pd
from pathlib import Path
import os
import yaml

def aggregate_csv_files(settings):
    dataset_output_dir = Path(settings["dataset_output_dir"])
    
    # Recursively find all CSV files in the dataset_output_dir and its subfolders
    all_csv_files = list(dataset_output_dir.rglob("*_trajectory_data.csv"))
    
    if not all_csv_files:
        print(f"No trajectory data CSV files found in {dataset_output_dir} or its subfolders.")
        return

    df_list = []
    for f in all_csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"Read {f.name}")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not df_list:
        print("No dataframes to concatenate.")
        return

    aggregated_df = pd.concat(df_list, ignore_index=True)

    output_csv_path = dataset_output_dir / "aggregated_trajectory_data.csv"
    aggregated_df.to_csv(output_csv_path, index=False)
    print(f"Aggregated data saved to {output_csv_path}")

if __name__ == "__main__":
    settings_file_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(settings_file_path, "r") as f:
        settings = yaml.safe_load(f)
    aggregate_csv_files(settings)


