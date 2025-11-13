import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import pickle
import os
import random
import hashlib
import json

def get_dataset_hash(robot_names, config_subset):
    key_data = {
        "robots": sorted(robot_names),
        **config_subset,
    }
    stringified = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(stringified.encode()).hexdigest()


def load_robot_data(robot_folder_path):
    trajectory_data_list = []

    """Load trajectory data and dynamic parameters for a single robot."""
    trajectory_files = [f for f in os.listdir(robot_folder_path) 
                       if f.startswith('trajectory_data_segment_') and f.endswith('.csv')]
    trajectory_files.sort()  # Ensure consistent ordering
    
    for f in trajectory_files:
        df = pd.read_csv(os.path.join(robot_folder_path, f))
        trajectory_data_list.append(df)

    # Load dynamic parameters
    dynamic_params_path = os.path.join(robot_folder_path, 'dynamic_parameters_normalized.csv')
    dynamic_params = pd.read_csv(dynamic_params_path)
    
    return trajectory_data_list, dynamic_params

def generate_sequences(trajectory_data, dynamic_params, sequence_length, overlap_ratio=0.5):
    """Generate overlapping sequences from trajectory data."""
    X_sequences = []
    
    # Calculate step size for overlapping sequences
    step_size = max(1, int(sequence_length * (1 - overlap_ratio)))
    
    for i in range(0, len(trajectory_data) - sequence_length + 1, step_size):
        sequence = trajectory_data.iloc[i:i + sequence_length]
        
        if len(sequence) == sequence_length:
            X_sequences.append(sequence)

    return np.array(X_sequences), np.tile(np.array(dynamic_params).reshape(1 , -1), (len(X_sequences), 1 , 1))

def load_robots_batch(robot_folders, input_features , output_features, sequence_length, overlap_ratio=0.5, sampling_rate=1 , secondary_sampling_rate=1):
    """Load and process a batch of robots."""
    batch_X, batch_y = [], []
    for robot_folder in tqdm(robot_folders, desc="Processing robots"):
        trajectory_data_list, dynamic_params = load_robot_data(robot_folder)
        for sampling_i in range(0, sampling_rate, secondary_sampling_rate):
            for trajectory_data in trajectory_data_list:
                X_seq, y_seq = generate_sequences(
                    trajectory_data.loc[sampling_i::sampling_rate, input_features], dynamic_params[output_features], sequence_length,
                    overlap_ratio
                )
                if len(X_seq) > 0 and X_seq.shape[0]:  # Only add if sequences were generated
                    batch_X.append(X_seq)
                    batch_y.append(y_seq)
    if batch_X:
        return np.concatenate(batch_X, axis=0), np.concatenate(batch_y, axis=0)
    else:
        return np.array([]), np.array([])

def preprocess_data(train_robots_chunk, input_features, output_features, config):
    """
    Memory-efficient batched preprocessing with caching to .npy files.
    """
    model_type = config["model_type"]
    sequence_length = config["sequence_length_transformer"] if model_type == "Transformer" else config["sequence_length_mamba"]
    sampling_rate = config["sampling_rate_transformer"] if model_type == "Transformer" else config["sampling_rate_mamba"]
    secondary_sampling_rate = config["secondary_sampling_rate_transformer"] if model_type == "Transformer" else config["secondary_sampling_rate_mamba"]

    overlap_ratio = config["overlap_ratio"]

    batch_size = config["batch_size_robots"]
    seed = config["seed"]
    cache_root = config["postprocessed_dataset_dir"]

    # Generate a hash key for this chunk and preprocessing configuration
    cache_key = get_dataset_hash(
        robot_names=train_robots_chunk,
        config_subset={
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "overlap_ratio": overlap_ratio,
            "input_features": input_features,
            "output_features": output_features,
            "secondary_sampling_rate": secondary_sampling_rate,
        }
    )
    cache_dir = os.path.join(cache_root, cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, "metadata.json")

    # === If cache exists, load and return ===
    if all(os.path.exists(os.path.join(cache_dir, f"{f}.npy")) for f in ["X_train", "y_train"]) and os.path.exists(metadata_path):
        print(f"[Cache Hit] Loading preprocessed data from {cache_dir}")
        X_train = torch.tensor(np.load(os.path.join(cache_dir, "X_train.npy")), dtype=torch.float32)
        y_train = torch.tensor(np.load(os.path.join(cache_dir, "y_train.npy")), dtype=torch.float32)

        return X_train, y_train

    print(f"[Cache Miss] Processing and saving to {cache_dir}")
    X_train_np, y_train_np = load_robots_batch(train_robots_chunk, input_features, output_features, sequence_length, overlap_ratio, sampling_rate , secondary_sampling_rate)

    # Save to disk
    np.save(os.path.join(cache_dir, "X_train.npy"), X_train_np)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train_np)

    with open(metadata_path, "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "overlap_ratio": overlap_ratio,
            "input_features": input_features,
            "output_features": output_features,
            "cache_key": cache_key,
            "robots": train_robots_chunk,
        }, f, indent=2)

    return (
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )

def load_test_dataset(test_robots, input_features, output_features, config):
    print("Loading test dataset...")
    """Load all test robots at once without batching, with caching support."""
    model_type = config["model_type"]
    sequence_length = config["sequence_length_transformer"] if model_type == "Transformer" else config["sequence_length_mamba"]
    sampling_rate = config["sampling_rate_transformer"] if model_type == "Transformer" else config["sampling_rate_mamba"]
    secondary_sampling_rate = config["secondary_sampling_rate_transformer"] if model_type == "Transformer" else config["secondary_sampling_rate_mamba"]

    overlap_ratio = config["overlap_ratio"]
    seed = config["seed"]
    cache_root = config["postprocessed_dataset_dir"]

    # Generate cache key for this configuration
    cache_key = get_dataset_hash(
        robot_names=test_robots,
        config_subset={
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "overlap_ratio": overlap_ratio,
            "input_features": input_features,
            "output_features": output_features,
            "test_split": True,  # Mark this as test dataset cache
            "secondary_sampling_rate": secondary_sampling_rate,
        }
    )
    cache_dir = os.path.join(cache_root, f"test_{cache_key}")
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, "metadata.json")

    # Check if cached test data exists
    if all(os.path.exists(os.path.join(cache_dir, f)) for f in ["X_test.npy", "y_test.npy"]) and os.path.exists(metadata_path):
        print(f"[Cache Hit] Loading preprocessed test data from {cache_dir}")
        X_test = np.load(os.path.join(cache_dir, "X_test.npy"))
        y_test = np.load(os.path.join(cache_dir, "y_test.npy"))
        
        # Load train/test split from metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
        
    print(f"[Cache Miss] Processing and saving test data to {cache_dir}")

    # Load all test data at once
    X_test, y_test = load_robots_batch(test_robots, input_features, output_features, 
                                     sequence_length, overlap_ratio, sampling_rate , secondary_sampling_rate)
    
    # Save to cache
    np.save(os.path.join(cache_dir, "X_test.npy"), X_test)
    np.save(os.path.join(cache_dir, "y_test.npy"), y_test)
    
    # Save metadata including train/test split
    with open(metadata_path, 'w') as f:
        json.dump({
            "test_split": True,
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "overlap_ratio": overlap_ratio,
            "input_features": input_features,
            "output_features": output_features,
            "cache_key": cache_key,
            "test_robots": test_robots,
        }, f, indent=2)
    
    return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


