import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
from utils import preprocess_data, load_test_dataset

sequence_length_transformer = [64 , 64 , 64, 64, 64] + [32 , 32 , 32, 32] + [128 , 128 , 128, 128] + [16 , 16 , 16, 16]  + [256 , 256 , 256, 256]
sampling_rate_transformer = [64 , 64 , 64, 32, 32]  + [32 , 64 , 64, 128] + [16 , 32 , 32, 64] + [32 , 64 , 128, 256]  + [16 , 16 , 32, 32]
secondary_sampling_rate_transformer = [16 , 32 , 64, 8 , 16] + [8 , 16 , 32, 128] + [8 , 8 , 16, 16]  + [8 , 16 , 32, 32]+ [8 , 16 , 8, 16]

I = 3
indicies = range(I , len(sequence_length_transformer) , 6)

sequence_length_transformer = [sequence_length_transformer[i] for i in indicies]
sampling_rate_transformer = [sampling_rate_transformer[i] for i in indicies]
secondary_sampling_rate_transformer = [secondary_sampling_rate_transformer[i] for i in indicies]

def get_robot_chunks(robots, chunk_size):
    for i in range(0, len(robots), chunk_size):
        yield robots[i:i + chunk_size]

if __name__ == "__main__":
    for i in range(len(sequence_length_transformer)):
        # Load configuration from YAML
        with open('/root/ws/deep_learning/config/deep_learning.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        config["sequence_length_transformer"] = sequence_length_transformer[i]
        config["sampling_rate_transformer"] = sampling_rate_transformer[i]
        config["secondary_sampling_rate_transformer"] = secondary_sampling_rate_transformer[i]

        print("="*24)
        print(f's{config["sequence_length_transformer"]}, sr{config["sampling_rate_transformer"]}, ssr{config["secondary_sampling_rate_transformer"]}')

        with open('/root/ws/deep_learning/config/data_preprocessor/raw_feature_analysis.yaml', 'r') as f:
            feature_analysis = yaml.safe_load(f)
        
        with open('/root/ws/deep_learning/config/dynamic_parameters.yaml', 'r') as f:
            dynamic_parameters = yaml.safe_load(f)

        with open('/root/ws/deep_learning/config/data_preprocessor/dynamic_parameters_analysis.yaml', 'r') as f:
            analyzed_dynamic_parameters = yaml.safe_load(f)

        # Set random seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset_root_path = config["dataset_root_path"]

        # Input and output features
        input_features = config["input_features"]
        input_features_exclude = config["input_features_exclude"]
        if input_features == "all":
            input_features = list(feature_analysis["dynamic_columns"])
            input_features.remove("time")
            input_features.remove("index")
        
        input_features = [
            f for f in input_features 
            if not any(elem in f for elem in input_features_exclude)
        ]

        output_features = config["output_features"]
        if output_features == "all":
            output_features = dynamic_parameters["dynamic_parameters"]
        elif output_features == "analyzed":
            output_features = analyzed_dynamic_parameters["dynamic_columns"]

        # Robot folders
        robot_folders = sorted([
            os.path.join(dataset_root_path, d) for d in os.listdir(dataset_root_path)
            if os.path.isdir(os.path.join(dataset_root_path, d))
        ])
        
        test_ratio = 1 - config["train_size"] / len(robot_folders)

        all_input_features = list(feature_analysis["dynamic_columns"])
        all_input_features.remove("time")
        all_input_features.remove("index")

        all_output_features = list(dynamic_parameters["dynamic_parameters"])
        output_indices = [all_output_features.index(f) for f in output_features]
        input_indices = [all_input_features.index(f) for f in input_features]

        # Split train/test robots
        train_robots, test_robots = train_test_split(robot_folders, test_size=test_ratio, random_state=seed)

        # Build test dataset
        X_test, y_test = load_test_dataset(test_robots, all_input_features, all_output_features, config)
        y_test = y_test[:, :, output_indices]
        X_test = X_test[:, :, input_indices]

        # Process training dataset in chunks
        chunk_size = config["batch_size_robots"]
        for chunk_index, robot_chunk in enumerate(get_robot_chunks(train_robots, chunk_size)):
            print(f"\n=== Processing Robot Chunk {chunk_index+1} ({len(robot_chunk)} robots) ===")
            
            X_train, y_train = preprocess_data(
                robot_chunk,
                all_input_features,
                all_output_features,
                config
            )
            y_train = y_train[:, :, output_indices]
            X_train = X_train[:, :, input_indices]