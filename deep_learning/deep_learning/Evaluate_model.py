import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from utils import preprocess_data, load_test_dataset
from model import UnifiedModel
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(model, loader, criterion, device, epoch=None, mode="Test", output_features=None):
    global dynamic_parameters_loggings
    
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    total_squared_error = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            total_squared_error += torch.sum((outputs - y_batch) ** 2).item()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            all_targets.append(y_batch)
            all_preds.append(outputs)

            total_samples += y_batch.numel()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = total_squared_error / total_samples
    rmse = mse ** 0.5
    r2 = r2_score(all_targets, all_preds).item()  # keep tensors

    print(f'{mode.capitalize()} Loss: {total_loss/len(loader):.4e}, '
          f'{mode.capitalize()} MSE: {mse:.4e}, {mode.capitalize()} RMSE: {rmse:.4e}, '
          f'{mode.capitalize()} R²: {r2:.4f}')

    num_sequences, num_joints, output_size = all_targets.shape
    all_targets_flat = all_targets.view(num_sequences * num_joints, output_size)
    all_preds_flat = all_preds.view(num_sequences * num_joints, output_size).detach()

    pearson_corr, _ = pearsonr(
        all_targets_flat.cpu().numpy().flatten(),
        all_preds_flat.cpu().numpy().flatten()
    )

    print(f'{mode.capitalize()} Pearson Correlation: {pearson_corr:.4f}')
    print("-" * 80)

    # ---- Per-feature R² (Validation only) ----
    per_feature_r2 = {}
    per_feature_rmse = {}
    if output_features is not None:
        for i, feature in enumerate(output_features):
            # R²
            r2_feat = r2_score(
                all_targets_flat[:, i],
                all_preds_flat[:, i]
            ).item()
            per_feature_r2[feature] = r2_feat

            # RMSE
            mse_feat = torch.mean((all_targets_flat[:, i] - all_preds_flat[:, i]) ** 2).item()
            rmse_feat = mse_feat ** 0.5
            per_feature_rmse[feature] = rmse_feat

            print(f"{mode.capitalize()} R² ({feature}): {r2_feat:.4f}, RMSE ({feature}): {rmse_feat:.4e}")

    if mode == "Val":
        all_targets = all_targets.view(num_sequences, num_joints, output_size)
        all_preds = all_preds.view(num_sequences, num_joints, output_size)
        predictions = torch.mean(all_preds, axis=0)
        dynamic_parameters_loggings[epoch] = predictions.detach().cpu().numpy()

    return total_loss / len(loader), mse, rmse, r2, pearson_corr

def get_robot_chunks(robots, chunk_size):
    for i in range(0, len(robots), chunk_size):
        yield robots[i:i + chunk_size]

if __name__ == "__main__":
    # Load configuration from YAML
    with open('/root/ws/deep_learning/config/deep_learning.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('/root/ws/deep_learning/config/data_preprocessor/raw_feature_analysis.yaml', 'r') as f:
        feature_analysis = yaml.safe_load(f)
    
    with open('/root/ws/deep_learning/config/dynamic_parameters.yaml', 'r') as f:
        dynamic_parameters = yaml.safe_load(f)

    with open('/root/ws/deep_learning/config/data_preprocessor/dynamic_parameters_analysis.yaml', 'r') as f:
        analyzed_dynamic_parameters = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available. Please check your setup.")
    
    print("Using device:", device)

    # Reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_root_path = config["dataset_root_path"]

    # Define model hyperparameters from config
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
    input_size = len(input_features)

    output_features = config["output_features"]
    if output_features == "all":
        output_features = dynamic_parameters["dynamic_parameters"]
    elif output_features == "analyzed":
        output_features = analyzed_dynamic_parameters["dynamic_columns"]
    output_size = len(output_features)

    model_type = config["model_type"]
    sequence_length = config["sequence_length_transformer"] if model_type == "Transformer" else config["sequence_length_mamba"]
    num_joints = config["num_joints"]
    m = config["m"]
    batch_size_training = config["batch_size_training"]
    version = config["version"]

    # Model directory and best model path
    secondery_model_save_dir = config["secondery_model_save_dir"]
    evaluation_model_path = config["Best_model"]

    model_dir = f"{config['model_type']}_s{sequence_length}_m{m}_e{config['num_epochs']}_b{batch_size_training}_v{version}"
    secondery_model_save_dir = os.path.join(secondery_model_save_dir, model_dir)

    # Prepare dataset
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

    # Split dataset
    _, test_robots = train_test_split(robot_folders, test_size=test_ratio, random_state=seed)
    X_test, y_test = load_test_dataset(test_robots, all_input_features, all_output_features, config)
    y_test = y_test[:, :, output_indices]
    X_test = X_test[:, :, input_indices]

    # Load model
    model = UnifiedModel(input_size, output_size, sequence_length=sequence_length, config=config, device=device)
    model.to(device)
    
    state_dict = torch.load(evaluation_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"Loaded best model from {evaluation_model_path}")

    # Evaluate on test set
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_training, shuffle=False)

    criterion = nn.MSELoss()
    evaluate_model(model, test_loader, criterion, device, mode="Test", output_features=output_features)

