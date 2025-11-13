import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import yaml
import wandb 
from sklearn.model_selection import train_test_split
from utils import preprocess_data, load_test_dataset
from model import UnifiedModel
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch.nn.functional as F

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def train_secondery_model(main_model , secondery_model, train_loader, val_loader, criterion, optimizer, device , model_save_dir , num_epochs=100 , model_save_freq=10):
    best_val_loss = float('inf')

    best_val_dir = os.path.join(model_save_dir, "best")
    checkpoint_dir = os.path.join(model_save_dir, "periodic")
    os.makedirs(best_val_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        secondery_model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute cosine similarity between main model predictions and ground truth
            outputs_main = main_model(X_batch)
            cos_sim = F.cosine_similarity(outputs_main, y_batch, dim=-1)  # returns (batch_size,)
            cos_sim = cos_sim.unsqueeze(1)

            outputs = secondery_model(X_batch)
            loss = criterion(outputs, cos_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(cos_sim.detach())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        train_mse = criterion(all_preds, all_targets).item()
        train_rmse = torch.sqrt(torch.tensor(train_mse)).item()
        train_r2   = r2_score(all_targets, all_preds).item()

        all_targets_flat = all_targets.cpu().numpy()
        all_preds_flat = all_preds.detach().cpu().numpy()
  
        train_pearson_corr, _ = pearsonr(all_targets_flat.flatten(), all_preds_flat.flatten())

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4e}, '
              f'Train MSE: {train_mse:.4e}, Train RMSE: {train_rmse:.4e}, Train R²: {train_r2:.4f}')
        
        mode = "Train"
        wandb.log({
            f"{mode}_Loss": epoch_loss / len(train_loader),
            f"{mode}_MSE": train_mse,
            f"{mode}_RMSE": train_rmse,
            f"{mode}_R2": train_r2,
            f"{mode}_Pearson_Correlation": train_pearson_corr,
        })

        val_loss, val_mse, val_rmse, val_r2, val_pearson_corr = evaluate_secondery_model(main_model , secondery_model, val_loader, criterion, device, epoch=epoch, mode="Val", output_features = output_features)

        # Save best validation model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(best_val_dir, "model.pth")
            torch.save(secondery_model.state_dict(), best_path)
            print(f"New best validation model saved at epoch {epoch + 1} to {best_path}")

        # Save checkpoint every n epochs or at last epoch
        if (epoch + 1) % model_save_freq == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"model.pth")
            torch.save(secondery_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

def evaluate_secondery_model(main_model , secondery_model, loader, criterion, device, epoch=None, mode="Test", output_features=None):
    global dynamic_parameters_loggings
    
    secondery_model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    total_squared_error = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs_main = main_model(X_batch)
            cos_sim = F.cosine_similarity(outputs_main, y_batch, dim=-1)  # returns (batch_size,)
            cos_sim = cos_sim.unsqueeze(1)

            outputs = secondery_model(X_batch)
            total_squared_error += torch.sum((outputs - cos_sim) ** 2).item()

            loss = criterion(outputs, cos_sim)

            total_loss += loss.item()
            
            all_targets.append(cos_sim.detach())
            all_preds.append(outputs)

            total_samples += cos_sim.numel()

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
        all_targets_flat.view(-1).cpu().numpy(),
        all_preds_flat.view(-1).cpu().numpy()
    )

    print(f'{mode.capitalize()} Pearson Correlation: {pearson_corr:.4f}')
    print("-" * 80)

    # ---- Per-feature R² (Validation only) ----
    per_feature_r2 = {}
    per_feature_rmse = {}
    # if output_features is not None:
    #     for i, feature in enumerate(output_features):
    #         # R²
    #         r2_feat = r2_score(
    #             all_targets_flat[:, i],
    #             all_preds_flat[:, i]
    #         ).item()
    #         per_feature_r2[feature] = r2_feat

    #         # RMSE
    #         mse_feat = torch.mean((all_targets_flat[:, i] - all_preds_flat[:, i]) ** 2).item()
    #         rmse_feat = mse_feat ** 0.5
    #         per_feature_rmse[feature] = rmse_feat

    #         print(f"{mode.capitalize()} R² ({feature}): {r2_feat:.4f}, RMSE ({feature}): {rmse_feat:.4e}")

    # ---- Logging ----
    if mode == "Test":
        wandb.log({
            f"{mode}_Metrics": wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Loss", total_loss / len(loader)],
                    ["MSE", mse],
                    ["RMSE", rmse],
                    ["R2", r2],
                    ["Pearson Correlation", pearson_corr]
                ]
            )
        })
    else:
        log_data = {
            f"{mode}_Loss": total_loss / len(loader),
            f"{mode}_MSE": mse,
            f"{mode}_RMSE": rmse,
            f"{mode}_R2": r2,
            f"{mode}_Pearson_Correlation": pearson_corr,
        }
        # Add per-feature R²
        if per_feature_r2:
            for feat in output_features:
                log_data[f"[debug]_{mode}_R2_{feat}"] = per_feature_r2[feat]
                log_data[f"[debug]_{mode}_RMSE_{feat}"] = per_feature_rmse[feat]
        
        wandb.log(log_data)

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
    
    # Load configuration from YAML
    with open('/root/ws/deep_learning/config/data_preprocessor/raw_feature_analysis.yaml', 'r') as f:
        feature_analysis = yaml.safe_load(f)
    
    # Load configuration from YAML
    with open('/root/ws/deep_learning/config/dynamic_parameters.yaml', 'r') as f:
        dynamic_parameters = yaml.safe_load(f)

    # Load configuration from YAML
    with open('/root/ws/deep_learning/config/data_preprocessor/dynamic_parameters_analysis.yaml', 'r') as f:
        analyzed_dynamic_parameters = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available. Please check your setup.")
    
    print("Using device:", device)

    # Set random seed for reproducibility
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
    
    input_size = len(input_features)  # Number of input features

    output_features = config["output_features"]
    if output_features == "all":
        output_features = dynamic_parameters["dynamic_parameters"]
    elif output_features == "analyzed":
        output_features = analyzed_dynamic_parameters["dynamic_columns"]
    output_size = len(output_features)  # Number of output features
    
    model_type = config["model_type"]
    sequence_length = config["sequence_length_transformer"] if model_type == "Transformer" else config["sequence_length_mamba"]

    num_joints = config["num_joints"]
    m = config["m"]
    
    num_epochs = config["num_epochs"]
    batch_size_training = config["batch_size_training"]
    lr = config["lr"]
    version = config["version"]

    model_save_dir = config["secondery_model_save_dir"]
    best_model_path = config["Best_model"]
    model_save_freq = config["model_save_freq"]
    
    model_dir = f"Secondery_{config['model_type']}_s{sequence_length}_m{m}_e{num_epochs}_b{batch_size_training}_v{version}"
    model_save_dir = os.path.join(model_save_dir, model_dir)
    logs_dir = os.path.join(model_save_dir , "logs")
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    chunk_size = config["batch_size_robots"]  # e.g., 128 robots at a time

    # Initialize Weights & Biases
    wandb.login()
    wandb.init(project=config["wandb_project"], name=model_dir)

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

    train_robots, test_robots = train_test_split(robot_folders, test_size=test_ratio, random_state=seed)
    X_test, y_test = load_test_dataset(test_robots, all_input_features, all_output_features, config)
    y_test = y_test[:, :, output_indices]
    X_test = X_test[:, :, input_indices]

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

        dynamic_parameters_loggings = {}

        main_model = UnifiedModel(input_size , output_size, sequence_length=sequence_length, config=config, device=device)
        main_model.to(device)
        state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
        main_model.load_state_dict(state_dict)
        main_model.eval()

        config['m'] = 2
        config['dropout'] = 0.33
        config['num_heads'] = 8
        config['ff_dim'] = 32
        config['embed_dim'] = 32
        
        secondery_model = UnifiedModel(input_size , 1, sequence_length=sequence_length, config=config, device=device)
        secondery_model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(secondery_model.parameters(), lr=lr, weight_decay=3e-6)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size_training, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_training, shuffle=False)

        train_secondery_model(main_model , secondery_model, train_loader, val_loader, criterion, optimizer, device, model_save_dir, num_epochs=num_epochs , model_save_freq=model_save_freq)

        logs_save_path = os.path.join(logs_dir, f"log.pkl")
        with open(logs_save_path, 'wb') as file:
            pickle.dump(dynamic_parameters_loggings, file)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_training, shuffle=False)
        test_loss, test_mse, test_rmse, test_r2, test_pearson_corr = evaluate_secondery_model(main_model, secondery_model, test_loader, criterion, device, mode="Test", output_features=output_features)

        wandb.finish()



