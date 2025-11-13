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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

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
    # Make predictions
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_training"], shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu())
            all_targets.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Flatten sequences and joints for plotting
    num_sequences, num_joints, output_size = all_targets.shape
    all_preds_flat = all_preds.view(-1, output_size).numpy()
    all_targets_flat = all_targets.view(-1, output_size).numpy()

    # Create folder for plots
    output_folder = "pred_vs_real_plots"
    os.makedirs(output_folder, exist_ok=True)

    # Plot each output feature
    for i, feature in enumerate(output_features):
        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets_flat[:, i], all_preds_flat[:, i], alpha=0.5)
        plt.xlabel("Real Value")
        plt.ylabel("Predicted Value")
        plt.title(f"Real vs Predicted: {feature}")
        plt.plot([all_targets_flat[:, i].min(), all_targets_flat[:, i].max()],
                 [all_targets_flat[:, i].min(), all_targets_flat[:, i].max()],
                 'r--', lw=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{feature}_pred_vs_real.png"))
        plt.close()

    print(f"Plots saved in folder: {output_folder}")


    # Create folder for plots
    output_folder = "pred_vs_real_3d_surface"
    os.makedirs(output_folder, exist_ok=True)

    # Plot each output feature as a 3D surface
    for i, feature in enumerate(output_features):
        x = all_targets_flat[:, i]
        y = all_preds_flat[:, i]

        # Compute point density using gaussian KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)

        # Create grid for surface
        x_grid = np.linspace(x.min(), x.max(), 100)
        y_grid = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # Define views (elevation, azimuth, roll)
        views = [
            (30, -135, "_v1"),
            (60, -135, "_v2"),
            (90, -135, "_v3")
        ]

        for elev, azim, suffix in views:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

            # Set labels and title
            ax.set_xlabel('Real Value')
            ax.set_ylabel('Predicted Value')
            ax.set_zlabel('Density')
            ax.set_title(f'3D Density Surface: {feature}')

            # Apply elevation and azimuth
            ax.view_init(elev=elev, azim=azim)

            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Density')

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{feature}_3d_surface{suffix}.png"))
            plt.close()

    print(f"3D surface plots saved in folder: {output_folder}")

    # Create folder for plots
    output_folder = "pred_vs_real_3d_plots"
    os.makedirs(output_folder, exist_ok=True)

    # Plot each output feature as 3D density
    for i, feature in enumerate(output_features):
        x = all_targets_flat[:, i]
        y = all_preds_flat[:, i]

        # Compute point density using gaussian KDE
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort points by density for better visualization
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('Real Value')
        ax.set_ylabel('Predicted Value')
        ax.set_zlabel('Density')
        ax.set_title(f'3D Density: {feature}')
        fig.colorbar(sc, ax=ax, label='Density')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{feature}_3d_density.png"))
        plt.close()

    print(f"3D density plots saved in folder: {output_folder}")
