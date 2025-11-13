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

from pathlib import Path
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

p.connect(p.DIRECT)

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def compute_cosine_similarity_per_point_per_output_all(
    model, loader, device,
    robot_id, urdf_joint_indices,
    joint_position_indices, joint_velocity_indices,
    last_link_index, output_size
):
    """
    Returns three lists of dictionaries (one per output) mapping 3D points to a list of cosine similarity scores:
    - positions
    - linear velocities
    - angular velocities
    """
    # Initialize dicts per output
    pos_scores_per_output = [{} for _ in range(output_size)]
    vel_scores_per_output = [{} for _ in range(output_size)]
    ang_vel_scores_per_output = [{} for _ in range(output_size)]

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)

            # Loop over samples in batch
            for i in range(X_batch.size(0)):
                x_seq = X_batch[i:i+1].cpu().numpy()
                y_seq = y_batch[i:i+1].cpu().numpy().flatten()
                y_pred_seq = outputs[i:i+1].cpu().numpy().flatten()

                # Get Cartesian positions, velocities, angular velocities
                cart_positions, _, cart_velocities, cart_angular_velocities = get_robot_cartesians(
                    x_seq, robot_id, urdf_joint_indices,
                    joint_position_indices, joint_velocity_indices,
                    last_link_index
                )

                # Flatten sequence
                cart_positions = cart_positions.reshape(-1, 3)
                cart_velocities = cart_velocities.reshape(-1, 3)
                cart_angular_velocities = cart_angular_velocities.reshape(-1, 3)

                # Loop over sequence points
                for j, pos in enumerate(cart_positions):
                    for out_idx in range(output_size):
                        # Scalar cosine similarity (1 if same, otherwise sign correlation)
                        true_val = float(y_seq[out_idx])
                        pred_val = float(y_pred_seq[out_idx])
                        cos_sim = 1.0 if np.isclose(true_val, pred_val, atol=1e-8) else true_val * pred_val / (abs(true_val) * abs(pred_val) + 1e-8)

                        key = tuple(pos)

                        # Position
                        if key not in pos_scores_per_output[out_idx]:
                            pos_scores_per_output[out_idx][key] = []
                        pos_scores_per_output[out_idx][key].append(cos_sim)

                        # Linear velocity
                        vel_key = tuple(cart_velocities[j])
                        if vel_key not in vel_scores_per_output[out_idx]:
                            vel_scores_per_output[out_idx][vel_key] = []
                        vel_scores_per_output[out_idx][vel_key].append(cos_sim)

                        # Angular velocity
                        ang_key = tuple(cart_angular_velocities[j])
                        if ang_key not in ang_vel_scores_per_output[out_idx]:
                            ang_vel_scores_per_output[out_idx][ang_key] = []
                        ang_vel_scores_per_output[out_idx][ang_key].append(cos_sim)

    # Average scores per point per output
    def average_scores(score_dicts):
        avg_scores = []
        for d in score_dicts:
            avg_scores.append({k: np.mean(v) for k, v in d.items()})
        return avg_scores

    avg_pos_scores = average_scores(pos_scores_per_output)
    avg_vel_scores = average_scores(vel_scores_per_output)
    avg_ang_vel_scores = average_scores(ang_vel_scores_per_output)

    return avg_pos_scores, avg_vel_scores, avg_ang_vel_scores

def plot_point_cloud_scores(avg_point_scores, output_file):
    """
    Plot 3D points colored by cosine similarity (green=high similarity)
    """
    points = np.array(list(avg_point_scores.keys()))
    scores = np.array(list(avg_point_scores.values()))
    if len(points) == 0:
        print(f"No points to plot for {output_file}")
        return

    # Normalize scores for color mapping
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    colors = np.zeros((len(points), 4))
    colors[:, 0] = 1 - scores_norm  # Red decreases as similarity increases
    colors[:, 1] = scores_norm      # Green increases with similarity
    colors[:, 2] = 0
    colors[:, 3] = scores_norm      # Alpha increases with similarity

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=5, depthshade=False)

    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved: {output_file}")

def get_robot_chunks(robots, chunk_size):
    for i in range(0, len(robots), chunk_size):
        yield robots[i:i + chunk_size]

def get_robot_cartesians(x, robot_id, urdf_joint_indices, joint_position_indices, joint_velocity_indices, last_link_index):
    """
    Convert joint states to end-effector Cartesian positions, orientations, and velocities.
    
    x shape: (batch_size, sequence_length, input_size)
    """
    batch_size, sequence_length, input_size = x.shape
    x = x.reshape(batch_size * sequence_length, input_size)

    all_joint_positions = x[:, joint_position_indices]
    all_joint_velocities = x[:, joint_velocity_indices]
    cart_positions, cart_orientations, cart_velocities, cart_angular_velocities = [], [], [], []

    for joint_positions, joint_velocities in zip(all_joint_positions, all_joint_velocities):
        for j, pos, vel in zip(urdf_joint_indices, joint_positions, joint_velocities):
            p.resetJointState(robot_id, j, pos, vel)

        link_state = p.getLinkState(robot_id, last_link_index, computeLinkVelocity=1)
        _, _, _, _, pos, quat, lin_vel, ang_vel = link_state

        cart_positions.append(pos)
        cart_orientations.append(quat)  
        cart_velocities.append(lin_vel)
        cart_angular_velocities.append(ang_vel)

    # Reshape back to (batch_size, sequence_length, ...)
    cart_positions = np.array(cart_positions).reshape(batch_size, sequence_length, 3)
    cart_orientations = np.array(cart_orientations).reshape(batch_size, sequence_length, 4)
    cart_velocities = np.array(cart_velocities).reshape(batch_size, sequence_length, 3)
    cart_angular_velocities = np.array(cart_angular_velocities).reshape(batch_size, sequence_length, 3)

    return cart_positions, cart_orientations, cart_velocities, cart_angular_velocities

if __name__ == "__main__":
    # Load configuration from YAML
    # === Load shared.yaml ===
    with open('/root/ws/deep_learning/config/shared.yaml', 'r') as f:
        shared_config = yaml.safe_load(f)

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

    joint_position_indices = [input_features.index(f"joint_{j}_position") for j in range(6)]
    joint_velocity_indices = [input_features.index(f"joint_{j}_velocity") for j in range(6)]

    print(joint_position_indices)
    print(joint_velocity_indices)

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

    ROBOTICS_DIR = Path(shared_config["robotics_dir"])
    robot_name = f"robot_{0}"
    robot_robotics_path = ROBOTICS_DIR / robot_name
    urdf_path = robot_robotics_path / "robotGA.urdf"
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)
    urdf_joint_indices = [i for i in range(p.getNumJoints(robot_id))
                    if p.getJointInfo(robot_id, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]
    last_link_index = list(urdf_joint_indices)[-1]


    # =========================
    # Example usage
    # =========================
    output_folder = Path("./cosine_pointclouds_per_output")
    output_folder.mkdir(exist_ok=True, parents=True)

    avg_pos_scores, avg_vel_scores, avg_ang_vel_scores = compute_cosine_similarity_per_point_per_output_all(
        model, test_loader, device,
        robot_id, urdf_joint_indices,
        joint_position_indices, joint_velocity_indices,
        last_link_index, output_size
    )    
    # Save the avg_point_scores_per_output as a .npz file
    np.savez(output_folder / "avg_pos_scores.npz", *avg_pos_scores)
    print(f"Saved avg_pos_scores to {output_folder / 'avg_pos_scores.npz'}")

    # Save the avg_point_scores_per_output as a .npz file
    np.savez(output_folder / "avg_vel_scores.npz", *avg_vel_scores)
    print(f"Saved avg_vel_scores to {output_folder / 'avg_vel_scores.npz'}")

    # Save the avg_point_scores_per_output as a .npz file
    np.savez(output_folder / "avg_ang_vel_scores.npz", *avg_ang_vel_scores)
    print(f"Saved avg_ang_vel_scores to {output_folder / 'avg_ang_vel_scores.npz'}")

    for out_idx, avg_pos_scores_for_output_i in enumerate(avg_pos_scores):
        output_file = output_folder / f"{output_features[out_idx]}.png"
        plot_point_cloud_scores(avg_pos_scores_for_output_i, output_file)
