# Dynamic Parameter Estimation Project Documentation - Execution Flow Guide

This document provides a concise guide on the execution flow for the Dynamic Parameter Estimation Project, focusing on the order of operations and key configurations.

---

## Setup: Conda Environments

This project uses three Conda environments for different stages:

| Environment | Purpose | YAML file |
|------------|---------|-----------|
| `rgen`     | Robot generation | `rgen.yml` |
| `tgen`     | Trajectory generation | `tgen.yml` |
| `dlenv`    | Deep learning | `dlenv.yml` |

### Create the Conda environments

```bash
# Create robot generation environment
conda env create -f rgen.yml

# Create trajectory generation environment
conda env create -f tgen.yml

# Create deep learning environment
conda env create -f dlenv.yml
```

### Activate an environment

Example: activate robot generation environment

```bash
conda activate rgen
```

## Section 1: Robot Folder Generation

First, activate the environment:

```bash
conda activate rgen
```

This section describes how to generate robot definition files. The main script for this is located in `robot_generator/main.py`.

### Step 1: Configure Robot Generation

The primary function used is `generate_n_kinematically_constant_robots`, which calls `make_robot_arm` internally. You can find the example usage in `robot_generator/main.py`.

**Example Configuration in `robot_generator/main.py`:**

```python
from robot_generator.robot.robot_arm_pack import generate_n_kinematically_constant_robots

def main():
    generate_n_kinematically_constant_robots(
        512, # Number of robots to generate
        "./robots", # Output directory for robot files
        [15 , 7.5 , 30 , 25 , 10 , 7.5 , 10], # Link lengths
        initial_diameter_range=(6, 12), # Range for initial diameters
        com_bias_range=(-0.35/6, 0), # Range of values defining link shape
        joints=["z" , "y" , "y" , "z" , "y" , "z"], # Rotation axis for each joint
    )

main()
```

### Step 2: Run Robot Generation

Execute the `main.py` script to generate the robot folders:

```bash
python3 robot_generator/main.py
```

This will generate robot folders (e.g., `robot_0`, `robot_1`, etc.) in the specified `parent_dir` (e.g., `./robots`).

### Step 3: Adapt Robots for Gazebo

After generation, the robots need to be adapted for Gazebo. The script for this is located at `robot_generator/robot_generator/gazebo/gazebo_adapter.py`.

**Usage:**

```python
from robot_generator.gazebo.gazebo_adapter import process_robot_pack

# Assuming 'output_dir' is the directory where robots were generated in Step 1
output_dir = "./robots"
process_robot_pack(output_dir)
```

**To run this adaptation:**

```bash
python3 -c "from robot_generator.gazebo.gazebo_adapter import process_robot_pack; process_robot_pack('./robots')"
```

This process generates a `robot_GA.urdf` file within each robot's folder, making it compatible with Gazebo. The typical folder structure after this step is:

```
Robots_main_folder:
    robot_i:
        stl/     -> STL meshes
        dae/     -> DAE meshes (includes color values)
        json/    -> Kinematic information (joint axes, link structure)
        stp/     -> Meshes for inertial parameter calculation (mass, inertia, COM)
        robot.urdf       -> Base URDF
        robot_GA.urdf    -> Gazebo-adapted URDF
```

---

## Section 2: Gazebo Trajectory Generator

First, activate the environment:

```bash
conda activate tgen
```

and colcon build the workspace

```bash
cd ws
colcon build
```

This component automates running trajectories for the generated robots inside Gazebo with ROS, collecting raw trajectory data. The main script is `ws/src/torque_controlled_arm/orchestrator/run.py`.

### Step 1: Configure `settings.yaml`

#### Experiment Control

* **`num_robots_in_experiment`**: Number of robots to simulate in one run.
* **`initial_robot_index`**: Starting index of robots (useful if not starting from 0).
* **`skip_generated_robots`**: If `True`, skips trajectory generation when the output folder already exists.
* **`max_failed_trajectories_per_robot`**: Maximum number of failed trajectories allowed per robot before skipping it.
* **`reset_gazebo_every`**: Number of robots after which Gazebo/ROS is reset to avoid cache buildup (e.g., 4).

#### Trajectory Generation

* **`num_trajectory_points_per_robot`**: Number of target points per robot. Robots follow a PID-generated path across these points.
* **`interpolation_steps`**: Number of interpolated steps between each pair of target points.
* **`intermediate_position_tolerance`**: Position tolerance for intermediate interpolated steps (used to avoid overshoot from proportional value of PID controller).
* **`target_position_tolerance`**: Position tolerance at each target point to consider it reached.
* **`target_velocity_tolerance`**: Velocity tolerance at each target point to consider it reached.
* **`timeout_trajectory_point`**: Timeout (in seconds) before triggering recovery if a target is not reached.
* **`interpolated_steps_latency_percentage`**: Synchronization tolerance between joints. Example: if set to `0.5` and `interpolation_steps=12`, then if a joint hasn’t reached step 3, others won’t proceed past step 9.

#### Dynamics and Friction

* **`coulomb_friction_coeff_range`**: Range of Coulomb friction coefficients applied to joints.
* **`coulomb_friction_decay_factor`**: Scaling factor for reducing Coulomb friction toward the distal joints.
* **`viscous_friction_coeff_range`**: Range of viscous friction coefficients applied to joints.

#### Execution and Control

* **`min_update_rate`**: Minimum acceptable update rate (Hz). Default update is 1 kHz; if it falls below this, the robot may be stuck, triggering recovery.
* **`print_frequency`**: Frequency of logging progress (in iterations).
* **`sampling_frequency`**: Subsampling factor for the 1 kHz control loop. Typically set to 1 (use every step).
* **`pid_velocity_limit_over_pi`**: Velocity limit for joints, in multiples of π rad/s.

#### Directories

* **`dataset_output_dir`**: Directory where datasets will be saved.
* **`robots_dir`**: Directory containing the robot definitions.


### Step 2: Run Trajectory Generation

Execute the orchestrator script:

```bash
python3 ws/src/torque_controlled_arm/orchestrator/run.py
```

This will simulate robots in Gazebo, generate trajectories, and save the raw data. The output structure for each robot will be:

```
output_dir:
   robot_i:
      dynamic_parameters.yaml   # Organized dynamic parameters for dataset generation
      target_points.yaml        # List of target points per joint (kept for analysis)
      trajectory_data.csv       # Time-series trajectory data
```

---

## Section 3: Dataset Preprocessor

First, activate the environment:

```bash
conda activate dlenv
```

This stage transforms raw trajectory datasets into a structured and normalized format suitable for deep learning. The main scripts are in `deep_learning/data_preprocessor`.

### Step 1: Configure Shared Paths

Edit `deep_learning/config/shared.yaml` to define the main directories:

```yaml
robotics_dir: "/media/raid/robots/robots_p6" # Path to robot URDFs (from Section 1)
raw_dataset_dir: "/media/raid/raw_trajectories/dataset_p6" # Path to raw trajectory data (from Section 2)
preprocessed_dataset_dir: "/media/raid/processed_trajectories" # Output directory for preprocessed data
```

### Step 2: Configure Data Preprocessing Parameters

Edit `deep_learning/config/data_preprocessing.yaml` to control preprocessing behavior:

```yaml
initial_naming_index: 5632
robot_range:
  start: 1032
  end: 1052   # exclusive
include:
  link_positions: False
  link_orientations: False
  link_velocities: False
  link_angular_velocities: False
  jacobians: True # Set to True to include Jacobians
sampling_frequency: 1000
debug_sampling:
  generate_samples: True
  random_pick: False
  sample_size: 500
```


### Step 3: Analyze Features for Constancy

After `jacob.py` has run for at least one robot, use `features_analyzer.py` to identify constant or nearly constant columns. This needs to be run on a sample preprocessed CSV file.

```bash
# Replace with an actual path to a generated segment CSV, e.g., from preprocessed_dataset_dir/robot_X/trajectory_data_segment_0.csv
python3 deep_learning/data_preprocessor/features_analyzer.py --csv_path /path/to/your/trajectory_data_segment_0.csv --update_raw_yaml True
```

This generates `raw_feature_analysis.yaml` (used by other scripts) and `humanized_feature_analysis.yaml` in `deep_learning/config/data_preprocessor`.

### Step 4: Analyze Dynamic Parameters

Run `output_analyzer.py` to determine which dynamic parameters are truly varying based on the feature analysis:

```bash
python3 deep_learning/data_preprocessor/output_analyzer.py
```

This creates `dynamic_parameters_analysis.yaml` in `deep_learning/config/data_preprocessor`.

### Step 5: Run Jacobian and Feature Engineering

Execute `jacob.py` to resample trajectories, compute Jacobians, and filter initial data:

```bash
python3 deep_learning/data_preprocessor/jacob.py
```

This script processes robots based on `robot_range` in `data_preprocessing.yaml` and saves segmented trajectory data and dynamic parameters in the `preprocessed_dataset_dir`.

### Step 6: Normalize Dynamic Parameters

Execute `dynamic_normalizer.py` to apply min-max normalization to the dynamic parameters across all robots:

```bash
python3 deep_learning/data_preprocessor/dynamic_normalizer.py
```

This saves `dynamic_parameters_normalized.csv` for each robot in its preprocessed folder and `dynamic_param_normalization.yml` (containing normalization criteria) in `deep_learning/config/data_preprocessor`.

---

## Section 4: Deep Learning and Dataset Caching

make sure you activated the enviroment:

```bash
conda activate dlenv
```

This stage handles loading preprocessed datasets, caching them, training deep learning models (Transformer or Mamba), and saving checkpoints. The main training script is `deep_learning/deep_learning/main.py`.

### Step 1: Configure Deep Learning Parameters

Edit `deep_learning/config/deep_learning.yaml`. Key parameters include:

#### Global
*   **`seed`**: Random seed for reproducibility.
*   **`version`**: Experiment version identifier.

#### Dataset
*   **`dataset_root_path`**: Path to preprocessed datasets (from Section 3).
*   **`model_save_dir`**: Directory to save trained models.
*   **`continue`**: Set to `True` to resume training from a saved model.
*   **`batch_size_robots`**: Number of robots loaded at once for preprocessing.
*   **`train_size`**: Proportion of robots for training (e.g., 0.8).

#### Model
*   **`model_type`**: `"Transformer"` or `"Mamba"`.
*   **`input_features`**: Input features (e.g., `"all"` or a list).
*   **`input_features_exclude`**: Features to exclude from input.
*   **`output_features`**: Target features (e.g., `"analyzed"`).
*   **`embed_dim`**, **`num_joints`**, **`m`**, **`decoder_size`**, **`dropout`**: Model architecture parameters.

#### Training
*   **`num_epochs`**: Total training epochs.
*   **`batch_size_training`**: Training batch size.
*   **`lr`**: Learning rate.

#### Logging
*   **`wandb_project`**: Weights & Biases project name.

#### Model-Specific Parameters
*   **`transformer_specific`** / **`mamba_specific`**: Parameters like `sequence_length`, `sampling_rate`, `num_heads`, `ff_dim` (for Transformer) or `sequence_length`, `sampling_rate` (for Mamba). Note that `main.py` might iterate through a predefined list of these parameters, overriding the YAML values for iterative experimentation.

### Step 2: Run Deep Learning Training

Execute the main training script:

```bash
python3 deep_learning/deep_learning/main.py
```

This script will:

1.  Load preprocessed datasets from `dataset_root_path`.
2.  Prepare data for training and validation.
3.  Initialize and train the specified deep learning model.
4.  Save model checkpoints and logs to `model_save_dir`.
5.  Log progress and metrics to Weights & Biases (if configured).

---


