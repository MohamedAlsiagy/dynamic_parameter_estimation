import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32
from ros_ign_interfaces.srv import ControlWorld
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from tabulate import tabulate
import os
import pybullet as p
import pybullet_data
import tempfile
from rcl_interfaces.srv import GetParameters
import xml.etree.ElementTree as ET
from io import StringIO
import random
import sys
from math import pi, nan
import csv
import yaml
import json
from pathlib import Path
import subprocess
import re
import psutil
import signal

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def kill_processes_by_name(name , exclude = "controller_node"):
    print(f"Trying to kill all processes matching '{name}'...")
    
    process_exist = False
    # First, try polite termination
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name in (proc.info.get('name') or '') or any(name in s for s in cmdline):
                if exclude in (proc.info.get('name') or '') or any(exclude in s for s in cmdline):
                    continue
                process_exist = True
                print(f"Killing process {proc.pid} ({proc.info['name']}) matching '{name}'")
                proc.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if process_exist:
        time.sleep(3)

    # Then, force kill if still running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name in (proc.info.get('name') or '') or any(name in s for s in cmdline):
                if exclude in (proc.info.get('name') or '') or any(exclude in s for s in cmdline):
                    continue
                print(f"Forcibly killing process {proc.pid} ({proc.info['name']}) matching '{name}'")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def is_successive_or_same(link_a, link_b):
    return abs(link_a - link_b) <= 1

class ControllerOutputFormatter:
    """Helper class to format controller output in a structured way"""
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.buffer = []
        
    def add_section(self, title, data=None):
        """Add a new section to the output"""
        self.buffer.append(f"\n=== {title} ===")
        if data:
            self.buffer.append(str(data))
            
    def add_table(self, headers, rows):
        """Add a table to the output"""
        self.buffer.append(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
        
    def add_metric(self, name, value):
        """Add a metric to the output"""
        self.buffer.append(f"{name}: {value}")
        
    def get_output(self):
        """Get the formatted output with markers"""
        return "'''\n" + "\n".join(self.buffer) + "\n'''"
        
    def clear(self):
        """Clear the buffer"""
        self.buffer = []

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def wait_until_controllers_reach_state(robot_name, desired_state, controllers, timeout=5):
    if desired_state not in ("active", "inactive", "unloaded"):
        raise ValueError("desired_state must be 'active', 'inactive', or 'unloaded'")
    if controllers == []:
        return True
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(["ros2", "control", "list_controllers" , "-c" , f"{robot_name}/controller_manager"],
                                    capture_output=True, text=True, check=True , timeout = 3)
            output = result.stdout.strip()
            output_clean = remove_ansi_escape_sequences(output)

            if desired_state in ["unloaded" , "inactive"]:
                for controller in controllers:
                    if controller not in output_clean:
                        return True  # Found controller still listed

            # Parse controller states for active/inactive check
            states = {}
            for line in output_clean.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] in controllers:
                    states[parts[0]] = parts[-1]

            if desired_state == "inactive":
                if all(states.get(c) in ("inactive", "unconfigured") for c in controllers):
                    return True
            else:
                if all(states.get(c) == desired_state for c in controllers):
                    return True

        except subprocess.CalledProcessError:
            pass

        time.sleep(0.5)

    print(f"[ERROR] Timeout: Controllers did not reach '{desired_state}' within {timeout} seconds.")
    return False

class CombinedControllerNode(Node):
    def __init__(self):
        super().__init__('combined_controller_node')
        
        # ROS parameters
        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("urdf_path", "/root/ws/src/torque_controlled_arm/robots/robot_0/robotGA.urdf")
        self.declare_parameter("plot", False)
        self.declare_parameter("x", 0)
        self.declare_parameter("y", 0)

        self.namespace = self.get_parameter("robot_name").get_parameter_value().string_value
        self.x = self.get_parameter("x").get_parameter_value().integer_value
        self.y = self.get_parameter("y").get_parameter_value().integer_value

        self.robot_name = self.namespace[:-2]

        self.urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        self.plot = self.get_parameter("plot").get_parameter_value().bool_value

        # Load settings
        base_dir = os.path.dirname(os.path.dirname(__file__))
        settings_file_path = os.path.join(base_dir, "config", "settings.yaml")
        with open(settings_file_path, 'r') as f:
            self.settings = yaml.safe_load(f)

        self.currently_failed = False
        self.failed_robot = False  # Initialize flag

        # Initialize configuration
        self._init_controller_params()
        self._init_data_structures()
        self._setup_pybullet()
        self._setup_ros_communications()

        # Output formatting
        self.output_formatter = ControllerOutputFormatter(self.robot_name)
        self.print_every = self.settings.get('print_frequency', 25)
        self.sampling_frequency = self.settings.get('sampling_frequency', 1)
        self.max_failed_trajectories = self.settings.get('max_failed_trajectories_per_robot', 3)
        self.update_count = 0

        # Start target generation
        self.get_logger().info(f"{self.num_trajectory_points_per_robot} random joint configurations are to be generated")
        
        self.setup_joint_trajectoriy_gen()
        self.prepare_next_trajectory()

    def _compute_dynamics(self, joint_positions):
        # Compute Jacobian for current joint positions:
        joint_positions_list = joint_positions.tolist()
        for i, q in zip(self.joint_indices, joint_positions_list):
            p.resetJointState(self.robot, i, q)

        zero_vec = [0.0] * len(self.joint_indices)
        jac_t, _ = p.calculateJacobian(
            self.robot,
            self.joint_indices[-1],
            localPosition=[0, 0, 0],
            objPositions=joint_positions_list,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jac_t = np.array(jac_t)
        self.dz_dj = jac_t[2, :]  # z-axis row of translational Jacobian

    def _limit_variable(self , x , limit):
        return max(-limit, min(x, limit))

    def _compute_joint_control(self, joint_index, position, velocity, dt):
        kp_joint_index = self.kpNG[joint_index] if self.dz_dj[joint_index] > 0 else self.kpWG[joint_index]
        error = self.target_angles[joint_index] - position
        self.integral[joint_index] += error * dt
        derivative = -velocity

        PI = (
            kp_joint_index * error +
            self.ki[joint_index] * self.integral[joint_index]
        )
        
        D = self.kd[joint_index] * derivative
        D = self._limit_variable(D , self.pid_velocity_limit)

        torque = PI + D

        return error, torque

    def compute_tip_position(self, joint_values):
        for i, q in zip(self.joint_indices, joint_values):
            p.resetJointState(self.robot, i, q)
        link_state = p.getLinkState(self.robot, self.joint_indices[-1])
        return link_state[4]

    def setup_joint_trajectoriy_gen(self):
        self.robot_base_pose = self.compute_tip_position([0.0] * len(self.joint_indices))
        ref_pose = self.compute_tip_position([0.0, pi/2, 0.0, 0.0, 0.0, 0.0])
        ref_dist = np.linalg.norm(np.array(ref_pose) - np.array(self.robot_base_pose))
        self.threshold_distance = 0.25 * ref_dist

        self.gen_attempts = 0
        self.last_valid_config = [0.0] * len(self.joint_indices)

    def generate_random_config(self):
        joint_positions = [random.uniform(-pi, pi) for _ in self.joint_indices]
        
        # Modify some joints to ensure interpolation variety
        for i in [1, 2]:  # Specific to original robot
            joint_positions[i] = self.last_valid_config[i] + random.uniform(-pi, pi) / 2

        # Check interpolated trajectory for Z validity
        for step in range(self.interpolation_steps):
            interp_joint_positions = [
                (1 - (step + 1) / self.interpolation_steps) * self.last_valid_config[j] +
                ((step + 1) / self.interpolation_steps) * joint_positions[j]
                for j in range(len(joint_positions))
            ]
            for i, q in zip(self.joint_indices, interp_joint_positions):
                p.resetJointState(self.robot, i, q)

            for i in range(3, p.getNumJoints(self.robot)):
                link_pos = p.getLinkState(self.robot, i)[4]
                if link_pos[2] < 0.6:
                    return None

        # Set the full configuration
        for i, q in zip(self.joint_indices, joint_positions):
            p.resetJointState(self.robot, i, q)

        # Check for self-collisions
        contacts = p.getClosestPoints(bodyA=self.robot, bodyB=self.robot, distance=0.001)
        real_collisions = [
            c for c in contacts if c[3] != c[4] and not is_successive_or_same(c[3], c[4])
        ]
        if real_collisions:
            return None

        # Check tip distance from base
        tip_pos = self.compute_tip_position(joint_positions)
        if np.linalg.norm(np.array(tip_pos) - np.array(self.robot_base_pose)) < self.threshold_distance:
            return None

        return joint_positions


    def prepare_next_trajectory(self):
        if len(self.valid_configs) <= self.num_trajectory_points_per_robot:
            result = None
            while result == None:
                result = self.generate_random_config()
                self.gen_attempts += 1
            self.valid_configs.append(result)
            self.last_valid_config = result
        else:
            self.get_logger().info("No more targets. Stopping contol.")
            self.save_robot_dataset()

            if self.plot:
                self.plot_all()

            msg = Int32()
            msg.data = 0 # Indicate completion
            self.trajectory_termination_pub.publish(msg)

            try:
                controllers = ['joint_state_broadcaster', 'torque_arm_controller']
                self.delete_robot_and_controllers(controllers)
            except Exception as e:
                print(f"Error during robot/controller deletion: {e}")

            self.done = True
            print(f"'''{self.robot_name} Done\n'''")
            return

        next_target = self.valid_configs[-1]
        self.all_target_points_followed.append(next_target) # Store the target point
        self.current_main_target = next_target

        self.joint_steps = [0] * 6
        current = self.target_angles

        for joint in range(6):
            self.step_targets[joint] = [
                (1 - ((step + 1) / self.interpolation_steps)) * current[joint] +
                ((step + 1) / self.interpolation_steps) * next_target[joint]
                for step in range(self.interpolation_steps)
            ]
            self.target_angles[joint] = self.step_targets[joint][0]

        self.get_logger().info(f"Starting interpolation to target {self.current_target_index}: {next_target}")
        
        self.reset_pid_state()

    def _init_controller_params(self):
        """Initialize controller parameters from settings"""
        # Timing and thresholds
        self.timeout_trajectory_point = self.settings['timeout_trajectory_point']
        self.min_update_rate = self.settings.get('min_update_rate', 100)  # Hz
        self.interpolation_steps = self.settings['interpolation_steps']
        self.num_trajectory_points_per_robot = self.settings['num_trajectory_points_per_robot']
        self.dataset_output_dir = self.settings['dataset_output_dir']
        
        # PID parameters
        self.kpNG = [5.0, 100.0, 100.0, 3.0, 3.0, 1.0] # 5.0
        self.kpWG = [5.0, 75.0, 75.0, 1.0, 1.0, 1.0]
        self.ki = [0.4, 10.0, 10.0, 0.10, 1.00, 0.10]
        self.kd = [0.4, 5.00, 5.00, 0.25, 0.25, 0.05]
        
        # Tolerance values
        self.pos_tol_rough_val = self.settings['intermediate_position_tolerance']
        self.pos_tol_final_val = self.settings['target_position_tolerance']
        self.vel_tol_final_val = self.settings['target_velocity_tolerance']
        self.interpolated_steps_latency_percentage = self.settings['interpolated_steps_latency_percentage']
        
        self.latency_threshold = math.ceil(self.interpolation_steps * self.interpolated_steps_latency_percentage)

        self.pid_velocity_limit = self.settings['pid_velocity_limit_over_pi'] * pi

    def _init_data_structures(self):
        """Initialize data collection structures"""
        # Controller state
        self.target_velocity = [0.0 for _ in range(6)]
        self.integral = [0.0] * 6
        self.prev_time = time.time()
        self.global_start_time = time.time()
        self.current_target_start_time = time.time()
        self.update_rate = 500
        self.reach_counter = 0
        self.done = False
        self.min_joint_step = 0
        
        # Data logging
        self.all_times = []
        self.all_positions = [[] for _ in range(6)]
        self.all_velocities = [[] for _ in range(6)]
        self.all_targets = [[] for _ in range(6)]
        self.all_index = [[] for _ in range(6)]
        self.all_torques = [[] for _ in range(6)]
        self.all_target_points_followed = []
        
        # Temporary storage for current trajectory point
        self.current_times = []
        self.current_positions = [[] for _ in range(6)]
        self.current_velocities = [[] for _ in range(6)]
        self.current_targets = [[] for _ in range(6)]
        self.current_index = [[] for _ in range(6)]
        self.current_torques = [[] for _ in range(6)]

        # Target tracking
        self.valid_configs = []
        self.current_target_index = 0
        self.joint_steps = [0] * 6
        self.step_targets = [[0.0] * self.interpolation_steps for _ in range(6)]
        self.target_angles = [0.0] * 6
        self.current_main_target = [0.0] * 6
        self.failed_count = 0
        self.stuck_count = 0
        self.deleted_robot = False

    def _setup_pybullet(self):
        """Initialize PyBullet simulation"""
        p.connect(p.DIRECT) 
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = [i for i in range(self.num_joints) 
                             if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

        if len(self.joint_indices) < 6:
            self.get_logger().error("Robot must have at least 6 movable joints.")
            sys.exit(1)

    def _setup_ros_communications(self):
        """Set up ROS publishers and subscribers"""
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, f'/{self.namespace}/joint_states', self.joint_state_callback, 1)
            
        # Publishers
        self.command_pub = self.create_publisher(
            Float64MultiArray, f'/{self.namespace}/torque_arm_controller/commands', 1)

        self.trajectory_termination_pub = self.create_publisher(
            Int32, f'/{self.namespace}/generation_state', 10)
        
    def shutdown_ros_communications(self):
        """Clean up ROS publishers and subscribers"""
        if hasattr(self, 'joint_state_sub') and self.joint_state_sub is not None:
            self.destroy_subscription(self.joint_state_sub)
            self.joint_state_sub = None

        if hasattr(self, 'command_pub') and self.command_pub is not None:
            self.destroy_publisher(self.command_pub)
            self.command_pub = None

        if hasattr(self, 'trajectory_termination_pub') and self.trajectory_termination_pub is not None:
            self.destroy_publisher(self.trajectory_termination_pub)
            self.trajectory_termination_pub = None


    def get_robot_id(self, robot_name):
        try:
            command = ["ign", "model", "-m", robot_name, "-p"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, _ = process.communicate()
            match = re.search(r"Model: \[([0-9]+)\]", output)
            if match:
                return match.group(1)
        except Exception as e:
            self.get_logger().info(f"Error getting robot ID for {robot_name}: {e}\n")
        return None

    def stop_and_unload_controllers_individually(self , controllers_to_remove):
        robot_name = self.namespace
        for controller in controllers_to_remove:
            # --- Step 1: Deactivate with retries ---
            print(f"[INFO] Deactivating controller: {controller}")
            deactivate_success = False
            for attempt in range(1, 4):
                print(f"[DEBUG] Attempt {attempt} to deactivate '{controller}'")
                stop_command = [
                    "ros2", "service", "call", f"/{robot_name}/controller_manager/switch_controller",
                    "controller_manager_msgs/srv/SwitchController",
                    f"{{deactivate_controllers: ['{controller}'], strictness: 2}}"
                ]
                try:
                    subprocess.run(stop_command, check=True, timeout=3)
                    if wait_until_controllers_reach_state(robot_name, "inactive", [controller]):
                        print(f"[INFO] Controller '{controller}' is inactive.")
                        deactivate_success = True
                        break
                except Exception as e:
                    print(f"[WARN] Failed to deactivate controller '{controller}' (attempt {attempt}): {e}")
            
            if not deactivate_success:
                print(f"[ERROR] Controller '{controller}' failed to deactivate after 3 attempts, trying to unload the controller now")

            # --- Step 2: Unload with retries ---
            print(f"[INFO] Unloading controller: {controller}")
            unload_success = False
            for attempt in range(1, 4):
                print(f"[DEBUG] Attempt {attempt} to unload '{controller}'")
                unload_command = [
                    "ros2", "service", "call", f"/{robot_name}/controller_manager/unload_controller",
                    "controller_manager_msgs/srv/UnloadController",
                    f"{{name: '{controller}'}}"
                ]
                try:
                    subprocess.run(unload_command, check=True, timeout=3)
                    if wait_until_controllers_reach_state(robot_name, "unloaded", [controller]):
                        print(f"[INFO] Controller '{controller}' successfully unloaded.")
                        unload_success = True
                        break
                except Exception as e:
                    print(f"[WARN] Failed to unload controller '{controller}' (attempt {attempt}): {e}")
            
            if not unload_success:
                print(f"[ERROR] Controller '{controller}' failed to unload after 3 attempts.")
                self.failed_robot = True

    def delete_robot(self , controllers_to_remove):
        robot_name = self.namespace
        urdf_path = self.urdf_path

        robot_id = self.get_robot_id(robot_name)

        # Step 3: Delete robot entity from Gazebo
        delete_command = [
            "ros2", "service", "call", "/world/default/remove",
            "ros_gz_interfaces/srv/DeleteEntity",
            f"{{entity: {{id: {robot_id}, name: '{robot_name}', type: 0}}}}"
        ]
        try:
            subprocess.run(delete_command, check=True, timeout = 3)
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] Failed to delete robot '{robot_name}': {e}")
        
        wait_until_controllers_reach_state(robot_name, "unloaded", controllers_to_remove)
        
            
    def spawn_robot(self , controllers):
        robot_name = self.namespace
        urdf_path = self.urdf_path

        spawn_command = [
            "ros2", "launch", "torque_controlled_arm", "robot.launch.py",
            f"urdf_path:={urdf_path}", f"robot_name:={robot_name}", f"x:={self.x}", f"y:={self.y}",
        ]

        try:
            self.spawn_process = subprocess.Popen(spawn_command)
            state = wait_until_controllers_reach_state(robot_name, "active", controllers)
            if state:
                print("[INFO] Robot spawned sucssefuly")
                self.deleted_robot = False
                return True
            else:
                self.delete_robot_and_controllers(controllers)
        except Exception as e:
            print(f"[ERROR] Failed to spawn robot '{robot_name}': {e}")

        return False
    
    def spawn_robot_model_only(self , controllers):
        robot_name = self.namespace
        urdf_path = self.urdf_path

        for attempt in range(1, 4):
            print(f"[INFO] Spawning robot '{robot_name}' from URDF at: {urdf_path}")

            spawn_command = [
                "ros2", "launch", "torque_controlled_arm", "robot_respawn.launch.py",
                f"robot_name:={robot_name}"
            ]

            try:
                self.spawn_process = subprocess.Popen(spawn_command)
                state = wait_until_controllers_reach_state(robot_name, "active", controllers)
                if state:
                    print("[INFO] Robot spawned succseefuly")
                    self.deleted_robot = False
                    return True
                else:
                    self.delete_robot_and_controllers(controllers)
            except Exception as e:
                print(f"[ERROR] Failed to spawn robot '{robot_name}': {e}")
        return False

    def delete_robot_and_controllers(self , controllers):
        if self.deleted_robot:
            return
        self.stop_and_unload_controllers_individually(controllers_to_remove = controllers)
        self.delete_robot(controllers)
        time.sleep(0.5)

        kill_processes_by_name(self.namespace)
        self.deleted_robot = True

    def _check_stuck_condition(self):
        """Check if the robot is stuck based on update rate"""
        if self.update_rate < self.min_update_rate:
            self.stuck_count += 1
            if self.stuck_count > 10:  # Require multiple consecutive low updates
                self.get_logger().warn(f"Low update rate detected: {self.update_rate:.1f} Hz")
                self.update_rate = 500
                return True
        else:
            self.stuck_count = 0
        return False

    def fail(self , type = 1):
        self.failed_robot = True
        msg = Int32()
        msg.data = type # Indicate failed robot type
        self.trajectory_termination_pub.publish(msg)

        self.done = True
        print(f"'''{self.robot_name} Done\n'''")

    def _handle_failure(self, reason):
        """Handle a failed trajectory point"""
        self.currently_failed = True
        self.failed_count += 1
        self.get_logger().error(f"Trajectory {self.current_target_index} failed: {reason}")
        
        # Publish failure progress
        progress_data = {
            'type': 'progress',
            'current': self.current_target_index,
            'total': self.num_trajectory_points_per_robot,
            'completion': 0.0,
            'failed': self.failed_count,
            'reason': reason
        }
        print(f"PROGRESS_JSON:{json.dumps(progress_data)}")

        # Move to next target
        controllers = ['joint_state_broadcaster', 'torque_arm_controller']
        self.delete_robot_and_controllers(controllers)
        # self.shutdown_ros_communications()
        # self.delete_robot([])

        if self.failed_count > self.max_failed_trajectories:
            self.fail(1)
            return
            
        # self.shutdown_ros_communications()
        self.namespace = self.robot_name + "_" + str(self.failed_count)
        self._setup_ros_communications()

        robot_spawned = self.spawn_robot(controllers)
        if not robot_spawned:
            self.fail(2)
            return

        # self.spawn_robot_model_only(controllers)      
        self._reset_current_trajectory_data()
        
        self.currently_failed = False
        self.valid_configs.pop()  # TODO: track failed trajectories
        self.prepare_next_trajectory()

        # Reset the current target start time
        self.current_target_start_time = time.time()

    def _generate_output(self, positions, velocities, torques, errors, dz_dj):
        """Generate formatted output for monitoring"""
        self.output_formatter.clear()
        
        # Basic info
        self.output_formatter.add_section("Robot Status", {
            'Name': self.robot_name,
            'Target': f"{self.current_target_index}/{self.num_trajectory_points_per_robot}",
            'Completion': f"{min(self.joint_steps)/self.interpolation_steps*100:.1f}%"
        })
        
        # Joint data table
        headers = ["Joint", "Εrror", "Velocity", "Torque", "Target", "Step", "dz/dj"]
        rows = []
        for i in range(6):
            step_info = "TR" if self.joint_steps[i] == self.interpolation_steps - 1 else str(self.joint_steps[i])
            rows.append([
                i,
                f"{errors[i]:+.3f}",
                f"{velocities[i]:+.3f}",
                f"{torques[i]:+.3f}",
                f"{self.target_angles[i]:+.3f}",
                step_info,
                f"{dz_dj[i]:+.4f}"
            ])
        self.output_formatter.add_table(headers, rows)
        
        # System metrics
        self.output_formatter.add_metric("Update Rate", f"{self.update_rate:.1f} Hz")
        self.output_formatter.add_metric("Time Elapsed", 
                                      f"{time.time() - self.current_target_start_time:.1f}s")
        
        return self.output_formatter.get_output()

    def _check_target_reached(self, i, error, velocity_i):
        pos_tol_rough = abs(error) < self.pos_tol_rough_val
        pos_tol_final = abs(error) < self.pos_tol_final_val
        vel_tol_final = abs(velocity_i) < self.vel_tol_final_val

        if self.joint_steps[i] < self.interpolation_steps - 1:
            if pos_tol_rough and self.joint_steps[i] - self.min_joint_step < self.latency_threshold:
                self.joint_steps[i] += 1
                self.target_angles[i] = self.step_targets[i][self.joint_steps[i]]

            return False
        else:
            if not (pos_tol_final and vel_tol_final):
                return False
        return True

    def _publish_commands(self, torques):
        msg = Float64MultiArray()
        msg.data = torques
        self.command_pub.publish(msg)

    def _commit_current_trajectory_data(self):
        """Commit the current trajectory point data to the main lists"""
        self.all_times.extend(self.current_times)
        for i in range(6):
            self.all_positions[i].extend(self.current_positions[i])
            self.all_velocities[i].extend(self.current_velocities[i])
            self.all_targets[i].extend(self.current_targets[i])
            self.all_index[i].extend(self.current_index[i])
            self.all_torques[i].extend(self.current_torques[i])
            # self.all_link_positions[i].extend(self.current_link_positions[i])
            # self.all_link_velocities[i].extend(self.current_link_velocities[i])
            # self.all_link_accelerations[i].extend(self.current_link_accelerations[i])
        
        # Commit Jacobian data
        # self.jacobian_collector.commit_current_data()

        # Reset current data
        self._reset_current_trajectory_data()

    def _reset_current_trajectory_data(self):
        """Reset the current trajectory data storage"""
        self.current_times = []
        self.current_positions = [[] for _ in range(6)]
        self.current_velocities = [[] for _ in range(6)]
        self.current_targets = [[] for _ in range(6)]
        self.current_index = [[] for _ in range(6)]
        self.current_torques = [[] for _ in range(6)]

    def joint_state_callback(self, msg):
        """Main control loop callback"""
        if self.done:
            return
        
        if self.currently_failed:
            return

        # Check for timeout on this trajectory point
        current_time = time.time()
        if (current_time - self.current_target_start_time) > self.timeout_trajectory_point:
            self._handle_failure("timeout")
            return
            
        # Check for stuck condition
        if self._check_stuck_condition():
            self._handle_failure("stuck (low update rate)")
            return
            
        # Calculate time delta safely
        dt = current_time - self.prev_time
        if dt <= 0:
            return
            
        # Update rate calculation
        self.update_rate = 0.9 * self.update_rate + 0.1 * (1/dt)
        
        # Main control logic
        positions = np.array(msg.position[:6])
        velocities = msg.velocity[:6]
        torques = []
        errors = []
        final_target_reached = True
        
        self.min_joint_step = min(self.joint_steps)

        # Compute Jacobian and dynamics
        self._compute_dynamics(positions)

        # Control logic for each joint
        for i in range(6):
            error, torque = self._compute_joint_control(i, positions[i], velocities[i], dt)
            errors.append(error)
            torques.append(torque)
            
            # Check if target reached
            if not self._check_target_reached(i, error, velocities[i]):
                final_target_reached = False
        
        # Publish commands
        self._publish_commands(torques)

        if self.update_count % self.sampling_frequency == 0:
            self.current_times.append(current_time - self.global_start_time)
            for i in range(6):
                # Store data for CSV
                self.current_positions[i].append(positions[i])
                self.current_velocities[i].append(velocities[i])
                self.current_targets[i].append(self.current_main_target[i])
                self.current_index[i].append(self.current_target_index)
                self.current_torques[i].append(torques[i])
                # self.current_link_positions[i].append(link_positions[i])
                # self.current_link_velocities[i].append(link_velocities[i])
                # self.current_link_accelerations[i].append(link_accelerations[i])

        # Log data and update display periodically
        if self.update_count % self.print_every == 0:
            print(self._generate_output(positions, velocities, torques, errors, self.dz_dj))
            
            # Publish progress update
            progress_data = {
                'type': 'progress',
                'current': self.current_target_index,
                'total': self.num_trajectory_points_per_robot,
                'completion': min(self.joint_steps) / self.interpolation_steps,
                'failed': self.failed_count
            }
            print(f"PROGRESS_JSON:{json.dumps(progress_data)}")
            
        # Handle target completion
        if final_target_reached:
            self.reach_counter += 1
            if self.reach_counter >= 10:  # Require stable reached state
                self.current_target_index += 1

                self._commit_current_trajectory_data()

                self.current_target_start_time = time.time()
                self.prepare_next_trajectory()
        else:
            self.reach_counter = 0
            
        self.prev_time = current_time
        self.update_count += 1
        

    def reset_pid_state(self):
        self.integral = [0.0] * 6
        self.prev_time = time.time()

    def save_robot_dataset(self):
        robot_name = self.robot_name
        robot_output_dir = Path(self.dataset_output_dir) / robot_name
        robot_output_dir.mkdir(parents=True, exist_ok=True)

        self._save_trajectory_csv(robot_output_dir, robot_name)
        self._save_target_points_yaml(robot_output_dir, robot_name)
        self._save_dynamic_parameters_yaml(robot_output_dir, robot_name)

        # # Save Jacobian data
        # self.jacobian_collector.save_to_file()

        failure_rate = self.failed_count / (len(self.valid_configs) + self.failed_count) if len(self.valid_configs) > 0 else 0
        self.get_logger().info(f"Trajectory completion rate: {(1-failure_rate)*100:.1f}%")

    def _save_trajectory_csv(self, output_dir, robot_name):
        csv_file_path = output_dir / 'trajectory_data.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['time', 'index']
            for i in range(6):
                header.extend([f'Joint_{i}_Position', f'Joint_{i}_Velocity', f'Joint_{i}_Torque'])
            writer.writerow(header)

            num_samples = len(self.all_times)
            for i in range(num_samples):
                row = [self.all_times[i], self.all_index[0][i]]
                for j in range(6):
                    row.extend([
                        self.all_positions[j][i],
                        self.all_velocities[j][i],
                        self.all_torques[j][i]
                    ])
                writer.writerow(row)
        self.get_logger().info(f'Trajectory data saved to {csv_file_path}')

    def _save_target_points_yaml(self, output_dir, robot_name):
        target_points_file_path = output_dir / 'target_points.yaml'
        with open(target_points_file_path, 'w') as yamlfile:
            yaml.dump({f"T{idx}" : {f"J{j_idx}" : joint for j_idx , joint in enumerate(target)} for idx , target in enumerate(self.all_target_points_followed)}, yamlfile, indent=4)
        self.get_logger().info(f'Target points saved to {target_points_file_path}')

    def _save_dynamic_parameters_yaml(self, output_dir, robot_name):
        def extract_dynamic_parameters(urdf_path):
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            dynamic_parameters = {"joints": {}, "links": {}}

            for j_idx, joint in enumerate(root.findall("joint")):
                joint_name = joint.get("name")
                dynamics = joint.find("dynamics")
                if dynamics is not None:
                    damping = float(dynamics.get("damping", 0.0))
                    friction = float(dynamics.get("friction", 0.0))
                    dynamic_parameters["joints"][joint_name] = {
                        "idx": j_idx - 1,
                        "coulomb_friction_coeff": friction,
                        "viscous_friction_coeff": damping
                    }

            for idx, link in enumerate(root.findall("link")):
                link_name = link.get("name")
                inertial = link.find("inertial")
                if inertial is not None:
                    mass = float(inertial.find("mass").get("value", 0.0))
                    inertia = inertial.find("inertia")
                    ixx = float(inertia.get("ixx", 0.0))
                    ixy = float(inertia.get("ixy", 0.0))
                    ixz = float(inertia.get("ixz", 0.0))
                    iyy = float(inertia.get("iyy", 0.0))
                    iyz = float(inertia.get("iyz", 0.0))
                    izz = float(inertia.get("izz", 0.0))
                    com_xyz = {"xyz"[i]: float(x) for i, x in enumerate(inertial.find("origin").get("xyz", "0 0 0").split())}

                    dynamic_parameters["links"][link_name] = {
                        "idx": idx,
                        "mass": mass,
                        "ixx": ixx,
                        "ixy": ixy,
                        "ixz": ixz,
                        "iyy": iyy,
                        "iyz": iyz,
                        "izz": izz,
                        "com": com_xyz
                    }

            return dynamic_parameters

        params = extract_dynamic_parameters(self.urdf_path)
        dyn_param_file = output_dir / "dynamic_parameters.yaml"
        with open(dyn_param_file, "w") as f:
            yaml.dump(params, f, indent=4)
        self.get_logger().info(f"Dynamic parameters saved to {dyn_param_file}")

    def plot_all(self):
        output_dir = Path(self.dataset_output_dir) / self.robot_name / "plotted_trajectories" 
        output_dir.mkdir(parents=True, exist_ok=True)
        for i_plt in range(6):
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (rad)', color='blue')
            ax1.plot(self.all_times, self.all_positions[i_plt], label='Position (rad)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            for i, t in enumerate(self.all_times):
                if i == 0 or self.all_targets[i_plt][i] != self.all_targets[i_plt][i - 1]:
                    ax1.axhline(y=self.all_targets[i_plt][i], linestyle='--', color='green', alpha=0.5)
                    ax1.text(t, self.all_targets[i_plt][i] + 0.05, f'target({self.all_index[i_plt][i]}):{self.all_targets[i_plt][i]:.2f}', color='green')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Velocity (rad/s)', color='orange')
            ax2.plot(self.all_times, self.all_velocities[i_plt], label='Velocity (rad/s)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            plt.title('Full Trajectory Over All Targets')
            fig.tight_layout()
            plt.grid(True)
            plt.savefig(output_dir / f'joint{i_plt}_pid_full_trajectory.png')
            plt.close()
        self.get_logger().info(f"Trajectory plots saved to {output_dir}")

def main(args=None):
    rclpy.init(args=args)
    node = CombinedControllerNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        try:
            controllers = ['joint_state_broadcaster', 'torque_arm_controller']
            node.delete_robot_and_controllers(controllers)
        except Exception as e:
            print(f"Error during robot/controller deletion: {e}")


        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()