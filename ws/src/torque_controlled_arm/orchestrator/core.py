import os
import yaml
from pathlib import Path
import time
from .robot_manager import RobotManager , Position
from .process_manager import ProcessManager
import tkinter as tk
import psutil
import signal
import psutil
import signal
import subprocess

def kill_processes_by_name(name):
    print(f"Trying to kill all processes matching '{name}'...")
    
    # First, try polite termination
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name in (proc.info.get('name') or '') or any(name in s for s in cmdline):
                print(f"Killing process {proc.pid} ({proc.info['name']}) matching '{name}'")
                proc.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(1)

    # Then, force kill if still running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name in (proc.info.get('name') or '') or any(name in s for s in cmdline):
                print(f"Forcibly killing process {proc.pid} ({proc.info['name']}) matching '{name}'")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


class DatasetOrchestrator:
    def __init__(self, gui_interface):
        self.gui = gui_interface
        self.robot_manager = RobotManager(self)
        self.process_manager = ProcessManager(self)
        self.running = False
        self.robot_process = None

        self.current_robot_spawn_failed = False

    def log_message(self, text, source=None):
        """Proxy method to pass messages to GUI with special handling for orchestrator events"""
        if source == 'orchestrator':
            # Add to orchestrator output panel
            self.gui.add_orchestrator_event(text)
        self.gui.log_message(text, source)

    def add_orchestrator_event(self, text):
        """Add an event to the orchestrator output panel"""
        self.gui.orchestrator_output.insert(tk.END, f"{text}\n", 'event')
        self.gui.orchestrator_output.see(tk.END)
        self.gui.root.update_idletasks()

    def update_status(self, message):
        """Proxy method to update GUI status"""
        self.gui.update_status(message)

    def update_trajectory_progress(self, current, total, completion, failed, reason=None):
        """Update trajectory progress in GUI with failure reason"""
        self.gui.update_trajectory_progress(
            current=current,
            total=total,
            completion=completion,
            failed=failed,
            reason=reason
        )
        
    def update_progress(self, current, total):
        """Proxy method to update GUI progress"""
        self.gui.update_progress(current, total)

    def update_current_robot(self, name):
        """Proxy method to update current robot in GUI"""
        self.gui.update_current_robot(name)

    def append_controller_output(self, text):
        """Proxy method for controller output"""
        self.gui.append_controller_output(text)

    def run(self):
        """Main execution method to be run in a thread"""
        if self.running:
            return

        self.running = True
        self.update_status("Starting dataset generation...")
        
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            settings_file_path = os.path.join(base_dir, "config", "settings.yaml")
            
            with open(settings_file_path, "r") as f:
                settings = yaml.safe_load(f)

            num_robots = settings.get("num_robots_in_experiment", 3)
            initial_robot_index = settings.get("initial_robot_index", 0)
            skip_generated_robots = settings.get("skip_generated_robots", True)
            dataset_output_dir = settings.get("dataset_output_dir", None)
            robots_dir = settings.get("robots_dir", "/root/ws/src/torque_controlled_arm/robots")
            reset_gazebo_every = settings.get("reset_gazebo_every", 5)
            
            robot_folders = self.get_robot_folders(robots_dir, num_robots)
            if not robot_folders:
                return

            self.process_robots(robot_folders , initial_robot_index , dataset_output_dir , skip_generated_robots , reset_gazebo_every)

        except Exception as e:
            self.log_message(f"Fatal error in orchestrator: {str(e)}\n", source='orchestrator')
        finally:
            self.running = False
            self.update_status("Ready")

    def get_robot_folders(self, robots_dir, num_robots):
        """Get and validate robot folders"""
        try:
            folders = [f.path for f in os.scandir(robots_dir) if f.is_dir()]
            folders = sorted(folders, key=self.robot_manager.natural_sort_key)
            return folders[:num_robots]
        except Exception as e:
            self.log_message(f"Error accessing robot directories: {e}\n", source='orchestrator')
            self.update_status("Failed to access robot directories")
            return []

    def process_robots(self, robot_folders , initial_robot_index = 0 , dataset_output_dir = None , skip_generated_robots = True , reset_gazebo_every = 5):
        gazebo_process = None
        for i, folder in enumerate(robot_folders[initial_robot_index:]):
            if i % reset_gazebo_every == 0 or self.current_robot_spawn_failed:
                if i != 0:
                    self.cleanup(gazebo_process)
                """Main processing loop for robots"""
                gazebo_process = self.launch_gazebo()

                if self.current_robot_spawn_failed:
                    self.current_robot_spawn_failed = False

            i += initial_robot_index
            if not self.running:  # Check if we should stop
                break
                
            robot_name = Path(folder).name

            # Check if robot is already processed
            if dataset_output_dir is not None and skip_generated_robots:
                robot_output_path = Path(dataset_output_dir) / robot_name
                if robot_output_path.exists():
                    print(f"[INFO] Skipping already processed robot: {robot_name}")
                    continue

            p = [0 , 2 , 0 , -2]
            position = Position(p[i%4] , p[(i + 1)%4])

            robot_name = robot_name + "_" + "0"
            self.current_robot_spawn_failed = not self.process_robot(i, len(robot_folders), folder, robot_name , position)

    def reset_ros2_daemon(self):
        subprocess.run(["ros2", "daemon", "stop"])
        time.sleep(1)
        subprocess.run(["ros2", "daemon", "start"])
        time.sleep(2)  # Give it a moment to initialize

    def launch_gazebo(self):
        """Launch Gazebo environment"""
        self.update_status("Launching Gazebo...")
        # self.reset_ros2_daemon()
        gazebo_cmd = ["ros2", "launch", "torque_controlled_arm", "bringup.launch.py", "ign_gz:=True"]
        process = self.process_manager.run_command(gazebo_cmd, source='gazebo')
        self.log_message("Launching Gazebo...\n", source='gazebo')
        time.sleep(10)  # Give Gazebo more time to start
        return process

    def process_robot(self, index, total, folder, robot_name , position):
        """Process a single robot"""
        self.update_current_robot(robot_name)
        self.update_progress(index+1, total)
        status_msg = f"Processing robot {index+1}/{total}: {robot_name}"
        self.update_status(status_msg)
        self.log_message(f"\n--- Processing Robot {index+1}/{total}: {robot_name} ---\n", 'orchestrator')
        self.add_orchestrator_event(f"Starting processing for robot: {robot_name}")

        urdf_path = os.path.join(folder, "robotGA.urdf")
        if not os.path.exists(urdf_path):
            error_msg = f"URDF file not found at {urdf_path}"
            self.log_message(f"{error_msg}\n", 'robot')
            self.add_orchestrator_event(f"ERROR: {error_msg}")
            return

        self.add_orchestrator_event(f"Spawning robot: {robot_name}")
        self.add_orchestrator_event(f"@ {urdf_path}")
        
        spawn_state , spawn_process = self.robot_manager.spawn_robot(urdf_path, robot_name , position)
        if spawn_state:
            self.add_orchestrator_event(f"Successfully spawned robot: {robot_name}")
            self.add_orchestrator_event(f"Starting controller for: {robot_name}")
            controller_state = self.robot_manager.run_combined_controller_node(urdf_path , robot_name , position)
            if controller_state:
                self.add_orchestrator_event(f"Deleting processed robot: {robot_name}")
                self.robot_manager.delete_robot(robot_name)
                self.add_orchestrator_event(f"Successfully deleted robot: {robot_name}")
            else:
                error_msg = f"Failed to start controller for {robot_name}"
                self.log_message(f"{error_msg}\n", 'controller')
                self.add_orchestrator_event(f"ERROR: {error_msg}")
            spawn_process.terminate()
        else:
            error_msg = f"Failed to spawn robot {robot_name}"
            self.log_message(f"{error_msg}\n", 'robot')
            self.add_orchestrator_event(f"ERROR: {error_msg}")
        
        self.add_orchestrator_event(f"Cleaning up processes containing: {robot_name}")
        kill_processes_by_name(robot_name[:-2] + "_")

        return spawn_state

    def cleanup(self, gazebo_process):
        self.add_orchestrator_event("Terminating Gazebo...")
        if gazebo_process:
            gazebo_process.terminate()
            gazebo_process.wait()

            kill_processes_by_name("ign")
            kill_processes_by_name("gazebo")
            kill_processes_by_name("ros")
            
            self.add_orchestrator_event("Gazebo terminated")

        self.add_orchestrator_event("Dataset generation completed!")
        self.update_status("Dataset generation completed")
        self.update_progress(1, 1)

    def stop(self):
        """Request the orchestrator to stop"""
        self.running = False