import time
import re
from pathlib import Path
import os
import yaml
import subprocess
import threading
import json

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def wait_until_controllers_reach_state(robot_name ,desired_state, controllers, timeout=5):
    if desired_state not in ("active", "inactive", "unloaded"):
        raise ValueError("desired_state must be 'active', 'inactive', or 'unloaded'")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(["ros2", "control", "list_controllers" , "-c" , f"{robot_name}/controller_manager"],
                                    capture_output=True, text=True, check=True)
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

class RobotManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.stream_output_buffer = ""
        self.stream_output_marker_index = 0
        self.send_buffer = False
        
    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', str(s))]
    
    def get_robot_id(self, robot_name):
        try:
            command = ["ign", "model", "-m", robot_name, "-p"]
            self.orchestrator.log_message(f"Checking robot ID for {robot_name}\n", 'command')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, _ = process.communicate()
            match = re.search(r"Model: \[([0-9]+)\]", output)
            if match:
                return match.group(1)
        except Exception as e:
            self.orchestrator.log_message(f"Error getting robot ID for {robot_name}: {e}\n", 'robot')
        return None

    def stop_and_unload_controllers_individually(self ,robot_name , controllers_to_remove):
        self.orchestrator.failed_robot = False  # Initialize flag

        for controller in controllers_to_remove:
            # --- Step 1: Deactivate with retries ---
            self.orchestrator.log_message(f"[INFO] Deactivating controller: {controller}")
            deactivate_success = False
            for attempt in range(1, 4):
                stop_command = [
                    "ros2", "service", "call", f"/{robot_name}/controller_manager/switch_controller",
                    "controller_manager_msgs/srv/SwitchController",
                    f"{{deactivate_controllers: ['{controller}'], strictness: 2}}"
                ]
                try:
                    subprocess.run(stop_command, check=True, timeout=3)
                    if wait_until_controllers_reach_state(robot_name ,"inactive", [controller]):
                        self.orchestrator.log_message(f"[INFO] Controller '{controller}' is inactive.")
                        deactivate_success = True
                        break
                except Exception as e:
                    self.orchestrator.log_message(f"[WARN] Failed to deactivate controller '{controller}' (attempt {attempt}): {e}")
            
            if not deactivate_success:
                self.orchestrator.log_message(f"[ERROR] Controller '{controller}' failed to deactivate after 3 attempts, trying to unload the controller now")

            # --- Step 2: Unload with retries ---
            self.orchestrator.log_message(f"[INFO] Unloading controller: {controller}")
            unload_success = False
            for attempt in range(1, 4):
                self.orchestrator.log_message(f"[DEBUG] Attempt {attempt} to unload '{controller}'")
                unload_command = [
                    "ros2", "service", "call", f"/{robot_name}/controller_manager/unload_controller",
                    "controller_manager_msgs/srv/UnloadController",
                    f"{{name: '{controller}'}}"
                ]
                try:
                    subprocess.run(unload_command, check=True, timeout=3)
                    if wait_until_controllers_reach_state(robot_name ,"unloaded", [controller]):
                        self.orchestrator.log_message(f"[INFO] Controller '{controller}' successfully unloaded.")
                        unload_success = True
                        break
                except Exception as e:
                    self.orchestrator.log_message(f"[WARN] Failed to unload controller '{controller}' (attempt {attempt}): {e}")
            
            if not unload_success:
                self.orchestrator.log_message(f"[ERROR] Controller '{controller}' failed to unload after 3 attempts.")
                self.orchestrator.failed_robot = True

    def delete_robot(self , robot_name , controllers_to_remove  = ['joint_state_broadcaster', 'torque_arm_controller']):
        robot_id = self.get_robot_id(robot_name)
        if robot_id:
            self.orchestrator.log_message(f"Deleting robot {robot_name} with ID {robot_id}...\n", 'robot')

            # Step 1&2: Delete robot entity from Gazebo
            self.stop_and_unload_controllers_individually(robot_name, controllers_to_remove = controllers_to_remove)
            
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
            
            wait_until_controllers_reach_state(robot_name ,"unloaded", controllers_to_remove)

    def spawn_robot(self, urdf_path, robot_name, position , controllers  = ['joint_state_broadcaster', 'torque_arm_controller']):
        for attempt in range(1, 4):
            self.orchestrator.log_message(f"[INFO] Attempt {attempt} Spawning robot '{robot_name}' from URDF at: {urdf_path}")
            spawn_command = [
                "ros2", "launch", "torque_controlled_arm", "robot.launch.py",
                f"robot_name:={robot_name}",
                f"urdf_path:={urdf_path}",
                f"x:={position.x}",
                f"y:={position.y}",
            ]

            try:
                process = subprocess.Popen(spawn_command)
                state = wait_until_controllers_reach_state(robot_name ,"active", controllers)
                if state:
                    self.orchestrator.log_message(f"Spawn process started for {robot_name} with PID: {process.pid}\n", 'robot')
                    return True , process
                else:
                    # self.delete_robot(robot_name)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    
            except Exception as e:
                self.orchestrator.log_message(f"Error spawning robot {robot_name}: {e}\n", 'robot')
        return False , None
    
    def stream_output(self, stream, process, is_stderr=False):
        try:
            while True:
                line = stream.readline()
                if not line:
                    if process.poll() is not None:
                        break  # Subprocess ended, and no more data
                    continue  # Wait for more output

                if "'''" in line:
                    self.stream_output_marker_index += 1
                    if self.stream_output_marker_index % 2 == 0:
                        self.stream_output_buffer += line
                        self.send_buffer = True

                if self.stream_output_marker_index % 2 == 1:
                    self.stream_output_buffer += line

                if "PROGRESS_JSON:" in line:
                    try:
                        progress_data = json.loads(line.split("PROGRESS_JSON:")[1])
                        self.orchestrator.update_trajectory_progress(
                            current=progress_data['current'],
                            total=progress_data['total'],
                            completion=progress_data['completion'],
                            failed=progress_data['failed'],
                            reason=progress_data.get('reason', None)
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        self.orchestrator.log_message(
                            f"Error parsing progress: {str(e)}",
                            'controller'
                        )

                if self.send_buffer:
                    self.orchestrator.append_controller_output(self.stream_output_buffer)
                    self.stream_output_buffer = ""
                    self.send_buffer = False

                if is_stderr or ('error' in line.lower() or 'warning' in line.lower()):
                    self.orchestrator.log_message(line, 'controller')

                if "Done" in line:
                    break  # End this thread
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def run_combined_controller_node(self, urdf_path, robot_name, position):
        self.orchestrator.log_message(f"Launching controller_node for {robot_name}...\n", 'controller')
        command = [
            "ros2", "run", "torque_controlled_arm", "controller_node",
            "--ros-args",
            "-p", f"robot_name:={robot_name}",
            "-p", f"urdf_path:= {urdf_path}",
            "-p", f"x:= {position.x}",
            "-p", f"y:= {position.y}",
            "-p", "plot:=false",
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_thread = threading.Thread(
                target=self.stream_output,
                args=(self.process.stdout, self.process, False),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self.stream_output,
                args=(self.process.stderr, self.process, True),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for the process to complete
            self.process.wait()

            # Ensure threads are done
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            self.stream_output_marker_index = 0

            print("done with threads")
            return self.process.returncode == 0

        except Exception as e:
            self.orchestrator.log_message(f"Error starting controller for {robot_name}: {e}\n", 'controller')
            return False
