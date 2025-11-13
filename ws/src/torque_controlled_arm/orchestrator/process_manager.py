import subprocess
import threading
import signal
import os

class ProcessManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.active_processes = []
        
    def run_command(self, command, capture_output=True, shell=False, cwd=None, source=None):
        self.orchestrator.log_message(f"Executing: {' '.join(command) if isinstance(command, list) else command}\n", 'command')
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=shell,
                cwd=cwd,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            self.active_processes.append(process)
            
            def stream_output(stream, is_stderr=False):
                try:
                    for line in iter(stream.readline, ''):
                        if line:
                            self.orchestrator.log_message(line, source)
                finally:
                    stream.close()

            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, False),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, True),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            return process
        except Exception as e:
            self.orchestrator.log_message(f"Error executing command: {e}\n", source)
            return None

    def kill_processes_by_name(self, names):
        for name in names:
            try:
                pids = subprocess.check_output(["pgrep", "-f", name]).decode().strip().split('\n')
                for pid in pids:
                    if pid.isdigit():
                        os.kill(int(pid), signal.SIGTERM)
                        self.orchestrator.log_message(f"Killed process {name} with PID {pid}\n", 'command')
            except subprocess.CalledProcessError:
                pass

    def cleanup(self):
        for process in self.active_processes:
            try:
                process.terminate()
            except:
                pass
        self.active_processes = []