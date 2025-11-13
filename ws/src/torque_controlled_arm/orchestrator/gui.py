import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import time
import psutil
import re
from .process_manager import ProcessManager
from .robot_manager import RobotManager
from .core import DatasetOrchestrator
import threading
import os
import psutil

class DatasetTrackerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Dataset Collection Controller")
        root.geometry("1200x800")
        
        # Configure style with bold fonts
        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'), padding=[10, 5])
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 11))
        self.style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 12, 'bold'))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Initialize orchestrator first
        self.orchestrator = DatasetOrchestrator(self)
        
        # Then let orchestrator create managers
        self.process_manager = self.orchestrator.process_manager
        self.robot_manager = self.orchestrator.robot_manager
        
        self.failed_trajectories = 0
        self.current_trajectory = 0
        self.total_trajectories = 0

        # Setup tabs
        self.setup_dataset_tab()
        self.setup_output_tab()
        self.setup_logs_tab()
        
        # Status bar
        self.setup_status_bar()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the process automatically
        self.start_main_thread()
    
    def update_trajectory_progress(self, current, total, completion, failed, reason=None):
        """Update the trajectory progress display with failure info"""
        self.current_trajectory = current
        self.total_trajectories = total
        
        # Update progress bar and labels
        self.trajectory_label.config(text=f"{current}/{total} targets completed")
        self.trajectory_progress['value'] = completion * 100
        self.failed_trajectories_label.config(text=f"Failed: {failed}")
        
        # Show failure reason if provided
        if reason:
            self.log_message(f"Trajectory {current} failed: {reason}\n", 'controller')
            failure_frame = ttk.Frame(self.trajectory_frame)
            failure_frame.pack(fill='x', pady=2)
            ttk.Label(failure_frame, text=f"Last failure: {reason}", 
                    foreground='red', font=('Helvetica', 9)).pack(side='left')
            
            # Remove previous failure message if exists
            if hasattr(self, 'last_failure_frame'):
                self.last_failure_frame.destroy()
            self.last_failure_frame = failure_frame
        
        self.root.update_idletasks()

    def increment_failed_trajectories(self):
        """Increment the failed trajectories counter"""
        self.failed_trajectories += 1
        self.failed_trajectories_label.config(text=f"Failed: {self.failed_trajectories}")
        self.root.update_idletasks()

    def add_orchestrator_event(self, text):
        """Add an event to the orchestrator output panel"""
        timestamp = time.strftime("%H:%M:%S")
        self.orchestrator_output.insert(tk.END, f"[{timestamp}] {text}\n", 'event')
        self.orchestrator_output.see(tk.END)
        self.root.update_idletasks()

    def _format_controller_output(self, content):
        """Apply formatting to controller output"""
        # Configure tags for different output parts
        self.subprocess_output_text.tag_config('section', foreground='cyan', font=('Helvetica', 10, 'bold'))
        self.subprocess_output_text.tag_config('metric', foreground='yellow')
        self.subprocess_output_text.tag_config('table', font=('Monospace', 9))
        
        # Process each line
        for line in content.split('\n'):
            if line.startswith('=== '):
                self.subprocess_output_text.insert(tk.END, line + '\n', 'section')
            elif ':' in line and any(c.isalpha() for c in line):
                # Likely a metric line
                self.subprocess_output_text.insert(tk.END, line + '\n', 'metric')
            else:
                # Treat as table or regular text
                self.subprocess_output_text.insert(tk.END, line + '\n', 'table')

    def append_controller_output(self, text):
        """Handle controller output display with better formatting"""
        if "'''" in text:
            # Clear previous output and insert new formatted output
            self.subprocess_output_text.delete("1.0", tk.END)
            
            # Extract content between markers and format with tags
            content = text.split("'''")[1]
            self._format_controller_output(content)
        else:
            # Append regular output
            self.subprocess_output_text.insert(tk.END, text)
        
        self.subprocess_output_text.see(tk.END)
        self.root.update_idletasks()

    def setup_status_bar(self):
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', padx=5, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Status: Initializing...", style='Status.TLabel')
        self.status_label.pack(side='left', padx=5)
        
        self.progress = ttk.Progressbar(self.status_frame, mode='determinate')
        self.progress.pack(side='right', fill='x', expand=True, padx=5)

    def setup_dataset_tab(self):
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset Generation')
        
        main_frame = ttk.Frame(self.dataset_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(title_frame, text="Dataset Generation Progress", style='Title.TLabel').pack(side='left')
        
        # Create a paned window for vertical split
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(expand=True, fill='both')
        
        # Top frame - current robot info
        top_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=1)
        
        # Current robot frame
        self.current_robot_frame = ttk.LabelFrame(top_frame, text="Current Robot")
        self.current_robot_frame.pack(fill='x', pady=5)
        
        self.robot_name_label = ttk.Label(self.current_robot_frame, text="None", font=("Helvetica", 14, 'bold'))
        self.robot_name_label.pack(pady=5)
        
        # Trajectory progress frame
        self.trajectory_frame = ttk.LabelFrame(top_frame, text="Trajectory Progress")
        self.trajectory_frame.pack(fill='x', pady=5)
        
        self.trajectory_progress = ttk.Progressbar(self.trajectory_frame, mode='determinate')
        self.trajectory_progress.pack(fill='x', padx=5, pady=5)
        
        self.trajectory_label = ttk.Label(self.trajectory_frame, text="0/0 targets completed")
        self.trajectory_label.pack()
        
        self.failed_trajectories_label = ttk.Label(self.trajectory_frame, text="Failed: 0")
        self.failed_trajectories_label.pack()
        
        # Bottom frame - system info
        bottom_frame = ttk.Frame(paned)
        paned.add(bottom_frame, weight=1)
        
        # Progress info
        self.progress_frame = ttk.LabelFrame(bottom_frame, text="Robot Progress")
        self.progress_frame.pack(fill='x', pady=5)
        
        self.robot_progress_label = ttk.Label(self.progress_frame, text="0/0 robots processed")
        self.robot_progress_label.pack(pady=5)
        
        # --- SYSTEM STATUS SPLIT VIEW ---
        self.system_status_frame = ttk.LabelFrame(bottom_frame, text="System Status")
        self.system_status_frame.pack(fill='x', pady=5, padx=10)

        # Paned window: two logical "tabs" side by side
        system_paned = ttk.PanedWindow(self.system_status_frame, orient=tk.HORIZONTAL)
        system_paned.pack(fill='both', expand=True)

        # --- LEFT SIDE: CPU & MEMORY INFO ---
        cpu_frame = ttk.Frame(system_paned, padding=10)
        system_paned.add(cpu_frame, weight=1)

        ttk.Label(cpu_frame, text="CPU & Memory Info", font=("Arial", 11, "bold")).pack(anchor='w')

        self.cpu_label = ttk.Label(cpu_frame, text="CPU Usage: -")
        self.cpu_label.pack(anchor='w', pady=5)

        self.mem_label = ttk.Label(cpu_frame, text="Memory Usage: -")
        self.mem_label.pack(anchor='w', pady=5)

        # --- RIGHT SIDE: THREAD MATRIX ---
        thread_frame = ttk.Frame(system_paned, padding=10)
        system_paned.add(thread_frame, weight=1)

        ttk.Label(thread_frame, text="CPU Threads", font=("Arial", 11, "bold")).pack(anchor='w')

        self.thread_canvas = tk.Canvas(thread_frame, width=280, height=240, bg="white", highlightthickness=1, relief="solid")
        self.thread_canvas.pack(fill='both', expand=True)

        self.update_thread_visualization()


        # Orchestrator output panel
        self.orchestrator_output_frame = ttk.LabelFrame(bottom_frame, text="Orchestrator Events")
        self.orchestrator_output_frame.pack(fill='both', expand=True, pady=5)
        
        self.orchestrator_output = ScrolledText(
            self.orchestrator_output_frame,
            height=8,
            font=("Monospace", 9),
            bg="black",
            fg="white"
        )
        self.orchestrator_output.pack(fill='both', expand=True, padx=5, pady=5)
        self.orchestrator_output.tag_config('event', foreground='cyan')
        
        
        self.update_system_info()

    def update_thread_visualization(self):
        self.thread_canvas.delete("all")

        total_threads = psutil.cpu_count(logical=True)
        used_threads = len(psutil.Process().threads())

        rows, cols = 4, 5
        box_size = 40
        padding = 10

        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            x1 = col * (box_size + padding) + padding
            y1 = row * (box_size + padding) + padding
            x2 = x1 + box_size
            y2 = y1 + box_size

            color = "red" if i < used_threads else "green"
            self.thread_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')

        self.root.after(2000, self.update_thread_visualization)


    def setup_output_tab(self):
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.output_tab, text='Controller Output')
        
        main_frame = ttk.Frame(self.output_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(title_frame, text="Controller Node Output", style='Title.TLabel').pack(side='left')
        
        # Configure text widget with better styling
        self.subprocess_output_text = ScrolledText(
            main_frame,
            height=25,
            width=120,
            wrap=tk.WORD,
            bg="black",
            fg="white",
            insertbackground="white",
            selectbackground="blue"
        )
        self.subprocess_output_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Configure tags for syntax highlighting
        self.subprocess_output_text.tag_config('section', foreground='cyan', font=('Helvetica', 10, 'bold'))
        self.subprocess_output_text.tag_config('metric', foreground='yellow')
        self.subprocess_output_text.tag_config('table', font=('Monospace', 9))
        self.subprocess_output_text.tag_config('error', foreground='red')
        self.subprocess_output_text.tag_config('warning', foreground='orange')

    def setup_logs_tab(self):
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_tab, text='System Logs')
        
        # Create inner notebook for different log types
        self.logs_notebook = ttk.Notebook(self.logs_tab)
        self.logs_notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create tabs for different components
        self.setup_component_log_tab('All', 'all_logs')
        self.setup_component_log_tab('Orchestrator', 'orchestrator')
        self.setup_component_log_tab('Gazebo', 'gazebo')
        self.setup_component_log_tab('Robot', 'robot')
        self.setup_component_log_tab('Controller', 'controller', show_info=False)
        
        # Setup filters and counters
        self.setup_log_filters()

    def setup_component_log_tab(self, name, source, show_info=True):
        """Create a tab for a specific component's logs"""
        tab = ttk.Frame(self.logs_notebook)
        self.logs_notebook.add(tab, text=name)
        
        # Create text widget for logs
        log_text = ScrolledText(tab, height=20, width=120, 
                              font=("Monospace", 10), bg="black")
        log_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Configure tags for different log levels
        log_text.tag_config('error', foreground='red')
        log_text.tag_config('warning', foreground='orange')
        log_text.tag_config('info', foreground='light gray')
        log_text.tag_config('command', foreground='cyan')
        log_text.tag_config('timestamp', foreground='yellow')
        
        # Store references
        setattr(self, f"{source}_text", log_text)
        setattr(self, f"{source}_logs", [])
        
        # Store configuration
        setattr(self, f"{source}_show_info", show_info)

    def setup_log_filters(self):
        """Setup filter controls for logs"""
        filter_frame = ttk.Frame(self.logs_tab)
        filter_frame.pack(fill='x', pady=5)
        
        self.show_errors = tk.BooleanVar(value=True)
        self.show_warnings = tk.BooleanVar(value=True)
        self.show_info = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(filter_frame, text="Errors", variable=self.show_errors, 
                       command=self.update_all_log_displays).pack(side='left', padx=5)
        ttk.Checkbutton(filter_frame, text="Warnings", variable=self.show_warnings,
                       command=self.update_all_log_displays).pack(side='left', padx=5)
        ttk.Checkbutton(filter_frame, text="Info", variable=self.show_info,
                       command=self.update_all_log_displays).pack(side='left', padx=5)
        
        counter_frame = ttk.Frame(self.logs_tab)
        counter_frame.pack(fill='x', pady=5)
        
        self.error_count = tk.IntVar(value=0)
        self.warning_count = tk.IntVar(value=0)
        self.info_count = tk.IntVar(value=0)
        
        ttk.Label(counter_frame, text="Errors:").pack(side='left', padx=5)
        ttk.Label(counter_frame, textvariable=self.error_count, foreground='red').pack(side='left', padx=5)
        
        ttk.Label(counter_frame, text="Warnings:").pack(side='left', padx=5)
        ttk.Label(counter_frame, textvariable=self.warning_count, foreground='orange').pack(side='left', padx=5)
        
        ttk.Label(counter_frame, text="Info:").pack(side='left', padx=5)
        ttk.Label(counter_frame, textvariable=self.info_count, foreground='white').pack(side='left', padx=5)
        
        ttk.Button(counter_frame, text="Clear All Logs", command=self.clear_all_logs).pack(side='right', padx=5)

    def update_system_info(self):
        cpu_percent = psutil.cpu_percent()
        mem_info = psutil.virtual_memory()
        
        self.cpu_label.config(text=f"CPU Usage: {cpu_percent}%")
        self.mem_label.config(text=f"Memory Usage: {mem_info.percent}% (Used: {mem_info.used/1024/1024:.1f} MB / Total: {mem_info.total/1024/1024:.1f} MB)")
        self.root.after(2000, self.update_system_info)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()

    def update_progress(self, current, total):
        self.robot_progress_label.config(text=f"{current}/{total} robots processed")
        self.progress['value'] = (current / total) * 100
        self.root.update_idletasks()

    def clear_all_logs(self):
        """Clear all log tabs"""
        for source in ['all_logs', 'orchestrator', 'gazebo', 'robot', 'controller']:
            text_widget = getattr(self, f"{source}_text")
            text_widget.delete('1.0', tk.END)
            setattr(self, f"{source}_logs", [])
        
        self.error_count.set(0)
        self.warning_count.set(0)
        self.info_count.set(0)

    def update_all_log_displays(self):
        """Update all log displays based on current filters"""
        for source in ['all_logs', 'orchestrator', 'gazebo', 'robot', 'controller']:
            self.update_log_display(source)

    def update_log_display(self, source):
        """Update a specific log display"""
        text_widget = getattr(self, f"{source}_text")
        logs = getattr(self, f"{source}_logs")
        show_info = getattr(self, f"{source}_show_info", True)
        
        text_widget.delete('1.0', tk.END)
        
        for entry in logs:
            text, tags = entry
            show = False
            
            if 'error' in tags and self.show_errors.get():
                show = True
            elif 'warning' in tags and self.show_warnings.get():
                show = True
            elif ('info' in tags or 'command' in tags) and self.show_info.get() and show_info:
                show = True
            
            if show:
                text_widget.insert(tk.END, text, tags)
        
        text_widget.see(tk.END)

    def log_message(self, text, source=None):
        timestamp = time.strftime("%H:%M:%S")
        full_text = f"[{timestamp}] {text}"
        
        # Determine log level and tags
        tags = []
        if source == 'command':
            tags = ['command', 'info']
            self.info_count.set(self.info_count.get() + 1)
        elif "error" in text.lower() or "exception" in text.lower():
            tags = ['error']
            self.error_count.set(self.error_count.get() + 1)
        elif "warning" in text.lower():
            tags = ['warning']
            self.warning_count.set(self.warning_count.get() + 1)
        else:
            tags = ['info']
            self.info_count.set(self.info_count.get() + 1)
        
        tags.append('timestamp')
        
        # Add to all logs
        self.add_log_entry('all_logs', full_text, tags)
        
        # Add to specific component log if source is specified
        if source and source != 'command':
            self.add_log_entry(source, full_text, tags)

    def add_log_entry(self, source, text, tags):
        """Add log entry to a specific source's log"""
        logs = getattr(self, f"{source}_logs")
        text_widget = getattr(self, f"{source}_text")
        
        # Limit log entries to prevent memory issues
        if len(logs) > 1000:
            logs.pop(0)
        
        logs.append((text, tuple(tags)))
        
        # Determine if we should show this entry
        show_info = getattr(self, f"{source}_show_info", True)
        show = False
        
        if 'error' in tags and self.show_errors.get():
            show = True
        elif 'warning' in tags and self.show_warnings.get():
            show = True
        elif ('info' in tags or 'command' in tags) and self.show_info.get() and show_info:
            show = True
            
        if show:
            text_widget.insert(tk.END, text, tuple(tags))
            text_widget.see(tk.END)
        
        self.root.update_idletasks()

    def on_closing(self):
        if messagebox.askyesno("Exit", "Are you sure you want to quit?\nAll robots and Gazebo will be terminated."):
            self.log_message("User requested shutdown. Cleaning up...\n", source='orchestrator')
            self.orchestrator.stop()
            self.process_manager.cleanup()
            self.root.destroy()
            os._exit(0)

    def update_current_robot(self, name):
        self.robot_name_label.config(text=name)
        self.root.update_idletasks()

    def start_main_thread(self):
        threading.Thread(target=self.orchestrator.run, daemon=True).start()
