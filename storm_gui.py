#!/usr/bin/env python3
"""
STORM Microscopy GUI Application - Fixed Version

A user-friendly graphical interface for STORM microscopy analysis with
improved file browsing and Windows path handling.

Author: ItzWhole
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
import logging
from typing import Optional, List
import json

# Add storm_core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from storm_core.data_processing import find_tiff_files, crop_and_sum_stack, extract_psf_cutouts, normalize_0_to_1
    from storm_core.neural_network import (
        build_astigmatic_psf_network, train_val_split_by_group, 
        build_augmenter, make_dataset, setup_callbacks
    )
    from storm_core.evaluation import rescale_01_to_nm, plot_true_vs_pred_heatmap, plot_random_psfs
    import numpy as np
    import tifffile as tiff
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tensorflow as tf
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running in the storm_env virtual environment")
    sys.exit(1)

class STORMConfig:
    """Configuration class for STORM microscopy analysis"""
    
    def __init__(self):
        self.distance = 24
        self.start_z = 0
        self.end_z = 161
        self.csum_slices = 30
        self.min_distance = 5
        self.prominence_sigma = 10.0
        self.support_radius = 2
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 1e-3
        self.val_split = 0.2
        
        # Paths
        self.data_path = None
        self.output_path = Path("./output")
        self.model_path = None

class WindowsPathHelper:
    """Helper class for Windows path handling in WSL and Windows"""
    
    @staticmethod
    def is_wsl():
        """Check if running in WSL environment"""
        return os.path.exists("/mnt/c")
    
    @staticmethod
    def get_windows_username():
        """Get the current Windows username"""
        try:
            # In WSL, we need to detect the actual Windows user, not the WSL user
            if WindowsPathHelper.is_wsl():
                users_dir = "/mnt/c/Users"
                if os.path.exists(users_dir):
                    try:
                        users = os.listdir(users_dir)
                        # Filter to real user accounts (exclude system accounts and temp files)
                        real_users = []
                        for d in users:
                            user_path = os.path.join(users_dir, d)
                            if (os.path.isdir(user_path) and 
                                d not in ['Public', 'Default', 'All Users', 'Default User'] and
                                not d.startswith('TEMP') and 
                                not d.startswith('UMFD') and
                                d != 'WsiAccount' and
                                not d.endswith('.ini')):
                                # Check if it has typical user directories
                                if (os.path.exists(os.path.join(user_path, 'Desktop')) or
                                    os.path.exists(os.path.join(user_path, 'Documents'))):
                                    real_users.append(d)
                        
                        if real_users:
                            return real_users[0]  # Return first real user
                    except PermissionError:
                        # If we can't list users, try common usernames
                        common_names = ['milab', 'user', 'admin', 'administrator']
                        for name in common_names:
                            test_path = f"/mnt/c/Users/{name}/Desktop"
                            if os.path.exists(test_path):
                                return name
            else:
                # Running on Windows directly
                username = os.environ.get('USERNAME', os.environ.get('USER', ''))
                if username:
                    return username
            
            return None
        except Exception as e:
            print(f"Error detecting Windows username: {e}")
            return None
    
    @staticmethod
    def get_desktop_path():
        """Get Windows Desktop path"""
        if WindowsPathHelper.is_wsl():
            username = WindowsPathHelper.get_windows_username()
            if username:
                desktop_path = f"/mnt/c/Users/{username}/Desktop"
                if os.path.exists(desktop_path):
                    return desktop_path
            # Fallback to a safe directory
            return "/mnt/c/Users"
        else:
            # Running on Windows directly
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if os.path.exists(desktop):
                return desktop
            return os.path.expanduser("~")
    
    @staticmethod
    def get_documents_path():
        """Get Windows Documents path"""
        if WindowsPathHelper.is_wsl():
            username = WindowsPathHelper.get_windows_username()
            if username:
                docs_path = f"/mnt/c/Users/{username}/Documents"
                if os.path.exists(docs_path):
                    return docs_path
            return "/mnt/c/Users"
        else:
            # Running on Windows directly
            documents = os.path.join(os.path.expanduser("~"), "Documents")
            if os.path.exists(documents):
                return documents
            return os.path.expanduser("~")
    
    @staticmethod
    def get_downloads_path():
        """Get Windows Downloads path"""
        if WindowsPathHelper.is_wsl():
            username = WindowsPathHelper.get_windows_username()
            if username:
                downloads_path = f"/mnt/c/Users/{username}/Downloads"
                if os.path.exists(downloads_path):
                    return downloads_path
            return "/mnt/c/Users"
        else:
            # Running on Windows directly
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            if os.path.exists(downloads):
                return downloads
            return os.path.expanduser("~")
    
    @staticmethod
    def get_default_data_path():
        """Get default path for data browsing"""
        if WindowsPathHelper.is_wsl():
            # Try to go directly to a user directory to avoid permission issues with C: root
            username = WindowsPathHelper.get_windows_username()
            if username:
                user_path = f"/mnt/c/Users/{username}"
                if os.path.exists(user_path):
                    return user_path
            return "/mnt/c/Users"
        else:
            return "C:\\"

class LogHandler(logging.Handler):
    """Custom logging handler to redirect logs to GUI"""
    
    def __init__(self, text_widget, queue_obj):
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue_obj
    
    def emit(self, record):
        msg = self.format(record)
        self.queue.put(('log', msg))

class STORMTrainerGUI:
    """GUI wrapper for STORM training functionality"""
    
    def __init__(self, config: STORMConfig, progress_callback=None, log_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.model = None
        self.training_data = None
        
    def load_training_data(self, train_files: List[Path]):
        """Load and process training data from TIFF files"""
        all_cutouts = []
        all_heights = []
        all_group_ids = []
        
        total_files = len(train_files)
        
        for i, tiff_path in enumerate(train_files):
            if self.progress_callback:
                progress = (i / total_files) * 50  # First 50% for data loading
                self.progress_callback(progress, f"Processing file {i+1}/{total_files}: {tiff_path.name}")
            
            if self.log_callback:
                self.log_callback(f"Loading TIFF file: {tiff_path.name}")
            
            # Load TIFF stack
            stack = tiff.imread(tiff_path)
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, self.config.start_z, self.config.end_z, self.config.csum_slices
            )
            
            # Extract PSF cutouts
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance,
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=False  # Don't show plots in GUI mode
            )
            
            # Process cutouts
            psfs = np.array([cutout[0] for cutout in cutouts])
            heights = np.array([cutout[1] for cutout in cutouts])
            
            # Adjust group IDs to be globally unique
            adjusted_group_ids = np.array(group_ids) + len(all_cutouts)
            
            all_cutouts.extend(psfs)
            all_heights.extend(heights)
            all_group_ids.extend(adjusted_group_ids)
        
        # Convert to arrays and normalize
        X = np.expand_dims(np.array(all_cutouts), axis=-1)
        y = np.array(all_heights)
        group_ids = np.array(all_group_ids)
        
        # Normalize data
        X = normalize_0_to_1(X)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        self.training_data = (X, y, group_ids)
        
        if self.log_callback:
            self.log_callback(f"Loaded {len(X)} training samples from {len(train_files)} files")
        
        return X, y, group_ids
    
    def train_model(self, X, y, group_ids):
        """Train the neural network model"""
        
        if self.progress_callback:
            self.progress_callback(50, "Preparing training data...")
        
        # Split data by groups
        (X_train, y_train), (X_val, y_val) = train_val_split_by_group(
            X, y, group_ids, val_size=self.config.val_split
        )
        
        if self.progress_callback:
            self.progress_callback(60, "Building neural network...")
        
        # Build model
        input_shape = (self.config.distance + 1, self.config.distance + 1, 1)
        self.model = build_astigmatic_psf_network(input_shape)
        
        # Setup data augmentation
        augmenter = build_augmenter(self.config.distance)
        
        # Create datasets
        train_ds = make_dataset(X_train, y_train, 
                              batch_size=self.config.batch_size, 
                              training=True, augmenter=augmenter)
        val_ds = make_dataset(X_val, y_val, 
                            batch_size=self.config.batch_size, 
                            training=False)
        
        if self.progress_callback:
            self.progress_callback(70, "Starting model training...")
        
        # Setup callbacks
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        model_path = self.config.output_path / f"model_distance_{self.config.distance}.keras"
        callbacks = setup_callbacks(str(model_path))
        
        # Custom callback for progress updates
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_func, log_func, total_epochs):
                super().__init__()
                self.progress_func = progress_func
                self.log_func = log_func
                self.total_epochs = total_epochs
            
            def on_epoch_end(self, epoch, logs=None):
                if self.progress_func:
                    progress = 70 + (epoch + 1) / self.total_epochs * 30  # Last 30% for training
                    self.progress_func(progress, f"Epoch {epoch + 1}/{self.total_epochs}")
                
                if self.log_func and logs:
                    self.log_func(f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
                                f"val_loss={logs.get('val_loss', 0):.4f}, "
                                f"mae={logs.get('mae', 0):.4f}")
        
        callbacks.append(ProgressCallback(self.progress_callback, self.log_callback, self.config.epochs))
        
        # Train model
        if self.log_callback:
            self.log_callback("Starting model training...")
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=0  # Suppress default output
        )
        
        if self.progress_callback:
            self.progress_callback(100, "Training completed!")
        
        if self.log_callback:
            self.log_callback(f"Training completed. Model saved to: {model_path}")
        
        return self.model, history

class STORMApp:
    """Main STORM Microscopy GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("STORM Microscopy Analysis - Height Regression")
        self.root.geometry("3600x2400")
        
        # Configuration
        self.config = STORMConfig()
        
        # Threading
        self.training_thread = None
        self.queue = queue.Queue()
        
        # Data
        self.tiff_files = []
        self.selected_files = []
        self.model = None
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.create_widgets()
        
        # Start queue processing
        self.process_queue()
    
    def setup_logging(self):
        """Setup logging to redirect to GUI"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handler that sends logs to queue
        handler = LogHandler(None, self.queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data Selection Tab
        self.create_data_tab(notebook)
        
        # Peak Configuration Tab (NEW)
        self.create_peak_config_tab(notebook)
        
        # Training Tab
        self.create_training_tab(notebook)
        
        # Prediction Tab
        self.create_prediction_tab(notebook)
    
    def create_peak_config_tab(self, notebook):
        """Create peak detection configuration and visualization tab"""
        peak_frame = ttk.Frame(notebook)
        notebook.add(peak_frame, text="Peak Configuration")
        
        # Peak detection parameters
        params_frame = ttk.LabelFrame(peak_frame, text="Peak Detection Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create grid of parameters
        params = [
            ("Cutout size:", "distance", self.config.distance),
            ("Prominence Sigma:", "prominence_sigma", self.config.prominence_sigma),
            ("Minimum distance between peaks:", "min_distance", self.config.min_distance),
            ("Radius used to reject hot pixels:", "support_radius", self.config.support_radius),
            ("Start Z:", "start_z", self.config.start_z),
            ("End Z:", "end_z", self.config.end_z),
            ("Sum Slices:", "csum_slices", self.config.csum_slices),
        ]
        
        self.param_vars = {}
        for i, (label, param, default) in enumerate(params):
            row = i // 2
            col = (i % 2) * 3
            
            ttk.Label(params_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 5), pady=5)
            var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
            self.param_vars[param] = var
            ttk.Entry(params_frame, textvariable=var, width=15).grid(row=row, column=col+1, padx=(0, 20), pady=5)
        
        # Visualization controls
        viz_frame = ttk.LabelFrame(peak_frame, text="Peak Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # File selection for visualization
        file_frame = ttk.Frame(viz_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Select TIFF file for visualization:").pack(anchor=tk.W)
        
        file_select_frame = ttk.Frame(viz_frame)
        file_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.viz_file_var = tk.StringVar()
        ttk.Entry(file_select_frame, textvariable=self.viz_file_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_select_frame, text="Browse", command=self.browse_viz_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_select_frame, text="Use Selected File", command=self.use_first_selected_file).pack(side=tk.LEFT)
        
        # Visualize button
        viz_button_frame = ttk.Frame(viz_frame)
        viz_button_frame.pack(fill=tk.X)
        
        self.visualize_button = ttk.Button(viz_button_frame, text="Visualize Peaks", command=self.visualize_peaks)
        self.visualize_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.viz_status_label = ttk.Label(viz_button_frame, text="Select a TIFF file to visualize peak detection")
        self.viz_status_label.pack(side=tk.LEFT)
    
    def create_data_tab(self, notebook):
        """Create data selection and preview tab"""
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Selection")
        
        # Directory selection
        dir_frame = ttk.LabelFrame(data_frame, text="Data Directory", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add quick access buttons for common Windows locations
        quick_frame = ttk.Frame(dir_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(quick_frame, text="Quick Access:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(quick_frame, text="C: Drive", command=lambda: self.set_directory("/mnt/c")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Desktop", command=self.go_to_desktop).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Documents", command=self.go_to_documents).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Downloads", command=self.go_to_downloads).pack(side=tk.LEFT, padx=(0, 5))
        
        self.data_dir_var = tk.StringVar()
        # Set default to a common location
        default_path = WindowsPathHelper.get_default_data_path()
        if os.path.exists(default_path):
            self.data_dir_var.set(default_path)
        
        entry_frame = ttk.Frame(dir_frame)
        entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(entry_frame, textvariable=self.data_dir_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(entry_frame, text="Browse", command=self.browse_data_directory).pack(side=tk.LEFT)
        ttk.Button(entry_frame, text="Scan Files", command=self.scan_tiff_files).pack(side=tk.LEFT, padx=(10, 0))
        
        # File list
        files_frame = ttk.LabelFrame(data_frame, text="Available TIFF Files", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for file selection
        columns = ('Index', 'Filename', 'Size', 'Selected')
        self.file_tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=10)
        
        # Set column widths: Filename gets 50%, others get 16.67% each
        column_widths = {
            'Index': 100,
            'Filename': 400,  # Much larger for filename
            'Size': 100,
            'Selected': 100
        }
        
        for col in columns:
            self.file_tree.heading(col, text=col)
            self.file_tree.column(col, width=column_widths[col])
        
        # Add double-click to toggle selection
        self.file_tree.bind('<Double-1>', self.toggle_file_selection)
        
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File selection buttons
        button_frame = ttk.Frame(data_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Select File", command=self.select_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preview Selected", command=self.preview_files).pack(side=tk.LEFT, padx=5)
    
    def create_training_tab(self, notebook):
        """Create training configuration and control tab"""
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="Training")
        
        # Training controls
        control_frame = ttk.LabelFrame(train_frame, text="Training Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.epochs_var = tk.IntVar(value=self.config.epochs)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.batch_size_var = tk.IntVar(value=self.config.batch_size)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(params_frame, text="Cutout size:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.distance_var = tk.IntVar(value=self.config.distance)
        ttk.Entry(params_frame, textvariable=self.distance_var, width=10).grid(row=0, column=5)
        
        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT)
        
        # Progress bar
        progress_frame = ttk.LabelFrame(train_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # Training log
        log_frame = ttk.LabelFrame(train_frame, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def create_prediction_tab(self, notebook):
        """Create prediction and evaluation tab"""
        pred_frame = ttk.Frame(notebook)
        notebook.add(pred_frame, text="Prediction")
        
        # File selection for prediction
        file_frame = ttk.LabelFrame(pred_frame, text="Select File for Prediction", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pred_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.pred_file_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_prediction_file).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Predict", command=self.make_prediction).pack(side=tk.LEFT, padx=(10, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(pred_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Placeholder for matplotlib plots
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_viz_file(self):
        """Browse for TIFF file to visualize"""
        try:
            self.root.update_idletasks()
            
            # Use current data directory as initial directory
            current_dir = self.data_dir_var.get()
            if current_dir and os.path.exists(current_dir):
                initial_dir = current_dir
            else:
                initial_dir = WindowsPathHelper.get_default_data_path()
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select TIFF file for peak visualization",
                initialdir=initial_dir,
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.viz_file_var.set(filename)
                self.viz_status_label.config(text=f"Selected: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")
    
    def use_first_selected_file(self):
        """Sync with the currently selected file from data selection tab"""
        selected_files = self.get_selected_files()
        if selected_files:
            self.viz_file_var.set(str(selected_files[0]))
            self.viz_status_label.config(text=f"Using: {selected_files[0].name}")
        else:
            messagebox.showwarning("Warning", "No file selected in Data Selection tab.\nPlease select a file first.")
    
    def visualize_peaks(self):
        """Visualize peak detection on selected file"""
        viz_file = self.viz_file_var.get()
        if not viz_file:
            messagebox.showwarning("Warning", "Please select a TIFF file first")
            return
        
        if not os.path.exists(viz_file):
            messagebox.showerror("Error", f"File not found: {viz_file}")
            return
        
        try:
            # Update status
            self.viz_status_label.config(text="Loading TIFF file...")
            self.root.update()
            
            # Update configuration from GUI
            self.update_config_from_gui()
            
            # Load TIFF stack
            stack = tiff.imread(viz_file)
            self.viz_status_label.config(text="Processing image...")
            self.root.update()
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, self.config.start_z, self.config.end_z, self.config.csum_slices
            )
            
            self.viz_status_label.config(text="Detecting peaks...")
            self.root.update()
            
            # Extract PSF cutouts with visualization
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance,
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=True  # This will show the matplotlib plot
            )
            
            # Update status with results
            self.viz_status_label.config(text=f"Found {len(peaks)} peaks, {len(cutouts)} cutouts")
            
            # Log results
            self.log_message(f"Peak visualization completed for {Path(viz_file).name}")
            self.log_message(f"Parameters: distance={self.config.distance}, prominence_sigma={self.config.prominence_sigma}")
            self.log_message(f"Results: {len(peaks)} peaks detected, {len(cutouts)} cutouts extracted")
            
        except Exception as e:
            self.viz_status_label.config(text="Error occurred")
            messagebox.showerror("Visualization Error", f"Error visualizing peaks: {str(e)}")
            self.log_message(f"Peak visualization error: {str(e)}")

    def go_to_desktop(self):
        """Navigate to Windows Desktop"""
        desktop_path = WindowsPathHelper.get_desktop_path()
        self.set_directory(desktop_path)
    
    def go_to_documents(self):
        """Navigate to Windows Documents"""
        docs_path = WindowsPathHelper.get_documents_path()
        self.set_directory(docs_path)
    
    def go_to_downloads(self):
        """Navigate to Windows Downloads"""
        downloads_path = WindowsPathHelper.get_downloads_path()
        self.set_directory(downloads_path)
    
    def set_directory(self, path):
        """Set directory and optionally scan"""
        try:
            if os.path.exists(path):
                # Test if we can actually access the directory
                try:
                    os.listdir(path)
                    self.data_dir_var.set(path)
                    # Auto-scan without asking
                    self.scan_tiff_files()
                except PermissionError:
                    messagebox.showerror("Permission Error", 
                                       f"Cannot access directory: {path}\n"
                                       f"Permission denied. Try a different location.")
                except Exception as e:
                    messagebox.showerror("Error", 
                                       f"Error accessing directory: {path}\n"
                                       f"Error: {str(e)}")
            else:
                messagebox.showwarning("Warning", f"Directory {path} not found")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
    
    def browse_data_directory(self):
        """Browse for data directory with better error handling"""
        try:
            # Force GUI update before opening dialog
            self.root.update_idletasks()
            
            # Use current directory from text field as initial directory
            current_dir = self.data_dir_var.get()
            if current_dir and os.path.exists(current_dir):
                initial_dir = current_dir
            else:
                # Fallback to default if current directory doesn't exist
                initial_dir = WindowsPathHelper.get_default_data_path()
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.expanduser("~")  # Final fallback to home directory
            
            directory = filedialog.askdirectory(
                title="Select TIFF Data Directory", 
                initialdir=initial_dir,
                parent=self.root
            )
            
            if directory:
                self.data_dir_var.set(directory)
                self.scan_tiff_files()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file dialog: {str(e)}")
            messagebox.showinfo("Alternative", 
                              "You can type the path directly in the text field and click 'Scan Files'")
    
    def toggle_file_selection(self, event):
        """Select single file on double-click (single selection mode)"""
        if not self.file_tree.selection():
            return
            
        clicked_item = self.file_tree.selection()[0]
        
        # First clear all selections
        for item in self.file_tree.get_children():
            values = list(self.file_tree.item(item, 'values'))
            values[3] = "No"
            self.file_tree.item(item, values=values)
        
        # Then select only the clicked item
        values = list(self.file_tree.item(clicked_item, 'values'))
        values[3] = "Yes"
        self.file_tree.item(clicked_item, values=values)
        
        # Update peak configuration tab with selected file
        self.update_peak_config_file()
    
    def select_file(self):
        """Select currently highlighted file (single selection)"""
        if not self.file_tree.selection():
            messagebox.showwarning("Warning", "Please highlight a file first")
            return
            
        selected_item = self.file_tree.selection()[0]
        
        # Clear all selections first
        for item in self.file_tree.get_children():
            values = list(self.file_tree.item(item, 'values'))
            values[3] = "No"
            self.file_tree.item(item, values=values)
        
        # Select only the highlighted item
        values = list(self.file_tree.item(selected_item, 'values'))
        values[3] = "Yes"
        self.file_tree.item(selected_item, values=values)
        
        # Update peak configuration tab with selected file
        self.update_peak_config_file()
    
    def clear_selection(self):
        """Clear file selection"""
        for item in self.file_tree.get_children():
            values = list(self.file_tree.item(item, 'values'))
            values[3] = "No"
            self.file_tree.item(item, values=values)
        
        # Clear peak configuration file as well
        self.viz_file_var.set("")
        self.viz_status_label.config(text="No file selected")
    
    def update_peak_config_file(self):
        """Update the peak configuration tab with the selected file"""
        selected_files = self.get_selected_files()
        if selected_files:
            self.viz_file_var.set(str(selected_files[0]))
            self.viz_status_label.config(text=f"Using: {selected_files[0].name}")
        else:
            self.viz_file_var.set("")
            self.viz_status_label.config(text="No file selected")
    
    def scan_tiff_files(self):
        """Scan directory for TIFF files"""
        data_dir = self.data_dir_var.get()
        if not data_dir:
            messagebox.showwarning("Warning", "Please select a data directory first")
            return
        
        try:
            self.tiff_files = find_tiff_files(Path(data_dir))
            self.update_file_list()
            self.log_message(f"Found {len(self.tiff_files)} TIFF files in {data_dir}")
        except PermissionError as e:
            # Handle permission errors gracefully
            self.tiff_files = []
            self.update_file_list()
            messagebox.showwarning("Permission Error", 
                                 f"Some files in {data_dir} could not be accessed due to permission restrictions.\n"
                                 f"Try selecting a more specific subdirectory.\n\n"
                                 f"Error details: {str(e)}")
            self.log_message(f"Permission error scanning {data_dir}: {str(e)}")
        except Exception as e:
            # Handle other errors
            self.tiff_files = []
            self.update_file_list()
            messagebox.showerror("Error", f"Error scanning directory: {str(e)}")
            self.log_message(f"Error scanning {data_dir}: {str(e)}")
    
    def update_file_list(self):
        """Update the file list display"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files
        for i, file_path in enumerate(self.tiff_files):
            try:
                size = file_path.stat().st_size / (1024 * 1024)  # MB
                size_str = f"{size:.1f} MB"
            except:
                size_str = "Unknown"
            
            self.file_tree.insert('', tk.END, values=(i, file_path.name, size_str, "No"))
    
    def get_selected_files(self):
        """Get list of selected files (should be only one)"""
        selected = []
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item, 'values')
            if values[3] == "Yes":
                index = int(values[0])
                selected.append(self.tiff_files[index])
        return selected
    
    def preview_files(self):
        """Preview selected file"""
        selected = self.get_selected_files()
        if not selected:
            messagebox.showwarning("Warning", "No file selected")
            return
        
        file_path = selected[0]
        message = f"Selected file for training:\n\n• {file_path.name}\n\nPath: {file_path}"
        messagebox.showinfo("Selected File", message)
    
    def start_training(self):
        """Start training in a separate thread"""
        selected_files = self.get_selected_files()
        if not selected_files:
            messagebox.showwarning("Warning", "Please select a file for training in the Data Selection tab")
            return
        
        if len(selected_files) > 1:
            messagebox.showerror("Error", "Multiple files selected. This is a single-stack model - please select only one file.")
            return
        
        # Update configuration from GUI
        self.update_config_from_gui()
        
        # Disable training button
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Log which file is being used for training
        self.log_message(f"Starting training with single stack: {selected_files[0].name}")
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread, 
            args=(selected_files,),
            daemon=True
        )
        self.training_thread.start()
    
    def train_model_thread(self, selected_files):
        """Training thread function"""
        try:
            trainer = STORMTrainerGUI(
                self.config,
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )
            
            # Load data
            X, y, group_ids = trainer.load_training_data(selected_files)
            
            # Train model
            model, history = trainer.train_model(X, y, group_ids)
            
            self.model = model
            self.queue.put(('training_complete', 'Training completed successfully!'))
            
        except Exception as e:
            self.queue.put(('error', f"Training error: {str(e)}"))
        finally:
            self.queue.put(('training_finished', None))
    
    def stop_training(self):
        """Stop training (placeholder - would need more complex implementation)"""
        messagebox.showinfo("Info", "Training stop requested (implementation needed)")
    
    def update_progress(self, value, message):
        """Update progress bar and message"""
        self.queue.put(('progress', (value, message)))
    
    def log_message(self, message):
        """Add message to log"""
        self.queue.put(('log', message))
    
    def update_config_from_gui(self):
        """Update configuration from GUI values"""
        self.config.epochs = self.epochs_var.get()
        self.config.batch_size = self.batch_size_var.get()
        self.config.distance = self.distance_var.get()
        
        for param, var in self.param_vars.items():
            setattr(self.config, param, var.get())
    
    def save_model(self):
        """Save trained model"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model to save")
            return
        
        try:
            self.root.update_idletasks()
            
            initial_dir = WindowsPathHelper.get_documents_path()
            
            filename = filedialog.asksaveasfilename(
                title="Save Model",
                initialdir=initial_dir,
                defaultextension=".keras",
                filetypes=[("Keras models", "*.keras"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.model.save(filename)
                messagebox.showinfo("Success", f"Model saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.root.update_idletasks()
            
            initial_dir = WindowsPathHelper.get_documents_path()
            
            filename = filedialog.askopenfilename(
                title="Load Model",
                initialdir=initial_dir,
                filetypes=[("Keras models", "*.keras"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.model = tf.keras.models.load_model(filename)
                messagebox.showinfo("Success", f"Model loaded from {filename}")
                self.log_message(f"Model loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def browse_prediction_file(self):
        """Browse for prediction file with better error handling"""
        try:
            self.root.update_idletasks()
            
            # Use current data directory as initial directory for consistency
            current_dir = self.data_dir_var.get()
            if current_dir and os.path.exists(current_dir):
                initial_dir = current_dir
            else:
                # Fallback to default if current directory doesn't exist
                initial_dir = WindowsPathHelper.get_default_data_path()
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select TIFF file for prediction",
                initialdir=initial_dir,
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.pred_file_var.set(filename)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file dialog: {str(e)}")
    
    def make_prediction(self):
        """Make prediction on selected file"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a trained model first")
            return
        
        pred_file = self.pred_file_var.get()
        if not pred_file:
            messagebox.showwarning("Warning", "Please select a file for prediction")
            return
        
        try:
            # This would implement the prediction logic
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Prediction on {pred_file}\n")
            self.results_text.insert(tk.END, "Implementation in progress...\n")
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
    
    def process_queue(self):
        """Process messages from worker threads"""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'progress':
                    value, message = data
                    self.progress_var.set(value)
                    self.progress_label.config(text=message)
                
                elif msg_type == 'log':
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, data + '\n')
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                
                elif msg_type == 'training_complete':
                    messagebox.showinfo("Success", data)
                
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                
                elif msg_type == 'training_finished':
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = STORMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()