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
    
    # Configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) detected and configured: {len(gpus)} device(s)")
            print(f"GPU devices: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected - using CPU (training will be slow)")
        
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
        self.root.geometry("1200x800")
        
        # Configuration
        self.config = STORMConfig()
        
        # Threading
        self.training_thread = None
        self.queue = queue.Queue()
        
        # Data
        self.tiff_files = []
        self.selected_files = []
        self.model = None
        
        # Training datasets (created by save_cutouts)
        self.train_ds = None
        self.val_ds = None
        self.training_data_ready = False
        self.training_stop_requested = False
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.create_widgets()
        
        # Start queue processing
        self.process_queue()
        
        # Log initial GPU status
        self.root.after(1000, self.log_initial_gpu_status)  # Delay to ensure GUI is ready
    
    def log_initial_gpu_status(self):
        """Log GPU status when GUI starts"""
        self.log_message("=== STORM Microscopy GUI Started ===")
        
        # Check TensorFlow and GPU status
        self.log_message(f"TensorFlow version: {tf.__version__}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            self.log_message(f"✅ {len(gpu_devices)} GPU(s) available for training:")
            for i, gpu in enumerate(gpu_devices):
                self.log_message(f"   GPU {i}: {gpu.name}")
        else:
            self.log_message("⚠️  No GPU detected - training will be slow on CPU")
            self.log_message("   Ensure CUDA is installed and WSL can access GPU")
        
        self.log_message("Ready to use. Select TIFF files to begin.")
    
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
        
        self.save_cutouts_button = ttk.Button(viz_button_frame, text="Save Cutouts", command=self.save_cutouts)
        self.save_cutouts_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.viz_status_label = ttk.Label(viz_button_frame, text="Select a TIFF file to visualize peak detection")
        self.viz_status_label.pack(side=tk.LEFT)
        
        # Log section (shared with training tab)
        log_frame = ttk.LabelFrame(peak_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text_peak = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_text_peak.pack(fill=tk.BOTH, expand=True)
    
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
        """Create training configuration and control tab with multi-stage training"""
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="Training")
        
        # Create scrollable frame for training blocks
        canvas = tk.Canvas(train_frame)
        scrollbar = ttk.Scrollbar(train_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # No cutouts alert
        self.no_cutouts_alert = ttk.Label(self.scrollable_frame, 
                                         text="⚠️ NO CUTOUTS SAVED! Please go to Peak Configuration and click 'Save Cutouts' first.",
                                         foreground="red", font=("TkDefaultFont", 10, "bold"))
        self.no_cutouts_alert.pack(fill=tk.X, padx=10, pady=10)
        
        # Training blocks container
        self.training_blocks_frame = ttk.Frame(self.scrollable_frame)
        self.training_blocks_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Initialize training blocks list
        self.training_blocks = []
        
        # Create default training block
        self.create_default_training_block()
        
        # Add block button
        add_block_frame = ttk.Frame(self.scrollable_frame)
        add_block_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(add_block_frame, text="➕ Add Training Block", command=self.add_training_block).pack(side=tk.LEFT)
        
        # Training controls
        control_frame = ttk.LabelFrame(self.scrollable_frame, text="Training Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Check GPU", command=self.check_gpu_status).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # Save model button (moved to end)
        save_frame = ttk.Frame(self.scrollable_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(save_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT)
        
        # Log
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_default_training_block(self):
        """Create the default (first) training block"""
        block_frame = ttk.LabelFrame(self.training_blocks_frame, text="Training Block 1 (Default)", padding=10)
        block_frame.pack(fill=tk.X, pady=5)
        
        # Parameters grid
        params_frame = ttk.Frame(block_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Epochs, Adam LR, Huber Delta
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        epochs_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=epochs_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(params_frame, text="Adam LR:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        adam_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(params_frame, textvariable=adam_lr_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(params_frame, text="Huber Delta:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        huber_delta_var = tk.DoubleVar(value=0.06)
        ttk.Entry(params_frame, textvariable=huber_delta_var, width=10).grid(row=0, column=5)
        
        # Row 2: LR Patience, Early Stopping
        ttk.Label(params_frame, text="LR Patience:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        lr_patience_var = tk.IntVar(value=25)
        ttk.Entry(params_frame, textvariable=lr_patience_var, width=10).grid(row=1, column=1, padx=(0, 20), pady=(5, 0))
        
        # Early stopping with enable checkbox
        early_stop_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Early Stopping:", variable=early_stop_enabled_var).grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        early_stop_patience_var = tk.IntVar(value=55)
        early_stop_entry = ttk.Entry(params_frame, textvariable=early_stop_patience_var, width=10, state=tk.DISABLED)
        early_stop_entry.grid(row=1, column=3, padx=(0, 20), pady=(5, 0))
        
        # Enable/disable early stopping entry based on checkbox
        def toggle_early_stop():
            early_stop_entry.config(state=tk.NORMAL if early_stop_enabled_var.get() else tk.DISABLED)
        early_stop_enabled_var.trace('w', lambda *args: toggle_early_stop())
        
        # Store block data
        block_data = {
            'frame': block_frame,
            'block_id': 1,
            'is_default': True,
            'epochs_var': epochs_var,
            'adam_lr_var': adam_lr_var,
            'huber_delta_var': huber_delta_var,
            'lr_patience_var': lr_patience_var,
            'early_stop_enabled_var': early_stop_enabled_var,
            'early_stop_patience_var': early_stop_patience_var
        }
        
        self.training_blocks.append(block_data)
    
    def add_training_block(self):
        """Add a new training block"""
        block_id = len(self.training_blocks) + 1
        
        block_frame = ttk.LabelFrame(self.training_blocks_frame, text=f"Training Block {block_id}", padding=10)
        block_frame.pack(fill=tk.X, pady=5)
        
        # Remove button (X)
        remove_frame = ttk.Frame(block_frame)
        remove_frame.pack(fill=tk.X, pady=(0, 5))
        
        remove_button = ttk.Button(remove_frame, text="❌ Remove Block", 
                                  command=lambda: self.remove_training_block(block_id))
        remove_button.pack(side=tk.RIGHT)
        
        # Parameters grid
        params_frame = ttk.Frame(block_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Adam LR, Huber Delta
        ttk.Label(params_frame, text="Adam LR:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        adam_lr_var = tk.DoubleVar(value=2e-4)
        ttk.Entry(params_frame, textvariable=adam_lr_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(params_frame, text="Huber Delta:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        huber_delta_var = tk.DoubleVar(value=0.02)
        ttk.Entry(params_frame, textvariable=huber_delta_var, width=10).grid(row=0, column=3)
        
        # Row 2: LR Patience, Early Stopping
        ttk.Label(params_frame, text="LR Patience:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        lr_patience_var = tk.IntVar(value=20)
        ttk.Entry(params_frame, textvariable=lr_patience_var, width=10).grid(row=1, column=1, padx=(0, 20), pady=(5, 0))
        
        # Early stopping with enable checkbox
        early_stop_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Early Stopping:", variable=early_stop_enabled_var).grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        early_stop_patience_var = tk.IntVar(value=40)
        early_stop_entry = ttk.Entry(params_frame, textvariable=early_stop_patience_var, width=10)
        early_stop_entry.grid(row=1, column=3, pady=(5, 0))
        
        # Enable/disable early stopping entry based on checkbox
        def toggle_early_stop():
            early_stop_entry.config(state=tk.NORMAL if early_stop_enabled_var.get() else tk.DISABLED)
        early_stop_enabled_var.trace('w', lambda *args: toggle_early_stop())
        
        # Store block data
        block_data = {
            'frame': block_frame,
            'block_id': block_id,
            'is_default': False,
            'adam_lr_var': adam_lr_var,
            'huber_delta_var': huber_delta_var,
            'lr_patience_var': lr_patience_var,
            'early_stop_enabled_var': early_stop_enabled_var,
            'early_stop_patience_var': early_stop_patience_var
        }
        
        self.training_blocks.append(block_data)
        
        # Update canvas scroll region
        self.scrollable_frame.update_idletasks()
    
    def remove_training_block(self, block_id):
        """Remove a training block"""
        # Find and remove the block
        for i, block in enumerate(self.training_blocks):
            if block['block_id'] == block_id and not block['is_default']:
                block['frame'].destroy()
                self.training_blocks.pop(i)
                break
        
        # Update canvas scroll region
        self.scrollable_frame.update_idletasks()
    
    def update_cutouts_alert(self):
        """Update the visibility of the no cutouts alert"""
        if self.training_data_ready:
            self.no_cutouts_alert.pack_forget()
        else:
            self.no_cutouts_alert.pack(fill=tk.X, padx=10, pady=10, before=self.training_blocks_frame)
    
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

    def save_cutouts(self):
        """Process cutouts and create training datasets"""
        viz_file = self.viz_file_var.get()
        if not viz_file:
            messagebox.showwarning("Warning", "Please select a TIFF file first")
            return
        
        if not os.path.exists(viz_file):
            messagebox.showerror("Error", f"File not found: {viz_file}")
            return
        
        try:
            # Update status
            self.viz_status_label.config(text="Processing cutouts...")
            self.root.update()
            
            # Update configuration from GUI
            self.update_config_from_gui()
            
            # Load TIFF stack
            stack = tiff.imread(viz_file)
            self.log_message(f"Loading TIFF stack: {Path(viz_file).name}")
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, self.config.start_z, self.config.end_z, self.config.csum_slices
            )
            
            self.viz_status_label.config(text="Extracting PSF cutouts...")
            self.root.update()
            
            # Extract PSF cutouts (without visualization this time)
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance,
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=False  # No visualization for data processing
            )
            
            self.log_message(f"Extracted {len(cutouts)} cutouts from {len(peaks)} peaks")
            
            # Process cutouts following the pipeline
            self.viz_status_label.config(text="Normalizing data...")
            self.root.update()
            
            # Extract PSFs and heights
            psfs = np.expand_dims(np.array([cutouts[i][0] for i in range(len(cutouts))]), axis=-1)
            heights = np.array([cutouts[i][1] for i in range(len(cutouts))])
            group_ids = np.array(group_ids)
            
            self.log_message(f"Height range: {np.min(heights):.1f} to {np.max(heights):.1f}")
            
            # Normalize PSFs (per-image normalization)
            psfs = normalize_0_to_1(psfs)
            
            # Normalize heights to [0, 1]
            heights = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))
            
            self.log_message(f"Normalized heights range: {np.min(heights):.3f} to {np.max(heights):.3f}")
            
            # Split into training and validation sets
            self.viz_status_label.config(text="Creating train/validation split...")
            self.root.update()
            
            (X_tr, y_tr), (X_va, y_va) = train_val_split_by_group(
                psfs, heights, group_ids, val_size=self.config.val_split
            )
            
            self.log_message(f"Training set: {len(X_tr)} samples")
            self.log_message(f"Validation set: {len(X_va)} samples")
            
            # Build augmenter
            augmenter = build_augmenter(self.config.distance)
            
            # Create datasets
            self.viz_status_label.config(text="Creating TensorFlow datasets...")
            self.root.update()
            
            train_ds = make_dataset(X_tr, y_tr, 
                                  batch_size=self.config.batch_size, 
                                  training=True, 
                                  augmenter=augmenter)
            
            val_ds = make_dataset(X_va, y_va, 
                                batch_size=self.config.batch_size, 
                                training=False)
            
            # Store datasets in memory for training
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.training_data_ready = True
            
            # Update the cutouts alert visibility
            self.update_cutouts_alert()
            
            # Update status with success
            self.viz_status_label.config(text=f"Datasets ready! Train: {len(X_tr)}, Val: {len(X_va)}")
            
            # Log success
            self.log_message("Cutouts processed and datasets created successfully!")
            self.log_message(f"Training dataset: {len(X_tr)} samples with augmentation")
            self.log_message(f"Validation dataset: {len(X_va)} samples")
            self.log_message("Ready for training!")
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Cutouts processed successfully!\n\n"
                              f"Training samples: {len(X_tr)}\n"
                              f"Validation samples: {len(X_va)}\n\n"
                              f"Datasets are ready for training.")
            
        except Exception as e:
            self.viz_status_label.config(text="Error processing cutouts")
            messagebox.showerror("Processing Error", f"Error processing cutouts: {str(e)}")
            self.log_message(f"Cutout processing error: {str(e)}")

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
        
        # Clear both log windows
        for log_widget in [self.log_text, self.log_text_peak]:
            log_widget.config(state=tk.NORMAL)
            log_widget.delete(1.0, tk.END)
            log_widget.config(state=tk.DISABLED)
        
        # Log which file is being used for training
        self.log_message(f"Starting training with single stack: {selected_files[0].name}")
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread, 
            args=(selected_files,),
            daemon=True
        )
        self.training_thread.start()
    
    def check_gpu_status(self):
        """Check and log GPU availability and status"""
        try:
            # Check if GPU is available
            gpu_available = tf.config.list_physical_devices('GPU')
            
            if gpu_available:
                self.log_message(f"✅ GPU detected: {len(gpu_available)} device(s)")
                for i, gpu in enumerate(gpu_available):
                    self.log_message(f"   GPU {i}: {gpu.name}")
                
                # Check if TensorFlow can actually use the GPU
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    result = tf.matmul(test_tensor, test_tensor)
                    self.log_message("✅ GPU computation test successful")
                
                # Log GPU memory info
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu_available[0])
                    if 'device_name' in gpu_details:
                        self.log_message(f"   GPU Name: {gpu_details['device_name']}")
                except:
                    pass
                    
                return True
            else:
                self.log_message("❌ No GPU detected - training will use CPU (very slow)")
                self.log_message("   Make sure CUDA is properly installed and configured")
                return False
                
        except Exception as e:
            self.log_message(f"❌ GPU check failed: {str(e)}")
            self.log_message("   Training will use CPU (very slow)")
            return False
    
    def train_model_thread(self, selected_files):
        """Multi-stage training thread function"""
        try:
            # Reset stop flag
            self.training_stop_requested = False
            
            # Check GPU status first
            self.log_message("=== System Check ===")
            gpu_available = self.check_gpu_status()
            
            if not gpu_available:
                # Ask user if they want to continue with CPU
                response = messagebox.askyesno(
                    "No GPU Detected", 
                    "No GPU detected! Training will be very slow on CPU.\n\n"
                    "Do you want to continue anyway?\n\n"
                    "Click 'No' to cancel and check your CUDA installation."
                )
                if not response:
                    self.log_message("Training cancelled by user due to no GPU")
                    return
                else:
                    self.log_message("User chose to continue training on CPU")
            
            # Check if we have pre-processed datasets from "Save Cutouts"
            if not self.training_data_ready or self.train_ds is None or self.val_ds is None:
                self.queue.put(('error', 'No cutouts saved! Please go to Peak Configuration and click "Save Cutouts" first.'))
                return
            
            self.log_message("Starting multi-stage training with pre-processed datasets")
            
            # Build initial model
            self.update_progress(10, "Building neural network...")
            input_shape = (self.config.distance + 1, self.config.distance + 1, 1)
            model = build_astigmatic_psf_network(input_shape)
            
            total_blocks = len(self.training_blocks)
            current_epoch = 0
            
            # Process each training block
            for block_idx, block in enumerate(self.training_blocks):
                # Check if stop was requested
                if self.training_stop_requested:
                    self.log_message("Training stopped by user before completing all blocks")
                    break
                    
                block_num = block_idx + 1
                self.log_message(f"=== Training Block {block_num}/{total_blocks} ===")
                
                # Get parameters from block
                if block['is_default']:
                    epochs = block['epochs_var'].get()
                    initial_epoch = current_epoch
                else:
                    # For fine-tuning blocks, use a default of 300 epochs
                    epochs = 300
                    initial_epoch = current_epoch
                
                adam_lr = block['adam_lr_var'].get()
                huber_delta = block['huber_delta_var'].get()
                lr_patience = block['lr_patience_var'].get()
                early_stop_enabled = block['early_stop_enabled_var'].get()
                early_stop_patience = block['early_stop_patience_var'].get() if early_stop_enabled else None
                
                self.log_message(f"Block {block_num} parameters:")
                self.log_message(f"  Epochs: {initial_epoch} → {epochs}")
                self.log_message(f"  Adam LR: {adam_lr}")
                self.log_message(f"  Huber Delta: {huber_delta}")
                self.log_message(f"  LR Patience: {lr_patience}")
                if early_stop_enabled:
                    self.log_message(f"  Early Stopping Patience: {early_stop_patience}")
                else:
                    self.log_message("  Early Stopping: Disabled")
                
                # Compile model with block-specific parameters
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(adam_lr),
                    loss=tf.keras.losses.Huber(delta=huber_delta),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
                )
                
                # Setup callbacks
                self.config.output_path.mkdir(parents=True, exist_ok=True)
                model_path = self.config.output_path / f"model_distance_{self.config.distance}.keras"
                
                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        str(model_path), 
                        monitor='val_mae', 
                        save_best_only=True, 
                        mode='min'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_mae', 
                        factor=0.5, 
                        patience=lr_patience, 
                        min_lr=1e-6
                    )
                ]
                
                # Add early stopping if enabled
                if early_stop_enabled:
                    callbacks.append(
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_mae', 
                            patience=early_stop_patience, 
                            restore_best_weights=True
                        )
                    )
                
                # Custom callback for progress updates and verbose logging
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, progress_func, log_func, block_num, total_blocks, initial_epoch, total_epochs, stop_flag_func):
                        super().__init__()
                        self.progress_func = progress_func
                        self.log_func = log_func
                        self.block_num = block_num
                        self.total_blocks = total_blocks
                        self.initial_epoch = initial_epoch
                        self.total_epochs = total_epochs
                        self.stop_flag_func = stop_flag_func
                    
                    def on_epoch_begin(self, epoch, logs=None):
                        # Check if stop was requested
                        if self.stop_flag_func():
                            self.log_func("Training stopped by user request")
                            self.model.stop_training = True
                            return
                        
                        current_epoch = self.initial_epoch + epoch + 1
                        self.log_func(f"Epoch {current_epoch}/{self.total_epochs}")
                    
                    def on_epoch_end(self, epoch, logs=None):
                        # Check if stop was requested
                        if self.stop_flag_func():
                            self.log_func("Training stopped by user request")
                            self.model.stop_training = True
                            return
                            
                        current_epoch = self.initial_epoch + epoch + 1
                        
                        # Calculate progress: 10% setup + 80% training + 10% final
                        block_progress = (self.block_num - 1) / self.total_blocks
                        epoch_progress = (epoch + 1) / (self.total_epochs - self.initial_epoch)
                        total_progress = 10 + 80 * (block_progress + epoch_progress / self.total_blocks)
                        
                        if self.progress_func:
                            self.progress_func(total_progress, 
                                             f"Block {self.block_num}/{self.total_blocks} - Epoch {current_epoch}")
                        
                        if self.log_func and logs:
                            # Format like Keras verbose output
                            loss_str = f"loss: {logs.get('loss', 0):.4f}"
                            val_loss_str = f"val_loss: {logs.get('val_loss', 0):.4f}"
                            mae_str = f"mae: {logs.get('mae', 0):.4f}"
                            val_mae_str = f"val_mae: {logs.get('val_mae', 0):.4f}"
                            
                            self.log_func(f"Epoch {current_epoch}/{self.total_epochs} - {loss_str} - {val_loss_str} - {mae_str} - {val_mae_str}")
                    
                    def on_train_begin(self, logs=None):
                        self.log_func(f"Starting training for Block {self.block_num}...")
                    
                    def on_train_end(self, logs=None):
                        if self.stop_flag_func():
                            self.log_func(f"Block {self.block_num} training stopped by user")
                        else:
                            self.log_func(f"Block {self.block_num} training completed")
                
                callbacks.append(ProgressCallback(
                    self.update_progress, self.log_message, 
                    block_num, total_blocks, initial_epoch, epochs,
                    lambda: self.training_stop_requested
                ))
                
                # Train model for this block
                progress_start = 10 + 80 * (block_idx / total_blocks)
                self.update_progress(progress_start, f"Training Block {block_num}/{total_blocks}...")
                
                history = model.fit(
                    self.train_ds,
                    validation_data=self.val_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    verbose=0  # Suppress default output
                )
                
                # Update current epoch for next block
                current_epoch = epochs
                
                self.log_message(f"Block {block_num} completed!")
                
                # Load best model for next block (if not the last block)
                if block_idx < total_blocks - 1:
                    self.log_message("Loading best model for next training block...")
                    model = tf.keras.models.load_model(str(model_path))
            
            # Final model loading and completion
            if not self.training_stop_requested:
                self.update_progress(90, "Loading final model...")
                self.model = tf.keras.models.load_model(str(model_path))
                
                self.update_progress(100, "Multi-stage training completed!")
                self.log_message(f"All {total_blocks} training blocks completed!")
                self.log_message(f"Final model saved to: {model_path}")
                
                self.queue.put(('training_complete', f'Multi-stage training completed! ({total_blocks} blocks)'))
            else:
                # Training was stopped
                self.update_progress(100, "Training stopped by user")
                self.log_message("Training stopped by user request")
                
                # Try to load the best model so far
                try:
                    self.model = tf.keras.models.load_model(str(model_path))
                    self.log_message(f"Loaded best model so far from: {model_path}")
                except:
                    self.log_message("No model checkpoint available")
                
                self.queue.put(('training_complete', 'Training stopped by user'))
            
        except Exception as e:
            self.queue.put(('error', f"Training error: {str(e)}"))
        finally:
            self.queue.put(('training_finished', None))
    
    def stop_training(self):
        """Stop training by setting a flag that will be checked by the training thread"""
        self.training_stop_requested = True
        self.log_message("Training stop requested by user...")
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showinfo("Stop Requested", "Training will stop after the current epoch completes.")
    
    def update_progress(self, value, message):
        """Update progress bar and message"""
        self.queue.put(('progress', (value, message)))
    
    def log_message(self, message):
        """Add message to log"""
        self.queue.put(('log', message))
    
    def update_config_from_gui(self):
        """Update configuration from GUI values (for peak configuration parameters)"""
        # Only update the peak detection parameters from the Peak Configuration tab
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
                    # Update both log windows (training tab and peak configuration tab)
                    for log_widget in [self.log_text, self.log_text_peak]:
                        log_widget.config(state=tk.NORMAL)
                        log_widget.insert(tk.END, data + '\n')
                        log_widget.see(tk.END)
                        log_widget.config(state=tk.DISABLED)
                
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