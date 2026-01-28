#!/usr/bin/env python3
"""
STORM Microscopy GUI Application - Time Series Analysis Version

A comprehensive GUI for STORM microscopy analysis with deep learning-based 
height regression and time series processing capabilities.

Author: ItzWhole
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import logging
from typing import Optional, List

# Add storm_core to path
brainstorm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BrainSTORM')
sys.path.append(brainstorm_dir)

try:
    from storm_core.data_processing import crop_and_sum_stack, extract_psf_cutouts, normalize_0_to_1
    from storm_core.neural_network import (
        build_astigmatic_psf_network, train_val_split_by_group, 
        build_augmenter, make_dataset, setup_callbacks
    )
    from storm_core.evaluation import plot_true_vs_pred_heatmap
    import numpy as np
    import tifffile as tiff
    import tensorflow as tf
    
    # Auto-fix CUDA library path issues
    def setup_cuda_environment():
        """Automatically setup CUDA environment variables and detect cuDNN"""
        import os
        import subprocess
        
        # Set CUDA_HOME if not set
        if 'CUDA_HOME' not in os.environ:
            if os.path.exists('/usr/local/cuda'):
                os.environ['CUDA_HOME'] = '/usr/local/cuda'
        
        # Common CUDA library paths
        cuda_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda-11.8/lib64',
            '/usr/local/cuda-11/lib64',
            '/opt/cuda/lib64',
            '/usr/lib/x86_64-linux-gnu'
        ]
        
        # Find existing CUDA libraries
        existing_cuda_paths = []
        cudnn_found = False
        
        for path in cuda_paths:
            if os.path.exists(path):
                # Check if it contains CUDA libraries
                try:
                    files = os.listdir(path)
                    has_cuda = any('libcuda' in f or 'libcublas' in f for f in files)
                    has_cudnn = any('libcudnn' in f for f in files)
                    
                    if has_cuda:
                        existing_cuda_paths.append(path)
                    if has_cudnn:
                        cudnn_found = True
                        
                except:
                    pass
        
        # Update LD_LIBRARY_PATH
        if existing_cuda_paths:
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            new_paths = []
            
            for path in existing_cuda_paths:
                if path not in current_ld_path:
                    new_paths.append(path)
            
            if new_paths:
                if current_ld_path:
                    os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + current_ld_path
                else:
                    os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
                
                print(f"Auto-configured CUDA paths: {new_paths}")
        
        # Check for cuDNN and suggest installation if missing
        if not cudnn_found:
            print("WARNING: cuDNN libraries not found in standard locations")
            print("   This may prevent GPU detection by TensorFlow")
            
            # Try to check if cuDNN is installed via package manager
            try:
                result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
                if result.returncode == 0 and 'libcudnn8' in result.stdout:
                    print("SUCCESS: cuDNN package found via dpkg")
                    cudnn_found = True
                else:
                    print("ERROR: cuDNN package not found via dpkg")
            except:
                pass
        
        return existing_cuda_paths, cudnn_found
    
    # Try to auto-fix CUDA setup
    cuda_paths, cudnn_found = setup_cuda_environment()
    
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
        if cuda_paths:
            print("CUDA paths were auto-configured, but GPU still not detected")
            if not cudnn_found:
                print("WARNING: cuDNN not found - this may be the cause")
                print("   Try: sudo apt install libcudnn8 libcudnn8-dev")
            print("You may need to restart the application for changes to take effect")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running in the storm_env virtual environment")
    sys.exit(1)

class STORMConfig:
    """Configuration class for STORM microscopy analysis"""
    
    def __init__(self):
        self.distance = 25  # GUI shows 25, function gets 24 for 25x25 cutouts
        self.start_z = 0
        self.end_z = 161
        self.csum_slices = 30
        self.min_distance = 5
        self.prominence_sigma = 10.0
        self.support_radius = 2
        self.nm_per_height = 25.0  # Distance in nm between z-heights
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
                stack, csum_image, self.config.distance - 1,  # GUI shows desired size, function gets size-1
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
        input_shape = (self.config.distance, self.config.distance, 1)  # GUI distance = actual cutout size
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
        self.prediction_model = None  # Separate model for predictions
        self.model_metadata = None  # Metadata for model validation
        
        # Training datasets (created by save_cutouts)
        self.train_ds = None
        self.val_ds = None
        self.training_data_ready = False
        self.training_stop_requested = False
        
        # Validation datasets (created by save_validation_cutouts)
        self.validation_cutouts = None
        self.validation_heights = None
        self.validation_data_ready = False
        
        # File selection variables
        self.peak_file_var = None
        self.validation_file_var = None
        self.viz_file_var = None
        self.pred_file_var = None
        
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
        self.log_message(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            self.log_message(f"SUCCESS: {len(gpu_devices)} GPU(s) available for training:")
            for i, gpu in enumerate(gpu_devices):
                self.log_message(f"   GPU {i}: {gpu.name}")
            self.log_message("Ready for fast GPU training!")
        else:
            self.log_message("WARNING: No GPU detected - training will use CPU (slower)")
            self.log_message("   Click 'Check GPU' button for detailed diagnostics")
            self.log_message("   App will still work, just slower for training")
        
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
        
        # Setup log file path
        self.setup_log_file()
    
    def setup_log_file(self):
        """Setup log file for troubleshooting"""
        import datetime
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"storm_gui_{timestamp}.log")
        
        # Initialize log file with header
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"STORM Microscopy GUI Log - Started: {datetime.datetime.now()}\n")
                f.write("=" * 60 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
            self.log_file_path = None
    
    def save_log_to_file(self, message):
        """Save log message to file for troubleshooting"""
        if not hasattr(self, 'log_file_path') or self.log_file_path is None:
            return
            
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
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
        
        # Time Series Analysis Tab
        self.create_time_series_tab(notebook)
    
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
            ("Distance in nm between heights:", "nm_per_height", self.config.nm_per_height),
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
        
        # Current file display
        current_file_frame = ttk.Frame(viz_frame)
        current_file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initialize viz_file_var for compatibility
        self.viz_file_var = tk.StringVar()
        
        ttk.Label(current_file_frame, text="Current file:").pack(side=tk.LEFT, padx=(0, 10))
        self.viz_status_label = ttk.Label(current_file_frame, text="No file selected - go to Data Selection tab")
        self.viz_status_label.pack(side=tk.LEFT)
        
        # Visualize button
        viz_button_frame = ttk.Frame(viz_frame)
        viz_button_frame.pack(fill=tk.X)
        
        self.visualize_button = ttk.Button(viz_button_frame, text="Visualize Peaks", command=self.visualize_peaks)
        self.visualize_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_cutouts_button = ttk.Button(viz_button_frame, text="Save Cutouts", command=self.save_cutouts)
        self.save_cutouts_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Log section (shared with training tab)
        log_frame = ttk.LabelFrame(peak_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text_peak = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_text_peak.pack(fill=tk.BOTH, expand=True)
    
    def create_data_tab(self, notebook):
        """Create simplified data selection tab with two file selectors"""
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Selection")
        
        # Peak Configuration File Selection
        peak_frame = ttk.LabelFrame(data_frame, text="Peak Configuration File", padding=10)
        peak_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(peak_frame, text="Select TIFF file for peak detection and training:").pack(anchor=tk.W, pady=(0, 5))
        
        peak_file_frame = ttk.Frame(peak_frame)
        peak_file_frame.pack(fill=tk.X)
        
        self.peak_file_var = tk.StringVar()
        ttk.Entry(peak_file_frame, textvariable=self.peak_file_var, width=70).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(peak_file_frame, text="Browse", command=self.browse_peak_file).pack(side=tk.LEFT)
        
        # Model Validation File Selection
        validation_frame = ttk.LabelFrame(data_frame, text="Model Validation File", padding=10)
        validation_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(validation_frame, text="Select TIFF file for model validation:").pack(anchor=tk.W, pady=(0, 5))
        
        validation_file_frame = ttk.Frame(validation_frame)
        validation_file_frame.pack(fill=tk.X)
        
        self.validation_file_var = tk.StringVar()
        ttk.Entry(validation_file_frame, textvariable=self.validation_file_var, width=70).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(validation_file_frame, text="Browse", command=self.browse_validation_file).pack(side=tk.LEFT)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(data_frame, text="Instructions", padding=10)
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        instructions_text = """File Selection Guide:

1. Peak Configuration File:
   • Used for peak detection visualization and training data generation
   • Select a TIFF stack with known height distribution
   • This file will be processed in the Peak Configuration tab

2. Model Validation File:
   • Used for testing trained models and generating validation heatmaps
   • Can be the same file as Peak Configuration or a different validation stack
   • This file will be processed in the Model Validation tab

Note: Both files should be TIFF stacks with the same Z-range and parameters for best results."""
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT)
        instructions_label.pack(anchor=tk.W)
    
    def browse_peak_file(self):
        """Browse for peak configuration TIFF file"""
        try:
            self.root.update_idletasks()
            
            # Use default data path as initial directory
            initial_dir = WindowsPathHelper.get_default_data_path()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select TIFF file for Peak Configuration",
                initialdir=initial_dir,
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.peak_file_var.set(filename)
                # Also update the peak configuration tab
                if hasattr(self, 'viz_file_var'):
                    self.viz_file_var.set(filename)
                if hasattr(self, 'viz_status_label'):
                    self.viz_status_label.config(text=f"Selected: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")
    
    def browse_validation_file(self):
        """Browse for model validation TIFF file"""
        try:
            self.root.update_idletasks()
            
            # Use default data path as initial directory
            initial_dir = WindowsPathHelper.get_default_data_path()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select TIFF file for Model Validation",
                initialdir=initial_dir,
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.validation_file_var.set(filename)
                # Also update the model validation tab
                if hasattr(self, 'pred_file_var'):
                    self.pred_file_var.set(filename)
                if hasattr(self, 'validation_status_label'):
                    self.validation_status_label.config(text=f"Selected: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")
    
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
                                         text="WARNING: NO CUTOUTS SAVED! Please go to Peak Configuration and click 'Save Cutouts' first.",
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
        
        ttk.Button(add_block_frame, text="+ Add Training Block", command=self.add_training_block).pack(side=tk.LEFT)
        
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
        
        ttk.Button(button_frame, text="Fix CUDA Path", command=self.fix_cuda_path).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # Save model section (moved to end)
        save_frame = ttk.LabelFrame(self.scrollable_frame, text="Save Model", padding=10)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Directory selection frame
        save_dir_frame = ttk.Frame(save_frame)
        save_dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_dir_var = tk.StringVar()
        # Set default save directory to Documents
        default_save_dir = WindowsPathHelper.get_documents_path()
        self.save_dir_var.set(default_save_dir)
        
        ttk.Entry(save_dir_frame, textvariable=self.save_dir_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(save_dir_frame, text="Browse", command=self.browse_save_directory).pack(side=tk.LEFT, padx=(0, 10))
        
        # Save button
        save_button_frame = ttk.Frame(save_frame)
        save_button_frame.pack(fill=tk.X)
        
        ttk.Button(save_button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT)
        
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
        
        remove_button = ttk.Button(remove_frame, text="X Remove Block", 
                                  command=lambda: self.remove_training_block(block_id))
        remove_button.pack(side=tk.RIGHT)
        
        # Parameters grid
        params_frame = ttk.Frame(block_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Epochs, Adam LR, Huber Delta
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        epochs_var = tk.IntVar(value=50)
        ttk.Entry(params_frame, textvariable=epochs_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(params_frame, text="Adam LR:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        adam_lr_var = tk.DoubleVar(value=2e-4)
        ttk.Entry(params_frame, textvariable=adam_lr_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(params_frame, text="Huber Delta:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        huber_delta_var = tk.DoubleVar(value=0.02)
        ttk.Entry(params_frame, textvariable=huber_delta_var, width=10).grid(row=0, column=5)
        
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
            'epochs_var': epochs_var,
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
        notebook.add(pred_frame, text="Model Validation")
        
        # Model selection section
        model_frame = ttk.LabelFrame(pred_frame, text="Select Model for Prediction", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model path selection frame
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pred_model_var = tk.StringVar()
        # Set default to Documents folder (will be updated if model is saved during training)
        default_model_dir = WindowsPathHelper.get_documents_path()
        self.pred_model_var.set(default_model_dir)
        
        ttk.Entry(model_path_frame, textvariable=self.pred_model_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # Load model button
        model_button_frame = ttk.Frame(model_frame)
        model_button_frame.pack(fill=tk.X)
        
        ttk.Button(model_button_frame, text="Load Model", command=self.load_prediction_model).pack(side=tk.LEFT)
        
        # Model status label
        self.model_status_var = tk.StringVar()
        self.model_status_var.set("No model loaded")
        self.model_status_label = ttk.Label(model_button_frame, textvariable=self.model_status_var, foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Current validation file display
        file_frame = ttk.LabelFrame(pred_frame, text="Current Validation File", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current file display
        current_val_file_frame = ttk.Frame(file_frame)
        current_val_file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initialize pred_file_var for compatibility
        self.pred_file_var = tk.StringVar()
        
        ttk.Label(current_val_file_frame, text="Current file:").pack(side=tk.LEFT, padx=(0, 10))
        self.validation_status_label = ttk.Label(current_val_file_frame, text="No file selected - go to Data Selection tab")
        self.validation_status_label.pack(side=tk.LEFT)
        
        # Validation processing buttons
        viz_buttons_frame = ttk.Frame(file_frame)
        viz_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(viz_buttons_frame, text="Visualize Validation Peaks", command=self.visualize_validation_peaks).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(viz_buttons_frame, text="Save Validation Cutouts", command=self.save_validation_cutouts).pack(side=tk.LEFT, padx=(0, 10))
        
        # Predict button
        predict_button_frame = ttk.Frame(file_frame)
        predict_button_frame.pack(fill=tk.X)
        
        ttk.Button(predict_button_frame, text="Generate Validation Heatmap", command=self.make_prediction).pack(side=tk.LEFT)
        
        # Info label about validation purpose
        info_label = ttk.Label(predict_button_frame, 
                              text="Note: Use training-like stacks to visualize model performance.",
                              foreground="blue")
        info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(pred_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Placeholder for matplotlib plots
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def create_time_series_tab(self, notebook):
        """Create time series analysis tab"""
        ts_frame = ttk.Frame(notebook)
        notebook.add(ts_frame, text="Time Series Analysis")
        
        # Time series file selection
        ts_file_frame = ttk.LabelFrame(ts_frame, text="Select Time Series", padding=10)
        ts_file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ts_file_frame, text="Select TIFF time series file:").pack(anchor=tk.W, pady=(0, 5))
        
        ts_file_selection_frame = ttk.Frame(ts_file_frame)
        ts_file_selection_frame.pack(fill=tk.X)
        
        self.ts_file_var = tk.StringVar()
        ttk.Entry(ts_file_selection_frame, textvariable=self.ts_file_var, width=70).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ts_file_selection_frame, text="Browse", command=self.browse_time_series_file).pack(side=tk.LEFT)
        
        # Model selection for time series
        ts_model_frame = ttk.LabelFrame(ts_frame, text="Select Model", padding=10)
        ts_model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ts_model_frame, text="Select trained model for Z-height prediction:").pack(anchor=tk.W, pady=(0, 5))
        
        ts_model_selection_frame = ttk.Frame(ts_model_frame)
        ts_model_selection_frame.pack(fill=tk.X)
        
        self.ts_model_var = tk.StringVar()
        ttk.Entry(ts_model_selection_frame, textvariable=self.ts_model_var, width=70).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ts_model_selection_frame, text="Browse", command=self.browse_ts_model_file).pack(side=tk.LEFT)
        
        # CSV save location
        csv_save_frame = ttk.LabelFrame(ts_frame, text="Output CSV Location", padding=10)
        csv_save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(csv_save_frame, text="Choose location to save analysis results:").pack(anchor=tk.W, pady=(0, 5))
        
        csv_save_selection_frame = ttk.Frame(csv_save_frame)
        csv_save_selection_frame.pack(fill=tk.X)
        
        self.csv_save_var = tk.StringVar()
        # Start with empty path - user must choose location
        
        ttk.Entry(csv_save_selection_frame, textvariable=self.csv_save_var, width=70).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(csv_save_selection_frame, text="Browse", command=self.browse_csv_save_location).pack(side=tk.LEFT)
        
        # Analysis buttons
        analyze_frame = ttk.Frame(ts_frame)
        analyze_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.analyze_button = ttk.Button(analyze_frame, text="Analyze", command=self.analyze_time_series)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_test_button = ttk.Button(analyze_frame, text="Analyze First 10 Frames", command=self.analyze_time_series_test)
        self.analyze_test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_analysis_button = ttk.Button(analyze_frame, text="Stop Analysis", command=self.stop_time_series_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Analysis control flag
        self.ts_analysis_stop_flag = False
        
        # Progress bar for time series analysis
        self.ts_progress_var = tk.DoubleVar()
        self.ts_progress_bar = ttk.Progressbar(analyze_frame, variable=self.ts_progress_var, maximum=100)
        self.ts_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Instructions
        instructions_frame = ttk.LabelFrame(ts_frame, text="Instructions", padding=10)
        instructions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        instructions_text = """Time Series Analysis Workflow:

1. Select Time Series: Choose a multi-frame TIFF file containing time series data
2. Select Model: Choose a trained .h5 model file for Z-height prediction
3. Output CSV Location: Specify where to save the analysis results
4. Click Analyze to start processing

The analysis will:
• Process each frame to detect PSF peaks using bandpass filtering
• Localize peaks with sub-pixel precision using iterative centroid method
• Extract cutouts around each peak and predict Z-heights using the selected model
• Generate a CSV file with columns: peak_id, frame, x, y, z

Note: This process may take several minutes depending on the number of frames and peaks."""
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT)
        instructions_label.pack(anchor=tk.W)
        
        # Results display for time series
        ts_results_frame = ttk.LabelFrame(ts_frame, text="Analysis Log", padding=10)
        ts_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ts_results_text = scrolledtext.ScrolledText(ts_results_frame, height=25, width=120, state=tk.DISABLED)
        self.ts_results_text.pack(fill=tk.BOTH, expand=True)
    
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
    
    def browse_time_series_file(self):
        """Browse for time series TIFF file"""
        try:
            self.root.update_idletasks()
            
            # Use default data path as initial directory
            initial_dir = WindowsPathHelper.get_default_data_path()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select Time Series TIFF File",
                initialdir=initial_dir,
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.ts_file_var.set(filename)
                self.ts_log_message(f"Selected time series file: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting time series file: {str(e)}")
    
    def browse_ts_model_file(self):
        """Browse for model file for time series analysis"""
        try:
            self.root.update_idletasks()
            
            # Use Documents as initial directory
            initial_dir = WindowsPathHelper.get_documents_path()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select Trained Model File",
                initialdir=initial_dir,
                filetypes=[("H5 model files", "*.h5"), ("Keras model files", "*.keras"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.ts_model_var.set(filename)
                self.ts_log_message(f"Selected model file: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting model file: {str(e)}")
    
    def browse_csv_save_location(self):
        """Browse for CSV save location"""
        try:
            self.root.update_idletasks()
            
            # Use Documents as initial directory
            initial_dir = WindowsPathHelper.get_documents_path()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
            
            filename = filedialog.asksaveasfilename(
                title="Save Analysis Results As",
                initialdir=initial_dir,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.csv_save_var.set(filename)
                self.ts_log_message(f"Results will be saved to: {Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting save location: {str(e)}")
    
    def ts_log_message(self, message):
        """Log message to time series results text widget and log file"""
        if hasattr(self, 'ts_results_text'):
            self.ts_results_text.config(state=tk.NORMAL)
            self.ts_results_text.insert(tk.END, f"{message}\n")
            self.ts_results_text.see(tk.END)
            self.ts_results_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        
        # Also save to log file for troubleshooting
        self.save_ts_log_to_file(message)
    
    def setup_ts_log_file(self):
        """Setup time series log file for troubleshooting"""
        import datetime
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Create timestamped log file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.ts_log_file_path = logs_dir / f"time_series_analysis_{timestamp}.log"
            
            # Write initial log entry
            with open(self.ts_log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Time Series Analysis Log - {datetime.datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                
        except Exception as e:
            print(f"Warning: Could not create time series log file: {e}")
            self.ts_log_file_path = None
    
    def save_ts_log_to_file(self, message):
        """Save time series log message to file for troubleshooting"""
        if not hasattr(self, 'ts_log_file_path') or self.ts_log_file_path is None:
            self.setup_ts_log_file()
        
        try:
            if self.ts_log_file_path:
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                with open(self.ts_log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Warning: Could not write to time series log file: {e}")
    
    def analyze_time_series(self):
        """Start time series analysis in a separate thread"""
        # Validate inputs
        ts_file = self.ts_file_var.get()
        model_file = self.ts_model_var.get()
        csv_file = self.csv_save_var.get()
        
        if not ts_file or not os.path.exists(ts_file):
            messagebox.showerror("Error", "Please select a valid time series TIFF file")
            return
        
        if not model_file or not os.path.exists(model_file):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        if not csv_file:
            messagebox.showerror("Error", "Please specify a CSV save location")
            return
        
        # Setup log file for this analysis session
        self.setup_ts_log_file()
        
        # Disable analyze buttons during processing
        self.analyze_button.config(state=tk.DISABLED)
        self.analyze_test_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.NORMAL)
        self.ts_analysis_stop_flag = False
        self.ts_progress_var.set(0)
        
        # Clear previous results
        self.ts_results_text.config(state=tk.NORMAL)
        self.ts_results_text.delete(1.0, tk.END)
        self.ts_results_text.config(state=tk.DISABLED)
        
        # Start analysis in separate thread
        self.ts_analysis_thread = threading.Thread(
            target=self.time_series_analysis_thread,
            args=(ts_file, model_file, csv_file),
            daemon=True
        )
        self.ts_analysis_thread.start()
    
    def analyze_time_series_test(self):
        """Start time series analysis for first 10 frames only (for testing)"""
        # Validate inputs
        ts_file = self.ts_file_var.get()
        model_file = self.ts_model_var.get()
        csv_file = self.csv_save_var.get()
        
        if not ts_file or not os.path.exists(ts_file):
            messagebox.showerror("Error", "Please select a valid time series TIFF file")
            return
        
        if not model_file or not os.path.exists(model_file):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        if not csv_file:
            messagebox.showerror("Error", "Please specify a CSV save location")
            return
        
        # Add "_test" to the CSV filename
        csv_path = Path(csv_file)
        test_csv_file = str(csv_path.parent / f"{csv_path.stem}_test{csv_path.suffix}")
        
        # Setup log file for this analysis session
        self.setup_ts_log_file()
        
        # Disable analyze buttons during processing
        self.analyze_button.config(state=tk.DISABLED)
        self.analyze_test_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.NORMAL)
        self.ts_analysis_stop_flag = False
        self.ts_progress_var.set(0)
        
        # Clear previous results
        self.ts_results_text.config(state=tk.NORMAL)
        self.ts_results_text.delete(1.0, tk.END)
        self.ts_results_text.config(state=tk.DISABLED)
        
        # Start analysis in separate thread (test mode - first 10 frames)
        self.ts_analysis_thread = threading.Thread(
            target=self.time_series_analysis_thread,
            args=(ts_file, model_file, test_csv_file, 10),  # Limit to 10 frames
            daemon=True
        )
        self.ts_analysis_thread.start()
    
    def stop_time_series_analysis(self):
        """Stop the current time series analysis"""
        self.ts_analysis_stop_flag = True
        self.ts_log_message("STOPPING: Analysis stop requested by user...")
        self.stop_analysis_button.config(state=tk.DISABLED)
    
    def time_series_analysis_thread(self, ts_file, model_file, csv_file, max_frames=None):
        """Time series analysis thread function"""
        try:
            import csv
            import pandas as pd
            
            # Import detection functions from detectionalgo.py
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from detectionalgo import (
                bandpass_filter, find_local_peaks, robust_max, 
                extract_rois, filter_rois_single_central_peak,
                iterative_weighted_centroid
            )
            
            self.ts_log_message("=== Starting Time Series Analysis ===")
            if max_frames:
                self.ts_log_message(f"TEST MODE: Processing first {max_frames} frames only")
            self.ts_log_message(f"Time series file: {Path(ts_file).name}")
            self.ts_log_message(f"Model file: {Path(model_file).name}")
            self.ts_log_message(f"Output CSV: {Path(csv_file).name}")
            
            # Load the trained model
            self.ts_log_message("Loading trained model...")
            model = tf.keras.models.load_model(model_file)
            
            # Detect model input dimensions
            input_shape = model.input_shape
            self.ts_log_message(f"Model input shape: {input_shape}")
            
            if len(input_shape) == 4:  # (batch, height, width, channels)
                expected_height = input_shape[1]
                expected_width = input_shape[2]
                if expected_height == expected_width:
                    cutout_size = expected_height
                    self.ts_log_message(f"Detected model expects {cutout_size}x{cutout_size} cutouts")
                else:
                    self.ts_log_message(f"WARNING: Model expects non-square input {expected_height}x{expected_width}, using {expected_height}x{expected_height}")
                    cutout_size = expected_height
            else:
                self.ts_log_message("WARNING: Unexpected model input shape, defaulting to 25x25 cutouts")
                cutout_size = 25
            
            # Load time series TIFF
            self.ts_log_message("Loading time series TIFF file...")
            with tiff.TiffFile(ts_file) as tif:
                time_series = tif.asarray()
            
            if len(time_series.shape) != 3:
                raise ValueError(f"Expected 3D time series (frames, height, width), got shape {time_series.shape}")
            
            n_frames, height, width = time_series.shape
            
            # Limit frames if in test mode
            if max_frames and max_frames < n_frames:
                n_frames = max_frames
                time_series = time_series[:max_frames]
                self.ts_log_message(f"Limited to first {n_frames} frames for testing")
            
            self.ts_log_message(f"Processing time series: {n_frames} frames, {height}x{width} pixels")
            self.ts_log_message(f"Using {cutout_size}x{cutout_size} cutouts to match model input")
            
            # Prepare CSV output
            results = []
            peak_counter = 0
            
            # Process each frame
            for frame_idx in range(n_frames):
                # Check stop flag
                if self.ts_analysis_stop_flag:
                    self.ts_log_message("Analysis stopped by user request")
                    break
                
                self.ts_log_message(f"Processing frame {frame_idx + 1}/{n_frames}...")
                
                # Update progress
                progress = (frame_idx / n_frames) * 90  # Reserve 10% for final steps
                self.root.after(0, lambda p=progress: self.ts_progress_var.set(p))
                
                frame = time_series[frame_idx]
                
                # Apply bandpass filter ONLY for peak detection
                filtered_frame = bandpass_filter(frame, sigma_small=1.2, sigma_large=3.0)
                
                # Find peaks using filtered frame
                threshold = 0.1 * robust_max(filtered_frame, k=10)
                peaks = find_local_peaks(filtered_frame, threshold=threshold, min_distance=4)
                
                if len(peaks) == 0:
                    self.ts_log_message(f"  No peaks found in frame {frame_idx + 1}")
                    continue
                
                # Extract ROIs from both filtered (for localization) and raw (for CNN) frames
                # Use the detected cutout_size instead of hardcoded 25
                filtered_rois, kept_peaks_filtered = extract_rois(filtered_frame, peaks, cutout_size)  # For localization
                raw_rois, kept_peaks_raw = extract_rois(frame, peaks, cutout_size)  # For CNN prediction
                
                if len(filtered_rois) == 0 or len(raw_rois) == 0:
                    self.ts_log_message(f"  No valid ROIs in frame {frame_idx + 1}")
                    continue
                
                # Filter ROIs to keep only those with single central peak (using filtered ROIs for detection)
                filtered_rois_clean, kept_indices, central_peaks, _ = filter_rois_single_central_peak(filtered_rois)
                
                if len(filtered_rois_clean) == 0:
                    self.ts_log_message(f"  No valid single-peak ROIs in frame {frame_idx + 1}")
                    continue
                
                self.ts_log_message(f"  Found {len(filtered_rois_clean)} valid peaks")
                
                # Process each valid ROI
                for roi_idx, (filtered_roi, raw_roi, peak_pos) in enumerate(zip(
                    filtered_rois_clean, 
                    raw_rois[kept_indices], 
                    kept_peaks_filtered[kept_indices]
                )):
                    # Check stop flag during processing
                    if self.ts_analysis_stop_flag:
                        self.ts_log_message("Analysis stopped by user request")
                        break
                    
                    # Get sub-pixel localization using iterative centroid on FILTERED ROI (better accuracy)
                    dy, dx = iterative_weighted_centroid(filtered_roi)
                    
                    # Calculate final coordinates (sub-pixel precision)
                    final_x = peak_pos[1] + dx  # peak_pos is (row, col) = (y, x)
                    final_y = peak_pos[0] + dy
                    
                    # Normalize RAW ROI for model prediction (same as training normalization)
                    roi_normalized = (raw_roi - raw_roi.min()) / (raw_roi.max() - raw_roi.min() + 1e-8)
                    roi_input = roi_normalized.reshape(1, cutout_size, cutout_size, 1)  # Use detected cutout_size
                    
                    # Predict Z-height
                    z_pred_normalized = model.predict(roi_input, verbose=0)[0, 0]
                    
                    # Denormalize Z prediction (assuming -2000 to +2000 nm range)
                    z_pred_nm = z_pred_normalized * 4000 - 2000  # Convert [0,1] to [-2000, +2000] nm
                    
                    # Store result
                    peak_counter += 1
                    results.append({
                        'peak_id': peak_counter,
                        'frame': frame_idx + 1,  # 1-based frame numbering
                        'x': final_x,
                        'y': final_y,
                        'z': z_pred_nm
                    })
                
                # Break out of frame loop if stopped
                if self.ts_analysis_stop_flag:
                    break
            
            # Save results to CSV (only if not stopped)
            if not self.ts_analysis_stop_flag:
                self.ts_log_message("Saving results to CSV...")
                self.root.after(0, lambda: self.ts_progress_var.set(95))
                
                df = pd.DataFrame(results)
                df.to_csv(csv_file, index=False)
                
                self.ts_log_message(f"Analysis complete! Processed {len(results)} peaks across {n_frames} frames")
                self.ts_log_message(f"Results saved to: {csv_file}")
                
                # Show summary statistics
                if len(results) > 0:
                    self.ts_log_message("\n=== Summary Statistics ===")
                    self.ts_log_message(f"Total peaks detected: {len(results)}")
                    self.ts_log_message(f"Average peaks per frame: {len(results)/n_frames:.1f}")
                    self.ts_log_message(f"Z-height range: {df['z'].min():.1f} to {df['z'].max():.1f} nm")
                    self.ts_log_message(f"X coordinate range: {df['x'].min():.1f} to {df['x'].max():.1f} pixels")
                    self.ts_log_message(f"Y coordinate range: {df['y'].min():.1f} to {df['y'].max():.1f} pixels")
                
                self.root.after(0, lambda: self.ts_progress_var.set(100))
            else:
                self.ts_log_message(f"Analysis stopped. Processed {len(results)} peaks from {frame_idx + 1} frames before stopping.")
                if len(results) > 0:
                    # Save partial results
                    df = pd.DataFrame(results)
                    partial_csv = csv_file.replace('.csv', '_partial.csv')
                    df.to_csv(partial_csv, index=False)
                    self.ts_log_message(f"Partial results saved to: {partial_csv}")
            
        except Exception as e:
            error_msg = f"ERROR: Time series analysis failed: {str(e)}"
            self.ts_log_message(error_msg)
            
            import traceback
            full_traceback = traceback.format_exc()
            self.ts_log_message("Full traceback:")
            self.ts_log_message(full_traceback)
            
            # Also log to main application log
            self.log_message(f"Time series analysis error: {str(e)}")
            
            # Show error dialog - fix lambda scope issue
            error_text = f"Time series analysis failed with error:\n\n{str(e)}\n\nCheck the analysis log and log files for details."
            self.root.after(0, lambda msg=error_text: messagebox.showerror(
                "Analysis Failed", msg
            ))
            
        finally:
            # Re-enable analyze buttons
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.analyze_test_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_analysis_button.config(state=tk.DISABLED))
            
            # Log completion
            if hasattr(self, 'ts_log_file_path') and self.ts_log_file_path:
                self.ts_log_message(f"\nLog file saved to: {self.ts_log_file_path}")
                self.ts_log_message("Analysis session completed.")
    
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
                stack, csum_image, self.config.distance - 1,  # GUI shows desired size, function gets size-1
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
                stack, csum_image, self.config.distance - 1,  # GUI shows desired size, function gets size-1
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
            
            # Normalize PSFs (per-image normalization)
            psfs = normalize_0_to_1(psfs)
            
            # Convert heights from z-indices to nm, then normalize to [0, 1]
            # Training stack: middle height = 0 nm, range is -2000 to +2000 nm
            middle_index = (self.config.end_z - self.config.start_z) / 2
            heights_nm = (heights - middle_index) * self.config.nm_per_height
            
            self.log_message(f"Height range in nm: {np.min(heights_nm):.1f} to {np.max(heights_nm):.1f}")
            
            # Normalize to [0, 1] using the training range
            nm_min = -2000.0
            nm_max = 2000.0
            heights = (heights_nm - nm_min) / (nm_max - nm_min)
            
            self.log_message(f"Normalized heights range: {np.min(heights):.3f} to {np.max(heights):.3f}")
            
            # Store the normalization parameters for later use
            self.training_nm_min = nm_min
            self.training_nm_max = nm_max
            
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
        """Get list of selected files from the new file selection system"""
        selected = []
        
        # Get the peak configuration file (used for training)
        peak_file = self.peak_file_var.get().strip() if hasattr(self, 'peak_file_var') and self.peak_file_var else ""
        
        if peak_file and os.path.exists(peak_file):
            selected.append(Path(peak_file))
        
        return selected
    
    def preview_files(self):
        """Preview selected file"""
        selected = self.get_selected_files()
        if not selected:
            messagebox.showwarning("Warning", "No peak configuration file selected.\nPlease go to Data Selection tab and select a file.")
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
        """Check and log GPU availability and status with detailed diagnostics"""
        try:
            self.log_message("=== GPU Diagnostic Information ===")
            
            # Check TensorFlow build info
            self.log_message(f"TensorFlow version: {tf.__version__}")
            self.log_message(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
            
            # Check if GPU is available
            gpu_available = tf.config.list_physical_devices('GPU')
            
            if gpu_available:
                self.log_message(f"SUCCESS: GPU detected: {len(gpu_available)} device(s)")
                for i, gpu in enumerate(gpu_available):
                    self.log_message(f"   GPU {i}: {gpu.name}")
                
                # Check if TensorFlow can actually use the GPU
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        result = tf.matmul(test_tensor, test_tensor)
                        self.log_message("SUCCESS: GPU computation test successful")
                    
                    # Log GPU memory info
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu_available[0])
                        if 'device_name' in gpu_details:
                            self.log_message(f"   GPU Name: {gpu_details['device_name']}")
                    except:
                        pass
                        
                    return True
                    
                except Exception as gpu_test_error:
                    self.log_message(f"ERROR: GPU computation test failed: {str(gpu_test_error)}")
                    self.log_message("   GPU detected but not usable for computation")
                    return False
            else:
                self.log_message("ERROR: No GPU detected by TensorFlow")
                
                # Additional diagnostics
                self.log_message("=== Diagnostic Steps ===")
                
                # Check CUDA installation
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.log_message("SUCCESS: nvidia-smi works - NVIDIA driver is installed")
                        # Extract GPU info from nvidia-smi
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                                gpu_info = line.strip()
                                self.log_message(f"   Hardware GPU found: {gpu_info}")
                    else:
                        self.log_message("ERROR: nvidia-smi failed - NVIDIA driver issue")
                except FileNotFoundError:
                    self.log_message("ERROR: nvidia-smi not found - NVIDIA driver not installed")
                except subprocess.TimeoutExpired:
                    self.log_message("ERROR: nvidia-smi timeout - driver issue")
                except Exception as e:
                    self.log_message(f"ERROR: nvidia-smi error: {str(e)}")
                
                # Check CUDA toolkit
                try:
                    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
                        if version_line:
                            self.log_message(f"SUCCESS: CUDA toolkit found: {version_line[0].strip()}")
                        else:
                            self.log_message("SUCCESS: CUDA toolkit found (version unclear)")
                    else:
                        self.log_message("ERROR: nvcc failed - CUDA toolkit issue")
                except FileNotFoundError:
                    self.log_message("ERROR: nvcc not found - CUDA toolkit not in PATH")
                except Exception as e:
                    self.log_message(f"ERROR: nvcc error: {str(e)}")
                
                # Check LD_LIBRARY_PATH
                ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                if 'cuda' in ld_path.lower():
                    self.log_message(f"SUCCESS: CUDA in LD_LIBRARY_PATH: {ld_path}")
                else:
                    self.log_message("ERROR: CUDA not in LD_LIBRARY_PATH")
                    self.log_message("   This might prevent TensorFlow from finding CUDA libraries")
                
                # Suggest solutions
                self.log_message("=== Possible Solutions ===")
                self.log_message("1. Restart WSL: 'wsl --shutdown' then reopen")
                self.log_message("2. Check if running in correct environment: 'source storm_env/bin/activate'")
                self.log_message("3. Verify WSL can access GPU: Windows 11 with WSL 2.0+ required")
                self.log_message("4. Update NVIDIA drivers to latest version")
                
                return False
                
        except Exception as e:
            self.log_message(f"ERROR: GPU check failed with error: {str(e)}")
            self.log_message("   Training will use CPU (very slow)")
            return False
    
    def fix_cuda_path(self):
        """Attempt to fix CUDA library path issues"""
        self.log_message("=== Attempting CUDA Path Fix ===")
        
        try:
            import os
            
            # Common CUDA library paths
            cuda_paths = [
                '/usr/local/cuda/lib64',
                '/usr/local/cuda-11.8/lib64', 
                '/usr/local/cuda-11/lib64',
                '/opt/cuda/lib64',
                '/usr/lib/x86_64-linux-gnu'
            ]
            
            self.log_message("Searching for CUDA libraries...")
            
            # Find existing CUDA libraries
            existing_cuda_paths = []
            for path in cuda_paths:
                if os.path.exists(path):
                    try:
                        files = os.listdir(path)
                        cuda_libs = [f for f in files if 'libcuda' in f or 'libcublas' in f or 'libcudnn' in f]
                        if cuda_libs:
                            existing_cuda_paths.append(path)
                            self.log_message(f"SUCCESS: Found CUDA libraries in: {path}")
                            self.log_message(f"   Libraries: {cuda_libs[:3]}{'...' if len(cuda_libs) > 3 else ''}")
                    except Exception as e:
                        self.log_message(f"ERROR: Cannot access {path}: {str(e)}")
            
            if existing_cuda_paths:
                # Update LD_LIBRARY_PATH
                current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                new_paths = []
                
                for path in existing_cuda_paths:
                    if path not in current_ld_path:
                        new_paths.append(path)
                
                if new_paths:
                    if current_ld_path:
                        new_ld_path = ':'.join(new_paths) + ':' + current_ld_path
                    else:
                        new_ld_path = ':'.join(new_paths)
                    
                    os.environ['LD_LIBRARY_PATH'] = new_ld_path
                    self.log_message(f"SUCCESS: Updated LD_LIBRARY_PATH with: {new_paths}")
                    self.log_message(f"   New LD_LIBRARY_PATH: {new_ld_path}")
                    
                    # Test if GPU is now detected
                    self.log_message("Testing GPU detection after path fix...")
                    
                    # Simple test - check if GPU is now available
                    try:
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpus:
                            self.log_message(f"🎉 SUCCESS! GPU now detected: {len(gpus)} device(s)")
                            for i, gpu in enumerate(gpus):
                                self.log_message(f"   GPU {i}: {gpu.name}")
                            
                            messagebox.showinfo("CUDA Fix Successful!", 
                                              f"GPU now detected!\n\n"
                                              f"Found {len(gpus)} GPU(s)\n"
                                              f"Training will now use GPU acceleration.")
                        else:
                            self.log_message("ERROR: GPU still not detected after path fix")
                            self.log_message("   You may need to restart the application")
                            messagebox.showinfo("Restart Required", 
                                              "CUDA paths updated but GPU not yet detected.\n\n"
                                              "Please restart the application to apply changes.")
                    except Exception as test_error:
                        self.log_message(f"ERROR: Error testing GPU after fix: {str(test_error)}")
                        messagebox.showinfo("Restart Required", 
                                          "CUDA paths updated.\n\n"
                                          "Please restart the application to apply changes.")
                else:
                    self.log_message("SUCCESS: CUDA paths already configured correctly")
                    self.log_message("   GPU detection issue may be elsewhere")
            else:
                self.log_message("ERROR: No CUDA libraries found in common locations")
                self.log_message("   CUDA toolkit may not be properly installed")
                
                # Suggest installation
                self.log_message("=== Installation Suggestions ===")
                self.log_message("1. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
                self.log_message("2. Or download from: https://developer.nvidia.com/cuda-downloads")
                self.log_message("3. Ensure WSL 2 with GPU support is enabled")
                
                messagebox.showwarning("CUDA Not Found", 
                                     "CUDA libraries not found in common locations.\n\n"
                                     "You may need to install the CUDA toolkit:\n"
                                     "sudo apt install nvidia-cuda-toolkit\n\n"
                                     "Check the log for detailed information.")
                
        except Exception as e:
            self.log_message(f"ERROR: CUDA path fix failed: {str(e)}")
            messagebox.showerror("Fix Failed", f"Could not fix CUDA paths: {str(e)}")

    def train_model_thread(self, selected_files):
        """Multi-stage training thread function"""
        try:
            # Reset stop flag
            self.training_stop_requested = False
            
            # Check GPU status first
            self.log_message("=== System Check ===")
            gpu_available = self.check_gpu_status()
            
            if not gpu_available:
                self.log_message("WARNING: Proceeding with CPU training (will be slower)")
                self.log_message("   Consider checking GPU setup for faster training")
            else:
                self.log_message("SUCCESS: GPU ready for fast training")
            
            # Check if we have pre-processed datasets from "Save Cutouts"
            if not self.training_data_ready or self.train_ds is None or self.val_ds is None:
                self.queue.put(('error', 'No cutouts saved! Please go to Peak Configuration and click "Save Cutouts" first.'))
                return
            
            self.log_message("Starting multi-stage training with pre-processed datasets")
            
            # Build initial model
            self.update_progress(10, "Building neural network...")
            input_shape = (self.config.distance, self.config.distance, 1)  # GUI distance = actual cutout size
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
                    # For fine-tuning blocks, use the epochs specified in the block
                    epochs = block['epochs_var'].get()
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
                model_path = self.config.output_path / f"model_distance_{self.config.distance}.h5"
                
                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        str(model_path), 
                        monitor='val_mae', 
                        save_best_only=True, 
                        mode='min',
                        save_format='h5'
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
        """Add message to log and save to file"""
        self.queue.put(('log', message))
        
        # Also save to log file for troubleshooting
        self.save_log_to_file(message)
    
    def update_config_from_gui(self):
        """Update configuration from GUI values (for peak configuration parameters)"""
        # Only update the peak detection parameters from the Peak Configuration tab
        for param, var in self.param_vars.items():
            setattr(self.config, param, var.get())
    
    def browse_save_directory(self):
        """Browse for directory to save model"""
        try:
            # Force GUI update before opening dialog
            self.root.update_idletasks()
            
            # Use current directory from text field as initial directory
            current_dir = self.save_dir_var.get()
            if current_dir and os.path.exists(current_dir):
                initial_dir = current_dir
            else:
                # Fallback to default if current directory doesn't exist
                initial_dir = WindowsPathHelper.get_documents_path()
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.expanduser("~")  # Final fallback to home directory
            
            directory = filedialog.askdirectory(
                title="Select Directory to Save Model", 
                initialdir=initial_dir,
                parent=self.root
            )
            
            if directory:
                self.save_dir_var.set(directory)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error opening directory dialog: {str(e)}")
            messagebox.showinfo("Alternative", 
                              "You can type the directory path directly in the text field")

    def save_model(self):
        """Save trained model to directory specified in text field"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model to save")
            return
        
        try:
            # Get directory from text field
            directory = self.save_dir_var.get().strip()
            
            if not directory:
                messagebox.showwarning("Warning", "Please select a directory to save the model")
                return
            
            if not os.path.exists(directory):
                messagebox.showerror("Error", f"Directory does not exist: {directory}")
                return
            
            # Create model filename with distance parameter
            model_filename = f"storm_model_distance_{self.config.distance}.h5"
            model_path = os.path.join(directory, model_filename)
            
            # Create metadata filename
            metadata_filename = f"storm_model_distance_{self.config.distance}_metadata.json"
            metadata_path = os.path.join(directory, metadata_filename)
            
            # Check if files already exist
            if os.path.exists(model_path) or os.path.exists(metadata_path):
                result = messagebox.askyesno("File Exists", 
                                           f"Model files already exist:\n{model_path}\n{metadata_path}\n\n"
                                           f"Do you want to overwrite them?")
                if not result:
                    self.log_message("Model save cancelled - files already exist")
                    return
            
            # Save the model
            self.model.save(model_path)
            
            # Create training metadata
            training_metadata = {
                "model_version": "1.0",
                "training_config": {
                    "distance": self.config.distance,
                    "start_z": self.config.start_z,
                    "end_z": self.config.end_z,
                    "z_range": self.config.end_z - self.config.start_z,
                    "nm_per_height": self.config.nm_per_height,
                    "nm_range_min": -2000.0,
                    "nm_range_max": 2000.0,
                    "csum_slices": self.config.csum_slices,
                    "min_distance": self.config.min_distance,
                    "prominence_sigma": self.config.prominence_sigma,
                    "support_radius": self.config.support_radius
                },
                "model_info": {
                    "input_shape": [self.config.distance, self.config.distance, 1],
                    "cutout_size": self.config.distance,
                    "expected_z_range": self.config.end_z - self.config.start_z
                }
            }
            
            # Save metadata as JSON
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2)
            
            # Update prediction tab with saved model path
            self.pred_model_var.set(model_path)
            
            # Log and notify user
            self.log_message(f"SUCCESS: Model saved successfully: {model_path}")
            self.log_message(f"SUCCESS: Metadata saved: {metadata_path}")
            messagebox.showinfo("Model Saved", 
                              f"Model saved successfully!\n\n"
                              f"Model: {model_path}\n"
                              f"Metadata: {metadata_path}\n"
                              f"Distance parameter: {self.config.distance}\n"
                              f"Z-range: {self.config.start_z}-{self.config.end_z}\n\n"
                              f"The model path has been set in the Model Validation tab.\n"
                              f"You can now load it for validation.")
                
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Save Error", error_msg)
    
    def load_model(self):
        """Load trained model"""
        try:
            self.root.update_idletasks()
            
            initial_dir = WindowsPathHelper.get_documents_path()
            
            filename = filedialog.askopenfilename(
                title="Load Model",
                initialdir=initial_dir,
                filetypes=[("Keras models", "*.h5"), ("Keras models (new)", "*.keras"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.model = tf.keras.models.load_model(filename)
                messagebox.showinfo("Success", f"Model loaded from {filename}")
                self.log_message(f"Model loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def browse_model_file(self):
        """Browse for model file to load for prediction"""
        try:
            # Force GUI update before opening dialog
            self.root.update_idletasks()
            
            # Use current directory from text field as initial directory
            current_path = self.pred_model_var.get()
            if current_path and os.path.exists(current_path):
                if os.path.isfile(current_path):
                    initial_dir = os.path.dirname(current_path)
                else:
                    initial_dir = current_path
            else:
                # Fallback to default if current path doesn't exist
                initial_dir = WindowsPathHelper.get_documents_path()
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.expanduser("~")
            
            filename = filedialog.askopenfilename(
                title="Select Model File",
                initialdir=initial_dir,
                filetypes=[("Keras models", "*.h5"), ("Keras models (new)", "*.keras"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                self.pred_model_var.set(filename)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file dialog: {str(e)}")
            messagebox.showinfo("Alternative", 
                              "You can type the model file path directly in the text field")

    def load_prediction_model(self):
        """Load model for prediction with metadata validation"""
        try:
            model_path = self.pred_model_var.get().strip()
            
            if not model_path:
                messagebox.showwarning("Warning", "Please select a model file to load")
                return
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file does not exist: {model_path}")
                return
            
            # Try to load metadata file
            metadata_path = model_path.replace('.h5', '_metadata.json')
            model_metadata = None
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        model_metadata = json.load(f)
                    self.log_message(f"SUCCESS: Metadata loaded: {metadata_path}")
                except Exception as e:
                    self.log_message(f"WARNING: Could not load metadata: {str(e)}")
            else:
                self.log_message(f"WARNING: No metadata file found: {metadata_path}")
                self.log_message("   Model validation will be limited")
            
            # Load the model
            self.prediction_model = tf.keras.models.load_model(model_path)
            
            # Store metadata for validation
            self.model_metadata = model_metadata
            
            # Validate model dimensions if metadata is available
            if model_metadata:
                expected_distance = model_metadata['training_config']['distance']
                expected_z_range = model_metadata['training_config']['z_range']
                expected_start_z = model_metadata['training_config']['start_z']
                expected_end_z = model_metadata['training_config']['end_z']
                
                # Check if current configuration matches training configuration
                current_z_range = self.config.end_z - self.config.start_z
                
                validation_warnings = []
                
                if expected_distance != self.config.distance:
                    validation_warnings.append(f"Distance mismatch: Model trained with {expected_distance}, current setting is {self.config.distance}")
                
                if expected_z_range != current_z_range:
                    validation_warnings.append(f"Z-range mismatch: Model trained with {expected_z_range} slices ({expected_start_z}-{expected_end_z}), current setting is {current_z_range} slices ({self.config.start_z}-{self.config.end_z})")
                
                if validation_warnings:
                    warning_msg = "WARNING: Configuration Mismatch Detected:\n\n" + "\n".join(validation_warnings)
                    warning_msg += "\n\nThis may cause prediction errors. Consider adjusting your configuration to match the training parameters."
                    messagebox.showwarning("Configuration Mismatch", warning_msg)
                    for warning in validation_warnings:
                        self.log_message(f"WARNING: {warning}")
                else:
                    self.log_message("SUCCESS: Configuration matches training parameters")
            
            # Update status
            model_name = os.path.basename(model_path)
            self.model_status_var.set(f"SUCCESS: Loaded: {model_name}")
            self.model_status_label.config(foreground="green")
            
            # Log the action
            self.log_message(f"SUCCESS: Prediction model loaded: {model_path}")
            
            success_msg = f"Model loaded successfully!\n\nFile: {model_name}\n"
            if model_metadata:
                success_msg += f"Training distance: {model_metadata['training_config']['distance']}\n"
                success_msg += f"Training Z-range: {model_metadata['training_config']['start_z']}-{model_metadata['training_config']['end_z']}\n"
            success_msg += "Ready for validation."
            
            messagebox.showinfo("Model Loaded", success_msg)
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_var.set("ERROR: Load failed")
            self.model_status_label.config(foreground="red")
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Load Error", error_msg)

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
    
    def visualize_validation_peaks(self):
        """Visualize peaks on validation file (like Peak Configuration)"""
        try:
            file_path = self.pred_file_var.get().strip()
            
            if not file_path:
                messagebox.showwarning("Warning", "Please select a TIFF file first")
                return
            
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File does not exist: {file_path}")
                return
            
            self.log_message(f"🔍 Visualizing validation peaks: {os.path.basename(file_path)}")
            
            # Update config from GUI
            self.update_config_from_gui()
            
            # Load TIFF stack
            stack = tiff.imread(file_path)
            self.log_message(f"Loaded validation stack: {stack.shape}")
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, 
                self.config.start_z, 
                self.config.end_z, 
                self.config.csum_slices
            )
            
            # Extract PSF cutouts with visualization
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance - 1,  # GUI shows desired size, function gets size-1
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=True  # This will show the visualization
            )
            
            self.log_message(f"SUCCESS: Detected {len(peaks)} peaks for validation")
            
        except Exception as e:
            error_msg = f"Error visualizing validation peaks: {str(e)}"
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Visualization Error", error_msg)
    
    def save_validation_cutouts(self):
        """Save validation cutouts for prediction (like Peak Configuration)"""
        try:
            file_path = self.pred_file_var.get().strip()
            
            if not file_path:
                messagebox.showwarning("Warning", "Please select a TIFF file first")
                return
            
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File does not exist: {file_path}")
                return
            
            self.log_message(f"💾 Processing validation cutouts: {os.path.basename(file_path)}")
            
            # Update config from GUI
            self.update_config_from_gui()
            
            # Load TIFF stack
            stack = tiff.imread(file_path)
            self.log_message(f"Loaded validation stack: {stack.shape}")
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, 
                self.config.start_z, 
                self.config.end_z, 
                self.config.csum_slices
            )
            
            # Extract PSF cutouts (without visualization this time)
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance - 1,  # GUI shows desired size, function gets size-1
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=False  # No visualization this time
            )
            
            if not cutouts:
                messagebox.showwarning("Warning", "No cutouts extracted. Try adjusting parameters.")
                return
            
            # Process cutouts for prediction
            psfs = np.expand_dims(np.array([cutouts[i][0] for i in range(len(cutouts))]), axis=-1)
            heights = np.array([cutouts[i][1] for i in range(len(cutouts))])
            
            # Normalize PSFs
            psfs = normalize_0_to_1(psfs)
            
            # Store validation data
            self.validation_cutouts = psfs
            self.validation_heights = heights
            self.validation_data_ready = True
            
            self.log_message(f"SUCCESS: Validation cutouts processed: {len(cutouts)} cutouts")
            self.log_message(f"   PSF shape: {psfs.shape}")
            self.log_message(f"   Height range: {heights.min():.1f} - {heights.max():.1f}")
            
            messagebox.showinfo("Validation Data Ready", 
                              f"Validation cutouts processed successfully!\n\n"
                              f"Cutouts: {len(cutouts)}\n"
                              f"PSF shape: {psfs.shape}\n"
                              f"Ready for prediction.")
            
        except Exception as e:
            error_msg = f"Error processing validation cutouts: {str(e)}"
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Processing Error", error_msg)

    def make_prediction(self):
        """Make prediction on validation data and show heatmap"""
        # Check if prediction model is loaded
        if not hasattr(self, 'prediction_model') or self.prediction_model is None:
            messagebox.showwarning("Warning", "Please load a prediction model first")
            return
        
        # Check if validation data is ready
        if not self.validation_data_ready or self.validation_cutouts is None:
            messagebox.showwarning("Warning", "Please process validation cutouts first using 'Save Validation Cutouts'")
            return
        
        # Validate dimensions if metadata is available
        if hasattr(self, 'model_metadata') and self.model_metadata:
            expected_distance = self.model_metadata['training_config']['distance']
            expected_z_range = self.model_metadata['training_config']['z_range']
            expected_start_z = self.model_metadata['training_config']['start_z']
            expected_end_z = self.model_metadata['training_config']['end_z']
            
            current_z_range = self.config.end_z - self.config.start_z
            
            # Check cutout size compatibility
            if self.validation_cutouts.shape[1] != expected_distance or self.validation_cutouts.shape[2] != expected_distance:
                error_msg = (f"Dimension Mismatch Error!\n\n"
                           f"Model expects cutouts of size {expected_distance}x{expected_distance}\n"
                           f"Validation data has cutouts of size {self.validation_cutouts.shape[1]}x{self.validation_cutouts.shape[2]}\n\n"
                           f"Please adjust the distance parameter in Peak Configuration to {expected_distance} "
                           f"and regenerate validation cutouts.")
                messagebox.showerror("Dimension Mismatch", error_msg)
                self.log_message(f"ERROR: Dimension mismatch: Expected {expected_distance}x{expected_distance}, got {self.validation_cutouts.shape[1]}x{self.validation_cutouts.shape[2]}")
                return
            
            # Check Z-range compatibility
            if current_z_range != expected_z_range:
                error_msg = (f"Z-Range Mismatch Error!\n\n"
                           f"Model was trained with Z-range: {expected_start_z}-{expected_end_z} ({expected_z_range} slices)\n"
                           f"Current validation data Z-range: {self.config.start_z}-{self.config.end_z} ({current_z_range} slices)\n\n"
                           f"Please adjust Start Z and End Z in Peak Configuration to match the training parameters:\n"
                           f"Start Z: {expected_start_z}\n"
                           f"End Z: {expected_end_z}")
                messagebox.showerror("Z-Range Mismatch", error_msg)
                self.log_message(f"ERROR: Z-range mismatch: Expected {expected_z_range} slices ({expected_start_z}-{expected_end_z}), got {current_z_range} slices ({self.config.start_z}-{self.config.end_z})")
                return
            
            self.log_message("SUCCESS: Dimension validation passed - model and validation data are compatible")
        
        try:
            self.log_message("PREDICTING: Making predictions on validation data...")
            
            # Make predictions using the loaded model
            y_pred_normalized = self.prediction_model.predict(self.validation_cutouts, verbose=0)
            y_pred_normalized = y_pred_normalized.flatten()
            
            # Convert true heights to nm first, then normalize to [0, 1]
            # Training stack: middle height (index 80 for 161 total) = 0 nm
            # Heights go from -2000 nm to +2000 nm in 25 nm increments
            middle_index = (self.config.end_z - self.config.start_z) / 2
            y_true_nm = (self.validation_heights - middle_index) * self.config.nm_per_height
            
            # Now normalize to [0, 1] using the same range as training
            nm_min = -2000.0  # Training range minimum
            nm_max = 2000.0   # Training range maximum
            y_true_normalized = (y_true_nm - nm_min) / (nm_max - nm_min)
            nm_range = (nm_min, nm_max)
            
            self.log_message(f"SUCCESS: Predictions completed: {len(y_pred_normalized)} predictions")
            self.log_message(f"   True height range: {y_true_normalized.min():.3f} - {y_true_normalized.max():.3f}")
            self.log_message(f"   Predicted range: {y_pred_normalized.min():.3f} - {y_pred_normalized.max():.3f}")
            
            # Calculate nm range for the heatmap
            self.log_message(f"PLOTTING: Creating prediction heatmap...")
            self.log_message(f"   Z-range: {self.config.start_z} - {self.config.end_z} ({self.config.end_z - self.config.start_z} slices)")
            self.log_message(f"   nm per height: {self.config.nm_per_height} nm")
            self.log_message(f"   Training nm range: {nm_range[0]:.1f} - {nm_range[1]:.1f} nm")
            self.log_message(f"   True heights in nm: {y_true_nm.min():.1f} - {y_true_nm.max():.1f} nm")
            
            # Create the heatmap using the evaluation function
            from storm_core.evaluation import plot_true_vs_pred_heatmap
            
            plot_true_vs_pred_heatmap(
                y_true=y_true_normalized,
                y_pred=y_pred_normalized,
                bins=100,
                figsize=(10, 8),
                log_scale=True,
                nm_range=nm_range,
                ignore_percent=0.0,
                start=self.config.start_z,
                end=self.config.end_z
            )
            
            # Update results display
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Prediction Results\n")
            self.results_text.insert(tk.END, f"==================\n\n")
            self.results_text.insert(tk.END, f"Validation file: {os.path.basename(self.pred_file_var.get())}\n")
            self.results_text.insert(tk.END, f"Model: {os.path.basename(self.pred_model_var.get())}\n\n")
            self.results_text.insert(tk.END, f"Data Statistics:\n")
            self.results_text.insert(tk.END, f"- Total predictions: {len(y_pred_normalized)}\n")
            self.results_text.insert(tk.END, f"- Z-range: {self.config.start_z} - {self.config.end_z} slices\n")
            self.results_text.insert(tk.END, f"- nm per height: {self.config.nm_per_height} nm\n")
            self.results_text.insert(tk.END, f"- Training range: {nm_range[0]:.1f} - {nm_range[1]:.1f} nm\n\n")
            self.results_text.insert(tk.END, f"True heights (nm): {y_true_nm.min():.1f} - {y_true_nm.max():.1f} nm\n")
            self.results_text.insert(tk.END, f"True heights (normalized): {y_true_normalized.min():.3f} - {y_true_normalized.max():.3f}\n")
            self.results_text.insert(tk.END, f"Predicted heights (normalized): {y_pred_normalized.min():.3f} - {y_pred_normalized.max():.3f}\n\n")
            self.results_text.insert(tk.END, f"SUCCESS: Heatmap displayed in separate window\n")
            self.results_text.insert(tk.END, f"Note: This visualization shows model performance on training-like data.\n")
            self.results_text.insert(tk.END, f"Real microscopy data may require different analysis approaches.\n")
            self.results_text.config(state=tk.DISABLED)
            
            self.log_message("SUCCESS: Prediction heatmap created successfully")
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Prediction Error", error_msg)
    
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
