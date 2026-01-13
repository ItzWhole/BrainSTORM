# STORM Microscopy GUI Application - Complete Specifications

## Project Overview

**Application Name**: STORM Microscopy Analysis - Height Regression  
**Purpose**: GUI application for STORM microscopy analysis using astigmatism to train neural networks for height regression  
**Environment**: WSL2 with CUDA support, Python virtual environment (storm_env)  
**Model Type**: Single-stack trained model (one TIFF file per training session)  

## System Architecture

### Core Components
- **Data Processing**: TIFF stack loading, peak detection, PSF cutout extraction
- **Neural Network**: Astigmatic PSF network for height regression
- **GUI Framework**: Tkinter with multi-tab interface
- **Training System**: Multi-stage training with configurable parameters
- **GPU Support**: CUDA acceleration with automatic path detection/fixing

### File Structure
```
BrainSTORM/
├── storm_gui.py                 # Main GUI application
├── launch_gui.py               # GUI launcher with environment checks
├── storm_core/                 # Core processing modules
│   ├── data_processing.py      # TIFF processing, peak detection
│   ├── neural_network.py       # Model architecture, training utilities
│   └── evaluation.py           # Model evaluation and visualization
├── requirements.txt            # Python dependencies
├── setup_wsl.sh               # WSL environment setup script
└── setup_gui.sh               # GUI setup script
```

## GUI Application Specifications

### Tab Structure

#### 1. Data Selection Tab
**Purpose**: Browse and select TIFF files for training

**Features**:
- File browser with Windows path integration (WSL ↔ Windows)
- Quick access buttons: C: Drive, Desktop, Documents, Downloads
- Single-file selection mode (radio button behavior)
- File tree with columns: Index, Filename (large), Size, Selected
- Double-click to select files
- Preview selected file information

**Path Handling**:
- WindowsPathHelper class for cross-platform paths
- WSL detection: `/mnt/c` vs Windows native paths
- Automatic username detection for user-specific folders
- Fallback paths for missing directories

#### 2. Peak Configuration Tab
**Purpose**: Configure peak detection parameters and visualize results

**Parameters**:
- **Cutout size**: PSF extraction window size (default: 24)
- **Prominence Sigma**: Peak detection threshold in noise sigmas (default: 10.0)
- **Minimum distance between peaks**: Spatial separation (default: 5)
- **Radius used to reject hot pixels**: Hot pixel filtering (default: 2)
- **Start Z**: First z-slice to include (default: 0)
- **End Z**: Last z-slice to include (default: 161)
- **Sum Slices**: Number of final slices to sum (default: 30)

**Actions**:
- **Visualize Peaks**: Opens matplotlib window showing detected peaks on summed image
- **Save Cutouts**: Processes data and creates training datasets in memory
- **Use Selected File**: Syncs with file from Data Selection tab

**Data Processing Pipeline** (Save Cutouts):
1. Extract PSF cutouts using `extract_psf_cutouts()`
2. Normalize PSFs using per-image normalization (`normalize_0_to_1`)
3. Normalize heights to [0,1] range
4. Split into train/validation sets using `train_val_split_by_group`
5. Create TensorFlow datasets with augmentation
6. Store `train_ds` and `val_ds` in memory

#### 3. Training Tab
**Purpose**: Multi-stage neural network training with configurable blocks

**Alert System**:
- Red warning: "⚠️ NO CUTOUTS SAVED!" when no preprocessed data
- Automatically hidden when cutouts are processed

**Training Block System**:
- **Default Block (Block 1)**:
  - Epochs: 100
  - Adam LR: 1e-3
  - Huber Delta: 0.06
  - LR Patience: 25
  - Early Stopping: Disabled (optional enable)

- **Additional Blocks** (➕ Add Training Block):
  - Adam LR: 2e-4 (fine-tuning default)
  - Huber Delta: 0.02 (fine-tuning default)
  - LR Patience: 20
  - Early Stopping: Enabled (40 patience)
  - ❌ Remove Block button

**Training Execution**:
- Sequential multi-stage training
- Model reloading between blocks for fine-tuning
- Real-time Keras-style logging: "Epoch X/Y - loss: X.XXXX - val_loss: X.XXXX"
- Progress tracking across all blocks
- Graceful stop functionality (like Spyder red square)

**GPU Management**:
- Automatic GPU detection and configuration
- CUDA path auto-fixing for common issues
- "Check GPU" button for diagnostics
- "Fix CUDA Path" button for manual repair
- CPU fallback with clear warnings

#### 4. Prediction Tab
**Purpose**: Model inference and evaluation (basic implementation)

**Features**:
- File selection for prediction
- Model loading
- Results display (placeholder for future expansion)

### Shared Components

#### Log System
- **Dual log windows**: Identical logs in Peak Configuration and Training tabs
- **Real-time updates**: All application activity logged
- **Synchronized**: Same content in both windows
- **Content**: File scanning, peak detection, training progress, errors

#### File Dialog Improvements
- **Freeze prevention**: `update_idletasks()` before opening dialogs
- **Parent window association**: Prevents dialogs from getting lost
- **Current directory awareness**: Opens in currently selected directory
- **Error handling**: Graceful fallback with user instructions

## Technical Specifications

### Neural Network Architecture
- **Model**: Astigmatic PSF network for height regression
- **Input Shape**: (cutout_size+1, cutout_size+1, 1)
- **Loss Function**: Huber loss with configurable delta
- **Optimizer**: Adam with configurable learning rate
- **Metrics**: Mean Absolute Error (MAE)

### Training Configuration
- **Callbacks**:
  - ModelCheckpoint: Save best model based on val_mae
  - ReduceLROnPlateau: Reduce LR when val_mae plateaus
  - EarlyStopping: Optional, configurable patience
  - Custom ProgressCallback: Real-time logging and progress updates

### Data Processing Pipeline
```python
# Peak Detection
csum_image = crop_and_sum_stack(stack, start_z, end_z, csum_slices)
cutouts, group_ids, peaks = extract_psf_cutouts(
    stack, csum_image, distance,
    min_distance=min_distance,
    prominence_sigma=prominence_sigma,
    support_radius=support_radius,
    start=start_z, end=end_z
)

# Normalization
psfs = normalize_0_to_1(psfs)  # Per-image normalization
heights = (heights - min) / (max - min)  # [0,1] normalization

# Train/Val Split
(X_train, y_train), (X_val, y_val) = train_val_split_by_group(
    psfs, heights, group_ids, val_size=0.2
)

# Dataset Creation
train_ds = make_dataset(X_train, y_train, training=True, augmenter=augmenter)
val_ds = make_dataset(X_val, y_val, training=False)
```

### GPU Support
- **Automatic Configuration**: Memory growth enabled, multi-GPU support
- **Path Detection**: Searches common CUDA library locations
- **Auto-fixing**: Updates LD_LIBRARY_PATH with found CUDA libraries
- **Diagnostics**: nvidia-smi, nvcc, library path verification
- **Fallback**: Graceful CPU operation with performance warnings

## Workflow Specifications

### Recommended User Workflow
1. **Data Selection**: Select single TIFF file
2. **Peak Configuration**: 
   - Adjust parameters
   - Click "Visualize Peaks" to verify detection
   - Click "Save Cutouts" to process data
3. **Training**:
   - Configure training blocks (default + optional fine-tuning)
   - Click "Start Training"
   - Monitor real-time progress in Log
4. **Model Management**: Save trained model

### Multi-Stage Training Example
```
Block 1: 100 epochs, LR=1e-3, Huber=0.06    # Initial training
Block 2: 300 epochs, LR=2e-4, Huber=0.02    # Fine-tuning
Block 3: 300 epochs, LR=1e-4, Huber=0.01    # Final fine-tuning
```

## Environment Requirements

### WSL2 Setup
- Ubuntu 20.04 LTS
- CUDA 11.8 toolkit
- NVIDIA drivers with WSL GPU support
- Python 3.8+ with virtual environment

### Python Dependencies
- TensorFlow 2.13.1 (GPU support)
- NumPy, SciPy, scikit-image
- matplotlib, tifffile
- tkinter (GUI framework)

### Hardware Requirements
- NVIDIA GPU (RTX series recommended)
- 8GB+ RAM
- Windows 11 with WSL 2.0+

## Error Handling & User Experience

### Robust Error Handling
- GPU detection failures with detailed diagnostics
- File access permission errors with alternatives
- Training interruption with model checkpoint recovery
- Path resolution across WSL/Windows boundary

### User-Friendly Features
- Clear progress indicators and status messages
- Comprehensive logging for troubleshooting
- Graceful degradation (CPU fallback)
- Intuitive single-file selection model
- Real-time training feedback

## Future Expansion Points

### Prediction Tab Enhancement
- Complete prediction pipeline implementation
- Visualization of height maps
- Model comparison tools
- Batch prediction capabilities

### Advanced Training Features
- Hyperparameter optimization
- Cross-validation support
- Model ensemble training
- Advanced augmentation options

### Data Management
- Training history tracking
- Model versioning
- Dataset management
- Export/import configurations

---

**Last Updated**: January 12, 2026  
**Version**: 1.0  
**Status**: Fully Implemented and Tested