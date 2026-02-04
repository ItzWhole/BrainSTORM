# STORM Microscopy Analysis GUI

A deep learning application for axial (Z-height) localization of point emitters in astigmatic STORM (Stochastic Optical Reconstruction Microscopy) using convolutional neural networks.

## What This Application Does

This application trains a specialized convolutional neural network (CNN) to predict the axial position (Z-height) of fluorescent point emitters from their astigmatic Point Spread Functions (PSFs). 

**Key capabilities:**
- **Extended axial range**: Operates across ±2000 nm (vs ±700 nm for traditional Gaussian fitting methods)
- **Consistent accuracy**: Maintains uniform error across the entire Z-range, unlike traditional methods that degrade at extremes
- **Astigmatism-based**: Uses optical astigmatism (cylindrical lens) to encode Z-information in PSF shape
- **Single-stack training**: Trains on one calibration TIFF stack with known Z-positions
- **Fluorophore-specific models**: Each model is optimized for specific fluorophores and imaging conditions

**How it works:**
1. Acquire a Z-stack of fluorescent emitters at known heights (e.g., -2000 to +2000 nm in 25 nm steps)
2. Detect PSF peaks and extract square cutouts (typically 25×25 pixels)
3. Train a CNN with multi-directional convolutional kernels to capture asymmetric PSF deformations
4. Use the trained model to predict Z-heights of unknown emitters from their PSF shapes

**Important limitations:**
- Models are highly specific to the fluorophore type, laser power, and optical setup used during training
- Different experimental conditions require retraining new models
- Multiple stacks cannot be combined unless the sample has uniform Z-height across all measurements

## Scientific Background

### Astigmatic STORM Microscopy

STORM achieves nanometer-scale resolution by sequentially activating and localizing individual fluorophores. While lateral (X-Y) localization is straightforward, axial (Z) information requires additional encoding.

**Astigmatism method**: A cylindrical lens introduces astigmatism in the optical path, causing PSFs to elongate:
- **Above focus**: PSFs elongate along X-axis
- **At focus**: PSFs are circular
- **Below focus**: PSFs elongate along Y-axis

### Traditional Approach vs Neural Network

**Traditional method (e.g., Picasso software)**:
- Fits 2D Gaussian to each PSF
- Extracts σ_x and σ_y (standard deviations)
- Uses calibration curve to map (σ_x, σ_y) → Z-height
- **Limitations**: 
  - Only works ±700 nm where PSFs remain Gaussian-like
  - Accuracy degrades significantly at extremes
  - Typical axial error: ~220 nm

**This neural network approach**:
- Uses CNN with anisotropic kernels (3×3, 3×5, 5×3) to capture directional deformations
- Learns complex PSF patterns beyond Gaussian approximations
- **Advantages**:
  - Works across ±2000 nm range (3× larger)
  - Consistent ~117 nm error across entire range
  - Better handles non-Gaussian PSF shapes at extremes

### Model Architecture

The CNN consists of:
1. **Initial block**: 64 filters (5×5 kernel) for low-level feature extraction
2. **Multi-directional blocks**: Three parallel convolutional paths per block
   - Standard path: 3×3 kernels
   - Horizontal path: 3×5 kernels (captures X-elongation)
   - Vertical path: 5×3 kernels (captures Y-elongation)
3. **Dense layers**: Global average pooling → 256 neurons → single Z-height output

**Training strategy**:
- Stage 1: 100 epochs with Huber loss (δ=0.06, ~240 nm) for coarse learning
- Stage 2: 300 epochs with Huber loss (δ=0.02, ~80 nm) for fine-tuning
- Data augmentation: Random rotation (±3°) and translation (±2 pixels) to prevent overfitting to noise

## Features

- **User-friendly GUI** with 5 main tabs for complete workflow
- **Peak detection and visualization** with configurable parameters
- **Multi-stage neural network training** with GPU acceleration
- **Model validation** with automatic dimension checking
- **Time series analysis** with frame-by-frame peak localization and Z-height prediction
- **Real-time training progress** with Keras logging
- **Automatic CUDA/GPU setup** and diagnostics
- **Model metadata storage** for reproducible results

## Hardware Requirements

⚠️ **IMPORTANT**: This application requires an NVIDIA GPU with CUDA support. AMD GPUs are not supported.

- **Operating System**: Windows 10/11 with WSL2 (Ubuntu 20.04 recommended)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+ (RTX 20/30/40 series recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended for large datasets
- **Storage**: 5GB free space for environment and models
- **NVIDIA Drivers**: Latest drivers from NVIDIA website (required for WSL GPU support)

## Installation

### 1. Enable WSL2 and Install Ubuntu

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-20.04
# Restart computer when prompted
```

### 2. Install NVIDIA Drivers

1. Download and install the latest NVIDIA drivers from [NVIDIA website](https://www.nvidia.com/drivers/)
2. Restart your computer
3. Verify installation in WSL:
```bash
nvidia-smi
```

### 3. Clone Repository and Setup Environment

```bash
# Navigate to your preferred directory (e.g., Windows C: drive)
cd /mnt/c/Users/YourUsername/Documents

# Clone the repository
git clone <your-repo-url>
cd BrainSTORM

# Run the automated setup script
chmod +x setup_wsl.sh
./setup_wsl.sh
```

### 4. Manual Installation (Alternative)

If you prefer manual installation or the setup script fails:

```bash
# Create virtual environment
python3 -m venv storm_env
source storm_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with CUDA support (pip-based method)
pip install tensorflow[and-cuda]==2.13.1

# Install NVIDIA CUDA libraries (more reliable than system packages)
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11

# Install application dependencies
pip install -r requirements.txt

# Configure CUDA library paths
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"' >> storm_env/bin/activate

# Reactivate environment to apply paths
deactivate
source storm_env/bin/activate
```

### 5. Verify Installation

```bash
# Test GPU detection
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print('SUCCESS: GPU detected!')
else:
    print('ERROR: No GPU detected - check NVIDIA drivers')
print('CUDA built with TensorFlow:', tf.test.is_built_with_cuda())
"

# Test STORM modules
python -c "
from storm_core import data_processing, neural_network, evaluation
print('SUCCESS: STORM core modules imported successfully')
"
```

**Note**: All dependencies (matplotlib, keras, numpy, etc.) are automatically installed via `requirements.txt`. The pip-based CUDA installation method is more reliable than system CUDA packages and handles version compatibility automatically.

## Usage

### Launch the Application

```bash
# Activate environment
source storm_env/bin/activate

# Launch GUI
python ../bin/storm_gui.py
```

### Workflow

#### 1. Data Selection Tab
- Select **Peak Configuration File**: TIFF stack for training and peak detection
- Select **Model Validation File**: TIFF stack for model testing
- Both files should have the same Z-range and parameters

#### 2. Peak Configuration Tab
- Configure detection parameters:
  - **Cutout size**: Size of PSF regions (default: 25)
  - **Prominence Sigma**: Peak detection sensitivity (default: 10.0)
  - **Start/End Z**: Z-stack range (default: 0-161)
  - **Distance in nm between heights**: Spacing between z-levels (default: 25.0 nm)
- Click **Visualize Peaks** to preview detection
- Click **Save Cutouts** to process training data

#### 3. Training Tab
- Configure training blocks:
  - **Block 1**: Initial training (default: 100 epochs)
  - **Additional blocks**: Fine-tuning with different parameters
- Click **Start Training** to begin
- Monitor progress in real-time
- Save trained model when complete

#### 4. Model Validation Tab
- Load trained model using **Browse** and **Load Model**
- Process validation data with **Save Validation Cutouts**
- Generate validation heatmap with **Generate Validation Heatmap**
- View true vs predicted height comparison

#### 5. Time Series Analysis Tab
- **Select Time Series**: Choose a multi-frame TIFF file for analysis
- **Select Model**: Load a trained .h5 model for Z-height prediction
- **Output CSV Location**: Specify where to save analysis results
- **Processing Parameters**: Configure detection and filtering parameters
  - **Sigma Small** (default: 1.2): PSF-scale smoothing (lower = sharper features)
  - **Sigma Large** (default: 3.0): Background smoothing (higher = more background removal)
  - **Threshold Factor** (default: 0.1): Peak detection sensitivity (lower = more sensitive)
  - **Min Distance** (default: 4): Minimum pixels between detected peaks
- **Frame Visualization**: Preview peak detection on individual frames
  - **Frame Number**: Select which frame to visualize (1-based indexing)
  - **Visualize Frame**: Display filtered frame with detected peaks overlaid
- **Analysis Options**:
  - **Analyze**: Process the complete time series with current parameters
  - **Analyze First 10 Frames**: Test mode for quick validation
  - **Stop Analysis**: Gracefully halt processing if needed
- **Real-time Analysis Log**: Monitor progress and view detailed processing information

**Time Series Analysis Workflow**:
1. **Parameter configuration**: Set bandpass filter and peak detection parameters
2. **Frame preview** (optional): Visualize detection results on sample frames
3. **Frame-by-frame processing**: Each frame is processed independently using configured parameters
4. **Bandpass filtering**: Applied only for peak detection using Sigma Small/Large values
5. **Peak detection**: Finds local maxima using Threshold Factor and Min Distance
6. **ROI extraction**: Extracts regions around detected peaks
   - Filtered ROIs used for sub-pixel localization
   - Raw ROIs fed to CNN for Z-height prediction
7. **Sub-pixel localization**: Uses iterative weighted centroid method on filtered data
8. **Z-height prediction**: CNN processes raw ROIs to predict axial position
9. **CSV output**: Results saved with columns: peak_id, frame, x, y, z

**Key Features**:
- **User-configurable parameters**: Full control over detection sensitivity and filtering
- **Real-time parameter preview**: Visualize detection results before running full analysis
- **Dynamic model detection**: Automatically detects expected cutout size from model input dimensions
- **Dual ROI processing**: Uses filtered data for localization, raw data for CNN prediction
- **Sub-pixel precision**: Achieves sub-pixel accuracy in X-Y localization
- **Test mode**: Process only first 10 frames for quick validation
- **Stop functionality**: Gracefully halt analysis and save partial results
- **Comprehensive logging**: Timestamped log files with parameter values for troubleshooting
- **Progress tracking**: Real-time progress bar and detailed status updates

### Tips for Best Results

1. **Data Quality**: Use high-quality Z-stacks with clear, isolated PSF patterns
2. **Parameter Optimization**: Use frame visualization to optimize detection parameters:
   - Start with default values (Sigma Small: 1.2, Sigma Large: 3.0, Threshold Factor: 0.1, Min Distance: 4)
   - Adjust **Threshold Factor** first: lower values detect more peaks, higher values are more selective
   - Tune **Sigma Small/Large** for optimal PSF enhancement: smaller sigma_small preserves finer details
   - Set **Min Distance** based on expected peak density: increase for crowded fields
3. **Frame Preview**: Always test parameters on representative frames before full analysis
4. **Cutout Size**: 25×25 pixels works best for typical single-molecule concentrations
5. **Fluorophore Specificity**: Train separate models for each fluorophore type and laser power
6. **Calibration Stack**: Ensure your training stack covers the full Z-range you want to measure
7. **Validation**: Always validate on independent data acquired under identical conditions
8. **Model Reuse**: Only use trained models on data acquired with the same:
   - Fluorophore type (e.g., ATTO647N vs beads will give different results)
   - Laser power (±2 mW tolerance recommended)
   - Optical setup (same cylindrical lens orientation and position)

### Expected Performance

Based on validation with ATTO647N single molecules:
- **Axial range**: -2000 to +2000 nm
- **Mean absolute error**: ~117 nm (ignoring 5% outliers: ~92 nm)
- **Comparison to Picasso**: ~47% error reduction across full range
- **Error distribution**: Nearly constant across Z-range (slight improvement below Z=0)

## Time Series Analysis

The Time Series Analysis tab provides comprehensive frame-by-frame processing of multi-frame TIFF files for single-molecule localization with Z-height prediction.

### Algorithm Overview

The time series analysis implements a sophisticated pipeline that combines traditional peak detection with deep learning-based height regression:

1. **Bandpass Filtering**: Uses Difference of Gaussians (DoG) filtering to enhance PSF contrast while suppressing background noise
2. **Peak Detection**: Employs adaptive thresholding based on robust maximum estimation to find local maxima
3. **ROI Extraction**: Extracts square regions around detected peaks with automatic size matching to model requirements
4. **Quality Filtering**: Removes ROIs without single central peaks to ensure reliable localization
5. **Sub-pixel Localization**: Uses iterative weighted centroid method on filtered data for precise X-Y positioning
6. **Z-height Prediction**: Feeds raw (unfiltered) ROIs to trained CNN for axial position estimation

### Key Technical Features

**Dual Processing Pipeline**:
- **Filtered data path**: Used for peak detection and X-Y localization (better contrast and noise reduction)
- **Raw data path**: Used for CNN Z-height prediction (preserves original PSF characteristics the model was trained on)

**Dynamic Model Adaptation**:
- Automatically detects expected cutout size from loaded model input dimensions
- Ensures compatibility between analysis parameters and trained model requirements
- Supports models trained with different ROI sizes (e.g., 25×25, 15×15, etc.)

**Robust Peak Detection**:
- Adaptive thresholding based on image statistics (10% of robust maximum)
- Configurable minimum distance between peaks to prevent over-detection
- Quality filtering to ensure single-peak ROIs for reliable localization

**Sub-pixel Localization Methods**:
The `detectionalgo.py` module implements multiple localization algorithms:
- **Iterative Weighted Centroid** (recommended): Iterative refinement with Gaussian weighting
- **Radial Symmetry**: Based on gradient analysis for symmetric PSFs
- **Image Moments**: Classical moment-based centroid calculation
- **Spline Interpolation**: High-resolution peak finding via interpolation
- **Adaptive Method**: Combines multiple methods for robust results

### Output Format

Results are saved as CSV files with the following columns:
- **peak_id**: Unique identifier for each detected peak
- **frame**: Frame number (1-based indexing)
- **x**: Sub-pixel X coordinate in the original image
- **y**: Sub-pixel Y coordinate in the original image  
- **z**: Predicted Z-height in nanometers (typically -2000 to +2000 nm range)

### Performance Characteristics

**Processing Speed**: Depends on frame size, peak density, and model complexity
- Typical processing: 1-10 frames per second on modern GPUs
- Bottlenecks: Peak detection (CPU) and CNN inference (GPU)

**Accuracy**: Inherits accuracy characteristics from the trained model
- X-Y localization: Sub-pixel precision (typically <0.1 pixel standard deviation)
- Z-height prediction: Model-dependent (typically ~100-200 nm for well-trained models)

**Memory Usage**: Scales with frame size and number of detected peaks
- Processes frames sequentially to minimize memory footprint
- Automatic garbage collection between frames

## Troubleshooting

### GPU Not Detected
1. **Check NVIDIA drivers**: `nvidia-smi` (should show GPU info)
2. **Verify WSL GPU support**: Ensure you have WSL2 with GPU passthrough enabled
3. **Use built-in diagnostics**: Click **Check GPU** button in Training tab
4. **Auto-fix CUDA paths**: Click **Fix CUDA Path** button to auto-configure library paths
5. **Manual CUDA path fix**: If automatic fix fails, run:
   ```bash
   export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
   ```

### Hardware Compatibility Issues
- **AMD GPUs**: Not supported - NVIDIA GPU required for CUDA acceleration
- **Older NVIDIA GPUs**: Must support CUDA Compute Capability 3.5+
- **Integrated Graphics**: Intel/AMD integrated graphics cannot run CUDA workloads

### Installation Issues
```bash
# If pip-based CUDA installation fails, try:
pip install --upgrade pip
pip install --force-reinstall tensorflow[and-cuda]==2.13.1
pip install --force-reinstall nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11

# If requirements.txt installation fails:
pip install -r requirements.txt --no-cache-dir
```

### Import Errors
```bash
# Ensure you're in the correct environment
source storm_env/bin/activate

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Memory Issues
- Reduce batch size in training configuration
- Close other applications
- Use smaller cutout sizes if necessary

### Path Issues
- Ensure TIFF files are accessible from WSL
- Use `/mnt/c/` prefix for Windows paths
- Check file permissions

### Time Series Analysis Issues
- **No peaks detected**: 
  - Use frame visualization to check detection on individual frames
  - Lower the **Threshold Factor** (try 0.05-0.08) for more sensitive detection
  - Adjust **Sigma Small/Large** values: try sigma_small=1.0, sigma_large=4.0
  - Check image quality and contrast
- **Too many false peaks**: 
  - Increase **Threshold Factor** (try 0.15-0.2) for more selective detection
  - Increase **Min Distance** to prevent over-detection in crowded areas
  - Adjust **Sigma Large** to remove more background noise
- **Parameter optimization**: Use frame visualization extensively to find optimal settings
- **Model dimension mismatch**: Ensure time series data matches training data characteristics
- **Memory errors**: Process smaller frame ranges or reduce image size
- **Slow processing**: Check GPU utilization and consider reducing peak detection sensitivity
- **Path errors**: Verify TIFF file paths are correctly formatted for WSL (use `/mnt/c/` prefix)
- **Analysis stops unexpectedly**: Check log files in `logs/` directory for detailed error messages
- **Visualization not working**: Ensure matplotlib backend is properly configured for display

## File Structure

```
BrainSTORM/
├── bin/
│   ├── storm_gui.py          # Main GUI application
│   └── detectionalgo.py      # Peak detection and localization algorithms
├── storm_core/
│   ├── data_processing.py    # TIFF processing, peak detection, PSF extraction
│   ├── neural_network.py     # CNN architecture and training routines
│   └── evaluation.py         # Model evaluation and visualization
├── brainstorm_original.py    # Original research code (for reference)
├── storm_env/                # Python virtual environment
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Model Output

Trained models (.h5 format) include metadata:
- **Training parameters**: Cutout size, Z-range, step size (nm between heights)
- **Stack dimensions**: Number of Z-slices used for training
- **Model architecture**: Layer configuration for reproducibility
- **Normalization scheme**: Z-heights normalized to [0,1] using fixed -2000 to +2000 nm range

## Limitations and Considerations

1. **Fluorophore specificity**: Models trained on one fluorophore type (e.g., ATTO647N) perform poorly on others (e.g., dark red beads). Train separate models for each fluorophore.

2. **Laser power sensitivity**: Changing laser power by >2 mW significantly degrades performance. Models are optimized for the specific photon count distribution of the training data.

3. **Single-stack training**: Combining multiple Z-stacks is problematic unless the sample has perfectly uniform height, as unknown Z-offsets between stacks corrupt training labels.

4. **Concentration trade-off**: 
   - Lower concentration → better isolated PSFs → larger cutouts possible → potentially better accuracy
   - Higher concentration → more training data per stack → faster acquisition → smaller cutouts required

5. **Optical setup dependency**: Any change to the cylindrical lens position, orientation, or objective requires retraining.

