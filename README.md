# STORM Microscopy Height Regression

A modular application for training a neural network capable of regressing the Z-values of PSFs in STORM microscopy setups. The model is trained based on one stack and saves it automatically as an h5 file. In the very near future I intend to implement automatic time series analysis as well. It was constructed with asymmetric astigmatic PSFs in mind, but should work for any setup where the deformations above Z=0 and below Z=0 are asymmetrical.

## Features

- **Peak Detection**: Robust local maxima detection with noise filtering
- **PSF Extraction**: Automated cutout extraction around detected peaks
- **Neural Network**: Custom astigmatic PSF regression model with multi-path architecture
- **WSL Compatible**: Designed to run on WSL2 with CUDA support
- **Modular Design**: Clean, maintainable code structure

## Requirements

- WSL2 with Ubuntu
- CUDA 11.8+ and cuDNN
- Python 3.8+
- TensorFlow 2.8+ with GPU support

Note: The project is tested with TensorFlow 2.13.1 and CUDA 11.8. Other versions may work but are not guaranteed.


## Installation

### 1. Setup WSL2 (if not already done)

```bash
# In Windows PowerShell (as Administrator)
wsl --install -d Ubuntu
```

### 2. Clone and Setup

```bash
# In WSL terminal
git clone <your-repo-url>
cd BrainSTORM

# Make setup script executable and run
chmod +x setup_wsl.sh
./setup_wsl.sh
```

### 3. Activate Environment

```bash
source storm_env/bin/activate
```

## Usage

### Training a Model

```bash
python storm_microscopy_app.py \
    --data-dir /path/to/tiff/files \
    --output-dir ./results \
    --distance 24 \
    --train \
    --train-indices 20 17
```

### Making Predictions

```bash
python storm_microscopy_app.py \
    --predict /path/to/test.tif \
    --model-path ./results/model_distance_18.keras \
    --output-dir ./predictions
```

### List Available Files

```bash
python storm_microscopy_app.py --data-dir /path/to/tiff/files
```

## Configuration

Key parameters can be adjusted in the `STORMConfig` class:

- `distance`: PSF cutout size (default: 18)
- `prominence_sigma`: Peak detection threshold (default: 10.0)
- `csum_slices`: Number of slices to sum for peak detection (default: 30)
- `batch_size`: Training batch size (default: 64)
- `epochs`: Training epochs (default: 100)

## File Structure

```
├── storm_microscopy_app.py     # Main application
├── storm_core/                 # Core modules
│   ├── __init__.py
│   ├── data_processing.py      # TIFF loading, peak detection
│   ├── neural_network.py       # Model architecture, training
│   └── evaluation.py           # Visualization, analysis
├── requirements.txt            # Python dependencies
├── setup_wsl.sh               # WSL setup script
└── README.md                  # This file
```

## Model Architecture

The neural network uses a multi-path architecture optimized for astigmatic PSF analysis:

<img width="907" height="611" alt="image" src="https://github.com/user-attachments/assets/1a5ae4e6-82c2-4773-b340-94d55c7fb1d2" />


This design captures the asymmetric nature of astigmatic PSFs at different z-heights.

## Data Processing Pipeline

1. **TIFF Loading**: Load z-stacks
2. **Peak Detection**: Sum final slices, detect local maxima
3. **Cutout Extraction**: Extract PSF regions around peaks
4. **Normalization**: Per-image [0,1] normalization
5. **Group Splitting**: Prevent data leakage by grouping peaks
6. **Augmentation**: Random translation and rotation

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver in WSL
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Issues

- Reduce `batch_size` in config
- Use smaller `distance` values

## Contributing

1. Follow the established code structure
2. Add type hints and docstrings
3. Test with sample data
4. Update documentation


## Experimental Results

This section demonstrates the performance of the trained neural network on real experimental data.
The model was evaluated using single-molecule measurements acquired with the **CIBION nanoscope** using **SMATTO640N** fluorophores.

The figure below is a true vs predicted graph showing axial localization results obtained from experimental PSFs, illustrating the model’s ability to recover z-positions from asymmetric astigmatic deformations.

<img width="779" height="588" alt="image" src="https://github.com/user-attachments/assets/2715afcc-2fc9-4f03-8b1b-f6b9465157ca" />


### Comparison with Picasso

The following figure compares the performance of several trained neural network models against **Picasso** (Jungmann Lab), a widely used software for 3D STORM localization.

<img width="904" height="628" alt="image" src="https://github.com/user-attachments/assets/4638df31-c290-48b2-a9be-3e5ae640defe" />


These results indicate that the proposed approach achieves improved axial localization performance compared to analytical fitting-based methods.


Note: These results are shown for representative datasets and do not constitute a full statistical benchmark.


