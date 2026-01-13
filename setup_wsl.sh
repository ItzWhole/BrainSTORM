#!/bin/bash
# WSL Setup Script for STORM Microscopy Application

set -e

echo "=== STORM Microscopy WSL Setup ==="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and essential tools
echo "Installing Python and development tools..."
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv storm_env
source storm_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with CUDA support (pip-based approach)
echo "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]==2.13.1

# Install NVIDIA CUDA libraries via pip (more reliable)
echo "Installing NVIDIA CUDA libraries..."
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11

# Install other requirements
echo "Installing application requirements..."
pip install -r requirements.txt

# Configure CUDA library paths
echo "Configuring CUDA library paths..."
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"' >> storm_env/bin/activate

# Reactivate environment to apply CUDA paths
deactivate
source storm_env/bin/activate

# Verify GPU setup
echo "Verifying GPU setup..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpu_devices)
if gpu_devices:
    print('SUCCESS: GPU detected!')
else:
    print('WARNING: No GPU detected - check NVIDIA drivers')
print('CUDA built with TensorFlow:', tf.test.is_built_with_cuda())
"

echo "=== Setup Complete ==="
echo "To activate the environment: source storm_env/bin/activate"
echo "To run the application: python storm_microscopy_app.py --help"