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

# Install CUDA (for TensorFlow GPU support)
echo "Installing CUDA toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# Install cuDNN
echo "Installing cuDNN..."
sudo apt install -y libcudnn8 libcudnn8-dev

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv storm_env
source storm_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]

# Install other requirements
echo "Installing application requirements..."
pip install -r requirements.txt

# Verify GPU setup
echo "Verifying GPU setup..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
print('CUDA built with TensorFlow:', tf.test.is_built_with_cuda())
"

echo "=== Setup Complete ==="
echo "To activate the environment: source storm_env/bin/activate"
echo "To run the application: python storm_microscopy_app.py --help"