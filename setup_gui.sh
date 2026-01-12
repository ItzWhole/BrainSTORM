#!/bin/bash
# GUI Setup Script for STORM Microscopy Application

set -e

echo "=== STORM Microscopy GUI Setup ==="

# Install tkinter for GUI
echo "Installing tkinter..."
sudo apt update
sudo apt install -y python3-tk

# Install X11 forwarding support (for WSL)
echo "Installing X11 support..."
sudo apt install -y x11-apps

# Activate virtual environment and install GUI dependencies
echo "Installing GUI-specific Python packages..."
source storm_env/bin/activate
pip install matplotlib

# Test GUI availability
echo "Testing GUI setup..."
python3 -c "
import tkinter as tk
print('✓ tkinter working')

import matplotlib
matplotlib.use('TkAgg')  # Use tkinter backend
print('✓ matplotlib with tkinter backend working')

print('GUI setup complete!')
"

echo ""
echo "=== GUI Setup Complete ==="
echo ""
echo "To run the GUI application:"
echo "1. Activate environment: source storm_env/bin/activate"
echo "2. Launch GUI: python launch_gui.py"
echo ""
echo "Note for WSL users:"
echo "- Install an X server like VcXsrv or use WSLg"
echo "- Set DISPLAY variable if needed: export DISPLAY=:0"