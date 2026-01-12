#!/usr/bin/env python3
"""
STORM Microscopy GUI Launcher

Simple launcher script for the STORM microscopy GUI application.
Checks dependencies and provides helpful error messages.
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check if we're in the correct environment"""
    try:
        import tkinter
        print("✓ tkinter available")
    except ImportError:
        print("✗ tkinter not available")
        print("Please install tkinter: sudo apt-get install python3-tk")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} available")
    except ImportError:
        print("✗ TensorFlow not available")
        print("Please activate the storm_env environment:")
        print("  source storm_env/bin/activate")
        return False
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import tifffile
        import scipy
        import sklearn
        print("✓ All scientific packages available")
    except ImportError as e:
        print(f"✗ Missing scientific package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    # Check if storm_core modules are available
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from storm_core import data_processing, neural_network, evaluation
        print("✓ STORM core modules available")
    except ImportError as e:
        print(f"✗ STORM core modules not available: {e}")
        print("Please ensure you're running from the correct directory")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("STORM Microscopy GUI Launcher")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed!")
        print("\nTo fix this:")
        print("1. Activate the virtual environment: source storm_env/bin/activate")
        print("2. Install tkinter: sudo apt-get install python3-tk")
        print("3. Run from the project directory")
        sys.exit(1)
    
    print("\nEnvironment check passed!")
    print("Starting STORM Microscopy GUI...")
    
    try:
        from storm_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"\nError starting GUI: {e}")
        print("\nTroubleshooting:")
        print("- Ensure you're in WSL with X11 forwarding enabled")
        print("- Try: export DISPLAY=:0")
        print("- Or use Windows Subsystem for Linux GUI (WSLg)")
        sys.exit(1)

if __name__ == "__main__":
    main()