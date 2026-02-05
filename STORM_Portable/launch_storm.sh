#!/bin/bash
# STORM Analysis Launcher for Native Linux
# This script is for Linux users (not Windows/WSL)

echo "========================================"
echo "   STORM Microscopy Analysis v4.0.0"
echo "   Native Linux Launcher"
echo "========================================"
echo ""

# Check if running on native Linux (not WSL)
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "⚠️  WARNING: You appear to be running in WSL"
    echo "   For Windows users, please use STORM_Launcher.bat instead"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Checking system requirements..."

# Check for X11 display
if [ -z "$DISPLAY" ]; then
    echo "❌ ERROR: No display detected (DISPLAY variable not set)"
    echo "   This application requires a GUI environment"
    echo "   If using SSH, try: ssh -X user@host"
    exit 1
fi
echo "✅ Display detected: $DISPLAY"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found"
    echo "   Install with: sudo apt install python3 python3-tk"
    exit 1
fi
echo "✅ Python 3 found"

# Check for NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
    else
        echo "⚠️  WARNING: nvidia-smi found but GPU not accessible"
        echo "   Application will run in CPU mode"
    fi
else
    echo "⚠️  WARNING: No NVIDIA GPU detected"
    echo "   Application will run in CPU mode (slower)"
fi

echo ""
echo "Starting STORM Analysis..."
echo "Please wait, this may take 1-2 minutes on first launch..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Make executable if not already
chmod +x STORM_Analysis 2>/dev/null

# Launch the application
./STORM_Analysis

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application failed to start"
    echo ""
    echo "Common issues:"
    echo "1. Missing dependencies - try: sudo apt install python3-tk libgl1-mesa-glx"
    echo "2. Library version mismatch - may need to rebuild for your system"
    echo "3. No GPU support - install NVIDIA drivers for better performance"
    echo ""
    echo "To rebuild for your system:"
    echo "  cd ../BrainSTORM"
    echo "  source storm_env/bin/activate"
    echo "  pyinstaller storm_gui.spec"
    echo ""
    read -p "Press Enter to exit..."
fi