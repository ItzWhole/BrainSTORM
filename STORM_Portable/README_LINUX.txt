STORM Microscopy Analysis - Linux Users Guide
==============================================

🐧 NATIVE LINUX INSTALLATION

Good news! The STORM_Analysis executable is a Linux binary, so it runs
NATIVELY on Linux systems (no WSL needed - that's only for Windows users).

QUICK START (Linux):
1. Open terminal in STORM_Portable directory
2. Run: chmod +x launch_storm.sh
3. Run: ./launch_storm.sh
4. Follow the prompts

SYSTEM REQUIREMENTS:
- Linux distribution: Ubuntu 20.04+, Debian 10+, or similar
- X11 display server (for GUI)
- Python 3.8+ with tkinter
- NVIDIA GPU + drivers (optional, for GPU acceleration)
- 8GB RAM minimum, 16GB recommended

INSTALLATION:

Option 1: Use Pre-built Executable (Easiest)
--------------------------------------------
# Install system dependencies
sudo apt update
sudo apt install python3-tk libgl1-mesa-glx

# Make launcher executable
chmod +x launch_storm.sh

# Launch application
./launch_storm.sh

Option 2: Build from Source (Most Compatible)
---------------------------------------------
If the pre-built executable doesn't work on your system (different distro,
library versions, etc.), you can rebuild it:

# Install build dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-tk

# Navigate to main directory
cd ../BrainSTORM

# Create virtual environment
python3 -m venv storm_env
source storm_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Build executable
pyinstaller storm_gui.spec

# Copy to portable folder
cp dist/STORM_Analysis ../STORM_Portable/

# Launch
cd ../STORM_Portable
./launch_storm.sh

GPU ACCELERATION (Optional but Recommended):
-------------------------------------------
For NVIDIA GPUs:

# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# Verify GPU detection
nvidia-smi

# The application will automatically use GPU if available

REMOTE/SSH USAGE:
----------------
If running over SSH without GUI:

# Enable X11 forwarding
ssh -X user@hostname

# Or use VNC/remote desktop for better performance

TROUBLESHOOTING:

"No display detected":
- Ensure X11 is running: echo $DISPLAY
- If using SSH: ssh -X user@host
- Or use VNC/remote desktop

"Library version mismatch":
- Rebuild for your system using Option 2 above
- The executable was built on Ubuntu 20.04

"No GPU detected":
- Install NVIDIA drivers: sudo ubuntu-drivers autoinstall
- Verify with: nvidia-smi
- Application will work in CPU mode (slower)

"Permission denied":
- Make executable: chmod +x STORM_Analysis launch_storm.sh

DIFFERENCES FROM WINDOWS VERSION:
---------------------------------
✅ Better performance (native execution, no WSL overhead)
✅ Simpler setup (no WSL2 installation needed)
✅ Direct GPU access (no virtualization layer)
❌ .bat launcher files won't work (use .sh instead)
❌ Windows-specific paths don't apply

DISTRIBUTION COMPATIBILITY:
--------------------------
✅ Ubuntu 20.04+ (tested, recommended)
✅ Debian 10+ (should work)
⚠️  Fedora/RHEL (may need rebuild)
⚠️  Arch Linux (may need rebuild)
⚠️  Other distros (rebuild recommended)

For best compatibility, rebuild the executable on your target system.

FEATURES:
--------
All features from Windows version are available:
- Complete STORM microscopy workflow
- Deep learning Z-height prediction
- Time series analysis with parameter controls
- Frame visualization and preview
- GPU acceleration support
- Real-time parameter tuning

For detailed usage instructions, see DOCUMENTATION.md

Version: 4.0.0
Platform: Linux x86_64
Build: PyInstaller 6.18.0 on Ubuntu 20.04