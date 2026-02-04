STORM Microscopy Analysis - Portable Version v3
================================================

QUICK START:
1. Double-click "launch_storm.bat" to start the application
2. The launcher will check system requirements automatically
3. If everything is OK, the STORM GUI will open

SYSTEM REQUIREMENTS:
- Windows 10/11 with WSL2 enabled
- Ubuntu 20.04 installed in WSL2
- NVIDIA GPU with latest drivers (recommended for GPU acceleration)
- At least 8GB RAM (16GB recommended)

INSTALLATION REQUIREMENTS:
This portable version requires WSL2 (Windows Subsystem for Linux) to be installed:

1. Open PowerShell as Administrator and run:
   wsl --install -d Ubuntu-20.04

2. Restart your computer when prompted

3. Install NVIDIA drivers from: https://www.nvidia.com/drivers/
   (Required for GPU acceleration)

4. Run launch_storm.bat

FEATURES:
- Complete STORM microscopy analysis workflow
- Deep learning-based Z-height prediction
- Time series analysis with configurable parameters
- Frame-by-frame visualization
- Real-time parameter tuning
- GPU acceleration support

TROUBLESHOOTING:
- If "WSL not found": Install WSL2 following the instructions above
- If "No GPU detected": Install NVIDIA drivers and restart
- If application doesn't start: Check that Ubuntu 20.04 is installed in WSL2
- For other issues: Check the log files created during analysis

USAGE:
The application provides 5 main tabs:
1. Data Selection - Choose your TIFF files
2. Peak Configuration - Set up detection parameters
3. Training - Train neural network models
4. Model Validation - Test your trained models
5. Time Series Analysis - Analyze multi-frame data with real-time preview

For detailed documentation, visit the GitHub repository or check the built-in help.

SUPPORT:
For issues and questions, please refer to the documentation in the GitHub repository.

Version: Time Series Analysis v3
Build Date: February 2026