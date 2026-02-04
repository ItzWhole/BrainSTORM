# Building the STORM_Analysis Executable

The `STORM_Analysis` executable is not included in the git repository due to GitHub's 100MB file size limit (the executable is ~535MB).

## Option 1: Build Locally

### Prerequisites:
- WSL2 with Ubuntu 20.04
- Python virtual environment with all dependencies installed

### Build Steps:
```bash
# Navigate to BrainSTORM directory
cd BrainSTORM

# Activate virtual environment
source storm_env/bin/activate

# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
pyinstaller storm_gui.spec

# Copy to portable folder
cp dist/STORM_Analysis STORM_Portable/

# Set executable permissions
chmod +x STORM_Portable/STORM_Analysis
```

## Option 2: Download from Releases

Check the [Releases page](https://github.com/ItzWhole/BrainSTORM/releases) for pre-built executables when available.

## Verification

After obtaining the executable, verify it works:
```bash
cd STORM_Portable
./STORM_Launcher.bat
```

The launcher should detect the executable and launch the application successfully.

## File Size Information

- **STORM_Analysis executable**: ~535MB (561,692,960 bytes)
- **Includes**: Complete Python environment, TensorFlow, all scientific libraries
- **Platform**: Linux x86_64 (runs in WSL2 on Windows)