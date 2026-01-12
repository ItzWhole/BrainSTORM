# STORM Microscopy GUI Interface

A modern, user-friendly graphical interface for STORM microscopy analysis that transforms the command-line application into an intuitive desktop application.

## 🎨 Interface Overview

The GUI provides a tabbed interface with four main sections:

### 1. **Data Selection Tab**
- **Directory Browser**: Select folders containing TIFF files
- **File Scanner**: Automatically find `*_MMStack_Default.ome.tif` files
- **Multi-Selection**: Choose specific files for training with checkboxes
- **File Preview**: View selected files and their properties
- **Batch Operations**: Select all, clear selection, preview selection

### 2. **Training Tab**
- **Parameter Controls**: Adjust epochs, batch size, distance, etc.
- **Training Controls**: Start/stop training with visual feedback
- **Progress Monitoring**: Real-time progress bar and status updates
- **Live Logging**: See training progress, loss values, and metrics
- **Model Management**: Save and load trained models
- **Threaded Execution**: Training runs in background without freezing GUI

### 3. **Prediction Tab**
- **File Selection**: Browse and select TIFF files for prediction
- **Model Loading**: Load previously trained models
- **Results Display**: View prediction results and visualizations
- **Batch Processing**: Process multiple files (planned feature)

### 4. **Configuration Tab**
- **Peak Detection**: Adjust prominence sigma, min distance, support radius
- **Z-Stack Settings**: Configure start/end Z, sum slices
- **Parameter Persistence**: Save/load configuration profiles
- **Reset Options**: Restore default settings

## 🚀 Getting Started

### Prerequisites
- WSL2 with Ubuntu 20.04
- STORM environment activated (`source storm_env/bin/activate`)
- X11 forwarding or WSLg for GUI display

### Installation

1. **Install GUI Dependencies**:
   ```bash
   chmod +x setup_gui.sh
   ./setup_gui.sh
   ```

2. **Launch the Application**:
   ```bash
   python launch_gui.py
   ```

### First-Time Setup

1. **Select Data Directory**: Use the "Browse" button to select your TIFF data folder
2. **Scan Files**: Click "Scan Files" to find available TIFF stacks
3. **Configure Parameters**: Adjust settings in the Configuration tab if needed
4. **Select Training Files**: Choose which files to use for training
5. **Start Training**: Click "Start Training" and monitor progress

## 🔧 Features

### User-Friendly Design
- **No Command Line**: Everything accessible through menus and buttons
- **Visual Feedback**: Progress bars, status messages, and real-time logs
- **Error Handling**: Clear error messages and troubleshooting hints
- **Intuitive Layout**: Logical organization with tabbed interface

### Advanced Functionality
- **Multi-Threading**: Training runs in background without blocking GUI
- **Real-Time Updates**: Live progress monitoring and logging
- **Configuration Management**: Save/load parameter sets
- **Model Persistence**: Easy model saving and loading
- **File Management**: Intelligent TIFF file discovery and selection

### Professional Features
- **Logging System**: Comprehensive logging with timestamps
- **Progress Tracking**: Detailed progress reporting during training
- **Error Recovery**: Graceful handling of errors and exceptions
- **Memory Management**: Efficient handling of large TIFF files

## 🎯 Workflow Example

### Training a New Model

1. **Data Selection**:
   - Browse to your TIFF data directory
   - Scan for available files
   - Select files for training (e.g., files 17 and 20)

2. **Configuration**:
   - Set training parameters (epochs: 100, batch size: 64)
   - Adjust peak detection settings if needed
   - Save configuration for future use

3. **Training**:
   - Click "Start Training"
   - Monitor progress in real-time
   - View training logs and metrics
   - Save model when training completes

4. **Prediction**:
   - Load your trained model
   - Select a test TIFF file
   - Run prediction and view results

## 🛠️ Technical Details

### Architecture
- **Main GUI**: `storm_gui.py` - Complete GUI application
- **Launcher**: `launch_gui.py` - Environment checker and startup
- **Setup**: `setup_gui.sh` - GUI dependencies installer

### Threading Model
- **Main Thread**: GUI interface and user interactions
- **Worker Thread**: Training and data processing
- **Queue Communication**: Safe message passing between threads

### Integration
- **Storm Core**: Uses existing `storm_core` modules
- **TensorFlow**: Full integration with training pipeline
- **Matplotlib**: Embedded plotting and visualization

## 🔍 Troubleshooting

### GUI Won't Start
```bash
# Check environment
python launch_gui.py

# Install missing dependencies
sudo apt-get install python3-tk
pip install matplotlib

# Set display (if needed)
export DISPLAY=:0
```

### Training Issues
- Ensure files are selected in Data Selection tab
- Check that virtual environment is activated
- Verify TIFF files are valid and accessible
- Monitor logs for specific error messages

### Display Problems (WSL)
- Install VcXsrv or use WSLg
- Enable X11 forwarding
- Check DISPLAY environment variable

## 📊 Performance

### Optimizations
- **Lazy Loading**: Files loaded only when needed
- **Threaded Processing**: Non-blocking training execution
- **Memory Efficient**: Proper cleanup and garbage collection
- **Progress Reporting**: Minimal overhead monitoring

### Scalability
- **Large Datasets**: Handles multiple large TIFF files
- **Long Training**: Progress tracking for extended training sessions
- **Multiple Models**: Easy switching between different models

## 🎉 Benefits

### For Researchers
- **No Programming Required**: Point-and-click interface
- **Visual Feedback**: See exactly what's happening
- **Easy Experimentation**: Quick parameter adjustments
- **Professional Results**: Same quality as command-line version

### For Workflow
- **Faster Setup**: No need to remember command syntax
- **Better Monitoring**: Real-time progress and logging
- **Easier Sharing**: GUI can be used by non-programmers
- **Reduced Errors**: Visual validation of selections

This GUI transforms your STORM microscopy application from a technical tool into an accessible, professional desktop application suitable for any researcher! 🔬✨