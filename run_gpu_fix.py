#!/usr/bin/env python3
"""
GPU Fix Runner - Attempts to fix GPU detection issues
"""

import subprocess
import sys
import os

def run_command(cmd, shell=True):
    """Run a command and return success, stdout, stderr"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_and_install_cudnn():
    """Check and install cuDNN if needed"""
    print("🔍 Checking cuDNN installation...")
    
    # Check if cuDNN is already installed
    success, output, error = run_command("dpkg -l | grep cudnn")
    if success and "libcudnn8" in output:
        print("✅ cuDNN already installed")
        return True
    
    print("📦 Installing cuDNN...")
    
    # Update package list
    success, output, error = run_command("sudo apt update")
    if not success:
        print(f"❌ Failed to update packages: {error}")
        return False
    
    # Install cuDNN
    success, output, error = run_command("sudo apt install -y libcudnn8 libcudnn8-dev")
    if success:
        print("✅ cuDNN installed successfully")
        return True
    else:
        print(f"❌ Failed to install cuDNN: {error}")
        return False

def reinstall_tensorflow():
    """Reinstall TensorFlow with CUDA support"""
    print("🔧 Reinstalling TensorFlow with CUDA support...")
    
    # Uninstall current TensorFlow
    success, output, error = run_command("pip uninstall -y tensorflow")
    if not success:
        print(f"⚠️  Warning: Could not uninstall TensorFlow: {error}")
    
    # Install TensorFlow with CUDA
    success, output, error = run_command("pip install tensorflow[and-cuda]==2.13.1")
    if success:
        print("✅ TensorFlow reinstalled successfully")
        return True
    else:
        print(f"❌ Failed to install TensorFlow: {error}")
        return False

def test_gpu_detection():
    """Test if GPU is now detected"""
    print("🧪 Testing GPU detection...")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✅ SUCCESS: {len(gpu_devices)} GPU(s) detected!")
            for i, gpu in enumerate(gpu_devices):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ No GPU devices detected")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
        return False

def main():
    print("STORM Microscopy GPU Detection Fix")
    print("=" * 40)
    
    # Set environment variables
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_paths = '/usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64'
    if cuda_paths not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_paths}:{current_ld_path}"
    
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print()
    
    # Step 1: Install cuDNN
    if not check_and_install_cudnn():
        print("❌ cuDNN installation failed. Manual intervention required.")
        return False
    
    # Step 2: Reinstall TensorFlow
    if not reinstall_tensorflow():
        print("❌ TensorFlow installation failed. Manual intervention required.")
        return False
    
    # Step 3: Test GPU detection
    if test_gpu_detection():
        print("\n🎉 GPU detection fix successful!")
        print("The STORM GUI should now detect your RTX 3060 GPU.")
        return True
    else:
        print("\n❌ GPU still not detected after fixes.")
        print("Additional steps may be required:")
        print("1. Restart WSL: wsl --shutdown (then reopen)")
        print("2. Update NVIDIA drivers")
        print("3. Verify Windows 11 with WSL 2.0+")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)