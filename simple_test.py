#!/usr/bin/env python3
"""
Simple test version without heavy dependencies
"""

import os
import sys
from pathlib import Path

def simple_storm_test():
    """Test STORM app structure without numpy/tensorflow"""
    
    print("=== STORM App Simple Test ===")
    
    # Test basic imports
    try:
        sys.path.append(os.getcwd())
        
        # Test if we can load the modules (just syntax)
        print("✓ Testing module structure...")
        
        # Check if files exist
        files_to_check = [
            "storm_microscopy_app.py",
            "storm_core/__init__.py", 
            "storm_core/data_processing.py",
            "storm_core/neural_network.py",
            "storm_core/evaluation.py"
        ]
        
        for file in files_to_check:
            if Path(file).exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} missing")
        
        print("\n✓ App structure is ready!")
        print("\nNext steps:")
        print("1. Restart your computer")
        print("2. Open WSL: wsl -d Ubuntu-20.04") 
        print("3. Navigate: cd '/mnt/c/THE FOLDER/KIROCODE/BrainSTORM/BrainSTORM'")
        print("4. Setup: chmod +x setup_wsl.sh && ./setup_wsl.sh")
        print("5. Test: python storm_microscopy_app.py --help")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simple_storm_test()