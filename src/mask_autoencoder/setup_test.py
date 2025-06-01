#!/usr/bin/env python3
"""
Setup script for v4 memory testing
Installs required dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install required packages for memory testing"""
    print("🔧 Setting up dependencies for v4 memory testing...")
    
    required_packages = [
        "psutil",           # Memory monitoring
        "torch",            # PyTorch (if not already installed)
        "numpy",            # Numerical operations
        "pandas",           # Data handling
        "scikit-learn",     # Metrics
        "matplotlib",       # Plotting (in v4.py)
        "tqdm"             # Progress bars
    ]
    
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        print("Please install them manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("\n✅ All dependencies installed successfully!")
        print("You can now run: python test_v4_memory.py")

if __name__ == '__main__':
    main() 