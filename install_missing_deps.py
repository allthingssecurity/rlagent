#!/usr/bin/env python3
"""
Quick script to install missing dependencies for ART framework.
Run this if you get import errors.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Installing missing dependencies for ART framework...")
    
    # List of packages that are often missing
    missing_packages = [
        "setproctitle>=1.3.0",
        "multiprocess>=0.70.0", 
        "polars>=0.20.0",
        "weave>=0.50.0",
        "litellm>=1.0.0",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers",
        "flash-attn --no-build-isolation",
    ]
    
    for package in missing_packages:
        print(f"📦 Installing {package.split('>=')[0].split('@')[0]}...")
        try:
            if "--no-build-isolation" in package:
                # Special handling for flash-attn
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package.replace(" --no-build-isolation", ""),
                    "--no-build-isolation"
                ])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ Installed successfully")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Failed to install {package} (might not be critical)")
            continue
    
    print("\n🧪 Testing imports...")
    
    # Test critical imports
    test_imports = [
        ("setproctitle", "setproctitle"),
        ("multiprocess", "multiprocess"),
        ("art", "art"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ]
    
    all_good = True
    for import_name, package_name in test_imports:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - still missing")
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run: python train_simple.py")
    else:
        print("\n⚠️ Some dependencies are still missing.")
        print("Try installing them manually or check the error messages above.")

if __name__ == "__main__":
    main()