#!/usr/bin/env python3
"""
Super quick fix for the torchtune dependency.
Just run this to install the missing package.
"""

import subprocess
import sys

def install_torchtune():
    """Install torchtune package."""
    print("🔧 Installing missing torchtune package...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torchtune>=0.1.0"
        ])
        print("✅ torchtune installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install torchtune: {e}")
        return False

def test_torchtune():
    """Test if torchtune can be imported."""
    try:
        import torchtune
        print(f"✅ torchtune {torchtune.__version__} working!")
        return True
    except ImportError as e:
        print(f"❌ torchtune import failed: {e}")
        return False

def main():
    print("⚡ Quick Fix for torchtune dependency")
    print("=" * 40)
    
    # Install torchtune
    if install_torchtune():
        # Test the import
        if test_torchtune():
            print("\n🎉 SUCCESS! torchtune is now working!")
            print("\n🚀 You can now run:")
            print("   python train_simple.py")
        else:
            print("\n⚠️ Installation succeeded but import failed.")
            print("Try restarting your Python session.")
    else:
        print("\n❌ Installation failed.")
        print("Try manual installation: pip install torchtune")

if __name__ == "__main__":
    main()