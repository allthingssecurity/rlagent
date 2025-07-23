#!/usr/bin/env python3
"""
Super quick fix for the torchtune dependency.
Just run this to install the missing package.
"""

import subprocess
import sys

def install_torch_deps():
    """Install torchao and torchtune packages."""
    print("🔧 Installing missing PyTorch dependencies...")
    
    # Install torchao first (required by torchtune)
    try:
        print("   Installing torchao...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torchao>=0.1.0"
        ])
        print("   ✅ torchao installed!")
    except subprocess.CalledProcessError:
        print("   ⚠️ torchao failed, trying alternative...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "torchao", "--pre", 
                "--index-url", "https://download.pytorch.org/whl/nightly/cu118"
            ])
            print("   ✅ torchao installed from nightly!")
        except subprocess.CalledProcessError:
            print("   ❌ torchao failed with all methods")
    
    # Install torchtune
    try:
        print("   Installing torchtune...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torchtune>=0.1.0"
        ])
        print("   ✅ torchtune installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ torchtune failed: {e}")
        return False

def test_torch_deps():
    """Test if torchao and torchtune can be imported."""
    success = True
    
    try:
        import torchao
        print(f"✅ torchao working!")
    except ImportError as e:
        print(f"❌ torchao import failed: {e}")
        success = False
    
    try:
        import torchtune
        print(f"✅ torchtune working!")
    except ImportError as e:
        print(f"❌ torchtune import failed: {e}")
        success = False
    
    return success

def main():
    print("⚡ Quick Fix for PyTorch dependencies")
    print("=" * 40)
    
    # Install dependencies
    if install_torch_deps():
        # Test the imports
        if test_torch_deps():
            print("\n🎉 SUCCESS! All PyTorch dependencies working!")
            print("\n🚀 You can now run:")
            print("   python train_simple.py")
        else:
            print("\n⚠️ Installation succeeded but some imports failed.")
            print("💡 Try:")
            print("   1. Restart your Python session")
            print("   2. Run: python install_torch_deps.py")
    else:
        print("\n❌ Installation failed.")
        print("💡 Try the comprehensive installer:")
        print("   python install_torch_deps.py")

if __name__ == "__main__":
    main()