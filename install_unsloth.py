#!/usr/bin/env python3
"""
Install Unsloth - the final missing dependency for ART framework.
Unsloth is Meta's fast LLM training library that ART depends on.
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and show progress."""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success!")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr.strip()[:200]}...")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def install_unsloth():
    """Install Unsloth with multiple methods."""
    print("🦥 Installing Unsloth (this may take 5-10 minutes)...")
    
    # Method 1: Direct install with colab extras
    methods = [
        ('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"', 
         "Installing Unsloth with colab extras"),
        
        ('pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"',
         "Installing basic Unsloth"),
        
        ('pip install unsloth',
         "Installing from PyPI"),
         
        ('pip install unsloth --pre',
         "Installing pre-release version"),
    ]
    
    for cmd, desc in methods:
        print(f"\n🚀 Attempt: {desc}")
        if run_command(cmd, desc):
            return True
        print("   ⚠️ This method failed, trying next...")
    
    print("\n❌ All Unsloth installation methods failed!")
    return False

def install_dependencies_first():
    """Install dependencies that Unsloth needs."""
    print("📦 Installing Unsloth dependencies first...")
    
    deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "accelerate>=0.20.0",
        "peft>=0.5.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "xformers",
    ]
    
    for dep in deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

def test_unsloth():
    """Test if Unsloth can be imported."""
    try:
        import unsloth
        print("✅ Unsloth imported successfully!")
        
        # Test specific functionality
        try:
            from unsloth import FastLanguageModel
            print("✅ FastLanguageModel import working!")
        except ImportError as e:
            print(f"⚠️ FastLanguageModel import failed: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Unsloth import failed: {e}")
        return False

def test_art_import():
    """Test if ART can now be imported."""
    try:
        import art
        print("✅ ART framework imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ ART import still failed: {e}")
        return False

def main():
    print("🦥 Unsloth Installation for ART Framework")
    print("=" * 50)
    print("Unsloth is Meta's library for fast LLM training.")
    print("ART framework requires it for efficient fine-tuning.")
    print("=" * 50)
    
    # Step 1: Install dependencies first
    install_dependencies_first()
    
    # Step 2: Install Unsloth
    if install_unsloth():
        print("\n🧪 Testing Unsloth import...")
        if test_unsloth():
            print("\n🧪 Testing ART framework import...")
            if test_art_import():
                print("\n🎉 SUCCESS! Everything is working!")
                print("\n🚀 You can now run:")
                print("   python train_simple.py")
                print("   python quick_start.py")
            else:
                print("\n⚠️ Unsloth works but ART still has issues.")
                print("💡 Try restarting your Python session.")
        else:
            print("\n⚠️ Unsloth installation succeeded but import failed.")
            print("💡 This might be due to:")
            print("   - CUDA version mismatch")  
            print("   - Insufficient GPU memory")
            print("   - Need to restart Python session")
    else:
        print("\n❌ Unsloth installation failed.")
        print("\n🔧 Manual installation options:")
        print('   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        print("   pip install unsloth")
        print("\n💡 If all fails, you can:")
        print("   1. Try without GPU: CUDA_VISIBLE_DEVICES='' python train_simple.py")
        print("   2. Use the simplified script: python quick_start.py")

if __name__ == "__main__":
    main()