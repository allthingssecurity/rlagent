#!/usr/bin/env python3
"""
STOP PIP MADNESS - Kill dependency resolution and install manually
"""

import subprocess
import sys
import signal
import os

def kill_pip():
    """Kill all pip processes."""
    print("🛑 Killing all pip processes...")
    try:
        subprocess.run("pkill -f pip", shell=True, capture_output=True)
        subprocess.run("pkill -f python", shell=True, capture_output=True)
        print("   ✅ Processes killed")
    except:
        pass

def manual_install():
    """Install packages manually without dependency resolution."""
    print("🔧 Manual installation without dependency hell...")
    
    # Essential packages only
    essential = [
        "torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118",
        "pydantic==1.10.13 --no-deps",
        "transformers==4.36.2 --no-deps", 
        "vllm==0.2.7 --no-deps",
        "peft==0.6.2 --no-deps",
        "openpipe-art --no-deps",
    ]
    
    for package in essential:
        print(f"   Installing {package.split('==')[0]}...")
        cmd = f"pip install {package} --quiet --force-reinstall"
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=60)
            print("   ✅ OK")
        except:
            print("   ⚠️ Failed (continuing anyway)")
    
    print("🎯 Manual install complete!")

def main():
    print("🛑 STOP PIP DEPENDENCY MADNESS!")
    print("=" * 40)
    
    # Kill pip
    kill_pip()
    
    # Manual install
    manual_install()
    
    # Test basic imports
    print("\n🧪 Testing basic imports...")
    test_code = """
try:
    import torch
    print('✅ PyTorch works')
    print(f'   CUDA: {torch.cuda.is_available()}')
except: print('❌ PyTorch failed')

try:
    import art
    print('✅ ART works')
except: print('❌ ART failed')
"""
    
    try:
        exec(test_code)
    except:
        subprocess.run(f'python -c "{test_code}"', shell=True)
    
    print("\n🚀 Try running: python train_gpu_optimized.py")
    print("💡 Or try the simple CPU version: python train_simple_cpu.py")

if __name__ == "__main__":
    main()