#!/usr/bin/env python3
"""
Install all PyTorch-related dependencies for ART framework.
This handles the complex dependency chain for torchtune/torchao.
"""

import subprocess
import sys
import importlib

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        importlib.import_module(module_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name}: {e}")
        return False

def main():
    print("🔥 Installing PyTorch ecosystem for ART framework")
    print("=" * 50)
    
    # Step 1: Install PyTorch with CUDA support
    print("\n🚀 Step 1: Installing PyTorch with CUDA...")
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        print("⚠️ CUDA PyTorch failed, trying CPU version...")
        cpu_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        run_command(cpu_cmd, "Installing PyTorch CPU version")
    
    # Step 2: Install torchao first (torchtune dependency)
    print("\n🔧 Step 2: Installing torchao...")
    torchao_commands = [
        "pip install torchao>=0.1.0",
        "pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cu118",
        "pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cpu"
    ]
    
    torchao_installed = False
    for cmd in torchao_commands:
        if run_command(cmd, f"Trying: {cmd}"):
            torchao_installed = True
            break
    
    if not torchao_installed:
        print("⚠️ torchao installation failed with all methods")
    
    # Step 3: Install torchtune
    print("\n⚡ Step 3: Installing torchtune...")
    torchtune_commands = [
        "pip install torchtune>=0.1.0",
        "pip install torchtune --pre",
        "pip install git+https://github.com/pytorch/torchtune.git"
    ]
    
    torchtune_installed = False
    for cmd in torchtune_commands:
        if run_command(cmd, f"Trying: {cmd}"):
            torchtune_installed = True
            break
    
    if not torchtune_installed:
        print("⚠️ torchtune installation failed with all methods")
    
    # Step 4: Install other training dependencies
    print("\n📚 Step 4: Installing training frameworks...")
    training_deps = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0", 
        "trl>=0.7.0",
        "peft>=0.5.0",
        "bitsandbytes>=0.41.0",
    ]
    
    for dep in training_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Step 5: Test all imports
    print("\n🧪 Testing PyTorch ecosystem imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchao", "TorchAO"), 
        ("torchtune", "TorchTune"),
        ("transformers", "Transformers"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
    ]
    
    all_working = True
    for module, name in imports_to_test:
        if not test_import(module, name):
            all_working = False
    
    # Test GPU availability
    print("\n🖥️ Testing GPU setup...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   🔥 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ CUDA not available - will use CPU (slower)")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
    
    print("\n" + "=" * 50)
    if all_working:
        print("🎉 SUCCESS! All PyTorch dependencies installed!")
        print("\n🚀 You can now run:")
        print("   python train_simple.py")
        print("   python quick_start.py")
    else:
        print("⚠️ Some dependencies are still missing.")
        print("\n💡 Try these fallback options:")
        print("1. Restart your Python session")
        print("2. Use CPU-only mode by setting CUDA_VISIBLE_DEVICES=''")
        print("3. Try the simplified training script: python quick_start.py")

if __name__ == "__main__":
    main()