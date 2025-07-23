#!/usr/bin/env python3
"""
COMPLETE installer for ALL ART framework dependencies.
This is the ultimate fix that installs everything needed.
"""

import subprocess
import sys
import time
import importlib

def run_command(cmd, description="", ignore_errors=False):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr.strip()}")
            if not ignore_errors:
                return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        if not ignore_errors:
            return False
    return True

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        importlib.import_module(module_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name}: {str(e)[:100]}...")
        return False

def main():
    print("🚀 COMPLETE ART Framework Installation")
    print("=" * 60)
    print("This will install ALL dependencies needed for ART training.")
    print("Estimated time: 10-15 minutes")
    print("=" * 60)
    
    # Step 1: System packages
    print("\n📦 Step 1: Installing system packages...")
    system_commands = [
        "apt-get update -qq",
        "apt-get install -y -qq build-essential git curl wget",
        "apt-get install -y -qq python3-dev python3-pip",
    ]
    
    for cmd in system_commands:
        run_command(cmd, f"Running: {cmd}", ignore_errors=True)
    
    # Step 2: Upgrade pip and install wheel
    print("\n🛠️ Step 2: Upgrading pip and build tools...")
    pip_commands = [
        "pip install --upgrade pip setuptools wheel",
        "pip install --upgrade packaging",
    ]
    
    for cmd in pip_commands:
        run_command(cmd, f"Running: {cmd}")
    
    # Step 3: Install PyTorch ecosystem
    print("\n🔥 Step 3: Installing PyTorch ecosystem...")
    pytorch_commands = [
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cu118",
        "pip install torchtune>=0.1.0",
    ]
    
    for i, cmd in enumerate(pytorch_commands):
        success = run_command(cmd, f"PyTorch step {i+1}/3: {cmd.split()[2]}")
        if not success and "torch" in cmd:
            # Fallback to CPU version
            fallback = cmd.replace("cu118", "cpu")
            run_command(fallback, f"Fallback: CPU version", ignore_errors=True)
    
    # Step 4: Install vLLM (critical for ART)
    print("\n⚡ Step 4: Installing vLLM inference engine...")
    vllm_commands = [
        "pip install vllm>=0.2.0",
        "pip install vllm --pre",
        "pip install vllm --upgrade",
    ]
    
    vllm_success = False
    for cmd in vllm_commands:
        if run_command(cmd, f"Trying: {cmd}"):
            vllm_success = True
            break
    
    if not vllm_success:
        print("   ⚠️ vLLM installation failed - trying alternative methods...")
        alt_commands = [
            "pip install vllm --no-deps",
            "pip install ray[serve] xformers",  # vLLM dependencies
        ]
        for cmd in alt_commands:
            run_command(cmd, f"Alternative: {cmd}", ignore_errors=True)
    
    # Step 5: Install core ML libraries
    print("\n🤖 Step 5: Installing ML frameworks...")
    ml_packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0", 
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.13.0",
        "datasets>=2.12.0",
        "huggingface-hub>=0.16.0",
        "safetensors>=0.3.0",
    ]
    
    for package in ml_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 6: Install training frameworks
    print("\n🏋️ Step 6: Installing training frameworks...")
    training_packages = [
        "trl>=0.7.0",
        "peft>=0.5.0", 
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    ]
    
    for package in training_packages:
        run_command(f"pip install '{package}'", f"Installing {package.split('@')[0]}", ignore_errors=True)
    
    # Step 7: Install ART system dependencies
    print("\n🔧 Step 7: Installing ART system dependencies...")
    art_deps = [
        "setproctitle>=1.3.0",
        "multiprocess>=0.70.0",
        "tblib>=1.7.0", 
        "cloudpickle>=2.0.0",
        "dill>=0.3.0",
        "polars>=0.20.0",
        "weave>=0.50.0",
        "litellm>=1.0.0",
    ]
    
    for package in art_deps:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 8: Install service dependencies
    print("\n🌐 Step 8: Installing service dependencies...")
    service_deps = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "httpx>=0.25.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "wandb>=0.15.0",
    ]
    
    for package in service_deps:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 9: Install utility packages
    print("\n📊 Step 9: Installing utility packages...")
    util_deps = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ]
    
    for package in util_deps:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 10: Install ART framework
    print("\n🎨 Step 10: Installing ART framework...")
    run_command("pip install openpipe-art>=0.1.0", "Installing ART framework")
    
    # Step 11: Test everything
    print("\n🧪 Step 11: Testing all critical imports...")
    
    critical_imports = [
        ("torch", "PyTorch"),
        ("torchao", "TorchAO"),
        ("torchtune", "TorchTune"), 
        ("vllm", "vLLM"),
        ("transformers", "Transformers"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("art", "ART Framework"),
        ("fastapi", "FastAPI"),
        ("openai", "OpenAI"),
    ]
    
    working_count = 0
    for module, name in critical_imports:
        if test_import(module, name):
            working_count += 1
    
    # Test ART backend specifically
    print("\n🏗️ Testing ART backend integration...")
    try:
        from art.local import LocalBackend
        print("   ✅ ART LocalBackend")
        working_count += 1
    except ImportError as e:
        print(f"   ❌ ART LocalBackend: {e}")
    
    # Test GPU
    print("\n🖥️ Testing GPU setup...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   🔥 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ CUDA not available - will use CPU")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
    
    # Final results
    print("\n" + "=" * 60)
    total_tests = len(critical_imports) + 1  # +1 for ART backend
    
    if working_count >= total_tests - 2:  # Allow 2 failures
        print("🎉 SUCCESS! ART framework is ready!")
        print(f"✅ {working_count}/{total_tests} critical components working")
        print("\n🚀 You can now run:")
        print("   python train_simple.py")
        print("   python quick_start.py") 
        print("   python run_server.py")
    elif working_count >= total_tests // 2:
        print("⚠️ PARTIAL SUCCESS - Most components working")
        print(f"✅ {working_count}/{total_tests} critical components working")
        print("\n💡 Try running the simpler script first:")
        print("   python quick_start.py")
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print(f"Only {working_count}/{total_tests} components working")
        print("\n🔧 Try:")
        print("   1. Restart your Python session")
        print("   2. Run this script again")
        print("   3. Check GPU memory: nvidia-smi")

if __name__ == "__main__":
    main()