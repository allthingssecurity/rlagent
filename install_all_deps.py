#!/usr/bin/env python3
"""
Comprehensive dependency installer for ART framework on RunPod.
This script installs ALL missing dependencies that ART needs.
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
    except ImportError:
        print(f"   ❌ {package_name}")
        return False

def main():
    print("🔧 Installing ALL ART framework dependencies...")
    print("=" * 50)
    
    # Complete list of dependencies needed for ART
    dependencies = [
        # Missing system deps
        "setproctitle>=1.3.0",
        "multiprocess>=0.70.0", 
        "tblib>=1.7.0",
        "cloudpickle>=2.0.0",
        "dill>=0.3.0",
        
        # ART core dependencies
        "polars>=0.20.0",
        "weave>=0.50.0",
        "litellm>=1.0.0",
        
        # ML framework deps
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.13.0",
        "datasets>=2.12.0",
        
        # Training framework deps
        "trl>=0.7.0",
        "peft>=0.5.0",
        "torchtune>=0.1.0",
        
        # API and service deps
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "httpx>=0.25.0",
        "openai>=1.0.0",
        
        # Utility deps
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "asyncio-throttle>=1.0.0",
        
        # Additional missing deps often needed
        "protobuf>=3.20.0",
        "packaging>=21.0",
        "typing-extensions>=4.0.0",
        "filelock>=3.0.0",
        "huggingface-hub>=0.16.0",
        "safetensors>=0.3.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ]
    
    # Install basic dependencies first
    print("\n🎯 Installing basic dependencies...")
    basic_deps = [
        "setproctitle>=1.3.0",
        "multiprocess>=0.70.0", 
        "tblib>=1.7.0",
        "cloudpickle>=2.0.0",
        "dill>=0.3.0",
        "packaging>=21.0",
        "typing-extensions>=4.0.0",
        "filelock>=3.0.0",
    ]
    
    for package in basic_deps:
        install_package(package)
    
    # Install ML dependencies
    print("\n🤖 Installing ML dependencies...")
    ml_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tokenizers>=0.13.0",
        "huggingface-hub>=0.16.0",
        "safetensors>=0.3.0",
    ]
    
    for package in ml_deps:
        install_package(package)
    
    # Install remaining dependencies
    print("\n📚 Installing remaining dependencies...")
    remaining_deps = [dep for dep in dependencies if dep not in basic_deps + ml_deps]
    
    for package in remaining_deps:
        install_package(package)
    
    # Try to install Unsloth (optional but recommended)
    print("\n🦥 Installing Unsloth (optional)...")
    unsloth_success = install_package("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    if not unsloth_success:
        print("   ⚠️ Unsloth installation failed (not critical for basic functionality)")
    
    # Install ART framework last
    print("\n🎨 Installing ART framework...")
    art_success = install_package("openpipe-art>=0.1.0")
    
    print("\n" + "=" * 50)
    print("🧪 Testing critical imports...")
    
    # Test critical imports
    critical_imports = [
        ("setproctitle", "setproctitle"),
        ("multiprocess", "multiprocess"),
        ("tblib", "tblib"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("art", "ART Framework"),
        ("fastapi", "FastAPI"),
        ("openai", "OpenAI"),
    ]
    
    all_good = True
    for module, name in critical_imports:
        if not test_import(module, name):
            all_good = False
    
    # Test ART backend specifically
    print("\n🏗️ Testing ART backend...")
    try:
        from art.local import LocalBackend
        print("   ✅ ART LocalBackend")
    except ImportError as e:
        print(f"   ❌ ART LocalBackend: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 SUCCESS! All dependencies installed and working!")
        print("\n📋 You can now run:")
        print("   python train_simple.py")
        print("   python quick_start.py")
        print("   python run_server.py")
    else:
        print("⚠️ Some dependencies are still missing.")
        print("Check the error messages above and try manual installation.")
        print("\n🔧 If issues persist, try:")
        print("   pip install --upgrade pip")
        print("   pip install --force-reinstall <package_name>")

if __name__ == "__main__":
    main()