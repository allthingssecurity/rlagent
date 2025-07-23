#!/usr/bin/env python3
"""
FAST GPU Installation - Bypasses dependency resolution hell
Installs packages with --no-deps to avoid conflicts
"""

import subprocess
import sys

def fast_install(package, desc=""):
    """Install package quickly without dependency resolution."""
    print(f"⚡ {desc}")
    cmd = f"pip install {package} --no-deps --force-reinstall --quiet"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("   ✅ SUCCESS")
        return True
    except:
        print("   ❌ FAILED")
        return False

def main():
    print("⚡ FAST GPU Installation (Bypasses Dependency Hell)")
    print("=" * 60)
    
    # Kill any running processes
    subprocess.run("pkill -f python", shell=True, capture_output=True)
    
    # Step 1: Remove problematic packages
    print("\n🗑️ Quick cleanup...")
    cleanup_packages = ["vllm", "peft", "trl", "openai", "pydantic", "fastapi"]
    for pkg in cleanup_packages:
        subprocess.run(f"pip uninstall -y {pkg} -q", shell=True, capture_output=True)
    
    # Step 2: Install core packages with --no-deps
    print("\n🔥 Installing core packages (no dependency resolution)...")
    
    core_packages = [
        ("pydantic==1.10.13", "Pydantic v1"),
        ("typing-extensions==4.8.0", "Typing extensions"),
        ("torch==2.1.2", "PyTorch"),
        ("transformers==4.36.2", "Transformers"),
        ("accelerate==0.25.0", "Accelerate"),
        ("tokenizers==0.15.0", "Tokenizers"),
        ("safetensors==0.4.1", "SafeTensors"),
    ]
    
    for package, desc in core_packages:
        fast_install(package, desc)
    
    # Step 3: Install training packages
    print("\n🏋️ Installing training packages...")
    training_packages = [
        ("peft==0.6.2", "PEFT"),
        ("trl==0.7.4", "TRL"), 
        ("bitsandbytes==0.41.3", "BitsAndBytes"),
    ]
    
    for package, desc in training_packages:
        fast_install(package, desc)
    
    # Step 4: Install vLLM carefully
    print("\n⚡ Installing vLLM...")
    vllm_success = False
    vllm_versions = ["0.2.7", "0.2.6", "0.2.5"]
    
    for version in vllm_versions:
        print(f"   Trying vLLM {version}...")
        if fast_install(f"vllm=={version}", f"vLLM {version}"):
            vllm_success = True
            break
    
    if not vllm_success:
        print("   ⚠️ vLLM failed - will try basic install")
        subprocess.run("pip install vllm --pre --quiet", shell=True, capture_output=True)
    
    # Step 5: Install service packages
    print("\n🌐 Installing service packages...")
    service_packages = [
        ("fastapi==0.68.2", "FastAPI"),
        ("uvicorn==0.15.0", "Uvicorn"),
        ("httpx==0.25.2", "HTTPX"),
        ("openai==1.6.1", "OpenAI"),
        ("python-dotenv==1.0.0", "Python-dotenv"),
    ]
    
    for package, desc in service_packages:
        fast_install(package, desc)
    
    # Step 6: Install ART dependencies
    print("\n🔧 Installing ART dependencies...")
    art_deps = [
        ("setproctitle==1.3.3", "SetProcTitle"),
        ("multiprocess==0.70.15", "Multiprocess"),
        ("tblib==2.0.0", "TBLib"),
        ("cloudpickle==3.0.0", "CloudPickle"),
        ("polars==0.20.6", "Polars"),
        ("weave==0.50.1", "Weave"),
        ("litellm==1.17.9", "LiteLLM"),
    ]
    
    for package, desc in art_deps:
        fast_install(package, desc)
    
    # Step 7: Install ART framework
    print("\n🎨 Installing ART framework...")
    fast_install("openpipe-art", "ART Framework")
    
    # Step 8: Install Unsloth
    print("\n🦥 Installing Unsloth...")
    unsloth_commands = [
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet',
        'pip install unsloth --quiet',
    ]
    
    for cmd in unsloth_commands:
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=300)
            print("   ✅ Unsloth installed")
            break
        except:
            print("   ⚠️ Method failed, trying next...")
            continue
    
    # Step 9: Quick test
    print("\n🧪 Quick test...")
    test_cmd = '''python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')

try:
    import vllm
    print('✅ vLLM imported')
except: print('❌ vLLM failed')

try:
    import peft
    print('✅ PEFT imported')
except: print('❌ PEFT failed')

try:
    import art
    print('✅ ART imported')
except: print('❌ ART failed')
"'''
    
    subprocess.run(test_cmd, shell=True)
    
    print("\n⚡ FAST INSTALLATION COMPLETE!")
    print("🚀 Try running: python train_gpu_optimized.py")

if __name__ == "__main__":
    main()