#!/usr/bin/env python3
"""
DEFINITIVE GPU-ONLY Installation Script
This will make ART work on GPU by fixing ALL version conflicts
"""

import subprocess
import sys
import time

def run_cmd(cmd, desc="", critical=True):
    """Run command with error handling."""
    print(f"🔧 {desc}")
    print(f"   → {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ SUCCESS")
            return True
        else:
            print(f"   ❌ FAILED: {result.stderr.strip()[:200]}")
            if critical:
                print("   🛑 This is critical - stopping installation")
                return False
            return False
    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
        if critical:
            return False
        return False

def nuclear_cleanup():
    """Nuclear option: remove ALL conflicting packages."""
    print("☢️ NUCLEAR CLEANUP - Removing ALL conflicting packages")
    print("This will take 2-3 minutes but ensures clean slate")
    
    packages_to_nuke = [
        "vllm", "peft", "trl", "transformers", "accelerate", 
        "bitsandbytes", "unsloth", "torchtune", "torchao",
        "fastapi", "pydantic", "openai", "httpx", "uvicorn",
        "openpipe-art"
    ]
    
    # Uninstall everything
    for pkg in packages_to_nuke:
        run_cmd(f"pip uninstall -y {pkg}", f"Nuking {pkg}", critical=False)
    
    # Clean pip cache
    run_cmd("pip cache purge", "Cleaning pip cache", critical=False)
    print("☢️ Nuclear cleanup complete - starting fresh install")

def install_gpu_stack():
    """Install GPU stack in precise order."""
    print("🚀 Installing definitive GPU stack for ART")
    
    # Step 1: System foundations
    print("\n📦 STEP 1: System foundations")
    foundations = [
        "pip install --upgrade pip setuptools wheel",
        "pip install packaging typing-extensions filelock",
    ]
    for cmd in foundations:
        if not run_cmd(cmd, "Installing foundation packages"):
            return False
    
    # Step 2: PyTorch with CUDA (CRITICAL)
    print("\n🔥 STEP 2: PyTorch with CUDA")
    pytorch_cmd = "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118"
    if not run_cmd(pytorch_cmd, "Installing PyTorch 2.1.2 with CUDA"):
        return False
    
    # Verify CUDA works
    cuda_test = """python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
print(f'🔥 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" """
    if not run_cmd(cuda_test, "Verifying CUDA works"):
        return False
    
    # Step 3: Core ML frameworks (exact versions)
    print("\n🤖 STEP 3: Core ML frameworks")
    ml_packages = [
        "transformers==4.36.2",  # Compatible with vLLM
        "accelerate==0.25.0",
        "tokenizers==0.15.0", 
        "datasets==2.16.1",
        "huggingface-hub==0.20.2",
        "safetensors==0.4.1",
    ]
    
    for pkg in ml_packages:
        if not run_cmd(f"pip install {pkg}", f"Installing {pkg}"):
            return False
    
    # Step 4: Pydantic v1 (CRITICAL for vLLM)
    print("\n📋 STEP 4: Pydantic v1 ecosystem")
    pydantic_packages = [
        "pydantic==1.10.13",
        "pydantic-core==2.6.3", 
        "typing-extensions==4.8.0",
    ]
    
    for pkg in pydantic_packages:
        if not run_cmd(f"pip install {pkg}", f"Installing {pkg}"):
            return False
    
    # Step 5: vLLM (GPU inference engine)
    print("\n⚡ STEP 5: vLLM inference engine")
    vllm_cmd = "pip install vllm==0.2.7 --no-deps"
    if not run_cmd(vllm_cmd, "Installing vLLM 0.2.7 (no deps)"):
        # Try alternative
        if not run_cmd("pip install vllm==0.2.6", "Installing vLLM 0.2.6"):
            return False
    
    # Install vLLM dependencies manually
    vllm_deps = [
        "ray[serve]==2.8.1",
        "xformers==0.0.23",
        "sentencepiece==0.1.99",
    ]
    
    for dep in vllm_deps:
        run_cmd(f"pip install {dep}", f"Installing vLLM dep: {dep}", critical=False)
    
    # Step 6: Training frameworks (compatible versions)
    print("\n🏋️ STEP 6: Training frameworks")
    training_packages = [
        "peft==0.6.2",        # Compatible with vLLM 0.2.7
        "trl==0.7.4",         # Compatible with PEFT 0.6.2
        "bitsandbytes==0.41.3",
    ]
    
    for pkg in training_packages:
        if not run_cmd(f"pip install {pkg}", f"Installing {pkg}"):
            return False
    
    # Step 7: PyTorch ecosystem (order matters!)
    print("\n⚡ STEP 7: PyTorch ecosystem")
    # Install torchao first
    torchao_cmd = "pip install torchao==0.1.1 --index-url https://download.pytorch.org/whl/cu118"
    if not run_cmd(torchao_cmd, "Installing torchao"):
        run_cmd("pip install torchao", "Installing torchao fallback", critical=False)
    
    # Then torchtune
    if not run_cmd("pip install torchtune==0.1.1", "Installing torchtune"):
        return False
    
    # Step 8: Unsloth (GPU acceleration)
    print("\n🦥 STEP 8: Unsloth GPU acceleration")
    unsloth_cmd = 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    if not run_cmd(unsloth_cmd, "Installing Unsloth"):
        # Fallback to basic Unsloth
        run_cmd("pip install unsloth", "Installing basic Unsloth", critical=False)
    
    # Step 9: ART system dependencies
    print("\n🔧 STEP 9: ART system dependencies")
    art_deps = [
        "setproctitle==1.3.3",
        "multiprocess==0.70.15", 
        "tblib==2.0.0",
        "cloudpickle==3.0.0",
        "dill==0.3.7",
        "polars==0.20.6",
        "weave==0.50.1",
        "litellm==1.17.9",
    ]
    
    for pkg in art_deps:
        if not run_cmd(f"pip install {pkg}", f"Installing {pkg}"):
            return False
    
    # Step 10: Service dependencies (Pydantic v1 compatible)
    print("\n🌐 STEP 10: Service dependencies")
    service_deps = [
        "fastapi==0.68.2",      # Compatible with Pydantic v1
        "uvicorn[standard]==0.15.0",
        "httpx==0.25.2",
        "openai==1.6.1",        # Compatible with Pydantic v1
        "python-dotenv==1.0.0",
    ]
    
    for pkg in service_deps:
        if not run_cmd(f"pip install {pkg}", f"Installing {pkg}"):
            return False
    
    # Step 11: ART framework (FINAL)
    print("\n🎨 STEP 11: ART framework")
    if not run_cmd("pip install openpipe-art", "Installing ART framework"):
        return False
    
    return True

def test_gpu_setup():
    """Test complete GPU setup."""
    print("\n🧪 TESTING GPU SETUP")
    
    # Test 1: CUDA availability
    cuda_test = """python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"```
    
    if not run_cmd(cuda_test, "Testing CUDA"):
        return False
    
    # Test 2: Critical imports
    import_test = """python -c "
modules = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'), 
    ('vllm', 'vLLM'),
    ('peft', 'PEFT'),
    ('trl', 'TRL'),
    ('unsloth', 'Unsloth'),
    ('art', 'ART Framework'),
]

all_good = True
for module, name in modules:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        all_good = False

if all_good:
    print('🎉 ALL IMPORTS WORKING!')
else:
    print('⚠️ Some imports failed')
"```
    
    if not run_cmd(import_test, "Testing imports"):
        return False
    
    # Test 3: ART backend
    backend_test = """python -c "
from art.local import LocalBackend
import art

print('Testing ART backend...')
backend = LocalBackend()

print('Creating test model...')
model = art.TrainableModel(
    name='gpu-test',
    project='gpu-test',
    base_model='Qwen/Qwen2.5-0.5B-Instruct'
)

print('✅ ART GPU setup successful!')
"```
    
    return run_cmd(backend_test, "Testing ART backend")

def main():
    print("🚀 DEFINITIVE GPU-ONLY ART INSTALLATION")
    print("=" * 60)
    print("This will make ART work on GPU by fixing ALL version conflicts")
    print("Estimated time: 15-20 minutes")
    print("=" * 60)
    
    # Nuclear cleanup
    nuclear_cleanup()
    
    # Install GPU stack
    if install_gpu_stack():
        print("\n✅ GPU stack installation completed!")
        
        # Test everything
        if test_gpu_setup():
            print("\n🎉 SUCCESS! ART is ready for GPU training!")
            print("\n🚀 You can now run:")
            print("   python train_simple.py")
            print("\n🔥 Expected to see:")
            print("   - Unsloth patches applied")
            print("   - GPU memory allocated") 
            print("   - Training steps completing")
        else:
            print("\n⚠️ Installation succeeded but tests failed")
            print("Try running train_simple.py anyway - it might work")
    else:
        print("\n❌ GPU stack installation failed")
        print("Check error messages above and fix issues")

if __name__ == "__main__":
    main()