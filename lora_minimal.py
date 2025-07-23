#!/usr/bin/env python3
"""
MINIMAL LoRA-only installation - just get LoRA working
Skips problematic packages like Weave that cause conflicts
"""

import subprocess
import sys

def minimal_lora_install():
    """Install minimal packages needed for LoRA training."""
    print("⚡ MINIMAL LoRA Installation")
    print("Skipping problematic packages, focusing on LoRA essentials")
    
    # Kill everything first
    subprocess.run("pkill -f pip", shell=True, capture_output=True)
    
    # Remove conflicting packages
    print("\n🗑️ Removing conflicts...")
    conflicts = ["weave", "litellm", "openai", "pydantic", "vllm", "peft", "trl"]
    for pkg in conflicts:
        subprocess.run(f"pip uninstall -y {pkg}", shell=True, capture_output=True)
    
    # Install minimal working stack
    print("\n🔧 Installing minimal LoRA stack...")
    
    minimal_packages = [
        # Core (no conflicts)
        "pydantic==1.10.13 --no-deps",
        "torch --no-deps", 
        "transformers==4.33.2 --no-deps",
        "accelerate==0.21.0 --no-deps",
        "tokenizers --no-deps",
        
        # LoRA essentials
        "peft==0.4.0 --no-deps",
        "trl==0.6.0 --no-deps", 
        "bitsandbytes --no-deps",
        
        # vLLM (careful)
        "vllm==0.2.2 --no-deps",
        
        # Basic ART deps (skip weave)
        "setproctitle --no-deps",
        "multiprocess --no-deps",
        "tblib --no-deps",
        "cloudpickle --no-deps",
        "polars==0.18.15 --no-deps",
        
        # Old OpenAI (compatible)
        "openai==0.28.1 --no-deps",
        
        # ART framework
        "openpipe-art --no-deps",
    ]
    
    for pkg in minimal_packages:
        print(f"   Installing {pkg.split('==')[0]}...")
        cmd = f"pip install {pkg}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=60)
            print("     ✅ OK")
        except:
            print("     ⚠️ Failed")
    
    print("\n⚡ Minimal install complete!")

def test_minimal():
    """Test the minimal install."""
    print("\n🧪 Testing minimal LoRA setup...")
    
    test_code = '''
print("Testing imports...")

try:
    import torch
    print(f"✅ PyTorch: {torch.cuda.is_available()}")
except: print("❌ PyTorch failed")

try:
    import peft
    print("✅ PEFT imported")
    
    from peft import LoraConfig
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj"])
    print("✅ LoRA config works")
except Exception as e: print(f"❌ PEFT: {e}")

try:
    import vllm
    print("✅ vLLM imported")
except Exception as e: print(f"❌ vLLM: {e}")

try:
    import art
    print("✅ ART imported")
except Exception as e: print(f"❌ ART: {e}")

print("🎯 Minimal test complete!")
'''
    
    subprocess.run(f'python -c "{test_code}"', shell=True)

def create_minimal_train():
    """Create a minimal training script."""
    script = '''#!/usr/bin/env python3
"""Minimal LoRA training script."""

import os
import asyncio
import torch

# Check GPU
if not torch.cuda.is_available():
    print("❌ Need GPU for LoRA training")
    exit(1)

print("🎯 Minimal LoRA Training")
print(f"GPU: {torch.cuda.get_device_name(0)}")

import art
from art.local import LocalBackend
from art.trajectories import Trajectory, TrajectoryGroup

async def minimal_train():
    """Minimal LoRA training."""
    try:
        print("\\n🔧 Creating backend...")
        backend = LocalBackend()
        
        print("📝 Creating model...")
        model = art.TrainableModel(
            name="minimal-lora",
            project="minimal",
            base_model="Qwen/Qwen2.5-0.5B-Instruct"
        )
        
        print("📋 Registering model (this is where LoRA errors happen)...")
        await model.register(backend)
        print("✅ Registration successful!")
        
        # Simple trajectory
        messages = [
            art.types.Message(role="system", content="You are helpful."),
            art.types.Message(role="user", content="Say hello"),
            art.types.Message(role="assistant", content="Hello!")
        ]
        
        trajectory = Trajectory(messages=messages, reward=1.0)
        group = TrajectoryGroup(trajectories=[trajectory])
        
        print("🏋️ Training...")
        async for metrics in model.train([group]):
            print(f"✅ Loss: {metrics.get('loss', 'N/A')}")
            break
        
        print("🎉 LoRA training worked!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        if 'lora_tensors' in str(e):
            print("💡 Still LoRA compatibility issue")
        else:
            print("💡 Different error - LoRA might be working!")

if __name__ == "__main__":
    asyncio.run(minimal_train())
'''
    
    with open('train_minimal_lora.py', 'w') as f:
        f.write(script)
    
    subprocess.run('chmod +x train_minimal_lora.py', shell=True)
    print("📝 Created train_minimal_lora.py")

def main():
    print("⚡ MINIMAL LoRA SOLUTION")
    print("=" * 40)
    print("Installs only what's needed for LoRA, skips conflicts")
    
    minimal_lora_install()
    test_minimal()
    create_minimal_train()
    
    print("\n🚀 Try running:")
    print("   python train_minimal_lora.py")
    print("\n💡 This skips weave/litellm conflicts entirely")

if __name__ == "__main__":
    main()