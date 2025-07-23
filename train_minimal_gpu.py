#!/usr/bin/env python3
"""
Minimal GPU LoRA Training Script for RunPod
Uses only essential packages, no weave/litellm/fastapi conflicts
"""

import os
import asyncio
import torch
from dotenv import load_dotenv

# Load environment
load_dotenv()

# GPU check
if not torch.cuda.is_available():
    print("❌ GPU required for LoRA training")
    exit(1)

print("🎯 MINIMAL GPU LORA TRAINING ON RUNPOD")
print("=" * 50)
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"🐍 PyTorch: {torch.__version__}")
print("=" * 50)

# Minimal imports (no weave conflicts)
import art
from art.local import LocalBackend
from art.trajectories import Trajectory, TrajectoryGroup

# Configuration
MODEL_NAME = "runpod-minimal-lora"
PROJECT = "runpod-gpu"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

async def runpod_lora_train():
    """Minimal LoRA training without problematic dependencies."""
    try:
        print("\n🔧 Creating LocalBackend...")
        backend = LocalBackend()
        
        print("📝 Creating TrainableModel...")
        model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT,
            base_model=BASE_MODEL
        )
        
        # GPU-optimized config
        print("⚙️ Configuring for GPU...")
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=1024,  # Conservative for stability
            )
        )
        
        print("📋 Registering model (critical LoRA test)...")
        print("⏳ This is where LoRA errors typically occur...")
        
        await model.register(backend)
        print("✅ MODEL REGISTERED SUCCESSFULLY!")
        print("🎉 No LoRA errors - minimal stack works!")
        
        # Simple training data
        print("\n📊 Creating training data...")
        messages = [
            art.types.Message(role="system", content="You are a helpful math assistant."),
            art.types.Message(role="user", content="What is 5 + 3?"),
            art.types.Message(role="assistant", content="5 + 3 = 8")
        ]
        
        trajectory = Trajectory(
            messages=messages, 
            reward=1.0,
            metadata={"runpod_training": True}
        )
        group = TrajectoryGroup(trajectories=[trajectory])
        
        print("🏋️ Starting LoRA training on GPU...")
        print("─" * 30)
        
        step_count = 0
        async for metrics in model.train([group]):
            step_count += 1
            print(f"📈 Step {step_count}:")
            if 'loss' in metrics:
                print(f"   📉 Loss: {metrics['loss']:.4f}")
            if 'learning_rate' in metrics:
                print(f"   📊 LR: {metrics['learning_rate']:.2e}")
            
            # GPU memory status
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"   🔥 GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
            
            print("   ✅ Training step completed!")
            
            # Just do one step for testing
            if step_count >= 1:
                break
        
        final_step = await model.get_step()
        print(f"\n🎉 RUNPOD LORA TRAINING SUCCESS!")
        print(f"📈 Final model step: {final_step}")
        print("🚀 Minimal dependency stack works perfectly!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Diagnostic information
        if 'lora_tensors' in str(e):
            print("\n💡 STILL LORA COMPATIBILITY ISSUE:")
            print("   The vLLM/PEFT versions are still incompatible")
            print("   Try: python3 runpod_fix_lora.py")
        elif 'weave' in str(e).lower():
            print("\n💡 WEAVE DEPENDENCY ISSUE:")
            print("   ART is trying to import weave (expected)")
            print("   The minimal stack successfully avoided this!")
        elif 'cuda' in str(e).lower():
            print("\n💡 GPU/CUDA ISSUE:")
            print("   GPU memory or CUDA compatibility problem")
        else:
            print(f"\n💡 DIFFERENT ERROR (this is progress!):")
            print("   Not the usual LoRA/weave issues")
            print("   The minimal stack may be working!")
        
        # Show system info for debugging
        print(f"\n🔍 System Info:")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Try to import key packages
        try:
            import peft, vllm
            print(f"   PEFT: {peft.__version__}")
            print(f"   vLLM: {vllm.__version__}")
        except:
            print("   ⚠️ PEFT/vLLM import issues")
        
        raise
    
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared")

if __name__ == "__main__":
    print("🚀 Starting RunPod minimal LoRA training...")
    asyncio.run(runpod_lora_train())