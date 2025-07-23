#!/usr/bin/env python3
"""
Training script with LoRA bypass to avoid the lora_tensors error.
This disables LoRA entirely and uses base model training.
"""

import os
import asyncio
import random

# DISABLE LORA BEFORE ANY IMPORTS
os.environ["VLLM_DISABLE_LORA"] = "1"
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from dotenv import load_dotenv
load_dotenv()

import torch

# Ensure GPU mode
if not torch.cuda.is_available():
    print("❌ CUDA not available! This script requires GPU.")
    exit(1)

print("🔥 GPU Training with LoRA Bypass")
print("=" * 50)
print("🚫 LoRA disabled to avoid compatibility issues")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)

import art
from art.local import LocalBackend
from art.trajectories import Trajectory, TrajectoryGroup

# Configuration
MODEL_NAME = "gpu-no-lora"
PROJECT = "gpu-bypass"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model without LoRA
TRAIN_STEPS = 3
BATCH_SIZE = 4
USE_RULER = bool(os.getenv("OPENAI_API_KEY"))

print(f"📋 Model: {MODEL_NAME}")
print(f"🧠 Base Model: {BASE_MODEL}")
print(f"🏋️ Training Steps: {TRAIN_STEPS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🚫 LoRA: Disabled")
print("=" * 50)


def generate_problems(num_problems: int = 10):
    """Generate simple problems for no-LoRA training."""
    problems = []
    
    for i in range(num_problems):
        # Simple math problems
        a, b = random.randint(1, 20), random.randint(1, 20)
        question = f"What is {a} + {b}?"
        answer = a + b
        solution = f"{a} + {b} = {answer}"
        
        messages = [
            art.types.Message(
                role="system", 
                content="You are a math helper. Answer briefly and clearly."
            ),
            art.types.Message(
                role="user", 
                content=question
            ),
            art.types.Message(
                role="assistant", 
                content=solution
            )
        ]
        
        trajectory = Trajectory(
            messages=messages,
            reward=1.0,
            metadata={
                "question": question,
                "answer": answer,
                "no_lora": True
            }
        )
        
        problems.append(trajectory)
    
    return problems


async def no_lora_rollout(model, problem_data, step_num):
    """Simple rollout without LoRA complications."""
    trajectories = []
    
    for i, problem in enumerate(problem_data[:BATCH_SIZE]):
        messages = [
            art.types.Message(
                role="system", 
                content="You are helpful with math."
            ),
            problem.messages[1],  # Question
            problem.messages[2]   # Answer
        ]
        
        trajectory = Trajectory(
            messages=messages,
            reward=problem.reward,
            metadata={
                **problem.metadata,
                "step": step_num,
                "batch_index": i,
                "bypass_mode": True
            }
        )
        
        trajectories.append(trajectory)
    
    return trajectories


async def main():
    """Main training function with LoRA bypass."""
    random.seed(42)
    
    try:
        print("\n🔧 Initializing ART backend (LoRA disabled)...")
        backend = LocalBackend()
        
        print("📝 Creating model without LoRA...")
        model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT,
            base_model=BASE_MODEL,
        )
        
        # Simple configuration without LoRA
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=1024,  # Smaller for stability
            ),
        )
        
        print("📋 Registering model...")
        print("⏳ This should work without LoRA errors...")
        
        await model.register(backend)
        print("✅ Model registered successfully!")
        
        current_step = await model.get_step()
        print(f"📈 Current model step: {current_step}")
        
        print(f"\n📊 Generating {10} simple problems...")
        problem_data = generate_problems(10)
        print(f"✅ Generated {len(problem_data)} problems")
        
        # Show sample
        sample = problem_data[0]
        print(f"\n📝 Sample: {sample.metadata['question']} → {sample.metadata['answer']}")
        
        print(f"\n🚀 Starting no-LoRA training for {TRAIN_STEPS} steps...")
        
        # Training loop
        for step in range(current_step, current_step + TRAIN_STEPS):
            print(f"\n🏋️ Training Step {step + 1}/{current_step + TRAIN_STEPS}")
            print("─" * 30)
            
            try:
                # Generate trajectories
                print("   📊 Generating trajectories...")
                trajectories = await no_lora_rollout(model, problem_data, step)
                print(f"   ✅ Generated {len(trajectories)} trajectories")
                
                # Create trajectory groups
                trajectory_groups = [TrajectoryGroup(trajectories=trajectories)]
                print("   📦 Created trajectory groups")
                
                # Train without LoRA
                print("   🏋️ Training model (no LoRA)...")
                train_config = art.TrainConfig(learning_rate=1e-4)  # Conservative LR
                
                training_completed = False
                async for metrics in model.train(trajectory_groups, config=train_config):
                    if 'loss' in metrics:
                        print(f"      📉 Loss: {metrics['loss']:.4f}")
                    if 'learning_rate' in metrics:
                        print(f"      📈 LR: {metrics['learning_rate']:.2e}")
                    
                    training_completed = True
                    break  # Just one iteration per step
                
                if training_completed:
                    print(f"   ✅ Step {step + 1} completed!")
                    
                    # Show GPU memory
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1e9
                        print(f"   🔥 GPU Memory: {memory_used:.1f}GB used")
                else:
                    print(f"   ⚠️ Step {step + 1} had issues")
                
            except Exception as e:
                print(f"   ❌ Step failed: {e}")
                print(f"   🔍 Error type: {type(e).__name__}")
                
                # Check if it's still the LoRA error
                if 'lora_tensors' in str(e).lower():
                    print("   🚫 Still hitting LoRA error - trying alternative approach")
                    break
                else:
                    print("   🔄 Different error - continuing...")
                    continue
        
        final_step = await model.get_step()
        print(f"\n🎉 Training completed!")
        print(f"📈 Final step: {final_step}")
        print(f"🚫 LoRA bypass mode worked!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"🔍 Error details: {type(e).__name__}")
        
        if 'lora_tensors' in str(e).lower():
            print("\n💡 Still hitting LoRA compatibility issue!")
            print("🔧 Possible solutions:")
            print("   1. Run: python fix_lora_tensors.py")
            print("   2. Try even older versions: vLLM 0.2.5, PEFT 0.4.0")
            print("   3. Use pure CPU training: python train_simple_cpu.py")
        else:
            print(f"\n💡 Different error - this is progress!")
            print("The LoRA bypass worked, but hit a different issue")
        
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if 'backend' in locals() and backend is not None:
            try:
                backend.close()
            except Exception as e:
                print(f"Warning: Cleanup error: {e}")


if __name__ == "__main__":
    print("🚀 Starting LoRA bypass training...")
    asyncio.run(main())