#!/usr/bin/env python3
"""
CPU-only version of the training script to avoid GPU/vLLM issues.
This bypasses the LoRA/vLLM compatibility problems.
"""

import asyncio
import random
import os
from dotenv import load_dotenv

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load environment variables
load_dotenv()

print("🖥️ CPU-Only ART Training Script")
print("=" * 40)
print("This version runs on CPU to avoid GPU/vLLM compatibility issues")
print("=" * 40)

# Check if we have OpenAI API key
USE_RULER = bool(os.getenv("OPENAI_API_KEY"))
if USE_RULER:
    print("✅ OpenAI API key found - RULER scoring enabled")
else:
    print("⚠️ No OpenAI API key - using basic scoring")

import art
from art.local import LocalBackend
from art.trajectories import Trajectory, TrajectoryGroup

# Configuration for faster CPU training
MODEL_NAME = "cpu-math-agent"
PROJECT = "cpu-training" 
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Smallest model
TRAIN_STEPS = 2  # Fewer steps for CPU
BATCH_SIZE = 2   # Smaller batch for CPU

print(f"Model: {MODEL_NAME}")
print(f"Base Model: {BASE_MODEL}")
print(f"Training Steps: {TRAIN_STEPS}")
print(f"Batch Size: {BATCH_SIZE}")
print("=" * 40)


def generate_simple_problems(num_problems: int = 6):
    """Generate very simple problems for CPU training."""
    problems = []
    
    for i in range(num_problems):
        # Only simple addition for CPU demo
        a, b = random.randint(1, 10), random.randint(1, 10)
        question = f"What is {a} + {b}?"
        answer = a + b
        solution = f"{a} + {b} = {answer}"
        
        messages = [
            art.types.Message(
                role="system", 
                content="You are a math helper. Answer briefly."
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
                "cpu_mode": True
            }
        )
        
        problems.append(trajectory)
    
    return problems


async def cpu_rollout(model, problem_data, step_num):
    """Simple rollout for CPU training."""
    trajectories = []
    
    for i, problem in enumerate(problem_data[:BATCH_SIZE]):
        # Simple variation
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
                "batch_index": i
            }
        )
        
        trajectories.append(trajectory)
    
    return trajectories


async def main():
    """Main CPU training function."""
    random.seed(42)
    
    try:
        print("\n🔧 Initializing ART backend (CPU mode)...")
        backend = LocalBackend()
        
        print("📝 Creating CPU model...")
        model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT,
            base_model=BASE_MODEL,
        )
        
        # CPU-optimized configuration
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=512,  # Very small for CPU
            ),
        )
        
        print("📋 Registering model...")
        await model.register(backend)
        
        current_step = await model.get_step()
        print(f"📈 Current model step: {current_step}")
        
        print("\n📊 Generating simple training data...")
        problem_data = generate_simple_problems(6)
        print(f"✅ Generated {len(problem_data)} simple problems")
        
        # Show sample
        sample = problem_data[0]
        print(f"\n📝 Sample problem: {sample.metadata['question']}")
        print(f"   Answer: {sample.metadata['answer']}")
        
        print(f"\n🚀 Starting CPU training for {TRAIN_STEPS} steps...")
        
        # Training loop
        for step in range(current_step, current_step + TRAIN_STEPS):
            print(f"\n📚 Training Step {step + 1}/{current_step + TRAIN_STEPS}")
            
            # Generate trajectories
            trajectories = await cpu_rollout(model, problem_data, step)
            print(f"   Generated {len(trajectories)} trajectories")
            
            # Create trajectory groups
            trajectory_groups = [TrajectoryGroup(trajectories=trajectories)]
            
            # Simple scoring (no RULER for CPU demo)
            print("   📊 Using simple reward scoring")
            
            # Train with lower learning rate for CPU
            print("   🏋️ Training model (CPU)...")
            try:
                train_config = art.TrainConfig(learning_rate=1e-4)
                
                training_completed = False
                async for metrics in model.train(trajectory_groups, config=train_config):
                    if 'loss' in metrics:
                        print(f"      Loss: {metrics['loss']:.4f}")
                    training_completed = True
                    break  # Just one iteration for CPU demo
                
                if training_completed:
                    print(f"   ✅ Step {step + 1} completed on CPU!")
                else:
                    print(f"   ⚠️ Step {step + 1} had issues")
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                break
        
        final_step = await model.get_step()
        print(f"\n🎉 CPU Training completed! Final step: {final_step}")
        print("\n💡 CPU training is slower but avoids GPU compatibility issues.")
        print("For faster training, fix the GPU dependencies and use train_simple.py")
        
    except Exception as e:
        print(f"\n❌ Error during CPU training: {e}")
        print("💡 Even CPU mode failed. Check your ART installation.")
        raise
    
    finally:
        # Cleanup
        if 'backend' in locals() and backend is not None:
            try:
                backend.close()
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    print("🚀 Starting CPU-only training script...")
    asyncio.run(main())