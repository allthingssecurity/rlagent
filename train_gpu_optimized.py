#!/usr/bin/env python3
"""
GPU-OPTIMIZED ART Training Script
This is specifically designed for GPU training with proper error handling.
"""

import asyncio
import random
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure GPU mode
if not torch.cuda.is_available():
    print("❌ CUDA not available! This script requires GPU.")
    print("Check your CUDA installation and GPU drivers.")
    exit(1)

print("🔥 GPU-OPTIMIZED ART Training Script")
print("=" * 50)
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"⚡ CUDA Version: {torch.version.cuda}")
print("=" * 50)

import art
from art.local import LocalBackend
from art.trajectories import Trajectory, TrajectoryGroup

# GPU-optimized configuration
MODEL_NAME = "gpu-math-agent"
PROJECT = "gpu-training" 
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Larger model for GPU
TRAIN_STEPS = 5
BATCH_SIZE = 8
USE_RULER = bool(os.getenv("OPENAI_API_KEY"))

print(f"📋 Model: {MODEL_NAME}")
print(f"🧠 Base Model: {BASE_MODEL}")
print(f"🏋️ Training Steps: {TRAIN_STEPS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"⚖️ RULER Scoring: {USE_RULER}")
print("=" * 50)


def generate_advanced_problems(num_problems: int = 20):
    """Generate more complex problems for GPU training."""
    problems = []
    
    problem_types = [
        "addition", "subtraction", "multiplication", 
        "division", "word_problem", "multi_step"
    ]
    
    for i in range(num_problems):
        problem_type = random.choice(problem_types)
        
        if problem_type == "addition":
            a, b = random.randint(10, 200), random.randint(10, 200)
            question = f"Calculate {a} + {b}"
            answer = a + b
            solution = f"To calculate {a} + {b}:\n{a} + {b} = {answer}"
            
        elif problem_type == "subtraction":
            a, b = random.randint(100, 500), random.randint(10, 99)
            question = f"Calculate {a} - {b}"
            answer = a - b
            solution = f"To calculate {a} - {b}:\n{a} - {b} = {answer}"
            
        elif problem_type == "multiplication":
            a, b = random.randint(5, 25), random.randint(5, 25)
            question = f"Calculate {a} × {b}"
            answer = a * b
            solution = f"To calculate {a} × {b}:\n{a} × {b} = {answer}"
            
        elif problem_type == "division":
            b = random.randint(5, 20)
            answer = random.randint(5, 50)
            a = b * answer
            question = f"Calculate {a} ÷ {b}"
            solution = f"To calculate {a} ÷ {b}:\n{a} ÷ {b} = {answer}"
            
        elif problem_type == "word_problem":
            items = random.randint(15, 50)
            used = random.randint(5, items - 5)
            remaining = items - used
            question = f"A store has {items} items. If {used} are sold, how many remain?"
            answer = remaining
            solution = f"Starting items: {items}\nSold: {used}\nRemaining: {items} - {used} = {remaining}"
            
        else:  # multi_step
            a, b, c = random.randint(5, 20), random.randint(3, 15), random.randint(2, 10)
            question = f"Calculate ({a} + {b}) × {c}"
            step1 = a + b
            answer = step1 * c
            solution = f"Step 1: {a} + {b} = {step1}\nStep 2: {step1} × {c} = {answer}"
        
        # Create trajectory with richer conversation
        messages = [
            art.types.Message(
                role="system", 
                content="You are an expert math tutor. Provide clear, step-by-step solutions to mathematical problems. Show your work and explain your reasoning."
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
            reward=1.0,  # Perfect score for correct solutions
            metadata={
                "problem_type": problem_type,
                "correct_answer": answer,
                "question": question,
                "difficulty": "intermediate" if problem_type in ["multi_step", "word_problem"] else "basic"
            }
        )
        
        problems.append(trajectory)
    
    return problems


async def gpu_rollout(model, problem_data, step_num):
    """GPU-optimized rollout function."""
    trajectories = []
    
    # Use more problems for GPU (better data variety)
    selected_problems = random.sample(problem_data, min(BATCH_SIZE, len(problem_data)))
    
    for i, problem in enumerate(selected_problems):
        # Add system prompt variations for diversity
        system_prompts = [
            "You are an expert mathematics teacher. Solve problems clearly and show all steps.",
            "You are a helpful math tutor. Break down complex problems into simple steps.",
            "You are a mathematical problem solver. Provide detailed explanations for your solutions.",
            "You are an experienced math instructor. Guide students through problem-solving processes.",
        ]
        
        # Create enhanced trajectory
        messages = [
            art.types.Message(
                role="system", 
                content=system_prompts[i % len(system_prompts)]
            ),
            problem.messages[1],  # User question
            problem.messages[2]   # Assistant solution
        ]
        
        trajectory = Trajectory(
            messages=messages,
            reward=problem.reward,
            metadata={
                **problem.metadata,
                "step": step_num,
                "batch_index": i,
                "gpu_optimized": True,
                "system_prompt_variant": i % len(system_prompts)
            }
        )
        
        trajectories.append(trajectory)
    
    return trajectories


async def main():
    """Main GPU training function."""
    random.seed(42)
    
    try:
        print("\n🔧 Initializing ART backend for GPU...")
        backend = LocalBackend()
        
        print("📝 Creating GPU-optimized model...")
        model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT,
            base_model=BASE_MODEL,
        )
        
        # GPU-optimized configuration
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=2048,  # Larger context for GPU
            ),
        )
        
        print("📋 Registering model with backend...")
        await model.register(backend)
        
        current_step = await model.get_step()
        print(f"📈 Current model step: {current_step}")
        
        print(f"\n📊 Generating {20} advanced training problems...")
        problem_data = generate_advanced_problems(20)
        print(f"✅ Generated problems with difficulty levels:")
        
        # Show problem distribution
        difficulty_count = {}
        for p in problem_data:
            diff = p.metadata['difficulty']
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        for diff, count in difficulty_count.items():
            print(f"   {diff}: {count} problems")
        
        # Show sample problems
        print(f"\n📝 Sample problems:")
        for i, sample in enumerate(random.sample(problem_data, 2)):
            print(f"   {i+1}. {sample.metadata['question']} (Answer: {sample.metadata['correct_answer']})")
        
        print(f"\n🚀 Starting GPU training for {TRAIN_STEPS} steps...")
        print("Expected behavior:")
        print("   - Unsloth will patch model layers")
        print("   - vLLM will start inference server") 
        print("   - GPU memory will be allocated")
        print("   - Training metrics will be displayed")
        
        # Training loop with better error handling
        for step in range(current_step, current_step + TRAIN_STEPS):
            print(f"\n🏋️ Training Step {step + 1}/{current_step + TRAIN_STEPS}")
            print("─" * 40)
            
            try:
                # Generate trajectories for this step
                print("   📊 Generating trajectories...")
                trajectories = await gpu_rollout(model, problem_data, step)
                print(f"   ✅ Generated {len(trajectories)} trajectories")
                
                # Create trajectory groups
                trajectory_groups = [TrajectoryGroup(trajectories=trajectories)]
                print(f"   📦 Created {len(trajectory_groups)} trajectory groups")
                
                # Apply RULER scoring if available
                if USE_RULER:
                    print("   ⚖️ Applying RULER scoring with GPT-4o...")
                    try:
                        from art.rewards import ruler_score_group
                        scored_groups = []
                        
                        for group in trajectory_groups:
                            scored_group = await ruler_score_group(
                                group,
                                "gpt-4o-mini",  # Faster for training
                                debug=False,
                                swallow_exceptions=True
                            )
                            if scored_group is not None:
                                scored_groups.append(scored_group)
                                # Show average score
                                avg_score = sum(t.reward for t in scored_group.trajectories) / len(scored_group.trajectories)
                                print(f"   📊 Average RULER score: {avg_score:.3f}")
                            else:
                                print("   ⚠️ RULER scoring failed, using original rewards")
                                scored_groups.append(group)
                        
                        trajectory_groups = scored_groups
                        
                    except Exception as e:
                        print(f"   ⚠️ RULER scoring error: {e}")
                        print("   📝 Continuing with original rewards")
                else:
                    print("   📊 Using original reward scores (no RULER)")
                
                # Train the model
                print("   🏋️ Training model on GPU...")
                train_config = art.TrainConfig(learning_rate=2e-5)  # Good for GPU
                
                training_metrics = []
                async for metrics in model.train(trajectory_groups, config=train_config):
                    training_metrics.append(metrics)
                    
                    # Display key metrics
                    if 'loss' in metrics:
                        print(f"      📉 Loss: {metrics['loss']:.4f}")
                    if 'learning_rate' in metrics:
                        print(f"      📈 LR: {metrics['learning_rate']:.2e}")
                    if 'step' in metrics:
                        print(f"      👣 Step: {metrics['step']}")
                
                print(f"   ✅ Step {step + 1} completed successfully!")
                
                # Display GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"   🔥 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                
            except Exception as e:
                print(f"   ❌ Training step failed: {e}")
                print(f"   🔍 Error type: {type(e).__name__}")
                
                # Try to continue with next step
                print("   🔄 Attempting to continue with next step...")
                continue
        
        final_step = await model.get_step()
        print(f"\n🎉 GPU Training completed!")
        print(f"📈 Final step: {final_step}")
        print(f"🏋️ Steps trained: {final_step - current_step}")
        
        # Final GPU memory check
        if torch.cuda.is_available():
            print(f"🔥 Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        print(f"\n💡 Model saved to: ./.art/{PROJECT}/{MODEL_NAME}/")
        print("🚀 You can now use this trained model for inference!")
        
    except Exception as e:
        print(f"\n❌ Critical error during GPU training: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        print("\n🛠️ Troubleshooting steps:")
        print("   1. Check GPU memory: nvidia-smi")
        print("   2. Verify CUDA installation: nvidia-smi")
        print("   3. Check vLLM/PEFT version compatibility")
        print("   4. Try smaller batch size or model")
        raise
    
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
        
        # Cleanup backend
        if 'backend' in locals() and backend is not None:
            try:
                backend.close()
                print("🔧 Backend closed successfully")
            except Exception as e:
                print(f"⚠️ Warning: Error during backend cleanup: {e}")


if __name__ == "__main__":
    print("🚀 Starting GPU-optimized training...")
    asyncio.run(main())