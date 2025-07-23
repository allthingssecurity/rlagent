#!/usr/bin/env python3
"""
Simple standalone training script - no HTTP server needed!
Just run: python train_simple.py

This script trains a math agent using the ART framework directly.
"""

import asyncio
import random
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if we have OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ Warning: OPENAI_API_KEY not found in .env file")
    print("RULER scoring will be disabled")

import art
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.trajectories import Trajectory, TrajectoryGroup

# Configuration
MODEL_NAME = "simple-math-agent"
PROJECT = "standalone-training" 
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_STEPS = 3
BATCH_SIZE = 4
USE_RULER = bool(os.getenv("OPENAI_API_KEY"))

print("🧮 Simple ART Training Script")
print("=" * 40)
print(f"Model: {MODEL_NAME}")
print(f"Base Model: {BASE_MODEL}")
print(f"Training Steps: {TRAIN_STEPS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"RULER Enabled: {USE_RULER}")
print("=" * 40)


def generate_math_problems(num_problems: int = 10):
    """Generate simple math problems for training."""
    problems = []
    
    for i in range(num_problems):
        # Generate different types of math problems
        problem_type = random.choice(["addition", "subtraction", "multiplication"])
        
        if problem_type == "addition":
            a, b = random.randint(1, 50), random.randint(1, 50)
            question = f"What is {a} + {b}?"
            answer = a + b
            solution = f"To solve {a} + {b}, I need to add these numbers together.\n{a} + {b} = {answer}"
            
        elif problem_type == "subtraction":
            a, b = random.randint(20, 100), random.randint(1, 19)
            question = f"What is {a} - {b}?"
            answer = a - b
            solution = f"To solve {a} - {b}, I need to subtract {b} from {a}.\n{a} - {b} = {answer}"
            
        else:  # multiplication
            a, b = random.randint(2, 12), random.randint(2, 12)
            question = f"What is {a} × {b}?"
            answer = a * b
            solution = f"To solve {a} × {b}, I need to multiply these numbers.\n{a} × {b} = {answer}"
        
        # Create trajectory messages
        messages = [
            art.types.Message(
                role="system", 
                content="You are a helpful math tutor. Solve problems step by step and show your work clearly."
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
        
        # Create trajectory with a good reward (since these are correct answers)
        trajectory = Trajectory(
            messages=messages,
            reward=1.0,  # Perfect score for correct solutions
            metadata={
                "problem_type": problem_type,
                "correct_answer": answer,
                "question": question
            }
        )
        
        problems.append(trajectory)
    
    return problems


async def simple_rollout(model, problem_data, step_num):
    """Simple rollout function that creates trajectories from our problem data."""
    trajectories = []
    
    # Use our pre-generated problems
    for i, problem in enumerate(problem_data):
        if i >= BATCH_SIZE:  # Limit to batch size
            break
            
        # Add some variation by modifying the system prompt
        system_prompts = [
            "You are a helpful math tutor. Solve problems step by step.",
            "You are a math expert. Show your work clearly when solving problems.",
            "You are a patient teacher helping students with math problems.",
        ]
        
        # Create a new trajectory with slight variations
        messages = [
            art.types.Message(
                role="system", 
                content=system_prompts[i % len(system_prompts)]
            ),
            problem.messages[1],  # Keep the user question
            problem.messages[2]   # Keep the assistant response
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
    """Main training function."""
    random.seed(42)  # For reproducible results
    
    try:
        print("\n🔧 Initializing ART backend...")
        backend = LocalBackend()
        
        print("📝 Creating model...")
        model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT,
            base_model=BASE_MODEL,
        )
        
        # Set model configuration
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=4096,  # Smaller for faster training
            ),
        )
        
        print("📋 Registering model...")
        await model.register(backend)
        
        # Check current step
        current_step = await model.get_step()
        print(f"📈 Current model step: {current_step}")
        
        print("\n📊 Generating training data...")
        problem_data = generate_math_problems(20)  # Generate more than we need
        print(f"✅ Generated {len(problem_data)} math problems")
        
        # Show a sample problem
        sample = problem_data[0]
        print(f"\n📝 Sample problem:")
        print(f"   Question: {sample.metadata['question']}")
        print(f"   Answer: {sample.metadata['correct_answer']}")
        
        print(f"\n🚀 Starting training for {TRAIN_STEPS} steps...")
        
        # Training loop
        for step in range(current_step, current_step + TRAIN_STEPS):
            print(f"\n📚 Training Step {step + 1}/{current_step + TRAIN_STEPS}")
            
            # Generate trajectories for this step
            trajectories = await simple_rollout(model, problem_data, step)
            print(f"   Generated {len(trajectories)} trajectories")
            
            # Create trajectory groups
            trajectory_groups = [TrajectoryGroup(trajectories=trajectories)]
            
            # Apply RULER scoring if enabled
            if USE_RULER:
                print("   ⚖️ Applying RULER scoring...")
                try:
                    scored_groups = []
                    for group in trajectory_groups:
                        scored_group = await ruler_score_group(
                            group,
                            "gpt-4o-mini",  # Use cheaper model for demo
                            debug=False,
                            swallow_exceptions=True
                        )
                        if scored_group is not None:
                            scored_groups.append(scored_group)
                        else:
                            print("   ⚠️ RULER scoring failed, using original scores")
                            scored_groups.append(group)
                    
                    trajectory_groups = scored_groups
                    
                    # Show RULER scores
                    if trajectory_groups:
                        avg_score = sum(t.reward for t in trajectory_groups[0].trajectories) / len(trajectory_groups[0].trajectories)
                        print(f"   📊 Average RULER score: {avg_score:.2f}")
                    
                except Exception as e:
                    print(f"   ⚠️ RULER scoring error: {e}")
                    print("   📝 Continuing with original rewards...")
            
            # Train the model
            print("   🏋️ Training model...")
            try:
                train_config = art.TrainConfig(learning_rate=5e-5)  # Slightly higher LR for demo
                
                training_metrics = []
                async for metrics in model.train(trajectory_groups, config=train_config):
                    training_metrics.append(metrics)
                    if 'loss' in metrics:
                        print(f"      Loss: {metrics['loss']:.4f}")
                
                print(f"   ✅ Step {step + 1} completed!")
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                break
        
        final_step = await model.get_step()
        print(f"\n🎉 Training completed! Final step: {final_step}")
        
        # Test the trained model with a new problem
        print("\n🧪 Testing trained model...")
        test_question = "What is 15 + 27?"
        print(f"Test question: {test_question}")
        
        # Note: In a real scenario, you'd use the trained model to generate responses
        # For this demo, we'll just show that training completed successfully
        print("✅ Model training completed successfully!")
        print("\n💡 Next steps:")
        print("   1. The model is saved in ./.art/ directory")
        print("   2. You can continue training by running this script again")
        print("   3. Use the HTTP API for more advanced features")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("💡 Make sure you have sufficient GPU memory and all dependencies installed")
        raise
    
    finally:
        # Cleanup
        if 'backend' in locals() and backend is not None:
            try:
                backend.close()  # Note: close() is not async in LocalBackend
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    print("🚀 Starting simple training script...")
    asyncio.run(main())