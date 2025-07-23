"""Example: Training a simple math problem-solving agent."""

import asyncio
import random
from typing import List
import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
MODEL_NAME = "math-solver-v1"
PROJECT = "math-examples"


async def generate_math_problems(num_problems: int = 20) -> List[dict]:
    """Generate simple math problems for training."""
    problems = []
    
    for i in range(num_problems):
        # Generate different types of math problems
        problem_type = random.choice(["addition", "subtraction", "multiplication", "word_problem"])
        
        if problem_type == "addition":
            a, b = random.randint(1, 100), random.randint(1, 100)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            
        elif problem_type == "subtraction":
            a, b = random.randint(10, 100), random.randint(1, 50)
            question = f"What is {a} - {b}?"
            answer = str(a - b)
            
        elif problem_type == "multiplication":
            a, b = random.randint(1, 12), random.randint(1, 12)
            question = f"What is {a} × {b}?"
            answer = str(a * b)
            
        else:  # word_problem
            apples = random.randint(5, 20)
            eaten = random.randint(1, apples - 1)
            question = f"Sarah has {apples} apples. She eats {eaten} of them. How many apples does she have left?"
            answer = str(apples - eaten)
        
        # Create a trajectory for this problem
        trajectory = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a math tutor. Solve math problems step by step and provide clear explanations."
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": f"Let me solve this step by step.\n\n{question}\n\nThe answer is {answer}."
                }
            ],
            "reward": 1.0,  # Perfect score for correct answers
            "metadata": {
                "problem_type": problem_type,
                "correct_answer": answer,
                "difficulty": "easy"
            }
        }
        
        problems.append(trajectory)
    
    return problems


async def train_math_agent():
    """Train a math-solving agent using the HTTP API."""
    async with httpx.AsyncClient() as client:
        
        print("🔍 Checking service health...")
        health_response = await client.get(f"{API_BASE_URL}/health")
        if health_response.status_code != 200:
            print("❌ Service is not healthy!")
            return
        
        health_data = health_response.json()
        print(f"✅ Service is healthy! Version: {health_data['version']}")
        
        print("\n📊 Generating training data...")
        training_trajectories = await generate_math_problems(15)
        print(f"Generated {len(training_trajectories)} training examples")
        
        print("\n🚀 Starting training...")
        training_request = {
            "model_name": MODEL_NAME,
            "project": PROJECT,
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "num_steps": 5,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "use_ruler": True,
            "judge_model": "gpt-4o",
            "task_prompt": "You are a math tutor. Solve math problems step by step and provide clear explanations.",
            "task_type": "math_solving",
            "trajectories": training_trajectories
        }
        
        train_response = await client.post(f"{API_BASE_URL}/train", json=training_request)
        
        if train_response.status_code != 200:
            print(f"❌ Training failed: {train_response.text}")
            return
        
        train_data = train_response.json()
        print(f"✅ Training started! Status: {train_data['status']}")
        print(f"📈 Current step: {train_data['current_step']}")
        
        print("\n📊 Monitoring training progress...")
        
        # Poll for training status
        while True:
            status_response = await client.get(
                f"{API_BASE_URL}/train/{PROJECT}/{MODEL_NAME}/status"
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']} | Step: {status_data['current_step']}")
                
                if status_data['status'] in ['success', 'error']:
                    break
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        print("\n🎯 Training completed!")
        
        print("\n📋 Listing all models...")
        models_response = await client.get(f"{API_BASE_URL}/models")
        if models_response.status_code == 200:
            models = models_response.json()
            for model in models:
                print(f"  - {model['project']}/{model['name']} (step {model['current_step']})")


async def test_rollout_generation():
    """Test the rollout generation endpoint."""
    async with httpx.AsyncClient() as client:
        
        print("\n🎲 Testing rollout generation...")
        
        rollout_request = {
            "model_name": "test-model",
            "project": "test-project",
            "num_rollouts": 3,
            "task_prompt": "You are a math tutor. Help students solve math problems step by step.",
            "max_turns": 5,
            "temperature": 0.7
        }
        
        rollout_response = await client.post(f"{API_BASE_URL}/rollout", json=rollout_request)
        
        if rollout_response.status_code == 200:
            rollout_data = rollout_response.json()
            print(f"✅ Generated {len(rollout_data['trajectories'])} rollouts")
            
            # Show first rollout as example
            if rollout_data['trajectories']:
                first_trajectory = rollout_data['trajectories'][0]
                print(f"\n📝 Example generated trajectory:")
                for msg in first_trajectory['messages'][:3]:  # Show first 3 messages
                    print(f"  {msg['role']}: {msg['content'][:100]}...")
        else:
            print(f"❌ Rollout generation failed: {rollout_response.text}")


async def test_evaluation():
    """Test the evaluation endpoint."""
    async with httpx.AsyncClient() as client:
        
        print("\n⚖️ Testing trajectory evaluation...")
        
        # Create some test trajectories
        test_trajectories = [
            {
                "messages": [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 5 + 3?"},
                    {"role": "assistant", "content": "5 + 3 = 8"}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 10 - 4?"},
                    {"role": "assistant", "content": "I don't know."}
                ]
            }
        ]
        
        eval_request = {
            "trajectories": test_trajectories,
            "judge_model": "gpt-4o",
            "evaluation_prompt": "Rate how well the assistant solved the math problem. Scale 0.0-1.0.",
            "scoring_criteria": "Correctness and clarity of the mathematical solution."
        }
        
        eval_response = await client.post(f"{API_BASE_URL}/evaluate", json=eval_request)
        
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            print(f"✅ Evaluation completed!")
            print(f"📊 Scores: {eval_data['scores']}")
            print(f"📈 Average score: {eval_data['average_score']:.2f}")
        else:
            print(f"❌ Evaluation failed: {eval_response.text}")


async def main():
    """Run the complete example."""
    print("🧮 Math Agent Training Example")
    print("=" * 40)
    
    # Test individual components first
    await test_rollout_generation()
    await test_evaluation()
    
    # Run the full training example
    await train_math_agent()


if __name__ == "__main__":
    asyncio.run(main())