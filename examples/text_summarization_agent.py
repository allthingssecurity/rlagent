"""Example: Training a text summarization agent using OpenAI for data generation."""

import asyncio
import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
MODEL_NAME = "summarizer-v1"
PROJECT = "text-summarization"


async def train_summarization_agent():
    """Train a text summarization agent using GPT-4o generated data."""
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
        
        print("📰 Training Text Summarization Agent")
        print("=" * 40)
        
        print("🔍 Checking service health...")
        health_response = await client.get(f"{API_BASE_URL}/health")
        health_data = health_response.json()
        print(f"✅ Service is healthy! OpenAI configured: {health_data['openai_configured']}")
        
        print("\n🚀 Starting training with automatic data generation...")
        
        training_request = {
            "model_name": MODEL_NAME,
            "project": PROJECT,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "num_steps": 3,
            "batch_size": 6,
            "learning_rate": 2e-5,
            "use_ruler": True,
            "judge_model": "gpt-4o",
            "task_prompt": """You are an expert text summarizer. Your task is to:
1. Read the given text carefully
2. Identify the key points and main ideas
3. Create a concise, accurate summary that captures the essence
4. Ensure the summary is about 1/3 the length of the original text
5. Maintain the original tone and important details""",
            "task_type": "text_summarization"
            # No trajectories provided - will use automatic generation
        }
        
        train_response = await client.post(f"{API_BASE_URL}/train", json=training_request)
        
        if train_response.status_code != 200:
            print(f"❌ Training failed: {train_response.text}")
            return
        
        train_data = train_response.json()
        print(f"✅ Training started! Status: {train_data['status']}")
        
        print("\n📊 Monitoring training progress...")
        
        # Monitor training progress
        step_count = 0
        while True:
            status_response = await client.get(
                f"{API_BASE_URL}/train/{PROJECT}/{MODEL_NAME}/status"
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                if status_data['current_step'] != step_count:
                    step_count = status_data['current_step']
                    print(f"📈 Step {step_count} completed")
                    
                    if 'training_metrics' in status_data and status_data['training_metrics']:
                        metrics = status_data['training_metrics']
                        if 'loss' in metrics:
                            print(f"   Loss: {metrics['loss']:.4f}")
                
                if status_data['status'] in ['success', 'error']:
                    print(f"\n🎯 Training {status_data['status']}!")
                    if status_data['status'] == 'error':
                        print(f"Error: {status_data.get('error_details', 'Unknown error')}")
                    break
            
            await asyncio.sleep(10)  # Check every 10 seconds


async def generate_summarization_data():
    """Generate training data specifically for summarization tasks."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        print("\n📊 Generating summarization training data...")
        
        rollout_request = {
            "model_name": "data-generator",
            "project": "data-generation",
            "num_rollouts": 8,
            "task_prompt": """Create diverse text summarization examples. Generate:
1. A substantial piece of text (news article, blog post, research abstract, etc.)
2. An expert-quality summary of that text

Topics should vary: technology, science, business, health, environment, etc.
Text lengths should vary: 200-800 words for source, 50-200 words for summary.

Format as a conversation where the user provides text and asks for a summary.""",
            "max_turns": 4,
            "temperature": 0.8
        }
        
        rollout_response = await client.post(f"{API_BASE_URL}/rollout", json=rollout_request)
        
        if rollout_response.status_code == 200:
            rollout_data = rollout_response.json()
            print(f"✅ Generated {len(rollout_data['trajectories'])} training examples")
            
            # Show example
            if rollout_data['trajectories']:
                example = rollout_data['trajectories'][0]
                print(f"\n📝 Example generated data:")
                for i, msg in enumerate(example['messages'][:2]):  # Show first 2 messages
                    content_preview = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                    print(f"  {msg['role']}: {content_preview}")
                
                print(f"🏆 Auto-evaluated score: {example.get('reward', 'N/A')}")
            
            return rollout_data['trajectories']
        else:
            print(f"❌ Data generation failed: {rollout_response.text}")
            return []


async def evaluate_summaries():
    """Test the evaluation system with sample summaries."""
    async with httpx.AsyncClient() as client:
        
        print("\n⚖️ Testing summary evaluation...")
        
        test_trajectories = [
            {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert text summarizer."
                    },
                    {
                        "role": "user", 
                        "content": "Please summarize this text: The rapid advancement of artificial intelligence has transformed various industries. From healthcare to finance, AI applications are becoming increasingly sophisticated. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. However, concerns about job displacement and ethical considerations remain significant challenges that society must address."
                    },
                    {
                        "role": "assistant", 
                        "content": "AI is rapidly advancing across industries like healthcare and finance, with ML algorithms processing data for pattern recognition and predictions. However, job displacement and ethical concerns remain key challenges for society."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert text summarizer."
                    },
                    {
                        "role": "user", 
                        "content": "Summarize this article about climate change and its effects on polar ice caps."
                    },
                    {
                        "role": "assistant", 
                        "content": "Climate change is bad and ice is melting."
                    }
                ]
            }
        ]
        
        eval_request = {
            "trajectories": test_trajectories,
            "judge_model": "gpt-4o",
            "evaluation_prompt": """Evaluate this text summarization based on:
1. Accuracy: Does the summary capture the key points?
2. Conciseness: Is it appropriately condensed?
3. Clarity: Is the summary clear and well-written?
4. Completeness: Are important details preserved?

Rate from 0.0 (poor) to 1.0 (excellent).""",
            "scoring_criteria": "Accuracy, conciseness, clarity, and completeness of the summary."
        }
        
        eval_response = await client.post(f"{API_BASE_URL}/evaluate", json=eval_request)
        
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            print(f"✅ Evaluation completed!")
            print(f"📊 Scores: {[f'{score:.2f}' for score in eval_data['scores']]}")
            print(f"📈 Average score: {eval_data['average_score']:.2f}")
        else:
            print(f"❌ Evaluation failed: {eval_response.text}")


async def train_with_custom_data():
    """Train using custom generated data instead of automatic generation."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        
        print("\n🎯 Training with custom generated data...")
        
        # First generate high-quality training data
        training_data = await generate_summarization_data()
        
        if not training_data:
            print("❌ No training data generated, skipping custom training")
            return
        
        training_request = {
            "model_name": f"{MODEL_NAME}-custom",
            "project": PROJECT,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "num_steps": 2,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "use_ruler": True,
            "judge_model": "gpt-4o",
            "task_prompt": """You are an expert text summarizer. Create concise, accurate summaries that capture the key points and main ideas while maintaining the original tone.""",
            "task_type": "text_summarization",
            "trajectories": training_data  # Use custom generated data
        }
        
        train_response = await client.post(f"{API_BASE_URL}/train", json=training_request)
        
        if train_response.status_code == 200:
            print("✅ Custom data training started!")
            
            # Monitor progress
            while True:
                status_response = await client.get(
                    f"{API_BASE_URL}/train/{PROJECT}/{MODEL_NAME}-custom/status"
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"Status: {status_data['status']} | Step: {status_data['current_step']}")
                    
                    if status_data['status'] in ['success', 'error']:
                        break
                
                await asyncio.sleep(8)
        else:
            print(f"❌ Custom training failed: {train_response.text}")


async def main():
    """Run the complete summarization training example."""
    print("📰 Text Summarization Agent Training")
    print("=" * 50)
    
    # Test evaluation system
    await evaluate_summaries()
    
    # Train with automatic data generation
    await train_summarization_agent()
    
    # Train with custom generated data
    await train_with_custom_data()
    
    print("\n📋 Final model list:")
    async with httpx.AsyncClient() as client:
        models_response = await client.get(f"{API_BASE_URL}/models")
        if models_response.status_code == 200:
            models = models_response.json()
            for model in models:
                if model['project'] == PROJECT:
                    print(f"  ✅ {model['name']} - Step {model['current_step']}")


if __name__ == "__main__":
    asyncio.run(main())