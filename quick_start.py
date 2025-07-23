#!/usr/bin/env python3
"""
Super quick start script - minimal training example.
Perfect for testing that everything works on RunPod!

Just run: python quick_start.py
"""

import asyncio
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Simple test without heavy dependencies first
def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🖥️ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️ CUDA not available - will use CPU (slower)")
    except ImportError:
        print("   ❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not found")
        return False
    
    try:
        import art
        print("   ✅ ART framework available")
    except ImportError:
        print("   ❌ ART framework not found")
        return False
    
    return True


async def minimal_training_test():
    """Minimal training test with tiny model and small data."""
    print("\n🚀 Running minimal training test...")
    
    try:
        import art
        from art.local import LocalBackend
        from art.trajectories import Trajectory, TrajectoryGroup
        
        # Use a very small model for quick testing
        print("📝 Setting up tiny test model...")
        backend = LocalBackend()
        
        model = art.TrainableModel(
            name="test-tiny",
            project="quick-test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Smallest Qwen model
        )
        
        # Minimal config for speed
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=512,  # Very small context
            ),
        )
        
        print("📋 Registering model...")
        await model.register(backend)
        
        print("📊 Creating minimal training data...")
        # Just 2 simple examples
        trajectories = [
            Trajectory(
                messages=[
                    art.types.Message(role="system", content="You are helpful."),
                    art.types.Message(role="user", content="Say hello"),
                    art.types.Message(role="assistant", content="Hello! How can I help you?")
                ],
                reward=1.0,
                metadata={"example": 1}
            ),
            Trajectory(
                messages=[
                    art.types.Message(role="system", content="You are helpful."),
                    art.types.Message(role="user", content="What is 2+2?"),
                    art.types.Message(role="assistant", content="2+2 equals 4.")
                ],
                reward=1.0,
                metadata={"example": 2}
            )
        ]
        
        print(f"   ✅ Created {len(trajectories)} training examples")
        
        # Single training step
        print("🏋️ Running 1 training step...")
        trajectory_group = TrajectoryGroup(trajectories=trajectories)
        
        train_config = art.TrainConfig(learning_rate=1e-4)
        
        step_completed = False
        async for metrics in model.train([trajectory_group], config=train_config):
            print(f"   📊 Training metrics: {metrics}")
            step_completed = True
            break  # Just do one step
        
        if step_completed:
            final_step = await model.get_step()
            print(f"✅ Training completed! Model at step: {final_step}")
            return True
        else:
            print("❌ Training did not complete")
            return False
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    finally:
        if 'backend' in locals():
            await backend.close()


async def main():
    """Main function to run all tests."""
    print("⚡ Quick Start Test Script")
    print("=" * 30)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import test failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    print("\n✅ All imports successful!")
    
    # Test minimal training
    success = await minimal_training_test()
    
    if success:
        print("\n🎉 SUCCESS! Everything is working!")
        print("\n📋 Next steps:")
        print("   1. Run full training: python train_simple.py")
        print("   2. Start HTTP service: python run_server.py")
        print("   3. Check examples: python examples/simple_math_agent.py")
    else:
        print("\n❌ Training test failed.")
        print("💡 This might be due to:")
        print("   - Insufficient GPU memory")
        print("   - Missing dependencies")
        print("   - Network issues downloading models")
        print("\n🔧 Try:")
        print("   - Use smaller model or reduce batch size")
        print("   - Check GPU memory: nvidia-smi")
        print("   - Check internet connection")


if __name__ == "__main__":
    asyncio.run(main())