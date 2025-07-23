#!/usr/bin/env python3
"""
Minimal installation that just gets ART working for training.
Skips HTTP API to avoid Pydantic conflicts.
"""

import subprocess
import sys

def install_minimal_art():
    """Install minimal ART dependencies without HTTP API conflicts."""
    print("⚡ Minimal ART Installation (Training Only)")
    print("=" * 50)
    print("This skips HTTP API to avoid Pydantic version conflicts")
    print("=" * 50)
    
    # Core packages needed for ART training only
    packages = [
        # PyTorch ecosystem
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tokenizers>=0.13.0",
        
        # Training frameworks
        "trl==0.7.4",
        "peft==0.6.2",
        
        # ART system deps
        "setproctitle>=1.3.0",
        "multiprocess>=0.70.0",
        "tblib>=1.7.0",
        "cloudpickle>=2.0.0",
        "dill>=0.3.0",
        "polars>=0.20.0",
        "weave>=0.50.0",
        "litellm>=1.0.0",
        
        # Basic utilities
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        
        # Minimal Pydantic (for ART only)
        "pydantic==1.10.13",
        
        # ART framework
        "openpipe-art>=0.1.0",
    ]
    
    print("📦 Installing packages one by one...")
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Installing {package.split('>=')[0].split('==')[0]}...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   ✅ Success")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Failed (not critical)")
    
    # Test ART specifically
    print("\n🧪 Testing ART framework...")
    try:
        import art
        from art.local import LocalBackend
        print("   ✅ ART framework working!")
        
        # Test model creation
        model = art.TrainableModel(
            name="test",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct"
        )
        print("   ✅ Model creation working!")
        return True
        
    except Exception as e:
        print(f"   ❌ ART test failed: {e}")
        return False

def main():
    if install_minimal_art():
        print("\n🎉 SUCCESS! ART framework is ready for training!")
        print("\n🚀 You can now run:")
        print("   python train_simple_cpu.py  # CPU training")
        print("   python train_simple.py      # GPU training (if GPU works)")
        print("\n💡 Note: HTTP API is not available in minimal install")
        print("   Use the training scripts directly instead")
    else:
        print("\n❌ Minimal installation failed")
        print("💡 Try individual package installation or check error messages")

if __name__ == "__main__":
    main()