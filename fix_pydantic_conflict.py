#!/usr/bin/env python3
"""
Fix Pydantic version conflicts between vLLM and FastAPI.
This creates a working environment by installing compatible versions.
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """Run a command and show result."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success")
            return True
        else:
            error_msg = result.stderr.strip()[:150]
            print(f"   ❌ Failed: {error_msg}...")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def fix_pydantic_conflicts():
    """Fix Pydantic version conflicts systematically."""
    print("🔧 Fixing Pydantic version conflicts...")
    print("This resolves conflicts between vLLM (needs Pydantic v1) and FastAPI (needs Pydantic v2)")
    
    # Step 1: Remove conflicting packages
    print("\n📦 Step 1: Removing conflicting packages...")
    packages_to_remove = [
        "fastapi", "pydantic", "vllm", "openai", "httpx"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"Removing {package}")
    
    # Step 2: Install Pydantic v1 first (required by vLLM)
    print("\n📦 Step 2: Installing Pydantic v1...")
    run_command("pip install pydantic==1.10.13", "Installing Pydantic v1.10.13")
    
    # Step 3: Install vLLM with Pydantic v1
    print("\n📦 Step 3: Installing vLLM...")
    vllm_success = run_command("pip install vllm==0.2.7", "Installing vLLM 0.2.7")
    if not vllm_success:
        print("   🔄 Trying alternative vLLM version...")
        run_command("pip install vllm==0.2.6", "Installing vLLM 0.2.6")
    
    # Step 4: Install FastAPI compatible with Pydantic v1
    print("\n📦 Step 4: Installing FastAPI...")
    fastapi_success = run_command("pip install fastapi==0.68.2", "Installing FastAPI 0.68.2")
    if not fastapi_success:
        print("   🔄 Trying newer FastAPI with Pydantic v1...")
        run_command("pip install 'fastapi<0.100.0'", "Installing FastAPI <0.100")
    
    # Step 5: Install other service dependencies
    print("\n📦 Step 5: Installing service dependencies...")
    service_deps = [
        ("uvicorn[standard]>=0.24.0", "Installing Uvicorn"),
        ("httpx>=0.25.0", "Installing HTTPX"),
        ("openai>=1.0.0", "Installing OpenAI"),
        ("python-dotenv>=1.0.0", "Installing python-dotenv"),
    ]
    
    for package, desc in service_deps:
        run_command(f"pip install '{package}'", desc)
    
    return True

def create_alternative_requirements():
    """Create alternative requirements without version conflicts."""
    print("\n📝 Creating alternative requirements.txt...")
    
    alt_requirements = """# Alternative requirements without version conflicts
# Core ML dependencies (no conflicts)
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
tokenizers>=0.13.0
datasets>=2.12.0
huggingface-hub>=0.16.0
safetensors>=0.3.0

# Training frameworks (compatible versions)
trl==0.7.4
peft==0.6.2

# ART system dependencies
setproctitle>=1.3.0
multiprocess>=0.70.0
tblib>=1.7.0
cloudpickle>=2.0.0
dill>=0.3.0
polars>=0.20.0
weave>=0.50.0
litellm>=1.0.0

# Service dependencies (Pydantic v1 compatible)
pydantic==1.10.13
fastapi==0.68.2
uvicorn[standard]>=0.24.0
httpx>=0.25.0
openai>=1.0.0
python-dotenv>=1.0.0

# vLLM (requires Pydantic v1)
vllm==0.2.7

# Utility packages
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
wandb>=0.15.0
psutil>=5.9.0

# ART framework
openpipe-art>=0.1.0
"""
    
    with open("requirements_alt.txt", "w") as f:
        f.write(alt_requirements)
    
    print("   ✅ Created requirements_alt.txt")
    return True

def test_compatibility():
    """Test if the fixed versions work together."""
    print("\n🧪 Testing version compatibility...")
    
    imports_to_test = [
        ("import pydantic", "Pydantic"),
        ("import fastapi", "FastAPI"),
        ("import vllm", "vLLM"),
        ("import openai", "OpenAI"),
        ("import art", "ART Framework"),
    ]
    
    all_good = True
    for import_cmd, name in imports_to_test:
        try:
            exec(import_cmd)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_good = False
    
    # Test version compatibility specifically
    try:
        import pydantic
        import fastapi
        print(f"   📋 Pydantic version: {pydantic.VERSION}")
        print(f"   📋 FastAPI version: {fastapi.__version__}")
        
        # Test that they work together
        from fastapi import FastAPI
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class TestModel(BaseModel):
            name: str
        
        print("   ✅ Pydantic + FastAPI compatibility confirmed")
        
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        all_good = False
    
    return all_good

def main():
    print("🔧 Pydantic Conflict Resolver")
    print("=" * 50)
    print("vLLM needs Pydantic v1, but newer FastAPI needs Pydantic v2")
    print("This script installs compatible versions that work together")
    print("=" * 50)
    
    # Fix conflicts
    fix_pydantic_conflicts()
    
    # Create alternative requirements
    create_alternative_requirements()
    
    # Test compatibility
    if test_compatibility():
        print("\n🎉 SUCCESS! All packages are compatible!")
        print("\n🚀 You can now run:")
        print("   python train_simple.py")
        print("   python train_simple_cpu.py")
        print("\n💡 Alternative installation:")
        print("   pip install -r requirements_alt.txt")
    else:
        print("\n⚠️ Some compatibility issues remain")
        print("\n💡 Try the CPU-only version:")
        print("   python train_simple_cpu.py")
        print("\n🔧 Or install from alternative requirements:")
        print("   pip install -r requirements_alt.txt")

if __name__ == "__main__":
    main()