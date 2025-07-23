#!/usr/bin/env python3
"""
Fix version conflicts between vLLM, PEFT, and other LoRA libraries.
This addresses the 'LoRARequest' object has no attribute 'lora_tensors' error.
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
            print(f"   ❌ Failed: {result.stderr.strip()[:100]}...")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def fix_version_conflicts():
    """Fix version conflicts by installing compatible versions."""
    print("🔧 Fixing version conflicts in LoRA/vLLM ecosystem...")
    
    # Step 1: Uninstall conflicting packages
    print("\n📦 Step 1: Removing conflicting packages...")
    packages_to_remove = ["vllm", "peft", "trl"]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"Removing {package}")
    
    # Step 2: Install compatible versions in specific order
    print("\n📦 Step 2: Installing compatible versions...")
    
    compatible_installs = [
        ("pip install peft==0.6.2", "Installing compatible PEFT"),
        ("pip install trl==0.7.4", "Installing compatible TRL"),
        ("pip install vllm==0.2.7", "Installing stable vLLM"),
    ]
    
    for cmd, desc in compatible_installs:
        if not run_command(cmd, desc):
            # Try alternative versions
            if "vllm" in cmd:
                print("   🔄 Trying alternative vLLM version...")
                run_command("pip install vllm==0.2.6", "Installing vLLM 0.2.6")
            elif "peft" in cmd:
                print("   🔄 Trying alternative PEFT version...")
                run_command("pip install peft==0.5.0", "Installing PEFT 0.5.0")
    
    # Step 3: Reinstall ART framework
    print("\n📦 Step 3: Reinstalling ART framework...")
    run_command("pip install --upgrade --force-reinstall openpipe-art", "Reinstalling ART")
    
    return True

def test_imports():
    """Test if the key imports work without the error."""
    print("\n🧪 Testing imports...")
    
    imports_to_test = [
        ("import torch", "PyTorch"),
        ("import vllm", "vLLM"),
        ("import peft", "PEFT"),
        ("import trl", "TRL"),
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
    
    return all_good

def test_art_backend():
    """Test the specific ART backend that was failing."""
    print("\n🏗️ Testing ART backend initialization...")
    
    try:
        from art.local import LocalBackend
        backend = LocalBackend()
        print("   ✅ LocalBackend created successfully")
        
        # Try to create a simple model
        import art
        model = art.TrainableModel(
            name="test-model",
            project="version-test", 
            base_model="Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
        )
        print("   ✅ Model creation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ART backend test failed: {e}")
        return False

def main():
    print("🔧 Version Conflict Fixer for ART Framework")
    print("=" * 50)
    print("This fixes the 'LoRARequest' object has no attribute 'lora_tensors' error")
    print("=" * 50)
    
    # Fix version conflicts
    if fix_version_conflicts():
        print("\n✅ Version conflict fixes applied")
        
        # Test imports
        if test_imports():
            print("\n✅ All imports working")
            
            # Test ART backend specifically
            if test_art_backend():
                print("\n🎉 SUCCESS! ART framework is working!")
                print("\n🚀 You can now run:")
                print("   python train_simple.py")
            else:
                print("\n⚠️ ART backend still has issues")
                print("💡 Try using a smaller model or CPU-only mode:")
                print("   CUDA_VISIBLE_DEVICES='' python train_simple.py")
        else:
            print("\n⚠️ Some imports still failing")
            print("💡 Try restarting your Python session")
    else:
        print("\n❌ Failed to fix version conflicts")
        print("💡 Try manual installation:")
        print("   pip install peft==0.6.2 trl==0.7.4 vllm==0.2.7")

if __name__ == "__main__":
    main()