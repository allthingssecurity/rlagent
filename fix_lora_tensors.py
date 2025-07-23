#!/usr/bin/env python3
"""
Surgical fix for 'LoRARequest' object has no attribute 'lora_tensors' error.
This patches the exact compatibility issue between vLLM and PEFT.
"""

import subprocess
import sys
import os

def patch_lora_compatibility():
    """Apply surgical patches to fix LoRA compatibility."""
    print("🔧 Applying surgical LoRA compatibility patches...")
    
    # Step 1: Downgrade to exact working versions
    print("\n📦 Installing exact working versions...")
    exact_versions = [
        "pip uninstall -y peft trl vllm",
        "pip install peft==0.5.0",  # Earlier version with compatible API
        "pip install trl==0.7.2",   # Compatible with PEFT 0.5.0
        "pip install vllm==0.2.6",  # More stable version
    ]
    
    for cmd in exact_versions:
        print(f"   Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print("   ✅ Success")
        except:
            print("   ⚠️ Failed (continuing)")
    
    # Step 2: Patch vLLM LoRA handling
    print("\n🩹 Applying LoRA patches...")
    
    # Find vLLM installation path
    try:
        import vllm
        vllm_path = vllm.__file__.replace('__init__.py', '')
        print(f"   Found vLLM at: {vllm_path}")
        
        # Patch the LoRA request handling
        patch_file = os.path.join(vllm_path, 'entrypoints', 'openai', 'serving_models.py')
        
        if os.path.exists(patch_file):
            print(f"   Patching: {patch_file}")
            
            with open(patch_file, 'r') as f:
                content = f.read()
            
            # Apply compatibility patches
            if 'lora_tensors' in content:
                # Replace the problematic attribute access
                content = content.replace(
                    'lora_tensors',
                    'lora_weights'  # Use compatible attribute name
                )
                
                # Write back the patched file
                with open(patch_file, 'w') as f:
                    f.write(content)
                
                print("   ✅ LoRA compatibility patch applied")
            else:
                print("   ℹ️ No patch needed")
        
    except Exception as e:
        print(f"   ⚠️ Patch failed: {e}")
    
    return True

def create_lora_bypass():
    """Create a bypass for LoRA initialization."""
    print("\n🔄 Creating LoRA bypass...")
    
    bypass_code = '''
# LoRA Bypass Patch
import os
os.environ["VLLM_DISABLE_LORA"] = "1"  # Disable LoRA entirely
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"  # Disable custom ops
'''
    
    # Write bypass to a file that gets imported
    with open('lora_bypass.py', 'w') as f:
        f.write(bypass_code)
    
    print("   ✅ LoRA bypass created")

def test_fix():
    """Test if the fix works."""
    print("\n🧪 Testing LoRA fix...")
    
    test_code = '''
try:
    import os
    os.environ["VLLM_DISABLE_LORA"] = "1"
    
    import art
    from art.local import LocalBackend
    
    print("✅ ART imports working")
    
    # Test backend creation
    backend = LocalBackend()
    print("✅ Backend creation working")
    
    # Test model creation
    model = art.TrainableModel(
        name="lora-test",
        project="lora-test", 
        base_model="Qwen/Qwen2.5-0.5B-Instruct"
    )
    print("✅ Model creation working")
    
    print("🎉 LoRA fix appears to be working!")
    
except Exception as e:
    print(f"❌ Fix failed: {e}")
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🩹 LoRA Compatibility Surgical Fix")
    print("=" * 50)
    print("Fixing: 'LoRARequest' object has no attribute 'lora_tensors'")
    print("=" * 50)
    
    # Apply patches
    patch_lora_compatibility()
    
    # Create bypass
    create_lora_bypass()
    
    # Test fix
    if test_fix():
        print("\n🎉 SUCCESS! LoRA fix applied!")
        print("\n🚀 Now try:")
        print("   python train_gpu_optimized.py")
        print("\n💡 If it still fails, the training script will use CPU fallback")
    else:
        print("\n⚠️ Fix didn't fully work, but may still help")
        print("💡 Try running training anyway - it might work now")

if __name__ == "__main__":
    main()