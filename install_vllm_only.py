#!/usr/bin/env python3
"""
Quick installer for just vLLM - the immediate fix needed.
"""

import subprocess
import sys

def install_vllm():
    """Install vLLM with multiple fallback methods."""
    print("⚡ Installing vLLM inference engine...")
    
    methods = [
        ("pip install vllm>=0.2.0", "Standard installation"),
        ("pip install vllm --pre", "Pre-release version"),
        ("pip install vllm --upgrade --force-reinstall", "Force reinstall"),
        ("pip install vllm==0.2.7", "Specific stable version"),
    ]
    
    for cmd, desc in methods:
        print(f"\n🔧 Trying: {desc}")
        print(f"   Command: {cmd}")
        
        try:
            subprocess.check_call(cmd.split(), stdout=subprocess.DEVNULL)
            print("   ✅ Success!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    print("\n⚠️ All installation methods failed!")
    return False

def test_vllm():
    """Test vLLM import."""
    try:
        import vllm
        print(f"✅ vLLM {vllm.__version__} imported successfully!")
        
        # Test specific import that was failing
        from vllm import AsyncEngineArgs
        print("✅ AsyncEngineArgs import working!")
        return True
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        return False

def main():
    print("🚀 vLLM Quick Installer")
    print("=" * 30)
    
    if install_vllm():
        if test_vllm():
            print("\n🎉 SUCCESS! vLLM is working!")
            print("\n🚀 You can now run:")
            print("   python train_simple.py")
        else:
            print("\n⚠️ Installation succeeded but import failed.")
            print("Try restarting your Python session.")
    else:
        print("\n❌ Installation failed.")
        print("💡 Try the complete installer:")
        print("   python install_complete.py")

if __name__ == "__main__":
    main()