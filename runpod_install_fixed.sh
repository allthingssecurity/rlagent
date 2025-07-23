#!/bin/bash
# RunPod GPU Installation Script - FIXED for OpenAI compatibility
# Run this on RunPod: bash runpod_install_fixed.sh

echo "🚀 RunPod GPU LoRA Installation (FIXED)"
echo "========================================"
echo "Using newer OpenAI library compatible with ART"
echo "Still avoiding weave/litellm pydantic conflicts"
echo "========================================"

# Update system
echo "📦 Updating system..."
apt-get update -qq

# Remove any conflicting packages
echo "🧹 Removing conflicting packages..."
pip uninstall -y weave litellm fastapi uvicorn httpx pydantic vllm peft trl transformers accelerate bitsandbytes torch openpipe-art openai 2>/dev/null || true

# Install exact working versions in order
echo "🔧 Installing PyTorch with CUDA..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo "🔧 Installing Transformers stack..."
pip install tokenizers==0.13.3
pip install transformers==4.33.2
pip install accelerate==0.21.0
pip install safetensors==0.3.2

echo "🔧 Installing LoRA stack..."
pip install peft==0.4.0
pip install trl==0.6.0
pip install bitsandbytes==0.41.1

echo "🔧 Installing vLLM with compatible Pydantic..."
pip install pydantic==1.10.13
pip install vllm==0.2.2

echo "🔧 Installing ART dependencies (no weave)..."
pip install setproctitle==1.3.3
pip install multiprocess==0.70.15
pip install tblib==2.0.0
pip install cloudpickle==3.0.0
pip install polars==0.18.15

echo "🔧 Installing NEW OpenAI (compatible with ART)..."
# ART needs the new OpenAI structure but we need to avoid conflicts
pip install openai==1.3.0  # Newer but not latest to avoid conflicts
pip install python-dotenv==1.0.0

echo "🔧 Installing ART framework..."
# Try with dependencies first, fallback to no-deps
pip install openpipe-art || pip install openpipe-art --no-deps

# Test installation
echo "🧪 Testing installation..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'✅ CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')

try:
    import openai
    print(f'✅ OpenAI {openai.__version__}')
    # Test the import that was failing
    from openai.types.chat.chat_completion import Choice
    print('✅ OpenAI types import works')
except Exception as e:
    print(f'❌ OpenAI: {e}')

try:
    import peft, trl, vllm
    print(f'✅ PEFT {peft.__version__}')
    print(f'✅ TRL {trl.__version__}')
    print(f'✅ vLLM {vllm.__version__}')
    
    # Test LoRA config
    from peft import LoraConfig
    config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
    print('✅ LoRA config works')
except Exception as e:
    print(f'❌ LoRA stack: {e}')

try:
    import art
    print('✅ ART imported successfully')
    
    # Test the specific import that was failing
    from art.types import Message
    print('✅ ART types import works')
    
    from art.local import LocalBackend
    print('✅ LocalBackend import works')
except Exception as e:
    print(f'❌ ART: {e}')

print('🎉 Installation test complete!')
"

echo ""
echo "🎉 RunPod installation complete!"
echo "Now run: python3 train_minimal_gpu.py"