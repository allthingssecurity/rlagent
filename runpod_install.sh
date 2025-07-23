#!/bin/bash
# RunPod GPU Installation Script - Bypass all dependency conflicts
# Run this on RunPod: bash runpod_install.sh

echo "🚀 RunPod GPU LoRA Installation"
echo "================================="
echo "Bypassing all pydantic/weave conflicts"
echo "Installing only essential packages for GPU LoRA training"
echo "================================="

# Update system
echo "📦 Updating system..."
apt-get update -qq

# Remove any conflicting packages
echo "🧹 Removing conflicting packages..."
pip uninstall -y weave litellm fastapi uvicorn httpx pydantic vllm peft trl transformers accelerate bitsandbytes torch openpipe-art 2>/dev/null || true

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

echo "🔧 Installing old OpenAI (compatible)..."
pip install openai==0.28.1
pip install python-dotenv==1.0.0

echo "🔧 Installing ART framework (no dependencies)..."
pip install openpipe-art --no-deps

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')

import peft, trl, vllm, art
print(f'✅ PEFT {peft.__version__}')
print(f'✅ TRL {trl.__version__}')
print(f'✅ vLLM {vllm.__version__}')
print('✅ ART imported')

# Test LoRA config
from peft import LoraConfig
config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
print('✅ LoRA config works')

print('🎉 Installation successful!')
"

echo ""
echo "🎉 RunPod installation complete!"
echo "Now run: python3 train_minimal_gpu.py"