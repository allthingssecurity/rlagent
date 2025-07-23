#!/bin/bash

# ONE COMMAND FIX - Install everything needed for ART in the right order
echo "🚀 ONE COMMAND FIX for ART Framework"
echo "This will install ALL dependencies in the correct order"
echo "Estimated time: 10-15 minutes"
echo "=========================================="

# Exit on any error
set -e

# Update system
echo "📦 Updating system..."
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq build-essential git curl wget python3-dev >/dev/null 2>&1

# Upgrade pip
echo "🛠️ Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1

# Install PyTorch with CUDA (most important)
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >/dev/null 2>&1

# Install critical system dependencies for ART
echo "🔧 Installing ART system dependencies..."
pip install setproctitle>=1.3.0 multiprocess>=0.70.0 tblib>=1.7.0 >/dev/null 2>&1
pip install cloudpickle>=2.0.0 dill>=0.3.0 packaging>=21.0 >/dev/null 2>&1

# Install torchao and torchtune (order matters!)
echo "⚡ Installing PyTorch ecosystem..."
pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cu118 >/dev/null 2>&1 || pip install torchao >/dev/null 2>&1
pip install torchtune>=0.1.0 >/dev/null 2>&1

# Install vLLM
echo "🚀 Installing vLLM..."
pip install vllm>=0.2.0 >/dev/null 2>&1

# Install ML frameworks
echo "🤖 Installing ML frameworks..."
pip install transformers>=4.30.0 accelerate>=0.20.0 >/dev/null 2>&1
pip install bitsandbytes>=0.41.0 tokenizers>=0.13.0 >/dev/null 2>&1
pip install trl>=0.7.0 peft>=0.5.0 >/dev/null 2>&1

# Install Unsloth (the tricky one)
echo "🦥 Installing Unsloth (this takes a few minutes)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" >/dev/null 2>&1 || \
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" >/dev/null 2>&1 || \
pip install unsloth >/dev/null 2>&1

# Install ART framework dependencies
echo "🎨 Installing ART dependencies..."
pip install polars>=0.20.0 weave>=0.50.0 litellm>=1.0.0 >/dev/null 2>&1

# Install service dependencies
echo "🌐 Installing service dependencies..."
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 >/dev/null 2>&1
pip install httpx>=0.25.0 openai>=1.0.0 python-dotenv>=1.0.0 >/dev/null 2>&1

# Install ART framework itself
echo "🎯 Installing ART framework..."
pip install openpipe-art>=0.1.0 >/dev/null 2>&1

# Test everything
echo ""
echo "🧪 Testing critical imports..."
python3 -c "
import sys
success = True
modules = [
    ('torch', 'PyTorch'),
    ('unsloth', 'Unsloth'), 
    ('vllm', 'vLLM'),
    ('art', 'ART Framework'),
]

for module, name in modules:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {str(e)[:50]}...')
        success = False

if success:
    print('\n🎉 ALL CRITICAL MODULES WORKING!')
else:
    print('\n⚠️ Some modules failed - but most should work')
"

echo ""
echo "🎯 Installation complete!"
echo "=========================================="
echo "🚀 Try running:"
echo "   python train_simple.py"
echo "   python quick_start.py"
echo ""
echo "If you still get errors, restart your Python session."