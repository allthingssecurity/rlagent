#!/bin/bash

# Comprehensive fix script for RunPod dependency issues
echo "🔧 Comprehensive dependency fix for RunPod..."

# Install missing system packages first
echo "📦 Installing system packages..."
apt-get update -qq
apt-get install -y -qq build-essential

# Install ALL missing Python packages for ART
echo "🐍 Installing ALL missing Python packages..."

# Critical system deps first
pip install setproctitle>=1.3.0
pip install multiprocess>=0.70.0
pip install tblib>=1.7.0
pip install cloudpickle>=2.0.0
pip install dill>=0.3.0
pip install packaging>=21.0
pip install typing-extensions>=4.0.0
pip install filelock>=3.0.0

# ART specific deps
pip install polars>=0.20.0
pip install weave>=0.50.0
pip install litellm>=1.0.0

# ML framework deps
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install tokenizers>=0.13.0
pip install huggingface-hub>=0.16.0
pip install safetensors>=0.3.0

# Training deps
pip install trl>=0.7.0
pip install peft>=0.5.0
pip install torchtune>=0.1.0

# Utility deps
pip install scipy>=1.9.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0

# Install ART framework
echo "🎨 Installing ART framework..."
pip install openpipe-art>=0.1.0

# Install Unsloth (might take a while)
echo "🦥 Installing Unsloth (this may take a few minutes)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "⚠️ Unsloth failed (not critical)"

# Try to install performance packages (optional)
echo "⚡ Installing performance packages (optional)..."
pip install xformers || echo "⚠️ xformers failed (not critical)"
pip install flash-attn --no-build-isolation || echo "⚠️ flash-attn failed (not critical)"

echo ""
echo "🧪 Testing all critical imports..."
python -c "
imports_to_test = [
    ('setproctitle', 'setproctitle'),
    ('multiprocess', 'multiprocess'),
    ('tblib', 'tblib'),
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('art', 'ART Framework'),
]

all_good = True
for module, name in imports_to_test:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        all_good = False

# Test ART backend specifically
try:
    from art.local import LocalBackend
    print('✅ ART LocalBackend')
except ImportError as e:
    print(f'❌ ART LocalBackend: {e}')
    all_good = False

if all_good:
    print('\n🎉 All dependencies working!')
else:
    print('\n⚠️ Some dependencies still missing')
"

echo ""
echo "🎯 Now try running: python train_simple.py"