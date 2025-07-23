#!/bin/bash

# Quick fix script for RunPod dependency issues
echo "🔧 Fixing missing dependencies on RunPod..."

# Install missing system packages first
echo "📦 Installing system packages..."
apt-get update -qq
apt-get install -y -qq build-essential

# Install the missing Python packages
echo "🐍 Installing missing Python packages..."
pip install setproctitle>=1.3.0
pip install multiprocess>=0.70.0
pip install polars>=0.20.0
pip install weave>=0.50.0
pip install litellm>=1.0.0

# Install Unsloth (might take a while)
echo "🦥 Installing Unsloth (this may take a few minutes)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Try to install xformers and flash-attn (optional but helpful)
echo "⚡ Installing performance packages (optional)..."
pip install xformers || echo "⚠️ xformers failed (not critical)"
pip install flash-attn --no-build-isolation || echo "⚠️ flash-attn failed (not critical)"

echo ""
echo "🧪 Testing imports..."
python -c "
try:
    import setproctitle
    print('✅ setproctitle')
except: print('❌ setproctitle')

try:
    import multiprocess
    print('✅ multiprocess')  
except: print('❌ multiprocess')

try:
    import art
    print('✅ art framework')
except Exception as e: 
    print(f'❌ art framework: {e}')

try:
    import torch
    print('✅ torch')
except: print('❌ torch')
"

echo ""
echo "🎯 Now try running: python train_simple.py"