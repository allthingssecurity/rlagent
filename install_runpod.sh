#!/bin/bash

# RunPod Installation Script for ART HTTP Training Service
# This script sets up everything needed to run the service on RunPod

set -e  # Exit on any error

echo "🚀 Setting up ART HTTP Training Service on RunPod..."
echo "================================================================"

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq git curl wget build-essential

# Install Python dependencies
echo "🐍 Installing Python dependencies..."

# First install PyTorch with CUDA support
echo "   Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "   Installing other ML dependencies..."
pip install transformers>=4.30.0 accelerate>=0.20.0 bitsandbytes>=0.41.0
pip install sentencepiece>=0.1.99 protobuf>=3.20.0 tokenizers>=0.13.0
pip install datasets>=2.12.0 numpy>=1.24.0 pandas>=2.0.0

# Install service dependencies
echo "   Installing service dependencies..."
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 openai>=1.0.0
pip install python-dotenv>=1.0.0 pydantic>=2.0.0 httpx>=0.25.0
pip install asyncio-throttle>=1.0.0 tqdm>=4.65.0 wandb>=0.15.0
pip install psutil>=5.9.0 GPUtil>=1.4.0

# Install ART framework
echo "   Installing ART framework..."
pip install openpipe-art>=0.1.0

# Install the service itself
echo "📁 Installing ART HTTP Training Service..."
pip install -e .

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p .art
mkdir -p logs

# Set up environment
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file - please edit with your API keys"
else
    echo "✅ .env file already exists"
fi

# Test GPU availability
echo "🖥️ Testing GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ No CUDA GPUs detected')
"

# Test ART installation
echo "🧪 Testing ART installation..."
python -c "
try:
    import art
    print('✅ ART framework imported successfully')
except ImportError as e:
    print(f'❌ ART import failed: {e}')
"

# Test service dependencies
echo "🧪 Testing service dependencies..."
python -c "
import sys
try:
    import fastapi, uvicorn, openai, transformers, torch
    print('✅ All core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Dependency import failed: {e}')
    sys.exit(1)
"

# Create startup script
echo "📝 Creating startup script..."
cat > start_service.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting ART HTTP Training Service..."

# Check environment
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please create it with your API keys."
    exit 1
fi

# Load environment variables
source .env

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ Warning: OPENAI_API_KEY not set. Some features may not work."
fi

# Start the service
python run_server.py
EOF

chmod +x start_service.sh

echo ""
echo "🎉 Installation complete!"
echo "================================================================"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Start the service:"
echo "   ./start_service.sh"
echo "   # OR"
echo "   python run_server.py"
echo ""
echo "3. Test the service:"
echo "   curl http://localhost:8000/health"
echo ""
echo "4. Run examples:"
echo "   python examples/simple_math_agent.py"
echo ""
echo "🔗 Service will be available at: http://localhost:8000"
echo "📖 API docs at: http://localhost:8000/docs"
echo ""
echo "💡 For RunPod access, use the public URL provided in your pod settings."