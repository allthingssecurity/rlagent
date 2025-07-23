# RunPod GPU LoRA Training

This project provides a minimal setup for GPU LoRA training on RunPod, avoiding all the pydantic/weave dependency conflicts.

## Quick Start on RunPod

1. Clone the repository:
```bash
git clone https://github.com/allthingssecurity/rlagent
cd rlagent
```

2. Run the FIXED installation script:
```bash
bash runpod_install_fixed.sh
```

3. Start training:
```bash
python3 train_minimal_gpu.py
```

## What This Does

### Installation (`runpod_install.sh`)
- **Removes conflicting packages**: weave, litellm, fastapi (cause pydantic conflicts)
- **Installs exact working versions**:
  - PyTorch 2.0.1 with CUDA 11.8
  - Transformers 4.33.2 + Tokenizers 0.13.3 (compatible)
  - PEFT 0.4.0 + TRL 0.6.0 (compatible LoRA stack)
  - vLLM 0.2.2 + Pydantic 1.10.13 (compatible)
  - ART framework (no weave dependencies)

### Training (`train_minimal_gpu.py`)
- **GPU-only LoRA training** using ART framework
- **Minimal dependencies** to avoid conflicts
- **Simple test case** to verify LoRA works
- **Comprehensive error diagnostics**

## Key Features

✅ **No Pydantic Conflicts**: Avoids weave/litellm that require pydantic>=2.0.0  
✅ **GPU Optimized**: Uses CUDA-enabled PyTorch with proper memory management  
✅ **LoRA Compatible**: Exact versions that work together  
✅ **RunPod Ready**: Designed for RunPod GPU instances  
✅ **Error Diagnostics**: Clear error messages for troubleshooting  

## Troubleshooting

If you still get LoRA errors after installation:

1. **Check GPU**: Ensure CUDA is available
2. **Check versions**: Run `pip list | grep -E "(peft|vllm|torch)"`
3. **Try older versions**: Some version combinations still don't work

## Why This Approach Works

The main issue was **dependency hell**:
- `weave>=0.50.0` requires `pydantic>=2.0.0`
- `vllm==0.2.2` requires `pydantic==1.10.13`
- This is an **impossible conflict**

Our solution:
- **Skip weave/litellm entirely** (not needed for basic training)
- **Use exact compatible versions** that work together
- **Install with --no-deps** when needed to bypass pip resolver

This gives you a working GPU LoRA training setup without the dependency conflicts.