# ART HTTP Training Service

A generic HTTP-based agent training service built on the [ART framework](https://github.com/openpipe/art) with OpenAI GPT-4o integration for evaluation and data generation.

## Features

- 🚀 **HTTP API** for agent training with RESTful endpoints
- 🤖 **OpenAI Integration** using GPT-4o for evaluation and synthetic data generation
- ⚖️ **RULER Support** for automatic trajectory scoring without hand-crafted rewards
- 📊 **Real-time Monitoring** with streaming training progress
- 🔧 **Flexible Configuration** supporting multiple model backends
- 📝 **Rich Examples** demonstrating various training scenarios

## Quick Start

### 1. Installation

```bash
# Clone or create the project
cd art-http-training

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configuration

Copy the environment template and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_api_key_here  # Optional
```

### 3. Start the Service

```bash
# With uv
uv run python -m art_http.api

# Or directly
python src/art_http/api.py
```

The service will start on `http://localhost:8000`

### 4. Verify Installation

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "art_backend": "local",
  "openai_configured": true,
  "wandb_configured": true
}
```

## API Endpoints

### Core Training Endpoints

- **POST `/train`** - Start training an agent
- **GET `/train/{project}/{model_name}/status`** - Get training status
- **GET `/train/{project}/{model_name}/stream`** - Stream real-time progress
- **DELETE `/models/{project}/{model_name}`** - Delete a model

### Data Generation & Evaluation

- **POST `/rollout`** - Generate training data using OpenAI
- **POST `/evaluate`** - Evaluate trajectories with GPT-4o as judge
- **GET `/models`** - List all registered models

### Utility

- **GET `/health`** - Service health check

## Usage Examples

### Basic Training Request

```python
import httpx

async def train_agent():
    async with httpx.AsyncClient() as client:
        request = {
            "model_name": "my-agent-v1",
            "project": "my-project",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "num_steps": 5,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "use_ruler": True,
            "judge_model": "gpt-4o",
            "task_prompt": "You are a helpful assistant that solves problems step by step."
        }
        
        response = await client.post("http://localhost:8000/train", json=request)
        print(response.json())
```

### Generate Training Data

```python
rollout_request = {
    "model_name": "data-gen",
    "project": "examples",
    "num_rollouts": 10,
    "task_prompt": "Generate diverse question-answer pairs for training.",
    "temperature": 0.8
}

response = await client.post("http://localhost:8000/rollout", json=rollout_request)
```

### Evaluate Trajectories

```python
eval_request = {
    "trajectories": [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4"}
            ]
        }
    ],
    "judge_model": "gpt-4o",
    "evaluation_prompt": "Rate the accuracy of this mathematical response."
}

response = await client.post("http://localhost:8000/evaluate", json=eval_request)
```

## Complete Examples

### Math Agent

Train an agent to solve math problems:

```bash
uv run python examples/simple_math_agent.py
```

This example demonstrates:
- Custom trajectory generation
- Training with pre-defined data
- Progress monitoring
- Model evaluation

### Text Summarization Agent

Train a text summarization agent with auto-generated data:

```bash
uv run python examples/text_summarization_agent.py
```

This example shows:
- Automatic data generation using OpenAI
- RULER-based evaluation
- Custom vs automatic training data
- Advanced task configuration

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o | Required |
| `WANDB_API_KEY` | Weights & Biases API key | Optional |
| `ART_BACKEND` | ART backend type (`local`/`skypilot`) | `local` |
| `ART_PATH` | Local storage path for ART | `./.art` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Model Configuration

The service supports any model compatible with the ART framework:

- **Qwen models**: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- **Llama models**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Custom models**: Any HuggingFace transformers compatible model

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_steps` | Number of training iterations | 10 |
| `batch_size` | Trajectories per training batch | 8 |
| `learning_rate` | Training learning rate | 1e-5 |
| `use_ruler` | Enable automatic scoring | true |
| `judge_model` | Model for evaluation | gpt-4o |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTTP Client   │────│  FastAPI Server  │────│  ART Framework  │
│                 │    │                  │    │                 │
│ - Training Req  │    │ - Route Handler  │    │ - Model Training│
│ - Status Check  │    │ - Job Management │    │ - LoRA/GRPO     │
│ - Data Gen      │    │ - Progress Track │    │ - Checkpoints   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  OpenAI Service  │
                       │                  │
                       │ - GPT-4o Judge   │
                       │ - Data Generation│
                       │ - RULER Support  │
                       └──────────────────┘
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests (when implemented)
uv run pytest
```

### Code Formatting

```bash
uv run ruff format .
uv run ruff check .
```

### Adding New Examples

1. Create a new file in `examples/`
2. Follow the pattern from existing examples
3. Document the specific use case and configuration

## Troubleshooting

### Common Issues

**Service won't start**
- Check that all required dependencies are installed
- Verify OpenAI API key is configured
- Ensure port 8000 is available

**Training fails immediately**
- Check model name is valid and accessible
- Verify sufficient disk space for model storage
- Check logs for CUDA/GPU related errors

**Poor training performance**
- Increase batch size if memory allows
- Adjust learning rate (try 5e-6 or 2e-5)
- Ensure training data quality is high
- Consider using RULER for better reward signals

**OpenAI API errors**
- Verify API key is valid and has sufficient credits
- Check rate limiting (add delays between requests)
- Ensure model name (gpt-4o) is available in your region

### Logs and Debugging

Logs are output to stdout. Increase verbosity with:

```bash
LOG_LEVEL=debug uv run python src/art_http/api.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes formatting checks
5. Submit a pull request

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## Acknowledgments

- [ART Framework](https://github.com/openpipe/art) for the core RL training infrastructure
- [OpenAI](https://openai.com/) for the evaluation and data generation capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework