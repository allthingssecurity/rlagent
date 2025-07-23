"""Configuration management for ART HTTP Training Service."""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7


class ARTConfig(BaseModel):
    """ART framework configuration."""
    backend: Literal["local", "skypilot"] = Field(
        default_factory=lambda: os.getenv("ART_BACKEND", "local")
    )
    path: str = Field(default_factory=lambda: os.getenv("ART_PATH", "./.art"))
    default_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    default_learning_rate: float = 1e-5
    max_seq_length: int = 8192


class ServerConfig(BaseModel):
    """HTTP server configuration."""
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("WANDB_API_KEY"))
    project: str = Field(default_factory=lambda: os.getenv("WANDB_PROJECT", "art-http-training"))
    enabled: bool = Field(default_factory=lambda: bool(os.getenv("WANDB_API_KEY")))


class Config(BaseModel):
    """Main application configuration."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    art: ARTConfig = Field(default_factory=ARTConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)


# Global config instance
config = Config()