"""Pydantic schemas for API requests and responses."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A conversation message."""
    role: Literal["system", "user", "assistant"]
    content: str


class TrajectoryRequest(BaseModel):
    """Request for creating a trajectory."""
    messages: List[Message]
    reward: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    """Request for training an agent."""
    model_name: str = Field(..., description="Name of the model to train")
    project: str = Field(..., description="Project name for organizing models")
    base_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Base model to fine-tune"
    )
    
    # Training parameters
    num_steps: int = Field(default=10, description="Number of training steps")
    batch_size: int = Field(default=8, description="Number of trajectories per batch")
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    
    # Evaluation parameters
    use_ruler: bool = Field(default=True, description="Use RULER for automatic scoring")
    judge_model: str = Field(default="gpt-4o", description="Model to use for evaluation")
    
    # Task configuration
    task_prompt: str = Field(..., description="System prompt defining the task")
    task_type: str = Field(default="general", description="Type of task for organization")
    
    # Optional trajectories (if you want to provide pre-generated data)
    trajectories: Optional[List[TrajectoryRequest]] = None


class RolloutRequest(BaseModel):
    """Request for generating rollouts."""
    model_name: str
    project: str
    num_rollouts: int = Field(default=5, description="Number of rollouts to generate")
    task_prompt: str
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class EvaluationRequest(BaseModel):
    """Request for evaluating trajectories."""
    trajectories: List[TrajectoryRequest]
    judge_model: str = Field(default="gpt-4o")
    evaluation_prompt: Optional[str] = None
    scoring_criteria: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response from training request."""
    status: Literal["success", "error", "in_progress"]
    message: str
    model_name: str
    current_step: int
    training_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[str] = None


class RolloutResponse(BaseModel):
    """Response from rollout request."""
    status: Literal["success", "error"]
    trajectories: List[TrajectoryRequest]
    metadata: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Response from evaluation request."""
    status: Literal["success", "error"]
    scores: List[float]
    average_score: float
    evaluation_details: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Information about a registered model."""
    name: str
    project: str
    base_model: str
    current_step: int
    created_at: str
    last_trained: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    version: str
    art_backend: str
    openai_configured: bool
    wandb_configured: bool