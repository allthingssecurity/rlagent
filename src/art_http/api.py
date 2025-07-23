"""Main FastAPI application with HTTP endpoints for agent training."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json

from . import __version__
from .config import config
from .schemas import (
    TrainingRequest, TrainingResponse, RolloutRequest, RolloutResponse,
    EvaluationRequest, EvaluationResponse, ModelInfo, HealthResponse,
    TrajectoryRequest
)
from .art_service import art_service
from .openai_service import openai_service

# Configure logging
logging.basicConfig(level=getattr(logging, config.server.log_level.upper()))
logger = logging.getLogger(__name__)

# Track ongoing training jobs
training_jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting ART HTTP Training Service...")
    
    try:
        # Initialize services
        await art_service.initialize()
        await openai_service.initialize()
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        await art_service.close()
        logger.info("ART HTTP Training Service stopped")


app = FastAPI(
    title="ART HTTP Training Service",
    description="Generic HTTP-based agent training service using ART framework",
    version=__version__,
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        art_backend=config.art.backend,
        openai_configured=bool(config.openai.api_key),
        wandb_configured=config.wandb.enabled
    )


@app.post("/train", response_model=TrainingResponse)
async def train_agent(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training an agent with the given configuration."""
    try:
        model_key = f"{request.project}/{request.model_name}"
        
        # Check if already training
        if model_key in training_jobs and training_jobs[model_key]["status"] == "in_progress":
            return TrainingResponse(
                status="error",
                message="Model is already being trained",
                model_name=request.model_name,
                current_step=training_jobs[model_key].get("current_step", 0),
                error_details="Training already in progress"
            )
        
        # Register the model first to get current step
        model = await art_service.register_model(
            request.model_name,
            request.project,
            request.base_model
        )
        current_step = await art_service.get_model_step(request.model_name, request.project)
        
        # Initialize training job tracking
        training_jobs[model_key] = {
            "status": "in_progress",
            "current_step": current_step,
            "target_steps": current_step + request.num_steps,
            "started_at": datetime.now().isoformat(),
            "request": request.model_dump()
        }
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            model_key,
            request
        )
        
        return TrainingResponse(
            status="in_progress",
            message=f"Training started for {request.model_name}",
            model_name=request.model_name,
            current_step=current_step
        )
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{project}/{model_name}/stream")
async def stream_training_progress(project: str, model_name: str):
    """Stream training progress in real-time."""
    model_key = f"{project}/{model_name}"
    
    if model_key not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    async def generate():
        """Generate training progress events."""
        last_step = -1
        
        while True:
            if model_key not in training_jobs:
                break
                
            job = training_jobs[model_key]
            
            if job["current_step"] != last_step:
                yield f"data: {json.dumps(job)}\n\n"
                last_step = job["current_step"]
            
            if job["status"] != "in_progress":
                break
                
            await asyncio.sleep(1)  # Poll every second
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.get("/train/{project}/{model_name}/status", response_model=TrainingResponse)
async def get_training_status(project: str, model_name: str):
    """Get the current training status for a model."""
    model_key = f"{project}/{model_name}"
    
    if model_key not in training_jobs:
        # Check if model exists but no active training
        try:
            current_step = await art_service.get_model_step(model_name, project)
            return TrainingResponse(
                status="success",
                message="No active training",
                model_name=model_name,
                current_step=current_step
            )
        except ValueError:
            raise HTTPException(status_code=404, detail="Model not found")
    
    job = training_jobs[model_key]
    return TrainingResponse(
        status=job["status"],
        message=job.get("message", "Training in progress"),
        model_name=model_name,
        current_step=job["current_step"],
        training_metrics=job.get("metrics")
    )


@app.post("/rollout", response_model=RolloutResponse)
async def generate_rollouts(request: RolloutRequest):
    """Generate rollouts using OpenAI for data generation."""
    try:
        trajectories = await openai_service.generate_training_data(
            task_prompt=request.task_prompt,
            num_examples=request.num_rollouts
        )
        
        return RolloutResponse(
            status="success",
            trajectories=trajectories,
            metadata={
                "num_generated": len(trajectories),
                "task_prompt": request.task_prompt,
                "model_used": config.openai.model
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to generate rollouts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_trajectories(request: EvaluationRequest):
    """Evaluate trajectories using GPT-4o as a judge."""
    try:
        scores = []
        
        for trajectory in request.trajectories:
            score = await openai_service.evaluate_trajectory(
                trajectory,
                request.evaluation_prompt,
                request.scoring_criteria
            )
            scores.append(score)
        
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationResponse(
            status="success",
            scores=scores,
            average_score=average_score,
            evaluation_details={
                "judge_model": request.judge_model,
                "num_trajectories": len(request.trajectories)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to evaluate trajectories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all registered models."""
    models = []
    
    for model_key, model in art_service.models.items():
        try:
            current_step = await art_service.get_model_step(model.name, model.project)
            
            models.append(ModelInfo(
                name=model.name,
                project=model.project,
                base_model=model.base_model,
                current_step=current_step,
                created_at=datetime.now().isoformat(),  # Placeholder
                last_trained=training_jobs.get(model_key, {}).get("started_at")
            ))
        except Exception as e:
            logger.warning(f"Failed to get info for model {model_key}: {e}")
            continue
    
    return models


@app.delete("/models/{project}/{model_name}")
async def delete_model(project: str, model_name: str):
    """Delete a model and stop any ongoing training."""
    model_key = f"{project}/{model_name}"
    
    # Stop training if in progress
    if model_key in training_jobs:
        training_jobs[model_key]["status"] = "cancelled"
        del training_jobs[model_key]
    
    # Remove from ART service
    if model_key in art_service.models:
        del art_service.models[model_key]
    
    return {"message": f"Model {model_key} deleted successfully"}


async def run_training_job(model_key: str, request: TrainingRequest):
    """Run a training job in the background."""
    try:
        training_jobs[model_key]["status"] = "in_progress"
        
        async for metrics in art_service.train_model(request, request.trajectories):
            if "error" in metrics:
                training_jobs[model_key]["status"] = "error"
                training_jobs[model_key]["message"] = metrics["error"]
                training_jobs[model_key]["error_details"] = metrics["error"]
                return
            
            training_jobs[model_key]["current_step"] = metrics.get("step", 0)
            training_jobs[model_key]["metrics"] = metrics
        
        # Training completed successfully
        training_jobs[model_key]["status"] = "success"
        training_jobs[model_key]["message"] = "Training completed successfully"
        training_jobs[model_key]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Training job {model_key} failed: {e}")
        training_jobs[model_key]["status"] = "error"
        training_jobs[model_key]["message"] = str(e)
        training_jobs[model_key]["error_details"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "art_http.api:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=True
    )