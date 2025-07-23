"""ART integration service for model management and training."""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator
from datetime import datetime

import art
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.trajectories import Trajectory, TrajectoryGroup

from .config import config
from .schemas import TrajectoryRequest, TrainingRequest, Message

logger = logging.getLogger(__name__)


class ARTService:
    """Service for managing ART models and training."""
    
    def __init__(self):
        self.backend: Optional[LocalBackend] = None
        self.models: Dict[str, art.TrainableModel] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the ART backend."""
        if self._initialized:
            return
            
        try:
            if config.art.backend == "local":
                self.backend = LocalBackend(path=config.art.path)
            else:
                # For SkyPilot backend, would need additional configuration
                raise NotImplementedError("SkyPilot backend not implemented yet")
            
            self._initialized = True
            logger.info(f"ART service initialized with {config.art.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize ART service: {e}")
            raise
    
    async def ensure_initialized(self):
        """Ensure the service is initialized."""
        if not self._initialized:
            await self.initialize()
    
    async def register_model(
        self, 
        name: str, 
        project: str, 
        base_model: str
    ) -> art.TrainableModel:
        """Register a new trainable model."""
        await self.ensure_initialized()
        
        model_key = f"{project}/{name}"
        
        if model_key not in self.models:
            model = art.TrainableModel(
                name=name,
                project=project,
                base_model=base_model,
            )
            
            # Set internal config for sequence length
            model._internal_config = art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(
                    max_seq_length=config.art.max_seq_length,
                ),
            )
            
            await model.register(self.backend)
            self.models[model_key] = model
            
            logger.info(f"Registered model: {model_key} with base model: {base_model}")
        
        return self.models[model_key]
    
    async def get_model_step(self, name: str, project: str) -> int:
        """Get the current training step for a model."""
        model_key = f"{project}/{name}"
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        return await self.models[model_key].get_step()
    
    def _trajectory_from_request(self, req: TrajectoryRequest) -> Trajectory:
        """Convert a trajectory request to an ART Trajectory."""
        messages = []
        for msg in req.messages:
            messages.append(art.types.Message(
                role=msg.role,
                content=msg.content
            ))
        
        return Trajectory(
            messages=messages,
            reward=req.reward or 0.0,
            metadata=req.metadata or {}
        )
    
    async def train_model(
        self,
        request: TrainingRequest,
        custom_trajectories: Optional[List[TrajectoryRequest]] = None
    ) -> AsyncIterator[Dict[str, float]]:
        """Train a model with provided or generated trajectories."""
        await self.ensure_initialized()
        
        # Register the model
        model = await self.register_model(
            request.model_name,
            request.project,
            request.base_model
        )
        
        current_step = await model.get_step()
        target_steps = current_step + request.num_steps
        
        logger.info(f"Training {request.model_name} from step {current_step} to {target_steps}")
        
        for step in range(current_step, target_steps):
            try:
                # Use custom trajectories if provided, otherwise generate them
                if custom_trajectories:
                    trajectories = [self._trajectory_from_request(t) for t in custom_trajectories]
                else:
                    # Generate trajectories through rollouts
                    trajectories = await self._generate_trajectories(
                        model, 
                        request.task_prompt,
                        request.batch_size,
                        step
                    )
                
                if not trajectories:
                    logger.warning(f"No trajectories generated for step {step}")
                    continue
                
                # Create trajectory groups
                train_groups = [TrajectoryGroup(trajectories=trajectories)]
                
                # Apply RULER scoring if enabled
                if request.use_ruler:
                    scored_groups = []
                    for group in train_groups:
                        try:
                            scored_group = await ruler_score_group(
                                group,
                                request.judge_model,
                                debug=True,
                                swallow_exceptions=True
                            )
                            if scored_group is not None:
                                scored_groups.append(scored_group)
                        except Exception as e:
                            logger.warning(f"RULER scoring failed: {e}")
                            scored_groups.append(group)
                    
                    train_groups = scored_groups
                
                if not train_groups:
                    logger.warning(f"No valid trajectory groups for step {step}")
                    continue
                
                # Train the model
                train_config = art.TrainConfig(learning_rate=request.learning_rate)
                
                async for metrics in model.train(train_groups, config=train_config):
                    yield {
                        "step": step,
                        "target_steps": target_steps,
                        **metrics
                    }
                
                logger.info(f"Completed training step {step}")
                
            except Exception as e:
                logger.error(f"Error during training step {step}: {e}")
                yield {
                    "step": step,
                    "error": str(e),
                    "status": "error"
                }
    
    async def _generate_trajectories(
        self,
        model: art.TrainableModel,
        task_prompt: str,
        batch_size: int,
        step: int
    ) -> List[Trajectory]:
        """Generate trajectories for training."""
        # This is a placeholder for trajectory generation
        # In a real implementation, you would:
        # 1. Use the model to generate responses to various prompts
        # 2. Simulate interactions with an environment
        # 3. Collect the conversation history as trajectories
        
        trajectories = []
        
        for i in range(batch_size):
            # Simple example: create a trajectory with system prompt and a dummy interaction
            messages = [
                art.types.Message(role="system", content=task_prompt),
                art.types.Message(role="user", content=f"Example task {i} for step {step}"),
                art.types.Message(role="assistant", content=f"Response {i} from step {step}")
            ]
            
            trajectory = Trajectory(
                messages=messages,
                reward=0.5,  # Placeholder reward
                metadata={"step": step, "batch_index": i}
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    async def close(self):
        """Clean up resources."""
        if self.backend:
            await self.backend.close()
        self._initialized = False
        logger.info("ART service closed")


# Global service instance
art_service = ARTService()