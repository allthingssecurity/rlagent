"""OpenAI integration service for evaluation and data generation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from .config import config
from .schemas import TrajectoryRequest, Message

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for OpenAI API interactions."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return
            
        if not config.openai.api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            self.client = AsyncOpenAI(
                api_key=config.openai.api_key,
                base_url=config.openai.base_url
            )
            
            # Test the connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("OpenAI service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {e}")
            raise
    
    async def _test_connection(self):
        """Test the OpenAI API connection."""
        try:
            response = await self.client.chat.completions.create(
                model=config.openai.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise
    
    async def ensure_initialized(self):
        """Ensure the service is initialized."""
        if not self._initialized:
            await self.initialize()
    
    async def generate_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a completion using OpenAI API."""
        await self.ensure_initialized()
        
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ]
            
            response = await self.client.chat.completions.create(
                model=model or config.openai.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            raise
    
    async def evaluate_trajectory(
        self,
        trajectory: TrajectoryRequest,
        evaluation_prompt: Optional[str] = None,
        scoring_criteria: Optional[str] = None
    ) -> float:
        """Evaluate a trajectory using GPT-4o as a judge."""
        await self.ensure_initialized()
        
        # Default evaluation prompt
        if not evaluation_prompt:
            evaluation_prompt = """
            You are an expert evaluator. Please evaluate the following conversation trajectory 
            based on how well the assistant performed the given task. 
            
            Rate the performance on a scale from 0.0 to 1.0, where:
            - 0.0 = Complete failure
            - 0.5 = Partial success
            - 1.0 = Perfect success
            
            Consider factors like:
            - Task completion
            - Response quality
            - Helpfulness
            - Accuracy
            
            Respond with only a single number between 0.0 and 1.0.
            """
        
        # Build the evaluation messages
        eval_messages = [
            Message(role="system", content=evaluation_prompt)
        ]
        
        if scoring_criteria:
            eval_messages.append(
                Message(role="user", content=f"Scoring criteria: {scoring_criteria}")
            )
        
        # Add the trajectory to evaluate
        trajectory_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in trajectory.messages
        ])
        
        eval_messages.append(
            Message(
                role="user", 
                content=f"Please evaluate this trajectory:\n\n{trajectory_text}"
            )
        )
        
        try:
            response = await self.generate_completion(
                eval_messages,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=10
            )
            
            # Extract the score
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse evaluation score: {response}, error: {e}")
            return 0.5  # Default score on parsing error
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            raise
    
    async def generate_training_data(
        self,
        task_prompt: str,
        num_examples: int = 10,
        diversity_prompt: Optional[str] = None
    ) -> List[TrajectoryRequest]:
        """Generate synthetic training data using GPT-4o."""
        await self.ensure_initialized()
        
        trajectories = []
        
        base_prompt = f"""
        You are helping to generate diverse training examples for the following task:
        
        Task: {task_prompt}
        
        Generate a realistic conversation that demonstrates this task being performed well.
        Include both user inputs and assistant responses.
        Make each example unique and diverse.
        """
        
        if diversity_prompt:
            base_prompt += f"\n\nAdditional guidance: {diversity_prompt}"
        
        for i in range(num_examples):
            try:
                messages = [
                    Message(role="system", content=base_prompt),
                    Message(
                        role="user", 
                        content=f"Generate training example #{i+1}. Make it different from previous examples."
                    )
                ]
                
                response = await self.generate_completion(
                    messages,
                    temperature=0.8,  # Higher temperature for diversity
                    max_tokens=2000
                )
                
                # Parse the response into a trajectory
                # This is a simplified parser - you might want to make it more robust
                trajectory_messages = self._parse_generated_trajectory(response, task_prompt)
                
                if trajectory_messages:
                    trajectory = TrajectoryRequest(
                        messages=trajectory_messages,
                        metadata={
                            "generated": True,
                            "example_number": i + 1,
                            "task": task_prompt
                        }
                    )
                    
                    # Auto-evaluate the generated trajectory
                    score = await self.evaluate_trajectory(trajectory)
                    trajectory.reward = score
                    
                    trajectories.append(trajectory)
                
            except Exception as e:
                logger.warning(f"Failed to generate training example {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(trajectories)} training trajectories")
        return trajectories
    
    def _parse_generated_trajectory(
        self, 
        response: str, 
        task_prompt: str
    ) -> List[Message]:
        """Parse a generated trajectory response into messages."""
        messages = [Message(role="system", content=task_prompt)]
        
        lines = response.strip().split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for role indicators
            if line.lower().startswith(('user:', 'human:', 'u:')):
                if current_role and current_content:
                    messages.append(Message(
                        role=current_role,
                        content='\n'.join(current_content).strip()
                    ))
                current_role = "user"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
            elif line.lower().startswith(('assistant:', 'ai:', 'a:', 'bot:')):
                if current_role and current_content:
                    messages.append(Message(
                        role=current_role,
                        content='\n'.join(current_content).strip()
                    ))
                current_role = "assistant"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
            else:
                if current_content:
                    current_content.append(line)
        
        # Add the last message
        if current_role and current_content:
            messages.append(Message(
                role=current_role,
                content='\n'.join(current_content).strip()
            ))
        
        return messages if len(messages) > 1 else []  # Need at least system + one other message


# Global service instance
openai_service = OpenAIService()