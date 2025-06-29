"""
Token Tracking Utility - Captures LLM call details with cost estimation
Senior AI Engineer Design: Non-intrusive token counting and cost tracking
"""

import time
import tiktoken
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage data for a single LLM call"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    cost_estimate: float
    timestamp: datetime


class TokenTracker:
    """
    Senior AI Engineer Design: Efficient token tracking with cost estimation
    
    Features:
    - Non-intrusive token counting
    - Accurate cost estimation for different models
    - Minimal performance impact
    - Support for all OpenAI models
    """
    
    # Cost per 1K tokens (as of 2024) - update as needed
    MODEL_COSTS = {
        # GPT-4 models
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        
        # GPT-3.5 models
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        
        # Default fallback
        "default": {"input": 0.0025, "output": 0.01}
    }
    
    def __init__(self):
        self.encoders = {}
        self._get_encoder("gpt-4o")  # Pre-load common encoder
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create tiktoken encoder for a model"""
        if model not in self.encoders:
            try:
                # Map model names to tiktoken encodings
                if "gpt-4" in model or "gpt-3.5" in model:
                    encoding_name = "cl100k_base"  # GPT-4 and GPT-3.5 use this
                else:
                    encoding_name = "cl100k_base"  # Default to GPT-4 encoding
                
                self.encoders[model] = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Created encoder for model: {model}")
            except Exception as e:
                logger.warning(f"Failed to create encoder for {model}, using default: {e}")
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model]
    
    def count_tokens(self, text: Union[str, list], model: str = "gpt-4o") -> int:
        """
        Count tokens in text using the appropriate encoder
        
        Args:
            text: Text to count tokens in (string or list of messages)
            model: Model name for encoding selection
            
        Returns:
            Number of tokens
        """
        try:
            encoder = self._get_encoder(model)
            
            if isinstance(text, str):
                return len(encoder.encode(text))
            elif isinstance(text, list):
                # Handle list of messages (e.g., chat history)
                total_tokens = 0
                for message in text:
                    if isinstance(message, dict):
                        # Handle message dict format
                        content = message.get('content', '')
                        role = message.get('role', '')
                        # Add tokens for role and content
                        total_tokens += len(encoder.encode(f"role: {role}\ncontent: {content}"))
                    elif isinstance(message, str):
                        total_tokens += len(encoder.encode(message))
                    else:
                        total_tokens += len(encoder.encode(str(message)))
                return total_tokens
            else:
                return len(encoder.encode(str(text)))
                
        except Exception as e:
            logger.error(f"Error counting tokens for model {model}: {e}")
            # Fallback: rough estimation (4 chars per token)
            if isinstance(text, str):
                return len(text) // 4
            elif isinstance(text, list):
                return sum(len(str(msg)) // 4 for msg in text)
            else:
                return len(str(text)) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
        """
        Estimate cost for token usage
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for cost lookup
            
        Returns:
            Estimated cost in USD
        """
        try:
            # Get model costs, fallback to default if not found
            model_costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["default"])
            
            input_cost = (input_tokens / 1000) * model_costs["input"]
            output_cost = (output_tokens / 1000) * model_costs["output"]
            
            total_cost = input_cost + output_cost
            return round(total_cost, 6)
            
        except Exception as e:
            logger.error(f"Error estimating cost for model {model}: {e}")
            # Fallback: use default GPT-4o pricing
            input_cost = (input_tokens / 1000) * 0.0025
            output_cost = (output_tokens / 1000) * 0.01
            return round(input_cost + output_cost, 6)
    
    def track_llm_call(
        self,
        input_text: Union[str, list],
        output_text: str,
        model: str = "gpt-4o",
        function_name: str = "unknown"
    ) -> TokenUsage:
        """
        Track a complete LLM call and return usage data
        
        Args:
            input_text: Input text or messages
            output_text: Output text from LLM
            model: Model name used
            function_name: Name of the function making the call
            
        Returns:
            TokenUsage object with all metrics
        """
        try:
            # Count tokens
            input_tokens = self.count_tokens(input_text, model)
            output_tokens = self.count_tokens(output_text, model)
            total_tokens = input_tokens + output_tokens
            
            # Estimate cost
            cost_estimate = self.estimate_cost(input_tokens, output_tokens, model)
            
            # Create usage object
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
                cost_estimate=cost_estimate,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Tracked LLM call - {function_name}: {input_tokens}+{output_tokens}={total_tokens} tokens, ${cost_estimate:.6f}")
            return usage
            
        except Exception as e:
            logger.error(f"Error tracking LLM call: {e}")
            # Return fallback usage data
            return TokenUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=model,
                cost_estimate=0.0,
                timestamp=datetime.now()
            )
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a model including costs"""
        costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["default"])
        return {
            "model": model,
            "input_cost_per_1k": costs["input"],
            "output_cost_per_1k": costs["output"],
            "encoder_available": model in self.encoders
        }


# Global token tracker instance
_token_tracker = None

def get_token_tracker() -> TokenTracker:
    """Get singleton token tracker instance"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker


# Convenience functions for easy integration
def count_tokens(text: Union[str, list], model: str = "gpt-4o") -> int:
    """Count tokens in text"""
    return get_token_tracker().count_tokens(text, model)

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """Estimate cost for token usage"""
    return get_token_tracker().estimate_cost(input_tokens, output_tokens, model)

def track_llm_call(input_text: Union[str, list], output_text: str, model: str = "gpt-4o", function_name: str = "unknown") -> TokenUsage:
    """Track a complete LLM call"""
    return get_token_tracker().track_llm_call(input_text, output_text, model, function_name) 