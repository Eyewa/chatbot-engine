#!/usr/bin/env python3

import logging
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    timestamp: str
    call_type: str
    function_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    cost_estimate: float
    prompt_content: str
    response_content: str

class TokenTracker:
    def __init__(self):
        self.usage_log: List[TokenUsage] = []
        # Use cl100k_base encoding which is used by GPT-4 models
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost per 1K tokens (approximate)
        self.cost_per_1k_tokens = {
            "gpt-4o": 0.005,  # $0.005 per 1K input, $0.015 per 1K output
            "gpt-4o-mini": 0.00015,  # $0.00015 per 1K input, $0.0006 per 1K output
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
        """Estimate cost based on token usage."""
        cost_rates = self.cost_per_1k_tokens.get(model, self.cost_per_1k_tokens["gpt-4o"])
        input_cost = (input_tokens / 1000) * cost_rates
        output_cost = (output_tokens / 1000) * (cost_rates * 3)  # Output is typically 3x input cost
        return input_cost + output_cost
    
    def log_call(self, call_type: str, function_name: str, prompt: str, response: str, model: str = "gpt-4o"):
        """Log a token usage call."""
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(response)
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens, model)
        
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            call_type=call_type,
            function_name=function_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            cost_estimate=cost,
            prompt_content=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            response_content=response[:200] + "..." if len(response) > 200 else response
        )
        
        self.usage_log.append(usage)
        
        logger.info(f"🔍 TOKEN USAGE: {call_type} | {function_name} | "
                   f"Input: {input_tokens} | Output: {output_tokens} | "
                   f"Total: {total_tokens} | Cost: ${cost:.4f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all token usage."""
        if not self.usage_log:
            return {"message": "No token usage logged yet"}
        
        total_calls = len(self.usage_log)
        total_input_tokens = sum(u.input_tokens for u in self.usage_log)
        total_output_tokens = sum(u.output_tokens for u in self.usage_log)
        total_tokens = sum(u.total_tokens for u in self.usage_log)
        total_cost = sum(u.cost_estimate for u in self.usage_log)
        
        # Group by call type
        call_type_summary = {}
        for usage in self.usage_log:
            if usage.call_type not in call_type_summary:
                call_type_summary[usage.call_type] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0
                }
            
            call_type_summary[usage.call_type]["calls"] += 1
            call_type_summary[usage.call_type]["input_tokens"] += usage.input_tokens
            call_type_summary[usage.call_type]["output_tokens"] += usage.output_tokens
            call_type_summary[usage.call_type]["total_tokens"] += usage.total_tokens
            call_type_summary[usage.call_type]["cost"] += usage.cost_estimate
        
        return {
            "summary": {
                "total_calls": total_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "total_cost_estimate": total_cost,
                "average_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0
            },
            "by_call_type": call_type_summary,
            "detailed_log": [asdict(u) for u in self.usage_log]
        }
    
    def print_summary(self):
        """Print a formatted summary of token usage."""
        summary = self.get_summary()
        
        if "message" in summary:
            print(summary["message"])
            return
        
        print("\n" + "="*80)
        print("🔍 TOKEN USAGE SUMMARY")
        print("="*80)
        
        s = summary["summary"]
        print(f"📊 Total Calls: {s['total_calls']}")
        print(f"📥 Total Input Tokens: {s['total_input_tokens']:,}")
        print(f"📤 Total Output Tokens: {s['total_output_tokens']:,}")
        print(f"🔢 Total Tokens: {s['total_tokens']:,}")
        print(f"💰 Estimated Cost: ${s['total_cost_estimate']:.4f}")
        print(f"📈 Average Tokens/Call: {s['average_tokens_per_call']:.1f}")
        
        print("\n📋 Breakdown by Call Type:")
        print("-" * 50)
        
        for call_type, stats in summary["by_call_type"].items():
            print(f"\n🔹 {call_type.upper()}:")
            print(f"   Calls: {stats['calls']}")
            print(f"   Input Tokens: {stats['input_tokens']:,}")
            print(f"   Output Tokens: {stats['output_tokens']:,}")
            print(f"   Total Tokens: {stats['total_tokens']:,}")
            print(f"   Cost: ${stats['cost']:.4f}")
            print(f"   Avg per call: {stats['total_tokens']/stats['calls']:.1f} tokens")
        
        print("\n" + "="*80)

# Global tracker instance
token_tracker = TokenTracker()

def track_llm_call(call_type: str, function_name: str, prompt: str, response: str, model: str = "gpt-4o"):
    """Decorator function to track LLM calls."""
    token_tracker.log_call(call_type, function_name, prompt, response, model)

class TokenTrackingLLMWrapper:
    """Wrapper class to automatically track token usage for LLM calls"""
    
    def __init__(self, llm_instance, call_type: str = "llm_call", function_name: str = "unknown"):
        self.llm = llm_instance
        self.call_type = call_type
        self.function_name = function_name
        self.model_name = getattr(llm_instance, 'model_name', 'gpt-4o')
        # Copy other important attributes
        self.temperature = getattr(llm_instance, 'temperature', 0)
        self.max_retries = getattr(llm_instance, 'max_retries', 3)
    
    def invoke(self, messages, **kwargs):
        """Track token usage for invoke calls"""
        # Convert messages to string for token counting
        prompt = self._messages_to_string(messages)
        
        # Make the actual LLM call
        response = self.llm.invoke(messages, **kwargs)
        
        # Track the usage
        response_str = str(response)
        track_llm_call(
            call_type=self.call_type,
            function_name=self.function_name,
            prompt=prompt,
            response=response_str,
            model=self.model_name
        )
        
        return response
    
    def __or__(self, other):
        """Support for the | operator used in LangChain chains"""
        # Delegate to the wrapped LLM
        return self.llm.__or__(other)
    
    def _messages_to_string(self, messages):
        """Convert message list to string for token counting"""
        if isinstance(messages, str):
            return messages
        
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            else:
                prompt_parts.append(str(msg))
        
        return "\n".join(prompt_parts)

def wrap_llm_with_tracking(llm_instance, call_type: str = "llm_call", function_name: str = "unknown"):
    """Wrap an LLM instance with token tracking"""
    return TokenTrackingLLMWrapper(llm_instance, call_type, function_name)

if __name__ == "__main__":
    # Test the token tracker
    print("Testing Token Tracker...")
    
    # Simulate some calls
    track_llm_call(
        "classification", 
        "classify_intent", 
        "Classify this query: show last two orders", 
        "live"
    )
    
    track_llm_call(
        "agent_execution", 
        "live_agent", 
        "Execute SQL query for orders", 
        "Here are the orders..."
    )
    
    track_llm_call(
        "response_formatting", 
        "format_response", 
        "Format this data as JSON", 
        '{"type": "orders_summary"}'
    )
    
    # Print summary
    token_tracker.print_summary() 