"""Factory for Winkly chatbot agent."""
from agent_router import get_routed_agent

def build_chatbot_agent():
    """Return the multi-DB agent with intent routing."""
    return get_routed_agent()
