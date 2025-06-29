#!/usr/bin/env python3

import logging
from agent.agent_router import get_routed_agent

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def test_agent_output():
    print("=== Testing Agent Output ===")
    
    # Create the agent
    agent = get_routed_agent()
    
    # Test with the order query
    input_dict = {
        "input": "show last two order and their separate order amount and customer name his loyalty card of customer 1338787",
        "chat_history": []
    }
    
    print(f"Input: {input_dict}")
    
    # Invoke the agent
    result = agent.invoke(input_dict)
    
    print(f"Agent result type: {type(result)}")
    print(f"Agent result: {result}")
    
    # Check if it has the expected structure
    if isinstance(result, dict):
        if "output" in result:
            print("✅ Agent returned structure with 'output' field")
            print(f"Output field: {result['output']}")
        else:
            print("❌ Agent returned dict but no 'output' field")
            print(f"Keys: {list(result.keys())}")
    else:
        print(f"❌ Agent returned non-dict: {type(result)}")

if __name__ == "__main__":
    test_agent_output() 