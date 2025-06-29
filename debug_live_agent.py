#!/usr/bin/env python3

import logging
import os
from agent.agent_router import _create_live_agent, extract_focused_prompt

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def test_live_agent():
    print("=== Testing Live Agent ===")
    
    # Test 1: Check if extract_focused_prompt works
    query = "show last two order and their separate order amount and customer name his loyalty card of customer 1338787"
    print(f"Query: {query}")
    
    live_prompt = extract_focused_prompt(query, db="live")
    print(f"Live prompt: {live_prompt}")
    
    if not live_prompt:
        print("❌ extract_focused_prompt returned None for live database")
        return
    
    # Test 2: Create live agent
    print("\n=== Creating Live Agent ===")
    try:
        live_agent = _create_live_agent()
        print("✅ Live agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create live agent: {e}")
        return
    
    # Test 3: Test live agent with the prompt
    print("\n=== Testing Live Agent with Prompt ===")
    try:
        result = live_agent.invoke({
            "input": live_prompt,
            "chat_history": []
        })
        print(f"Live agent result: {result}")
    except Exception as e:
        print(f"❌ Live agent failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_agent() 