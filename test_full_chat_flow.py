#!/usr/bin/env python3

import requests
import json
import time

def test_chat_endpoint():
    """Test the chat endpoint with a query that should return both orders and loyalty data"""
    
    # Start the server if not running
    try:
        response = requests.get("http://localhost:8000/ping", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running. Please start it with: python main.py")
        return
    
    # Test query that should return both orders and loyalty data
    test_query = "show last two order and their separate order amount and customer name his loyalty card of customer 1338787"
    
    print(f"🔍 Testing query: {test_query}")
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": test_query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received:")
            print(json.dumps(result, indent=2))
            
            # Check if we got both orders and loyalty data
            if isinstance(result.get("output"), list):
                print("✅ Got list response with multiple data sources")
                for i, item in enumerate(result["output"]):
                    print(f"  Item {i+1}: {item.get('type', 'unknown')}")
            elif isinstance(result.get("output"), dict):
                print("✅ Got single response")
                print(f"  Type: {result['output'].get('type', 'unknown')}")
            else:
                print("❌ Unexpected response format")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chat_endpoint() 