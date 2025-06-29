#!/usr/bin/env python3
"""
Quick test for both customer cases
"""

import requests
import json

def quick_test():
    """Quick test for both customers"""
    
    test_cases = [
        {
            "name": "Customer 1338787 (has orders)",
            "customer_id": "1338787",
            "expected": ["orders_summary", "customer_summary", "loyalty_summary"]
        },
        {
            "name": "Customer 2555880 (no orders)", 
            "customer_id": "2555880",
            "expected": ["customer_summary", "loyalty_summary"]
        }
    ]
    
    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        print("-" * 50)
        
        payload = {
            "input": f"show last two order and their separate order amount and customer name his loyalty card of customer {test['customer_id']}",
            "chat_history": ["string"],
            "summarize": False,
            "conversationId": f"test_{test['customer_id']}"
        }
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if response is a list
                if isinstance(result['output'], list):
                    print(f"✅ Response is a list with {len(result['output'])} items")
                    
                    # Get actual types
                    actual_types = [item.get('type') for item in result['output'] if isinstance(item, dict)]
                    print(f"📋 Found types: {actual_types}")
                    
                    # Check expected types
                    missing = [t for t in test['expected'] if t not in actual_types]
                    if missing:
                        print(f"❌ Missing: {missing}")
                    else:
                        print(f"✅ All expected types present")
                    
                    # Show summary
                    for item in result['output']:
                        if isinstance(item, dict):
                            item_type = item.get('type')
                            if item_type == 'orders_summary':
                                orders = item.get('orders', [])
                                print(f"   📦 Orders: {len(orders)} found")
                            elif item_type == 'customer_summary':
                                name = item.get('customer_name', 'N/A')
                                print(f"   👤 Customer: {name}")
                            elif item_type == 'loyalty_summary':
                                card = item.get('card_number', 'N/A')
                                status = item.get('status', 'N/A')
                                print(f"   💳 Loyalty: {card} ({status})")
                else:
                    print(f"❌ Response is not a list: {type(result['output'])}")
                    print(f"   Content: {result['output']}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Quick Test for Both Customer Cases")
    print("=" * 60)
    quick_test()
    print("\n🎉 Test completed!") 