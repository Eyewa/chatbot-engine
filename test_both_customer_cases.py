#!/usr/bin/env python3
"""
Test script for both customer cases:
1. Customer 1338787 - has orders (should return orders_summary, customer_summary, loyalty_summary)
2. Customer 2555880 - no orders (should return customer_summary, loyalty_summary)
"""

import requests
import json
import time

def test_customer_with_orders():
    """Test customer 1338787 who has orders"""
    print("🧪 Testing Customer 1338787 (has orders)...")
    
    payload = {
        "input": "show last two order and their separate order amount and customer name his loyalty card of customer 1338787",
        "chat_history": ["string"],
        "summarize": False,
        "conversationId": "test_1338787"
    }
    
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Response received successfully")
        print(f"Response structure: {type(result['output'])}")
        
        # Check if response is a list (as expected)
        if isinstance(result['output'], list):
            print(f"✅ Response is a list with {len(result['output'])} items")
            
            # Check for required summary types
            summary_types = [item.get('type') for item in result['output'] if isinstance(item, dict)]
            print(f"Summary types found: {summary_types}")
            
            # Validate expected types
            expected_types = ['orders_summary', 'customer_summary', 'loyalty_summary']
            missing_types = [t for t in expected_types if t not in summary_types]
            
            if not missing_types:
                print("✅ All expected summary types found")
            else:
                print(f"❌ Missing summary types: {missing_types}")
            
            # Detailed validation
            for item in result['output']:
                if isinstance(item, dict):
                    item_type = item.get('type')
                    print(f"\n📋 {item_type.upper()}:")
                    
                    if item_type == 'orders_summary':
                        orders = item.get('orders', [])
                        print(f"   - Orders count: {len(orders)}")
                        if orders:
                            print(f"   - First order ID: {orders[0].get('order_id')}")
                            print(f"   - First order amount: {orders[0].get('order_amount')}")
                            print(f"   - Customer name in orders: {orders[0].get('customer_name')}")
                    
                    elif item_type == 'customer_summary':
                        print(f"   - Customer name: {item.get('customer_name')}")
                        print(f"   - Customer ID: {item.get('customer_id')}")
                        print(f"   - Email: {item.get('email', 'Not provided')}")
                        print(f"   - Mobile: {item.get('mobile_number', 'Not provided')}")
                        print(f"   - Country: {item.get('country_code', 'Not provided')}")
                    
                    elif item_type == 'loyalty_summary':
                        print(f"   - Card number: {item.get('card_number')}")
                        print(f"   - Status: {item.get('status')}")
                        print(f"   - Points balance: {item.get('points_balance', 'Not provided')}")
                        print(f"   - Expiry date: {item.get('expiry_date', 'Not provided')}")
        else:
            print("❌ Response is not a list as expected")
            print(f"Response type: {type(result['output'])}")
            print(f"Response content: {result['output']}")
    else:
        print(f"❌ Request failed with status code: {response.status_code}")
        print(f"Error: {response.text}")
    
    print("\n" + "="*60 + "\n")

def test_customer_without_orders():
    """Test customer 2555880 who has no orders"""
    print("🧪 Testing Customer 2555880 (no orders)...")
    
    payload = {
        "input": "show last two order and their separate order amount and customer name his loyalty card of customer 2555880",
        "chat_history": ["string"],
        "summarize": False,
        "conversationId": "test_2555880"
    }
    
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Response received successfully")
        print(f"Response structure: {type(result['output'])}")
        
        # Check if response is a list (as expected)
        if isinstance(result['output'], list):
            print(f"✅ Response is a list with {len(result['output'])} items")
            
            # Check for required summary types
            summary_types = [item.get('type') for item in result['output'] if isinstance(item, dict)]
            print(f"Summary types found: {summary_types}")
            
            # Validate expected types (no orders_summary expected)
            expected_types = ['customer_summary', 'loyalty_summary']
            missing_types = [t for t in expected_types if t not in summary_types]
            
            if not missing_types:
                print("✅ All expected summary types found")
            else:
                print(f"❌ Missing summary types: {missing_types}")
            
            # Check that orders_summary is NOT present (since no orders)
            if 'orders_summary' in summary_types:
                print("⚠️  orders_summary found but customer has no orders")
            else:
                print("✅ orders_summary correctly not present (no orders)")
            
            # Detailed validation
            for item in result['output']:
                if isinstance(item, dict):
                    item_type = item.get('type')
                    print(f"\n📋 {item_type.upper()}:")
                    
                    if item_type == 'orders_summary':
                        orders = item.get('orders', [])
                        print(f"   - Orders count: {len(orders)}")
                        if len(orders) == 0:
                            print("   ✅ Orders array is empty (correct for customer with no orders)")
                    
                    elif item_type == 'customer_summary':
                        print(f"   - Customer name: {item.get('customer_name')}")
                        print(f"   - Customer ID: {item.get('customer_id')}")
                        print(f"   - Email: {item.get('email', 'Not provided')}")
                        print(f"   - Mobile: {item.get('mobile_number', 'Not provided')}")
                        print(f"   - Country: {item.get('country_code', 'Not provided')}")
                    
                    elif item_type == 'loyalty_summary':
                        print(f"   - Card number: {item.get('card_number')}")
                        print(f"   - Status: {item.get('status')}")
                        print(f"   - Points balance: {item.get('points_balance', 'Not provided')}")
                        print(f"   - Expiry date: {item.get('expiry_date', 'Not provided')}")
        else:
            print("❌ Response is not a list as expected")
            print(f"Response type: {type(result['output'])}")
            print(f"Response content: {result['output']}")
    else:
        print(f"❌ Request failed with status code: {response.status_code}")
        print(f"Error: {response.text}")
    
    print("\n" + "="*60 + "\n")

def test_response_structure_validation():
    """Validate that responses follow the correct structure"""
    print("🔍 Validating Response Structure...")
    
    test_cases = [
        {
            "name": "Customer with Orders (1338787)",
            "customer_id": "1338787",
            "expected_types": ["orders_summary", "customer_summary", "loyalty_summary"],
            "should_have_orders": True
        },
        {
            "name": "Customer without Orders (2555880)",
            "customer_id": "2555880", 
            "expected_types": ["customer_summary", "loyalty_summary"],
            "should_have_orders": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        payload = {
            "input": f"show last two order and their separate order amount and customer name his loyalty card of customer {test_case['customer_id']}",
            "chat_history": ["string"],
            "summarize": False,
            "conversationId": f"test_{test_case['customer_id']}"
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Validate response is a list
            if not isinstance(result['output'], list):
                print(f"❌ {test_case['name']}: Response is not a list")
                continue
            
            # Validate summary types
            actual_types = [item.get('type') for item in result['output'] if isinstance(item, dict)]
            missing_types = [t for t in test_case['expected_types'] if t not in actual_types]
            unexpected_types = [t for t in actual_types if t not in test_case['expected_types']]
            
            if missing_types:
                print(f"❌ {test_case['name']}: Missing types: {missing_types}")
            elif unexpected_types:
                print(f"⚠️  {test_case['name']}: Unexpected types: {unexpected_types}")
            else:
                print(f"✅ {test_case['name']}: All expected types present")
            
            # Validate orders logic
            if test_case['should_have_orders']:
                orders_summary = next((item for item in result['output'] if item.get('type') == 'orders_summary'), None)
                if orders_summary and orders_summary.get('orders'):
                    print(f"✅ {test_case['name']}: Has orders as expected")
                else:
                    print(f"❌ {test_case['name']}: Expected orders but none found")
            else:
                orders_summary = next((item for item in result['output'] if item.get('type') == 'orders_summary'), None)
                if orders_summary and not orders_summary.get('orders'):
                    print(f"✅ {test_case['name']}: Correctly has empty orders array")
                elif not orders_summary:
                    print(f"✅ {test_case['name']}: Correctly has no orders_summary")
                else:
                    print(f"❌ {test_case['name']}: Has orders when none expected")
        else:
            print(f"❌ {test_case['name']}: Request failed with status {response.status_code}")

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Customer Tests")
    print("="*60)
    
    # Test individual cases with detailed output
    test_customer_with_orders()
    test_customer_without_orders()
    
    # Test structure validation
    test_response_structure_validation()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 