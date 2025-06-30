import requests
import pytest

API_URL = "http://localhost:8000/chat/"

@pytest.mark.integration
@pytest.mark.parametrize("customer_id, expect_orders", [
    (1338787, True),   # Customer with orders
    (2555880, False),  # Customer with no orders
])
def test_customer_summary_and_orders(customer_id, expect_orders):
    """
    Integration test requiring a running FastAPI server.
    Run with: pytest -m integration
    Or start the server first: python main_new.py
    """
    try:
        payload = {
            "input": f"show last two order and their separate order amount and customer name his loyalty card of customer {customer_id}",
            "chat_history": [],
            "summarize": False,
            "conversationId": None
        }
        response = requests.post(API_URL, json=payload, timeout=120)
        assert response.status_code == 200
        data = response.json()["output"]

        # Always expect a customer_summary
        found_customer = any(
            (isinstance(item, dict) and item.get("type") == "customer_summary")
            for item in (data if isinstance(data, list) else [data])
        )
        assert found_customer, f"customer_summary not found in response for customer {customer_id}"

        # Always expect orders_summary (even if empty)
        found_orders_summary = any(
            (isinstance(item, dict) and item.get("type") == "orders_summary")
            for item in (data if isinstance(data, list) else [data])
        )
        assert found_orders_summary, f"orders_summary not found for customer {customer_id}"

        # Check if orders array has actual orders
        orders_summary = next(
            (item for item in (data if isinstance(data, list) else [data]) 
             if isinstance(item, dict) and item.get("type") == "orders_summary"), 
            None
        )
        has_orders = orders_summary and orders_summary.get("orders") and len(orders_summary.get("orders", [])) > 0
        
        if expect_orders:
            assert has_orders, f"orders_summary has no orders for customer {customer_id} (should have orders)"
        else:
            assert not has_orders, f"orders_summary has orders for customer {customer_id} (should have no orders)"
    except requests.exceptions.ConnectionError:
        pytest.skip("FastAPI server not running. Start with: python main_new.py") 