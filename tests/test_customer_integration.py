import requests
import pytest

API_URL = "http://localhost:8000/chat/"

@pytest.mark.parametrize("customer_id, expect_orders", [
    (1338787, True),   # Customer with orders
    (2555880, False),  # Customer with no orders
])
def test_customer_summary_and_orders(customer_id, expect_orders):
    payload = {
        "input": f"show last two order and their separate order amount and customer name his loyalty card of customer {customer_id}",
        "chat_history": [],
        "summarize": False,
        "conversationId": None
    }
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 200
    data = response.json()["output"]

    # Always expect a customer_summary
    found_customer = any(
        (isinstance(item, dict) and item.get("type") == "customer_summary")
        for item in (data if isinstance(data, list) else [data])
    )
    assert found_customer, f"customer_summary not found in response for customer {customer_id}"

    # Orders summary presence depends on test case
    found_orders = any(
        (isinstance(item, dict) and item.get("type") == "orders_summary" and item.get("orders"))
        for item in (data if isinstance(data, list) else [data])
    )
    if expect_orders:
        assert found_orders, f"orders_summary not found for customer {customer_id} (should have orders)"
    else:
        assert not found_orders, f"orders_summary found for customer {customer_id} (should have no orders)" 