# tools/sql_tool.py
from langchain_core.tools import tool

@tool
def query_order_status(order_id: str) -> str:
    """Lookup the status of an order by its ID."""
    # Stub implementation: return a fake status for demonstration
    if order_id == "1001":
        return "Order 1001 is currently *Shipped*."
    elif order_id == "1002":
        return "Order 1002 is currently *Processing*."
    else:
        return f"Order {order_id} status is *Unknown*."
