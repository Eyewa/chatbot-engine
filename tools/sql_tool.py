from langchain.tools import tool

@tool
def query_order_status(order_id: str) -> str:
    """Return the status of an order given its ID."""
    # Placeholder logic for demonstration purposes
    if not order_id:
        return "No order ID provided."
    return f"Status for order {order_id}: Shipped"
