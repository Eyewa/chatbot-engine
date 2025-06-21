# tools/es_tool.py
from typing import List
from langchain_core.tools import tool

@tool
def search_products_by_filters(category: str = None, price_min: float = None, price_max: float = None) -> List[dict]:
    """Search for products by category and/or price range. Returns a list of matching products."""
    # Stub implementation: return dummy products based on filters
    products = []
    # Example dummy data
    catalog = [
        {"id": "P100", "name": "Smartphone", "category": "electronics", "price": 299.99},
        {"id": "P200", "name": "Laptop", "category": "electronics", "price": 999.00},
        {"id": "P300", "name": "Coffee Mug", "category": "home", "price": 12.50}
    ]
    for item in catalog:
        # Filter by category if provided
        if category and item["category"].lower() != category.lower():
            continue
        # Filter by price range if provided
        if price_min is not None and item["price"] < price_min:
            continue
        if price_max is not None and item["price"] > price_max:
            continue
        products.append(item)
    return products
