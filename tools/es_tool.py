from langchain.tools import tool

@tool
def search_products_by_filters(category: str | None = None, price_max: float | None = None) -> str:
    """Search for products given optional filters."""
    filters = []
    if category:
        filters.append(f"category={category}")
    if price_max is not None:
        filters.append(f"price<=${price_max}")
    if not filters:
        return "Showing all products"
    return f"Products filtered by: {', '.join(filters)}"
