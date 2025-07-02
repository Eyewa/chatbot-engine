import pytest
from tools.custom_table_info_loader import extract_table_names
from agent.core.agent_router import _combine_responses


def test_extract_table_names_simple():
    sql = "SELECT * FROM sales_order"
    assert extract_table_names(sql) == ["sales_order"]


def test_extract_table_names_with_join():
    sql = (
        "SELECT so.increment_id FROM sales_order so "
        "JOIN customer_entity ce ON so.customer_id = ce.entity_id"
    )
    names = extract_table_names(sql)
    assert set(names) == {"sales_order", "customer_entity"}


def test_extract_table_names_complex():
    sql = (
        "SELECT so.increment_id, ce.firstname FROM sales_order AS so "
        "INNER JOIN customer_entity AS ce ON so.customer_id = ce.entity_id "
        "LEFT JOIN sales_order_payment sop ON so.entity_id = sop.parent_id"
    )
    names = extract_table_names(sql)
    assert set(names) == {
        "sales_order",
        "customer_entity",
        "sales_order_payment",
    }


def test_combine_responses_filters_and_merges():
    # Only allowed fields will be present in the output per response_types.yaml
    live = '{"type": "orders_summary", "orders": [{"order_id": 1, "grand_total": 100}]}'
    common = '{"type": "loyalty_summary", "card_number": "777", "first_name": "John", "activation_date": "2023-01-01", "expiry_date": "2024-01-01", "status": "active"}'

    out = _combine_responses(live, common)
    result = out
    assert result is not None
    assert isinstance(result, list)
    assert any(r.get("type") == "orders_summary" for r in result)
    assert any(r.get("type") == "loyalty_summary" for r in result)
    # Optionally, check the fields in each summary
    orders = next(r for r in result if r.get("type") == "orders_summary")
    loyalty = next(r for r in result if r.get("type") == "loyalty_summary")
    assert "orders" in orders
    assert "card_number" in loyalty
    assert loyalty["card_number"] == "777"
