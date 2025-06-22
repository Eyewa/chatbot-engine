import pytest
from tools.sql_tool import _validate_query

def test_validate_query_rejects_invalid_column():
    schema = {"customer_entity": ["entity_id", "email"]}
    with pytest.raises(ValueError):
        _validate_query("SELECT ce.loyalty_card FROM customer_entity as ce", schema)

def test_validate_query_allows_valid_columns():
    schema = {"customer_entity": ["entity_id", "email"]}
    assert _validate_query("SELECT ce.email FROM customer_entity ce", schema) is None


def test_validate_query_resolves_aliases():
    schema = {
        "sales_order": ["entity_id", "customer_id"],
        "customer_entity": ["entity_id"],
    }
    q = (
        "SELECT so.entity_id FROM sales_order so "
        "JOIN customer_entity ce ON so.customer_id = ce.entity_id"
    )
    assert _validate_query(q, schema) is None


def test_validate_query_ignores_select_aliases():
    schema = {"sales_order": ["entity_id"]}
    q = "SELECT so.entity_id AS order_id FROM sales_order so"
    assert _validate_query(q, schema) is None
