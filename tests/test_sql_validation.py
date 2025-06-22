import pytest
from tools.sql_tool import _validate_query

def test_validate_query_rejects_invalid_column():
    schema = {"customer_entity": ["entity_id", "email"]}
    with pytest.raises(ValueError):
        _validate_query("SELECT ce.loyalty_card FROM customer_entity as ce", schema)

def test_validate_query_allows_valid_columns():
    schema = {"customer_entity": ["entity_id", "email"]}
    assert _validate_query("SELECT ce.email FROM customer_entity ce", schema) is None
