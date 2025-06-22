import pytest
from tools.sql_tool import extract_table_names


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
