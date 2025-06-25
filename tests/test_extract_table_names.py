import pytest
from tools.custom_table_info_loader import extract_table_names


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
