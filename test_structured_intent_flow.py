from agent.intent_to_sql import intent_to_sql
from agent.config_loader import config_loader
import pytest

# Sample intent for a simple join
simple_intent = {
    "tables": ["sales_order", "customer_entity"],
    "fields": [
        {"table": "sales_order", "field": "increment_id"},
        {"table": "sales_order", "field": "grand_total"},
        {"table": "customer_entity", "field": "firstname"},
        {"table": "customer_entity", "field": "lastname"}
    ],
    "joins": [
        {
            "from_table": "sales_order",
            "from_field": "customer_id",
            "to_table": "customer_entity",
            "to_field": "entity_id"
        }
    ],
    "filters": {"customer_entity.entity_id": 1338787},
    "limit": 2,
    "order_by": "sales_order.created_at DESC"
}

# Sample intent with aggregation
agg_intent = {
    "tables": ["sales_order"],
    "fields": [
        {"table": "sales_order", "field": "customer_id"},
        {"table": "sales_order", "field": "grand_total"},
        {"table": "sales_order", "field": "entity_id"}
    ],
    "joins": [],
    "filters": {},
    "limit": None,
    "order_by": None,
    "aggregation": {
        "function": "SUM",
        "field": "grand_total",
        "group_by": "customer_id"
    }
}

# Sample intent with an invalid field
invalid_intent = {
    "tables": ["sales_order"],
    "fields": [
        {"table": "sales_order", "field": "not_a_field"}
    ],
    "joins": [],
    "filters": {},
    "limit": 1,
    "order_by": None
}

# Sample intent with an invalid join
invalid_join_intent = {
    "tables": ["sales_order", "customer_entity"],
    "fields": [
        {"table": "sales_order", "field": "increment_id"},
        {"table": "customer_entity", "field": "firstname"}
    ],
    "joins": [
        {
            "from_table": "sales_order",
            "from_field": "not_a_field",
            "to_table": "customer_entity",
            "to_field": "entity_id"
        }
    ],
    "filters": {},
    "limit": 1,
    "order_by": None
}

# Sample intent with an invalid filter
invalid_filter_intent = {
    "tables": ["sales_order"],
    "fields": [
        {"table": "sales_order", "field": "increment_id"}
    ],
    "joins": [],
    "filters": {"sales_order.not_a_field": 123},
    "limit": 1,
    "order_by": None
}

def test_intent_to_sql():
    print("\n--- Simple Join Example ---")
    try:
        sql = intent_to_sql(simple_intent)
        print(sql)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Aggregation Example ---")
    try:
        sql = intent_to_sql(agg_intent)
        print(sql)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Aggregation with HAVING Example ---")
    try:
        having_intent = agg_intent.copy()
        having_intent["aggregation"] = {
            "function": "SUM",
            "field": "grand_total",
            "group_by": "customer_id",
            "having": "SUM(grand_total) > 1000"
        }
        sql = intent_to_sql(having_intent)
        print(sql)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Multiple Joins Example ---")
    try:
        multi_join_intent = {
            "tables": ["sales_order", "customer_entity", "sales_order_payment"],
            "fields": [
                {"table": "sales_order", "field": "increment_id"},
                {"table": "customer_entity", "field": "firstname"},
                {"table": "sales_order_payment", "field": "amount_paid"}
            ],
            "joins": [
                {
                    "from_table": "sales_order",
                    "from_field": "customer_id",
                    "to_table": "customer_entity",
                    "to_field": "entity_id"
                },
                {
                    "from_table": "sales_order_payment",
                    "from_field": "parent_id",
                    "to_table": "sales_order",
                    "to_field": "entity_id"
                }
            ],
            "filters": {"customer_entity.entity_id": 1338787},
            "limit": 1,
            "order_by": None
        }
        sql = intent_to_sql(multi_join_intent)
        print(sql)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Invalid Aggregation Field Example ---")
    try:
        bad_agg_intent = agg_intent.copy()
        bad_agg_intent["aggregation"] = {
            "function": "SUM",
            "field": "not_a_field",
            "group_by": "customer_id"
        }
        sql = intent_to_sql(bad_agg_intent)
        print(sql)
    except Exception as e:
        print(f"Error: {e}")

def test_invalid_field():
    with pytest.raises(ValueError) as excinfo:
        intent_to_sql(invalid_intent)
    assert "Unknown field" in str(excinfo.value)

def test_invalid_join():
    with pytest.raises(ValueError) as excinfo:
        intent_to_sql(invalid_join_intent)
    assert "Invalid join" in str(excinfo.value)

def test_invalid_filter():
    with pytest.raises(ValueError) as excinfo:
        intent_to_sql(invalid_filter_intent)
    assert "Unknown field in filter" in str(excinfo.value)

if __name__ == "__main__":
    test_intent_to_sql() 