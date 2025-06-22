import pytest
from agent.schema_validator import validate_schema_yaml

@pytest.fixture
def schema():
    return validate_schema_yaml("config/schema/schema.yaml")

def test_tables_exist(schema):
    assert "tables" in schema
    assert isinstance(schema["tables"], dict)

@pytest.mark.parametrize("table", ["sales_order", "customer_entity", "customer_wallet"])
def test_table_fields_nonempty(schema, table):
    assert table in schema["tables"]
    assert schema["tables"][table]["fields"]

@pytest.mark.parametrize("table", ["sales_order", "customer_wallet"])
def test_join_structure_valid(schema, table):
    joins = schema["tables"].get(table, {}).get("joins", [])
    for join in joins:
        assert "from_field" in join and "to_table" in join and "to_field" in join