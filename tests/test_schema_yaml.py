import pytest

from agent.core.schema_validator import validate_schema_yaml


@pytest.fixture
def schema():
    return validate_schema_yaml("config/schema/schema.yaml")


def test_tables_exist(schema):
    assert "live" in schema
    assert "tables" in schema["live"]
    assert isinstance(schema["live"]["tables"], dict)


@pytest.mark.parametrize(
    "db,table",
    [
        ("live", "sales_order"),
        ("live", "customer_entity"),
        ("common", "customer_wallet"),
    ],
)
def test_table_fields_nonempty(schema, db, table):
    assert table in schema[db]["tables"]
    assert schema[db]["tables"][table]["fields"]


@pytest.mark.parametrize("table", ["sales_order", "customer_wallet"])
def test_join_structure_valid(schema, table):
    # Check both live and common
    db = "live" if table in schema["live"]["tables"] else "common"
    joins = schema[db]["tables"].get(table, {}).get("joins", [])
    for join in joins:
        assert "from_field" in join and "to_table" in join and "to_field" in join


def test_schema_integrity(schema):
    for db in ("live", "common"):
        for table, meta in schema.get(db, {}).get("tables", {}).items():
            assert "fields" in meta and isinstance(
                meta["fields"], list
            ), f"Missing or invalid fields in {table}"
            for join in meta.get("joins", []):
                assert "from_field" in join, f"Missing from_field in join for {table}"
                assert "to_table" in join, f"Missing to_table in join for {table}"
                assert "to_field" in join, f"Missing to_field in join for {table}"
            if "customInfo" in meta:
                assert isinstance(meta["customInfo"], str)
