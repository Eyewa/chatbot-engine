# schema_utils.py

try:
    import yaml  # type: ignore
except Exception:  # PyYAML may be unavailable
    yaml = None
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

def validate_schema_yaml(path: str) -> Dict[str, Any]:
    """Validates the structure of schema.yaml and returns parsed content."""
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"❌ Schema file not found: {path}")

    if yaml is not None:
        with schema_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        json_path = schema_path.with_suffix(".json")
        with json_path.open("r", encoding="utf-8") as jf:
            data = json.load(jf)

    assert isinstance(data, dict), "❌ Top-level YAML structure must be a dictionary"
    assert "tables" in data, "❌ Missing 'tables' key in schema"
    assert isinstance(data["tables"], dict), "❌ 'tables' must be a dictionary"

    for table, meta in data["tables"].items():
        assert isinstance(meta, dict), f"❌ Table '{table}' definition must be a dictionary"
        assert "fields" in meta, f"❌ Missing 'fields' for table: {table}"
        assert isinstance(meta["fields"], list), f"❌ 'fields' in {table} must be a list"
        for field in meta["fields"]:
            assert isinstance(field, str), f"❌ Each field in {table} must be a string"

        if "joins" in meta:
            assert isinstance(meta["joins"], list), f"❌ 'joins' in {table} must be a list"
            for join in meta["joins"]:
                assert isinstance(join, dict), f"❌ Each join in {table} must be a dictionary"
                assert all(k in join for k in ["from_field", "to_table", "to_field"]), (
                    f"❌ Each join in {table} must contain 'from_field', 'to_table', and 'to_field'"
                )

    logging.info("✅ schema.yaml is valid and well-structured!")
    return data

def generate_openapi_schema(schema_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generates an OpenAPI-compatible schema from YAML schema."""
    properties = {}
    for table, meta in schema_data.get("tables", {}).items():
        properties[table] = {
            "type": "object",
            "description": meta.get("description", ""),
            "properties": {field: {"type": "string"} for field in meta.get("fields", [])}
        }

    return {
        "openapi": "3.0.0",
        "info": {"title": "Eyewa Schema", "version": "1.0.0"},
        "components": {"schemas": properties}
    }

def generate_custom_table_info(schema_data: Dict[str, Any]) -> Dict[str, str]:
    """Creates LangChain-compatible custom_table_info from YAML schema."""
    table_info = {}
    for table, meta in schema_data.get("tables", {}).items():
        fields = ", ".join(meta.get("fields", []))
        desc = meta.get("description", f"{table} table.")
        table_info[table] = f"{table}: {desc}\nColumns: {fields}"
    return table_info

def generate_prompt_context(schema_data: Dict[str, Any], allowed_tables: Optional[list] = None) -> str:
    """Creates a concise join summary for prompting agents."""
    join_lines = []
    for table, meta in schema_data.get("tables", {}).items():
        if allowed_tables and table not in allowed_tables:
            continue
        for join in meta.get("joins", []):
            if allowed_tables and join.get("to_table") not in allowed_tables:
                continue
            join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
    return "Use these joins when needed:\n" + "\n".join(join_lines)

# --------------------------
# CLI testing utility
# --------------------------

if __name__ == "__main__":
    schema = validate_schema_yaml("config/schema/schema.yaml")

    print("\n🔧 LangChain custom_table_info:")
    print(json.dumps(generate_custom_table_info(schema), indent=2))

    print("\n📘 OpenAPI schema:")
    print(json.dumps(generate_openapi_schema(schema), indent=2))

    print("\n📎 Prompt join summary:")
    print(generate_prompt_context(schema))
