"""Utilities for generating LangChain custom_table_info strings."""

import json
import re
from pathlib import Path
from typing import List

import yaml

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML isn't installed
    import simple_yaml as yaml


def load_custom_table_info(
    schema_path: str, table_names: List[str], db: str = "live"
) -> str:
    """Load schema.yaml and build a custom_table_info string for selected tables."""
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Parse YAML or fallback to JSON if PyYAML is unavailable
    if yaml is not None and path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
    else:
        with path.with_suffix(".json").open("r", encoding="utf-8") as f:
            schema = json.load(f)

    tables = schema.get(db, {}).get("tables", {})
    lines: List[str] = []
    for name in table_names:
        meta = tables.get(name)
        if not meta:
            continue

        desc = meta.get("description", f"{name} table.")
        fields = ", ".join(meta.get("fields", []))

        lines.append(f"Table: {name}")
        lines.append(f"Description: {desc}")
        lines.append(f"Fields: {fields}")

        joins = meta.get("joins", [])
        if joins:
            lines.append("Joins:")
            for join in joins:
                lines.append(
                    f"  - {name}.{join['from_field']} \u2192 {join['to_table']}.{join['to_field']}"
                )
        lines.append("")

    return "\n".join(lines).strip()


def extract_table_names(sql: str):
    # Very basic: finds all table names after FROM or JOIN
    return re.findall(r"(?:from|join)\s+([\w_]+)", sql, re.IGNORECASE)
