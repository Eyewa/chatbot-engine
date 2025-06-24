import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML isn't installed
    import simple_yaml as yaml


class PromptBuilder:
    """Load prompt configuration from YAML files with hot-reload support."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _timestamps: Dict[str, float] = {}

    def __init__(self, base_dir: str = "config"):
        self.base_dir = Path(base_dir)
        self.response_cfg = self._load_yaml(
            self.base_dir / "templates" / "response_types.yaml"
        )
        self.schema_cfg = self._load_yaml(self.base_dir / "schema" / "schema.yaml")

    @classmethod
    def _load_yaml(cls, path: Path) -> Dict[str, Any]:
        key = str(path.resolve())
        try:
            mtime = path.stat().st_mtime
            if key not in cls._timestamps or cls._timestamps[key] < mtime:
                if yaml is not None:
                    with path.open("r", encoding="utf-8") as f:
                        cls._cache[key] = yaml.safe_load(f) or {}
                else:
                    json_path = path.with_suffix(".json")
                    with json_path.open("r", encoding="utf-8") as f:
                        cls._cache[key] = json.load(f)
                cls._timestamps[key] = mtime
                logging.info(f"🔄 Reloaded config file: {path.name}")
            return cls._cache[key]
        except Exception as e:
            logging.warning(f"⚠️ Failed to load YAML: {path.name} — {e}")
            return {}

    def build_system_prompt(
        self, db: str = "", allowed_tables: Optional[List[str]] = None, extra_examples: Optional[list] = None
    ) -> str:
        types = ", ".join(self.response_cfg.keys())
        lines = [
            "You are Winkly, a structured data assistant for customer and order data. Use only the schema and rules below.",
            "Rules:",
            "- Use only the tables and fields listed below. Never invent or guess columns, tables, or values.",
            "- For order history, use sales_order.increment_id as the order ID. Only use customer_loyalty_ledger.order_id for loyalty transactions, and join to sales_order.increment_id if order details are needed.",
            "- If only loyalty card details are needed, do not join the ledger table.",
            "- Never join across _live and _common databases.",
            "- Do NOT hallucinate tables, fields, or values. Only use what is listed.",
            "- Always return valid JSON with double quotes. No code blocks, no SQL, no prose.",
            "- Output must match one of the allowed response_types. Top-level key must be 'type'.",
            f"- Valid types: {types}.",
        ]
        if allowed_tables:
            lines.append(f"Database: `{db}`. Allowed tables: {', '.join(allowed_tables)}.")
            table_info = []
            for table in allowed_tables:
                meta = self.schema_cfg.get("tables", {}).get(table, {})
                custom = meta.get("customInfo")
                if custom:
                    table_info.append(f"- `{table}`: {custom}")
                else:
                    fields = meta.get("fields", [])
                    field_list = ", ".join(fields)
                    table_info.append(f"- `{table}`: {field_list}")
            if table_info:
                lines.append("Allowed tables and fields:")
                lines.extend(table_info)
        # Add a single, clear join example if available
        if extra_examples and len(extra_examples) > 0:
            lines.append("Example join:")
            lines.append(extra_examples[0])
        join_lines = []
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if allowed_tables and table not in allowed_tables:
                continue
            for join in meta.get("joins", []):
                if join.get("to_table") in allowed_tables:
                    join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
        if join_lines:
            lines.append("Known joins:")
            lines.extend(join_lines)
        return "\n".join(lines)

    def build_custom_table_info(
        self, allowed_tables: Optional[List[str]] = None
    ) -> Dict[str, str]:
        info = {}
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if allowed_tables and table not in allowed_tables:
                continue
            custom = meta.get("customInfo")
            if custom:
                info[table] = custom
                continue

            description = meta.get("description", f"{table} table.")
            fields = meta.get("fields", [])
            field_list = ", ".join(fields)
            joins = meta.get("joins", [])
            join_info = "; ".join(
                f"{table}.{j['from_field']} -> {j['to_table']}.{j['to_field']}" for j in joins
            )
            text = f"{table}: {description}\nColumns: {field_list}"
            if join_info:
                text += f"\nJoins: {join_info}"
            info[table] = text
        return info

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        logging.debug(f"📦 Attempting to parse response: {text!r}")

        # Try JSON directly
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except json.JSONDecodeError as e:
            logging.warning(f"⚠️ Direct JSON parse failed: {e}")

        # Try parsing as Python dict literal
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "output" in parsed:
                inner = parsed["output"]
                if isinstance(inner, str):
                    try:
                        json_data = json.loads(inner)
                        if json_data.get("type") in self.response_cfg:
                            return json_data
                    except Exception as inner_e:
                        logging.error(f"❌ Failed to parse inner output: {inner_e}")
        except Exception as e2:
            logging.error(f"❌ literal_eval failed: {e2}")

        return {"type": "text_response", "message": text.strip()}

    def assert_valid_schema(self):
        assert isinstance(self.schema_cfg, dict), "Schema must be a dictionary"
        for table, meta in self.schema_cfg.get("tables", {}).items():
            assert "fields" in meta, f"Missing fields for table {table}"
            assert isinstance(
                meta.get("fields"), list
            ), f"Fields for table {table} must be a list"
            if "joins" in meta:
                assert isinstance(
                    meta["joins"], list
                ), f"Joins for table {table} must be a list"
                for join in meta["joins"]:
                    assert (
                        "from_field" in join
                        and "to_table" in join
                        and "to_field" in join
                    ), f"Invalid join format in {table}"
