# agent/prompt_builder.py
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml

class PromptBuilder:
    """Load prompt configuration from YAML files and build system prompts."""

    def __init__(self, base_dir: str = "config"):
        self.response_cfg = self._load_yaml(Path(base_dir) / "templates" / "response_types.yaml")
        self.schema_cfg = self._load_yaml(Path(base_dir) / "schema" / "schema.yaml")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def build_system_prompt(self, db: str = "", allowed_tables: List[str] = None) -> str:
        types = ", ".join(self.response_cfg.keys())
        lines = [
            "You are Winkly — an intelligent, structured response assistant.",
            f"Always respond using JSON with a top-level 'type'. Valid types are: {types}.",
        ]
        if allowed_tables:
            lines.append(f"You are using the `{db}` database with access to: {', '.join(allowed_tables)}.")
        join_lines = []
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if allowed_tables and table not in allowed_tables:
                continue
            for join in meta.get("joins", []):
                join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
        if join_lines:
            lines.append("Use these known joins when needed:")
            lines.extend(join_lines)
        return "\n".join(lines)

    def build_custom_table_info(self) -> Dict[str, str]:
        table_info = {}
        for table, meta in self.schema_cfg.get("tables", {}).items():
            description = meta.get("description", f"{table} table.")
            fields = meta.get("fields", [])
            field_list = ", ".join(fields)
            table_info[table] = f"{table}: {description}\nColumns: {field_list}"
        return table_info

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}