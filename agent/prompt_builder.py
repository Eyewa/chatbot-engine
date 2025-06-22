import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import yaml

class PromptBuilder:
    """Load prompt configuration from YAML files with hot-reload support."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _timestamps: Dict[str, float] = {}

    def __init__(self, base_dir: str = "config"):
        self.base_dir = Path(base_dir)
        self.response_cfg = self._load_yaml(self.base_dir / "templates" / "response_types.yaml")
        self.schema_cfg = self._load_yaml(self.base_dir / "schema" / "schema.yaml")

    @classmethod
    def _load_yaml(cls, path: Path) -> Dict[str, Any]:
        key = str(path.resolve())
        try:
            mtime = path.stat().st_mtime
            if key not in cls._timestamps or cls._timestamps[key] < mtime:
                with path.open("r", encoding="utf-8") as f:
                    cls._cache[key] = yaml.safe_load(f) or {}
                    cls._timestamps[key] = mtime
                    logging.info(f"🔄 Reloaded config file: {path.name}")
            return cls._cache[key]
        except Exception as e:
            logging.warning(f"⚠️ Failed to load YAML: {path.name} — {e}")
            return {}

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

    def build_custom_table_info_filtered(self, allowed_tables: List[str]) -> Dict[str, str]:
        """Return custom_table_info for only a subset of allowed tables."""
        full_info = self.build_custom_table_info()
        return {table: info for table, info in full_info.items() if table in allowed_tables}

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}
