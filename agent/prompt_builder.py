import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self.response_cfg = self._load_yaml(self.base_dir / "templates" / "response_types.yaml")
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

    def build_system_prompt(self, db: str = "", allowed_tables: Optional[List[str]] = None) -> str:
        types = ", ".join(self.response_cfg.keys())
        lines = [
            "You are Winkly — an intelligent, structured response assistant.",
            f"Always respond using JSON with a top-level 'type'. Valid types are: {types}.",
        ]
        if allowed_tables:
            lines.append(f"You are using the `{db}` database with access to: {', '.join(allowed_tables)}.")
        else:
            allowed_tables = []

        # Prevent cross-database join hallucination
        lines.append("⚠️ Tables with `_live` and `_common` suffixes belong to separate databases. Never join across them.")

        join_lines = []
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if allowed_tables and table not in allowed_tables:
                continue
            for join in meta.get("joins", []):
                if join.get("to_table") in allowed_tables:
                    join_lines.append(f"{table}.{join['from_field']} → {join['to_table']}.{join['to_field']}")
        if join_lines:
            lines.append("Use these known joins when needed:")
            lines.extend(join_lines)

        # Anti-hallucination
        lines.append(
            "🚫 Do NOT hallucinate tables or fields. Only query the tables and columns listed above."
        )

        if db == "eyewa_common":
            lines.append(
                "You MUST only query these tables and fields. Do NOT assume or guess column names. "
                "No cross-database joins are allowed. If loyalty info is not present, fallback to: "
                "SELECT card_number FROM customer_loyalty_card WHERE customer_id = {X};"
            )

        return "\n".join(lines)

    def build_custom_table_info(self, allowed_tables: Optional[List[str]] = None) -> Dict[str, str]:
        """Return LangChain-compatible table info for allowed tables only."""
        info = {}
        for table, meta in self.schema_cfg.get("tables", {}).items():
            if allowed_tables and table not in allowed_tables:
                continue
            description = meta.get("description", f"{table} table.")
            fields = meta.get("fields", [])
            field_list = ", ".join(fields)
            info[table] = f"{table}: {description}\nColumns: {field_list}"
        return info

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}

    def assert_valid_schema(self):
        assert isinstance(self.schema_cfg, dict), "Schema must be a dictionary"
        for table, meta in self.schema_cfg.get("tables", {}).items():
            assert "fields" in meta, f"Missing fields for table {table}"
            assert isinstance(meta.get("fields"), list), f"Fields for table {table} must be a list"
            if "joins" in meta:
                assert isinstance(meta["joins"], list), f"Joins for table {table} must be a list"
                for join in meta["joins"]:
                    assert "from_field" in join and "to_table" in join and "to_field" in join, f"Invalid join format in {table}"
