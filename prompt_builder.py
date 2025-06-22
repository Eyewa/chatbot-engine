import json
from pathlib import Path
from typing import Any, Dict

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

    def build_system_prompt(self) -> str:
        types = ", ".join(self.response_cfg.keys())
        joins = []
        for table, info in self.schema_cfg.get("tables", {}).items():
            if isinstance(info, dict) and "join" in info:
                joins.append(f"{table} -> {info['join']}")
        joins_str = "; ".join(joins)
        prompt = (
            "You are Winkly — an intelligent, frontend-aware AI assistant. "
            "Always respond with a single JSON object containing a 'type' field. "
            f"Valid types: {types}. "
            "Use the following joins when needed: " + joins_str
        )
        return prompt

    def translate_freeform(self, text: str) -> Dict[str, Any]:
        """Convert free-form text into a valid response object."""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") in self.response_cfg:
                return data
        except Exception:
            pass
        return {"type": "text_response", "message": text.strip()}
