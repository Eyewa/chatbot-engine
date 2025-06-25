"""Configuration loader for chatbot engine."""

import os
import logging
from typing import Dict, Any, List, Optional

try:
    import yaml
except Exception:
    import simple_yaml as yaml


class ConfigLoader:
    """Centralized configuration loader for the chatbot engine."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self._cache: Dict[str, Any] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files."""
        config_files = [
            "intent_registry.yaml",
            "query_routing.yaml",
            "templates/response_types.yaml",
            "schema/schema.yaml"
        ]
        
        for config_file in config_files:
            self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load a single configuration file."""
        config_path = os.path.join(self.config_dir, config_file)
        
        if not os.path.exists(config_path):
            logging.warning(f"⚠️ Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                self._cache[config_file] = config
                logging.info(f"✅ Loaded config: {config_file}")
                return config
        except Exception as e:
            logging.error(f"❌ Failed to load config {config_file}: {e}")
            return {}
    
    def get_config(self, config_file: str) -> Dict[str, Any]:
        """Get a specific configuration."""
        return self._cache.get(config_file, {})
    
    def reload_config(self, config_file: Optional[str] = None):
        """Reload configuration files."""
        if config_file:
            self._load_config(config_file)
        else:
            self._cache.clear()
            self._load_all_configs()
    
    # Convenience methods for specific configs
    def get_intent_registry(self) -> Dict[str, Any]:
        """Get intent registry configuration."""
        return self.get_config("intent_registry.yaml")
    
    def get_query_routing(self) -> Dict[str, Any]:
        """Get query routing configuration."""
        return self.get_config("query_routing.yaml")
    
    def get_response_types(self) -> Dict[str, Any]:
        """Get response types configuration."""
        return self.get_config("templates/response_types.yaml")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema configuration."""
        return self.get_config("schema/schema.yaml")
    
    # Query routing specific methods
    def get_databases(self) -> Dict[str, Any]:
        """Get database definitions."""
        routing_config = self.get_query_routing()
        return routing_config.get("databases", {})
    
    def get_classification_rules(self) -> Dict[str, Any]:
        """Get classification rules."""
        routing_config = self.get_query_routing()
        return routing_config.get("classification_rules", {})
    
    def get_query_scoping(self) -> Dict[str, Any]:
        """Get query scoping rules."""
        routing_config = self.get_query_routing()
        return routing_config.get("query_scoping", {})
    
    def get_intent_database_mapping(self) -> Dict[str, List[str]]:
        """Get intent to database mapping."""
        routing_config = self.get_query_routing()
        return routing_config.get("intent_database_mapping", {})
    
    def get_fallback_queries(self) -> Dict[str, str]:
        """Get fallback queries."""
        scoping_config = self.get_query_scoping()
        return scoping_config.get("fallback_queries", {})
    
    def get_both_databases_config(self) -> Dict[str, str]:
        """Get both databases configuration."""
        scoping_config = self.get_query_scoping()
        return scoping_config.get("both_databases", {})
    
    def get_keywords(self, keyword_type: str) -> List[str]:
        """Get keywords of a specific type."""
        rules = self.get_classification_rules()
        return rules.get("keywords", {}).get(keyword_type, [])
    
    def needs_ledger_data(self, query: str) -> bool:
        """Check if query needs ledger data based on keywords."""
        requires_ledger_keywords = self.get_keywords("requires_ledger")
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in requires_ledger_keywords)
    
    def get_database_tables(self, db_name: str) -> List[str]:
        """Get tables for a specific database."""
        databases = self.get_databases()
        return databases.get(db_name, {}).get("tables", [])


# Global config loader instance
config_loader = ConfigLoader() 