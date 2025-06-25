from agent.config_loader import config_loader

def validate_intent_registry():
    schema = config_loader.get_schema()
    registry = config_loader.get_intent_registry()
    schema_tables = schema.get('tables', {})
    errors = []
    for intent, subintents in registry.items():
        for sub, cfg in subintents.items():
            db = cfg.get('db')
            table = cfg.get('table')
            fields = cfg.get('fields', [])
            if table not in schema_tables:
                errors.append(f"Unknown table '{table}' for {intent}.{sub}")
                continue
            schema_fields = schema_tables[table].get('fields', [])
            for field in fields:
                if field not in schema_fields:
                    errors.append(f"Unknown field '{field}' in table '{table}' for {intent}.{sub}")
    if errors:
        print("Validation errors:")
        for e in errors:
            print("  -", e)
    else:
        print("All intent mappings are valid!")

if __name__ == "__main__":
    validate_intent_registry() 