from agent.config_loader import config_loader

def validate_intent_registry():
    schema = config_loader.get_schema()
    registry = config_loader.get_intent_registry()
    errors = []
    for intent, subintents in registry.items():
        for sub, cfg in subintents.items():
            db = cfg.get('db', 'live')
            table = cfg.get('table')
            fields = cfg.get('fields', [])
            db_tables = schema.get(db, {}).get('tables', {})
            if table not in db_tables:
                errors.append(f"Unknown table '{table}' for {intent}.{sub} in db '{db}'")
                continue
            schema_fields = db_tables[table].get('fields', [])
            for field in fields:
                if field not in schema_fields:
                    errors.append(f"Unknown field '{field}' in table '{table}' for {intent}.{sub} in db '{db}'")
    if errors:
        print("Validation errors:")
        for e in errors:
            print("  -", e)
    else:
        print("All intent mappings are valid!")

if __name__ == "__main__":
    validate_intent_registry() 