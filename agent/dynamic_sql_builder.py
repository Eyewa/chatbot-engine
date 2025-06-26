import yaml
import os

def load_schema(path=None, db_key="live"):
    if path is None:
        path = os.path.join("config", "schema", "schema.yaml")
    with open(path) as f:
        schema = yaml.safe_load(f)
    return schema[db_key]["tables"]

def get_field_alias(table, field, schema):
    aliases = schema[table].get("field_aliases", {})
    real = aliases.get(field)
    if real:
        return real if isinstance(real, list) else [real]
    return [field]

def find_field_table(field, schema):
    for table, meta in schema.items():
        if field in meta.get("fields", []):
            return table, field
        aliases = meta.get("field_aliases", {})
        for alias, real in aliases.items():
            if (isinstance(real, list) and field in real) or field == real:
                return table, real if isinstance(real, str) else field
    return None, None

def find_join_path(schema, from_table, to_table):
    for join in schema[from_table].get("joins", []):
        if join["to_table"] == to_table:
            return join
    return None

def build_dynamic_sql(user_fields, main_table, filters, limit, schema):
    select_fields = []
    joins = []
    used_tables = {main_table}
    join_clauses = []

    for field in user_fields:
        real_fields = get_field_alias(main_table, field, schema)
        for rf in real_fields:
            if rf in schema[main_table]["fields"]:
                select_fields.append(f"{main_table}.{rf} AS {rf}")
            else:
                target_table, _ = find_field_table(rf, schema)
                if not target_table:
                    raise Exception(f"Field {rf} not found in schema.")
                join_info = find_join_path(schema, main_table, target_table)
                if join_info:
                    jt = join_info["to_table"]
                    if jt not in used_tables:
                        join_clauses.append(f"JOIN {jt} ON {main_table}.{join_info['from_field']} = {jt}.{join_info['to_field']}")
                        used_tables.add(jt)
                    select_fields.append(f"{jt}.{rf} AS {rf}")
                else:
                    raise Exception(f"No join path from {main_table} to {target_table} for field {rf}.")

    sql = f"SELECT {', '.join(select_fields)} FROM {main_table} "
    if join_clauses:
        sql += " ".join(join_clauses) + " "
    if filters:
        where_clause = " AND ".join(f"{main_table}.{k} = {repr(v)}" for k, v in filters.items())
        sql += f"WHERE {where_clause} "
    sql += f"ORDER BY {main_table}.created_at DESC LIMIT {limit};"
    return sql 