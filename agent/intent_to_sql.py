import logging
from typing import Any, Dict, List

from agent.core.config_loader import config_loader


def validate_intent_schema(intent: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    logging.debug(f"[validate_intent_schema] Validating intent: {intent}")
    errors = []
    db = intent.get("db", "live")
    schema_tables = schema.get(db, {}).get("tables", {})
    # Validate tables
    for table in intent.get("tables", []):
        if table not in schema_tables:
            errors.append(f"Unknown table: {table} in db: {db}")
    # Validate fields
    for f in intent.get("fields", []):
        table = f["table"]
        field = f["field"]
        if table not in schema_tables:
            errors.append(f"Unknown table: {table} for field: {field} in db: {db}")
        elif field not in schema_tables[table]["fields"]:
            errors.append(f"Unknown field: {field} in table: {table} in db: {db}")
    # Validate joins
    for join in intent.get("joins", []):
        from_table = join["from_table"]
        from_field = join["from_field"]
        to_table = join["to_table"]
        to_field = join["to_field"]
        if from_table not in schema_tables or to_table not in schema_tables:
            errors.append(
                f"Unknown table in join: {from_table} or {to_table} in db: {db}"
            )
            continue
        found = False
        for j in schema_tables[from_table].get("joins", []):
            if (
                j["to_table"] == to_table
                and j["from_field"] == from_field
                and j["to_field"] == to_field
            ):
                found = True
                break
        if not found:
            errors.append(
                f"Invalid join: {from_table}.{from_field} -> {to_table}.{to_field} in db: {db}"
            )
    # Validate filters
    for k in intent.get("filters", {}).keys():
        if "." in k:
            tbl, fld = k.split(".", 1)
            if tbl not in schema_tables:
                errors.append(f"Unknown table in filter: {tbl}")
            elif fld not in schema_tables[tbl]["fields"]:
                errors.append(f"Unknown field in filter: {fld} for table: {tbl}")
        else:
            # Could be a generic filter, skip
            pass
    # Validate aggregation
    agg = intent.get("aggregation")
    if agg:
        agg_table = intent["tables"][0]
        agg_field = agg["field"]
        if agg_table not in schema_tables:
            errors.append(f"Unknown table for aggregation: {agg_table}")
        elif agg_field not in schema_tables[agg_table]["fields"]:
            errors.append(
                f"Unknown aggregation field: {agg_field} in table: {agg_table}"
            )
    logging.debug(f"[validate_intent_schema] Validation errors: {errors}")
    return errors


def intent_to_sql(intent: Dict[str, Any]) -> str:
    """
    Translate a structured intent dict to a SQL query using schema.yaml.
    Supports aggregation, GROUP BY, and HAVING.
    """
    logging.info(f"[intent_to_sql] Received intent: {intent}")
    schema = config_loader.get_schema()
    db = intent.get("db", "live")
    errors = validate_intent_schema(intent, schema)
    if errors:
        logging.error(f"[intent_to_sql] Validation failed: {errors}")
        raise ValueError("; ".join(errors))
    schema_tables = schema.get(db, {}).get("tables", {})

    # Validate tables
    for table in intent.get("tables", []):
        if table not in schema_tables:
            raise ValueError(f"Unknown table: {table} in db: {db}")

    # Validate fields
    select_fields = []
    for f in intent.get("fields", []):
        table = f["table"]
        field = f["field"]
        if table not in schema_tables or field not in schema_tables[table]["fields"]:
            raise ValueError(f"Unknown field: {field} in table: {table} in db: {db}")
        alias = table[:2]
        select_fields.append(f"{alias}.{field}")

    # Aggregation
    agg = intent.get("aggregation")
    group_by_clause = ""
    having_clause = ""
    if agg:
        func = agg["function"].upper()
        agg_field = agg["field"]
        group_by = agg.get("group_by")
        having = agg.get("having")
        # Validate aggregation field
        agg_table = intent["tables"][0]
        if agg_field not in schema_tables[agg_table]["fields"]:
            raise ValueError(
                f"Unknown aggregation field: {agg_field} in table: {agg_table} in db: {db}"
            )
        select_fields = [f"{group_by}", f"{func}({agg_field}) as total"]
        if group_by:
            group_by_clause = f"GROUP BY {group_by}"
        if having:
            having_clause = f"HAVING {having}"

    # FROM and JOINs
    main_table = intent["tables"][0]
    main_alias = main_table[:2]
    from_clause = f"{main_table} {main_alias}"
    join_clauses = []
    for join in intent.get("joins", []):
        from_table = join["from_table"]
        from_field = join["from_field"]
        to_table = join["to_table"]
        to_field = join["to_field"]
        join_clause = f"JOIN {to_table} {to_table[:2]} ON {from_table[:2]}.{from_field} = {to_table[:2]}.{to_field}"
        join_clauses.append(join_clause)

    # WHERE
    where_clauses = []
    for k, v in intent.get("filters", {}).items():
        if "." in k:
            tbl, fld = k.split(".", 1)
            alias = tbl[:2]
            where_clauses.append(f"{alias}.{fld} = {repr(v)}")
        else:
            where_clauses.append(f"{k} = {repr(v)}")

    # ORDER BY
    order_by = intent.get("order_by")
    order_by_clause = f"ORDER BY {order_by}" if order_by else ""

    # LIMIT
    limit = intent.get("limit")
    limit_clause = f"LIMIT {limit}" if limit else ""

    # Build SQL
    sql = f"SELECT {', '.join(select_fields)} FROM {from_clause}"
    if join_clauses:
        sql += " " + " ".join(join_clauses)
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    if group_by_clause:
        sql += f" {group_by_clause}"
    if having_clause:
        sql += f" {having_clause}"
    if order_by_clause:
        sql += f" {order_by_clause}"
    if limit_clause:
        sql += f" {limit_clause}"
    sql = sql.strip()
    logging.info(f"[intent_to_sql] Generated SQL: {sql}")
    return sql
