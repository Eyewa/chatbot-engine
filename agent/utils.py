def filter_response_by_type(response_json: dict) -> dict:
    # Import RESPONSE_TYPES locally to avoid circular import
    from simple_yaml import safe_load
    import os
    RESPONSE_TYPES = None
    if os.path.exists(os.path.join("config", "templates", "response_types.yaml")):
        with open(os.path.join("config", "templates", "response_types.yaml")) as f:
            RESPONSE_TYPES = safe_load(f)
    if RESPONSE_TYPES is None:
        return response_json
    response_type = response_json.get("type")
    if not response_type or response_type not in RESPONSE_TYPES:
        return response_json
    allowed_fields = RESPONSE_TYPES[response_type].get("fields", [])
    filtered = {"type": response_type}
    for key in allowed_fields:
        # Always include the field if present, even if it's an empty list or None
        if key in response_json:
            filtered[key] = response_json[key]
    return filtered 