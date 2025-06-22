"""Very small YAML subset parser used when PyYAML is unavailable."""

from typing import Any, List, Tuple


def safe_load(stream: Any) -> Any:
    """Parse a tiny subset of YAML supporting dictionaries and lists."""
    if hasattr(stream, "read"):
        stream = stream.read()

    raw_lines = [line.split("#", 1)[0].rstrip() for line in str(stream).splitlines()]
    lines = [ln for ln in raw_lines if ln.strip()]

    tokens: List[Tuple[int, str]] = []
    for ln in lines:
        indent = len(ln) - len(ln.lstrip(" "))
        tokens.append((indent, ln.lstrip()))

    index = 0

    def parse_value(val: str):
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                return []
            return [parse_value(v.strip()) for v in inner.split(",")]
        try:
            if val.startswith("0") and val != "0":
                raise ValueError
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val.strip('"').strip("'")

    def parse_block(indent: int):
        nonlocal index
        if index >= len(tokens):
            return None
        container = None
        # Decide container type
        if tokens[index][0] >= indent and tokens[index][1].startswith("- "):
            container = []
            while index < len(tokens) and tokens[index][0] >= indent and tokens[index][1].startswith("- "):
                ind, text = tokens[index]
                item_text = text[2:].strip()
                index += 1
                if index < len(tokens) and tokens[index][0] > ind:
                    if item_text.endswith(":"):
                        key = item_text[:-1].strip()
                        value = parse_block(ind + 2)
                        container.append({key: value})
                    elif ":" in item_text:
                        key, val = item_text.split(":", 1)
                        item = {key.strip(): parse_value(val.strip())}
                        extra_indent = tokens[index][0]
                        if extra_indent > ind:
                            extra = parse_block(extra_indent)
                            if isinstance(extra, dict):
                                item.update(extra)
                        container.append(item)
                    else:
                        value = parse_block(ind + 2)
                        container.append(value)
                else:
                    if item_text.endswith(":"):
                        container.append({item_text[:-1].strip(): {}})
                    elif ":" in item_text:
                        key, val = item_text.split(":", 1)
                        container.append({key.strip(): parse_value(val.strip())})
                    else:
                        container.append(parse_value(item_text))
        else:
            container = {}
            while index < len(tokens) and tokens[index][0] >= indent:
                ind, text = tokens[index]
                if ind < indent:
                    break
                if text.startswith("- ") and ind == indent:
                    break
                if ":" not in text:
                    index += 1
                    continue
                key, val = text.split(":", 1)
                key = key.strip()
                val = val.strip()
                index += 1
                if val in {'>', '|'}:
                    parts = []
                    while index < len(tokens) and tokens[index][0] > ind:
                        parts.append(tokens[index][1])
                        index += 1
                    container[key] = " ".join(p.strip() for p in parts)
                elif val:
                    container[key] = parse_value(val)
                else:
                    if index < len(tokens) and tokens[index][0] > ind:
                        container[key] = parse_block(ind + 2)
                    else:
                        container[key] = {}
        return container

    result = parse_block(0)
    return result if result is not None else {}
