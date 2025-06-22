import re

# Minimal YAML loader for offline environments.
# Supports the limited structure used in config/schema/schema.yaml.

def safe_load(stream):
    if hasattr(stream, 'read'):
        stream = stream.read()
    lines = [line.split('#', 1)[0].rstrip() for line in stream.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    data = {}
    current_table = None
    current_join = None

    for line in lines:
        if not line.startswith(' '):
            if line.startswith('schema_version:'):
                val = line.split(':', 1)[1].strip()
                try:
                    data['schema_version'] = float(val)
                except ValueError:
                    data['schema_version'] = val
            elif line.startswith('tables:'):
                data['tables'] = {}
        elif line.startswith('  ') and not line.startswith('    '):
            if line.strip().endswith(':'):
                current_table = line.strip()[:-1]
                data['tables'][current_table] = {}
        elif line.startswith('    ') and not line.startswith('      '):
            stripped = line.strip()
            if stripped.startswith('description:'):
                data['tables'][current_table]['description'] = stripped.split(':',1)[1].strip()
            elif stripped.startswith('fields:'):
                fields = stripped.split(':',1)[1].strip()
                fields = [f.strip() for f in fields.strip('[]').split(',') if f.strip()]
                data['tables'][current_table]['fields'] = fields
            elif stripped.startswith('joins:'):
                data['tables'][current_table]['joins'] = []
                current_join = None
            elif stripped.startswith('- '):
                key, val = stripped[2:].split(':',1)
                current_join = {key.strip(): val.strip()}
                data['tables'][current_table].setdefault('joins', []).append(current_join)
            elif ':' in stripped and current_join is not None:
                key, val = stripped.split(':',1)
                current_join[key.strip()] = val.strip()
        elif line.startswith('      '):
            stripped = line.strip()
            if stripped.startswith('- '):
                key, val = stripped[2:].split(':',1)
                current_join = {key.strip(): val.strip()}
                data['tables'][current_table].setdefault('joins', []).append(current_join)
            elif ':' in stripped and current_join is not None:
                key, val = stripped.split(':',1)
                current_join[key.strip()] = val.strip()
    return data
