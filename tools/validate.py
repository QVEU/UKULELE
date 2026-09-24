"""Validate an entry YAML file against entry.schema.json."""
import sys, json, yaml
from jsonschema import validate, ValidationError

def load_schema(path="schema/entry.schema.json"):
    with open(path) as f:
        return json.load(f)

def validate_entry(entry_path, schema_path="schema/entry.schema.json"):
    schema = load_schema(schema_path)
    with open(entry_path) as f:
        entry = yaml.safe_load(f)
    try:
        validate(instance=entry, schema=schema)
        print(f"OK: {entry_path} is valid.")
        return True
    except ValidationError as e:
        print(f"INVALID: {entry_path}\n  {e.message}\n  at: {'/'.join(map(str, e.path))}")
        return False

if __name__ == "__main__":
    results = [validate_entry(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
