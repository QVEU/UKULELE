"""Validate entry YAML files against entry.schema.json, then check links across the whole set."""
import sys, json, re, yaml
from jsonschema import validate, ValidationError

LINEAGE = {"tests", "motivated_by", "revises"}
ENTRY_ID = re.compile(r"^ku-[a-z0-9]*-[0-9a-f]{8}$")
PLACEHOLDER = re.compile(r"^ku-[a-z0-9]*-0{8}$")

def load_schema(path="schema/entry.schema.json"):
    with open(path) as f:
        return json.load(f)

def load_entry(path):
    with open(path) as f:
        return yaml.safe_load(f)

def validate_entry(entry_path, schema_path="schema/entry.schema.json"):
    schema = load_schema(schema_path)
    entry = load_entry(entry_path)
    try:
        validate(instance=entry, schema=schema)
        print(f"OK: {entry_path} is valid.")
        return True
    except ValidationError as e:
        print(f"INVALID: {entry_path}\n  {e.message}\n  at: {'/'.join(map(str, e.path))}")
        return False

def _first_cycle(edges):
    state = {}  # node -> 1 while on the current path, 2 when finished
    for start in sorted(edges):
        if start in state:
            continue
        path, stack = [start], [iter(sorted(edges[start]))]
        state[start] = 1
        while stack:
            nxt = next(stack[-1], None)
            if nxt is None:
                state[path.pop()] = 2
                stack.pop()
            elif state.get(nxt) == 1:
                return path[path.index(nxt):] + [nxt]
            elif nxt not in state:
                state[nxt] = 1
                path.append(nxt)
                stack.append(iter(sorted(edges.get(nxt, ()))))
    return None

def check_graph(entries):
    """Cross-entry checks a per-file schema can't express. `entries` maps a file name to its parsed entry."""
    entries = {name: e for name, e in entries.items() if isinstance(e, dict)}
    errors, known = [], {}
    for name, e in entries.items():
        eid = str(e.get("id") or "")
        if ENTRY_ID.match(eid) and not PLACEHOLDER.match(eid):
            if eid in known:
                errors.append(f"{name}: duplicate id {eid} (also in {known[eid]})")
            known[eid] = name

    edges = {}
    for name, e in entries.items():
        eid = str(e.get("id") or "")
        node = eid if eid in known else name  # pending entries share a placeholder id, so key them by file
        for link in e.get("links") or []:
            if not isinstance(link, dict):
                continue
            target = str(link.get("target", ""))
            if not ENTRY_ID.match(target):
                continue  # PMID / PMC / DOI targets point outside the database
            if PLACEHOLDER.match(target):
                errors.append(f"{name}: links to placeholder id {target}; link only to merged entries")
            elif target == eid:
                errors.append(f"{name}: links to itself")
            elif target not in known:
                errors.append(f"{name}: links to unknown entry {target}")
            elif link.get("relation") in LINEAGE:
                edges.setdefault(node, set()).add(target)

    cycle = _first_cycle(edges)
    if cycle:
        errors.append("lineage links (tests/motivated_by/revises) form a cycle: " + " -> ".join(cycle))
    return errors

if __name__ == "__main__":
    paths = sys.argv[1:]
    results = [validate_entry(p) for p in paths]
    graph_errors = check_graph({p: load_entry(p) for p in paths})
    for err in graph_errors:
        print(f"INVALID LINKS: {err}")
    sys.exit(0 if all(results) and not graph_errors else 1)
