"""Assign IDs in-place to entries that don't have a real one yet."""
import sys, glob, os, re, hashlib, yaml

PLACEHOLDER = re.compile(r"^ku-[a-z0-9]*-0{8}$|^\s*$")

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())[:12] or "misc"

def make_id(entry: dict) -> str:
    domain = slugify(entry.get("domain", "misc"))
    basis = "|".join([
        entry.get("title", ""), entry.get("observation", ""),
        entry.get("contributor", {}).get("name", ""), entry.get("date", ""),
    ])
    h = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:8]
    return f"ku-{domain}-{h}"

def process(path):
    with open(path) as f:
        raw = f.read()
    entry = yaml.safe_load(raw)
    current = str(entry.get("id", "") or "")
    if not PLACEHOLDER.match(current):
        return False  # already has a real ID; leave untouched
    new_id = make_id(entry)
    # Surgical replace of the id line to preserve formatting/comments elsewhere
    if re.search(r"(?m)^id:.*$", raw):
        raw = re.sub(r"(?m)^id:.*$", f"id: {new_id}", raw, count=1)
    else:
        raw = f"id: {new_id}\n" + raw
    with open(path, "w") as f:
        f.write(raw)
    print(f"Assigned {new_id} -> {path}")
    return True

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "entries/"
    files = glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "*.yml"))
    any(process(p) for p in files)
