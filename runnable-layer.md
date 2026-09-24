# Knowledge Universe — Runnable Layer

This adds the two pieces that turn the scaffold into a working system:

1. **CI** (`.github/workflows/`) — validates every PR against the schema and assigns IDs on merge. This is what makes PR-as-peer-review run unattended.
2. **Reference retrieval** (`retrieval/`) — build an index from entries, then query it. Enforces the four rules: claim-type/evidence-layer awareness, confidence surfacing, citation grounding, and anti-false-consensus.

```
knowledge-universe/
├── .github/workflows/
│   ├── validate.yml
│   └── assign-ids.yml
├── requirements.txt
├── retrieval/
│   ├── build_index.py
│   ├── query.py
│   ├── synthesize.py        # replaceable LLM seam
│   └── index/               # generated (gitignored)
└── ... (scaffold from before)
```

---

## `requirements.txt`

```
pyyaml>=6.0
jsonschema>=4.0
sentence-transformers>=2.2
numpy>=1.24
```

> `sentence-transformers` pulls in torch; if you want a lighter footprint for CI, the validation workflow only needs `pyyaml` + `jsonschema`. Retrieval deps are only needed where you build/query the index.

---

## CI — validation on every PR (`.github/workflows/validate.yml`)

```yaml
name: Validate entries
on:
  pull_request:
    paths: ["entries/**"]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install pyyaml jsonschema
      - name: Validate changed entries
        run: |
          # Validate every entry file so a schema change can't silently break existing data
          shopt -s nullglob
          files=(entries/*.yaml entries/*.yml)
          if [ ${#files[@]} -eq 0 ]; then echo "No entries."; exit 0; fi
          python tools/validate.py "${files[@]}"
```

---

## CI — assign IDs on merge (`.github/workflows/assign-ids.yml`)

```yaml
name: Assign IDs
on:
  push:
    branches: [main]
    paths: ["entries/**"]

permissions:
  contents: write

jobs:
  assign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install pyyaml
      - name: Assign IDs to entries missing a real one
        run: python tools/assign_id_inplace.py entries/
      - name: Commit any ID assignments
        run: |
          if [ -n "$(git status --porcelain)" ]; then
            git config user.name  "ku-bot"
            git config user.email "ku-bot@users.noreply.github.com"
            git add entries/
            git commit -m "chore: assign entry IDs [skip ci]"
            git push
          else
            echo "No IDs to assign."
          fi
```

---

## `tools/assign_id_inplace.py`

Writes the deterministic ID back into any entry that lacks a real one (placeholder IDs all-zeros or blank), preserving the file otherwise.

```python
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
```

---

## Retrieval — build the index (`retrieval/build_index.py`)

```python
"""
Build an embedding index from all entries.

Stores, per entry: the embedding vector, the composed text, and the STRUCTURED
fields the query layer needs to reason about (claim_type, evidence_layer,
confidence, conditions, related_positive). We embed prose but we RETRIEVE with
structure — that separation is the whole point.
"""
import glob, os, json, yaml
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, offline-friendly; swap freely
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

def compose_embedding_text(e: dict) -> str:
    parts = [
        f"CLAIM TYPE: {e['claim_type']}",
        f"EVIDENCE LAYER: {e['evidence_layer']}",
        f"DOMAIN: {e['domain']}",
        f"METHOD: {e.get('method', {}).get('name', 'unspecified')}",
        f"CONFIDENCE: {e.get('confidence', {}).get('level', 'unspecified')}",
        f"TITLE: {e['title']}",
        f"OBSERVATION: {e['observation']}",
        f"CONDITIONS: {e.get('conditions', '')}",
        f"CAVEATS: {e.get('caveats', '')}",
        f"ALTERNATIVES: {e.get('alternatives', '')}",
    ]
    return "\n".join(p for p in parts if p.strip())

def load_entries(path="entries/"):
    out = []
    for f in glob.glob(os.path.join(path, "*.y*ml")):
        with open(f) as fh:
            out.append(yaml.safe_load(fh))
    return out

def main():
    entries = load_entries()
    if not entries:
        print("No entries to index."); return
    model = SentenceTransformer(MODEL_NAME)
    texts = [compose_embedding_text(e) for e in entries]
    vecs = model.encode(texts, normalize_embeddings=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vecs)
    # Keep the structured metadata the query layer reasons over
    meta = [{
        "id": e["id"],
        "title": e["title"],
        "claim_type": e["claim_type"],
        "evidence_layer": e["evidence_layer"],
        "domain": e["domain"],
        "confidence": e.get("confidence", {}),
        "conditions": e.get("conditions", ""),
        "caveats": e.get("caveats", ""),
        "alternatives": e.get("alternatives", ""),
        "related_positive": e.get("related_positive", []),
    } for e in entries]
    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "entries": meta}, f, indent=2)
    print(f"Indexed {len(entries)} entries with {MODEL_NAME}.")

if __name__ == "__main__":
    main()
```

---

## Retrieval — query with the four rules enforced (`retrieval/query.py`)

```python
"""
Query the index. This layer enforces the rules that make retrieval trustworthy:

  RULE 1 (grounding):   every returned item carries its entry ID for citation.
  RULE 2 (confidence):  confidence + positive-control status are surfaced, never hidden.
  RULE 3 (layers):      you can filter by evidence_layer / claim_type so a
                        functional null and a binding result are not conflated.
  RULE 4 (anti-consensus): we compute whether a cluster is a real 'raft' (compatible
                        structure) or just topical proximity, and label it as such.
"""
import os, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

def load_index():
    vecs = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
    with open(os.path.join(INDEX_DIR, "meta.json")) as f:
        meta = json.load(f)
    return vecs, meta["entries"], meta["model"]

def confidence_flag(conf: dict) -> str:
    """Turn structured confidence into a plain-language trust flag."""
    level = conf.get("level", "unknown")
    pc = conf.get("positive_control", {}) or {}
    if pc.get("present") is False:
        return f"{level} confidence — WEAKENED: no positive control (assay sensitivity unproven)"
    if pc.get("present") and pc.get("worked") is False:
        return f"{level} confidence — CAUTION: positive control did not work"
    if pc.get("present") and pc.get("worked"):
        return f"{level} confidence — positive control validated the assay"
    return f"{level} confidence"

def raft_assessment(hits: list) -> str:
    """
    RULE 4: proximity != agreement. A genuine raft requires the hits to share
    evidence_layer AND claim_type AND have no linked contradicting positives.
    Otherwise we explicitly warn against reading the cluster as consensus.
    """
    if len(hits) < 2:
        return "Single result — no consensus claim possible."
    layers = {h["evidence_layer"] for h in hits}
    claims = {h["claim_type"] for h in hits}
    has_contradiction = any(h.get("related_positive") for h in hits)
    if len(layers) > 1:
        return ("MIXED EVIDENCE LAYERS ({}). These answer different questions and "
                "must NOT be read as agreeing.".format(", ".join(sorted(layers))))
    if len(claims) > 1:
        return ("Mixed claim types ({}). Related but not equivalent evidence."
                .format(", ".join(sorted(claims))))
    if has_contradiction:
        return ("Consistent layer/claim, BUT one or more entries link a contradicting "
                "positive result. Not settled — inspect related_positive.")
    return ("Genuine raft: consistent evidence layer and claim type, no linked "
            "contradictions. Reasonable to treat as accumulating evidence — "
            "still check conditions for comparability.")

def query(text, top_k=5, evidence_layer=None, claim_type=None):
    vecs, entries, model_name = load_index()
    model = SentenceTransformer(model_name)
    q = model.encode([text], normalize_embeddings=True)[0]

    idx = list(range(len(entries)))
    # RULE 3: structural filtering BEFORE semantic ranking
    if evidence_layer:
        idx = [i for i in idx if entries[i]["evidence_layer"] == evidence_layer]
    if claim_type:
        idx = [i for i in idx if entries[i]["claim_type"] == claim_type]
    if not idx:
        return [], "No entries match the requested filters."

    sims = vecs[idx] @ q
    order = np.argsort(-sims)[:top_k]
    hits = []
    for o in order:
        e = entries[idx[o]]
        hits.append({**e, "score": float(sims[o])})
    return hits, raft_assessment(hits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--layer", default=None, help="filter: binding/functional/phenotypic/computational/observational")
    ap.add_argument("--claim", default=None, help="filter: real_null/not_reproducible/not_meaningful/method_failure")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    hits, raft = query(args.question, top_k=args.k,
                       evidence_layer=args.layer, claim_type=args.claim)

    print(f"\nQ: {args.question}")
    print(f"\n[RAFT ASSESSMENT] {raft}\n")
    for h in hits:
        print(f"  [{h['id']}]  (score {h['score']:.3f})")
        print(f"    {h['title']}")
        print(f"    layer={h['evidence_layer']}  claim={h['claim_type']}")
        print(f"    {confidence_flag(h['confidence'])}")
        if h.get("related_positive"):
            print(f"    ⚠ linked contradicting/positive: {h['related_positive']}")
        print()

if __name__ == "__main__":
    main()
```

---

## Retrieval — the replaceable LLM seam (`retrieval/synthesize.py`)

Kept deliberately separate and provider-agnostic. The conversational answer is built *only* from retrieved, cited entries — the function refuses to add anything not grounded in a hit.

```python
"""
Turn retrieved hits into a cited, confidence-aware answer.

This is the SWAPPABLE layer. `synthesize()` builds a strict, grounded prompt;
plug in whatever LLM you want at `call_llm()`. The contract: the model may only
use the supplied entries, must cite [id] for every claim, must surface
confidence/positive-control caveats, and must not assert consensus the raft
assessment doesn't support.
"""
from query import query, confidence_flag

SYSTEM_RULES = """You answer ONLY from the provided negative-result entries.
Rules you must follow:
1. Cite the entry [id] for every factual claim. Never state anything not traceable to a supplied entry.
2. Always surface confidence and positive-control status; never present a weakened null as settled.
3. Different evidence layers (binding vs functional vs phenotypic) answer different questions — never treat them as agreeing.
4. Do not claim consensus beyond what the RAFT ASSESSMENT supports.
5. Where entries suggest alternatives, relay them. If evidence is thin or weakened, say so plainly."""

def build_prompt(question, hits, raft):
    ctx = [f"RAFT ASSESSMENT: {raft}\n"]
    for h in hits:
        ctx.append(
            f"[{h['id']}] {h['title']}\n"
            f"  layer={h['evidence_layer']} claim={h['claim_type']}\n"
            f"  {confidence_flag(h['confidence'])}\n"
            f"  conditions: {h['conditions']}\n"
            f"  caveats: {h['caveats']}\n"
            f"  alternatives: {h['alternatives']}\n"
            f"  related_positive: {h.get('related_positive', [])}\n"
        )
    return (f"{SYSTEM_RULES}\n\nQUESTION: {question}\n\n"
            f"ENTRIES:\n" + "\n".join(ctx) +
            "\n\nAnswer with inline [id] citations.")

def call_llm(prompt: str) -> str:
    # Replace with your provider of choice. Left unbound on purpose.
    raise NotImplementedError(
        "Plug in an LLM here. The prompt is fully grounded and ready to send.")

def synthesize(question, top_k=5, evidence_layer=None, claim_type=None):
    hits, raft = query(question, top_k=top_k,
                       evidence_layer=evidence_layer, claim_type=claim_type)
    if not hits:
        return "No relevant entries found. (No answer invented — that's the point.)"
    prompt = build_prompt(question, hits, raft)
    return call_llm(prompt)  # or just return `prompt` to inspect it without an LLM
```

---

## Try it end to end

```bash
pip install -r requirements.txt

# 1. validate (same check CI runs)
python tools/validate.py entries/*.yaml

# 2. build the index from your entries
python retrieval/build_index.py

# 3. ask it something — watch the rules fire
python retrieval/query.py "does PTBP2 affect poliovirus IRES translation?"

# 4. same question, but only functional evidence
python retrieval/query.py "PTBP2 poliovirus IRES" --layer functional

# 5. ask a binding question — your functional null should NOT dominate
python retrieval/query.py "does PTBP2 bind the poliovirus IRES?" --layer binding
```

Step 5 is the one to watch: with only your functional entry indexed and a `--layer binding` filter, the system correctly returns *nothing* rather than handing back your functional null as if it answered a binding question. That refusal is the anti-false-negative machinery working.
