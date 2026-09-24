# Knowledge Universe — v2 additions

Three upgrades, in dependency order:

1. **Conditions-comparability** — rafts now check whether entries are *actually comparable*, not just topically close.
2. **Live `call_llm()`** — a real, provider-agnostic talking demo.
3. **FAISS** — swap the flat numpy index for something that scales, behind the same interface.

---

## 1. Conditions-comparability — making rafts honest

The hard truth from earlier: two nulls in the same evidence layer and claim type still may **not** corroborate each other if they were run under incompatible conditions (different buffers, systems, doses, coverage). "Are these conditions comparable?" is a domain-specific judgment, so the honest design is a **two-tier** approach: cheap structural signals first, optional LLM judgment second. Never pretend structural overlap *is* scientific comparability — label the confidence of the comparability claim itself.

### Schema nudge (optional but recommended)

Comparability works far better on *structured* conditions than on prose. Add an optional structured block that contributors fill where they can — the free text stays as the fallback:

```json
"conditions_structured": {
  "type": "object",
  "description": "Optional structured conditions to enable comparability checks. Free-text 'conditions' remains the fallback.",
  "additionalProperties": true,
  "properties": {
    "system":     { "type": "string", "description": "e.g. RRL, wheat-germ, HEK293T, 40 field sites" },
    "key_params": { "type": "object", "additionalProperties": true, "description": "domain-specific: buffer, [Mg2+], dose range, temperature, season..." }
  }
}
```

### `retrieval/comparability.py`

```python
"""
Assess whether two entries are COMPARABLE (their nulls can corroborate each other),
distinct from whether they are SIMILAR (topically close).

Two tiers:
  Tier 1 - structural: cheap, deterministic signals from structured fields.
  Tier 2 - semantic (optional): an LLM judges condition compatibility from prose.

The output always carries its OWN confidence. Structural agreement is a weak
proxy; we say so.
"""
from itertools import combinations

def _params(e):
    cs = e.get("conditions_structured", {}) or {}
    return cs.get("system"), (cs.get("key_params") or {})

def structural_comparability(a, b):
    """Return (score 0..1, reasons[]). Deterministic, no model needed."""
    reasons, score, checks = [], 0.0, 0

    # Same evidence layer is a prerequisite for corroboration at all.
    if a["evidence_layer"] != b["evidence_layer"]:
        return 0.0, ["different evidence layers — not corroborating"]

    sys_a, par_a = _params(a)
    sys_b, par_b = _params(b)

    if sys_a and sys_b:
        checks += 1
        if sys_a.strip().lower() == sys_b.strip().lower():
            score += 1; reasons.append(f"same system ({sys_a})")
        else:
            reasons.append(f"different systems ({sys_a} vs {sys_b}) — may not corroborate")

    shared = set(par_a) & set(par_b)
    for k in shared:
        checks += 1
        if str(par_a[k]).strip().lower() == str(par_b[k]).strip().lower():
            score += 1; reasons.append(f"matching {k}")
        else:
            reasons.append(f"differing {k} ({par_a[k]} vs {par_b[k]})")

    if checks == 0:
        return 0.0, ["no structured conditions to compare — comparability UNKNOWN"]
    return score / checks, reasons

def semantic_comparability(a, b, call_llm):
    """Tier 2: ask an LLM to judge condition compatibility from prose. Optional."""
    prompt = (
        "Two negative experimental results are below. Judging ONLY by their "
        "conditions, could they corroborate each other, or are conditions too "
        "different for one to reinforce the other? Answer COMPARABLE / "
        "NOT_COMPARABLE / UNCLEAR and one sentence why.\n\n"
        f"A ({a['id']}): {a.get('conditions','')}\n\n"
        f"B ({b['id']}): {b.get('conditions','')}"
    )
    return call_llm(prompt)

def assess_raft(hits, call_llm=None):
    """
    Upgrade of the old raft_assessment: now reports pairwise comparability,
    not just shared labels.
    """
    if len(hits) < 2:
        return {"verdict": "single", "detail": "One result — no consensus possible."}

    layers = {h["evidence_layer"] for h in hits}
    if len(layers) > 1:
        return {"verdict": "mixed_layers",
                "detail": f"Different evidence layers ({sorted(layers)}); different questions."}

    pairs, weakest = [], 1.0
    for a, b in combinations(hits, 2):
        s, reasons = structural_comparability(a, b)
        weakest = min(weakest, s)
        entry = {"pair": [a["id"], b["id"]], "structural_score": round(s, 2), "reasons": reasons}
        if call_llm and s < 1.0:  # only spend a model call where structure is ambiguous
            entry["semantic"] = semantic_comparability(a, b, call_llm)
        pairs.append(entry)

    if weakest == 0.0:
        verdict = "topical_only"; msg = "Entries are on-topic but NOT shown to be comparable — do not read as consensus."
    elif weakest < 0.6:
        verdict = "weak_raft"; msg = "Partial comparability — treat as suggestive, not settled."
    else:
        verdict = "genuine_raft"; msg = "Comparable conditions — reasonable to treat as accumulating evidence."

    return {"verdict": verdict, "detail": msg, "pairs": pairs}
```

The key honesty: when there are **no structured conditions to compare**, it returns `UNKNOWN` rather than defaulting to "comparable." Silence is not agreement.

---

## 2. Live `call_llm()` — the talking demo

Provider-agnostic via environment variables. Works with any OpenAI-compatible endpoint (OpenAI, local Ollama, vLLM, etc.); an Anthropic path is shown too. Nothing about the grounding rules changes — only where the prompt gets sent.

### `retrieval/llm.py`

```python
"""
A single call_llm(prompt) -> str, chosen by env var. This is the ONLY place a
provider is named; everything upstream is provider-agnostic.

  KU_LLM=openai   + OPENAI_API_KEY   (+ optional OPENAI_BASE_URL for local/compatible)
  KU_LLM=anthropic+ ANTHROPIC_API_KEY
  KU_LLM=echo     -> returns the prompt (no network; for inspecting grounding)
"""
import os

def _openai(prompt):
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))  # None = default
    r = client.chat.completions.create(
        model=os.getenv("KU_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # grounded synthesis wants determinism, not creativity
    )
    return r.choices[0].message.content

def _anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=os.getenv("KU_LLM_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in r.content if hasattr(b, "text"))

def call_llm(prompt: str) -> str:
    backend = os.getenv("KU_LLM", "echo").lower()
    if backend == "openai":    return _openai(prompt)
    if backend == "anthropic": return _anthropic(prompt)
    if backend == "echo":      return "[echo — no LLM called]\n\n" + prompt
    raise ValueError(f"Unknown KU_LLM backend: {backend}")
```

### Wire it into `synthesize.py`

Replace the stub `call_llm` import and add comparability to the prompt:

```python
from query import query, confidence_flag
from llm import call_llm
from comparability import assess_raft

SYSTEM_RULES = """You answer ONLY from the provided negative-result entries.
1. Cite the entry [id] for every factual claim. Never state anything not traceable to a supplied entry.
2. Always surface confidence and positive-control status; never present a weakened null as settled.
3. Different evidence layers answer different questions — never treat them as agreeing.
4. Respect the RAFT VERDICT: if 'topical_only' or 'weak_raft', explicitly avoid claiming consensus.
5. If comparability is UNKNOWN, say the evidence cannot yet be pooled.
6. Relay suggested alternatives; if evidence is weak/weakened, say so plainly."""

def build_prompt(question, hits, raft):
    ctx = [f"RAFT VERDICT: {raft['verdict']} — {raft['detail']}\n"]
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
    return (f"{SYSTEM_RULES}\n\nQUESTION: {question}\n\nENTRIES:\n"
            + "\n".join(ctx) + "\n\nAnswer with inline [id] citations.")

def synthesize(question, top_k=5, evidence_layer=None, claim_type=None):
    hits, _ = query(question, top_k=top_k, evidence_layer=evidence_layer, claim_type=claim_type)
    if not hits:
        return "No relevant entries found. (No answer invented — that's the point.)"
    raft = assess_raft(hits, call_llm=call_llm)   # comparability-aware
    return call_llm(build_prompt(question, hits, raft))
```

### Talk to it

```bash
export KU_LLM=anthropic          # or openai, or leave as echo to inspect
export ANTHROPIC_API_KEY=...
python -c "from retrieval.synthesize import synthesize; \
print(synthesize('does PTBP2 affect poliovirus IRES translation?'))"
```

With `KU_LLM=echo` you see the exact grounded prompt and can verify the rules travel with it before spending a single API call.

---

## 3. FAISS — scale behind the same interface

The flat numpy dot-product is fine to ~a few thousand entries. FAISS takes you to millions without changing anything upstream — same `query()` signature, same filtering-before-ranking logic.

### `requirements.txt` additions

```
faiss-cpu>=1.7
openai>=1.0        # only if using KU_LLM=openai
anthropic>=0.39    # only if using KU_LLM=anthropic
```

### `retrieval/build_index.py` — write a FAISS index too

Append after the numpy save:

```python
import faiss  # add at top

# ... after vecs are computed and normalized ...
index = faiss.IndexFlatIP(vecs.shape[1])   # inner product == cosine (vecs normalized)
index = faiss.IndexIDMap(index)            # lets us map back to entry rows
index.add_with_ids(vecs.astype("float32"), np.arange(len(vecs)))
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
print("Wrote FAISS index.")
```

### `retrieval/query.py` — read FAISS if present, else fall back

The critical subtlety: **filtering must still happen, and metadata filters interact awkwardly with ANN search.** The honest pattern at this scale is *over-fetch then filter*: ask FAISS for more neighbors than you need, then apply evidence-layer/claim-type filters, then trim to `top_k`. For hard filters on huge corpora you'd eventually want per-layer sub-indices, but over-fetch is correct and simple to start.

```python
import os, faiss, numpy as np

def _faiss_search(q, entries, top_k, evidence_layer, claim_type, overfetch=10):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    # over-fetch because we filter AFTER ANN retrieval
    k = min(len(entries), top_k * overfetch)
    scores, ids = index.search(q.astype("float32").reshape(1, -1), k)
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        e = entries[i]
        if evidence_layer and e["evidence_layer"] != evidence_layer:
            continue
        if claim_type and e["claim_type"] != claim_type:
            continue
        hits.append({**e, "score": float(score)})
        if len(hits) >= top_k:
            break
    return hits

def query(text, top_k=5, evidence_layer=None, claim_type=None):
    vecs, entries, model_name = load_index()
    model = SentenceTransformer(model_name)
    q = model.encode([text], normalize_embeddings=True)[0]

    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(faiss_path):
        hits = _faiss_search(q, entries, top_k, evidence_layer, claim_type)
    else:
        # original flat path (unchanged) ...
        idx = list(range(len(entries)))
        if evidence_layer: idx = [i for i in idx if entries[i]["evidence_layer"] == evidence_layer]
        if claim_type:     idx = [i for i in idx if entries[i]["claim_type"] == claim_type]
        if not idx:        return [], "No entries match the requested filters."
        sims = vecs[idx] @ q
        order = np.argsort(-sims)[:top_k]
        hits = [{**entries[idx[o]], "score": float(sims[o])} for o in order]

    from comparability import assess_raft
    return hits, assess_raft(hits)  # raft assessment now comparability-aware
```

---

## Where this leaves you

- **Rafts are now honest**: they distinguish topical proximity from real comparability, and admit UNKNOWN when structured conditions are missing.
- **It talks**: one env var switches providers; `echo` lets you audit the grounded prompt for free.
- **It scales**: FAISS behind the same `query()` interface, with over-fetch-then-filter so structural filtering survives ANN search.

The one honest gap remaining is upstream of all this: comparability is only as good as the structured conditions contributors provide. The Tier-2 LLM judge softens that, but the real lever is making `conditions_structured` easy and rewarding to fill at submission time — which loops back to the incentive problem that was the biggest risk from the very start.
