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
    cs = (e.get("system") or {}).get("conditions_structured") or {}
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
