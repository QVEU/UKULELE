"""
Turn retrieved hits into a cited, confidence-aware answer.

Grounded synthesis: the model may only use the supplied entries, must cite [id]
for every claim, must surface confidence/positive-control caveats, and must not
assert consensus the comparability-aware raft verdict doesn't support. The LLM
call itself lives in llm.py, chosen by the KU_LLM env var.
"""
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
