"""Find results that disagree about the same subject, and classify each disagreement.

Works on findings (schema/finding.schema.json) and on UKULELE entries, which share
outcome, claim_type, evidence_layer, subjects, and effect. Classes:

  layer_explained        different evidence layers: the results answer different questions
  genuine_conflict       same layer and comparable conditions: a real paradox
  condition_dependent    same layer, conditions differ: the disagreement may be explained by them
  comparability_unknown  same layer, but no structured conditions to compare
  unresolved             one side has no evidence layer, so the disagreement can't be classified
"""
import re
from itertools import combinations

from comparability import UNKNOWN_REASON, structural_comparability

GENUINE_THRESHOLD = 0.6  # the cut-off assess_raft uses for a genuine raft
OPPOSITE = {("increase", "decrease"), ("decrease", "increase")}

def _subjects(item, role):
    ids, labels = set(), set()
    for s in item.get("subjects") or []:
        if isinstance(s, dict) and s.get("role") == role:
            if s.get("id"):
                ids.add(s["id"])
            if s.get("label"):
                labels.add(re.sub(r"\s+", " ", s["label"].strip().lower()))
    return ids, labels

def _role_match(a, b, role):
    a_ids, a_labels = _subjects(a, role)
    b_ids, b_labels = _subjects(b, role)
    if a_ids and b_ids:  # identifiers are authoritative when both sides have them
        return "id" if a_ids & b_ids else None
    return "label" if a_labels & b_labels else None

def same_subject(a, b):
    """'id' or 'label' if a and b share an agent and a target, else None. 'label' is weaker."""
    matches = [_role_match(a, b, role) for role in ("agent", "target")]
    if None in matches:
        return None
    return "id" if all(m == "id" for m in matches) else "label"

def conflict(a, b):
    """How a and b disagree, or None. Inconclusive results never conflict."""
    outcomes = {a.get("outcome"), b.get("outcome")}
    if outcomes == {"positive", "negative"}:
        return "positive vs negative"
    if outcomes == {"positive"}:
        da = (a.get("effect") or {}).get("direction")
        db = (b.get("effect") or {}).get("direction")
        if (da, db) in OPPOSITE:
            return f"opposite directions ({da} vs {db})"
    return None

def _comparable(item):
    cs = item.get("conditions_structured")
    system = item.get("system")
    if cs is None and isinstance(system, dict):
        cs = system.get("conditions_structured")
    return {"evidence_layer": item.get("evidence_layer"), "system": {"conditions_structured": cs}}

def classify(a, b):
    """Return (verdict, reasons) for two results already known to disagree about the same subject."""
    la, lb = a.get("evidence_layer"), b.get("evidence_layer")
    if not la or not lb:
        return "unresolved", ["evidence layer unknown for at least one side"]
    if la != lb:
        return "layer_explained", [f"different evidence layers ({la} vs {lb}): they answer different questions"]
    score, reasons = structural_comparability(_comparable(a), _comparable(b))
    if reasons == [UNKNOWN_REASON]:
        return "comparability_unknown", reasons
    if score >= GENUINE_THRESHOLD:
        return "genuine_conflict", reasons
    return "condition_dependent", reasons

def _ident(item, index):
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    return item.get("id") or source.get("passage_id") or f"#{index}"

def find_paradoxes(items):
    """Every pair of items that disagree about the same subject, with its classification."""
    found = []
    for (i, a), (j, b) in combinations(enumerate(items), 2):
        kind = conflict(a, b)
        if not kind:
            continue
        match = same_subject(a, b)
        if not match:
            continue
        verdict, reasons = classify(a, b)
        found.append({"pair": [_ident(a, i), _ident(b, j)], "conflict": kind,
                      "verdict": verdict, "match": match, "reasons": reasons})
    return found
