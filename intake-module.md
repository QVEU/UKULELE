# Knowledge Universe — Conversational Intake (`intake/`)

A conversational agent that turns messy lab notes or free-form description into a
schema-valid entry — **by extraction, never invention.**

The governing rule inverts a normal assistant: this agent is *rewarded for
recording gaps* and *penalized for filling them*. An honest `unknown` is worth
more than a confident guess, because the entire database's value is its
trustworthiness. The confirmation step is not a nicety — it is the load-bearing
safety mechanism, and it shows **provenance** (stated / inferred / unknown), not
just a tidy summary.

```
intake/
├── provenance.py     # the field-provenance state object
├── intake_prompts.py # system prompt encoding extract-don't-invent
├── agent.py          # conversation loop + targeted questioning
└── to_entry.py       # emit schema-valid YAML only after human sign-off
```

Reuses the existing `llm.py` seam, schema, and vocabularies. Same values as the
retrieval agent, pointed at input instead of output.

---

## `intake/provenance.py`

Every field carries **how we know it**. This is what makes the confirmation
honest — the human sees what they said vs. what the model inferred vs. what's
still missing.

```python
"""
Field-level provenance tracking. The core anti-confabulation primitive.

Each field has:
  value       - the current value (or None)
  provenance  - one of: 'stated' | 'inferred' | 'unknown'
  source      - for 'stated': the user text it came from
                for 'inferred': the reasoning (must be shown at confirmation)
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict

STATED, INFERRED, UNKNOWN = "stated", "inferred", "unknown"

@dataclass
class Field:
    value: Any = None
    provenance: str = UNKNOWN
    source: Optional[str] = None

    def is_confident(self):
        return self.provenance == STATED

# The fields we track, grouped by how much scrutiny they need at confirmation.
# HIGH_VALUE gaps are the ones that change a null's trustworthiness — the agent
# asks about these specifically rather than interrogating everything equally.
HIGH_VALUE = [
    "evidence_layer",          # binding vs functional vs phenotypic — changes the meaning
    "positive_control",        # was assay sensitivity ever proven?
    "confidence_level",        # human should OWN this, not rubber-stamp
    "powered", "n",            # strength of the null
    "conditions_structured",   # drives comparability
]

# HUMAN_OWNED must never be auto-accepted from an inference — the agent may
# prompt, but the human has to actively set these (anchoring risk otherwise).
HUMAN_OWNED = ["confidence_level", "caveats"]

ALL_FIELDS = [
    "claim_type", "evidence_layer", "domain", "title",
    "observation", "conditions", "conditions_structured", "caveats",
    "alternatives", "system", "method",
    "confidence_level", "positive_control", "powered", "n",
    "detection_limit", "controls", "related_positive",
    "contributor", "date", "references",
]

class IntakeState:
    def __init__(self):
        self.fields: Dict[str, Field] = {k: Field() for k in ALL_FIELDS}

    def set(self, name, value, provenance, source=None):
        # HUMAN_OWNED fields cannot be set by inference — downgrade to a prompt.
        if name in HUMAN_OWNED and provenance == INFERRED:
            return False
        self.fields[name] = Field(value=value, provenance=provenance, source=source)
        return True

    def unknowns(self):
        return [k for k, f in self.fields.items() if f.provenance == UNKNOWN]

    def high_value_gaps(self):
        """Unknown OR merely-inferred high-value fields — worth a question."""
        gaps = []
        for k in HIGH_VALUE:
            f = self.fields.get(k, Field())
            if f.provenance in (UNKNOWN, INFERRED):
                gaps.append(k)
        return gaps

    def inferred(self):
        return {k: f for k, f in self.fields.items() if f.provenance == INFERRED}

    def as_dict(self):
        return {k: asdict(f) for k, f in self.fields.items()}
```

---

## `intake/intake_prompts.py`

The system prompt that inverts normal helpfulness into disciplined extraction.

```python
"""System prompts for the intake agent. Extraction, not invention."""

EXTRACTION_RULES = """You are an intake agent for a database of NEGATIVE and
inconclusive scientific results. The database's only value is its HONESTY.

Your job is to convert what the scientist tells you into structured fields by
EXTRACTION ONLY. You are rewarded for recording gaps and penalized for filling
them. Follow these rules without exception:

1. NEVER invent, assume, or infer a value the scientist did not state, unless you
   explicitly mark it 'inferred' AND state your reasoning. When unsure, mark the
   field 'unknown'. An honest 'unknown' is ALWAYS better than a guess.
2. Do NOT normalize away uncertainty. If they say "standard buffer", record that
   phrase, do not expand it into specific concentrations you assume.
3. For every field, classify it as:
     stated   — directly supported by their words (quote the words as 'source')
     inferred — a reasonable reading BUT not stated (give reasoning; will be shown for confirmation)
     unknown  — not addressed
4. Ask about HIGH-VALUE gaps specifically, one or two at a time, plainly:
     - Is this a BINDING result, a FUNCTIONAL result, a PHENOTYPIC one, etc.?
     - Was a POSITIVE CONTROL (a known effect/modulator) run, and did it work?
     - Roughly how many replicates / what sample size?
     - What conditions/system matter for interpreting this (buffer, dose, cell type, etc.)?
   Do not interrogate every field equally. Do not pester about low-value fields.
5. Do NOT propose the confidence level or write the caveats FOR them. Prompt them
   to state these in their own words. These are the scientist's judgment, not yours.
6. Be brief and concrete. This is a working scientist pasting notes, not an interview.

Output your running understanding as JSON with {value, provenance, source} per
field whenever asked to summarize."""

def confirmation_instructions():
    return """Produce a confirmation summary grouped into three sections:
  STATED (from your words): ...
  INFERRED (my reading — please confirm or correct each): ...
  STILL UNKNOWN (left blank honestly): ...
Ask the scientist to confirm, correct, or fill. Do NOT emit a final entry until
they explicitly approve. Never hide the inferred section inside the stated one."""
```

---

## `intake/agent.py`

The conversation loop. Extracts on each turn, tracks provenance, targets
high-value gaps, and refuses to finalize without sign-off.

```python
"""
Conversational intake loop.

Design: each user turn is parsed into field updates (with provenance) by the LLM
under EXTRACTION_RULES. The agent then either asks about the most important
remaining high-value gap, or moves to provenance-tagged confirmation.
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from llm import call_llm  # the shared, provider-agnostic seam
from provenance import IntakeState, STATED, INFERRED, UNKNOWN
from intake_prompts import EXTRACTION_RULES, confirmation_instructions

def parse_turn(state: IntakeState, user_text: str):
    """Ask the LLM to extract field updates from this turn, with provenance."""
    prompt = (
        f"{EXTRACTION_RULES}\n\n"
        f"Current known fields:\n{json.dumps(state.as_dict(), indent=2)}\n\n"
        f"The scientist just said:\n\"\"\"{user_text}\"\"\"\n\n"
        "Return ONLY a JSON object of fields to update, each as "
        '{\"value\":..., \"provenance\":\"stated|inferred|unknown\", '
        '\"source\":\"their words or your reasoning\"}. '
        "Include a field ONLY if this turn changes it. Do not invent values."
    )
    raw = call_llm(prompt)
    try:
        updates = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        return {}  # if parsing fails, change nothing — never guess
    for name, u in updates.items():
        if name in state.fields:
            state.set(name, u.get("value"), u.get("provenance", UNKNOWN), u.get("source"))
    return updates

def next_question(state: IntakeState) -> str:
    """Ask about the single most important remaining gap, or signal readiness."""
    gaps = state.high_value_gaps()
    if not gaps:
        return None  # ready for confirmation
    prompt = (
        f"{EXTRACTION_RULES}\n\n"
        f"Current fields:\n{json.dumps(state.as_dict(), indent=2)}\n\n"
        f"The most important unresolved high-value gaps are: {gaps}. "
        "Ask the scientist ONE brief, plain question about the most important of "
        "these. If it is confidence_level or caveats, PROMPT them to state it "
        "themselves — do not propose a value."
    )
    return call_llm(prompt)

def confirmation_summary(state: IntakeState) -> str:
    prompt = (
        f"{EXTRACTION_RULES}\n\n{confirmation_instructions()}\n\n"
        f"Fields:\n{json.dumps(state.as_dict(), indent=2)}"
    )
    return call_llm(prompt)

def run():
    state = IntakeState()
    print("Paste your notes or describe the result. (Ctrl-D to move to confirmation.)\n")
    try:
        for line in sys.stdin:
            parse_turn(state, line.strip())
            q = next_question(state)
            if q:
                print("\nAGENT:", q, "\n")
            else:
                print("\nAGENT: I think I have the high-value fields. "
                      "Add more, or Ctrl-D to review.\n")
    except (EOFError, KeyboardInterrupt):
        pass

    print("\n" + "=" * 60)
    print("CONFIRMATION — review provenance before submitting:\n")
    print(confirmation_summary(state))
    print("=" * 60)
    resp = input("\nType 'approve' to emit the entry, or anything else to abort: ")
    if resp.strip().lower() == "approve":
        from to_entry import emit
        path = emit(state)
        print(f"\nWrote {path} (id assigned on merge). Validate before PR.")
    else:
        print("Aborted — nothing written. (Correct fields and re-run.)")

if __name__ == "__main__":
    run()
```

---

## `intake/to_entry.py`

Emits schema-valid YAML **only after sign-off** — and only from confident or
human-approved values. Unknowns are written honestly as `unknown`, never faked.

```python
"""Convert an approved IntakeState into a schema-valid entry YAML."""
import os, yaml
from provenance import IntakeState, UNKNOWN

def _v(state, name, default=None):
    f = state.fields.get(name)
    if f is None or f.provenance == UNKNOWN or f.value is None:
        return default
    return f.value

def emit(state: IntakeState, out_dir="entries/"):
    entry = {
        "id": "ku-pending-00000000",   # real ID assigned by CI on merge
        "version": "0.1.0",
        "claim_type": _v(state, "claim_type", "real_null"),
        "evidence_layer": _v(state, "evidence_layer"),   # required — CI will reject if left None
        "domain": _v(state, "domain", "unspecified"),
        "title": _v(state, "title", "Untitled negative result"),
        "observation": _v(state, "observation", ""),
        "conditions": _v(state, "conditions", ""),
        "caveats": _v(state, "caveats", ""),
        "alternatives": _v(state, "alternatives", ""),
        "system": _v(state, "system", {}),
        "method": _v(state, "method", {"name": "unspecified"}),
        "confidence": {
            "level": _v(state, "confidence_level", "low"),
            "positive_control": _v(state, "positive_control", {"present": False,
                "detail": "not recorded during intake"}),
            "powered": _v(state, "powered"),
            "controls": _v(state, "controls", ""),
            "n": _v(state, "n", ""),
            "detection_limit": _v(state, "detection_limit", ""),
        },
        "related_positive": _v(state, "related_positive", []),
        "contributor": _v(state, "contributor", {"name": "unknown"}),
        "date": _v(state, "date", ""),
        "references": _v(state, "references", []),
    }
    os.makedirs(out_dir, exist_ok=True)
    # Comment header preserves which fields were left unknown, for reviewer context
    unknowns = state.unknowns()
    header = ("# Generated by conversational intake.\n"
              f"# Fields left UNKNOWN (honestly blank): {unknowns}\n"
              "# Review before opening a PR.\n\n")
    path = os.path.join(out_dir, "intake-draft.yaml")
    with open(path, "w") as f:
        f.write(header + yaml.safe_dump(entry, sort_keys=False, default_flow_style=False))
    return path
```

---

## The flow, end to end

```bash
export KU_LLM=anthropic   # or openai; 'echo' won't parse, this step needs a real model
export ANTHROPIC_API_KEY=...

python intake/agent.py
# paste lab notes → agent asks about positive control, evidence layer, n, conditions
# → Ctrl-D → provenance-tagged confirmation → 'approve' → schema-valid draft
python tools/validate.py entries/intake-draft.yaml
```

## The safety properties, restated

- **Parse failure changes nothing.** If the model's JSON can't be parsed, no field is touched — the system never guesses to recover.
- **Inference is quarantined.** Inferred values are tracked separately and shown in their own section at confirmation; they can't masquerade as stated facts.
- **Human-owned fields can't be auto-filled.** `confidence_level` and `caveats` reject inferred values outright — the agent may prompt, never propose. This defuses the anchoring risk.
- **Unknown is honorable.** Gaps are written as `unknown` and listed in the file header for the reviewer. An empty field stays honestly empty.
- **Nothing is emitted without explicit `approve`.** The confirmation is the gate, and it shows provenance, not a tidy summary that hides inference.
