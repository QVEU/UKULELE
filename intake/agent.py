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
        if name in state.fields and isinstance(u, dict):
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
