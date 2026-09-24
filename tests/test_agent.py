import io
import json

import pytest

import agent
from provenance import HIGH_VALUE, STATED, UNKNOWN, IntakeState
from validate import validate_entry


def _update(**fields):
    return json.dumps({k: {"value": v, "provenance": "stated", "source": f"said {k}"}
                       for k, v in fields.items()})


def test_parse_turn_applies_updates_embedded_in_prose(monkeypatch):
    monkeypatch.setattr(agent, "call_llm", lambda p: "Sure!\n" + _update(evidence_layer="functional") + "\nDone.")
    state = IntakeState()
    agent.parse_turn(state, "we ran a reporter assay")
    assert state.fields["evidence_layer"].value == "functional"
    assert state.fields["evidence_layer"].provenance == STATED


def test_parse_turn_prompt_carries_rules_and_user_text(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "call_llm", lambda p: seen.append(p) or "{}")
    agent.parse_turn(IntakeState(), "UNIQUE-NOTE-TEXT")
    assert agent.EXTRACTION_RULES in seen[0] and "UNIQUE-NOTE-TEXT" in seen[0]


@pytest.mark.parametrize("raw", ["no json here", "{not valid json}", ""])
def test_unparseable_response_changes_nothing(monkeypatch, raw):
    monkeypatch.setattr(agent, "call_llm", lambda p: raw)
    state = IntakeState()
    before = state.as_dict()
    assert agent.parse_turn(state, "notes") == {}
    assert state.as_dict() == before


def test_malformed_update_items_are_skipped(monkeypatch):
    raw = json.dumps({"n": "3", "domain": None, "powered": ["yes"],
                      "title": {"value": "T", "provenance": "stated", "source": "x"}})
    monkeypatch.setattr(agent, "call_llm", lambda p: raw)
    state = IntakeState()
    agent.parse_turn(state, "notes")
    assert state.fields["n"].provenance == UNKNOWN
    assert state.fields["title"].value == "T"


def test_unknown_field_names_are_ignored(monkeypatch):
    monkeypatch.setattr(agent, "call_llm", lambda p: _update(favourite_colour="blue"))
    state = IntakeState()
    agent.parse_turn(state, "notes")
    assert "favourite_colour" not in state.fields


def test_inferred_confidence_is_never_accepted(monkeypatch):
    raw = json.dumps({"confidence_level": {"value": "high", "provenance": "inferred", "source": "looks solid"}})
    monkeypatch.setattr(agent, "call_llm", lambda p: raw)
    state = IntakeState()
    agent.parse_turn(state, "it worked great")
    assert state.fields["confidence_level"].provenance == UNKNOWN


def test_next_question_asks_about_gaps(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "call_llm", lambda p: seen.append(p) or "Did you run a positive control?")
    assert agent.next_question(IntakeState()) == "Did you run a positive control?"
    assert str(HIGH_VALUE) in seen[0]


def test_next_question_signals_ready_when_high_value_fields_stated(monkeypatch):
    monkeypatch.setattr(agent, "call_llm", lambda p: pytest.fail("should not call the LLM"))
    state = IntakeState()
    for name in HIGH_VALUE:
        state.set(name, "x", STATED, "said")
    assert agent.next_question(state) is None


class ScriptedLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if "Return ONLY a JSON object" in prompt:
            return _update(
                claim_type="real_null", evidence_layer="functional", domain="virology",
                title="No effect", observation="Nothing changed.", conditions="In vitro.",
                caveats="Functional only.", system={"model": "RRL"},
                conditions_structured={"system": "RRL"}, method={"name": "IVT"},
                confidence_level="medium", positive_control={"present": False},
                powered=True, n="3", contributor={"name": "Tester"}, date="2026-09-24")
        if "confirmation summary" in prompt:
            return "STATED: ...\nINFERRED: none\nSTILL UNKNOWN: ..."
        return "Any more detail?"


@pytest.mark.parametrize("answer,writes", [("approve", True), ("no", False), ("", False)])
def test_run_end_to_end(monkeypatch, tmp_path, schema_path, capsys, answer, writes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "call_llm", ScriptedLLM())
    monkeypatch.setattr("sys.stdin", io.StringIO("PTBP2 did nothing to IRES translation\n"))
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    agent.run()
    out = capsys.readouterr().out
    assert "CONFIRMATION" in out and "STILL UNKNOWN" in out
    draft = tmp_path / "entries" / "intake-draft.yaml"
    assert draft.exists() is writes
    if writes:
        assert validate_entry(str(draft), str(schema_path))
