import pytest

from provenance import (ALL_FIELDS, HIGH_VALUE, HUMAN_OWNED, INFERRED, STATED,
                        UNKNOWN, IntakeState)


def test_new_state_is_entirely_unknown():
    state = IntakeState()
    assert set(state.unknowns()) == set(ALL_FIELDS)
    assert state.high_value_gaps() == HIGH_VALUE
    assert state.inferred() == {}


def test_stated_value_resolves_gap():
    state = IntakeState()
    assert state.set("evidence_layer", "functional", STATED, "a reporter assay") is True
    field = state.fields["evidence_layer"]
    assert field.is_confident() and field.source == "a reporter assay"
    assert "evidence_layer" not in state.unknowns()
    assert "evidence_layer" not in state.high_value_gaps()


def test_inferred_high_value_field_is_still_a_gap():
    state = IntakeState()
    state.set("n", "3", INFERRED, "sounds like triplicate")
    assert "n" in state.high_value_gaps()
    assert set(state.inferred()) == {"n"}
    assert not state.fields["n"].is_confident()


@pytest.mark.parametrize("name", HUMAN_OWNED)
def test_human_owned_fields_reject_inference(name):
    state = IntakeState()
    assert state.set(name, "high", INFERRED, "results look solid") is False
    assert state.fields[name].provenance == UNKNOWN
    assert state.fields[name].value is None


@pytest.mark.parametrize("name", HUMAN_OWNED)
def test_human_owned_fields_accept_stated_values(name):
    state = IntakeState()
    assert state.set(name, "low", STATED, "I'd call it low") is True
    assert state.fields[name].value == "low"


@pytest.mark.parametrize("label", ["assumed", "guess", "inferred-from-context", "", None])
def test_unrecognized_provenance_changes_nothing(label):
    # Otherwise a label like "assumed" would sneak past the HUMAN_OWNED guard and read as resolved.
    state = IntakeState()
    assert state.set("confidence_level", "high", label, "?") is False
    assert state.set("n", "3", label, "?") is False
    assert state.fields["confidence_level"].provenance == UNKNOWN
    assert "n" in state.high_value_gaps()


def test_provenance_labels_are_case_insensitive():
    state = IntakeState()
    assert state.set("n", "3", " Stated ", "three replicates") is True
    assert state.fields["n"].provenance == STATED


def test_as_dict_shape():
    state = IntakeState()
    state.set("title", "T", STATED, "their words")
    d = state.as_dict()
    assert set(d) == set(ALL_FIELDS)
    assert d["title"] == {"value": "T", "provenance": STATED, "source": "their words"}
