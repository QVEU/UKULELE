import yaml

from provenance import INFERRED, STATED, IntakeState
from to_entry import emit
from validate import validate_entry


def _stated(**fields):
    state = IntakeState()
    for name, value in fields.items():
        assert state.set(name, value, STATED, f"said {name}")
    return state


def _complete_state(**overrides):
    fields = dict(
        claim_type="real_null", evidence_layer="functional", domain="molecular virology",
        title="No effect of PTBP2", observation="No change in reporter output.",
        conditions="In vitro translation.", caveats="Functional only.",
        alternatives="Try EMSA.", system={"model": "RRL"},
        method={"name": "in vitro translation"}, confidence_level="medium",
        positive_control={"present": True, "worked": True, "detail": "known ITAF"},
        powered=True, n="3", contributor={"name": "Tester"}, date="2026-09-24",
    )
    fields.update(overrides)
    return _stated(**fields)


def _load(path):
    return yaml.safe_load(open(path))


def test_complete_intake_emits_valid_entry(tmp_path, schema_path):
    path = emit(_complete_state(), out_dir=str(tmp_path))
    assert validate_entry(path, str(schema_path))


def test_conditions_structured_nested_under_system(tmp_path, schema_path):
    cs = {"system": "RRL", "key_params": {"Mg2+": "2 mM"}}
    state = _complete_state(conditions_structured=cs)
    entry = _load(emit(state, out_dir=str(tmp_path)))
    assert "conditions_structured" not in entry
    assert entry["system"] == {"model": "RRL", "conditions_structured": cs}
    assert state.fields["system"].value == {"model": "RRL"}
    assert validate_entry(str(tmp_path / "intake-draft.yaml"), str(schema_path))


def test_conditions_structured_with_unknown_system(tmp_path):
    state = _complete_state(conditions_structured={"system": "RRL"})
    state.fields["system"].provenance = "unknown"
    entry = _load(emit(state, out_dir=str(tmp_path)))
    assert entry["system"] == {"conditions_structured": {"system": "RRL"}}


def test_non_dict_system_does_not_crash_or_drop_conditions(tmp_path, schema_path):
    # Nothing is invented or lost: both stay visible and validation flags the shape problem.
    state = _complete_state(system="HEK293T cells", conditions_structured={"system": "HEK293T"})
    path = emit(state, out_dir=str(tmp_path))
    entry = _load(path)
    assert entry["system"] == "HEK293T cells"
    assert entry["conditions_structured"] == {"system": "HEK293T"}
    assert not validate_entry(path, str(schema_path))


def test_unknown_powered_is_omitted_not_null(tmp_path, schema_path):
    state = _complete_state()
    state.fields["powered"].provenance = "unknown"
    path = emit(state, out_dir=str(tmp_path))
    assert "powered" not in _load(path)["confidence"]
    assert validate_entry(path, str(schema_path))


def test_empty_intake_uses_conservative_defaults_and_lists_unknowns(tmp_path, schema_path):
    path = emit(IntakeState(), out_dir=str(tmp_path))
    text = open(path).read()
    entry = yaml.safe_load(text)
    assert entry["id"] == "ku-pending-00000000"
    assert entry["confidence"]["level"] == "low"
    assert entry["confidence"]["positive_control"]["present"] is False
    assert entry["evidence_layer"] is None
    assert "Fields left UNKNOWN" in text and "'evidence_layer'" in text
    assert not validate_entry(path, str(schema_path))  # CI must reject a missing evidence layer


def test_inferred_values_are_emitted_but_not_listed_unknown(tmp_path):
    state = _complete_state()
    state.set("n", "about 3", INFERRED, "they said 'a few'")
    text = open(emit(state, out_dir=str(tmp_path))).read()
    assert yaml.safe_load(text)["confidence"]["n"] == "about 3"
    assert "'n'" not in text.splitlines()[1]
