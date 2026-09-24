import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from conftest import ROOT
from validate import validate_entry

REQUIRED = ["id", "version", "claim_type", "evidence_layer", "domain", "title",
            "observation", "system", "method", "conditions", "confidence",
            "contributor", "date"]


def test_schema_is_a_valid_json_schema(schema_path):
    Draft202012Validator.check_schema(json.loads(schema_path.read_text()))


@pytest.mark.parametrize("entry_path", sorted((ROOT / "entries").glob("*.y*ml")),
                         ids=lambda p: p.name)
def test_committed_entries_validate(entry_path, schema_path):
    assert validate_entry(str(entry_path), str(schema_path))


def test_minimal_entry_validates(make_entry, write_entry, schema_path):
    assert validate_entry(write_entry(make_entry()), str(schema_path))


def test_structured_conditions_under_system_validate(make_entry, write_entry, schema_path):
    entry = make_entry(system={"model": "RRL", "conditions_structured": {
        "system": "RRL", "key_params": {"Mg2+": "2 mM"}, "extra": "allowed"}})
    assert validate_entry(write_entry(entry), str(schema_path))


@pytest.mark.parametrize("field", REQUIRED)
def test_missing_required_field_rejected(field, make_entry, write_entry, schema_path):
    entry = make_entry()
    del entry[field]
    assert not validate_entry(write_entry(entry), str(schema_path))


@pytest.mark.parametrize("overrides", [
    {"claim_type": "maybe"},
    {"evidence_layer": "vibes"},
    {"evidence_layer": None},
    {"version": "9.9.9"},
    {"id": "KU-Upper-00000000"},
    {"id": "ku-misc-1234567"},
    {"id": "ku-misc-zzzzzzzz"},
    {"title": "x" * 201},
    {"unexpected_top_level": 1},
    {"conditions_structured": {"system": "RRL"}},
    {"system": "HEK293T"},
    {"confidence": {"level": "medium"}},
    {"confidence": {"level": "certain", "positive_control": {"present": False}}},
    {"confidence": {"level": "low", "positive_control": {}}},
    {"confidence": {"level": "low", "positive_control": {"present": False}, "powered": None}},
    {"method": {}},
    {"contributor": {}},
    {"related_positive": "doi:10.1/abc"},
], ids=lambda o: ",".join(o))
def test_invalid_entries_rejected(overrides, make_entry, write_entry, schema_path):
    assert not validate_entry(write_entry(make_entry(**overrides)), str(schema_path))


def test_unquoted_yaml_date_is_rejected(tmp_path, schema_path, make_entry, write_entry):
    # YAML parses a bare 2026-09-24 as a date object, not a string; contributors must quote it.
    path = write_entry(make_entry())
    text = open(path).read().replace("date: '2026-09-24'", "date: 2026-09-24")
    assert "date: 2026-09-24" in text
    open(path, "w").write(text)
    assert not validate_entry(path, str(schema_path))


def _run_cli(*paths):
    return subprocess.run([sys.executable, "tools/validate.py", *paths],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_exit_codes(make_entry, write_entry):
    good = write_entry(make_entry())
    bad = write_entry(make_entry(claim_type="maybe"))
    assert _run_cli(good).returncode == 0
    assert _run_cli(good, bad).returncode == 1


def test_cli_reports_every_invalid_file(make_entry, write_entry):
    bad1 = write_entry(make_entry(claim_type="maybe"))
    bad2 = write_entry(make_entry(evidence_layer="vibes"))
    result = _run_cli(bad1, bad2)
    assert result.returncode == 1
    assert bad1 in result.stdout and bad2 in result.stdout
