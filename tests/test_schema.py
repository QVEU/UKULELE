import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from conftest import ROOT
from validate import validate_entry

REQUIRED = ["id", "version", "outcome", "claim_type", "evidence_layer", "domain", "title",
            "observation", "system", "method", "conditions", "confidence",
            "contributor", "date"]
FINDING_SCHEMA = ROOT / "schema" / "finding.schema.json"
SHARED_DEFS = ["outcome", "claim_type", "evidence_layer", "outcome_matches_claim_type",
               "curie", "subject", "effect", "conditions_structured", "passage_id"]


@pytest.mark.parametrize("path", [ROOT / "schema" / "entry.schema.json", FINDING_SCHEMA], ids=lambda p: p.name)
def test_schema_is_a_valid_json_schema(path):
    Draft202012Validator.check_schema(json.loads(path.read_text()))


@pytest.mark.parametrize("name", SHARED_DEFS)
def test_shared_definitions_are_identical(name, schema_path):
    entry_defs = json.loads(schema_path.read_text())["$defs"]
    finding_defs = json.loads(FINDING_SCHEMA.read_text())["$defs"]
    assert entry_defs[name] == finding_defs[name]


def _finding(**overrides):
    finding = {
        "statement": "PTBP2 did not change IRES-driven translation.",
        "outcome": "negative", "claim_type": "real_null", "evidence_layer": "functional",
        "subjects": [{"role": "agent", "label": "PTBP2", "id": "NCBIGene:58155"}],
        "source": {"pmid": "12345", "passage_id": "PMID:12345:ab", "quote": "did not change"},
    }
    finding.update(overrides)
    return finding


def _finding_errors(finding):
    validator = Draft202012Validator(json.loads(FINDING_SCHEMA.read_text()))
    return list(validator.iter_errors(finding))


def test_valid_findings():
    assert _finding_errors(_finding()) == []
    assert _finding_errors(_finding(outcome="positive", claim_type="effect_observed",
                                    effect={"direction": "increase", "magnitude": "2-fold"})) == []
    assert _finding_errors(_finding(source={"pmid": "1", "pmcid": "PMC9", "passage_id": "PMC9:s3.2:p4",
                                            "quote": "x"})) == []
    assert _finding_errors(_finding(source={"pmcid": "PMC9", "passage_id": "PMC9:ab:p1", "quote": "x"})) == []


@pytest.mark.parametrize("overrides", [
    {"source": {"pmid": "12345", "passage_id": "PMID:12345:ab"}},
    {"source": {"pmid": "12345", "passage_id": "abstract", "quote": "q"}},
    {"source": {"pmid": "PMID12345", "passage_id": "PMID:12345:ab", "quote": "q"}},
    {"source": {"passage_id": "PMC9:ab:p1", "quote": "q"}},
    {"subjects": []},
    {"outcome": "positive"},
    {"claim_type": "effect_observed"},
    {"statement": ""},
], ids=lambda o: ",".join(o))
def test_invalid_findings(overrides):
    assert _finding_errors(_finding(**overrides))


@pytest.mark.parametrize("pid,ok", [
    ("PMID:22745249:ti", True), ("PMID:22745249:ab", True),
    ("PMC5736707:ti", True), ("PMC5736707:ab:p1", True),
    ("PMC5736707:s3:p2", True), ("PMC5736707:s3.2.1:p10", True), ("PMC5736707:s0:p1", True),
    ("PMID:0123:ab", False), ("PMC5736707:s3:p0", False), ("PMC5736707:s03:p1", False),
    ("PMC5736707:body:p1", False), ("5736707:s1:p1", False),
])
def test_passage_id_format(pid, ok):
    errors = _finding_errors(_finding(source={"pmid": "1", "passage_id": pid, "quote": "q"}))
    assert (errors == []) is ok


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


@pytest.mark.parametrize("overrides", [
    {"outcome": "positive", "claim_type": "effect_observed", "effect": {"direction": "binds"}},
    {"outcome": "inconclusive", "claim_type": "method_failure"},
    {"subjects": [{"role": "agent", "label": "PTBP2", "id": "NCBIGene:58155"},
                  {"role": "system", "label": "HeLa", "id": "cellosaurus:CVCL_0030"},
                  {"role": "target", "label": "poliovirus IRES"}]},
    {"links": [{"relation": "tests", "target": "ku-misc-1a2b3c4d", "note": "follow-up"},
               {"relation": "contradicts", "target": "PMID:29259203"},
               {"relation": "supports", "target": "PMC5736707"},
               {"relation": "challenges", "target": "DOI:10.1038/s41467-017-02329-y"}]},
    {"source": {"kind": "contributed"}},
    {"source": {"kind": "literature", "pmid": "29259203", "pmcid": "PMC5736707",
                "passage_id": "PMC5736707:s2:p3", "quotes": ["no mutations were detected"]}},
    {"source": {"kind": "literature", "doi": "10.1101/2024.01.01.000001", "quotes": ["no effect"]}},
], ids=lambda o: ",".join(o))
def test_valid_v02_fields(overrides, make_entry, write_entry, schema_path):
    assert validate_entry(write_entry(make_entry(**overrides)), str(schema_path))


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
    {"version": "0.1.0"},
    {"outcome": "maybe"},
    {"outcome": "positive"},
    {"outcome": "negative", "claim_type": "effect_observed"},
    {"effect": {"direction": "sideways"}},
    {"effect": {"magnitude": "2-fold"}},
    {"subjects": [{"role": "enzyme", "label": "PTBP2"}]},
    {"subjects": [{"role": "agent"}]},
    {"subjects": [{"role": "agent", "label": "PTBP2", "id": "PTBP2"}]},
    {"subjects": [{"role": "agent", "label": "PTBP2", "id": "NCBIGene: 58155"}]},
    {"subjects": [{"role": "agent", "label": "PTBP2", "species": "human"}]},
    {"links": [{"relation": "inspired", "target": "ku-misc-1a2b3c4d"}]},
    {"links": [{"relation": "tests", "target": "ptbp2-entry.yaml"}]},
    {"links": [{"relation": "tests"}]},
    {"source": {"kind": "rumor"}},
    {"source": {"kind": "literature", "pmid": "29259203"}},
    {"source": {"kind": "literature", "pmid": "29259203", "quotes": []}},
    {"source": {"kind": "literature", "quotes": ["no effect"]}},
], ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items())[:80])
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
