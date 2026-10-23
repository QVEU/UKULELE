import subprocess
import sys

from conftest import ROOT
from validate import check_graph

A, B, C = "ku-misc-aaaaaaaa", "ku-misc-bbbbbbbb", "ku-misc-cccccccc"


def _db(make_entry, **links_by_id):
    return {f"{eid}.yaml": make_entry(id=eid, links=[{"relation": r, "target": t} for r, t in links])
            for eid, links in links_by_id.items()}


def test_valid_dag_and_external_links_pass(make_entry):
    db = _db(make_entry, **{
        A: [],
        B: [("tests", A), ("contradicts", "PMID:29259203")],
        C: [("revises", B), ("motivated_by", A), ("supports", "DOI:10.1/x")],
    })
    assert check_graph(db) == []


def test_evidence_relations_may_run_both_ways(make_entry):
    db = _db(make_entry, **{A: [("contradicts", B)], B: [("contradicts", A)]})
    assert check_graph(db) == []


def test_lineage_cycle_is_rejected(make_entry):
    db = _db(make_entry, **{A: [("tests", C)], B: [("motivated_by", A)], C: [("revises", B)]})
    errors = check_graph(db)
    assert len(errors) == 1 and "cycle" in errors[0]
    for eid in (A, B, C):
        assert eid in errors[0]


def test_dangling_self_and_placeholder_links_are_rejected(make_entry):
    db = _db(make_entry, **{A: [("tests", "ku-misc-dddddddd"), ("supports", A),
                                ("motivated_by", "ku-pending-00000000")]})
    errors = check_graph(db)
    assert any("unknown entry ku-misc-dddddddd" in e for e in errors)
    assert any("links to itself" in e for e in errors)
    assert any("placeholder" in e for e in errors)


def test_pending_entries_can_link_to_merged_ones(make_entry):
    db = {
        "a.yaml": make_entry(id=A),
        "new1.yaml": make_entry(id="ku-pending-00000000", links=[{"relation": "tests", "target": A}]),
        "new2.yaml": make_entry(id="ku-pending-00000000", links=[{"relation": "revises", "target": A}]),
    }
    assert check_graph(db) == []


def test_duplicate_ids_are_rejected(make_entry):
    db = {"one.yaml": make_entry(id=A), "two.yaml": make_entry(id=A)}
    assert any("duplicate id" in e for e in check_graph(db))


def test_unparseable_entries_are_skipped():
    assert check_graph({"bad.yaml": None, "list.yaml": ["not", "an", "entry"]}) == []


def test_cli_fails_on_graph_errors(make_entry, write_entry):
    a = write_entry(make_entry(id=A, links=[{"relation": "tests", "target": B}]))
    b = write_entry(make_entry(id=B, links=[{"relation": "tests", "target": A}]))
    result = subprocess.run([sys.executable, "tools/validate.py", a, b],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert "INVALID LINKS" in result.stdout and "cycle" in result.stdout


def test_committed_database_has_no_link_errors():
    from validate import load_entry
    paths = sorted((ROOT / "entries").glob("*.y*ml"))
    assert check_graph({str(p): load_entry(p) for p in paths}) == []
