import pytest

from paradox import classify, conflict, find_paradoxes, same_subject

AGENT = {"role": "agent", "label": "Ibrutinib", "id": "MESH:C551803"}
TARGET = {"role": "target", "label": "BTK", "id": "NCBIGene:695"}


def finding(pid, outcome, layer="functional", subjects=(AGENT, TARGET), direction=None, cs=None):
    f = {"statement": pid, "outcome": outcome,
         "claim_type": "effect_observed" if outcome == "positive" else "real_null",
         "evidence_layer": layer, "subjects": list(subjects),
         "source": {"pmid": "1", "passage_id": pid, "quote": "q" * 20}}
    if direction:
        f["effect"] = {"direction": direction}
    if cs is not None:
        f["conditions_structured"] = cs
    return f


RRL = {"system": "RRL", "key_params": {"Mg2+": "2 mM"}}


@pytest.mark.parametrize("a,b,verdict", [
    (finding("PMID:1:ab", "positive", "binding"), finding("PMID:2:ab", "negative", "functional"),
     "layer_explained"),
    (finding("PMID:1:ab", "positive", cs=RRL), finding("PMID:2:ab", "negative", cs=RRL), "genuine_conflict"),
    (finding("PMID:1:ab", "positive", cs=RRL),
     finding("PMID:2:ab", "negative", cs={"system": "HeLa", "key_params": {"Mg2+": "5 mM"}}),
     "condition_dependent"),
    (finding("PMID:1:ab", "positive"), finding("PMID:2:ab", "negative"), "comparability_unknown"),
    (finding("PMID:1:ab", "positive", layer=None), finding("PMID:2:ab", "negative"), "unresolved"),
])
def test_each_verdict(a, b, verdict):
    assert classify(a, b)[0] == verdict
    [p] = find_paradoxes([a, b])
    assert p["verdict"] == verdict and p["pair"] == ["PMID:1:ab", "PMID:2:ab"]


def test_opposite_directions_conflict_but_same_direction_does_not():
    up, down = finding("A", "positive", direction="increase"), finding("B", "positive", direction="decrease")
    assert conflict(up, down) == "opposite directions (increase vs decrease)"
    assert conflict(up, finding("C", "positive", direction="increase")) is None
    assert conflict(up, finding("D", "positive")) is None


def test_inconclusive_results_never_conflict():
    assert conflict(finding("A", "inconclusive"), finding("B", "positive")) is None
    assert find_paradoxes([finding("A", "inconclusive"), finding("B", "negative")]) == []


def test_matching_by_id_label_and_authoritative_ids():
    a = finding("A", "positive")
    assert same_subject(a, finding("B", "negative")) == "id"
    by_label = [{"role": "agent", "label": "  ibrutinib "}, {"role": "target", "label": "btk"}]
    assert same_subject(a, finding("C", "negative", subjects=by_label)) == "label"
    other_gene = [AGENT, {"role": "target", "label": "BTK", "id": "NCBIGene:12229"}]
    assert same_subject(a, finding("D", "negative", subjects=other_gene)) is None


def test_different_subjects_or_missing_target_are_not_paradoxes():
    other = [AGENT, {"role": "target", "label": "PLCG2", "id": "NCBIGene:5336"}]
    no_target = [AGENT]
    items = [finding("A", "positive"), finding("B", "negative", subjects=other),
             finding("C", "negative", subjects=no_target)]
    assert find_paradoxes(items) == []


def test_works_on_entries_with_conditions_under_system(make_entry):
    a = make_entry(id="ku-misc-aaaaaaaa", outcome="positive", claim_type="effect_observed",
                   subjects=[AGENT, TARGET], system={"conditions_structured": RRL})
    b = make_entry(id="ku-misc-bbbbbbbb", subjects=[AGENT, TARGET], system={"conditions_structured": RRL})
    [p] = find_paradoxes([a, b])
    assert p["pair"] == ["ku-misc-aaaaaaaa", "ku-misc-bbbbbbbb"]
    assert p["verdict"] == "genuine_conflict" and p["conflict"] == "positive vs negative"
