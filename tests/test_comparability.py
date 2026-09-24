import pytest

from comparability import assess_raft, semantic_comparability, structural_comparability


def hit(id_, layer="functional", system=None, **key_params):
    cs = {}
    if system:
        cs["system"] = system
    if key_params:
        cs["key_params"] = key_params
    return {"id": id_, "evidence_layer": layer, "conditions": f"conditions of {id_}",
            "system": {"conditions_structured": cs} if cs else {}}


def test_different_layers_never_corroborate():
    score, reasons = structural_comparability(hit("a", "functional", "RRL"), hit("b", "binding", "RRL"))
    assert score == 0.0 and "different evidence layers" in reasons[0]


def test_no_structured_conditions_is_unknown_not_comparable():
    score, reasons = structural_comparability(hit("a"), hit("b"))
    assert score == 0.0 and "UNKNOWN" in reasons[0]


def test_matching_system_and_params():
    score, reasons = structural_comparability(hit("a", system="RRL", mg="2 mM"),
                                              hit("b", system=" rrl ", mg="2 MM"))
    assert score == 1.0
    assert any("same system" in r for r in reasons) and "matching mg" in reasons


def test_partial_match_scores_fractionally():
    score, reasons = structural_comparability(hit("a", system="RRL", mg="2 mM", k="100 mM"),
                                              hit("b", system="RRL", mg="5 mM", k="100 mM"))
    assert score == pytest.approx(2 / 3)
    assert any("differing mg" in r for r in reasons)


def test_only_shared_params_are_compared():
    score, _ = structural_comparability(hit("a", system="RRL", mg="2 mM"), hit("b", system="RRL", temp="30C"))
    assert score == 1.0


def test_reads_conditions_from_system_block_like_the_schema():
    a = {"id": "a", "evidence_layer": "functional", "conditions_structured": {"system": "RRL"}}
    b = {"id": "b", "evidence_layer": "functional", "conditions_structured": {"system": "RRL"}}
    assert structural_comparability(a, b)[0] == 0.0  # top-level placement is not schema-valid
    assert structural_comparability(hit("a", system="RRL"), hit("b", system="RRL"))[0] == 1.0


@pytest.mark.parametrize("hits,verdict", [
    ([hit("a", system="RRL")], "single"),
    ([hit("a", system="RRL"), hit("b", "binding", system="RRL")], "mixed_layers"),
    ([hit("a"), hit("b")], "topical_only"),
    ([hit("a", system="RRL", mg="2"), hit("b", system="WGE", mg="5")], "topical_only"),
    ([hit("a", system="RRL", mg="2", k="1"), hit("b", system="RRL", mg="5", k="9")], "weak_raft"),
    ([hit("a", system="RRL", mg="2"), hit("b", system="RRL", mg="2")], "genuine_raft"),
    ([hit("a", system="RRL"), hit("b", system="RRL"), hit("c")], "topical_only"),
])
def test_assess_raft_verdicts(hits, verdict):
    assert assess_raft(hits)["verdict"] == verdict


def test_assess_raft_reports_every_pair():
    result = assess_raft([hit("a", system="RRL"), hit("b", system="RRL"), hit("c", system="RRL")])
    assert [p["pair"] for p in result["pairs"]] == [["a", "b"], ["a", "c"], ["b", "c"]]


def test_semantic_tier_only_called_where_structure_is_ambiguous():
    calls = []
    llm = lambda prompt: calls.append(prompt) or "UNCLEAR"
    result = assess_raft([hit("a", system="RRL"), hit("b", system="RRL"), hit("c", system="WGE")], call_llm=llm)
    assert len(calls) == 2
    by_pair = {tuple(p["pair"]): p for p in result["pairs"]}
    assert "semantic" not in by_pair[("a", "b")]
    assert by_pair[("a", "c")]["semantic"] == "UNCLEAR"


def test_semantic_prompt_includes_both_entries():
    prompt = semantic_comparability(hit("a"), hit("b"), lambda p: p)
    assert "A (a): conditions of a" in prompt and "B (b): conditions of b" in prompt
