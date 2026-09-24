import json
import os
import sys

import pytest
import yaml

faiss = pytest.importorskip("faiss")

import build_index  # noqa: E402
import query as q  # noqa: E402
import synthesize  # noqa: E402


def _entry(make_entry, id_, layer, claim, title, system=None, pc=None):
    return make_entry(
        id=id_, evidence_layer=layer, claim_type=claim, title=title, observation=title,
        system={"conditions_structured": system} if system else {},
        confidence={"level": "medium", "positive_control": pc or {"present": False}},
    )


@pytest.fixture
def corpus(make_entry):
    rrl = {"system": "RRL", "key_params": {"Mg2+": "2 mM"}}
    return [
        _entry(make_entry, "ku-virology-aaaaaaaa", "functional", "real_null",
               "PTBP2 has no effect on poliovirus IRES translation", rrl),
        _entry(make_entry, "ku-virology-bbbbbbbb", "functional", "real_null",
               "PTBP2 fragments do not alter poliovirus IRES translation", rrl,
               pc={"present": True, "worked": True}),
        _entry(make_entry, "ku-virology-cccccccc", "binding", "real_null",
               "PTBP2 shows no binding to the poliovirus IRES by EMSA"),
        _entry(make_entry, "ku-zoology-dddddddd", "phenotypic", "not_reproducible",
               "Fin regeneration speedup in zebrafish could not be reproduced"),
    ]


@pytest.fixture
def build(tmp_path, monkeypatch, fake_encoder):
    index_dir = tmp_path / "index"
    monkeypatch.chdir(tmp_path)
    for mod in (build_index, q):
        monkeypatch.setattr(mod, "SentenceTransformer", fake_encoder)
        monkeypatch.setattr(mod, "INDEX_DIR", str(index_dir))

    def _build(entries):
        (tmp_path / "entries").mkdir(exist_ok=True)
        for e in entries:
            (tmp_path / "entries" / f"{e['id']}.yaml").write_text(yaml.safe_dump(e))
        build_index.main()
        return index_dir
    return _build


def _ids(hits):
    return {h["id"] for h in hits}


def test_build_index_writes_all_artifacts(build, corpus):
    index_dir = build(corpus)
    for name in ("vectors.npy", "faiss.index", "meta.json"):
        assert (index_dir / name).exists()
    meta = json.loads((index_dir / "meta.json").read_text())
    assert len(meta["entries"]) == 4
    assert all("system" in e for e in meta["entries"])


def test_build_index_with_no_entries(build, capsys):
    index_dir = build([])
    assert "No entries to index" in capsys.readouterr().out
    assert not index_dir.exists()


def test_embedding_text_leads_with_structure(corpus):
    text = build_index.compose_embedding_text(corpus[0])
    assert text.startswith("CLAIM TYPE: real_null\nEVIDENCE LAYER: functional")


@pytest.mark.parametrize("use_faiss", [True, False])
def test_layer_filter_excludes_other_layers(build, corpus, use_faiss):
    index_dir = build(corpus)
    if not use_faiss:
        os.remove(index_dir / "faiss.index")
    hits, _ = q.query("PTBP2 poliovirus IRES", top_k=5, evidence_layer="binding")
    assert _ids(hits) == {"ku-virology-cccccccc"}
    hits, _ = q.query("anything", top_k=5, claim_type="not_reproducible")
    assert _ids(hits) == {"ku-zoology-dddddddd"}


@pytest.mark.parametrize("use_faiss", [True, False])
def test_functional_null_does_not_answer_a_binding_question(build, corpus, use_faiss):
    index_dir = build(corpus[:2])
    if not use_faiss:
        os.remove(index_dir / "faiss.index")
    hits, raft = q.query("does PTBP2 bind the poliovirus IRES?", evidence_layer="binding")
    assert hits == []
    assert raft["verdict"] == "no_match"


def test_faiss_and_flat_paths_agree(build, corpus):
    index_dir = build(corpus)
    with_faiss, _ = q.query("PTBP2 IRES translation", top_k=4)
    os.remove(index_dir / "faiss.index")
    flat, _ = q.query("PTBP2 IRES translation", top_k=4)
    assert [h["id"] for h in with_faiss] == [h["id"] for h in flat]
    for a, b in zip(with_faiss, flat):
        assert a["score"] == pytest.approx(b["score"], abs=1e-5)


def test_top_k_is_respected(build, corpus):
    build(corpus)
    assert len(q.query("PTBP2", top_k=2)[0]) == 2
    assert len(q.query("PTBP2", top_k=50)[0]) == 4


def test_structured_conditions_survive_indexing_into_a_genuine_raft(build, corpus):
    build(corpus)
    hits, raft = q.query("PTBP2 IRES translation", top_k=2, evidence_layer="functional")
    assert _ids(hits) == {"ku-virology-aaaaaaaa", "ku-virology-bbbbbbbb"}
    assert raft["verdict"] == "genuine_raft"


def test_unfiltered_cluster_across_layers_is_not_consensus(build, corpus):
    build(corpus)
    _, raft = q.query("PTBP2 poliovirus IRES", top_k=4)
    assert raft["verdict"] == "mixed_layers"


@pytest.mark.parametrize("conf,expected", [
    ({"level": "high", "positive_control": {"present": False}}, "WEAKENED"),
    ({"level": "high", "positive_control": {"present": True, "worked": False}}, "CAUTION"),
    ({"level": "high", "positive_control": {"present": True, "worked": True}}, "validated"),
    ({"level": "low"}, "low confidence"),
    ({"level": "low", "positive_control": None}, "low confidence"),
    ({}, "unknown confidence"),
])
def test_confidence_flag(conf, expected):
    assert expected in q.confidence_flag(conf)


@pytest.mark.parametrize("argv,expected", [
    (["PTBP2 IRES", "--layer", "functional", "--k", "2"], "genuine_raft"),
    (["PTBP2 IRES", "--layer", "computational"], "no_match"),
])
def test_query_cli(build, corpus, monkeypatch, capsys, argv, expected):
    build(corpus)
    monkeypatch.setattr(sys, "argv", ["query.py", *argv])
    q.main()
    out = capsys.readouterr().out
    assert "[RAFT ASSESSMENT]" in out and expected in out


def test_build_prompt_carries_grounding_rules(corpus):
    prompt = synthesize.build_prompt("Q?", corpus[:2], {"verdict": "weak_raft", "detail": "partial"})
    assert synthesize.SYSTEM_RULES in prompt
    assert "RAFT VERDICT: weak_raft — partial" in prompt
    assert "[ku-virology-aaaaaaaa]" in prompt and "[ku-virology-bbbbbbbb]" in prompt
    assert "WEAKENED" in prompt


def test_synthesize_refuses_without_hits(build, corpus, monkeypatch):
    build(corpus)
    monkeypatch.setattr(synthesize, "call_llm", lambda p: pytest.fail("LLM must not be called"))
    assert "No relevant entries found" in synthesize.synthesize("x", evidence_layer="computational")


def test_synthesize_sends_grounded_prompt(build, corpus, monkeypatch):
    build(corpus)
    monkeypatch.setenv("KU_LLM", "echo")
    answer = synthesize.synthesize("PTBP2 IRES translation", top_k=2, evidence_layer="functional")
    assert answer.startswith("[echo — no LLM called]")
    assert "RAFT VERDICT: genuine_raft" in answer
