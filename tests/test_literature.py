import gzip
import json
import re
import shutil

import pytest

import pmc
import pubmed_baseline as pb
from conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures"
PUBMED = FIXTURES / "pubmed_sample.xml"
JATS = FIXTURES / "pmc_sample.xml"
PASSAGE_ID = re.compile(json.loads((ROOT / "schema" / "finding.schema.json").read_text())
                        ["$defs"]["passage_id"]["pattern"])


@pytest.fixture
def records():
    stats = {}
    return list(pb.iter_records(str(PUBMED), stats)), stats


def test_preprints_and_books_are_skipped(records):
    recs, stats = records
    assert [r["pmid"] for r in recs] == ["90000001", "90000004"]
    assert stats == {"preprints": 1, "PubmedBookArticle": 1}


def test_record_fields(records):
    first, old = records[0]
    assert first["title"] == "Kinase XYZ1 inhibition does not alter replication of a model virus."
    assert first["abstract"] == ("BACKGROUND: XYZ1 was proposed to support viral replication. "
                                 "RESULTS: Inhibiting XYZ1 produced no change in viral titer across three doses.")
    assert first["year"] == 2021 and first["journal"] == "Journal of Invented Results"
    assert first["doi"] == "10.9999/jir.2021.001" and first["pmcid"] == "PMC9000001"
    assert [m["ui"] for m in first["mesh"]] == ["D014779", "D011493"]
    assert old["year"] == 1998 and old["abstract"] == "" and old["doi"] is None and old["pmcid"] is None
    assert [p["name"] for p in old["pub_types"]] == ["Journal Article", "Review"]


def test_gzipped_input_matches_plain(tmp_path, records):
    gz = tmp_path / "pubmed26n0001.xml.gz"
    with open(PUBMED, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    assert list(pb.iter_records(str(gz))) == records[0]


def test_pubmed_passages_use_frozen_ids(records):
    first, old = records[0]
    assert [pid for pid, _ in pb.passages(first)] == ["PMID:90000001:ti", "PMID:90000001:ab"]
    assert [pid for pid, _ in pb.passages(old)] == ["PMID:90000004:ti"]
    assert all(PASSAGE_ID.match(pid) for r in records[0] for pid, _ in pb.passages(r))


def test_pubmed_cli_writes_pyserini_jsonl(tmp_path):
    pb.main([str(PUBMED), "--out", str(tmp_path)])
    with gzip.open(tmp_path / "pubmed_sample.jsonl.gz", "rt") as fh:
        docs = [json.loads(line) for line in fh]
    assert [d["id"] for d in docs] == ["90000001", "90000004"]
    assert docs[0]["contents"].startswith("Kinase XYZ1 inhibition") and "RESULTS:" in docs[0]["contents"]


@pytest.fixture
def article():
    return pmc.parse_jats(JATS.read_bytes())


def test_jats_ids_and_title(article):
    assert article["pmcid"] == "PMC9000001" and article["pmid"] == "90000001"
    assert article["doi"] == "10.9999/jir.2021.001"
    assert article["title"] == "Kinase XYZ1 inhibition does not alter replication"


def test_jats_passages_have_stable_ids(article):
    ids = [p["passage_id"] for p in article["passages"]]
    assert ids == ["PMC9000001:ti", "PMC9000001:ab:p1", "PMC9000001:ab:p2", "PMC9000001:s0:p1",
                   "PMC9000001:s1:p1", "PMC9000001:s1:p2",
                   "PMC9000001:s2.1:p1", "PMC9000001:s2.1:p2", "PMC9000001:s2.2:p1"]
    assert all(PASSAGE_ID.match(pid) for pid in ids)
    assert pmc.parse_jats(JATS.read_bytes())["passages"] == article["passages"]


def test_jats_passage_text_and_sections(article):
    by_id = {p["passage_id"]: p for p in article["passages"]}
    assert by_id["PMC9000001:s1:p1"]["text"] == "Prior work suggested a role for XYZ1 [1]."
    assert by_id["PMC9000001:s2.1:p2"] == {"passage_id": "PMC9000001:s2.1:p2", "section": "Results > Titer",
                                         "text": "No dose produced a change."}
    assert by_id["PMC9000001:s2.2:p1"]["section"] == "Results > Positive control"
    texts = " ".join(p["text"] for p in article["passages"])
    assert "Graphical abstract" not in texts and "Figure caption" not in texts


def test_jats_cited_pmids_are_deduplicated(article):
    assert article["cited_pmids"] == ["90000004", "90000005"]


def test_jats_pmcid_from_argument_or_error():
    xml = b"<article><front><article-meta><title-group><article-title>T</article-title></title-group>" \
          b"</article-meta></front></article>"
    assert pmc.parse_jats(xml, pmcid="5")["passages"] == [{"passage_id": "PMC5:ti", "section": "Title", "text": "T"}]
    with pytest.raises(ValueError):
        pmc.parse_jats(xml)


def test_load_challenge_pmcids(tmp_path):
    good = tmp_path / "Challenge_PMCIDs.txt"
    good.write_text("PMCID\nPMC1\n2\n\n# comment\npmc3\n")
    assert pmc.load_challenge_pmcids(good) == {"PMC1", "PMC2", "PMC3"}
    bad = tmp_path / "bad.txt"
    bad.write_text("PMC1\nnot-an-id\n")
    with pytest.raises(ValueError, match=":2:"):
        pmc.load_challenge_pmcids(bad)


@pytest.mark.parametrize("listed,kept", [("PMC9000001\n", 1), ("PMC1\n", 0)])
def test_pmc_cli_filters_by_challenge_list(tmp_path, listed, kept):
    ids = tmp_path / "ids.txt"
    ids.write_text(listed)
    out = tmp_path / "passages.jsonl.gz"
    pmc.main([str(JATS), "--pmcids", str(ids), "--out", str(out)])
    with gzip.open(out, "rt") as fh:
        assert len(fh.readlines()) == kept
