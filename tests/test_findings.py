import pytest

from findings import normalize_text, verify_quotes

PASSAGES = {
    "PMID:1:ab": "Ibrutinib  resistance was associated with\nmutations in BTK and PLCG2 in 5 of 6 patients.",
    "PMC9:s2:p1": "No change in reporter output was observed — the “null” held for every fragment.",
}


def _f(pid, quote):
    return {"statement": "s", "source": {"pmid": "1", "passage_id": pid, "quote": quote}}


def test_verbatim_quote_is_kept_across_whitespace_differences():
    kept, rejected = verify_quotes([_f("PMID:1:ab", "resistance was associated with mutations in BTK")], PASSAGES)
    assert len(kept) == 1 and rejected == []


def test_typographic_punctuation_is_normalized():
    kept, _ = verify_quotes([_f("PMC9:s2:p1", 'observed - the "null" held for every fragment')], PASSAGES)
    assert len(kept) == 1


@pytest.mark.parametrize("quote,reason", [
    ("resistance was associated with mutations in BTK, PLCG2 and ITPKB", "not found"),
    ("Resistance was associated with mutations", "not found"),
    ("resistance was ... mutations in BTK", "not found"),
    ("in BTK", "shorter"),
    ("", "shorter"),
])
def test_altered_stitched_or_trivial_quotes_are_rejected(quote, reason):
    kept, rejected = verify_quotes([_f("PMID:1:ab", quote)], PASSAGES)
    assert kept == [] and reason in rejected[0]["reason"]


def test_unknown_passage_and_malformed_findings_are_rejected():
    _, rejected = verify_quotes([_f("PMID:2:ab", "resistance was associated"), "text", {"statement": "x"}],
                                PASSAGES)
    assert [r["reason"] for r in rejected] == ["unknown passage 'PMID:2:ab'", "no source", "no source"]


def test_normalize_text():
    assert normalize_text("  a b\t‘c’  ") == "a b 'c'"
    assert normalize_text(None) == ""
