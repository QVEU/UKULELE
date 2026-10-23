import pytest

from hybrid import rrf_fuse


def test_documents_ranked_well_by_both_retrievers_win():
    bm25 = ["a", "b", "c"]
    dense = ["b", "d", "a"]
    fused = rrf_fuse([bm25, dense], k=60)
    assert [doc for doc, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_weights_limit_and_duplicates():
    fused = rrf_fuse([["a", "a", "b"], ["b"]], k=1, weights=[1.0, 3.0], limit=1)
    assert fused == [("b", pytest.approx(1 / 3 + 3 / 2))]


def test_ties_keep_first_seen_order():
    assert [d for d, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]


def test_weight_count_must_match():
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])
