import pytest

from metrics import mean_reciprocal_rank, reciprocal_rank


def test_reciprocal_rank():
    assert reciprocal_rank(["9", "22745249", "3"], {"22745249"}) == 0.5
    assert reciprocal_rank(["9", "3"], {"22745249"}) == 0.0
    assert reciprocal_rank(["9", "3", "22745249"], {"22745249"}, cutoff=2) == 0.0


def test_mean_reciprocal_rank_penalizes_missing_questions_and_skips_empty_qrels():
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}, "q4": set()}
    run = {"q1": ["a"], "q2": ["x", "b"]}
    assert mean_reciprocal_rank(run, qrels) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mean_reciprocal_rank_needs_relevant_documents():
    with pytest.raises(ValueError):
        mean_reciprocal_rank({}, {"q1": set()})
