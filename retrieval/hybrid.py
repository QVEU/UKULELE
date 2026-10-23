"""Combine ranked result lists from different retrievers (e.g. BM25 and dense)."""

def rrf_fuse(rankings, k=60, weights=None, limit=None):
    """Reciprocal-rank fusion: score(d) = sum over rankings of w / (k + rank of d), ranks from 1.

    Repeats within one ranking (e.g. several passages from the same paper) are dropped
    before ranking, so each document is ranked among distinct documents. Returns
    [(doc_id, score)] best first; ties keep the order in which documents were first seen.
    """
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")
    scores, first_seen = {}, {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc, len(first_seen))
    fused = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return fused[:limit] if limit else fused
