"""Retrieval metrics for benchmarking literature search (e.g. against TREC BioGen topics)."""

def reciprocal_rank(ranked, relevant, cutoff=None):
    """1/rank of the first relevant document in `ranked` (best first), or 0 if none is found."""
    for rank, doc in enumerate(ranked[:cutoff] if cutoff else ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(run, qrels, cutoff=None):
    """MRR over questions that have at least one relevant document.

    run:   {question_id: [doc_id, ...]} best first
    qrels: {question_id: {relevant doc_id, ...}}
    A question missing from the run scores 0, so dropped questions are penalized.
    """
    questions = [q for q, relevant in qrels.items() if relevant]
    if not questions:
        raise ValueError("qrels has no question with a relevant document")
    return sum(reciprocal_rank(run.get(q, []), qrels[q], cutoff) for q in questions) / len(questions)
