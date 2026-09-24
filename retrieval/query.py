"""
Query the index. This layer enforces the rules that make retrieval trustworthy:

  RULE 1 (grounding):   every returned item carries its entry ID for citation.
  RULE 2 (confidence):  confidence + positive-control status are surfaced, never hidden.
  RULE 3 (layers):      you can filter by evidence_layer / claim_type so a
                        functional null and a binding result are not conflated.
  RULE 4 (anti-consensus): comparability.assess_raft() decides whether a cluster is a
                        real 'raft' (comparable conditions) or just topical proximity.
"""
import os, json, argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from comparability import assess_raft

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

def load_index():
    vecs = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
    with open(os.path.join(INDEX_DIR, "meta.json")) as f:
        meta = json.load(f)
    return vecs, meta["entries"], meta["model"]

def confidence_flag(conf: dict) -> str:
    """Turn structured confidence into a plain-language trust flag."""
    level = conf.get("level", "unknown")
    pc = conf.get("positive_control", {}) or {}
    if pc.get("present") is False:
        return f"{level} confidence — WEAKENED: no positive control (assay sensitivity unproven)"
    if pc.get("present") and pc.get("worked") is False:
        return f"{level} confidence — CAUTION: positive control did not work"
    if pc.get("present") and pc.get("worked"):
        return f"{level} confidence — positive control validated the assay"
    return f"{level} confidence"

def _faiss_search(q, entries, top_k, evidence_layer, claim_type, overfetch=10):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    # over-fetch because we filter AFTER ANN retrieval
    k = min(len(entries), top_k * overfetch)
    scores, ids = index.search(q.astype("float32").reshape(1, -1), k)
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        e = entries[i]
        if evidence_layer and e["evidence_layer"] != evidence_layer:
            continue
        if claim_type and e["claim_type"] != claim_type:
            continue
        hits.append({**e, "score": float(score)})
        if len(hits) >= top_k:
            break
    return hits

def query(text, top_k=5, evidence_layer=None, claim_type=None):
    vecs, entries, model_name = load_index()
    model = SentenceTransformer(model_name)
    q = model.encode([text], normalize_embeddings=True)[0]

    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(faiss_path):
        hits = _faiss_search(q, entries, top_k, evidence_layer, claim_type)
    else:
        # original flat path (unchanged)
        idx = list(range(len(entries)))
        if evidence_layer: idx = [i for i in idx if entries[i]["evidence_layer"] == evidence_layer]
        if claim_type:     idx = [i for i in idx if entries[i]["claim_type"] == claim_type]
        if not idx:        idx = []
        if idx:
            sims = vecs[idx] @ q
            order = np.argsort(-sims)[:top_k]
            hits = [{**entries[idx[o]], "score": float(sims[o])} for o in order]
        else:
            hits = []

    if not hits:
        return [], {"verdict": "no_match", "detail": "No entries match the requested filters."}
    return hits, assess_raft(hits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--layer", default=None, help="filter: binding/functional/phenotypic/computational/observational")
    ap.add_argument("--claim", default=None, help="filter: real_null/not_reproducible/not_meaningful/method_failure")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    hits, raft = query(args.question, top_k=args.k,
                       evidence_layer=args.layer, claim_type=args.claim)

    print(f"\nQ: {args.question}")
    print(f"\n[RAFT ASSESSMENT] {raft['verdict']} — {raft['detail']}\n")
    for h in hits:
        print(f"  [{h['id']}]  (score {h['score']:.3f})")
        print(f"    {h['title']}")
        print(f"    layer={h['evidence_layer']}  claim={h['claim_type']}")
        print(f"    {confidence_flag(h['confidence'])}")
        if h.get("related_positive"):
            print(f"    ⚠ linked contradicting/positive: {h['related_positive']}")
        print()

if __name__ == "__main__":
    main()
