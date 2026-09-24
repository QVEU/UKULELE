"""
Build an embedding index from all entries.

Stores, per entry: the embedding vector, the composed text, and the STRUCTURED
fields the query layer needs to reason about (claim_type, evidence_layer,
confidence, conditions, related_positive). We embed prose but we RETRIEVE with
structure — that separation is the whole point.
"""
import glob, os, json, yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, offline-friendly; swap freely
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

def compose_embedding_text(e: dict) -> str:
    parts = [
        f"CLAIM TYPE: {e['claim_type']}",
        f"EVIDENCE LAYER: {e['evidence_layer']}",
        f"DOMAIN: {e['domain']}",
        f"METHOD: {e.get('method', {}).get('name', 'unspecified')}",
        f"CONFIDENCE: {e.get('confidence', {}).get('level', 'unspecified')}",
        f"TITLE: {e['title']}",
        f"OBSERVATION: {e['observation']}",
        f"CONDITIONS: {e.get('conditions', '')}",
        f"CAVEATS: {e.get('caveats', '')}",
        f"ALTERNATIVES: {e.get('alternatives', '')}",
    ]
    return "\n".join(p for p in parts if p.strip())

def load_entries(path="entries/"):
    out = []
    for f in glob.glob(os.path.join(path, "*.y*ml")):
        with open(f) as fh:
            out.append(yaml.safe_load(fh))
    return out

def main():
    entries = load_entries()
    if not entries:
        print("No entries to index."); return
    model = SentenceTransformer(MODEL_NAME)
    texts = [compose_embedding_text(e) for e in entries]
    vecs = model.encode(texts, normalize_embeddings=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])   # inner product == cosine (vecs normalized)
    index = faiss.IndexIDMap(index)            # lets us map back to entry rows
    index.add_with_ids(vecs.astype("float32"), np.arange(len(vecs)))
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    print("Wrote FAISS index.")

    # Keep the structured metadata the query layer reasons over
    meta = [{
        "id": e["id"],
        "title": e["title"],
        "claim_type": e["claim_type"],
        "evidence_layer": e["evidence_layer"],
        "domain": e["domain"],
        "confidence": e.get("confidence", {}),
        "conditions": e.get("conditions", ""),
        "caveats": e.get("caveats", ""),
        "alternatives": e.get("alternatives", ""),
        "related_positive": e.get("related_positive", []),
    } for e in entries]
    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "entries": meta}, f, indent=2)
    print(f"Indexed {len(entries)} entries with {MODEL_NAME}.")

if __name__ == "__main__":
    main()
