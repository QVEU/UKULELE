"""Guard against invented findings: each one must quote text that is really in its passage.

Findings follow schema/finding.schema.json. Normalization is limited to what can't
change meaning (Unicode compatibility forms, typographic quotes and dashes,
whitespace), so a quote that paraphrases, stitches fragments, or changes a word fails.
"""
import re
import unicodedata

MIN_QUOTE_CHARS = 12

_PUNCT = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
})

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()

def verify_quotes(findings, passages, min_chars=MIN_QUOTE_CHARS):
    """Split findings into (kept, rejected). `passages` maps passage_id -> text.

    Each rejected item is {"finding": ..., "reason": ...} so the reason can be logged
    or shown to a reviewer.
    """
    normalized = {}
    kept, rejected = [], []
    for f in findings:
        source = f.get("source") if isinstance(f, dict) else None
        if not isinstance(source, dict):
            rejected.append({"finding": f, "reason": "no source"})
            continue
        pid, quote = source.get("passage_id"), normalize_text(source.get("quote"))
        if pid not in passages:
            reason = f"unknown passage {pid!r}"
        elif len(quote) < min_chars:
            reason = f"quote shorter than {min_chars} characters"
        else:
            if pid not in normalized:
                normalized[pid] = normalize_text(passages[pid])
            if quote in normalized[pid]:
                kept.append(f)
                continue
            reason = "quote not found verbatim in passage"
        rejected.append({"finding": f, "reason": reason})
    return kept, rejected
