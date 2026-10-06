"""Turn PMC JATS XML articles into retrievable passages with stable IDs.

Passage IDs follow schema/finding.schema.json:
  <PMCID>:ti            article title
  <PMCID>:ab:p<n>       nth abstract paragraph
  <PMCID>:s<path>:p<n>  nth paragraph of body section <path> (1-based, nested as 3.2);
                        s0 holds paragraphs that sit directly in <body>

Usage: python literature/pmc.py --pmcids Challenge_PMCIDs.txt --out passages.jsonl.gz FILE.xml [...]
"""
import argparse, gzip, json, re
import xml.etree.ElementTree as ET

_PMCID = re.compile(r"^(?:PMC)?([0-9]+)$", re.IGNORECASE)
_SKIP_ABSTRACT_TYPES = {"graphical", "teaser", "toc"}
_CONTAINERS = {"list", "list-item", "disp-quote", "boxed-text", "def-list", "def-item", "def"}

def normalize_pmcid(value):
    m = _PMCID.match((value or "").strip())
    return f"PMC{m.group(1)}" if m else None

def load_challenge_pmcids(path):
    """Read the challenge's PMCID list. A non-ID first line is treated as a header."""
    pmcids = set()
    with open(path) as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pmcid = normalize_pmcid(line)
            if pmcid:
                pmcids.add(pmcid)
            elif n > 1:
                raise ValueError(f"{path}:{n}: not a PMCID: {line!r}")
    return pmcids

def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""

def _paragraphs(el):
    """Paragraph elements belonging to `el` itself: direct <p> and <p> inside lists/quotes/boxes."""
    for child in el:
        if child.tag == "p":
            yield child
        elif child.tag in _CONTAINERS:
            yield from _paragraphs(child)

def _section_passages(sec, path, pmcid, title_trail):
    out, n = [], 0
    for p in _paragraphs(sec):
        text = _text(p)
        if text:
            n += 1
            out.append({"passage_id": f"{pmcid}:s{path}:p{n}", "section": " > ".join(title_trail), "text": text})
    for i, sub in enumerate(sec.findall("sec"), start=1):
        sub_path = f"{path}.{i}" if path != "0" else str(i)
        out += _section_passages(sub, sub_path, pmcid, title_trail + [_text(sub.find("title"))])
    return out

def parse_jats(xml_bytes, pmcid=None):
    """Parse one JATS article into ids, title, passages, and the PMIDs it cites."""
    root = ET.fromstring(xml_bytes)
    meta = root.find("front/article-meta")
    ids = {}
    for el in meta.findall("article-id") if meta is not None else []:
        ids.setdefault(el.get("pub-id-type"), _text(el))
    pmcid = normalize_pmcid(pmcid or ids.get("pmcid") or ids.get("pmc"))
    if not pmcid:
        raise ValueError("article has no PMCID; pass pmcid=")

    title = _text(meta.find("title-group/article-title")) if meta is not None else ""
    passages = [{"passage_id": f"{pmcid}:ti", "section": "Title", "text": title}] if title else []

    n = 0
    for abstract in meta.findall("abstract") if meta is not None else []:
        if abstract.get("abstract-type") in _SKIP_ABSTRACT_TYPES:
            continue
        for p in abstract.iter("p"):
            text = _text(p)
            if text:
                n += 1
                passages.append({"passage_id": f"{pmcid}:ab:p{n}", "section": "Abstract", "text": text})

    body = root.find("body")
    if body is not None:
        passages += _section_passages(body, "0", pmcid, [])

    cited = []
    for el in root.iterfind("back/ref-list//pub-id[@pub-id-type='pmid']"):
        pmid = _text(el)
        if pmid.isdigit() and pmid not in cited:
            cited.append(pmid)

    pmid = ids.get("pmid")
    return {"pmcid": pmcid, "pmid": pmid if pmid and pmid.isdigit() else None,
            "doi": ids.get("doi") or None, "title": title, "passages": passages, "cited_pmids": cited}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="local JATS XML files")
    ap.add_argument("--pmcids", help="Challenge_PMCIDs.txt; files for other articles are skipped")
    ap.add_argument("--out", required=True, help="output .jsonl.gz, one article per line")
    args = ap.parse_args(argv)
    allowed = load_challenge_pmcids(args.pmcids) if args.pmcids else None
    kept = skipped = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for path in args.files:
            with open(path, "rb") as fh:
                article = parse_jats(fh.read())
            if allowed is not None and article["pmcid"] not in allowed:
                skipped += 1
                continue
            out.write(json.dumps(article, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} articles to {args.out}; skipped {skipped} not in the challenge list")

if __name__ == "__main__":
    main()
