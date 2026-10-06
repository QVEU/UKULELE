"""Stream records out of PubMed baseline XML files (plain or .gz) and write them as JSONL.

Each output line has Pyserini's JsonCollection fields ("id", "contents") plus metadata,
so the files can be indexed for BM25 directly and reused for dense encoding.

Usage: python literature/pubmed_baseline.py pubmed26n0001.xml.gz [...] --out corpus/pubmed/
"""
import argparse, gzip, json, os, re
import xml.etree.ElementTree as ET

PREPRINT_UI = "D000076942"  # MeSH publication type "Preprint"; excluded from the SPARK corpus
_YEAR = re.compile(r"(1[89]|20)\d\d")

def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""

def _year(article):
    for path in ("Journal/JournalIssue/PubDate/Year", "Journal/JournalIssue/PubDate/MedlineDate",
                 "ArticleDate/Year"):
        m = _YEAR.search(_text(article.find(path)))
        if m:
            return int(m.group(0))
    return None

def _abstract(article):
    parts = []
    for node in article.findall("Abstract/AbstractText"):
        text = _text(node)
        if text:
            label = node.get("Label")
            parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)

def parse_article(elem):
    """Turn one <PubmedArticle> element into a record dict, or None if it is a preprint."""
    citation = elem.find("MedlineCitation")
    article = citation.find("Article")
    pub_types = [{"ui": p.get("UI"), "name": _text(p)} for p in article.findall("PublicationTypeList/PublicationType")]
    if any(p["ui"] == PREPRINT_UI for p in pub_types):
        return None
    ids = {i.get("IdType"): _text(i) for i in elem.findall("PubmedData/ArticleIdList/ArticleId")}
    pmcid = ids.get("pmc")
    return {
        "pmid": _text(citation.find("PMID")),
        "title": _text(article.find("ArticleTitle")),
        "abstract": _abstract(article),
        "journal": _text(article.find("Journal/Title")),
        "year": _year(article),
        "pub_types": pub_types,
        "mesh": [{"ui": d.get("UI"), "name": _text(d)}
                 for d in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")],
        "doi": ids.get("doi") or None,
        "pmcid": pmcid if pmcid and pmcid.startswith("PMC") else (f"PMC{pmcid}" if pmcid else None),
    }

def iter_records(path, stats=None):
    """Yield records from one baseline file. `stats` (a dict) collects skip counts if given."""
    stats = {} if stats is None else stats
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        root = None
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            if root is None:
                root = elem
            if event != "end" or elem.tag not in ("PubmedArticle", "PubmedBookArticle", "DeleteCitation"):
                continue
            if elem.tag == "PubmedArticle":
                record = parse_article(elem)
                if record is None:
                    stats["preprints"] = stats.get("preprints", 0) + 1
                else:
                    yield record
            else:
                stats[elem.tag] = stats.get(elem.tag, 0) + 1
            root.clear()  # free parsed articles; baseline files hold ~30k each

def passages(record):
    """The retrievable passages of one record: (passage_id, text) for the title and the abstract."""
    out = [(f"PMID:{record['pmid']}:ti", record["title"])]
    if record["abstract"]:
        out.append((f"PMID:{record['pmid']}:ab", record["abstract"]))
    return out

def to_json_line(record):
    contents = f"{record['title']}\n{record['abstract']}".strip()
    return json.dumps({"id": record["pmid"], "contents": contents, **record}, ensure_ascii=False)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True, help="directory for <name>.jsonl.gz files")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.files:
        stats, count = {}, 0
        name = os.path.basename(path).split(".xml")[0]
        with gzip.open(os.path.join(args.out, f"{name}.jsonl.gz"), "wt", encoding="utf-8") as out:
            for record in iter_records(path, stats):
                out.write(to_json_line(record) + "\n")
                count += 1
        print(f"{path}: {count} records, skipped {stats or 'none'}")

if __name__ == "__main__":
    main()
