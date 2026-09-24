import copy
import re
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

# The project modules use flat imports relative to their own directories.
for sub in ("", "intake", "retrieval", "tools"):
    sys.path.insert(0, str(ROOT / sub))


class FakeEncoder:
    """Deterministic bag-of-words embedder so tests never download a model."""

    DIM = 32

    def __init__(self, name=None):
        self.name = name

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), self.DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# sentence-transformers pulls in torch; stub it when absent so retrieval logic is testable anywhere.
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _stub = types.ModuleType("sentence_transformers")
    _stub.SentenceTransformer = FakeEncoder
    sys.modules["sentence_transformers"] = _stub


BASE_ENTRY = {
    "id": "ku-molvirology-00000000",
    "version": "0.1.0",
    "claim_type": "real_null",
    "evidence_layer": "functional",
    "domain": "molecular virology",
    "title": "No effect of PTBP2 on poliovirus IRES translation",
    "observation": "Reporter output unchanged across the concentration series.",
    "conditions": "In vitro translation of an IRES reporter RNA.",
    "caveats": "Functional readout only.",
    "alternatives": "Confirm binding orthogonally.",
    "system": {},
    "method": {"name": "in vitro translation"},
    "confidence": {"level": "medium", "positive_control": {"present": False}},
    "related_positive": [],
    "contributor": {"name": "Tester"},
    "date": "2026-09-24",
    "references": [],
}


@pytest.fixture
def make_entry():
    def _make(**overrides):
        entry = copy.deepcopy(BASE_ENTRY)
        entry.update(overrides)
        return entry
    return _make


@pytest.fixture
def schema_path():
    return ROOT / "schema" / "entry.schema.json"


@pytest.fixture
def write_entry(tmp_path):
    counter = {"n": 0}

    def _write(entry, directory=None):
        directory = Path(directory or tmp_path)
        directory.mkdir(parents=True, exist_ok=True)
        counter["n"] += 1
        path = directory / f"entry-{counter['n']}.yaml"
        path.write_text(yaml.safe_dump(entry, sort_keys=False))
        return str(path)
    return _write


@pytest.fixture
def fake_encoder():
    return FakeEncoder
