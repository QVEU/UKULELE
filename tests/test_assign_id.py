import re
import subprocess
import sys

import pytest
import yaml

import assign_id_inplace as aid
from conftest import ROOT
from validate import validate_entry


def test_slugify():
    assert aid.slugify("Molecular Virology / RNA biology") == "molecularvir"
    assert aid.slugify("") == "misc"
    assert aid.slugify(None) == "misc"
    assert aid.slugify("!!!") == "misc"


def test_make_id_is_deterministic_and_content_sensitive(make_entry):
    entry = make_entry()
    assert aid.make_id(entry) == aid.make_id(make_entry())
    assert aid.make_id(entry) != aid.make_id(make_entry(title="Something else"))
    assert re.fullmatch(r"ku-[a-z0-9]*-[0-9a-f]{8}", aid.make_id(entry))


@pytest.mark.parametrize("value,is_placeholder", [
    ("ku-pending-00000000", True),
    ("ku-molvirology-00000000", True),
    ("ku--00000000", True),
    ("", True),
    ("   ", True),
    ("ku-molvirology-1a2b3c4d", False),
    ("ku-pending-000000001", False),
])
def test_placeholder_pattern(value, is_placeholder):
    assert bool(aid.PLACEHOLDER.match(value)) is is_placeholder


def test_process_assigns_id_and_preserves_formatting(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text("# header comment\nid: ku-pending-00000000\n"
                    "title: A title   # inline comment\ndomain: cell biology\n")
    assert aid.process(str(path)) is True
    text = path.read_text()
    assert "# header comment" in text and "# inline comment" in text
    new_id = yaml.safe_load(text)["id"]
    assert new_id.startswith("ku-cellbiology-") and not new_id.endswith("00000000")

    assert aid.process(str(path)) is False
    assert path.read_text() == text


def test_process_adds_missing_id_line(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text("title: No id yet\ndomain: ecology\n")
    assert aid.process(str(path)) is True
    assert yaml.safe_load(path.read_text())["id"].startswith("ku-ecology-")


def test_process_leaves_real_ids_alone(tmp_path):
    path = tmp_path / "e.yaml"
    original = "id: ku-ecology-1a2b3c4d\ntitle: Done\n"
    path.write_text(original)
    assert aid.process(str(path)) is False
    assert path.read_text() == original


def test_assigned_id_passes_schema(tmp_path, schema_path):
    path = tmp_path / "ptbp2.yaml"
    text = (ROOT / "entries" / "ptbp2-entry.yaml").read_text()
    path.write_text(re.sub(r"(?m)^id:.*$", "id: ku-pending-00000000", text, count=1))
    assert aid.process(str(path)) is True
    assert validate_entry(str(path), str(schema_path))


def test_cli_assigns_ids_to_every_placeholder_entry(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.yaml").write_text(
            f"id: ku-pending-00000000\ntitle: Entry {name}\ndomain: ecology\n")
    subprocess.run([sys.executable, "tools/assign_id_inplace.py", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True)
    ids = [yaml.safe_load((tmp_path / f"{n}.yaml").read_text())["id"] for n in "abc"]
    assert not any(i.endswith("00000000") for i in ids), ids
    assert len(set(ids)) == 3
