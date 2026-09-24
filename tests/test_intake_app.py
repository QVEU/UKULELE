import importlib
import json
import os
import sys

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from validate import validate_entry  # noqa: E402

PASSWORD = "test-secret"
AUTH = {"X-KU-Password": PASSWORD}


def _import_app():
    sys.modules.pop("intake_app", None)
    return importlib.import_module("intake_app")


@pytest.fixture
def app_mod(monkeypatch, tmp_path):
    monkeypatch.setenv("KU_APP_PASSWORD", PASSWORD)
    monkeypatch.setenv("KU_ENTRIES_DIR", str(tmp_path / "entries"))
    return _import_app()


@pytest.fixture
def client(app_mod):
    return TestClient(app_mod.app)


def _update(**fields):
    return json.dumps({k: {"value": v, "provenance": "stated", "source": f"said {k}"}
                       for k, v in fields.items()})


COMPLETE = dict(
    claim_type="real_null", evidence_layer="functional", domain="virology",
    title="No effect", observation="Nothing changed.", conditions="In vitro.",
    caveats="Functional only.", system={"model": "RRL"},
    conditions_structured={"system": "RRL", "key_params": {"Mg2+": "2 mM"}},
    method={"name": "IVT"}, confidence_level="medium",
    positive_control={"present": False}, powered=True, n="3",
    contributor={"name": "Tester"}, date="2026-09-24",
)


def scripted_llm(updates):
    def _llm(prompt):
        if "Return ONLY a JSON object" in prompt:
            return updates
        if "confirmation summary" in prompt:
            return "STATED: ...\nINFERRED: ...\nSTILL UNKNOWN: ..."
        return "Was a positive control run?"
    return _llm


def test_refuses_to_start_without_password(monkeypatch):
    monkeypatch.delenv("KU_APP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="KU_APP_PASSWORD"):
        _import_app()


def test_home_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Knowledge Universe" in r.text


@pytest.mark.parametrize("path", ["/api/start", "/api/message", "/api/confirm", "/api/submit"])
@pytest.mark.parametrize("headers", [{}, {"X-KU-Password": "wrong"}])
def test_api_requires_password(client, path, headers):
    body = {"session_id": "x", "message": "hi"}
    assert client.post(path, json=body, headers=headers).status_code == 401


def test_full_flow_writes_valid_entry(client, app_mod, monkeypatch, schema_path):
    monkeypatch.setattr(app_mod, "call_llm", scripted_llm(_update(**COMPLETE)))
    sid = client.post("/api/start", headers=AUTH).json()["session_id"]

    r = client.post("/api/message", json={"session_id": sid, "message": "notes"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["ready_for_confirmation"] is True

    summary = client.post("/api/confirm", json={"session_id": sid, "message": ""}, headers=AUTH).json()
    assert "STILL UNKNOWN" in summary["summary"]

    written = client.post("/api/submit", json={"session_id": sid, "message": ""}, headers=AUTH).json()
    assert os.path.dirname(written["written"]) == app_mod.ENTRIES_DIR
    entry = yaml.safe_load(written["yaml"])
    assert "conditions_structured" not in entry
    assert entry["system"]["conditions_structured"] == COMPLETE["conditions_structured"]
    assert validate_entry(written["written"], str(schema_path))


def test_unfinished_session_asks_a_question(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "call_llm", scripted_llm("{}"))
    sid = client.post("/api/start", headers=AUTH).json()["session_id"]
    r = client.post("/api/message", json={"session_id": sid, "message": "notes"}, headers=AUTH).json()
    assert r == {"agent": "Was a positive control run?", "ready_for_confirmation": False}


def test_unknown_powered_still_validates(client, app_mod, monkeypatch, schema_path):
    fields = {k: v for k, v in COMPLETE.items() if k != "powered"}
    monkeypatch.setattr(app_mod, "call_llm", scripted_llm(_update(**fields)))
    sid = client.post("/api/start", headers=AUTH).json()["session_id"]
    client.post("/api/message", json={"session_id": sid, "message": "notes"}, headers=AUTH)
    written = client.post("/api/submit", json={"session_id": sid, "message": ""}, headers=AUTH).json()
    assert "powered" not in yaml.safe_load(written["yaml"])["confidence"]
    assert validate_entry(written["written"], str(schema_path))


def test_submit_filename_cannot_escape_entries_dir(client, app_mod):
    sid = client.post("/api/start", headers=AUTH).json()["session_id"]
    written = client.post("/api/submit", json={"session_id": sid, "message": "../../etc/passwd"},
                          headers=AUTH).json()["written"]
    assert os.path.dirname(written) == app_mod.ENTRIES_DIR


def test_unknown_session_is_404(client):
    for path in ("/api/message", "/api/confirm", "/api/submit"):
        assert client.post(path, json={"session_id": "nope", "message": "x"}, headers=AUTH).status_code == 404


def test_bill_guards(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "call_llm", scripted_llm("{}"))
    sid = client.post("/api/start", headers=AUTH).json()["session_id"]
    too_long = "x" * (app_mod.MAX_MSG_CHARS + 1)
    assert client.post("/api/message", json={"session_id": sid, "message": too_long}, headers=AUTH).status_code == 413
    app_mod.SESSIONS[sid].turns = app_mod.MAX_TURNS_PER_SESSION
    assert client.post("/api/message", json={"session_id": sid, "message": "hi"}, headers=AUTH).status_code == 429


def test_parse_turn_safety(app_mod, monkeypatch):
    raw = json.dumps({
        "confidence_level": {"value": "high", "provenance": "inferred", "source": "looks solid"},
        "caveats": {"value": "none", "provenance": "assumed", "source": "?"},
        "n": "3",
        "favourite_colour": {"value": "blue", "provenance": "stated"},
        "title": {"value": "T", "provenance": "Stated", "source": "their words"},
    })
    monkeypatch.setattr(app_mod, "call_llm", lambda p: raw)
    sess = app_mod.Session()
    app_mod.parse_turn(sess, "notes")
    assert sess.fields["confidence_level"]["provenance"] == "unknown"
    assert sess.fields["caveats"]["provenance"] == "unknown"
    assert sess.fields["n"]["provenance"] == "unknown"
    assert "favourite_colour" not in sess.fields
    assert sess.fields["title"] == {"value": "T", "provenance": "stated", "source": "their words"}


def test_parse_failure_changes_nothing(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "call_llm", lambda p: "I could not parse that")
    sess = app_mod.Session()
    before = json.dumps(sess.fields, sort_keys=True)
    app_mod.parse_turn(sess, "notes")
    assert json.dumps(sess.fields, sort_keys=True) == before
