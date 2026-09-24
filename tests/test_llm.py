import sys
import types

import pytest

import llm


def test_defaults_to_echo(monkeypatch):
    monkeypatch.delenv("KU_LLM", raising=False)
    assert llm.call_llm("hello") == "[echo — no LLM called]\n\nhello"


def test_backend_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("KU_LLM", "ECHO")
    assert llm.call_llm("hi").endswith("hi")


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("KU_LLM", "llamacorp")
    with pytest.raises(ValueError, match="llamacorp"):
        llm.call_llm("hi")


def test_anthropic_backend(monkeypatch):
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            text = types.SimpleNamespace(text="grounded answer")
            return types.SimpleNamespace(content=[text, types.SimpleNamespace(type="other")])

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("KU_LLM", "anthropic")
    monkeypatch.delenv("KU_LLM_MODEL", raising=False)

    assert llm.call_llm("prompt") == "grounded answer"
    assert captured["temperature"] == 0
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["messages"] == [{"role": "user", "content": "prompt"}]


def test_openai_backend_honours_model_and_base_url(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = types.SimpleNamespace(content="ok")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def OpenAI(base_url=None):
        captured["base_url"] = base_url
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("KU_LLM", "openai")
    monkeypatch.setenv("KU_LLM_MODEL", "local-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    assert llm.call_llm("prompt") == "ok"
    assert captured["model"] == "local-model"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["temperature"] == 0
