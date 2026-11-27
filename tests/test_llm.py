"""LLM backend tests with mocked HTTP (no network, no keys)."""

from __future__ import annotations

import pytest
import requests

from asrs_rag import llm as llm_mod
from asrs_rag.llm import (
    AnthropicLLM,
    LLMError,
    OllamaLLM,
    OpenAICompatibleLLM,
    get_llm,
)


class Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "ASRS_RAG_LLM",
        "ASRS_RAG_LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OLLAMA_HOST",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_ollama_request_shape(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, body=json)
        return Resp({"message": {"content": '{"labels": []}'}})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    out = OllamaLLM("m1").generate("hi", system="sys", json_mode=True, max_tokens=7)
    assert out == '{"labels": []}'
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["format"] == "json"
    assert seen["body"]["options"]["temperature"] == 0
    assert seen["body"]["options"]["num_predict"] == 7
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_errors_are_wrapped(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(LLMError):
        OllamaLLM().generate("hi")


def test_ollama_available_checks_model(monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests, "get", lambda *a, **k: Resp({"models": [{"name": "qwen2.5:1.5b"}]})
    )
    assert OllamaLLM("qwen2.5:1.5b").available()
    assert not OllamaLLM("other").available()


def test_api_backends_need_keys():
    with pytest.raises(LLMError):
        OpenAICompatibleLLM()
    with pytest.raises(LLMError):
        AnthropicLLM()


def test_openai_uses_env_key_in_header_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, body=json, headers=headers)
        return Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    llm = OpenAICompatibleLLM("gpt-test")
    assert llm.generate("q", json_mode=True) == "ok"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert "sk-test" not in str(seen["body"]) and "sk-test" not in llm.name
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-test")
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, body=json, headers=headers)
        return Resp({"content": [{"type": "text", "text": "hello"}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    assert AnthropicLLM("model-x").generate("q", system="s") == "hello"
    assert seen["headers"]["x-api-key"] == "key-test"
    assert seen["body"]["system"] == "s" and seen["body"]["model"] == "model-x"


def test_get_llm_none_and_auto_fallback(monkeypatch):
    assert get_llm("none") is None
    monkeypatch.setattr(OllamaLLM, "available", lambda self: False)
    assert get_llm() is None  # auto, no server, no keys -> no-LLM mode
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(get_llm(), AnthropicLLM)
    monkeypatch.setattr(OllamaLLM, "available", lambda self: True)
    assert isinstance(get_llm(), OllamaLLM)


def test_get_llm_env_and_model_override(monkeypatch):
    monkeypatch.setenv("ASRS_RAG_LLM", "ollama")
    monkeypatch.setenv("ASRS_RAG_LLM_MODEL", "llama3.2:3b")
    llm = get_llm()
    assert isinstance(llm, OllamaLLM) and llm.model == "llama3.2:3b"
    with pytest.raises(ValueError):
        get_llm("bogus")
