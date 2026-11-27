"""Pluggable LLM backends, selected with ``ASRS_RAG_LLM``.

* ``ollama`` - local Ollama server at ``OLLAMA_HOST``; no credentials.
* ``transformers`` - small Hugging Face instruct model run in-process on CPU;
  no credentials, no server.
* ``openai`` - any OpenAI-compatible chat API; key from ``OPENAI_API_KEY``
  (``OPENAI_BASE_URL`` for self-hosted endpoints).
* ``anthropic`` - Anthropic Messages API; key from ``ANTHROPIC_API_KEY``.
* ``none`` - no LLM. Retrieval, the supervised classifier and template
  summaries keep working.
* ``auto`` (default) - Ollama if reachable, else an API whose key is set,
  else ``none``.

Keys are only ever read from environment variables; nothing is hard-coded or
logged. ``ASRS_RAG_LLM_MODEL`` overrides the default model of any backend.
All calls use temperature 0 so results are as repeatable as the backend allows.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import requests

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "ollama": "qwen2.5:1.5b",
    "transformers": "Qwen/Qwen2.5-1.5B-Instruct",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}


class LLMError(RuntimeError):
    """Raised when a backend call fails."""


class LLM(Protocol):
    """Interface every backend implements."""

    name: str

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:  # pragma: no cover - protocol
        """Return the model's text response."""
        ...


def _model(backend: str, model: str | None) -> str:
    return model or os.environ.get("ASRS_RAG_LLM_MODEL") or DEFAULT_MODELS[backend]


class OllamaLLM:
    """Chat with a local Ollama server."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300):
        self.model = _model("ollama", model)
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def available(self) -> bool:
        """True if the server answers and has the model pulled."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            r.raise_for_status()
        except requests.RequestException:
            return False
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int = 512, json_mode: bool = False
    ) -> str:
        """Call ``/api/chat`` (non-streaming)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "seed": 13, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc


class OpenAICompatibleLLM:
    """Any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, vLLM, LM Studio...)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: int = 120):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.model = _model("openai", model)
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self.name = f"openai:{self.model}"

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int = 512, json_mode: bool = False
    ) -> str:
        """Call ``/chat/completions``."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"OpenAI-compatible call failed: {exc}") from exc


class AnthropicLLM:
    """Anthropic Messages API (key from ``ANTHROPIC_API_KEY``)."""

    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, timeout: int = 120):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = _model("anthropic", model)
        self.timeout = timeout
        self.name = f"anthropic:{self.model}"

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int = 512, json_mode: bool = False
    ) -> str:
        """Call ``/v1/messages``; ``json_mode`` is enforced through the prompt."""
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc


class TransformersLLM:
    """Run a small instruct model in-process with Hugging Face ``transformers`` (CPU).

    Slower than Ollama but needs no server; used to produce this project's
    published LLM results in an environment where Ollama could not be run.
    """

    def __init__(self, model: str | None = None, threads: int | None = None):
        self.model_id = _model("transformers", model)
        self.name = f"transformers:{self.model_id}"
        self.threads = threads
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.threads:
            torch.set_num_threads(self.threads)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16)
        self._model.eval()

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int = 512, json_mode: bool = False
    ) -> str:
        """Greedy decoding with the model's chat template."""
        import torch

        self._load()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        ids = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.inference_mode():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


def get_llm(backend: str | None = None, model: str | None = None) -> LLM | None:
    """Return the configured backend, or ``None`` for no-LLM mode.

    ``backend`` defaults to ``$ASRS_RAG_LLM`` or ``auto``.
    """
    backend = (backend or os.environ.get("ASRS_RAG_LLM") or "auto").lower()
    if backend == "none":
        return None
    if backend == "ollama":
        return OllamaLLM(model)
    if backend == "transformers":
        return TransformersLLM(model)
    if backend == "openai":
        return OpenAICompatibleLLM(model)
    if backend == "anthropic":
        return AnthropicLLM(model)
    if backend != "auto":
        raise ValueError(f"Unknown LLM backend {backend!r}")

    ollama = OllamaLLM(model)
    if ollama.available():
        return ollama
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleLLM(model)
    log.info("No LLM available; running in no-LLM mode")
    return None
