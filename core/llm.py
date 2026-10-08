"""
Minimal client for any OpenAI-compatible Chat Completions API.

Works with Groq (default, has a free tier), OpenAI, Together, OpenRouter, or a local
Ollama server, by changing three environment variables:

    LLM_API_KEY   the key (for Ollama any non-empty string)
    LLM_BASE_URL  default https://api.groq.com/openai/v1   (Ollama: http://localhost:11434/v1)
    LLM_MODEL     default llama-3.3-70b-versatile

The key is read on the server only and is never sent to the browser.
With no key set, is_configured() is False and the app uses its offline fallback.
"""
from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None, post=None):
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT", "25"))
        self._post = post or requests.post  # injectable so tests never touch the network

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2,
             max_tokens: int = 700) -> dict:
        """Returns the assistant message: {"role": "assistant", "content": str|None, "tool_calls": [...]?}"""
        if not self.is_configured:
            raise LLMError("No LLM_API_KEY set")
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            res = self._post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        except requests.RequestException as e:
            raise LLMError(f"LLM request failed: {e.__class__.__name__}") from e
        if res.status_code != 200:
            try:
                detail = res.json().get("error", {}).get("message", "")
            except ValueError:
                detail = ""
            raise LLMError(f"LLM API returned {res.status_code}{': ' + detail if detail else ''}")
        try:
            return res.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("Unexpected LLM response shape") from e
