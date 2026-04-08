"""Thin OpenAI chat-completions wrapper (httpx-based, no SDK dependency)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from llqm.utils import load_dotenv_file


class LLMClient:
    """Stateless OpenAI chat-completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4-mini",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Send a chat completion request and return the assistant message."""
        resp = self._client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self._model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> Any:
        """Send a chat request that returns structured JSON."""
        resp = self._client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self._model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system + "\nRespond in JSON."},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)


def build_llm() -> LLMClient | None:
    """Build an LLM client from env if OPENAI_API_KEY is available."""
    load_dotenv_file()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip()
    return LLMClient(api_key=key, model=model)
