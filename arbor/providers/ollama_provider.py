"""
Ollama provider — local models, completely free, no API key needed.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.

Quickstart:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull qwen2.5:7b
    3. Use: OllamaProvider(model="qwen2.5:7b")

Recommended models for Arbor (structured JSON output):
    qwen2.5:7b      — Best for structured tasks, fits in 8GB VRAM
    qwen2.5:14b     — Better quality, needs 16GB VRAM
    llama3.1:8b     — Good general purpose
    mistral:7b      — Fast, decent JSON compliance
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from arbor.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Local Ollama provider. Zero cost, runs fully offline.

    Ollama uses the OpenAI-compatible API format, so we reuse the openai client
    with a local base_url and a dummy API key.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,   # Local inference can be slow on CPU
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = openai.AsyncOpenAI(
            api_key="ollama",   # Ollama doesn't validate the API key
            base_url=f"{self._base_url}/v1",
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        content, _ = await self.complete_with_finish_reason(
            prompt, temperature, max_tokens, chat_history
        )
        return content

    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        messages = list(chat_history or [])
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = "stop" if choice.finish_reason == "stop" else "length"
        return content, finish_reason

    def count_tokens(self, text: str) -> int:
        # Rough approximation — Ollama doesn't expose a tokenizer API
        return len(text) // 4

    async def list_models(self) -> list[str]:
        """Return list of models available in the local Ollama instance."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from exc

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models = await self.list_models()
            return any(self.model in m for m in models)
        except Exception:
            return False
