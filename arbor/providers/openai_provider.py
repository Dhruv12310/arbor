"""
OpenAI-compatible provider.

Covers: OpenAI, Groq (free tier), Together AI, Perplexity,
any server that exposes an OpenAI-compatible /v1/chat/completions endpoint.

The Groq provider is just GroqProvider(api_key=...) — same code, different base_url.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from arbor.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    Standard OpenAI provider.

    Default model: gpt-4o-mini (cheap, capable).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        self.model = model
        self._timeout = timeout
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

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
        if _TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.encoding_for_model(self.model)
                return len(enc.encode(text))
            except Exception:
                pass
        return len(text) // 4


class GroqProvider(LLMProvider):
    """
    Groq provider — OpenAI-compatible API with free tier.

    Default model: llama-3.1-70b-versatile (free, very capable).
    Get a free API key at: https://console.groq.com

    Rate limits (free tier as of 2024):
        6000 tokens/min, 14400 requests/day per model.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-70b-versatile",
        timeout: float = 120.0,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        self.model = model
        self._timeout = timeout
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var. "
                "Get a free key at https://console.groq.com"
            )
        self._client = openai.AsyncOpenAI(
            api_key=resolved_key,
            base_url=self.GROQ_BASE_URL,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"groq/{self.model}"

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
        # Groq models are Llama/Mixtral — tiktoken doesn't support them natively.
        # Use a slightly more accurate heuristic: ~3.5 chars per token for English.
        return max(1, len(text) // 4)


class OpenAICompatibleProvider(LLMProvider):
    """
    Generic OpenAI-compatible provider.

    Use this for Together AI, Perplexity, Anyscale, vLLM, LM Studio,
    or any server exposing the /v1/chat/completions endpoint.

    Example:
        provider = OpenAICompatibleProvider(
            api_key="...",
            base_url="http://localhost:8000/v1",
            model="mistralai/Mistral-7B-Instruct-v0.2",
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        self.model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"openai-compat/{self.model}"

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
        return len(text) // 4
