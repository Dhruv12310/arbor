"""
Abstract LLM provider interface.

All providers implement this interface so the rest of Arbor is LLM-agnostic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """
    Abstract interface for LLM backends.

    Arbor works with any LLM by implementing this interface.
    Built-in providers: OpenAI, Groq, Ollama, Anthropic, HuggingFace.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Send a prompt and return the completion text.

        Args:
            prompt: The user prompt to send.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate. None = model default.
            chat_history: Prior messages as [{"role": "user"|"assistant", "content": str}].
                          Appended before the current prompt.

        Returns:
            The generated text (stripped).
        """
        ...

    @abstractmethod
    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        """
        Send a prompt and return (content, finish_reason).

        finish_reason values:
            "stop"   — completed normally
            "length" — hit max_tokens limit (output may be truncated)

        Used for detecting truncated TOC outputs that need continuation.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging."""
        ...

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for the given text.

        Override this method for exact counting (e.g. tiktoken for OpenAI models).
        Default: rough 4-chars-per-token approximation.
        """
        return len(text) // 4

    async def complete_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> str:
        """
        complete() with automatic retry on transient failures.

        Uses the same retry pattern as PageIndex (10 retries, 1s sleep).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return await self.complete(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chat_history=chat_history,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        raise RuntimeError(
            f"[{self.name}] Failed after {max_retries} attempts: {last_exc}"
        ) from last_exc
