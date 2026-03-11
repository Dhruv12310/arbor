"""
Anthropic (Claude) provider.

Uses the anthropic SDK directly (not OpenAI-compatible shim) for better
control over system prompts, extended thinking, and token counting.

Install: pip install arbor-rag[anthropic]
         or: pip install anthropic
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from arbor.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Claude provider via the Anthropic SDK.

    Default model: claude-haiku-4-5-20251001 (fast + cheap for structured tasks).
    For best quality: claude-sonnet-4-6 or claude-opus-4-6.
    """

    # Map of short aliases to full model IDs
    MODEL_ALIASES: dict[str, str] = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required. "
                "Install with: pip install anthropic  or  pip install arbor-rag[anthropic]"
            )
        # Resolve model aliases
        self.model = self.MODEL_ALIASES.get(model, model)
        self._default_max_tokens = max_tokens
        self._timeout = timeout
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

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
        # Build messages list: history + current prompt
        messages: list[dict] = list(chat_history or [])
        messages.append({"role": "user", "content": prompt})

        response = await self._client.messages.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self._default_max_tokens,
        )

        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text

        # Anthropic stop reasons: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
        finish_reason = "stop" if response.stop_reason == "end_turn" else "length"
        return content_text.strip(), finish_reason

    def count_tokens(self, text: str) -> int:
        """Anthropic uses ~3.5 chars per token for English. Rough approximation."""
        return max(1, len(text) // 4)
