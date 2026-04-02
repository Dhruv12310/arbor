"""Google Gemini provider via the google-genai SDK.

Install: pip install google-genai
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from arbor.providers.base import LLMProvider


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg


class GeminiProvider(LLMProvider):
    """Google Gemini provider via the google-genai SDK.

    Default model: gemini-2.5-flash-preview-04-17 (fast + cheap).
    Install: pip install google-genai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        max_tokens: int = 4096,
    ):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )
        self.model = model
        self._max_tokens = max_tokens
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    @property
    def name(self) -> str:
        return f"gemini/{self.model}"

    def _build_contents(
        self, prompt: str, chat_history: Optional[list[dict]]
    ) -> list[dict]:
        contents = []
        if chat_history:
            for msg in chat_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents

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
        contents = self._build_contents(prompt, chat_history)
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or self._max_tokens,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        candidate = response.candidates[0] if response.candidates else None
        finish = "stop"
        if candidate and hasattr(candidate, "finish_reason"):
            finish = "length" if str(candidate.finish_reason) in ("MAX_TOKENS", "2") else "stop"
        return text.strip(), finish

    async def complete_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> str:
        """Exponential backoff on rate limit errors, fixed delay on others."""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return await self.complete(prompt, temperature, max_tokens, chat_history)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = min(retry_delay * (2 ** attempt), 60.0) if _is_rate_limit(exc) else retry_delay
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"[{self.name}] Failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
