"""
Token counting utilities.

Uses tiktoken for exact counts when available (optional dep).
Falls back to a fast character-based approximation otherwise.

The approximation (len // 4) is the same heuristic PageIndex uses for
non-OpenAI providers. It's accurate enough for chunking decisions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# Default model for tiktoken encoding (matches PageIndex's default)
_DEFAULT_MODEL = "gpt-4o"

# Fallback: characters per token (conservative estimate)
_CHARS_PER_TOKEN = 4


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in the given text.

    Uses tiktoken if available, otherwise falls back to len(text) // 4.

    Args:
        text: The text to count tokens for.
        model: Model name for tiktoken encoding. Defaults to gpt-4o.

    Returns:
        Approximate or exact token count (always >= 1 for non-empty text).
    """
    if not text:
        return 0

    if _TIKTOKEN_AVAILABLE:
        try:
            enc = _get_encoding(model or _DEFAULT_MODEL)
            return len(enc.encode(text))
        except Exception:
            pass  # Fall through to approximation

    return max(1, len(text) // _CHARS_PER_TOKEN)


def count_tokens_approx(text: str) -> int:
    """
    Fast character-based token approximation. Never uses tiktoken.

    Use when speed matters more than accuracy (e.g., quick size checks).
    """
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


def is_tiktoken_available() -> bool:
    """Return True if tiktoken is installed."""
    return _TIKTOKEN_AVAILABLE


@lru_cache(maxsize=8)
def _get_encoding(model: str):
    """Cached tiktoken encoding lookup."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Model not found — use cl100k_base (GPT-4 encoding, works for most models)
        return tiktoken.get_encoding("cl100k_base")
