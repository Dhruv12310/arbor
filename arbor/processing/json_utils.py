"""
JSON extraction and LLM output normalization utilities.

LLMs often return slightly malformed JSON:
  - Wrapped in ```json ... ``` code fences
  - Python-style None instead of JSON null
  - Trailing commas before } or ]
  - Single quotes instead of double quotes

This module handles all of that robustly.
Also contains the continuation logic for truncated LLM outputs.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from arbor.providers.base import LLMProvider


# ─── JSON Extraction ──────────────────────────────────────────────────────────

def extract_json(content: str) -> Any:
    """
    Extract and parse JSON from LLM output.

    Handles:
      - ```json ... ``` or ``` ... ``` code fences
      - Python None/True/False → JSON null/true/false
      - Trailing commas before } and ]
      - Leading/trailing whitespace

    Args:
        content: Raw LLM output string.

    Returns:
        Parsed Python object (dict, list, str, etc.)

    Raises:
        ValueError: If the content cannot be parsed as JSON after all fixes.
    """
    text = content.strip()

    # Strip code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Replace Python None/True/False with JSON equivalents
    # Use word boundaries to avoid replacing substrings
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try ast.literal_eval as fallback (handles single quotes, etc.)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    # Last resort: find the first JSON object or array in the text
    for pattern in (r'\{.*\}', r'\[.*\]'):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(0)
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            candidate = re.sub(r'\bNone\b', 'null', candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"Could not parse JSON from LLM output. "
        f"Content (first 500 chars): {content[:500]!r}"
    )


def safe_extract_json(content: str, default: Any = None) -> Any:
    """
    Like extract_json() but returns `default` instead of raising on failure.
    """
    try:
        return extract_json(content)
    except (ValueError, Exception):
        return default


# ─── Continuation Logic ────────────────────────────────────────────────────────

CONTINUATION_PROMPT = (
    "Please continue the generation. Output only the remaining part, "
    "starting from where you left off."
)

TOC_CONTINUATION_PROMPT = (
    "please continue the generation of table of contents , "
    "directly output the remaining part of the structure"
)


async def continue_if_truncated(
    provider: LLMProvider,
    original_prompt: str,
    partial_response: str,
    finish_reason: str,
    continuation_prompt: str = CONTINUATION_PROMPT,
    max_continuations: int = 5,
) -> str:
    """
    If finish_reason == "length", continue the LLM output via chat history.

    This is critical for long TOC generation — the LLM may hit its output
    limit mid-JSON. We send the partial response back as assistant history
    and ask for the remainder.

    Args:
        provider: The LLM provider to use.
        original_prompt: The original user prompt.
        partial_response: The truncated assistant response.
        finish_reason: "stop" or "length".
        continuation_prompt: What to ask the LLM to continue.
        max_continuations: Maximum continuation rounds (PageIndex uses 5).

    Returns:
        The complete response (partial + all continuations joined).
    """
    if finish_reason == "stop":
        return partial_response

    full_response = partial_response
    chat_history = [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": partial_response},
    ]

    for _ in range(max_continuations):
        continuation, next_finish = await provider.complete_with_finish_reason(
            continuation_prompt,
            temperature=0.0,
            chat_history=chat_history,
        )
        full_response += continuation
        if next_finish == "stop":
            break
        # Update history for next round
        chat_history.append({"role": "assistant", "content": continuation})
        chat_history.append({"role": "user", "content": continuation_prompt})

    return full_response


async def complete_with_continuation(
    provider: LLMProvider,
    prompt: str,
    continuation_prompt: str = CONTINUATION_PROMPT,
    max_continuations: int = 5,
    temperature: float = 0.0,
) -> str:
    """
    Call provider.complete_with_finish_reason() and auto-continue if truncated.

    Convenience wrapper around continue_if_truncated().
    """
    response, finish_reason = await provider.complete_with_finish_reason(
        prompt, temperature=temperature
    )
    return await continue_if_truncated(
        provider, prompt, response, finish_reason,
        continuation_prompt=continuation_prompt,
        max_continuations=max_continuations,
    )
