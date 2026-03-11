"""
Async concurrency utilities.

semaphore_gather() — Run coroutines with a concurrency cap.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")


async def semaphore_gather(
    coros: list[Coroutine[Any, Any, T]],
    semaphore: Optional[asyncio.Semaphore] = None,
) -> list[T]:
    """
    Run a list of coroutines concurrently, with optional semaphore limiting.

    If semaphore is None, runs all coroutines fully concurrently (asyncio.gather).
    If semaphore is provided, at most semaphore._value tasks run simultaneously.

    This is critical for Groq (rate-limited) and Ollama (single model instance).
    """
    if semaphore is None:
        return list(await asyncio.gather(*coros))

    async def _limited(coro: Coroutine) -> Any:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(_limited(c) for c in coros)))


def make_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Create a semaphore for limiting concurrent LLM calls."""
    return asyncio.Semaphore(max_concurrent)
