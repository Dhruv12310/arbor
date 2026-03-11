"""
TOC verification and error correction.

verify_toc()       — Check what % of TOC entries map to correct pages
fix_incorrect()    — Re-run mapping for wrong entries (up to 3 retries)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from arbor.core.types import ArborConfig, PageContent
from arbor.extraction.text_utils import tag_page, parse_physical_index
from arbor.processing.json_utils import extract_json, safe_extract_json
from arbor.prompts.tree_generation import (
    check_title_appearance_prompt,
    check_title_at_start_prompt,
    fix_toc_entry_prompt,
)
from arbor.providers.base import LLMProvider
from arbor.utils.async_helpers import semaphore_gather


async def verify_toc(
    pages: list[PageContent],
    toc_items: list[dict],
    provider: LLMProvider,
    start_index: int = 1,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> tuple[float, list[dict]]:
    """
    Verify that each TOC item's physical_index points to the right page.

    For each item, asks the LLM: "Does this section title appear on this page?"
    Computes accuracy = correct / total.

    Args:
        pages: List of PageContent objects.
        toc_items: Flat list of {title, physical_index, ...} dicts.
        provider: LLM provider for verification.
        start_index: 1-based offset for sub-range processing.
        semaphore: Optional concurrency limiter.

    Returns:
        (accuracy, incorrect_items)
        accuracy: float in [0.0, 1.0]
        incorrect_items: list of items that failed verification
    """
    if not toc_items:
        return 1.0, []

    tasks = [
        _verify_single(item, pages, provider, start_index)
        for item in toc_items
    ]
    results = await semaphore_gather(tasks, semaphore)

    correct = sum(1 for ok in results if ok)
    incorrect = [item for item, ok in zip(toc_items, results) if not ok]
    accuracy = correct / len(toc_items) if toc_items else 1.0

    return accuracy, incorrect


async def _verify_single(
    item: dict,
    pages: list[PageContent],
    provider: LLMProvider,
    start_index: int,
) -> bool:
    """Verify one TOC item. Returns True if correct."""
    title = item.get("title", "")
    physical_index = parse_physical_index(item.get("physical_index"))

    if physical_index is None:
        return False

    page_offset = physical_index - start_index
    if page_offset < 0 or page_offset >= len(pages):
        return False

    page_text = pages[page_offset].text
    prompt = check_title_appearance_prompt(title, page_text)

    try:
        response = await provider.complete_with_retry(prompt)
        data = safe_extract_json(response, {})
        answer = str(data.get("answer", "no")).lower().strip()
        return answer == "yes"
    except Exception:
        return False


async def check_appear_at_start(
    items: list[dict],
    pages: list[PageContent],
    provider: LLMProvider,
    start_index: int = 1,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> list[dict]:
    """
    For each TOC item, determine whether its section starts at the TOP of its page.

    Sets item["appear_start"] = "yes" | "no".
    Used in post_processing() to decide whether end_index = next.start - 1 or next.start.
    """
    if not items:
        return items

    tasks = [
        _check_single_appear_start(item, pages, provider, start_index)
        for item in items
    ]
    results = await semaphore_gather(tasks, semaphore)

    updated = []
    for item, appear_start in zip(items, results):
        item = dict(item)
        item["appear_start"] = appear_start
        updated.append(item)

    return updated


async def _check_single_appear_start(
    item: dict,
    pages: list[PageContent],
    provider: LLMProvider,
    start_index: int,
) -> str:
    """Returns "yes" if section starts at top of page, "no" otherwise."""
    title = item.get("title", "")
    physical_index = parse_physical_index(item.get("physical_index"))

    if physical_index is None:
        return "no"

    page_offset = physical_index - start_index
    if page_offset < 0 or page_offset >= len(pages):
        return "no"

    page_text = pages[page_offset].text
    prompt = check_title_at_start_prompt(title, page_text)

    try:
        response = await provider.complete_with_retry(prompt)
        data = safe_extract_json(response, {})
        start_begin = str(data.get("start_begin", "no")).lower().strip()
        return "yes" if start_begin == "yes" else "no"
    except Exception:
        return "no"


async def fix_incorrect_entries(
    toc_items: list[dict],
    pages: list[PageContent],
    incorrect_items: list[dict],
    provider: LLMProvider,
    start_index: int = 1,
    max_retries: int = 3,
    window_pages: int = 5,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Attempt to fix incorrectly-mapped TOC entries.

    For each incorrect item, shows the LLM a window of pages around the
    expected location and asks for the correct physical_index.

    Runs up to max_retries rounds, each time re-verifying and re-fixing.

    Returns:
        (updated_toc_items, remaining_incorrect)
    """
    # Build mutable copy with index lookup
    item_by_title: dict[str, int] = {
        item["title"]: i for i, item in enumerate(toc_items)
    }
    toc_items = [dict(item) for item in toc_items]

    remaining = list(incorrect_items)

    for attempt in range(max_retries):
        if not remaining:
            break

        fix_tasks = [
            _fix_single_entry(item, pages, provider, start_index, window_pages)
            for item in remaining
        ]
        fixed_indices = await semaphore_gather(fix_tasks, semaphore)

        next_remaining = []
        for item, new_idx in zip(remaining, fixed_indices):
            if new_idx is not None:
                toc_items[item_by_title[item["title"]]]["physical_index"] = new_idx
            else:
                next_remaining.append(item)

        # Re-verify the items we just fixed
        just_fixed = [
            toc_items[item_by_title[item["title"]]]
            for item, new_idx in zip(remaining, fixed_indices)
            if new_idx is not None
        ]
        if just_fixed:
            _, still_wrong = await verify_toc(pages, just_fixed, provider, start_index, semaphore)
            next_remaining.extend(still_wrong)

        remaining = next_remaining

    return toc_items, remaining


async def _fix_single_entry(
    item: dict,
    pages: list[PageContent],
    provider: LLMProvider,
    start_index: int,
    window_pages: int,
) -> Optional[int]:
    """Try to find the correct physical_index for a wrong entry. Returns int or None."""
    title = item.get("title", "")
    current_idx = parse_physical_index(item.get("physical_index")) or start_index

    # Build a window of pages around the current (wrong) location
    window_start = max(start_index, current_idx - window_pages)
    window_end = min(start_index + len(pages) - 1, current_idx + window_pages)

    window_text = ""
    for page_num in range(window_start, window_end + 1):
        offset = page_num - start_index
        if 0 <= offset < len(pages):
            window_text += tag_page(pages[offset].text, page_num)

    prompt = fix_toc_entry_prompt(title, window_text)

    try:
        response = await provider.complete_with_retry(prompt)
        data = safe_extract_json(response, {})
        raw_idx = data.get("physical_index")
        return parse_physical_index(raw_idx)
    except Exception:
        return None
