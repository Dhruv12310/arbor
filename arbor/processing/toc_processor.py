"""
Three-mode TOC processing pipeline.

Mode 1: TOC_WITH_PAGES  — Document has a TOC with page numbers
Mode 2: TOC_NO_PAGES    — Document has a TOC but no page numbers
Mode 3: NO_TOC          — No TOC; generate structure from scratch

Each mode returns a flat list of {structure, title, physical_index} dicts.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from arbor.core.types import ArborConfig, PageContent, ProcessingMode
from arbor.extraction.text_utils import group_page_contents, parse_physical_index
from arbor.processing.json_utils import (
    extract_json,
    safe_extract_json,
    complete_with_continuation,
    TOC_CONTINUATION_PROMPT,
)
from arbor.processing.tree_builder import validate_and_clamp_indices
from arbor.processing.verification import verify_toc, fix_incorrect_entries, check_appear_at_start
from arbor.prompts.tree_generation import (
    toc_transformer_prompt,
    toc_index_extractor_prompt,
    add_page_number_prompt,
    generate_toc_init_prompt,
    generate_toc_continue_prompt,
)
from arbor.providers.base import LLMProvider
from arbor.utils.async_helpers import make_semaphore, semaphore_gather


async def meta_processor(
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    mode: ProcessingMode = ProcessingMode.NO_TOC,
    toc_content: Optional[str] = None,
    toc_page_list: Optional[list[int]] = None,
    start_index: int = 1,
) -> list[dict]:
    """
    Orchestrate TOC processing with accuracy-based fallback cascade.

    Fallback order:
        TOC_WITH_PAGES → TOC_NO_PAGES → NO_TOC (on accuracy < 0.6)

    If accuracy is between 0.6–1.0, attempts error correction (up to 3 retries).

    Args:
        pages: Document pages.
        provider: LLM provider.
        config: Arbor configuration.
        mode: Initial processing mode to try.
        toc_content: Raw TOC text (for TOC_WITH_PAGES / TOC_NO_PAGES modes).
        toc_page_list: Page numbers where TOC was found.
        start_index: 1-based offset for the first page.

    Returns:
        Flat list of {structure, title, physical_index} dicts, ready for post_processing().
    """
    semaphore = make_semaphore(config.max_concurrent_llm_calls)

    toc_items = await _run_mode(
        pages, provider, config, mode, toc_content, toc_page_list, start_index
    )
    toc_items = validate_and_clamp_indices(toc_items, len(pages), start_index)

    if not toc_items:
        return await _fallback(pages, provider, config, mode, start_index)

    accuracy, incorrect = await verify_toc(pages, toc_items, provider, start_index, semaphore)

    if accuracy == 1.0:
        return toc_items

    if accuracy > 0.6:
        # Good enough — fix the bad entries
        toc_items, _ = await fix_incorrect_entries(
            toc_items, pages, incorrect, provider, start_index,
            max_retries=3, semaphore=semaphore
        )
        return toc_items

    # accuracy <= 0.6 — fall back to next mode
    return await _fallback(pages, provider, config, mode, start_index, toc_content, toc_page_list)


async def _fallback(
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    current_mode: ProcessingMode,
    start_index: int,
    toc_content: Optional[str] = None,
    toc_page_list: Optional[list[int]] = None,
) -> list[dict]:
    if current_mode == ProcessingMode.TOC_WITH_PAGES:
        return await meta_processor(
            pages, provider, config,
            mode=ProcessingMode.TOC_NO_PAGES,
            toc_content=toc_content,
            toc_page_list=toc_page_list,
            start_index=start_index,
        )
    elif current_mode == ProcessingMode.TOC_NO_PAGES:
        return await meta_processor(
            pages, provider, config,
            mode=ProcessingMode.NO_TOC,
            start_index=start_index,
        )
    else:
        raise RuntimeError(
            "All three processing modes failed. "
            "The document may not have extractable structure."
        )


async def _run_mode(
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    mode: ProcessingMode,
    toc_content: Optional[str],
    toc_page_list: Optional[list[int]],
    start_index: int,
) -> list[dict]:
    if mode == ProcessingMode.TOC_WITH_PAGES:
        return await process_toc_with_pages(
            toc_content or "", toc_page_list or [], pages, provider, config, start_index
        )
    elif mode == ProcessingMode.TOC_NO_PAGES:
        return await process_toc_no_pages(
            toc_content or "", pages, provider, config, start_index
        )
    else:
        return await process_no_toc(pages, provider, config, start_index)


# ─── Mode 1: TOC with page numbers ────────────────────────────────────────────

async def process_toc_with_pages(
    toc_content: str,
    toc_page_list: list[int],
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    start_index: int = 1,
) -> list[dict]:
    """
    Map TOC items to physical pages using the TOC's own page numbers.

    Sends chunks of tagged pages to the LLM along with the TOC,
    asking it to fill in the physical_index for each entry.
    """
    groups = group_page_contents(
        pages, config.max_tokens_per_node, config.overlap_pages, start_index
    )

    # Start with the TOC structure (no physical indices yet)
    # We'll ask the LLM to fill them in chunk by chunk
    toc_items: list[dict] = []

    for group_text in groups:
        prompt = toc_index_extractor_prompt(toc_content, group_text)
        try:
            response = await complete_with_continuation(
                provider, prompt, TOC_CONTINUATION_PROMPT
            )
            data = safe_extract_json(response, [])
            if isinstance(data, list):
                toc_items = _merge_toc_items(toc_items, data)
        except Exception:
            continue

    return _normalize_items(toc_items)


def _merge_toc_items(existing: list[dict], new_items: list[dict]) -> list[dict]:
    """
    Merge new TOC items into existing, updating physical_index where missing.
    """
    if not existing:
        return new_items

    # Build lookup by title for fast update
    by_title = {item.get("title", ""): i for i, item in enumerate(existing)}
    result = [dict(item) for item in existing]

    for new_item in new_items:
        title = new_item.get("title", "")
        new_idx = parse_physical_index(new_item.get("physical_index"))
        if new_idx is None:
            continue
        if title in by_title:
            # Update existing entry
            result[by_title[title]]["physical_index"] = new_idx
        else:
            # New entry not in existing TOC
            result.append(new_item)

    return result


# ─── Mode 2: TOC without page numbers ─────────────────────────────────────────

async def process_toc_no_pages(
    toc_content: str,
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    start_index: int = 1,
) -> list[dict]:
    """
    Transform a page-number-free TOC to JSON, then scan the document to find
    where each section starts.
    """
    # Step 1: Transform raw TOC text to structured JSON
    prompt = toc_transformer_prompt(toc_content)
    response = await complete_with_continuation(
        provider, prompt, TOC_CONTINUATION_PROMPT
    )
    toc_data = safe_extract_json(response, {})

    if isinstance(toc_data, dict):
        toc_items = toc_data.get("table_of_contents", [])
    elif isinstance(toc_data, list):
        toc_items = toc_data
    else:
        return []

    # Step 2: Scan document pages to find physical_index for each section
    groups = group_page_contents(
        pages, config.max_tokens_per_node, config.overlap_pages, start_index
    )

    current_structure = list(toc_items)
    for group_text in groups:
        prompt = add_page_number_prompt(group_text, current_structure)
        try:
            response = await complete_with_continuation(provider, prompt)
            updated = safe_extract_json(response, current_structure)
            if isinstance(updated, list):
                current_structure = updated
        except Exception:
            continue

    return _normalize_items(current_structure)


# ─── Mode 3: No TOC — generate from scratch ───────────────────────────────────

async def process_no_toc(
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    start_index: int = 1,
) -> list[dict]:
    """
    Generate a tree structure by scanning the full document text.

    Processes the first chunk with generate_toc_init_prompt(),
    then continues chunk-by-chunk with generate_toc_continue_prompt().
    """
    groups = group_page_contents(
        pages, config.max_tokens_per_node, config.overlap_pages, start_index
    )

    if not groups:
        return []

    # First chunk: initialize
    prompt = generate_toc_init_prompt(groups[0])
    response = await complete_with_continuation(provider, prompt, TOC_CONTINUATION_PROMPT)
    toc_items = safe_extract_json(response, [])
    if not isinstance(toc_items, list):
        toc_items = []

    # Subsequent chunks: continue
    for group_text in groups[1:]:
        prompt = generate_toc_continue_prompt(toc_items, group_text)
        try:
            response = await complete_with_continuation(
                provider, prompt, TOC_CONTINUATION_PROMPT
            )
            additional = safe_extract_json(response, [])
            if isinstance(additional, list):
                toc_items.extend(additional)
        except Exception:
            continue

    return _normalize_items(toc_items)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_items(items: list[dict]) -> list[dict]:
    """
    Normalize a flat TOC list:
    - Ensure each item has 'structure', 'title', 'physical_index'
    - Convert physical_index to int
    - Remove items with None physical_index
    """
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = parse_physical_index(item.get("physical_index"))
        if idx is None:
            continue
        normalized = {
            "structure": str(item.get("structure", "")),
            "title": str(item.get("title", "")),
            "physical_index": idx,
        }
        result.append(normalized)
    return result
