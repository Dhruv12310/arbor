"""
TOC detection: scan the first N pages of a document to find a table of contents.
"""

from __future__ import annotations

from arbor.core.types import ArborConfig, PageContent
from arbor.processing.json_utils import extract_json, safe_extract_json
from arbor.prompts.tree_generation import (
    toc_detector_prompt,
    extract_toc_prompt,
    detect_page_index_prompt,
    check_toc_complete_prompt,
)
from arbor.providers.base import LLMProvider


async def check_toc(
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
) -> dict:
    """
    Scan the first toc_check_pages pages for a table of contents.

    Returns a dict with:
        toc_content: str | None        — raw TOC text if found
        toc_page_list: list[int]        — page indices that contain TOC
        page_index_given_in_toc: str   — "yes" | "no"
    """
    check_pages = pages[:config.toc_check_pages]

    # Scan each page individually for TOC detection
    toc_pages: list[int] = []
    toc_text_parts: list[str] = []

    for page in check_pages:
        prompt = toc_detector_prompt(page.text)
        try:
            response = await provider.complete_with_retry(prompt)
            data = safe_extract_json(response, {})
            detected = str(data.get("toc_detected", "no")).lower().strip()
            if detected == "yes":
                toc_pages.append(page.page_number)
                toc_text_parts.append(page.text)
        except Exception:
            continue

    if not toc_pages:
        return {
            "toc_content": None,
            "toc_page_list": [],
            "page_index_given_in_toc": "no",
        }

    # Extract clean TOC content from the detected pages
    raw_toc_text = "\n".join(toc_text_parts)
    extract_prompt = extract_toc_prompt(raw_toc_text)
    try:
        toc_content = await provider.complete_with_retry(extract_prompt)
    except Exception:
        toc_content = raw_toc_text

    # Check whether the TOC contains page numbers
    page_index_given = "no"
    if toc_content:
        pi_prompt = detect_page_index_prompt(toc_content)
        try:
            pi_response = await provider.complete_with_retry(pi_prompt)
            pi_data = safe_extract_json(pi_response, {})
            page_index_given = str(
                pi_data.get("page_index_given_in_toc", "no")
            ).lower().strip()
        except Exception:
            pass

    return {
        "toc_content": toc_content,
        "toc_page_list": toc_pages,
        "page_index_given_in_toc": page_index_given,
    }
