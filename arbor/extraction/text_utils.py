"""
Text processing utilities for Arbor's document pipeline.

Key functions:
    tag_page()      — Wrap page text in <physical_index_N> markers
    tag_pages()     — Tag a list of PageContent objects
    group_pages()   — Chunk pages into LLM-context-sized groups with overlap
    parse_physical_index() — Extract integer from "<physical_index_5>" strings
"""

from __future__ import annotations

import math
import re
from typing import Union

from arbor.core.types import PageContent
from arbor.utils.token_counter import count_tokens


# ─── Page Tagging ─────────────────────────────────────────────────────────────

def tag_page(text: str, page_number: int) -> str:
    """
    Wrap a single page's text in physical_index markers.

    Format (exact same as PageIndex — both open and close tags are identical):
        <physical_index_5>
        [page text]
        <physical_index_5>

    This is what PageIndex prompts reference when the LLM needs to report
    which page a section starts on.
    """
    return f"<physical_index_{page_number}>\n{text}\n<physical_index_{page_number}>\n\n"


def tag_pages(pages: list[PageContent]) -> list[str]:
    """
    Tag a list of PageContent objects with physical_index markers.

    Returns one tagged string per page, preserving original page_number values.
    """
    return [tag_page(p.text, p.page_number) for p in pages]


def tag_pages_range(
    pages: list[PageContent],
    start_index: int = 1,
) -> list[str]:
    """
    Tag pages but use start_index as the offset for page numbering.

    Used when processing a sub-range of pages (e.g., for large-node subdivision):
    the physical_index in the tag matches the original document page numbers,
    not the local position in the slice.
    """
    return [tag_page(p.text, start_index + i) for i, p in enumerate(pages)]


# ─── Page Grouping (Chunking) ─────────────────────────────────────────────────

def group_pages(
    tagged_pages: list[str],
    token_lengths: list[int],
    max_tokens: int = 20000,
    overlap_pages: int = 1,
) -> list[str]:
    """
    Split tagged pages into LLM-context-sized groups with overlap.

    This is a direct port of PageIndex's page_list_to_group_text() algorithm.

    Algorithm:
        1. If total tokens <= max_tokens: return single group (all pages)
        2. Otherwise:
           a. Compute N = ceil(total / max_tokens)  — minimum number of groups
           b. Target per-group = ceil((total/N + max_tokens) / 2)
              This is the average of the "even split" and max_tokens,
              so groups are larger than the minimum split but never exceed max.
           c. Fill groups page-by-page; when adding a page would exceed target,
              close current group and start new one with overlap_pages of overlap.

    Args:
        tagged_pages: List of tagged page strings (from tag_pages()).
        token_lengths: Token count for each tagged page (same length as tagged_pages).
        max_tokens: Maximum tokens per group (context window budget).
        overlap_pages: Number of pages to repeat at the start of each new group.

    Returns:
        List of joined page-group strings, ready to send to an LLM.
    """
    if not tagged_pages:
        return []

    total_tokens = sum(token_lengths)

    # Fast path: everything fits in one group
    if total_tokens <= max_tokens:
        return ["".join(tagged_pages)]

    # Compute target group size (PageIndex formula)
    n_groups = math.ceil(total_tokens / max_tokens)
    target_per_group = math.ceil(((total_tokens / n_groups) + max_tokens) / 2)

    groups: list[str] = []
    current_pages: list[str] = []
    current_tokens: int = 0

    for i, (page_text, page_tokens) in enumerate(zip(tagged_pages, token_lengths)):
        if current_tokens + page_tokens > target_per_group and current_pages:
            # Close current group
            groups.append("".join(current_pages))
            # Start new group with overlap (go back overlap_pages)
            overlap_start = max(i - overlap_pages, 0)
            current_pages = list(tagged_pages[overlap_start:i])
            current_tokens = sum(token_lengths[overlap_start:i])

        current_pages.append(page_text)
        current_tokens += page_tokens

    # Flush remaining pages
    if current_pages:
        groups.append("".join(current_pages))

    return groups


def group_page_contents(
    pages: list[PageContent],
    max_tokens: int = 20000,
    overlap_pages: int = 1,
    start_index: int = 1,
) -> list[str]:
    """
    Convenience wrapper: tag pages then group them.

    Args:
        pages: List of PageContent objects.
        max_tokens: Max tokens per group.
        overlap_pages: Pages of overlap between groups.
        start_index: Page number for the first page (for sub-range processing).

    Returns:
        List of joined tagged-page strings, one per group.
    """
    tagged = [tag_page(p.text, start_index + i) for i, p in enumerate(pages)]
    token_lengths = [count_tokens(t) for t in tagged]
    return group_pages(tagged, token_lengths, max_tokens, overlap_pages)


# ─── Physical Index Parsing ────────────────────────────────────────────────────

def parse_physical_index(value: Union[str, int, None]) -> Union[int, None]:
    """
    Extract the integer page number from a physical_index value.

    Handles all formats the LLM might return:
        "<physical_index_5>"  → 5
        "physical_index_5"    → 5
        5                     → 5   (already an int)
        None                  → None

    This is necessary because LLMs sometimes return the raw tag string
    rather than extracting the integer.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    match = re.search(r'physical_index_(\d+)', str(value))
    if match:
        return int(match.group(1))
    # Last resort: try parsing as a plain integer string
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


# ─── Text Utilities ────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to single space; normalize line endings."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
