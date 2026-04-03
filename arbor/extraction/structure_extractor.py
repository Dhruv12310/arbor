"""
Structure extractor — zero-LLM-call PDF structure extraction.

Replaces LLM-based tree generation for documents with native structure.
Output is a DocumentTree identical in format to TreeGen output, so the
existing search_tree() / multihop navigation model works with zero changes.

Extraction cascade (tries in order, returns first success):
  1. Embedded PDF bookmarks  — PyMuPDF get_toc(), most PDFs have this
  2. Font-size heading detection — visual hierarchy from text block analysis
  3. Regex printed TOC — scans first 20 pages for printed table of contents
  4. Fixed-size page chunking — universal fallback, always succeeds

Usage:
    from arbor.extraction.structure_extractor import extract_structure

    tree = extract_structure("document.pdf")
    # tree is a DocumentTree — pass directly to search_tree()
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Union

try:
    import pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        _PYMUPDF_AVAILABLE = True
    except ImportError:
        _PYMUPDF_AVAILABLE = False

from arbor.core.types import DocumentTree, TreeNode
from arbor.extraction.pdf_extractor import get_pdf_name


# ── Public entry point ────────────────────────────────────────────────────────

def extract_structure(
    source: Union[str, BytesIO],
    chunk_size: int = 10,
    min_sections: int = 3,
) -> DocumentTree:
    """
    Extract document structure from a PDF without any LLM calls.

    Tries four strategies in order:
      1. Embedded bookmarks (get_toc)
      2. Font-size heading detection
      3. Regex printed TOC
      4. Fixed-size page chunking (fallback)

    Args:
        source:       PDF file path or BytesIO.
        chunk_size:   Pages per chunk for the fallback strategy.
        min_sections: Minimum sections required for strategies 1-3 to succeed.
                      If fewer are found, falls through to the next strategy.

    Returns:
        DocumentTree compatible with search_tree(multihop=True).
    """
    if not _PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF is required for structure extraction.\n"
            "Install with: pip install pymupdf"
        )

    doc_name = get_pdf_name(source)

    if isinstance(source, BytesIO):
        source.seek(0)
        doc = pymupdf.open(stream=source.read(), filetype="pdf")
    else:
        doc = pymupdf.open(source)

    total_pages = len(doc)

    # Strategy 1: embedded bookmarks
    nodes, strategy = _try_embedded_toc(doc, total_pages, min_sections)

    # Strategy 2: multi-line TOC (SEC filing format — title then page on next line)
    if nodes is None:
        nodes, strategy = _try_multiline_toc(doc, total_pages, min_sections)

    # Strategy 3: font-size heading detection
    if nodes is None:
        nodes, strategy = _try_font_headings(doc, total_pages, min_sections)

    # Strategy 4: regex single-line TOC ("Title ..... 47")
    if nodes is None:
        nodes, strategy = _try_regex_toc(doc, total_pages, min_sections)

    # Strategy 5: page chunking (always succeeds)
    if nodes is None:
        nodes, strategy = _chunk_by_pages(total_pages, chunk_size), "page_chunks"

    doc.close()

    # Assign sequential node IDs across all nodes
    _assign_node_ids(nodes)

    tree = DocumentTree(doc_name=doc_name, structure=nodes)
    print(f"[StructDirect] {doc_name}: {strategy}, "
          f"{_count_nodes(nodes)} sections, {total_pages} pages")
    return tree


# ── Strategy 1: Embedded PDF bookmarks ───────────────────────────────────────

def _try_embedded_toc(
    doc, total_pages: int, min_sections: int
) -> tuple[list[TreeNode] | None, str]:
    """
    Use PyMuPDF get_toc() to extract embedded PDF bookmarks.
    Returns physical page numbers — not TOC-printed numbers.
    This eliminates the page-offset bug that plagued LLM-generated trees.
    """
    toc = doc.get_toc(simple=True)   # [[level, title, page], ...]
    if not toc or len(toc) < min_sections:
        return None, "embedded_toc"

    # Filter out invalid entries (page <= 0 or title empty)
    toc = [(lvl, title.strip(), pg) for lvl, title, pg in toc
           if pg > 0 and title.strip()]

    if len(toc) < min_sections:
        return None, "embedded_toc"

    nodes = _toc_entries_to_tree(toc, total_pages)
    return nodes, "embedded_toc"


def _toc_entries_to_tree(
    toc: list[tuple[int, str, int]], total_pages: int
) -> list[TreeNode]:
    """
    Convert flat [(level, title, start_page), ...] list into nested TreeNode tree.
    End pages are inferred: end of section = start of next sibling - 1.
    """
    # Build flat list with end pages first
    entries = []
    for i, (level, title, start_page) in enumerate(toc):
        # Find end page: start of next entry at same or lower level
        end_page = total_pages
        for j in range(i + 1, len(toc)):
            next_level, _, next_start = toc[j]
            if next_level <= level:
                end_page = next_start - 1
                break
        end_page = max(start_page, end_page)  # never end before start
        entries.append((level, title, start_page, end_page))

    # Build nested tree via stack
    root_nodes: list[TreeNode] = []
    stack: list[tuple[int, TreeNode]] = []   # (level, node)

    for level, title, start_page, end_page in entries:
        node = TreeNode(title=title, start_index=start_page, end_index=end_page)

        # Pop stack until parent level found
        while stack and stack[-1][0] >= level:
            stack.pop()

        if not stack:
            root_nodes.append(node)
        else:
            stack[-1][1].nodes.append(node)

        stack.append((level, node))

    return root_nodes


# ── Title heuristic ──────────────────────────────────────────────────────────

# Patterns that strongly indicate a real section title
_TITLE_PATTERNS = re.compile(
    r'^(ITEM\s|PART\s|Note\s|Exhibit\s|Schedule\s|Appendix\s|Section\s|Chapter\s)',
    re.IGNORECASE,
)

def _looks_like_section_title(text: str) -> bool:
    """
    Return True if text looks like a section heading, not body text.

    Rejects:
    - Lines ending with a period (full sentences)
    - Lines starting with lowercase (sentence continuations)
    - Lines with obvious body-text connectors mid-sentence
    - Very long lines (>100 chars — likely a full sentence)
    """
    t = text.strip()
    if not t or len(t) < 4:
        return False
    # Reject full sentences (end with period, question mark, etc.)
    if t[-1] in '.?,:;':
        return False
    # Reject lines starting lowercase — continuation of prior sentence
    if t[0].islower():
        return False
    # Reject very long lines — likely a sentence, not a title
    if len(t) > 100:
        return False
    # Reject obvious body-text phrases
    _BODY_PHRASES = (
        ' as well as ', ' and to ', ' in the ', ' of the ',
        ' for the ', ' with the ', ' by the ', ' to the ',
    )
    tl = t.lower()
    if any(p in tl for p in _BODY_PHRASES) and not _TITLE_PATTERNS.match(t):
        return False
    return True


# ── Strategy 2: Multi-line TOC (SEC filing format) ───────────────────────────

def _try_multiline_toc(
    doc, total_pages: int, min_sections: int, scan_pages: int = 6
) -> tuple[list[TreeNode] | None, str]:
    """
    Handle TOCs where title and page number are on separate lines.

    SEC 10-K/10-Q filings use this format:
        ITEM 1
        Business
        4
        ITEM 1A
        Risk Factors
        10

    Also handles financial statement subsections:
        Consolidated Statement of Cash Flows...
        60
    """
    entries: list[tuple[str, int]] = []

    for page_idx in range(min(scan_pages, total_pages)):
        page = doc[page_idx]
        lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]

        for i, line in enumerate(lines):
            # Check if this line is a standalone page number
            try:
                pg = int(line)
                if not (1 <= pg <= total_pages):
                    continue
            except ValueError:
                continue

            # Build the title from the line(s) before the page number
            if i == 0:
                continue

            prev = lines[i - 1]

            # Skip if previous line is also a number or too short
            if prev.isdigit() or len(prev) < 3:
                continue

            # Skip body text masquerading as a title
            if not _looks_like_section_title(prev):
                continue

            # "ITEM 1A\nRisk Factors\n10" — combine item label + descriptive title
            if i >= 2 and re.match(r'^(ITEM\s+\d|PART\s+[IVX])', lines[i - 2], re.I):
                label = lines[i - 2].strip()
                title = f"{label} — {prev}"
            else:
                title = prev

            entries.append((title, pg))

    if not entries:
        return None, "multiline_toc"

    # Sort by page number, deduplicate, drop obvious noise
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for title, pg in sorted(entries, key=lambda x: x[1]):
        if pg not in seen and len(title) > 3:
            unique.append((title, pg))
            seen.add(pg)

    if len(unique) < min_sections:
        return None, "multiline_toc"

    # Build flat TreeNodes with inferred end pages
    nodes: list[TreeNode] = []
    for i, (title, start_page) in enumerate(unique):
        end_page = unique[i + 1][1] - 1 if i + 1 < len(unique) else total_pages
        end_page = max(start_page, end_page)
        nodes.append(TreeNode(title=title, start_index=start_page, end_index=end_page))

    return nodes, "multiline_toc"


# ── Strategy 3: Font-size heading detection ───────────────────────────────────

def _try_font_headings(
    doc, total_pages: int, min_sections: int
) -> tuple[list[TreeNode] | None, str]:
    """
    Detect headings by font size. Spans significantly larger than body text
    are treated as section headings. Builds a flat or shallow hierarchy.
    """
    # Collect all spans with their font sizes and page numbers
    spans: list[tuple[float, str, int]] = []  # (font_size, text, page_num)

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", flags=0).get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text and size > 0:
                        spans.append((size, text, page_num))

    if not spans:
        return None, "font_headings"

    # Determine body text font size (most common size)
    size_counts = Counter(round(s, 1) for s, _, _ in spans)
    body_size = size_counts.most_common(1)[0][0]

    # Headings are spans >= 20% larger than body text
    heading_threshold = body_size * 1.12

    # Collect headings with their page numbers
    headings: list[tuple[str, int]] = []  # (title, page_num)
    seen_pages: set[int] = set()

    for size, text, page_num in spans:
        if size >= heading_threshold and len(text) > 3:
            # Deduplicate: one heading per page
            if page_num not in seen_pages:
                headings.append((text, page_num))
                seen_pages.add(page_num)

    if len(headings) < min_sections:
        return None, "font_headings"

    # Build flat TreeNodes with inferred end pages
    nodes: list[TreeNode] = []
    for i, (title, start_page) in enumerate(headings):
        end_page = headings[i + 1][1] - 1 if i + 1 < len(headings) else total_pages
        end_page = max(start_page, end_page)
        nodes.append(TreeNode(title=title, start_index=start_page, end_index=end_page))

    return nodes, "font_headings"


# ── Strategy 3: Regex printed TOC ────────────────────────────────────────────

# Matches lines like: "Introduction ......... 5" or "1.2 Methods    12"
_TOC_LINE_RE = re.compile(
    r"^(.{3,60}?)\s*[.\s]{2,}\s*(\d{1,4})\s*$"
)
_TOC_LINE_SHORT = re.compile(
    r"^(.{3,60}?)\s{2,}(\d{1,4})\s*$"
)


def _try_regex_toc(
    doc, total_pages: int, min_sections: int
) -> tuple[list[TreeNode] | None, str]:
    """
    Scan the first 20 pages for a printed table of contents.
    Matches lines of the form: "Title ..... 47" or "Title    47".
    """
    toc_entries: list[tuple[str, int]] = []   # (title, page_number)
    scan_pages = min(20, total_pages)

    for page_num in range(scan_pages):
        page = doc[page_num]
        text = page.get_text()
        for line in text.splitlines():
            line = line.strip()
            m = _TOC_LINE_RE.match(line) or _TOC_LINE_SHORT.match(line)
            if m:
                title = m.group(1).strip(" .")
                try:
                    page_ref = int(m.group(2))
                    if 1 <= page_ref <= total_pages and len(title) > 3:
                        toc_entries.append((title, page_ref))
                except ValueError:
                    pass

    # Deduplicate and sort by page number
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for title, pg in sorted(toc_entries, key=lambda x: x[1]):
        if pg not in seen:
            unique.append((title, pg))
            seen.add(pg)

    if len(unique) < min_sections:
        return None, "regex_toc"

    # Build flat TreeNodes
    nodes: list[TreeNode] = []
    for i, (title, start_page) in enumerate(unique):
        end_page = unique[i + 1][1] - 1 if i + 1 < len(unique) else total_pages
        end_page = max(start_page, end_page)
        nodes.append(TreeNode(title=title, start_index=start_page, end_index=end_page))

    return nodes, "regex_toc"


# ── Strategy 4: Fixed-size page chunking (fallback) ───────────────────────────

def _chunk_by_pages(total_pages: int, chunk_size: int) -> list[TreeNode]:
    """
    Split document into fixed-size page chunks.
    Universal fallback — always produces a valid structure.
    """
    nodes: list[TreeNode] = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        nodes.append(TreeNode(
            title=f"Pages {start}–{end}",
            start_index=start,
            end_index=end,
        ))
    return nodes


# ── Node ID assignment ────────────────────────────────────────────────────────

def _assign_node_ids(nodes: list[TreeNode]) -> None:
    """
    Assign sequential 4-digit node IDs ("0001", "0002", ...) in DFS order.
    Modifies nodes in-place.
    """
    counter = [1]

    def _assign(node_list: list[TreeNode]) -> None:
        for node in node_list:
            node.node_id = f"{counter[0]:04d}"
            counter[0] += 1
            if node.nodes:
                _assign(node.nodes)

    _assign(nodes)


def _count_nodes(nodes: list[TreeNode]) -> int:
    total = 0
    for node in nodes:
        total += 1 + _count_nodes(node.nodes)
    return total


# ── Convenience: extract and save to JSON ────────────────────────────────────

def extract_and_save(
    source: Union[str, BytesIO],
    output_path: str,
    **kwargs,
) -> DocumentTree:
    """
    Extract structure from PDF and save to a JSON file.
    Useful for pre-caching structures before running queries.
    """
    import json
    from pathlib import Path

    tree = extract_structure(source, **kwargs)
    Path(output_path).write_text(
        json.dumps(tree.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[StructDirect] Saved to {output_path}")
    return tree
