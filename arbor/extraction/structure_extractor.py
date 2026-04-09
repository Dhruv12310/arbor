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

    # Fill gaps: if a node has children but its start_page precedes the first child,
    # insert a chunk covering the gap so no pages are unreachable
    _fill_parent_gaps(nodes)

    # embedded_toc-specific fixes: fill sibling gaps, collapse over-granular
    # single-page siblings, and expand single-page leaves by ±1 page
    if strategy == "embedded_toc":
        _fill_sibling_gaps(nodes)
        _expand_single_page_leaves(nodes, total_pages)

    # Fill preamble gap: if first node doesn't start at page 0 or 1, insert a
    # "Cover / Preamble" node so pages before the first TOC entry are reachable.
    # Applies to multiline_toc and font_headings (embedded_toc handles gaps via
    # _fill_sibling_gaps; page_chunks covers everything by definition).
    if strategy in ("multiline_toc", "font_headings") and nodes:
        first_start = nodes[0].start_index
        if first_start > 1:
            nodes.insert(0, TreeNode(
                title="Cover / Preamble",
                start_index=0,
                end_index=first_start - 1,
            ))

    # Refine: any node spanning >15 pages with no children gets sub-section detection
    _refine_large_nodes(doc, nodes, max_span=15)

    doc.close()

    # Assign sequential node IDs across all nodes
    _assign_node_ids(nodes)

    tree = DocumentTree(doc_name=doc_name, structure=nodes, extraction_strategy=strategy)
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

    # Pre-processing: prune uninformative/redundant bookmark titles
    _prune_uninformative_nodes(nodes)

    # Quality gate: reject pathologically large/flat/noisy trees
    passed, reason = _score_embedded_toc_quality(nodes, total_pages)
    if not passed:
        print(f"[StructDirect] embedded_toc rejected ({reason}) — falling through")
        return None, "embedded_toc"

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


# ── embedded_toc helpers ─────────────────────────────────────────────────────

def _prune_uninformative_nodes(nodes: list[TreeNode]) -> None:
    """
    Remove or flatten redundant nodes before the quality gate runs.

    Two cases:
      1. A child whose title is a substring of its parent's title AND the child
         has its own children → promote grandchildren up (remove the redundant
         intermediate node).  Example: parent "Part I", child "Part I" (with
         sub-children) → move sub-children directly under parent.
      2. Nodes whose title is < 3 characters after stripping → remove entirely.

    Modifies nodes in-place.
    """
    i = 0
    while i < len(nodes):
        node = nodes[i]
        title_clean = node.title.strip()

        # Remove noise nodes (title too short)
        if len(title_clean) < 3:
            nodes.pop(i)
            continue

        # Check if any child is a redundant wrapper
        j = 0
        while j < len(node.nodes):
            child = node.nodes[j]
            child_title = child.title.strip()
            if (
                child_title.lower() in title_clean.lower()
                and child.nodes  # only collapse if child has its own children
            ):
                # Promote grandchildren in place of the child
                grandchildren = child.nodes
                node.nodes[j : j + 1] = grandchildren
                # Don't advance j — need to check the promoted children too
            else:
                j += 1

        # Recurse into remaining children
        _prune_uninformative_nodes(node.nodes)
        i += 1


def _score_embedded_toc_quality(
    nodes: list[TreeNode], total_pages: int
) -> tuple[bool, str]:
    """
    Reject embedded_toc trees that will confuse the navigator.

    Returns (True, "") if acceptable, or (False, reason) if the tree
    should be rejected and the cascade should try the next strategy.

    Checks (any failure → reject):
      A. Node count cap:       total nodes > max(60, total_pages * 2)
      B. Flat structure:       > 25 root nodes AND max depth == 1
      C. Single-page dominance: > 80% of all leaf nodes span 1 page AND total > 40
      D. Title duplication:    most common title appears > 5 times AND
                               unique-title ratio < 0.7
    """
    all_nodes = list(_iter_all_nodes_flat(nodes))
    total = len(all_nodes)
    if total == 0:
        return False, "no nodes"

    # A. Node count cap: navigator handles ~60 nodes well; allow up to 1 node
    #    per 3 pages for larger docs (a well-structured tree is much sparser).
    cap = max(60, total_pages // 3)
    if total > cap:
        return False, f"too many nodes ({total} > {cap})"

    # B. Flat structure check
    root_count = len(nodes)
    max_depth = _embedded_max_depth(nodes)
    if root_count > 25 and max_depth <= 1:
        return False, f"flat structure ({root_count} roots, depth={max_depth})"

    # C. Single-page dominance
    if total > 40:
        leaves = [n for n in all_nodes if not n.nodes]
        single_page_leaves = [n for n in leaves if n.start_index == n.end_index]
        if leaves and len(single_page_leaves) / len(leaves) > 0.80:
            return False, (
                f"single-page dominated ({len(single_page_leaves)}/{len(leaves)} leaves = 1 page)"
            )

    # D. Title duplication
    title_counts = Counter(n.title.strip().lower() for n in all_nodes)
    most_common_count = title_counts.most_common(1)[0][1]
    unique_ratio = len(title_counts) / total
    if most_common_count > 5 and unique_ratio < 0.7:
        return False, (
            f"duplicate titles (most common appears {most_common_count}×, "
            f"unique ratio={unique_ratio:.2f})"
        )

    return True, ""


def _iter_all_nodes_flat(nodes: list[TreeNode]):
    """Yield every node in the tree (depth-first)."""
    for node in nodes:
        yield node
        yield from _iter_all_nodes_flat(node.nodes)


def _embedded_max_depth(nodes: list[TreeNode], depth: int = 1) -> int:
    if not nodes:
        return depth - 1
    return max(_embedded_max_depth(n.nodes, depth + 1) for n in nodes)


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
    # Reject parentheticals like "(ii)", "(iv)", "(i.e. ...)"
    if t.startswith('('):
        return False
    # Reject pure integers
    if t.isdigit():
        return False
    # Reject float/decimal patterns: "14.8", "3.14", "1.2.3", "14.8.1"
    if re.match(r'^\d[\d\.]*$', t):
        return False
    # Reject titles where <30% of characters are letters and the title is short
    # catches: "8ρgd4", "§3.2", "—IV—", mixed symbol garbage from bad TOC parsing
    alpha_ratio = sum(c.isalpha() for c in t) / len(t)
    if alpha_ratio < 0.3 and len(t) < 20:
        return False
    # Reject very long lines — likely a sentence, not a title
    if len(t) > 100:
        return False
    # Reject all-caps lines shorter than 4 words that look like legal boilerplate
    words = t.split()
    if t.isupper() and len(words) > 8:
        return False
    # Reject URLs and path-like strings
    if re.search(r'https?://|www\.|\.com/|\.org/|\.gov/', t, re.I):
        return False
    if t.startswith('//'):
        return False
    # Reject footnote citations: line starting with a number immediately followed
    # by a word (e.g. "11Anthropic", "3See", "42Id") — common in academic papers
    if re.match(r'^\d+[A-Za-z]', t):
        return False
    # Reject date/preprint stamps like "A PREPRINT - MARCH 31, 2026"
    if re.search(r'\b(preprint|arxiv|submitted|revised)\b', t, re.I):
        return False
    # Reject obvious body-text phrases
    _BODY_PHRASES = (
        ' as well as ', ' and to ', ' in the ', ' of the ',
        ' for the ', ' with the ', ' by the ', ' to the ',
        ' shall be ', ' will be ', ' may be ', ' referred to as ',
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

    Produces a flat list — flat structure is better for navigation because the
    navigator can directly scan all section titles without needing to commit to
    a branch at each level.  (Hierarchy was tested and caused regression: Haiku
    and our fine-tuned model both navigate flat title lists better than deep
    trees for this strategy's typical output.)
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

    # Sort by page number, deduplicate, drop obvious noise.
    # Deduplicate on (title, page) NOT page alone — multiple short sections
    # (e.g., 10-K Items 3, 4, 5) legitimately share the same TOC page number
    # and must all be preserved.
    seen: set[tuple] = set()
    unique: list[tuple[str, int]] = []
    for title, pg in sorted(entries, key=lambda x: x[1]):
        key = (title.strip().lower(), pg)
        if key not in seen and len(title) > 3:
            unique.append((title, pg))
            seen.add(key)

    if len(unique) < min_sections:
        return None, "multiline_toc"

    # Quality gate: reject if >50% of titles are repeated (running headers/footers)
    # e.g. "A PREPRINT - MARCH 31, 2026" appearing on every page gets extracted
    # multiple times and dominates the entry list — useless for navigation.
    title_counts: dict[str, int] = {}
    for title, _ in unique:
        title_counts[title.strip().lower()] = title_counts.get(title.strip().lower(), 0) + 1
    repeated = sum(1 for cnt in title_counts.values() if cnt > 1)
    if len(unique) > 0 and repeated / len(unique) > 0.5:
        return None, "multiline_toc"

    # Separate top-level entries (ITEM X / PART X) from sub-entries
    # Top-level sections get end_page = start of next top-level section - 1
    # Sub-entries get end_page = start of next entry - 1 (original behavior)
    _TOP_LEVEL_RE = re.compile(r'^(ITEM\s+\d|PART\s+[IVX])', re.IGNORECASE)

    def _is_top_level(title: str) -> bool:
        return bool(_TOP_LEVEL_RE.match(title))

    top_level_pages = [pg for title, pg in unique if _is_top_level(title)]

    nodes: list[TreeNode] = []
    for i, (title, start_page) in enumerate(unique):
        if _is_top_level(title):
            # End at the start of the next top-level section - 1
            next_top = next((pg for pg in top_level_pages if pg > start_page), None)
            end_page = (next_top - 1) if next_top else total_pages
        else:
            # Sub-entry: end at next entry start - 1
            end_page = unique[i + 1][1] - 1 if i + 1 < len(unique) else total_pages
        end_page = max(start_page, end_page)
        nodes.append(TreeNode(title=title, start_index=start_page, end_index=end_page))

    return nodes, "multiline_toc"


# ── Strategy 3: Font-size heading detection ───────────────────────────────────

def _try_font_headings(
    doc, total_pages: int, min_sections: int
) -> tuple[list[TreeNode] | None, str]:
    """
    Detect headings by font properties and build a flat list of sections.

    A span is treated as a heading if it meets EITHER criterion:
      • Size criterion : font size >= body_size * 1.12  (the original check)
      • Bold criterion : font is bold (flags & 16) AND size >= body_size * 0.95
                        catches same-size bold headers common in earnings releases

    One heading per page (first qualifying span wins).  If the resulting node
    count exceeds max(60, total_pages // 3) the tree is too dense to navigate
    and we fall through to the next strategy.
    """
    # Collect all spans: (font_size, bold_flag, text, page_num)
    raw_spans: list[tuple[float, bool, str, int]] = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict", flags=0).get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text  = span.get("text", "").strip()
                    size  = span.get("size", 0)
                    flags = span.get("flags", 0)
                    bold  = bool(flags & 16)
                    if text and size > 0:
                        raw_spans.append((size, bold, text, page_num))

    if not raw_spans:
        return None, "font_headings"

    # Body text = most common rounded font size
    size_counts = Counter(round(s, 1) for s, _, _, _ in raw_spans)
    body_size   = size_counts.most_common(1)[0][0]

    size_threshold = body_size * 1.12   # ≥12% larger → heading by size
    bold_threshold = body_size * 0.95   # bold spans near body size also count

    # One heading per page: first qualifying span wins.
    # Strategy: try size-only first.  If that produces too few sections, fall
    # back to size-OR-bold so that flat-font documents (earnings releases) can
    # use same-size bold text as section markers.
    def _collect_headings(use_bold: bool) -> list[tuple[str, int]]:
        seen: set[int] = set()
        result: list[tuple[str, int]] = []
        for size, bold, text, page_num in raw_spans:
            if page_num in seen or len(text) <= 3:
                continue
            is_heading = (size >= size_threshold) or (use_bold and bold and size >= bold_threshold)
            if is_heading:
                result.append((text, page_num))
                seen.add(page_num)
        return result

    headings = _collect_headings(use_bold=False)
    if len(headings) < min_sections:
        headings = _collect_headings(use_bold=True)

    if len(headings) < min_sections:
        return None, "font_headings"

    # Quality gate: reject over-dense trees (same formula as embedded_toc)
    cap = max(60, total_pages // 3)
    if len(headings) > cap:
        return None, "font_headings"

    # Build flat TreeNodes with inferred end pages
    nodes: list[TreeNode] = []
    for i, (title, start_page) in enumerate(headings):
        end_page = headings[i + 1][1] - 1 if i + 1 < len(headings) else total_pages
        end_page = max(start_page, end_page)
        nodes.append(TreeNode(title=title, start_index=start_page, end_index=end_page))

    # Quality gate: reject if headings only cover a small fraction of the document
    # (e.g. headings only found in an appendix/exhibits section)
    covered_pages = set()
    for n in nodes:
        for p in range(n.start_index, n.end_index + 1):
            covered_pages.add(p)
    if len(covered_pages) < total_pages * 0.5:
        return None, "font_headings"

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


# ── Fill parent-to-first-child gaps ──────────────────────────────────────────

def _fill_parent_gaps(nodes: list[TreeNode]) -> None:
    """
    If a parent node starts before its first child, the pages between
    parent.start and first_child.start - 1 are unreachable via child navigation.
    Insert a chunk node covering that gap as the first child.
    Recurses into existing children.
    """
    for node in nodes:
        if not node.nodes:
            continue
        first_child_start = node.nodes[0].start_index
        if first_child_start > node.start_index + 1:
            # Gap exists — insert a chunk covering it
            gap_node = TreeNode(
                title=f"{node.title} — Cover / Intro (pages {node.start_index}–{first_child_start - 1})",
                start_index=node.start_index,
                end_index=first_child_start - 1,
            )
            node.nodes.insert(0, gap_node)
        # Recurse
        _fill_parent_gaps(node.nodes)


# ── Fill sibling gaps ────────────────────────────────────────────────────────

def _fill_sibling_gaps(nodes: list[TreeNode]) -> None:
    """
    Fill page gaps between consecutive sibling nodes at every level.

    Example: if sibling A ends at page 57 and sibling B starts at page 63,
    pages 58-62 are unreachable. Insert a 'Pages 58-62' chunk to cover them.

    Also recurses into all children so gaps are filled at every depth.
    """
    if not nodes:
        return

    # Collect gap insertions first (reverse order so indices stay valid)
    inserts: list[tuple[int, TreeNode]] = []
    for i in range(len(nodes) - 1):
        gap_start = nodes[i].end_index + 1
        gap_end   = nodes[i + 1].start_index - 1
        if gap_start <= gap_end:
            inserts.append((i + 1, TreeNode(
                title=f"Pages {gap_start}–{gap_end}",
                start_index=gap_start,
                end_index=gap_end,
            )))

    for idx, gap_node in reversed(inserts):
        nodes.insert(idx, gap_node)

    # Recurse into children (including any newly inserted gap nodes)
    for node in nodes:
        if node.nodes:
            _fill_sibling_gaps(node.nodes)


# ── Collapse single-page sibling groups ──────────────────────────────────────

def _collapse_single_page_siblings(nodes: list[TreeNode], max_consecutive: int = 3) -> None:
    """
    Merge runs of consecutive single-page leaf siblings into one grouped node.

    When more than `max_consecutive` consecutive leaf nodes each span exactly
    1 page, they are merged into a single parent node whose title is
    "<first_title> ... <last_title>" and whose page range covers all of them.
    The originals become children of the merged node so pages stay reachable.

    This reduces the visible node count at each level (navigator sees fewer
    choices) while preserving full coverage.
    """
    if not nodes:
        return

    i = 0
    while i < len(nodes):
        # Find start of a run of single-page leaves
        if nodes[i].nodes or nodes[i].start_index != nodes[i].end_index:
            _collapse_single_page_siblings(nodes[i].nodes, max_consecutive)
            i += 1
            continue

        # Count consecutive single-page leaves starting at i
        run_end = i
        while (
            run_end + 1 < len(nodes)
            and not nodes[run_end + 1].nodes
            and nodes[run_end + 1].start_index == nodes[run_end + 1].end_index
        ):
            run_end += 1

        run_length = run_end - i + 1
        if run_length > max_consecutive:
            run = nodes[i : run_end + 1]
            merged = TreeNode(
                title=f"{run[0].title} … {run[-1].title}",
                start_index=run[0].start_index,
                end_index=run[-1].end_index,
                nodes=run,
            )
            nodes[i : run_end + 1] = [merged]
            # Don't recurse into the merged node's children again — they are
            # already single-page leaves and don't need further collapsing
            i += 1
        else:
            i += 1


def _expand_single_page_leaves(nodes: list[TreeNode], total_pages: int) -> None:
    """
    Expand single-page leaf nodes by ±1 page to fix off-by-one bookmark misses.

    PDF bookmarks often point to the first page of a section, but the evidence
    may be on the page just before (e.g., the section header page).  Expanding
    by 1 page in each direction creates a small overlap that covers these cases.

    Expansion is clamped to [1, total_pages] and to the parent's page range
    (passed via the `parent_start` / `parent_end` parameters on recursion).
    """
    _expand_leaves_recursive(nodes, parent_start=1, parent_end=total_pages, total_pages=total_pages)


def _expand_leaves_recursive(
    nodes: list[TreeNode],
    parent_start: int,
    parent_end: int,
    total_pages: int,
) -> None:
    for idx, node in enumerate(nodes):
        if node.nodes:
            # Recurse — child range is constrained by this node's range
            _expand_leaves_recursive(node.nodes, node.start_index, node.end_index, total_pages)
        else:
            # Leaf node: only expand if it's a single page
            if node.start_index == node.end_index:
                # Determine safe bounds from neighbours
                prev_end   = nodes[idx - 1].end_index   if idx > 0              else parent_start - 1
                next_start = nodes[idx + 1].start_index if idx + 1 < len(nodes) else parent_end + 1

                new_start = max(node.start_index - 1, prev_end + 1,   1)
                new_end   = min(node.end_index   + 1, next_start - 1, total_pages)

                # Only expand if it doesn't invert the range
                if new_start <= node.start_index:
                    node.start_index = new_start
                if new_end >= node.end_index:
                    node.end_index = new_end


# ── Fix collapsed "Financial Statements" sections ─────────────────────────────

_FINANCIAL_STMT_RE = re.compile(
    r'financial\s+statement', re.IGNORECASE
)

def _fix_collapsed_financial_statements(
    nodes: list[TreeNode], total_pages: int
) -> None:
    """
    In some SEC 10-K filings (e.g. Activision Blizzard), the multiline_toc
    collapses 'Item 8. Financial Statements' to a single page because Item 9
    starts at the same TOC-printed page number.

    Detection: a top-level node whose title matches 'Financial Statement*'
    and whose page span is <= 2 pages, while total_pages > 50.

    Fix: extend Item 8's end_index to just before the next PART or Exhibit
    section, absorbing the collapsed Items 9/9A/9B/PART-III nodes as children.
    The absorbed nodes become children of Item 8 so navigation can still reach
    them individually.
    """
    if total_pages < 50:
        return  # Only applies to full 10-K filings

    for i, node in enumerate(nodes):
        span = node.end_index - node.start_index
        if not _FINANCIAL_STMT_RE.search(node.title):
            continue
        if span > 2:
            continue  # Already has reasonable extent — skip

        # Find how far to extend: absorb consecutive small Items (9, 9A, 9B)
        # until we hit a PART or Exhibit or a section with span > 5 pages
        absorb_until = i  # last index to absorb (inclusive)
        for j in range(i + 1, len(nodes)):
            sibling = nodes[j]
            sibling_span = sibling.end_index - sibling.start_index
            title_upper = sibling.title.upper()
            # Stop at PART X, Exhibit, or any large section
            if (re.match(r'PART\s+[IVX]', title_upper)
                    or 'EXHIBIT' in title_upper
                    or sibling_span > 5):
                break
            absorb_until = j

        if absorb_until <= i:
            continue  # Nothing to absorb

        # Extend Item 8's end_index to include absorbed siblings
        new_end = nodes[absorb_until].end_index
        absorbed = nodes[i + 1: absorb_until + 1]

        # Move absorbed nodes as children of Item 8
        node.end_index = new_end
        node.nodes = absorbed + node.nodes  # prepend so order is chronological

        # Remove absorbed nodes from top-level list
        del nodes[i + 1: absorb_until + 1]


# ── Recursive refinement of large nodes ──────────────────────────────────────

def _refine_large_nodes(
    doc, nodes: list[TreeNode], max_span: int = 15, chunk_size: int = 10
) -> None:
    """
    For any leaf node spanning more than max_span pages, try to find sub-sections:
      1. Font-heading detection within the page range (preferred — semantic titles)
      2. Fixed-size page chunking within the range (fallback — always works)
    Ensures no pages are orphaned: gaps before the first heading are filled
    with a chunk node. Modifies nodes in-place. Recurses into existing children.
    """
    for node in nodes:
        if node.nodes:
            _refine_large_nodes(doc, node.nodes, max_span, chunk_size)
            continue
        span = node.end_index - node.start_index
        if span <= max_span:
            continue
        # Try font headings first
        sub = _font_headings_in_range(doc, node.start_index, node.end_index)
        if len(sub) >= 2:
            # Fill any gap between parent start and first heading
            if sub[0].start_index > node.start_index:
                gap_end = sub[0].start_index - 1
                prefix = _chunk_range(node.start_index, gap_end, chunk_size)
                node.nodes = prefix + sub
            else:
                node.nodes = sub
        else:
            # Fallback: chunk within this node's page range
            node.nodes = _chunk_range(node.start_index, node.end_index, chunk_size)


def _font_headings_in_range(
    doc, start_page: int, end_page: int
) -> list[TreeNode]:
    """
    Detect sub-headings by font size within a specific page range (1-indexed).
    Uses a slightly stricter threshold (1.15x) than the top-level scan.
    """
    spans: list[tuple[float, str, int]] = []
    for page_num in range(start_page - 1, min(end_page, len(doc))):  # 0-indexed
        page = doc[page_num]
        blocks = page.get_text("dict", flags=0).get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text and size > 0:
                        spans.append((size, text, page_num + 1))  # back to 1-indexed

    if not spans:
        return []

    size_counts = Counter(round(s, 1) for s, _, _ in spans)
    body_size = size_counts.most_common(1)[0][0]
    heading_threshold = body_size * 1.15

    headings: list[tuple[str, int]] = []
    seen_pages: set[int] = set()
    for size, text, page_num in spans:
        if (size >= heading_threshold
                and len(text) > 3
                and _looks_like_section_title(text)
                and page_num not in seen_pages):
            headings.append((text, page_num))
            seen_pages.add(page_num)

    if len(headings) < 2:
        return []

    nodes: list[TreeNode] = []
    for i, (title, sp) in enumerate(headings):
        ep = headings[i + 1][1] - 1 if i + 1 < len(headings) else end_page
        ep = max(sp, ep)
        nodes.append(TreeNode(title=title, start_index=sp, end_index=ep))
    return nodes


def _chunk_range(start_page: int, end_page: int, chunk_size: int) -> list[TreeNode]:
    """Split a page range into fixed-size chunks (used as refinement fallback)."""
    nodes: list[TreeNode] = []
    for sp in range(start_page, end_page + 1, chunk_size):
        ep = min(sp + chunk_size - 1, end_page)
        nodes.append(TreeNode(
            title=f"Pages {sp}–{ep}",
            start_index=sp,
            end_index=ep,
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
