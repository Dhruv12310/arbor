"""
Markdown document extraction.

Extracts hierarchical structure from Markdown using regex (no LLM needed).
Markdown headers define the tree — # = level 1, ## = level 2, etc.

Mirrors PageIndex's page_index_md.py but adapted for Arbor's types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from arbor.core.types import TreeNode, DocumentTree
from arbor.utils.token_counter import count_tokens


@dataclass
class _MarkdownNode:
    """Intermediate node during Markdown parsing."""
    title: str
    level: int          # 1-6 (from # count)
    line_num: int       # 1-based line number of the header
    text: str = ""      # Text content under this header
    nodes: list[_MarkdownNode] = field(default_factory=list)


def extract_from_markdown(
    content: str,
    doc_name: str = "document",
    min_section_tokens: int = 0,
    add_node_ids: bool = True,
) -> DocumentTree:
    """
    Parse a Markdown document into a DocumentTree.

    Algorithm:
        1. Split into lines
        2. Detect code blocks (``` ... ```) — skip headers inside them
        3. Extract all headers (# through ######) with their line numbers
        4. Assign text content between headers to each section
        5. Build nested tree from header levels
        6. Optionally merge tiny sections (< min_section_tokens)

    Args:
        content: Full Markdown document text.
        doc_name: Name for the DocumentTree (e.g. filename without extension).
        min_section_tokens: Merge sections with fewer tokens than this into parent.
        add_node_ids: Assign zero-padded 4-digit node_id to each node.

    Returns:
        DocumentTree with nested TreeNode structure.
    """
    flat_nodes = _extract_flat_nodes(content)
    if not flat_nodes:
        # No headers — return single root node with all content
        return DocumentTree(
            doc_name=doc_name,
            structure=[
                TreeNode(
                    title=doc_name,
                    start_index=1,
                    end_index=1,
                    text=content,
                    node_id="0001" if add_node_ids else None,
                )
            ],
        )

    tree_nodes = _build_tree(flat_nodes, add_node_ids)

    if min_section_tokens > 0:
        tree_nodes = _thin_small_nodes(tree_nodes, min_section_tokens)

    return DocumentTree(doc_name=doc_name, structure=tree_nodes)


def _extract_flat_nodes(content: str) -> list[_MarkdownNode]:
    """
    Extract a flat list of sections from Markdown content.

    Respects code fences (``` blocks) — headers inside code blocks are ignored.
    """
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    code_fence_pattern = re.compile(r'^```')

    lines = content.split('\n')
    in_code_block = False
    sections: list[tuple[int, int, str]] = []  # (level, line_num, title)

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if code_fence_pattern.match(stripped):
            in_code_block = not in_code_block
            continue
        if not stripped or in_code_block:
            continue
        match = header_pattern.match(stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append((level, line_num, title))

    if not sections:
        return []

    # Build flat nodes with text content
    flat_nodes: list[_MarkdownNode] = []
    for i, (level, line_num, title) in enumerate(sections):
        # Text goes from just after this header's line to just before the next
        start_line = line_num  # exclusive (the header line itself)
        end_line = sections[i + 1][1] - 1 if i + 1 < len(sections) else len(lines)
        section_lines = lines[start_line:end_line]
        text = '\n'.join(section_lines).strip()

        flat_nodes.append(_MarkdownNode(
            title=title,
            level=level,
            line_num=line_num,
            text=text,
        ))

    return flat_nodes


def _build_tree(
    flat_nodes: list[_MarkdownNode],
    add_node_ids: bool,
) -> list[TreeNode]:
    """
    Convert flat list (with .level) to a nested TreeNode tree.

    Uses a stack to track the current parent chain.
    Node IDs are assigned in depth-first pre-order (same as PageIndex's write_node_id).
    """
    # Stack entries: (node, level)
    stack: list[tuple[TreeNode, int]] = []
    roots: list[TreeNode] = []
    counter = [1]  # mutable for nested function

    def make_tree_node(mn: _MarkdownNode) -> TreeNode:
        node_id = str(counter[0]).zfill(4) if add_node_ids else None
        counter[0] += 1
        return TreeNode(
            title=mn.title,
            start_index=mn.line_num,  # line_num as start (Markdown has no pages)
            end_index=mn.line_num,
            node_id=node_id,
            text=mn.text if mn.text else None,
            nodes=[],
        )

    for mn in flat_nodes:
        node = make_tree_node(mn)

        # Pop stack until we find a node at a lower level (parent candidate)
        while stack and stack[-1][1] >= mn.level:
            stack.pop()

        if not stack:
            roots.append(node)
        else:
            parent_node, _ = stack[-1]
            parent_node.nodes.append(node)

        stack.append((node, mn.level))

    return roots


def _thin_small_nodes(
    nodes: list[TreeNode],
    min_tokens: int,
) -> list[TreeNode]:
    """
    Merge nodes with fewer than min_tokens into their parent.

    Works recursively, bottom-up. Tiny leaf nodes get their text folded
    into the parent's text content.
    """
    result: list[TreeNode] = []
    for node in nodes:
        # Recurse first (process children)
        if node.nodes:
            node.nodes = _thin_small_nodes(node.nodes, min_tokens)

        node_tokens = count_tokens(node.text or "")
        if node_tokens < min_tokens and not node.nodes and result:
            # Merge into previous sibling's text (simplified thinning)
            prev = result[-1]
            combined = (prev.text or "") + "\n\n" + (node.text or "")
            prev.text = combined.strip()
            prev.end_index = node.end_index
        else:
            result.append(node)

    return result
