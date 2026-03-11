"""
Tree utility functions.

write_node_ids()      — Assign depth-first pre-order zero-padded IDs
create_node_mapping() — node_id → TreeNode lookup dict
add_node_text()       — Inject page text into tree nodes
remove_node_text()    — Strip text fields (for search prompts)
print_tree()          — Pretty-print tree to stdout
remove_fields()       — Strip arbitrary fields from tree dicts (for prompts)
"""

from __future__ import annotations

from typing import Optional

from arbor.core.types import PageContent, TreeNode


# ─── Node ID Assignment ───────────────────────────────────────────────────────

def write_node_ids(nodes: list[TreeNode], counter: Optional[list[int]] = None) -> None:
    """
    Assign depth-first pre-order zero-padded 4-digit node IDs.

    "0001", "0002", ..., "0999", "1000", ...

    Modifies nodes in-place. Counter is a mutable list[int] so it can be
    shared across recursive calls.
    """
    if counter is None:
        counter = [1]
    for node in nodes:
        node.node_id = str(counter[0]).zfill(4)
        counter[0] += 1
        if node.nodes:
            write_node_ids(node.nodes, counter)


# ─── Node Mapping ─────────────────────────────────────────────────────────────

def create_node_mapping(nodes: list[TreeNode]) -> dict[str, TreeNode]:
    """
    Build a flat node_id → TreeNode lookup dict from a tree.

    Traverses depth-first. Only includes nodes that have a node_id set.
    """
    mapping: dict[str, TreeNode] = {}
    _collect_nodes(nodes, mapping)
    return mapping


def _collect_nodes(nodes: list[TreeNode], mapping: dict[str, TreeNode]) -> None:
    for node in nodes:
        if node.node_id:
            mapping[node.node_id] = node
        if node.nodes:
            _collect_nodes(node.nodes, mapping)


# ─── Text Injection / Removal ─────────────────────────────────────────────────

def add_node_text(nodes: list[TreeNode], pages: list[PageContent]) -> None:
    """
    Inject raw page text into each node based on its page range.

    node.text = joined text of pages[start_index-1 : end_index]

    Modifies nodes in-place (recursive).
    """
    for node in nodes:
        start = node.start_index - 1  # Convert to 0-based index
        end = node.end_index          # Inclusive → exclusive slice
        node_pages = pages[start:end]
        node.text = "\n\n".join(p.text for p in node_pages).strip() or None
        if node.nodes:
            add_node_text(node.nodes, pages)


def remove_node_text(nodes: list[TreeNode]) -> None:
    """
    Strip the text field from all nodes (in-place, recursive).

    Used before serializing the tree for search prompts — the text field
    is large and we only want titles/summaries/node_ids for the LLM.
    """
    for node in nodes:
        node.text = None
        if node.nodes:
            remove_node_text(node.nodes)


# ─── Pretty Printing ──────────────────────────────────────────────────────────

def print_tree(
    nodes: list[TreeNode],
    indent: int = 0,
    exclude_fields: Optional[list[str]] = None,
) -> None:
    """
    Pretty-print a tree to stdout.

    Example:
        [0001] Chapter 1 (pp. 1-15)
          [0002] 1.1 Introduction (pp. 1-3)
          [0003] 1.2 Background (pp. 4-8)
        [0004] Chapter 2 (pp. 16-30)
    """
    prefix = "  " * indent
    for node in nodes:
        id_part = f"[{node.node_id}] " if node.node_id else ""
        page_part = f" (pp. {node.start_index}-{node.end_index})"
        print(f"{prefix}{id_part}{node.title}{page_part}")
        if node.nodes:
            print_tree(node.nodes, indent + 1, exclude_fields)


def tree_to_search_dict(nodes: list[TreeNode]) -> list[dict]:
    """
    Convert tree to a list of dicts suitable for search prompts.

    Includes: node_id, title, summary, nodes (recursive).
    Excludes: text (too large for prompts).
    """
    result = []
    for node in nodes:
        d: dict = {"title": node.title}
        if node.node_id:
            d["node_id"] = node.node_id
        if node.summary:
            d["summary"] = node.summary
        if node.nodes:
            d["nodes"] = tree_to_search_dict(node.nodes)
        result.append(d)
    return result


def remove_fields(obj: dict | list, fields: list[str]) -> dict | list:
    """
    Recursively remove specified fields from a nested dict/list structure.

    Used to strip 'text' from tree dicts before sending to search LLM
    (same as PageIndex's utils.remove_fields()).
    """
    if isinstance(obj, list):
        return [remove_fields(item, fields) for item in obj]
    if isinstance(obj, dict):
        return {
            k: remove_fields(v, fields)
            for k, v in obj.items()
            if k not in fields
        }
    return obj


def count_nodes(nodes: list[TreeNode]) -> int:
    """Count total number of nodes in the tree (including all descendants)."""
    total = len(nodes)
    for node in nodes:
        total += count_nodes(node.nodes)
    return total
