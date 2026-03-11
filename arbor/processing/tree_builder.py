"""
Convert a flat list of TOC items into a nested TreeNode tree.

This is a direct port of PageIndex's list_to_tree() and post_processing().
"""

from __future__ import annotations

from arbor.core.types import FlatTOCItem, TreeNode
from arbor.extraction.text_utils import parse_physical_index


def post_processing(
    items: list[dict],
    total_pages: int,
    start_index: int = 1,
) -> list[TreeNode]:
    """
    Convert a flat list of TOC items to a nested TreeNode tree.

    Steps:
    1. Convert physical_index strings to integers
    2. Compute end_index for each item:
       - If next item appears at start of its page (appear_start="yes"):
         end_index = next.start_index - 1
       - Otherwise:
         end_index = next.start_index  (they share the boundary page)
    3. Last item: end_index = total_pages
    4. Call list_to_tree() to build hierarchy from dot-notation structure

    Args:
        items: List of dicts with keys: structure, title, physical_index, appear_start
        total_pages: Total number of pages in the document (for last item's end_index)
        start_index: 1-based offset (for sub-range processing)

    Returns:
        List of root TreeNode objects.
    """
    # Resolve physical_index to int start_index
    valid_items = []
    for item in items:
        idx = parse_physical_index(item.get("physical_index"))
        if idx is None:
            continue
        item = dict(item)  # don't mutate caller's data
        item["start_index"] = idx
        valid_items.append(item)

    if not valid_items:
        return []

    # Compute end_index for each item
    for i, item in enumerate(valid_items):
        if i < len(valid_items) - 1:
            next_item = valid_items[i + 1]
            next_start = next_item["start_index"]
            appear_start = next_item.get("appear_start", "no")
            if appear_start == "yes":
                item["end_index"] = next_start - 1
            else:
                item["end_index"] = next_start
        else:
            item["end_index"] = total_pages

        # Clamp: end_index must be >= start_index
        item["end_index"] = max(item["end_index"], item["start_index"])

    tree = list_to_tree(valid_items)
    if tree:
        return tree

    # Fallback: return flat list as top-level nodes
    result = []
    for item in valid_items:
        node = TreeNode(
            title=item.get("title", ""),
            start_index=item["start_index"],
            end_index=item["end_index"],
        )
        result.append(node)
    return result


def list_to_tree(items: list[dict]) -> list[TreeNode]:
    """
    Convert a flat list of {structure, title, start_index, end_index} items
    to a nested tree using dot-notation structure keys.

    structure "1"     → root
    structure "1.1"   → child of "1"
    structure "1.1.2" → child of "1.1"

    Handles orphaned nodes (parent doesn't exist) by promoting to root.
    """
    def get_parent_key(structure: str) -> str | None:
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    node_map: dict[str, TreeNode] = {}
    root_nodes: list[TreeNode] = []

    for item in items:
        structure = str(item.get("structure", ""))
        node = TreeNode(
            title=item.get("title", ""),
            start_index=item.get("start_index", 1),
            end_index=item.get("end_index", 1),
        )
        node_map[structure] = node

        parent_key = get_parent_key(structure)
        if parent_key and parent_key in node_map:
            node_map[parent_key].nodes.append(node)
        else:
            root_nodes.append(node)

    # Clean up empty nodes lists
    def clean(node: TreeNode) -> TreeNode:
        if not node.nodes:
            node.nodes = []
        else:
            node.nodes = [clean(n) for n in node.nodes]
        return node

    return [clean(n) for n in root_nodes]


def add_preface_if_needed(items: list[dict], start_index: int = 1) -> list[dict]:
    """
    If the document has content before the first detected section, add a
    "Preface" node covering pages 1 through first_section.start - 1.

    This matches PageIndex's add_preface_if_needed() behavior.
    """
    if not items:
        return items

    first_idx = parse_physical_index(items[0].get("physical_index"))
    if first_idx is None:
        first_idx = items[0].get("start_index", start_index)

    if first_idx <= start_index:
        return items

    preface = {
        "structure": "0",
        "title": "Preface",
        "physical_index": start_index,
        "start_index": start_index,
        "appear_start": "yes",
    }
    return [preface] + items


def validate_and_clamp_indices(
    items: list[dict],
    total_pages: int,
    start_index: int = 1,
) -> list[dict]:
    """
    Remove items with invalid/out-of-range physical indices.
    Clamp valid indices to [start_index, start_index + total_pages - 1].
    """
    valid = []
    max_page = start_index + total_pages - 1
    for item in items:
        idx = parse_physical_index(item.get("physical_index"))
        if idx is None:
            continue
        if idx < start_index or idx > max_page:
            continue
        item = dict(item)
        item["physical_index"] = idx
        valid.append(item)
    return valid
