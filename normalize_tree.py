"""
Arbor TreeGen Output Normalizer

Takes raw TreeGen-v2 output (which may use inconsistent field names,
miss page ranges, have over-deep nesting, etc.) and normalizes it into
the exact canonical schema that TreeSearch expects:

    {
        "title": str,
        "node_id": str (zero-padded 4-digit),
        "start_index": int (1-based page number),
        "end_index": int (1-based page number, inclusive),
        "nodes": [... recursive children ...]
    }

Usage:
    from normalize_tree import normalize_tree
    clean_tree = normalize_tree(raw_tree_output, chunk_start_page=1, chunk_end_page=4)
"""

import json
import re
from typing import Any

# --- Field aliases the model invents ---
CHILDREN_ALIASES = {"children", "sections", "sub_nodes", "subnodes", "subsections",
                    "items", "content_nodes", "elements", "parts", "sub_sections"}
TITLE_ALIASES = {"title", "name", "heading", "section_title", "label", "section_name", "header"}
START_ALIASES = {"start_index", "start_page", "page_start", "begin_page", "from_page", "page"}
END_ALIASES = {"end_index", "end_page", "page_end", "to_page"}
SUMMARY_ALIASES = {"summary", "description", "content", "text", "overview", "detail", "details"}

# Fields to strip (not part of canonical schema, create noise for TreeSearch)
STRIP_FIELDS = {"type", "value", "metrics", "data", "figures", "tables",
                "references", "footnotes", "level", "depth", "parent_id",
                "content", "text", "summary", "description", "overview",
                "detail", "details", "key_points", "highlights"}


def _find_field(node: dict, aliases: set) -> Any:
    """Find first matching field from a set of aliases."""
    for alias in aliases:
        if alias in node:
            return node[alias]
    return None


def _find_children(node: dict) -> list:
    """Extract children from any alias the model might use."""
    # Check canonical first
    if "nodes" in node and isinstance(node["nodes"], list):
        return node["nodes"]
    # Check aliases
    for alias in CHILDREN_ALIASES:
        if alias in node and isinstance(node[alias], list):
            return node[alias]
    return []


def _extract_title(node: dict) -> str:
    """Get a title from whatever field the model used."""
    val = _find_field(node, TITLE_ALIASES)
    if val and isinstance(val, str):
        return val.strip()
    # If no title field, try to synthesize from other fields
    if "node_id" in node:
        return f"Section {node['node_id']}"
    return "Untitled Section"


def _extract_page_range(node: dict, parent_start: int, parent_end: int) -> tuple[int, int]:
    """Extract or infer page range from node."""
    start = _find_field(node, START_ALIASES)
    end = _find_field(node, END_ALIASES)

    # Convert to int, handling string page numbers
    if start is not None:
        try:
            start = int(start)
        except (ValueError, TypeError):
            start = None
    if end is not None:
        try:
            end = int(end)
        except (ValueError, TypeError):
            end = None

    # If we have start but no end, use parent end
    if start is not None and end is None:
        end = parent_end
    # If we have end but no start, use parent start
    if start is None and end is not None:
        start = parent_start
    # If we have neither, inherit parent range
    if start is None and end is None:
        start = parent_start
        end = parent_end

    # Clamp to parent range
    start = max(start, parent_start)
    end = min(end, parent_end)
    # Ensure start <= end
    if start > end:
        start = end

    return start, end


def _flatten_overdeep(nodes: list[dict], max_depth: int = 5, current_depth: int = 0) -> list[dict]:
    """
    Flatten nodes that exceed max_depth by promoting their children.
    Prevents 11-level deep trees for trivial content.
    """
    result = []
    for node in nodes:
        children = _find_children(node)
        if current_depth >= max_depth and children:
            # This node is too deep and has children — absorb children's titles
            # into this node and make it a leaf
            child_titles = [_extract_title(c) for c in children]
            existing_title = _extract_title(node)
            if child_titles:
                node_copy = dict(node)
                # Keep the parent title, drop children (they become implicit)
                for alias in CHILDREN_ALIASES | {"nodes"}:
                    node_copy.pop(alias, None)
                result.append(node_copy)
            else:
                result.append(node)
        else:
            node_copy = dict(node)
            if children:
                flattened_children = _flatten_overdeep(children, max_depth, current_depth + 1)
                # Replace whatever alias was used with canonical "nodes"
                for alias in CHILDREN_ALIASES | {"nodes"}:
                    node_copy.pop(alias, None)
                node_copy["nodes"] = flattened_children
            result.append(node_copy)
    return result


def _is_hallucinated_content(node: dict) -> bool:
    """
    Detect obvious hallucinated placeholder content.
    E.g., "Revenue: $X billion", "Amount: $XX million"
    """
    title = _extract_title(node)
    # Check for placeholder patterns
    placeholder_patterns = [
        r'\$X+\b',           # $X, $XX, $XXX
        r'\$\d+\.?\d*\s*(billion|million|thousand)',  # fabricated numbers
        r'TBD',
        r'N/A',
        r'\[.*?\]',          # [placeholder]
        r'Lorem ipsum',
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return True

    # Check summary/content fields for hallucination markers
    for alias in SUMMARY_ALIASES:
        val = node.get(alias, "")
        if isinstance(val, str):
            for pattern in placeholder_patterns:
                if re.search(pattern, val, re.IGNORECASE):
                    return True
    return False


def _normalize_node(node: dict, parent_start: int, parent_end: int,
                    counter: list[int], max_depth: int = 5, depth: int = 0) -> dict | None:
    """Recursively normalize a single node into canonical schema."""
    if not isinstance(node, dict):
        return None

    # Skip hallucinated content
    if _is_hallucinated_content(node):
        return None

    title = _extract_title(node)
    start, end = _extract_page_range(node, parent_start, parent_end)

    # Assign sequential node_id
    counter[0] += 1
    node_id = str(counter[0]).zfill(4)

    # Process children
    children = _find_children(node)
    normalized_children = []

    if children and depth < max_depth:
        # Distribute page ranges across children if they don't have their own
        for i, child in enumerate(children):
            normalized = _normalize_node(child, start, end, counter, max_depth, depth + 1)
            if normalized is not None:
                normalized_children.append(normalized)

        # Fix child page ranges to be non-overlapping and cover parent range
        if normalized_children:
            _fix_child_ranges(normalized_children, start, end)

    # Build canonical node
    result = {
        "title": title,
        "node_id": node_id,
        "start_index": start,
        "end_index": end,
    }
    if normalized_children:
        result["nodes"] = normalized_children

    return result


def _fix_child_ranges(children: list[dict], parent_start: int, parent_end: int):
    """
    Ensure children's page ranges:
    1. Are non-overlapping
    2. Are ordered by start_index
    3. Cover the parent's range reasonably
    """
    if not children:
        return

    # Sort by start_index
    children.sort(key=lambda c: c["start_index"])

    # If all children have the same range (inherited from parent),
    # distribute pages evenly
    all_same = all(
        c["start_index"] == children[0]["start_index"] and
        c["end_index"] == children[0]["end_index"]
        for c in children
    )
    if all_same and len(children) > 1:
        total_pages = parent_end - parent_start + 1
        pages_per_child = max(1, total_pages // len(children))
        for i, child in enumerate(children):
            child["start_index"] = parent_start + i * pages_per_child
            if i < len(children) - 1:
                child["end_index"] = parent_start + (i + 1) * pages_per_child - 1
            else:
                child["end_index"] = parent_end

    # Ensure non-overlapping: each child's end <= next child's start - 1
    for i in range(len(children) - 1):
        if children[i]["end_index"] >= children[i + 1]["start_index"]:
            children[i]["end_index"] = children[i + 1]["start_index"] - 1
            if children[i]["end_index"] < children[i]["start_index"]:
                children[i]["end_index"] = children[i]["start_index"]

    # Clamp all to parent range
    for child in children:
        child["start_index"] = max(child["start_index"], parent_start)
        child["end_index"] = min(child["end_index"], parent_end)
        if child["start_index"] > child["end_index"]:
            child["start_index"] = child["end_index"]


def normalize_tree(raw_output: dict | list | str,
                   chunk_start_page: int = 1,
                   chunk_end_page: int = 999,
                   max_depth: int = 5) -> list[dict]:
    """
    Main entry point: normalize raw TreeGen output into canonical schema.

    Args:
        raw_output: The raw model output (dict, list, or JSON string)
        chunk_start_page: First page number of the chunk
        chunk_end_page: Last page number of the chunk
        max_depth: Maximum nesting depth (default 5, prevents 11-level trees)

    Returns:
        List of normalized root nodes in canonical schema
    """
    # Parse if string
    if isinstance(raw_output, str):
        try:
            raw_output = json.loads(raw_output)
        except json.JSONDecodeError:
            return []

    # Handle different root shapes
    nodes_to_normalize = []

    if isinstance(raw_output, list):
        nodes_to_normalize = raw_output
    elif isinstance(raw_output, dict):
        # Could be {"structure": [...]} or {"nodes": [...]} or a single node
        if "structure" in raw_output and isinstance(raw_output["structure"], list):
            nodes_to_normalize = raw_output["structure"]
        elif "nodes" in raw_output and isinstance(raw_output["nodes"], list):
            # Model wrapped everything in a root object
            nodes_to_normalize = raw_output["nodes"]
        else:
            # Check aliases
            for alias in CHILDREN_ALIASES:
                if alias in raw_output and isinstance(raw_output[alias], list):
                    nodes_to_normalize = raw_output[alias]
                    break
            if not nodes_to_normalize:
                # Single node — treat as root
                nodes_to_normalize = [raw_output]

    # First pass: flatten over-deep nesting
    nodes_to_normalize = _flatten_overdeep(nodes_to_normalize, max_depth)

    # Second pass: normalize each root node
    counter = [0]  # mutable counter for sequential node_ids
    normalized = []
    for node in nodes_to_normalize:
        result = _normalize_node(node, chunk_start_page, chunk_end_page,
                                 counter, max_depth, depth=0)
        if result is not None:
            normalized.append(result)

    return normalized


def merge_chunk_trees(chunk_trees: list[tuple[int, int, list[dict]]],
                      doc_title: str = "Document") -> dict:
    """
    Merge normalized trees from multiple chunks into a single document tree.

    Args:
        chunk_trees: List of (start_page, end_page, normalized_nodes) tuples
        doc_title: Title for the root document node

    Returns:
        Complete document tree in canonical schema
    """
    # Sort chunks by start page
    chunk_trees.sort(key=lambda x: x[0])

    # Collect all root nodes from all chunks
    all_roots = []
    for start_page, end_page, nodes in chunk_trees:
        all_roots.extend(nodes)

    # Reassign node_ids sequentially across the whole tree
    counter = [0]

    def reassign_ids(node):
        counter[0] += 1
        node["node_id"] = str(counter[0]).zfill(4)
        for child in node.get("nodes", []):
            reassign_ids(child)

    for root in all_roots:
        reassign_ids(root)

    # Build final structure
    if not all_roots:
        return {"structure": []}

    total_start = min(r["start_index"] for r in all_roots)
    total_end = max(r["end_index"] for r in all_roots)

    return {
        "structure": all_roots,
    }


# --- Test / demo ---
if __name__ == "__main__":
    # Test with messy model output
    messy_output = {
        "title": "Financial Report 2023",
        "children": [  # Wrong key!
            {
                "title": "Revenue Overview",
                "content": "Revenue was $X billion",  # Hallucinated
                "start_page": 1,  # Wrong key
                "end_page": 3,
                "sections": [  # Wrong key!
                    {"name": "Q1 Revenue", "page": 1},  # Wrong key, missing end
                    {"name": "Q2 Revenue", "start_index": 2, "end_index": 3},
                ]
            },
            {
                "title": "Risk Factors",
                "type": "section",  # Extra field
                "value": "high",   # Extra field
                "start_index": 4,
                "end_index": 6,
            }
        ]
    }

    result = normalize_tree(messy_output, chunk_start_page=1, chunk_end_page=6)
    print(json.dumps(result, indent=2))
    print(f"\nTotal nodes: {sum(1 for _ in _count_nodes(result))}")


def _count_nodes(nodes):
    for n in nodes:
        yield n
        yield from _count_nodes(n.get("nodes", []))
