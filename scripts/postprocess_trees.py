"""
Arbor Tree Post-Processor — Clean TreeGen output for TreeSearch training

Fixes:
  1. Inverted page ranges (start_index > end_index) — swaps them
  2. Junk title nodes — removes "Table of Contents", "Page X", "Untitled Section", etc.
  3. Duplicate title spam — collapses runs of identical titles
  4. Table-header titles — removes "Amount", "Total", "Product" (column headers, not sections)
  5. Empty/orphaned nodes — removes nodes with no title or no page range
  6. Re-numbers node_ids sequentially after all removals
  7. Propagates page ranges from parent → child when child has broken range
  8. Removes completely duplicate subtrees

Usage:
    python postprocess_trees.py trees/           # process all JSON files in directory
    python postprocess_trees.py trees/3M.json    # process single file

    # Or import:
    from postprocess_trees import postprocess_tree
    clean_structure = postprocess_tree(raw_structure)
"""

import json
import os
import re
import sys
import copy
from pathlib import Path


# ── Junk title patterns ──────────────────────────────────────────────────────

# Exact matches — always remove
JUNK_TITLES_EXACT = {
    "Table of Contents",
    "Untitled Section",
    "Amount",
    "Total",
    "Product",
    "",
}

# Prefix matches — remove if title starts with these
JUNK_TITLE_PREFIXES = [
    "Page ",          # "Page 12", "Page 55"
]

# Regex patterns — remove if title matches
JUNK_TITLE_PATTERNS = [
    r"^Page \d+$",                           # "Page 12"
    r"^\d{4}$",                              # "2019", "2018" (bare year)
    r"^For the Years? Ended",                # "For the Years Ended December 31"
    r"^Fiscal Year \d{4}$",                  # "Fiscal Year 2015"
]
_JUNK_REGEXES = [re.compile(p, re.IGNORECASE) for p in JUNK_TITLE_PATTERNS]


def is_junk_title(title: str) -> bool:
    """Check if a node title is junk that should be removed."""
    title = title.strip()
    
    if title in JUNK_TITLES_EXACT:
        return True
    
    for prefix in JUNK_TITLE_PREFIXES:
        if title.startswith(prefix) and title[len(prefix):].strip().isdigit():
            return True
    
    for regex in _JUNK_REGEXES:
        if regex.match(title):
            return True
    
    return False


def is_repetitive_boilerplate(title: str) -> bool:
    """
    Detect titles that are document boilerplate, not real sections.
    These appear many times because they're headers/footers extracted as content.
    """
    boilerplate = {
        "Notes to Consolidated Financial Statements (Continued)",
        "Notes to Consolidated Financial Statements (continued)",
        "Notes to Consolidated Financial Statements",
    }
    return title.strip() in boilerplate


# ── Core fixes ────────────────────────────────────────────────────────────────

def fix_page_range(node: dict, parent_start: int = None, parent_end: int = None) -> None:
    """
    Fix inverted page ranges in-place.
    Rules:
      1. If start > end, swap them
      2. If child range exceeds parent range, clamp to parent
      3. If start or end is missing/None, inherit from parent
    """
    si = node.get("start_index")
    ei = node.get("end_index")
    
    # Handle missing values
    if si is None:
        si = parent_start or 1
    if ei is None:
        ei = parent_end or si
    
    # Fix inversion
    if si > ei:
        si, ei = ei, si
    
    # Clamp to parent range
    if parent_start is not None:
        si = max(si, parent_start)
    if parent_end is not None:
        ei = min(ei, parent_end)
    
    # Ensure still valid after clamping
    if si > ei:
        si = ei
    
    node["start_index"] = si
    node["end_index"] = ei
    
    # Recurse into children
    for child in node.get("nodes", []):
        fix_page_range(child, si, ei)


def remove_junk_nodes(nodes: list[dict]) -> list[dict]:
    """
    Remove nodes with junk titles. 
    When removing a parent node, promote its children to take its place.
    """
    result = []
    for node in nodes:
        title = node.get("title", "").strip()
        children = node.get("nodes", [])
        
        # Recursively clean children first
        clean_children = remove_junk_nodes(children)
        
        if is_junk_title(title):
            # Remove this node but keep its children (promote them)
            result.extend(clean_children)
        else:
            # Keep this node with cleaned children
            if clean_children:
                node["nodes"] = clean_children
            elif "nodes" in node:
                del node["nodes"]
            result.append(node)
    
    return result


def remove_boilerplate_nodes(nodes: list[dict]) -> list[dict]:
    """
    Remove repetitive boilerplate nodes (e.g., repeated 'Notes to Consolidated
    Financial Statements (Continued)' that appear as headers on every page).
    
    Keep the FIRST occurrence, remove subsequent ones.
    """
    seen_boilerplate = set()
    result = []
    
    for node in nodes:
        title = node.get("title", "").strip()
        children = node.get("nodes", [])
        
        # Recurse first
        if children:
            node["nodes"] = remove_boilerplate_nodes(children)
            if not node["nodes"]:
                del node["nodes"]
        
        if is_repetitive_boilerplate(title):
            if title in seen_boilerplate:
                # Skip duplicate boilerplate, but promote any children
                promoted = node.get("nodes", [])
                result.extend(promoted)
                continue
            seen_boilerplate.add(title)
        
        result.append(node)
    
    return result


def collapse_duplicate_siblings(nodes: list[dict]) -> list[dict]:
    """
    When consecutive sibling nodes have identical titles (e.g., multiple
    'ACTIVISION BLIZZARD, INC. AND SUBSIDIARIES' nodes), merge them into one
    node spanning the combined page range, combining their children.
    """
    if not nodes:
        return nodes
    
    result = []
    i = 0
    while i < len(nodes):
        current = nodes[i]
        title = current.get("title", "")
        
        # Look ahead for consecutive duplicates
        j = i + 1
        while j < len(nodes) and nodes[j].get("title", "") == title:
            j += 1
        
        if j > i + 1:
            # Found duplicates: merge nodes[i:j]
            merged = copy.deepcopy(current)
            all_children = list(current.get("nodes", []))
            min_start = current.get("start_index", 9999)
            max_end = current.get("end_index", 0)
            
            for k in range(i + 1, j):
                dup = nodes[k]
                all_children.extend(dup.get("nodes", []))
                min_start = min(min_start, dup.get("start_index", 9999))
                max_end = max(max_end, dup.get("end_index", 0))
            
            merged["start_index"] = min_start
            merged["end_index"] = max_end
            if all_children:
                merged["nodes"] = collapse_duplicate_siblings(all_children)
            elif "nodes" in merged:
                del merged["nodes"]
            
            result.append(merged)
            i = j
        else:
            # No duplicates — recurse into children
            if current.get("nodes"):
                current["nodes"] = collapse_duplicate_siblings(current["nodes"])
            result.append(current)
            i += 1
    
    return result


def remove_single_page_spam(nodes: list[dict], min_title_length: int = 3) -> list[dict]:
    """
    Remove nodes that:
    - Span exactly 1 page
    - Have a very short title (< min_title_length chars)  
    - Have no children
    These are typically table cell extractions, not real sections.
    """
    result = []
    for node in nodes:
        title = node.get("title", "").strip()
        si = node.get("start_index", 0)
        ei = node.get("end_index", 0)
        children = node.get("nodes", [])
        
        # Recurse
        if children:
            node["nodes"] = remove_single_page_spam(children, min_title_length)
            if not node["nodes"]:
                del node["nodes"]
                children = []
        
        # Filter: skip tiny leaf nodes
        is_single_page_leaf = (si == ei) and not children
        is_tiny_title = len(title) < min_title_length
        
        if is_single_page_leaf and is_tiny_title:
            continue
        
        result.append(node)
    
    return result


def remove_empty_nodes(nodes: list[dict]) -> list[dict]:
    """Remove nodes that have no title and no children."""
    result = []
    for node in nodes:
        children = node.get("nodes", [])
        if children:
            node["nodes"] = remove_empty_nodes(children)
            if not node["nodes"]:
                del node["nodes"]
        
        title = node.get("title", "").strip()
        has_children = bool(node.get("nodes"))
        
        if not title and not has_children:
            continue
        
        result.append(node)
    
    return result


def reassign_node_ids(nodes: list[dict], counter: list = None) -> None:
    """Reassign sequential node IDs after all removals."""
    if counter is None:
        counter = [0]
    
    for node in nodes:
        counter[0] += 1
        node["node_id"] = str(counter[0]).zfill(4)
        if node.get("nodes"):
            reassign_node_ids(node["nodes"], counter)


def strip_summary_field(nodes: list[dict]) -> None:
    """
    Remove 'summary' field from all nodes.
    TreeSearch training data doesn't use summaries — only title, node_id, 
    start_index, end_index, nodes.
    """
    for node in nodes:
        node.pop("summary", None)
        if node.get("nodes"):
            strip_summary_field(node["nodes"])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def get_covered_pages(nodes: list[dict]) -> set[int]:
    """Collect every page number covered by any node in the tree."""
    covered = set()
    for n in nodes:
        si = n.get("start_index", 0)
        ei = n.get("end_index", 0)
        covered.update(range(si, ei + 1))
        covered.update(get_covered_pages(n.get("nodes", [])))
    return covered


def fill_page_gaps(nodes: list[dict], total_pages: int) -> list[dict]:
    """
    Add placeholder root nodes for page ranges not covered by any existing node.
    Without these, TreeSearch can never route to those pages — they're black holes.
    Covered pages are detected across the full tree (not just root level).
    """
    if not nodes or total_pages < 1:
        return nodes

    covered = get_covered_pages(nodes)
    gaps = sorted(set(range(1, total_pages + 1)) - covered)
    if not gaps:
        return nodes

    # Group consecutive gap pages into ranges
    gap_nodes = []
    start = gaps[0]
    prev = gaps[0]
    for page in gaps[1:]:
        if page != prev + 1:
            gap_nodes.append({
                "title": f"Filing content (pages {start}-{prev})",
                "node_id": "TEMP",
                "start_index": start,
                "end_index": prev,
            })
            start = page
        prev = page
    gap_nodes.append({
        "title": f"Filing content (pages {start}-{prev})",
        "node_id": "TEMP",
        "start_index": start,
        "end_index": prev,
    })

    # Merge gap nodes into main structure, sorted by start page
    combined = nodes + gap_nodes
    combined.sort(key=lambda n: n.get("start_index", 0))
    return combined


def postprocess_tree(structure: list[dict], total_pages: int = 0,
                     strip_summaries: bool = False) -> list[dict]:
    """
    Full post-processing pipeline for a tree structure.

    Args:
        structure: List of root-level tree nodes
        total_pages: Total page count of the document (for gap-filling; 0 = skip)
        strip_summaries: Whether to remove summary fields.
                         Default False — existing training data preserves summaries.

    Returns:
        Cleaned tree structure
    """
    if not structure:
        return structure

    # Deep copy to avoid mutating the original
    nodes = copy.deepcopy(structure)

    # Step 1: Fix all page ranges (inversions, clamping)
    for node in nodes:
        fix_page_range(node)

    # Step 2: Remove junk title nodes (promotes children)
    nodes = remove_junk_nodes(nodes)

    # Step 3: Remove boilerplate repetitions
    nodes = remove_boilerplate_nodes(nodes)

    # Step 4: Collapse consecutive duplicate siblings
    nodes = collapse_duplicate_siblings(nodes)

    # Step 5: Remove single-page spam with tiny titles
    nodes = remove_single_page_spam(nodes)

    # Step 6: Remove empty nodes
    nodes = remove_empty_nodes(nodes)

    # Step 7: Gap-fill uncovered page ranges
    if total_pages > 0:
        nodes = fill_page_gaps(nodes, total_pages)

    # Step 8: Strip summaries only if explicitly requested
    if strip_summaries:
        strip_summary_field(nodes)

    # Step 9: Reassign sequential node IDs
    reassign_node_ids(nodes)

    return nodes


def count_nodes(nodes: list[dict]) -> int:
    """Count total nodes in a tree."""
    total = 0
    for n in nodes:
        total += 1
        total += count_nodes(n.get("nodes", []))
    return total


def count_inversions(nodes: list[dict]) -> int:
    """Count nodes with inverted page ranges."""
    total = 0
    for n in nodes:
        if n.get("start_index", 0) > n.get("end_index", 0):
            total += 1
        total += count_inversions(n.get("nodes", []))
    return total


def count_junk_titles(nodes: list[dict]) -> int:
    """Count nodes with junk titles."""
    total = 0
    for n in nodes:
        if is_junk_title(n.get("title", "")):
            total += 1
        total += count_junk_titles(n.get("nodes", []))
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def process_file(filepath: str) -> dict:
    """Process a single tree JSON file and return stats."""
    with open(filepath) as f:
        data = json.load(f)
    
    tree = data.get("tree", data)
    structure = tree.get("structure", [])
    
    if not structure:
        return {"error": "No structure found"}
    
    # Before stats
    before_nodes = count_nodes(structure)
    before_inversions = count_inversions(structure)
    before_junk = count_junk_titles(structure)
    
    # Infer total_pages from the deepest end_index in the tree
    def max_page(nodes):
        m = 0
        for n in nodes:
            m = max(m, n.get("end_index", 0))
            m = max(m, max_page(n.get("nodes", [])))
        return m

    total_pages = tree.get("total_pages") or max_page(structure)

    # Process
    clean = postprocess_tree(structure, total_pages=total_pages)
    
    # After stats
    after_nodes = count_nodes(clean)
    after_inversions = count_inversions(clean)
    after_junk = count_junk_titles(clean)
    
    # Update the data
    tree["structure"] = clean
    if "tree" in data:
        data["tree"] = tree
    else:
        data = tree
    
    # Write back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {
        "before_nodes": before_nodes,
        "after_nodes": after_nodes,
        "removed_nodes": before_nodes - after_nodes,
        "before_inversions": before_inversions,
        "after_inversions": after_inversions,
        "before_junk": before_junk,
        "after_junk": after_junk,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python postprocess_trees.py <path_to_json_or_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if target.is_dir():
        files = sorted(target.glob("*.json"))
    elif target.is_file():
        files = [target]
    else:
        print(f"Not found: {target}")
        sys.exit(1)
    
    print(f"Processing {len(files)} files...\n")
    print(f"{'File':<40s} {'Before':>8s} {'After':>8s} {'Removed':>8s} {'Inv.Fix':>8s} {'Junk.Fix':>9s}")
    print("─" * 85)
    
    total_before = total_after = total_inv_fixed = total_junk_fixed = 0
    
    for fpath in files:
        stats = process_file(str(fpath))
        if "error" in stats:
            print(f"{fpath.name:<40s} ERROR: {stats['error']}")
            continue
        
        name = fpath.stem
        inv_fixed = stats["before_inversions"] - stats["after_inversions"]
        junk_fixed = stats["before_junk"] - stats["after_junk"]
        
        print(f"{name:<40s} {stats['before_nodes']:>8d} {stats['after_nodes']:>8d} "
              f"{stats['removed_nodes']:>8d} {inv_fixed:>8d} {junk_fixed:>9d}")
        
        total_before += stats["before_nodes"]
        total_after += stats["after_nodes"]
        total_inv_fixed += inv_fixed
        total_junk_fixed += junk_fixed
    
    print("─" * 85)
    print(f"{'TOTAL':<40s} {total_before:>8d} {total_after:>8d} "
          f"{total_before - total_after:>8d} {total_inv_fixed:>8d} {total_junk_fixed:>9d}")


if __name__ == "__main__":
    main()
