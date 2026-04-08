# gen_dagger_v15_local.py — LOCAL version (Windows, no Colab/Drive)
# Run this directly: python scripts/gen_dagger_v15_local.py
# Requires: pip install -e . (from repo root)

import json, os, sys, random

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_FILE  = os.path.join(REPO_ROOT, "financebench_ensemble_results.json")
OUTPUT_FILE    = os.path.join(REPO_ROOT, "data", "financebench", "dagger_v15_targeted.jsonl")
REPO_TREE_DIR  = os.path.join(REPO_ROOT, "data", "financebench", "trees")
# LOCAL_TREE_CACHE: download trees from Drive and put them here for local use
# Drive path: MyDrive/arbor-training-data/multidomaintest_trees/
LOCAL_TREE_CACHE = os.path.join(REPO_ROOT, "data", "financebench", "trees_cache")
QA_FILE        = os.path.join(REPO_ROOT, "data", "financebench", "financebench_open_source.jsonl")

sys.path.insert(0, REPO_ROOT)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

ensemble = json.loads(open(ENSEMBLE_FILE, encoding="utf-8").read())
qa_list = [
    json.loads(line)
    for line in open(QA_FILE, encoding="utf-8").read().strip().splitlines()
    if line.strip()
]
qa_by_q = {i + 1: qa for i, qa in enumerate(qa_list)}

from arbor.core.types import DocumentTree, TreeNode

tree_cache = {}
search_dirs = [REPO_TREE_DIR]
if os.path.exists(LOCAL_TREE_CACHE):
    search_dirs.append(LOCAL_TREE_CACHE)
else:
    print(f"  NOTE: No local tree cache at {LOCAL_TREE_CACHE}")
    print(f"  To get all 84 trees: download MyDrive/arbor-training-data/multidomaintest_trees/ to that path")

for tree_dir in search_dirs:
    for doc in os.listdir(tree_dir):
        if doc.endswith(".json") and doc[:-5] not in tree_cache:
            try:
                raw = json.loads(open(os.path.join(tree_dir, doc), encoding="utf-8").read())
                if "tree" in raw and isinstance(raw["tree"], dict):
                    raw = raw["tree"]
                tree_cache[doc[:-5]] = DocumentTree.from_dict(raw)
            except Exception as e:
                print(f"  Skipping {doc}: {e}")

print(f"Trees loaded: {len(tree_cache)}")
print(f"QA pairs loaded: {len(qa_list)}")

# ── Oracle path tracer ────────────────────────────────────────────────────────
_INFERENCE_SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)


def find_all_target_nodes(tree, evidence_pages):
    """
    For each evidence page, find the deepest tree node whose range covers it.
    Returns a deduplicated list of target nodes (one per distinct evidence location).
    Handles multi-evidence questions by returning multiple nodes when evidence
    spans different sections (e.g., Q144 Verizon: pages 62 AND 93).
    """
    evidence_set = set(evidence_pages)

    # Collect all nodes that overlap with any evidence page, with depth
    all_matches = []
    def _search(node, depth):
        node_pages = set(range(node.start_index, node.end_index + 1))
        if node_pages & evidence_set:
            all_matches.append((node, depth, node_pages & evidence_set))
        for child in (node.nodes or []):
            _search(child, depth + 1)
    for top_node in tree.structure:
        _search(top_node, 0)

    # For each evidence page, find the deepest node covering it
    page_to_best: dict = {}
    for node, depth, covered_pages in all_matches:
        for p in covered_pages:
            if p not in page_to_best or depth > page_to_best[p][1]:
                page_to_best[p] = (node, depth)

    # Deduplicate: return each unique target node once
    seen: set = set()
    targets = []
    for node, _ in page_to_best.values():
        if node.node_id and node.node_id not in seen:
            seen.add(node.node_id)
            targets.append(node)
    return targets


def trace_path_to_node(tree, target_node_id):
    """
    Trace path from top level to target node.
    Returns list of (parent_node_or_None, correct_child_id) pairs.
    None parent means the step is at the top level (tree.structure).
    """
    path = []

    # Check if target is a top-level node
    for node in tree.structure:
        if node.node_id == target_node_id:
            return []  # Target IS the top level — no navigation needed

    # First step: which top-level node leads to target?
    for top_node in tree.structure:
        sub_path = []
        def _find(node, target, current_path):
            if node.node_id == target:
                return True
            for child in (node.nodes or []):
                current_path.append((node, child.node_id))
                if _find(child, target, current_path):
                    return True
                current_path.pop()
            return False
        if _find(top_node, target_node_id, sub_path):
            # Prepend: top-level selection step (parent=None means show tree.structure)
            path = [(None, top_node.node_id)] + sub_path
            break

    return path


def format_sections_at_level(parent_node, tree_structure=None):
    """
    Format children of a node as section list.
    If parent_node is None, format tree.structure (top-level).
    """
    if parent_node is None:
        nodes = tree_structure or []
    else:
        nodes = parent_node.nodes or []
    lines = []
    for child in nodes:
        suffix = " [has sub-sections]" if child.nodes else ""
        lines.append(
            f"[{child.node_id}] {child.title} (pages {child.start_index}-{child.end_index}){suffix}"
        )
    return "\n".join(lines)


def generate_oracle_examples(question, tree, evidence_pages, doc_name=""):
    """
    Generate oracle navigation examples for a question.
    Supports multi-evidence: if evidence spans multiple sections, generates
    training examples that select ALL relevant sections at each shared level.
    """
    target_nodes = find_all_target_nodes(tree, evidence_pages)
    if not target_nodes:
        return []

    # Trace path from root to each target node
    all_paths = []
    for target in target_nodes:
        path = trace_path_to_node(tree, target.node_id)
        if path:  # empty path means target is top-level — no navigation needed
            all_paths.append(path)
    if not all_paths:
        return []

    # Merge paths: group by parent key → collect all child IDs to select at that level.
    # Using ordered dict so root level comes first, deeper levels follow.
    from collections import OrderedDict
    parent_to_children: OrderedDict = OrderedDict()
    parent_node_lookup: dict = {}

    for path in all_paths:
        for parent_node, child_id in path:
            key = parent_node.node_id if parent_node else None
            if key not in parent_to_children:
                parent_to_children[key] = []
                parent_node_lookup[key] = parent_node
            if child_id not in parent_to_children[key]:
                parent_to_children[key].append(child_id)

    # Generate one training example per unique parent level
    examples = []
    for parent_key, child_ids in parent_to_children.items():
        parent_node = parent_node_lookup[parent_key]
        sections_text = format_sections_at_level(parent_node, tree.structure)
        if not sections_text:
            continue

        siblings = tree.structure if parent_node is None else (parent_node.nodes or [])
        child_nodes = [c for c in siblings if c.node_id in child_ids]
        child_titles = [c.title for c in child_nodes]

        if len(child_titles) == 1:
            thinking = (
                f"The question asks about '{question[:60]}'. "
                f"'{child_titles[0]}' is most likely to contain the answer."
            )
        else:
            thinking = (
                f"The question asks about '{question[:60]}'. "
                f"The answer spans multiple sections: "
                + ", ".join(f"'{t}'" for t in child_titles) + "."
            )

        examples.append({
            "messages": [
                {"role": "system",    "content": _INFERENCE_SYSTEM},
                {"role": "user",      "content": (
                    f"Question: {question}\n\n"
                    f"Sections at this level:\n{sections_text}\n\n"
                    f"Which sections should we explore next?"
                )},
                {"role": "assistant", "content": json.dumps({
                    "thinking": thinking,
                    "navigate_to": child_ids
                })},
            ]
        })
    return examples


# ── Build target list ─────────────────────────────────────────────────────────
neither_qs  = ensemble.get("neither", [])
v14_only_qs = ensemble.get("v14_only", [])

all_targets = (
    [{"q": r["q"], "doc": r["doc"], "priority": "regression"} for r in v14_only_qs] +
    [{"q": r["q"], "doc": r["doc"], "priority": "new_failure"} for r in neither_qs]
)

print(f"\nDAgger targets: {len(all_targets)}")
print(f"  Regressions (v14-only): {len(v14_only_qs)}")
print(f"  New failures (neither): {len(neither_qs)}")
print(f"\nGenerating oracle corrections...\n")

# ── Generate ─────────────────────────────────────────────────────────────────
all_examples = []
skipped = []

for target in all_targets:
    q_num    = target["q"]
    doc      = target["doc"]
    priority = target["priority"]
    qa       = qa_by_q.get(q_num)

    if not qa or doc not in tree_cache:
        skipped.append((q_num, doc, "no tree" if doc not in tree_cache else "no qa"))
        continue

    question       = qa["question"]
    evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
    tree           = tree_cache[doc]

    examples = generate_oracle_examples(question, tree, evidence_pages, doc)

    if examples:
        all_examples.extend(examples)
        print(f"  [{priority:12s}] Q{q_num:03d} {doc[:40]:<40} -> {len(examples)} examples  (evid={evidence_pages})")
    else:
        skipped.append((q_num, doc, f"no path found (evid={evidence_pages})"))
        print(f"  [SKIP        ] Q{q_num:03d} {doc[:40]:<40} - no oracle path (evid={evidence_pages})")

# ── Save ──────────────────────────────────────────────────────────────────────
random.seed(42)
random.shuffle(all_examples)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"  Generated : {len(all_examples)} training examples")
print(f"  Skipped   : {len(skipped)} questions")
if len(all_targets) - len(skipped) > 0:
    print(f"  Avg hops  : {len(all_examples) / (len(all_targets) - len(skipped)):.1f} per question")
print(f"  Output    : {OUTPUT_FILE}")
print(f"{'='*60}")

if skipped:
    print(f"\nSkipped details:")
    for q, doc, reason in skipped:
        print(f"  Q{q:03d} {doc}: {reason}")
