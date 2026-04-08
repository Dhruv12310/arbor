# gen_success_replay_local.py — LOCAL version (Windows, no Colab/Drive)
# Run this directly: python scripts/gen_success_replay_local.py
# Requires: pip install -e . (from repo root)

import json, os, sys, random

REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_FILE    = os.path.join(REPO_ROOT, "financebench_ensemble_results.json")
OUTPUT_FILE      = os.path.join(REPO_ROOT, "data", "financebench", "v14b_success_replay.jsonl")
REPO_TREE_DIR    = os.path.join(REPO_ROOT, "data", "financebench", "trees")
LOCAL_TREE_CACHE = os.path.join(REPO_ROOT, "data", "financebench", "trees_cache")
QA_FILE          = os.path.join(REPO_ROOT, "data", "financebench", "financebench_open_source.jsonl")

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

from arbor.core.types import DocumentTree

tree_cache = {}
search_dirs = [REPO_TREE_DIR]
if os.path.exists(LOCAL_TREE_CACHE):
    search_dirs.append(LOCAL_TREE_CACHE)
else:
    print(f"  NOTE: No local tree cache at {LOCAL_TREE_CACHE}")
    print(f"  Download MyDrive/arbor-training-data/multidomaintest_trees/ there for all 84 trees")

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
per_q = {r["q"]: r for r in ensemble.get("per_question", [])}

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
    For each evidence page, find the deepest tree node covering it.
    Returns deduplicated list — supports multi-evidence questions.
    """
    evidence_set = set(evidence_pages)
    all_matches = []
    def _search(node, depth):
        node_pages = set(range(node.start_index, node.end_index + 1))
        covered = node_pages & evidence_set
        if covered:
            all_matches.append((node, depth, covered))
        for child in (node.nodes or []):
            _search(child, depth + 1)
    for top_node in tree.structure:
        _search(top_node, 0)

    page_to_best: dict = {}
    for node, depth, covered in all_matches:
        for p in covered:
            if p not in page_to_best or depth > page_to_best[p][1]:
                page_to_best[p] = (node, depth)

    seen: set = set()
    targets = []
    for node, _ in page_to_best.values():
        if node.node_id and node.node_id not in seen:
            seen.add(node.node_id)
            targets.append(node)
    return targets


def trace_path_to_node(tree, target_node_id):
    for node in tree.structure:
        if node.node_id == target_node_id:
            return []  # top-level node, no navigation step needed
    path = []
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
            path = [(None, top_node.node_id)] + sub_path
            break
    return path


def format_sections_at_level(parent_node, tree_structure=None):
    nodes = tree_structure if parent_node is None else (parent_node.nodes or [])
    lines = []
    for child in nodes:
        suffix = " [has sub-sections]" if child.nodes else ""
        lines.append(
            f"[{child.node_id}] {child.title} (pages {child.start_index}-{child.end_index}){suffix}"
        )
    return "\n".join(lines)


def generate_oracle_examples(question, tree, evidence_pages):
    """Generate oracle navigation examples. Supports multi-evidence questions."""
    target_nodes = find_all_target_nodes(tree, evidence_pages)
    if not target_nodes:
        return []

    all_paths = []
    for target in target_nodes:
        path = trace_path_to_node(tree, target.node_id)
        if path:
            all_paths.append(path)
    if not all_paths:
        return []

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
                f"'{child_titles[0]}' is most likely to contain the relevant information."
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


# ── Build replay target list ──────────────────────────────────────────────────
# Priority: AMD_2022 regressions + JNJ wins + v14b_only gains + sample of both-pass
PRIORITY_DOCS = {
    "AMD_2022_10K", "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",
    "BESTBUY_2017_10K", "BESTBUY_2023_10K",
    "WALMART_2018_10K", "WALMART_2019_10K",
    "AMERICANWATERWORKS_2020_10K", "AMERICANWATERWORKS_2022_10K",
}

v14b_only = [(r["q"], r["doc"]) for r in ensemble.get("v14b_only", [])]
both_pass = [
    (r["q"], r["doc"]) for r in ensemble.get("per_question", [])
    if r.get("v14_perfect") and r.get("v14b_perfect")
]

replay_targets = []

# High priority: priority docs where v14b succeeds
for q_num, r in per_q.items():
    doc = r["doc"]
    if doc in PRIORITY_DOCS and r.get("v14b_perfect"):
        replay_targets.append({"q": q_num, "doc": doc, "priority": "high"})

# Medium: all v14b_only gains
for q_num, doc in v14b_only:
    if not any(t["q"] == q_num for t in replay_targets):
        replay_targets.append({"q": q_num, "doc": doc, "priority": "medium"})

# Low: sample 50 from both-pass (stable anchors)
random.seed(42)
for q_num, doc in random.sample(both_pass, min(50, len(both_pass))):
    if not any(t["q"] == q_num for t in replay_targets):
        replay_targets.append({"q": q_num, "doc": doc, "priority": "low"})

print(f"\nReplay targets: {len(replay_targets)}")
print(f"  High: {sum(1 for t in replay_targets if t['priority']=='high')}")
print(f"  Medium: {sum(1 for t in replay_targets if t['priority']=='medium')}")
print(f"  Low: {sum(1 for t in replay_targets if t['priority']=='low')}")
print(f"\nGenerating replay examples...\n")

# ── Generate ──────────────────────────────────────────────────────────────────
all_replay = []

for target in replay_targets:
    q_num = target["q"]
    doc   = target["doc"]
    qa    = qa_by_q.get(q_num)
    if not qa or doc not in tree_cache:
        continue
    question       = qa["question"]
    evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
    tree           = tree_cache[doc]
    examples       = generate_oracle_examples(question, tree, evidence_pages)
    if examples:
        all_replay.extend(examples)
        print(f"  [{target['priority']:6s}] Q{q_num:03d} {doc[:45]:<45} -> {len(examples)} examples")

# ── Save ──────────────────────────────────────────────────────────────────────
random.shuffle(all_replay)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in all_replay:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"  Generated : {len(all_replay)} replay examples")
print(f"  Output    : {OUTPUT_FILE}")
print(f"  These will be mixed into v15 training at ~5% to prevent forgetting.")
print(f"{'='*60}")
