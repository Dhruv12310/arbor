# gen_dagger_v17_local.py — LOCAL version (Windows, no Colab/Drive)
# Generates DAgger correction examples for all 56 v16 failures.
# Run: python scripts/gen_dagger_v17_local.py
# Requires: pip install -e . (from repo root)

import json, os, sys, random
from collections import OrderedDict

REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V16_RESULTS_FILE = os.path.join(REPO_ROOT, "financebench_v16_results.json")
OUTPUT_FILE      = os.path.join(REPO_ROOT, "data", "financebench", "dagger_v17_targeted.jsonl")
REPO_TREE_DIR    = os.path.join(REPO_ROOT, "data", "financebench", "trees")
LOCAL_TREE_CACHE = os.path.join(REPO_ROOT, "data", "financebench", "trees_cache")
QA_FILE          = os.path.join(REPO_ROOT, "data", "financebench", "financebench_open_source.jsonl")

sys.path.insert(0, REPO_ROOT)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

v16_results = json.loads(open(V16_RESULTS_FILE, encoding="utf-8").read())
qa_list = [
    json.loads(line)
    for line in open(QA_FILE, encoding="utf-8").read().strip().splitlines()
    if line.strip()
]
qa_by_q = {i + 1: qa for i, qa in enumerate(qa_list)}

from arbor.core.types import DocumentTree, TreeNode

tree_cache = {}
search_dirs = [d for d in [REPO_TREE_DIR, LOCAL_TREE_CACHE] if os.path.exists(d)]

for tree_dir in search_dirs:
    for fname in os.listdir(tree_dir):
        if fname.endswith(".json") and fname[:-5] not in tree_cache:
            try:
                raw = json.loads(open(os.path.join(tree_dir, fname), encoding="utf-8").read())
                if "tree" in raw and isinstance(raw["tree"], dict):
                    raw = raw["tree"]
                tree_cache[fname[:-5]] = DocumentTree.from_dict(raw)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

print(f"Trees loaded: {len(tree_cache)}")
print(f"QA pairs loaded: {len(qa_list)}")

# ── Oracle path helpers ───────────────────────────────────────────────────────
_INFERENCE_SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)


def find_all_target_nodes(tree, evidence_pages):
    """For each evidence page, find deepest tree node covering it. Returns deduped list."""
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
    """Return list of (parent_node_or_None, correct_child_id) from root to target."""
    for node in tree.structure:
        if node.node_id == target_node_id:
            return []  # top-level — no navigation step needed

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
            return [(None, top_node.node_id)] + sub_path

    return []


def format_sections_at_level(parent_node, tree_structure=None):
    nodes = tree_structure if parent_node is None else (parent_node.nodes or [])
    lines = []
    for child in nodes:
        suffix = " [has sub-sections]" if child.nodes else ""
        lines.append(
            f"[{child.node_id}] {child.title} (pages {child.start_index}-{child.end_index}){suffix}"
        )
    return "\n".join(lines)


def generate_oracle_examples(question, tree, evidence_pages, doc_name=""):
    """Generate oracle navigation training examples. Supports multi-evidence questions."""
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
                    "thinking":    thinking,
                    "navigate_to": child_ids,
                })},
            ]
        })
    return examples


# ── Build target list from v16 failures ──────────────────────────────────────
failures = [r for r in v16_results if not r["perfect"]]
print(f"\nv16 failures: {len(failures)}")

all_targets = []
skipped_unreachable = []

for r in failures:
    q_num = r["q"]
    doc   = r["doc"]
    qa    = qa_by_q.get(q_num)
    if not qa:
        continue
    evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
    # Skip cover-page questions (evidence=[0]) — structurally unreachable
    if not evidence_pages or evidence_pages == [0]:
        skipped_unreachable.append((q_num, doc, evidence_pages))
        continue
    all_targets.append({"q": q_num, "doc": doc, "evidence": evidence_pages})

print(f"Targetable failures: {len(all_targets)}")
print(f"Skipped (unreachable evid=[0]): {len(skipped_unreachable)}")
if skipped_unreachable:
    for q, doc, evid in skipped_unreachable:
        print(f"  Q{q:03d} {doc}: evid={evid}")

print(f"\nGenerating oracle corrections...\n")

# ── Generate ─────────────────────────────────────────────────────────────────
all_examples = []
skipped = []

for target in all_targets:
    q_num          = target["q"]
    doc            = target["doc"]
    evidence_pages = target["evidence"]
    qa             = qa_by_q.get(q_num)

    if doc not in tree_cache:
        skipped.append((q_num, doc, "no tree in cache"))
        continue

    question = qa["question"]
    tree     = tree_cache[doc]
    examples = generate_oracle_examples(question, tree, evidence_pages, doc)

    if examples:
        all_examples.extend(examples)
        print(f"  Q{q_num:03d} {doc[:45]:<45} -> {len(examples)} examples  (evid={evidence_pages})")
    else:
        skipped.append((q_num, doc, f"no oracle path (evid={evidence_pages})"))
        print(f"  [SKIP] Q{q_num:03d} {doc[:45]:<45} - no path (evid={evidence_pages})")

# ── Save ──────────────────────────────────────────────────────────────────────
random.seed(42)
random.shuffle(all_examples)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

targetable = len(all_targets)
generated  = targetable - len(skipped)

print(f"\n{'='*60}")
print(f"  v16 failures    : {len(failures)}")
print(f"  Unreachable skip: {len(skipped_unreachable)}")
print(f"  Targetable      : {targetable}")
print(f"  Generated paths : {generated}")
print(f"  Skipped (no path): {len(skipped)}")
print(f"  Training examples: {len(all_examples)}")
if generated > 0:
    print(f"  Avg hops/question: {len(all_examples) / generated:.1f}")
print(f"  Output          : {OUTPUT_FILE}")
print(f"{'='*60}")

if skipped:
    print(f"\nSkipped details:")
    for q, doc, reason in skipped:
        print(f"  Q{q:03d} {doc}: {reason}")
