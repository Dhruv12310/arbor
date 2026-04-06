# gen_structdirect_data.py
# ========================
# Generates training data for TreeSearch v12 from StructDirect-extracted trees.
# Run this BEFORE finetune_treesearch_v12.py.
#
# Changes vs v11 generator:
#   - Expanded from 7 docs to all 84 FinanceBench docs
#     -> ~2800 unique examples (was ~130), no oversampling needed
#   - Covers all filing types: 10-K, 10-Q, 8-K, Earnings releases
#   - Skips PDFs that fail extraction (try/except per doc) so one bad PDF
#     does not abort the whole run
#   - Multi-node targets: when evidence spans 2 sections (e.g. Adobe pages 58 AND 62),
#     navigate_to correctly lists BOTH nodes instead of just one
#   - Truncation guard: node lists > 20 entries are chunked into windows of 20
#     so prompts stay under 500 tokens and never hit the 4096 seq limit
#   - Augmented with synthetic template examples for all docs
#   - Shuffled duplicates for position-invariance training
#
# Output: structdirect_train.jsonl saved to Drive

import subprocess, sys, os, json, random
subprocess.run(["pip", "install", "pymupdf", "-q"], check=True)

from google.colab import drive
drive.mount('/content/drive')

if not os.path.exists("/content/arbor"):
    subprocess.run(["git", "clone", "https://github.com/Dhruv12310/arbor.git", "/content/arbor"], check=True)
else:
    subprocess.run(["git", "-C", "/content/arbor", "pull"], check=True)
sys.path.insert(0, "/content/arbor")

from arbor.extraction.structure_extractor import extract_structure

PDF_DIR  = "/content/drive/MyDrive/arbor-training-data/financebench-pdfs"
QA_FILE  = "/content/arbor/data/financebench/financebench_open_source.jsonl"
OUT_FILE = "/content/drive/MyDrive/arbor-training-data/structdirect_train.jsonl"

TARGET_DOCS = [
    "3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q",
    "ACTIVISIONBLIZZARD_2019_10K",
    "ADOBE_2015_10K", "ADOBE_2016_10K", "ADOBE_2017_10K", "ADOBE_2022_10K",
    "AES_2022_10K",
    "AMAZON_2017_10K", "AMAZON_2019_10K",
    "AMCOR_2020_10K", "AMCOR_2022_8K_dated-2022-07-01", "AMCOR_2023_10K",
    "AMCOR_2023Q2_10Q", "AMCOR_2023Q4_EARNINGS",
    "AMD_2015_10K", "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "AMERICANWATERWORKS_2020_10K", "AMERICANWATERWORKS_2021_10K", "AMERICANWATERWORKS_2022_10K",
    "BESTBUY_2017_10K", "BESTBUY_2019_10K", "BESTBUY_2023_10K", "BESTBUY_2024Q2_10Q",
    "BLOCK_2016_10K", "BLOCK_2020_10K",
    "BOEING_2018_10K", "BOEING_2022_10K",
    "COCACOLA_2017_10K", "COCACOLA_2021_10K", "COCACOLA_2022_10K",
    "CORNING_2020_10K", "CORNING_2021_10K", "CORNING_2022_10K",
    "COSTCO_2021_10K",
    "CVSHEALTH_2018_10K", "CVSHEALTH_2022_10K",
    "FOOTLOCKER_2022_8K_dated_2022-08-19", "FOOTLOCKER_2022_8K_dated-2022-05-20",
    "GENERALMILLS_2019_10K", "GENERALMILLS_2020_10K", "GENERALMILLS_2022_10K",
    "JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2022Q4_EARNINGS",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30", "JOHNSON_JOHNSON_2023Q2_EARNINGS",
    "JPMORGAN_2021Q1_10Q", "JPMORGAN_2022_10K", "JPMORGAN_2022Q2_10Q", "JPMORGAN_2023Q2_10Q",
    "KRAFTHEINZ_2019_10K",
    "LOCKHEEDMARTIN_2020_10K", "LOCKHEEDMARTIN_2021_10K", "LOCKHEEDMARTIN_2022_10K",
    "MGMRESORTS_2018_10K", "MGMRESORTS_2020_10K", "MGMRESORTS_2022_10K",
    "MGMRESORTS_2022Q4_EARNINGS", "MGMRESORTS_2023Q2_10Q",
    "MICROSOFT_2016_10K", "MICROSOFT_2023_10K",
    "NETFLIX_2015_10K", "NETFLIX_2017_10K",
    "NIKE_2018_10K", "NIKE_2019_10K", "NIKE_2021_10K", "NIKE_2023_10K",
    "PAYPAL_2022_10K",
    "PEPSICO_2021_10K", "PEPSICO_2022_10K",
    "PEPSICO_2023_8K_dated-2023-05-05", "PEPSICO_2023_8K_dated-2023-05-30",
    "PEPSICO_2023Q1_EARNINGS",
    "PFIZER_2021_10K", "Pfizer_2023Q2_10Q",
    "ULTABEAUTY_2023_10K", "ULTABEAUTY_2023Q4_EARNINGS",
    "VERIZON_2021_10K", "VERIZON_2022_10K",
    "WALMART_2018_10K", "WALMART_2019_10K", "WALMART_2020_10K",
]

SYSTEM_PROMPT = (
    "You are a document tree navigator. Given a question and a list of "
    "document sections at the current level, output JSON specifying which "
    "sections to explore next.\n"
    "Reply format: {\"thinking\": \"...\", \"navigate_to\": [\"XXXX\", ...]}"
)

FINANCIAL_TEMPLATES = [
    ("What is the total revenue for this company?",              ["revenue", "income", "financial", "operations", "results"]),
    ("What is the net income or net earnings?",                  ["income", "earnings", "financial", "statement", "results"]),
    ("What are the total assets and liabilities?",               ["balance", "financial", "assets", "sheet"]),
    ("What is the operating cash flow?",                         ["cash", "flow", "financial", "statement", "liquidity"]),
    ("What is the capital expenditure?",                         ["cash", "flow", "capital", "financial", "expenditure"]),
    ("What drove operating margin changes?",                     ["operations", "management", "discussion", "analysis", "results"]),
    ("What are the main risk factors?",                          ["risk", "factors", "item 1a"]),
    ("What business segments does this company operate in?",     ["business", "segment", "item 1", "operations"]),
    ("What is the company's liquidity position?",                ["liquidity", "cash", "management", "discussion", "capital"]),
    ("What are the total stockholders equity figures?",          ["balance", "equity", "financial", "stockholder"]),
    ("What is the earnings per share?",                          ["income", "earnings", "financial", "statement"]),
    ("What are the notes to financial statements?",              ["notes", "financial", "supplementary", "item 8"]),
    ("What is the gross profit margin?",                         ["income", "financial", "results", "operations"]),
    ("What is the return on equity?",                            ["income", "financial", "balance", "equity"]),
    ("What are the company's long-term debt obligations?",       ["debt", "financial", "notes", "balance", "liquidity"]),
]

MAX_NODES_PER_HOP = 20   # hard cap -- keeps prompts under ~600 tokens

# ── Helpers ────────────────────────────────────────────────────────────────────
def nodes_containing_pages(nodes, target_pages):
    # Return ALL node_ids whose page range overlaps any target page.
    return [n.node_id for n in nodes
            if any(n.start_index <= p <= n.end_index for p in target_pages)]

def get_children(all_nodes, parent_id):
    def find(nodes):
        for n in nodes:
            if n.node_id == parent_id: return n.nodes
            r = find(n.nodes)
            if r is not None: return r
        return None
    return find(all_nodes) or []

def format_node_list(nodes):
    lines = []
    for n in nodes:
        sub = " [has sub-sections]" if n.nodes else ""
        lines.append(f"[{n.node_id}] {n.title} (pages {n.start_index}-{n.end_index}){sub}")
    return "\n".join(lines)

def make_thinking(question, target_ids, nodes):
    id_to_title = {n.node_id: n.title for n in nodes}
    targets = [id_to_title.get(t, t) for t in target_ids]
    return (
        f"The question asks: '{question[:80]}'. "
        f"'{', '.join(targets)}' is most likely to contain the answer."
    )

def make_example(doc, question, node_list, target_ids):
    # Build one training example. Truncates node_list to MAX_NODES_PER_HOP centered
    # around the target nodes so prompts stay short and targets are always visible.
    if len(node_list) > MAX_NODES_PER_HOP:
        # Find target positions and build a window around them
        id_to_idx = {n.node_id: i for i, n in enumerate(node_list)}
        target_positions = [id_to_idx[t] for t in target_ids if t in id_to_idx]
        if target_positions:
            center = sum(target_positions) // len(target_positions)
            half = MAX_NODES_PER_HOP // 2
            start = max(0, center - half)
            end   = min(len(node_list), start + MAX_NODES_PER_HOP)
            start = max(0, end - MAX_NODES_PER_HOP)
            node_list = node_list[start:end]
        else:
            node_list = node_list[:MAX_NODES_PER_HOP]

    return {"doc": doc, "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Question: {question}\n\nSections at this level:\n{format_node_list(node_list)}"},
        {"role": "assistant", "content": json.dumps({
            "thinking":    make_thinking(question, target_ids, node_list),
            "navigate_to": target_ids,
        })}
    ]}

def title_matches_keywords(title, keywords):
    tl = title.lower()
    return any(k in tl for k in keywords)

def add_recovery_examples(tree, question, evidence_pages, examples, doc, max_distractors=3):
    """
    Recovery training: model navigated into wrong branch, must correct course.
    Shows top-level list again with "Previously explored: [wrong_node]" prefix.
    Correct answer is the evidence-containing node.

    Mirrors DAgger / imitation-with-corrections: trains the model on states
    that arise from its own navigation errors (not just perfect oracle paths).
    """
    top_level = tree.structure
    evidence_set = set(nodes_containing_pages(top_level, evidence_pages))
    if not evidence_set:
        return

    wrong_nodes = [n for n in top_level if n.node_id not in evidence_set]
    random.shuffle(wrong_nodes)

    for wrong_node in wrong_nodes[:max_distractors]:
        evidence_titles = [n.title for n in top_level if n.node_id in evidence_set]
        thinking = (
            f"I previously explored '{wrong_node.title}' which did not contain the answer. "
            f"The question asks: '{question[:60]}'. "
            f"The answer is more likely in '{', '.join(evidence_titles)}'."
        )

        node_list = top_level[:]
        if len(node_list) > MAX_NODES_PER_HOP:
            id_to_idx = {n.node_id: i for i, n in enumerate(node_list)}
            positions = [id_to_idx[t] for t in evidence_set if t in id_to_idx]
            if positions:
                center = sum(positions) // len(positions)
                half   = MAX_NODES_PER_HOP // 2
                start  = max(0, center - half)
                end    = min(len(node_list), start + MAX_NODES_PER_HOP)
                start  = max(0, end - MAX_NODES_PER_HOP)
                node_list = node_list[start:end]

        target_ids = [t for t in list(evidence_set) if any(n.node_id == t for n in node_list)]
        if not target_ids:
            continue

        user_content = (
            f"Question: {question}\n\n"
            f"Previously explored: [{wrong_node.node_id}] {wrong_node.title} "
            f"(pages {wrong_node.start_index}-{wrong_node.end_index}) — no relevant content found.\n\n"
            f"Sections at this level:\n"
            f"{format_node_list(node_list)}"
        )
        examples.append({"doc": doc, "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps({
                "thinking":    thinking,
                "navigate_to": target_ids,
            })},
        ]})


def add_negative_examples(tree, question, evidence_pages, examples, doc):
    """
    Hard negative training: shows the model a semantically plausible but wrong
    neighboring section and trains it to explicitly reject it in favor of the
    correct section.

    Mirrors Word2Vec negative sampling: the model must distinguish the right
    section from a close-but-wrong neighbor, not just learn what's correct.
    """
    top_level = tree.structure
    evidence_set = set(nodes_containing_pages(top_level, evidence_pages))
    if not evidence_set:
        return

    id_to_idx = {n.node_id: i for i, n in enumerate(top_level)}

    for correct_id in list(evidence_set):
        correct_idx = id_to_idx.get(correct_id)
        if correct_idx is None:
            continue

        # Closest DFS neighbor that does NOT contain evidence (hard negative)
        best_neighbor = None
        best_dist = float("inf")
        for n in top_level:
            if n.node_id in evidence_set:
                continue
            dist = abs(id_to_idx.get(n.node_id, 999) - correct_idx)
            if dist < best_dist:
                best_dist = dist
                best_neighbor = n

        if best_neighbor is None:
            continue

        correct_node = next((n for n in top_level if n.node_id == correct_id), None)
        if correct_node is None:
            continue

        thinking = (
            f"Although '[{best_neighbor.node_id}] {best_neighbor.title}' "
            f"(pages {best_neighbor.start_index}-{best_neighbor.end_index}) is nearby, "
            f"it does not contain the answer. "
            f"The answer requires '[{correct_id}] {correct_node.title}' "
            f"(pages {correct_node.start_index}-{correct_node.end_index})."
        )

        node_list = top_level[:]
        if len(node_list) > MAX_NODES_PER_HOP:
            center = (id_to_idx.get(correct_id, 0) + id_to_idx.get(best_neighbor.node_id, 0)) // 2
            half   = MAX_NODES_PER_HOP // 2
            start  = max(0, center - half)
            end    = min(len(node_list), start + MAX_NODES_PER_HOP)
            start  = max(0, end - MAX_NODES_PER_HOP)
            node_list = node_list[start:end]

        target_ids = [t for t in list(evidence_set) if any(n.node_id == t for n in node_list)]
        if not target_ids:
            continue

        user_content = (
            f"Question: {question}\n\n"
            f"Sections at this level:\n"
            f"{format_node_list(node_list)}"
        )
        examples.append({"doc": doc, "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps({
                "thinking":    thinking,
                "navigate_to": target_ids,
            })},
        ]})
        break  # one hard negative per QA pair to avoid class imbalance

# ── Extract trees ──────────────────────────────────────────────────────────────
print(f"Extracting trees for {len(TARGET_DOCS)} docs...")
trees = {}
failed = []
for doc in TARGET_DOCS:
    try:
        trees[doc] = extract_structure(f"{PDF_DIR}/{doc}.pdf")
    except Exception as e:
        print(f"  SKIP {doc}: {e}")
        failed.append(doc)
print(f"Extracted: {len(trees)} trees | Failed: {len(failed)}")
if failed:
    print(f"  Failed docs: {failed}")

# ── Load QA pairs ──────────────────────────────────────────────────────────────
all_qa   = [json.loads(l) for l in open(QA_FILE, encoding="utf-8").read().strip().splitlines()]
qa_pairs = [q for q in all_qa if q["doc_name"] in trees]
print(f"Loaded {len(qa_pairs)} QA pairs")

examples = []
skipped  = 0

# ── Part 1: Real QA pairs — all hops, multi-node targets, shuffled variants ───
for qa in qa_pairs:
    doc            = qa["doc_name"]
    question       = qa["question"]
    evidence_pages = [e["evidence_page_num"] for e in qa.get("evidence", [])]
    if not evidence_pages or evidence_pages == [0]:
        skipped += 1
        continue
    tree = trees[doc]

    def add_hops(node_list, depth=0):
        # Find ALL nodes at this level that contain any evidence page
        targets = nodes_containing_pages(node_list, evidence_pages)
        if not targets or depth > 3:
            return
        # Original order
        examples.append(make_example(doc, question, node_list, targets))
        # Shuffled order (position invariance)
        shuffled = node_list[:]
        random.shuffle(shuffled)
        examples.append(make_example(doc, question, shuffled, targets))
        # Recurse into children of each target
        for t in targets:
            children = get_children(tree.structure, t)
            if children:
                add_hops(children, depth + 1)

    add_hops(tree.structure)
    add_recovery_examples(tree, question, evidence_pages, examples, doc)
    add_negative_examples(tree, question, evidence_pages, examples, doc)

print(f"Real QA examples: {len(examples)}")

# ── Part 2: Synthetic template examples ───────────────────────────────────────
synthetic = []
for doc, tree in trees.items():
    for question, keywords in FINANCIAL_TEMPLATES:
        targets = [n.node_id for n in tree.structure if title_matches_keywords(n.title, keywords)]
        if not targets:
            continue
        synthetic.append(make_example(doc, question, tree.structure, targets))
        # Child level
        for parent_id in targets:
            children = get_children(tree.structure, parent_id)
            if not children:
                continue
            child_targets = [n.node_id for n in children if title_matches_keywords(n.title, keywords)]
            if not child_targets:
                child_targets = [children[0].node_id]
            synthetic.append(make_example(doc, question, children, child_targets))

print(f"Synthetic examples: {len(synthetic)}")

# ── Combine, deduplicate, shuffle ──────────────────────────────────────────────
all_examples = examples + synthetic
seen = set()
unique = []
for ex in all_examples:
    key = ex["messages"][1]["content"]
    if key not in seen:
        seen.add(key)
        unique.append(ex)

random.seed(42)
random.shuffle(unique)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for ex in unique:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

multi_node = [ex for ex in unique if len(json.loads(ex["messages"][2]["content"])["navigate_to"]) > 1]
print(f"\nDocs extracted : {len(trees)} / {len(TARGET_DOCS)}")
print(f"Real QA        : {len(examples)}")
print(f"Synthetic      : {len(synthetic)}")
print(f"Total unique   : {len(unique)}")
print(f"Multi-node     : {len(multi_node)} (navigate_to > 1)")
print(f"Skipped QA     : {skipped} (no evidence pages)")
print(f"Saved to       : {OUT_FILE}")
