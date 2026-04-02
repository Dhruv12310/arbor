"""
Build TreeSearch multi-hop training dataset.

WHY MULTI-HOP?
  One-shot approach (full tree → pick nodes) fails because:
  - FinanceBench 10-Ks have 497 nodes, median ~8700 tokens → 52% of examples truncated
  - Model never sees a complete example, can't learn the task

  Multi-hop (level-by-level) fixes this:
  - Each call: question + 10-20 nodes at current level (~500 tokens)
  - No truncation ever, even for 1000-node trees
  - Simpler task: "which of these 10 sections leads to the answer?"
  - Each tree × question generates 3-4 training examples (one per level)

WHAT THIS SCRIPT DOES:
  1. Load 141 arXiv trees from data/training_v6/
  2. Load 7 FinanceBench trees from data/financebench/trees/
  3. Load 150 FinanceBench Q&A pairs from financebench_open_source.jsonl
  4. For arXiv: generate 5 questions per tree via Gemini (free)
  5. For all questions: use Gemini to identify ground-truth answer nodes
  6. Convert to multi-hop level-by-level training examples
  7. Split 90/10 train/eval
  8. Save to data/finetune/treesearch_multihop_{train,eval}.jsonl

EXPECTED OUTPUT:
  ~141 trees × 5 questions × 3 levels = ~2100 arXiv examples
  ~7 trees × 21 questions × 3 levels = ~440 FinanceBench examples (150 Q&A / 7 docs)
  Total: ~2500 examples, all under 512 tokens

COST: $0 (Gemini 2.0 Flash free: 1500 req/day, 1M tokens/min)
TIME: ~2-3 hours on free tier rate limits

USAGE:
  export GOOGLE_API_KEY=your_key
  python scripts/build_multihop_dataset.py

  # Dry run (first 3 trees only):
  python scripts/build_multihop_dataset.py --dry-run
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# Load .env from project root if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from google import genai
from google.genai import types as genai_types

# ── Config ──────────────────────────────────────────────────────────────────
ARXIV_TREES_DIR      = Path("data/training_v6")
DOMAIN_TREES_DIR     = Path("data/training_domain")   # new sector trees (legal, health, etc.)
FB_TREES_DIR         = Path("data/financebench/trees")
FB_QA_FILE           = Path("data/financebench/financebench_open_source.jsonl")
OUTPUT_DIR           = Path("data/finetune")
PROGRESS_FILE        = Path("data/multihop_progress.json")

QUESTIONS_PER_ARXIV_DOC = 5
TRAIN_SPLIT              = 0.9
GEMINI_MODEL             = "gemini-2.5-flash-lite"
RATE_LIMIT_DELAY         = 0.1   # seconds between calls (4K RPM limit = 0.015s/req, 0.1s is safe)

# ── System prompt (used at inference time too — keep stable) ─────────────────
NAVIGATOR_SYSTEM_PROMPT = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)

# ── Gemini prompts ────────────────────────────────────────────────────────────
QUESTION_GEN_PROMPT = """\
You are given a document tree structure. Generate exactly {n} specific, answerable questions.

Requirements:
- Questions must be answerable from the document (not require external knowledge)
- Cover different parts of the document
- Mix of factual ("what is X"), analytical ("why did X happen"), and comparative ("how does X differ from Y")
- Specific enough that 1-3 sections would contain the answer

Document: {title}

Tree structure:
{tree_summary}

Return JSON only:
{{"questions": ["...", "...", ...]}}"""

NODE_SELECTION_PROMPT = """\
Given a document tree and a question, find which specific nodes contain the answer.

Question: {question}

Full tree (node_id | title | pages):
{tree_summary}

Instructions:
- Identify the 1-3 most specific nodes that DIRECTLY contain the answer
- Prefer deep/specific nodes over broad parent nodes
- If the answer requires multiple nodes, list all of them

Return JSON only:
{{"thinking": "brief reasoning", "answer_nodes": ["node_id1", ...]}}"""


# ── Tree utilities ────────────────────────────────────────────────────────────
def flatten_tree(nodes: list, depth: int = 0) -> list:
    """Return flat list of all nodes with _depth field added."""
    result = []
    for node in nodes:
        result.append({**node, "_depth": depth})
        if node.get("nodes"):
            result.extend(flatten_tree(node["nodes"], depth + 1))
    return result


def tree_to_summary(structure: list, max_nodes: int = 80) -> str:
    """Compact text representation of a tree for Gemini prompts."""
    flat = flatten_tree(structure)[:max_nodes]
    lines = []
    for n in flat:
        indent = "  " * n["_depth"]
        has_children = "[+]" if n.get("nodes") else "   "
        lines.append(
            f"{indent}{has_children} [{n['node_id']}] {n['title']} "
            f"(p{n.get('start_index', '?')}-{n.get('end_index', '?')})"
        )
    if len(flatten_tree(structure)) > max_nodes:
        lines.append(f"  ... ({len(flatten_tree(structure)) - max_nodes} more nodes)")
    return "\n".join(lines)


def get_node_by_id(structure: list, node_id: str) -> dict | None:
    for node in structure:
        if node["node_id"] == node_id:
            return node
        if node.get("nodes"):
            found = get_node_by_id(node["nodes"], node_id)
            if found:
                return found
    return None


def is_ancestor_of_any(node: dict, target_ids: set) -> bool:
    """True if node has any descendant in target_ids."""
    if not node.get("nodes"):
        return False
    for child in node["nodes"]:
        if child["node_id"] in target_ids:
            return True
        if is_ancestor_of_any(child, target_ids):
            return True
    return False


MAX_NODES_PER_LEVEL = 20   # Chunk wider levels into windows of this size


def tree_quality_ok(structure: list) -> bool:
    """
    Filter out trees that are too tiny for any meaningful navigation.
    Requires: at least 8 total nodes.
    Depth-1 trees (flat sections list) are still valid — single-hop selection.
    """
    flat = flatten_tree(structure)
    return len(flat) >= 8


def build_multihop_examples(structure: list, question: str, answer_node_ids: list) -> list:
    """
    Convert (tree, question, answer_nodes) into level-by-level training examples.

    At each level, if any node on the path to an answer exists in the sibling list,
    create a training example: question + siblings → which siblings to explore.

    Wide levels (>MAX_NODES_PER_LEVEL) are chunked into overlapping windows so the
    model never sees more than MAX_NODES_PER_LEVEL sections at once. This also handles
    flat FinanceBench trees that have 150-290 top-level nodes.
    """
    target_set = set(answer_node_ids)
    examples = []
    seen_levels = set()

    def format_sections(nodes: list) -> str:
        lines = []
        for n in nodes:
            sub = " [has sub-sections]" if n.get("nodes") else ""
            lines.append(
                f"[{n['node_id']}] {n['title']} "
                f"(pages {n.get('start_index', '?')}-{n.get('end_index', '?')}){sub}"
            )
        return "\n".join(lines)

    def make_example(nodes: list, navigate_to: list) -> dict:
        selected_titles = [n["title"] for n in nodes if n["node_id"] in navigate_to]
        thinking = (
            f"The answer is likely in: {', '.join(selected_titles)}. "
            f"Other sections cover unrelated content."
        )
        return {
            "messages": [
                {"role": "system", "content": NAVIGATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Sections at this level:\n{format_sections(nodes)}\n\n"
                        f"Which sections should we explore next?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"thinking": thinking, "navigate_to": navigate_to}
                    ),
                },
            ]
        }

    def process_level(nodes: list):
        """Handle one level — chunks wide levels into MAX_NODES_PER_LEVEL windows."""
        if not nodes:
            return

        level_key = tuple(n["node_id"] for n in nodes)
        if level_key in seen_levels:
            return
        seen_levels.add(level_key)

        # Chunk wide levels into windows
        if len(nodes) > MAX_NODES_PER_LEVEL:
            windows = [
                nodes[i : i + MAX_NODES_PER_LEVEL]
                for i in range(0, len(nodes), MAX_NODES_PER_LEVEL)
            ]
        else:
            windows = [nodes]

        navigate_to_all = []  # Collect all nodes to recurse into

        for window in windows:
            navigate_to = []
            for node in window:
                if node["node_id"] in target_set:
                    navigate_to.append(node["node_id"])
                elif is_ancestor_of_any(node, target_set):
                    navigate_to.append(node["node_id"])

            if navigate_to:
                examples.append(make_example(window, navigate_to))
                navigate_to_all.extend(navigate_to)
            elif len(windows) > 1:
                # For wide levels: also add a "none here" negative example
                # (teaches the model to skip irrelevant windows)
                # Only do this for a fraction to avoid class imbalance
                if random.random() < 0.15:
                    negative = make_example(window, [])
                    # Overwrite navigate_to in assistant response for negatives
                    negative["messages"][-1]["content"] = json.dumps(
                        {"thinking": "None of these sections contain the answer.", "navigate_to": []}
                    )
                    examples.append(negative)

        # Recurse into children of selected nodes
        for node in nodes:
            if node["node_id"] in navigate_to_all and node.get("nodes"):
                process_level(node["nodes"])

    process_level(structure)
    return examples


# ── Gemini helpers ────────────────────────────────────────────────────────────
def call_gemini(client, prompt: str, retries: int = 3) -> str | None:
    """Call Gemini with retries and rate limit handling."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            time.sleep(RATE_LIMIT_DELAY)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                wait = 30 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  [gemini error] {err[:120]}")
                return None
    return None


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from Gemini response (handles markdown code blocks)."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def generate_questions(client, tree_summary: str, title: str, n: int = 5) -> list[str]:
    prompt = QUESTION_GEN_PROMPT.format(
        n=n, title=title, tree_summary=tree_summary
    )
    resp = call_gemini(client, prompt)
    parsed = parse_json_response(resp)
    if parsed and "questions" in parsed:
        return [q for q in parsed["questions"] if isinstance(q, str)][:n]
    return []


def get_answer_nodes(client, tree_summary: str, question: str) -> list[str]:
    prompt = NODE_SELECTION_PROMPT.format(
        question=question, tree_summary=tree_summary
    )
    resp = call_gemini(client, prompt)
    parsed = parse_json_response(resp)
    if parsed and "answer_nodes" in parsed:
        return [str(nid) for nid in parsed["answer_nodes"]]
    return []


# ── FinanceBench helpers ──────────────────────────────────────────────────────
def load_financebench_qa() -> dict[str, list]:
    """Load Q&A pairs grouped by doc_name."""
    qa_by_doc = defaultdict(list)
    if not FB_QA_FILE.exists():
        print(f"[warn] FinanceBench QA file not found: {FB_QA_FILE}")
        return qa_by_doc
    with open(FB_QA_FILE) as f:
        for line in f:
            item = json.loads(line)
            doc_name = item.get("doc_name", "")
            if doc_name:
                qa_by_doc[doc_name].append({
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                })
    return qa_by_doc


def load_tree_structure(tree_path: Path) -> list | None:
    """Load tree structure from a JSON file (handles multiple formats)."""
    with open(tree_path) as f:
        data = json.load(f)
    # Handle different formats
    if "structure" in data:
        return data["structure"]
    if "tree" in data:
        t = data["tree"]
        if isinstance(t, dict) and "structure" in t:
            return t["structure"]
        if isinstance(t, list):
            return t
    return None


# ── Progress tracking ─────────────────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": [], "examples": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 3 arXiv trees only")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and start from scratch")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

    client = genai.Client(api_key=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("[reset] Progress cleared.")

    progress = load_progress()
    all_examples = progress.get("examples", [])
    processed = set(progress.get("processed", []))

    # ── 1. Process arXiv + domain trees ──────────────────────────────────────
    arxiv_paths = sorted(ARXIV_TREES_DIR.glob("*.json"))

    # Add domain trees (legal, healthcare, government, energy, insurance, real_estate, automotive)
    domain_paths = []
    if DOMAIN_TREES_DIR.exists():
        domain_paths = sorted(DOMAIN_TREES_DIR.rglob("*.json"))

    all_tree_paths = arxiv_paths + domain_paths

    if args.dry_run:
        arxiv_paths = arxiv_paths[:3]
        domain_paths = domain_paths[:2]
        all_tree_paths = arxiv_paths + domain_paths
        print(f"[dry-run] Processing {len(all_tree_paths)} trees ({len(arxiv_paths)} arXiv + {len(domain_paths)} domain)")
    else:
        print(f"[arXiv] {len(arxiv_paths)} trees found")
        print(f"[domain] {len(domain_paths)} trees found across sectors")

    arxiv_paths = all_tree_paths

    for i, tree_path in enumerate(arxiv_paths):
        doc_id = tree_path.stem
        if doc_id in processed:
            continue

        print(f"[{i+1}/{len(arxiv_paths)}] {doc_id}", end="  ", flush=True)

        structure = load_tree_structure(tree_path)
        if not structure:
            print("skip (no structure)")
            processed.add(doc_id)
            continue

        if not tree_quality_ok(structure):
            flat = flatten_tree(structure)
            print(f"skip (too flat: {len(flat)} nodes, depth {max((n['_depth'] for n in flat), default=0)})")
            processed.add(doc_id)
            continue

        tree_summary = tree_to_summary(structure)

        # Extract a readable title
        with open(tree_path) as f:
            raw = json.load(f)
        title = raw.get("arxiv_id", doc_id)
        if raw.get("document_text"):
            title = raw["document_text"][:100].replace("\n", " ")

        # Generate questions
        questions = generate_questions(client, tree_summary, title, QUESTIONS_PER_ARXIV_DOC)
        if not questions:
            print("skip (no questions generated)")
            processed.add(doc_id)
            continue

        doc_examples = 0
        for q in questions:
            # Get ground-truth answer nodes
            answer_nodes = get_answer_nodes(client, tree_summary, q)
            if not answer_nodes:
                continue

            # Validate node IDs exist in tree
            valid_nodes = [nid for nid in answer_nodes if get_node_by_id(structure, nid)]
            if not valid_nodes:
                continue

            # Build multi-hop examples
            examples = build_multihop_examples(structure, q, valid_nodes)
            all_examples.extend(examples)
            doc_examples += len(examples)

        print(f"{doc_examples} examples from {len(questions)} questions")

        processed.add(doc_id)
        progress["processed"] = list(processed)
        progress["examples"] = all_examples
        save_progress(progress)

    # ── 2. Process FinanceBench trees ─────────────────────────────────────────
    fb_qa = load_financebench_qa()
    fb_paths = sorted(FB_TREES_DIR.glob("*.json"))
    if args.dry_run:
        fb_paths = fb_paths[:1]
    print(f"\n[FinanceBench] {len(fb_paths)} trees, {sum(len(v) for v in fb_qa.values())} Q&A pairs")

    for tree_path in fb_paths:
        doc_id = f"fb_{tree_path.stem}"
        if doc_id in processed:
            continue

        structure = load_tree_structure(tree_path)
        if not structure:
            continue

        # Match tree to Q&A pairs (doc_name matches filename stem)
        doc_name = tree_path.stem
        questions = fb_qa.get(doc_name, [])
        if not questions:
            print(f"  {doc_name}: no Q&A pairs found, skipping")
            processed.add(doc_id)
            continue

        tree_summary = tree_to_summary(structure)
        print(f"  {doc_name}: {len(questions)} questions", end="  ")

        doc_examples = 0
        for qa in questions:
            q = qa["question"]

            answer_nodes = get_answer_nodes(client, tree_summary, q)
            if not answer_nodes:
                continue

            valid_nodes = [nid for nid in answer_nodes if get_node_by_id(structure, nid)]
            if not valid_nodes:
                continue

            examples = build_multihop_examples(structure, q, valid_nodes)
            all_examples.extend(examples)
            doc_examples += len(examples)

        print(f"{doc_examples} examples")

        processed.add(doc_id)
        progress["processed"] = list(processed)
        progress["examples"] = all_examples
        save_progress(progress)

    # ── 3. Split and save ─────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(all_examples)

    n_train = int(len(all_examples) * TRAIN_SPLIT)
    train_set = all_examples[:n_train]
    eval_set  = all_examples[n_train:]

    train_path = OUTPUT_DIR / "treesearch_multihop_train.jsonl"
    eval_path  = OUTPUT_DIR / "treesearch_multihop_eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_set:
            f.write(json.dumps(ex) + "\n")

    # Token length distribution
    if all_examples:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            lengths = sorted(
                len(enc.encode(" ".join(m["content"] for m in ex["messages"])))
                for ex in all_examples
            )
            p50 = lengths[len(lengths) // 2]
            p90 = lengths[int(len(lengths) * 0.9)]
            p99 = lengths[int(len(lengths) * 0.99)]
            print(f"\n  Token lengths — p50: {p50}, p90: {p90}, p99: {p99}")
        except ImportError:
            pass

    print(f"\n{'='*50}")
    print(f"  Total examples : {len(all_examples)}")
    print(f"  Train          : {len(train_set)}  ->  {train_path}")
    print(f"  Eval           : {len(eval_set)}   ->  {eval_path}")
    print(f"{'='*50}")
    print("\nNext step: run scripts/finetune_treesearch_v7.py on Colab")


if __name__ == "__main__":
    main()
