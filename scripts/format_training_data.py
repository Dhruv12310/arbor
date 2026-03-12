"""
Convert Arbor training pairs into HuggingFace-compatible fine-tuning datasets.

Usage:
    python scripts/format_training_data.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAINING_DIR = ROOT / "data" / "training"
FINETUNE_DIR = ROOT / "data" / "finetune"

TREEGEN_SYSTEM = (
    "You are an expert document structure analyzer. Given document text with page markers, "
    "generate a hierarchical tree structure as JSON."
)
TREESEARCH_SYSTEM = (
    "You are a document retrieval expert. Given a document tree structure and a question, "
    "identify which nodes contain the answer. Reply with JSON: "
    '{"thinking": "...", "node_list": [...]}'
)

QUESTION_TEMPLATES = [
    "What is discussed in the {title} section?",
    "What are the key points of {title}?",
    "Summarize the {title} section.",
    "What does the document say about {title}?",
    "Explain {title} as described in this document.",
]

MAX_USER_TOKENS = 8000


def truncate(text: str, max_tokens: int = MAX_USER_TOKENS) -> str:
    limit = max_tokens * 4  # chars
    return text[:limit] + "\n[truncated]" if len(text) > limit else text


def flatten_nodes(nodes: list[dict]) -> list[dict]:
    """Recursively yield all nodes with a node_id and summary."""
    result = []
    for node in nodes:
        if node.get("node_id") and node.get("summary"):
            result.append(node)
        result.extend(flatten_nodes(node.get("nodes", [])))
    return result


def strip_text(nodes: list[dict]) -> list[dict]:
    """Return nodes without 'text' field (for search input)."""
    out = []
    for n in nodes:
        c = {k: v for k, v in n.items() if k != "text"}
        if "nodes" in c:
            c["nodes"] = strip_text(c["nodes"])
        out.append(c)
    return out


def make_treegen_example(doc: dict) -> dict:
    user_text = truncate(doc["document_text"])
    tree_out = {k: v for k, v in doc["tree"].items() if k != "doc_description"}
    return {"messages": [
        {"role": "system",  "content": TREEGEN_SYSTEM},
        {"role": "user",    "content": user_text},
        {"role": "assistant", "content": json.dumps(tree_out, separators=(",", ":"))},
    ]}


def make_treesearch_examples(doc: dict) -> list[dict]:
    nodes = flatten_nodes(doc["tree"].get("structure", []))
    tree_for_search = {"structure": strip_text(doc["tree"].get("structure", []))}
    tree_json = json.dumps(tree_for_search, separators=(",", ":"))
    examples = []
    for node in nodes:
        title = node["title"]
        node_id = node["node_id"]
        for template in QUESTION_TEMPLATES:
            question = template.format(title=title)
            answer = json.dumps({
                "thinking": f"The question asks about {title}, which corresponds to node {node_id}.",
                "node_list": [node_id],
            }, separators=(",", ":"))
            examples.append({"messages": [
                {"role": "system",  "content": TREESEARCH_SYSTEM},
                {"role": "user",    "content": f"Question: {question}\n\nDocument tree structure:\n{tree_json}"},
                {"role": "assistant", "content": answer},
            ]})
    return examples


def write_split(examples: list[dict], name: str, eval_ratio: float = 0.1) -> tuple[int, int]:
    random.shuffle(examples)
    split = max(1, int(len(examples) * eval_ratio))
    eval_set, train_set = examples[:split], examples[split:]
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for subset, suffix in [(train_set, "train"), (eval_set, "eval")]:
        path = FINETUNE_DIR / f"{name}_{suffix}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in subset))
    return len(train_set), len(eval_set)


def main():
    files = sorted(TRAINING_DIR.glob("*.json"))
    if not files:
        print(f"No training files found in {TRAINING_DIR}")
        return

    treegen, treesearch = [], []
    for f in files:
        doc = json.loads(f.read_text())
        treegen.append(make_treegen_example(doc))
        treesearch.extend(make_treesearch_examples(doc))

    tg_train, tg_eval = write_split(treegen, "treegen")
    ts_train, ts_eval = write_split(treesearch, "treesearch")

    print(f"TreeGen    dataset: {len(treegen)} examples  (train {tg_train} / eval {tg_eval})")
    print(f"TreeSearch dataset: {len(treesearch)} examples (train {ts_train} / eval {ts_eval})")
    print(f"Output: {FINETUNE_DIR}")


if __name__ == "__main__":
    main()
