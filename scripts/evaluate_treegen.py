"""
Evaluate TreeGen model output quality.

Loads model using standard transformers (NOT Unsloth — Unsloth has a RoPE
inference bug with Qwen2: RuntimeError: output with shape [1, 28, 1, 128]
doesn't match the broadcast shape [1, 28, 474, 128]).

Usage:
    python scripts/evaluate_treegen.py --model TStark12310/arbor-treegen-7b-v2
    python scripts/evaluate_treegen.py --model TStark12310/arbor-treegen-7b-v2 --doc data/training/2603.08012v1.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent.parent

TREEGEN_SYSTEM = '''You are a document structure analyzer. Given document text with [Page N] markers, extract the hierarchical structure as a JSON tree.

Output format:
{
  "doc_name": "<filename>",
  "structure": [
    {
      "title": "<section title>",
      "node_id": "<4-digit zero-padded ID, e.g. 0001>",
      "start_index": <first page number>,
      "end_index": <last page number>,
      "summary": "<1-2 sentence summary of section content>",
      "nodes": [<child sections, same format>]
    }
  ]
}

Rules:
- node_id: Sequential in depth-first order starting from "0001"
- start_index / end_index: Physical page numbers (1-based) from [Page N] markers
- summary: Specific to THIS section, not the overall document
- nodes: Nested children. Leaf nodes have "nodes": []
- Create proper hierarchy: chapters > sections > subsections
- Page ranges must match where content actually appears'''

BUILTIN_TEST = """[Page 1]
Attention Is All You Need
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
We propose the Transformer, based solely on attention mechanisms.

[Page 2]
1 Introduction
Recurrent neural networks, long short-term memory networks, have been firmly established as state of the art.
Attention mechanisms allow modeling of dependencies without regard to their distance in sequences.

[Page 3]
2 Model Architecture
2.1 Encoder and Decoder Stacks
The encoder maps an input sequence to a sequence of continuous representations.
2.2 Attention
An attention function maps a query and a set of key-value pairs to an output.

[Page 4]
3 Training
3.1 Training Data
We trained on the standard WMT 2014 English-German dataset.
3.2 Results
On the WMT 2014 English-to-German translation task, we achieve 28.4 BLEU.
"""


def count_nodes(nodes):
    total = 0
    for n in nodes:
        total += 1
        total += count_nodes(n.get("nodes", []))
    return total


def max_depth(nodes, d=0):
    mx = d
    for n in nodes:
        if n.get("nodes"):
            mx = max(mx, max_depth(n["nodes"], d + 1))
    return mx


def collect_summaries(nodes, out=None):
    if out is None:
        out = []
    for n in nodes:
        s = n.get("summary", "")
        if s:
            out.append(s)
        collect_summaries(n.get("nodes", []), out)
    return out


def check_page_ranges(nodes):
    issues = 0
    for n in nodes:
        if n.get("start_index", 0) > n.get("end_index", 0):
            issues += 1
        issues += check_page_ranges(n.get("nodes", []))
    return issues


def check_node_ids(nodes, out=None):
    if out is None:
        out = []
    for n in nodes:
        nid = n.get("node_id", "")
        out.append(nid)
        check_node_ids(n.get("nodes", []), out)
    return out


def score_tree(tree_json: str) -> dict:
    """Score a generated tree JSON string."""
    scores = {
        "valid_json": False,
        "has_structure": False,
        "node_count": 0,
        "max_depth": 0,
        "has_summaries": False,
        "summaries_unique": False,
        "page_ranges_valid": False,
        "node_ids_present": False,
        "verdict": "BAD",
    }

    try:
        data = json.loads(tree_json)
        scores["valid_json"] = True
    except Exception:
        # Try to find JSON in response
        start = tree_json.find("{")
        end = tree_json.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(tree_json[start:end])
                scores["valid_json"] = True
            except Exception:
                return scores
        else:
            return scores

    structure = data.get("structure", [])
    scores["has_structure"] = len(structure) > 0

    if not structure:
        return scores

    nodes = count_nodes(structure)
    scores["node_count"] = nodes
    scores["max_depth"] = max_depth(structure)

    summaries = collect_summaries(structure)
    scores["has_summaries"] = len(summaries) > 0
    if summaries:
        scores["summaries_unique"] = len(set(summaries)) >= len(summaries) * 0.8

    scores["page_ranges_valid"] = check_page_ranges(structure) == 0

    node_ids = check_node_ids(structure)
    scores["node_ids_present"] = all(nid for nid in node_ids)

    # Verdict
    good_checks = sum([
        scores["valid_json"],
        scores["has_structure"],
        nodes >= 3,
        scores["max_depth"] >= 1,
        scores["has_summaries"],
        scores["summaries_unique"],
        scores["page_ranges_valid"],
        scores["node_ids_present"],
    ])

    if good_checks >= 7:
        scores["verdict"] = "GOOD"
    elif good_checks >= 5:
        scores["verdict"] = "OKAY"
    else:
        scores["verdict"] = "BAD"

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate TreeGen model quality")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--doc", type=Path, default=None,
                        help="JSON file with document_text + tree (uses built-in test if not provided)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load model with standard transformers (NOT Unsloth)
    print(f"Loading {args.model} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    # Get test document
    if args.doc and args.doc.exists():
        doc = json.loads(args.doc.read_text(encoding="utf-8"))
        doc_text = doc.get("document_text", "")
        page_count = doc.get("page_count", 1)
        # Build paged text heuristically
        if page_count > 1:
            chars_per_page = len(doc_text) // page_count
            parts = []
            for i in range(page_count):
                start = i * chars_per_page
                end = (i + 1) * chars_per_page if i < page_count - 1 else len(doc_text)
                parts.append(f"[Page {i+1}]\n{doc_text[start:end].strip()}")
            input_text = "\n\n".join(parts)
        else:
            input_text = f"[Page 1]\n{doc_text}"
        baseline_tree = doc.get("tree", {})
        doc_name = args.doc.name
    else:
        print("Using built-in 4-page test document (Attention Is All You Need)...")
        input_text = BUILTIN_TEST
        baseline_tree = None
        doc_name = "builtin_test"

    # Truncate if needed
    max_chars = 6000 * 4
    if len(input_text) > max_chars:
        input_text = input_text[:max_chars] + "\n[Document truncated]"

    # Generate
    messages = [
        {"role": "system", "content": TREEGEN_SYSTEM},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating tree...")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"  GENERATED OUTPUT (first 1000 chars)")
    print(f"{'='*60}")
    print(response[:1000])
    if len(response) > 1000:
        print(f"  ... [{len(response)-1000} more chars]")

    # Score
    scores = score_tree(response)

    print(f"\n{'='*60}")
    print(f"  QUALITY SCORES — {doc_name}")
    print(f"{'='*60}")
    print(f"  Valid JSON:          {'YES' if scores['valid_json'] else 'NO'}")
    print(f"  Has structure:       {'YES' if scores['has_structure'] else 'NO'}")
    print(f"  Node count:          {scores['node_count']}")
    print(f"  Max depth:           {scores['max_depth']}")
    print(f"  Has summaries:       {'YES' if scores['has_summaries'] else 'NO'}")
    print(f"  Summaries unique:    {'YES' if scores['summaries_unique'] else 'NO'}")
    print(f"  Page ranges valid:   {'YES' if scores['page_ranges_valid'] else 'NO'}")
    print(f"  Node IDs present:    {'YES' if scores['node_ids_present'] else 'NO'}")
    print(f"")
    print(f"  VERDICT: {scores['verdict']}")
    print(f"{'='*60}")

    # Compare with baseline if available
    if baseline_tree and baseline_tree.get("structure"):
        baseline_structure = baseline_tree["structure"]
        b_nodes = count_nodes(baseline_structure)
        b_depth = max_depth(baseline_structure)
        b_summaries = collect_summaries(baseline_structure)
        print(f"\n  BASELINE COMPARISON:")
        print(f"  {'Metric':<25} {'Baseline':>10} {'Generated':>10}")
        print(f"  {'-'*47}")
        print(f"  {'Node count':<25} {b_nodes:>10} {scores['node_count']:>10}")
        print(f"  {'Max depth':<25} {b_depth:>10} {scores['max_depth']:>10}")
        print(f"  {'Has summaries':<25} {'YES' if b_summaries else 'NO':>10} {'YES' if scores['has_summaries'] else 'NO':>10}")


if __name__ == "__main__":
    main()
