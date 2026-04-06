#!/usr/bin/env python3
# gen_academic_data_local.py
# ==========================
# Standalone version of gen_academic_data.py — runs locally, no Colab needed.
# Generates academic paper navigation training data from 210 real arXiv PDFs.
#
# Usage:
#   cd C:\Users\dhruv\arbor
#   set ANTHROPIC_API_KEY=sk-ant-...
#   python scripts/gen_academic_data_local.py
#
# Output: data/academic_train.jsonl
# Requirements: pip install pymupdf anthropic requests

import json
import os
import random
import sys
import time

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PDF_DIR       = os.path.join(REPO_ROOT, "data", "domain_pdfs")
TREE_CACHE_DIR = os.path.join(REPO_ROOT, "data", "academic_trees")
OUT_FILE      = os.path.join(REPO_ROOT, "data", "academic_train.jsonl")

os.makedirs(PDF_DIR,        exist_ok=True)
os.makedirs(TREE_CACHE_DIR, exist_ok=True)

# ── API key ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    raise ValueError("Set ANTHROPIC_API_KEY environment variable before running:\n"
                     "  Windows: set ANTHROPIC_API_KEY=sk-ant-...\n"
                     "  Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...")

import anthropic as _ant
claude = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)

from arbor.extraction.structure_extractor import extract_structure
from arbor.core.types import DocumentTree

# ── Domain PDFs ───────────────────────────────────────────────────────────────
DOMAIN_PDFS = {
    "automotive": [
        "2603.29165v1","2603.29192v1","2603.29213v1","2603.29226v1","2603.29227v1",
        "2603.29254v1","2603.29272v1","2603.29281v1","2603.29315v1","2603.29332v1",
        "2603.29363v1","2603.29383v1","2603.29391v1","2603.29409v1","2603.29414v1",
        "2603.29419v1","2603.29452v1","2603.29474v1","2603.29499v1","2603.29512v1",
        "2603.29533v1","2603.29560v1","2603.29627v1","2603.29646v1","2603.29708v1",
        "2603.29808v1","2603.29844v1","2603.29882v1","2603.30022v1","2603.30042v1",
    ],
    "energy": [
        "2603.24489v1","2603.24785v1","2603.25192v1","2603.25211v2","2603.25259v1",
        "2603.25287v1","2603.25430v2","2603.26050v1","2603.26081v1","2603.26264v1",
        "2603.26339v1","2603.26578v1","2603.26998v1","2603.27305v1","2603.27442v1",
        "2603.27446v1","2603.27471v1","2603.27816v1","2603.27831v1","2603.28217v1",
        "2603.28286v1","2603.28323v1","2603.28562v1","2603.28945v1","2603.29354v1",
        "2603.29712v1","2603.29745v1","2603.29752v1","2603.29812v1","2603.29851v1",
    ],
    "government": [
        "2603.27717v1","2603.27801v1","2603.27806v1","2603.27994v1","2603.28123v1",
        "2603.28317v1","2603.28371v1","2603.28374v1","2603.28428v1","2603.28434v1",
        "2603.28553v1","2603.28605v1","2603.28679v1","2603.28755v1","2603.28916v1",
        "2603.28928v1","2603.28930v1","2603.29121v1","2603.29141v1","2603.29159v1",
        "2603.29285v1","2603.29288v1","2603.29470v1","2603.29545v1","2603.29559v1",
        "2603.29762v1","2603.29874v1","2603.29935v1","2603.30004v1","2603.30028v1",
    ],
    "healthcare": [
        "2603.24828v1","2603.24846v1","2603.25186v1","2603.25780v1","2603.25821v1",
        "2603.25923v1","2603.26071v1","2603.26254v1","2603.26351v1","2603.26475v1",
        "2603.26483v1","2603.26809v1","2603.26952v1","2603.27114v1","2603.27142v1",
        "2603.27143v1","2603.27589v1","2603.28040v1","2603.28057v1","2603.28115v1",
        "2603.28167v1","2603.28315v1","2603.28334v1","2603.28387v1","2603.28532v1",
        "2603.28589v1","2603.29041v1","2603.29432v1","2603.29709v1","2603.29793v1",
    ],
    "insurance": [
        "2603.11897v1","2603.12767v1","2603.14546v2","2603.15369v1","2603.15839v1",
        "2603.15947v1","2603.15963v1","2603.17792v1","2603.17954v1","2603.18962v1",
        "2603.18969v1","2603.19414v1","2603.19984v1","2603.20319v1","2603.20580v1",
        "2603.22569v1","2603.23842v2","2603.24154v1","2603.24640v1","2603.25217v1",
        "2603.25285v1","2603.25320v1","2603.25350v1","2603.26309v1","2603.26491v1",
        "2603.26901v1","2603.27274v1","2603.27370v1","2603.29430v1","2603.29994v1",
    ],
    "legal": [
        "2602.24130v1","2603.04259v1","2603.04982v2","2603.09492v1","2603.10653v1",
        "2603.10807v1","2603.15705v1","2603.15937v1","2603.16938v1","2603.17444v1",
        "2603.18116v1","2603.18914v1","2603.18964v2","2603.20957v3","2603.21270v1",
        "2603.21280v1","2603.21656v1","2603.21990v1","2603.22714v1","2603.22716v1",
        "2603.22912v1","2603.22920v1","2603.23857v1","2603.24580v1","2603.26420v1",
        "2603.26983v1","2603.27075v1","2603.27094v1","2603.28669v1","2603.29278v1",
    ],
    "real_estate": [
        "2502.16810v5","2502.21141v2","2503.18332v2","2504.13459v1","2504.17113v1",
        "2504.20429v1","2506.00572v1","2506.08131v1","2506.09875v1","2506.15354v1",
        "2507.17624v1","2508.07082v1","2509.02800v1","2509.21460v2","2510.22817v1",
        "2511.01133v1","2511.10808v1","2511.13200v1","2511.18614v1","2512.03088v2",
        "2512.17037v1","2512.23078v1","2601.00914v1","2601.04579v1","2601.08957v2",
        "2602.08631v1","2602.20356v1","2603.03260v1","2603.22880v1","2603.22886v1",
    ],
}

ALL_PDFS = []
for domain, ids in DOMAIN_PDFS.items():
    for arxiv_id in ids:
        ALL_PDFS.append({
            "id":       arxiv_id,
            "domain":   domain,
            "pdf_path": os.path.join(PDF_DIR, f"{arxiv_id}.pdf"),
        })

print(f"Total PDFs: {len(ALL_PDFS)} across {len(DOMAIN_PDFS)} domains")

# ── Download from arXiv (skip if already cached) ──────────────────────────────
print("Downloading missing PDFs from arXiv...")
errors = []
for entry in ALL_PDFS:
    if os.path.exists(entry["pdf_path"]):
        continue
    url = f"https://arxiv.org/pdf/{entry['id']}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(entry["pdf_path"], "wb") as f:
            f.write(r.content)
        print(f"  downloaded {entry['id']} ({entry['domain']}, {len(r.content)//1024}KB)")
        time.sleep(0.4)
    except Exception as e:
        print(f"  FAILED {entry['id']}: {e}")
        errors.append(entry["id"])

print(f"Download done. Errors: {len(errors)}")

# ── Extract StructDirect trees (cached to disk) ───────────────────────────────
print("Extracting trees...")
tree_cache = {}
skipped    = []

for entry in ALL_PDFS:
    pdf_id     = entry["id"]
    cache_path = os.path.join(TREE_CACHE_DIR, f"{pdf_id}.json")

    if not os.path.exists(entry["pdf_path"]):
        skipped.append(pdf_id)
        continue

    if os.path.exists(cache_path):
        raw  = open(cache_path, encoding="utf-8").read()
        tree = DocumentTree.from_dict(json.loads(raw))
        tree_cache[pdf_id] = tree
        continue

    try:
        t0   = time.monotonic()
        tree = extract_structure(entry["pdf_path"])
        ms   = (time.monotonic() - t0) * 1000
        tree_cache[pdf_id] = tree
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)
        def _count(ns): return sum(1 + _count(n.nodes) for n in ns)
        print(f"  OK [{tree.extraction_strategy:20s}] {pdf_id}: {_count(tree.structure)} sections, {ms:.0f}ms")
    except Exception as e:
        print(f"  ERR {pdf_id}: {e}")
        skipped.append(pdf_id)

# Skip page_chunks trees — no semantic titles
useful_trees = {
    k: v for k, v in tree_cache.items()
    if v.extraction_strategy != "page_chunks"
}

from collections import Counter
strat_counts = Counter(v.extraction_strategy for v in tree_cache.values())
print(f"\nTotal extracted      : {len(tree_cache)}")
print(f"Skipped (no PDF)     : {len(skipped)}")
print(f"Useful (non-chunks)  : {len(useful_trees)}")
for s, c in strat_counts.most_common():
    print(f"  {s}: {c}")

# ── Shared constants ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a document tree navigator. Given a question and a list of "
    "document sections at the current level, output JSON specifying which "
    "sections to explore next.\n"
    "Reply format: {\"thinking\": \"...\", \"navigate_to\": [\"XXXX\", ...]}"
)

MAX_NODES_PER_HOP = 20

ACADEMIC_TEMPLATES = [
    (
        "What research problem or gap does this paper address?",
        ["introduction", "motivation", "background", "problem", "overview", "challenge"],
    ),
    (
        "What is the proposed method, model, or approach in this paper?",
        ["method", "approach", "model", "architecture", "framework", "algorithm",
         "system", "design", "proposed", "technique", "network"],
    ),
    (
        "What datasets or benchmarks were used for evaluation?",
        ["dataset", "data", "corpus", "benchmark", "testbed", "collection"],
    ),
    (
        "What were the main quantitative results or performance metrics?",
        ["result", "experiment", "evaluation", "performance", "finding",
         "analysis", "benchmark", "assessment"],
    ),
    (
        "How does this approach compare to existing baselines or prior work?",
        ["comparison", "baseline", "related", "prior", "previous", "versus"],
    ),
    (
        "What are the limitations or future directions of this research?",
        ["limitation", "future", "discussion", "constraint", "open problem"],
    ),
    (
        "What is the conclusion or main takeaway of this paper?",
        ["conclusion", "summary", "discussion", "finding", "takeaway"],
    ),
    (
        "Who provided funding or acknowledgments for this research?",
        ["acknowledgment", "acknowledgement", "funding", "grant", "support", "sponsor"],
    ),
    (
        "What prior or related work does this paper build upon?",
        ["related", "background", "prior", "literature", "survey", "previous work"],
    ),
    (
        "What implementation details or experimental setup were used?",
        ["implement", "training", "setup", "detail", "hyperparameter",
         "configuration", "experiment", "protocol"],
    ),
    (
        "What ablation study or component analysis was conducted?",
        ["ablation", "analysis", "study", "component", "investigation"],
    ),
    (
        "What are the names and institutional affiliations of the authors?",
        [],  # special case: handled by page-1 node
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_node_list(nodes):
    lines = []
    for n in nodes:
        sub = " [has sub-sections]" if n.nodes else ""
        lines.append(f"[{n.node_id}] {n.title} (pages {n.start_index}-{n.end_index}){sub}")
    return "\n".join(lines)


def title_matches(title, keywords):
    tl = title.lower()
    return any(k in tl for k in keywords)


def find_node_covering_page(nodes, page):
    for n in nodes:
        if n.start_index <= page <= n.end_index:
            return n
    return None


def make_example(pdf_id, domain, question, node_list, target_ids, thinking=None):
    if len(node_list) > MAX_NODES_PER_HOP:
        id_to_idx = {n.node_id: i for i, n in enumerate(node_list)}
        positions = [id_to_idx[t] for t in target_ids if t in id_to_idx]
        if positions:
            center = sum(positions) // len(positions)
            half   = MAX_NODES_PER_HOP // 2
            start  = max(0, center - half)
            end    = min(len(node_list), start + MAX_NODES_PER_HOP)
            start  = max(0, end - MAX_NODES_PER_HOP)
            node_list = node_list[start:end]
        else:
            node_list = node_list[:MAX_NODES_PER_HOP]

    if thinking is None:
        id_to_title = {n.node_id: n.title for n in node_list}
        targets_str = ", ".join(id_to_title.get(t, t) for t in target_ids if t in id_to_title)
        thinking = (
            f"The question asks: '{question[:70]}'. "
            f"'{targets_str}' is most likely to contain the answer."
        )

    return {
        "doc":    pdf_id,
        "domain": domain,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Question: {question}\n\nSections at this level:\n{format_node_list(node_list)}"},
            {"role": "assistant", "content": json.dumps({"thinking": thinking, "navigate_to": target_ids})},
        ],
    }


def get_children_of(nodes, parent_id):
    for n in nodes:
        if n.node_id == parent_id:
            return n.nodes
        result = get_children_of(n.nodes, parent_id)
        if result is not None:
            return result
    return None


def add_hops_from_targets(pdf_id, domain, question, top_level, target_ids_top, examples, depth=0):
    if not target_ids_top or depth > 3:
        return
    examples.append(make_example(pdf_id, domain, question, top_level, target_ids_top))
    shuffled = top_level[:]
    random.shuffle(shuffled)
    examples.append(make_example(pdf_id, domain, question, shuffled, target_ids_top))
    for t in target_ids_top:
        children = get_children_of(top_level, t) or []
        if children:
            # At child level: navigate to first child (pass-through)
            add_hops_from_targets(pdf_id, domain, question, children,
                                  [children[0].node_id], examples, depth + 1)


def flatten_tree(nodes):
    result = {}
    for n in nodes:
        if n.node_id:
            result[n.node_id] = n
        result.update(flatten_tree(n.nodes))
    return result


def format_tree_compact(nodes, max_nodes=30, indent="", _count=None):
    """Compact tree for haiku prompt. Uses explicit counter to avoid mutable default."""
    if _count is None:
        _count = [0]
    lines = []
    for n in nodes:
        if _count[0] >= max_nodes:
            lines.append(f"{indent}  ... more sections ...")
            break
        sub = f" [+{len(n.nodes)} sub]" if n.nodes else ""
        lines.append(f"{indent}[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index}){sub}")
        _count[0] += 1
        if n.nodes and _count[0] < max_nodes:
            child_str = format_tree_compact(n.nodes[:4], max_nodes, indent + "  ", _count)
            if child_str:
                lines.extend(child_str.split("\n"))
    return "\n".join(lines)


def validate_label(question, node_id, flat_nodes):
    node = flat_nodes.get(node_id)
    if node is None:
        return False, "hallucinated_node_id"
    if len(node.nodes) > 3:
        return False, "parent_too_coarse"
    metadata_keywords = ["author", "affiliation", "doi", "orcid", "published",
                         "journal", "conference", "funding", "grant", "acknowledgment"]
    if any(kw in question.lower() for kw in metadata_keywords):
        if node.start_index > 5:
            return False, "metadata_not_on_first_pages"
    return True, "ok"


def get_path_to_node(nodes, target_id, path=None):
    if path is None:
        path = []
    for n in nodes:
        if n.node_id == target_id:
            return path + [n]
        result = get_path_to_node(n.nodes, target_id, path + [n])
        if result is not None:
            return result
    return None


def generate_path_examples(pdf_id, domain, question, tree_structure, target_id, thinking, examples):
    path = get_path_to_node(tree_structure, target_id)
    if not path:
        return
    current_level = tree_structure
    for i, node_in_path in enumerate(path):
        target_at_level = [node_in_path.node_id]
        examples.append(make_example(
            pdf_id, domain, question, current_level,
            target_at_level, thinking if i == len(path) - 1 else None,
        ))
        current_level = node_in_path.nodes
        if not current_level:
            break


def extract_first_pages(pdf_path, n_pages=3):
    import pymupdf
    doc  = pymupdf.open(pdf_path)
    text = ""
    for i in range(min(n_pages, len(doc))):
        text += doc[i].get_text()
    doc.close()
    return text[:3500]


# ── Tier 1: Template-based examples (FREE) ────────────────────────────────────
print("\n--- Tier 1: Template examples ---")
tier1_examples = []

for pdf_id, tree in useful_trees.items():
    domain = next((e["domain"] for e in ALL_PDFS if e["id"] == pdf_id), "unknown")
    top    = tree.structure

    for question, keywords in ACADEMIC_TEMPLATES:
        if not keywords:
            # Author affiliations → node covering page 1
            node = find_node_covering_page(top, 1)
            if node:
                add_hops_from_targets(pdf_id, domain, question, top,
                                      [node.node_id], tier1_examples)
            continue

        targets = [n.node_id for n in top if title_matches(n.title, keywords)]
        if not targets:
            continue
        add_hops_from_targets(pdf_id, domain, question, top, targets, tier1_examples)

seen_t1 = set()
tier1_unique = []
for ex in tier1_examples:
    key = ex["messages"][1]["content"]
    if key not in seen_t1:
        seen_t1.add(key)
        tier1_unique.append(ex)

print(f"Tier 1 (templates) — raw: {len(tier1_examples)}, unique: {len(tier1_unique)}")


# ── Tier 2: Haiku auto-labeled + validation ───────────────────────────────────
print("\n--- Tier 2: Haiku auto-labeling ---")
tier2_examples = []
tier2_stats = {
    "generated": 0, "passed": 0,
    "failed_hallucinated_node_id": 0,
    "failed_parent_too_coarse": 0,
    "failed_metadata_not_on_first_pages": 0,
    "api_errors": 0,
}

for i, (pdf_id, tree) in enumerate(useful_trees.items()):
    entry = next((e for e in ALL_PDFS if e["id"] == pdf_id), None)
    if not entry or not os.path.exists(entry["pdf_path"]):
        continue

    domain     = entry["domain"]
    flat_nodes = flatten_tree(tree.structure)
    first_text = extract_first_pages(entry["pdf_path"], n_pages=3)
    compact_tree = format_tree_compact(tree.structure, max_nodes=30)  # fresh counter each call

    prompt = f"""Document: {pdf_id} (domain: {domain})

Tree structure:
{compact_tree}

First pages text (context only):
---
{first_text[:2000]}
---

Generate exactly 3 factual questions whose answers are in SPECIFIC sections of this document.
For each question:
- It must be specific enough to point to ONE particular section, not the whole paper
- The node_id MUST exist in the tree above
- Prefer LEAF nodes (no sub-sections) or nodes with <= 3 sub-sections
- Cover DIFFERENT parts of the document (do NOT pick the same node twice)
- The reasoning should explain WHY that node contains the answer

Return ONLY valid JSON (no markdown):
[
  {{"question": "...", "node_id": "XXXX", "reasoning": "why this node contains the answer"}},
  {{"question": "...", "node_id": "XXXX", "reasoning": "..."}},
  {{"question": "...", "node_id": "XXXX", "reasoning": "..."}}
]"""

    try:
        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw.strip())

        for item in items:
            q      = item.get("question", "").strip()
            nid    = str(item.get("node_id", "")).strip()
            reason = item.get("reasoning", "").strip()
            if not q or not nid:
                continue

            tier2_stats["generated"] += 1
            ok, reason_fail = validate_label(q, nid, flat_nodes)

            if not ok:
                tier2_stats[f"failed_{reason_fail}"] = tier2_stats.get(f"failed_{reason_fail}", 0) + 1
                continue

            tier2_stats["passed"] += 1
            thinking = (
                f"The question asks: '{q[:70]}'. "
                f"According to the document structure, [{nid}] {flat_nodes[nid].title} "
                f"contains this information: {reason[:100]}"
            )
            generate_path_examples(pdf_id, domain, q, tree.structure, nid,
                                   thinking, tier2_examples)

        time.sleep(0.25)

    except Exception as e:
        tier2_stats["api_errors"] += 1
        print(f"  ERR {pdf_id}: {e}")

    if (i + 1) % 30 == 0:
        print(f"  [{i+1}/{len(useful_trees)}] generated: {tier2_stats['generated']}, "
              f"passed: {tier2_stats['passed']}")

seen_t2 = set()
tier2_unique = []
for ex in tier2_examples:
    key = ex["messages"][1]["content"]
    if key not in seen_t2:
        seen_t2.add(key)
        tier2_unique.append(ex)

print("\nTier 2 stats:")
for k, v in tier2_stats.items():
    print(f"  {k}: {v}")
print(f"Tier 2 unique examples: {len(tier2_unique)}")


# ── Combine, deduplicate, save ────────────────────────────────────────────────
all_examples = tier1_unique + tier2_unique
seen_final   = set()
final        = []
for ex in all_examples:
    key = ex["messages"][1]["content"]
    if key not in seen_final:
        seen_final.add(key)
        final.append(ex)

random.seed(42)
random.shuffle(final)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for ex in final:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

from collections import Counter
domain_counts = Counter(ex.get("domain", "unknown") for ex in final)

print(f"\n{'='*60}")
print(f"  ACADEMIC TRAINING DATA — COMPLETE")
print(f"{'='*60}")
print(f"  Tier 1 (templates, free) : {len(tier1_unique):5d} examples")
print(f"  Tier 2 (haiku, specific) : {len(tier2_unique):5d} examples")
print(f"  Total unique             : {len(final):5d} examples")
print(f"\n  By domain:")
for domain, count in sorted(domain_counts.items()):
    print(f"    {domain:<20}: {count}")
print(f"\n  Saved to: {OUT_FILE}")
print(f"{'='*60}")
