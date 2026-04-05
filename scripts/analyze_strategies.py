# analyze_strategies.py
# =====================
# Run in Colab after eval_full_84.py to see which strategy each doc used
# and correlate strategies with pass/fail rates.
#
# CELL 1 — Mount Drive + load results
"""
from google.colab import drive
drive.mount('/content/drive')

import json, os, re
from collections import defaultdict

DRIVE_DIR  = "/content/drive/MyDrive/arbor-training-data"
TREE_DIR   = f"{DRIVE_DIR}/sd_trees"
RESULTS    = f"{DRIVE_DIR}/eval_full_84_results.json"

results = json.loads(open(RESULTS, encoding="utf-8").read())
print(f"Loaded {len(results)} results")
"""


# CELL 2 — Infer strategy from each cached tree + build per-doc table
"""
def infer_strategy(tree_data: dict) -> str:
    '''
    Infer which extraction strategy was used based on tree structure patterns.

    Heuristics:
    - page_chunks:    all top-level titles match "Pages N-M"
    - multiline_toc:  top-level titles contain "ITEM" or "PART" patterns
    - embedded_toc:   deep hierarchy (children have children), any title format
    - font_headings:  flat structure, titles don't match ITEM/PART or "Pages N-M"
    - regex_toc:      flat structure, titles match "word ..... N" format (harder to distinguish from font_headings)
    '''
    nodes = tree_data.get("structure", [])
    if not nodes:
        return "unknown"

    top_titles = [n.get("title", "") for n in nodes]

    # page_chunks: all titles are "Pages X-Y" or "Pages X–Y"
    chunk_pattern = re.compile(r'^Pages\s+\d+[-–]\d+$')
    if all(chunk_pattern.match(t) for t in top_titles):
        return "page_chunks"

    # multiline_toc: significant portion of top-level titles start with ITEM or PART
    item_part_re = re.compile(r'^(ITEM\s+\d|PART\s+[IVX])', re.IGNORECASE)
    item_count = sum(1 for t in top_titles if item_part_re.match(t))
    if item_count / len(top_titles) >= 0.3:
        return "multiline_toc"

    # embedded_toc: has deep nesting (any node has grandchildren)
    def has_deep_nesting(nodes, depth=0):
        for n in nodes:
            children = n.get("nodes", [])
            if children and depth >= 1:
                return True
            if has_deep_nesting(children, depth + 1):
                return True
        return False

    if has_deep_nesting(nodes):
        return "embedded_toc"

    # regex_toc vs font_headings — both are flat with arbitrary titles
    # regex_toc tends to have more entries (it finds every TOC line)
    # font_headings is capped at 1 per page
    # Best proxy: if #sections > #pages/3, likely regex_toc
    # (can't determine page count here, so just call both "flat")
    return "font_headings_or_regex_toc"


# Load trees, infer strategy, count sections
doc_info = {}  # doc -> {strategy, n_sections, n_pages}

for fname in sorted(os.listdir(TREE_DIR)):
    if not fname.endswith(".json"):
        continue
    doc = fname[:-5]
    raw = json.loads(open(f"{TREE_DIR}/{fname}", encoding="utf-8").read())

    def count_nodes(nodes):
        return sum(1 + count_nodes(n.get("nodes", [])) for n in nodes)

    n_sections = count_nodes(raw.get("structure", []))
    strategy = infer_strategy(raw)
    doc_info[doc] = {"strategy": strategy, "n_sections": n_sections}

print(f"Loaded {len(doc_info)} trees\n")

# Build per-doc pass/fail from results
doc_results = defaultdict(lambda: {"exact": 0, "tol": 0, "miss": 0, "total": 0})
for r in results:
    doc = r["doc"]
    doc_results[doc]["total"] += 1
    if r["status"] == "EXACT":
        doc_results[doc]["exact"] += 1
    elif r["status"] == "+/-1":
        doc_results[doc]["tol"] += 1
    else:
        doc_results[doc]["miss"] += 1

print(f"{'Doc':<45} {'Strategy':<30} {'Sec':>4}  {'Q':>3}  {'Ex':>3}  {'~1':>3}  {'Ms':>3}")
print("-" * 100)
for doc in sorted(doc_info):
    info = doc_info[doc]
    dr   = doc_results.get(doc, {})
    n    = dr.get("total", 0)
    ex   = dr.get("exact", 0)
    tol  = dr.get("tol", 0)
    ms   = dr.get("miss", 0)
    flag = " ←" if ms > 0 else ""
    print(f"{doc:<45} {info['strategy']:<30} {info['n_sections']:4d}  {n:3d}  {ex:3d}  {tol:3d}  {ms:3d}{flag}")
"""


# CELL 3 — Strategy-level summary: how many questions passed/failed per strategy
"""
from collections import Counter

strategy_stats = defaultdict(lambda: {"exact": 0, "tol": 0, "miss": 0, "total": 0, "docs": set()})

for r in results:
    doc = r["doc"]
    if doc not in doc_info:
        continue
    strat = doc_info[doc]["strategy"]
    strategy_stats[strat]["total"] += 1
    strategy_stats[strat]["docs"].add(doc)
    if r["status"] == "EXACT":
        strategy_stats[strat]["exact"] += 1
    elif r["status"] == "+/-1":
        strategy_stats[strat]["tol"] += 1
    else:
        strategy_stats[strat]["miss"] += 1

print(f"\n{'Strategy':<35} {'Docs':>5}  {'Q':>4}  {'Ex%':>6}  {'Tol%':>6}  {'Miss%':>7}")
print("-" * 70)
for strat, s in sorted(strategy_stats.items(), key=lambda x: -x[1]["miss"]):
    n    = s["total"]
    docs = len(s["docs"])
    ex   = s["exact"]
    tol  = s["tol"]
    ms   = s["miss"]
    print(f"{strat:<35} {docs:5d}  {n:4d}  {ex/n:6.1%}  {tol/n:6.1%}  {ms/n:7.1%}")

# Strategy miss counts (what the user specifically asked for)
print(f"\n=== Strategy Miss Counts ===")
for strat, s in sorted(strategy_stats.items(), key=lambda x: -x[1]["miss"]):
    print(f"  {strat:<35}: {s['miss']:3d} misses / {s['total']:3d} questions  ({s['miss']/s['total']:.1%} miss rate)")
"""


# CELL 4 — Section count vs accuracy: do docs with more sections do better?
"""
import matplotlib.pyplot as plt

x_sections = []
y_recall   = []
labels     = []

for doc in sorted(doc_info):
    dr = doc_results.get(doc, {})
    n  = dr.get("total", 0)
    if n == 0:
        continue
    ex  = dr.get("exact", 0) + dr.get("tol", 0)
    rec = ex / n
    x_sections.append(doc_info[doc]["n_sections"])
    y_recall.append(rec)
    labels.append(doc)

plt.figure(figsize=(12, 6))
plt.scatter(x_sections, y_recall, alpha=0.6)
for i, lbl in enumerate(labels):
    if y_recall[i] == 0 or y_recall[i] == 1.0:
        plt.annotate(lbl[:15], (x_sections[i], y_recall[i]), fontsize=7)
plt.xlabel("Number of sections in extracted tree")
plt.ylabel("Recall (EXACT + ±1) / total questions")
plt.title("Section count vs accuracy per doc")
plt.tight_layout()
plt.show()

# Print docs with 0% recall
zero_recall = [(doc, doc_info[doc]["n_sections"], doc_info[doc]["strategy"])
               for doc in doc_info
               if doc_results.get(doc, {}).get("total", 0) > 0
               and doc_results[doc]["exact"] + doc_results[doc]["tol"] == 0]
print(f"\nDocs with 0% accuracy ({len(zero_recall)}):")
for doc, secs, strat in sorted(zero_recall):
    n = doc_results[doc]["total"]
    print(f"  {doc:<45} {secs:3d} sections  [{strat}]  ({n} questions)")
"""
