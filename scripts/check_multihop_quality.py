"""
Quick quality check for treesearch_multihop_{train,eval}.jsonl.
Run after build_multihop_dataset.py completes.

Usage: python scripts/check_multihop_quality.py
"""
import json
from pathlib import Path

TRAIN = Path("data/finetune/treesearch_multihop_train.jsonl")
EVAL  = Path("data/finetune/treesearch_multihop_eval.jsonl")

train = [json.loads(l) for l in open(TRAIN)] if TRAIN.exists() else []
eval_ = [json.loads(l) for l in open(EVAL)]  if EVAL.exists()  else []
all_ex = train + eval_

if not all_ex:
    print("No examples found. Run build_multihop_dataset.py first.")
    exit(1)

issues = []
token_lens = []
nav_sizes = []

for i, ex in enumerate(all_ex):
    msgs = ex["messages"]
    if len(msgs) != 3:
        issues.append(f"[{i}] wrong message count: {len(msgs)}")
        continue
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
        issues.append(f"[{i}] wrong roles")
        continue

    user_content = msgs[1]["content"]
    try:
        parsed = json.loads(msgs[2]["content"])
    except json.JSONDecodeError:
        issues.append(f"[{i}] assistant output is not valid JSON")
        continue

    nav = parsed.get("navigate_to")
    if nav is None:
        issues.append(f"[{i}] missing navigate_to")
        continue
    if not isinstance(nav, list):
        issues.append(f"[{i}] navigate_to is not a list")
        continue
    for nid in nav:
        if f"[{nid}]" not in user_content:
            issues.append(f"[{i}] node {nid} in navigate_to not found in sections list")
    if not parsed.get("thinking", "").strip():
        issues.append(f"[{i}] empty thinking field")

    token_lens.append(sum(len(m["content"]) for m in msgs) // 4)
    nav_sizes.append(len(nav))

token_lens.sort()
p50  = token_lens[len(token_lens) // 2]
p90  = token_lens[int(len(token_lens) * 0.9)]
pmax = token_lens[-1]
over = sum(1 for t in token_lens if t > 768)
avg_nav = sum(nav_sizes) / max(len(nav_sizes), 1)
empty_nav = sum(1 for n in nav_sizes if n == 0)

print(f"Examples : {len(all_ex)}  ({len(train)} train / {len(eval_)} eval)")
print(f"Tokens   : p50={p50}  p90={p90}  max={pmax}  over_768={over}")
print(f"navigate_to: avg={avg_nav:.1f} nodes  empty={empty_nav}")
print()
if issues:
    print(f"ISSUES ({len(issues)}):")
    for iss in issues[:15]:
        print(f"  {iss}")
    if len(issues) > 15:
        print(f"  ... and {len(issues)-15} more")
else:
    print("Quality check: PASSED")
