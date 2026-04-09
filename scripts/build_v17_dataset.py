#!/usr/bin/env python
# build_v17_dataset.py — Assemble v17_train.jsonl from all data sources.
# Run: python scripts/build_v17_dataset.py
# Output: data/financebench/v17_train.jsonl

import json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FB   = REPO / "data" / "financebench"

# (path, multiplier, label)
SOURCES = [
    (FB / "diverse_train.jsonl",         1,  "diverse"),
    (FB / "structdirect_train.jsonl",    8,  "structdirect"),
    (FB / "dagger_v17_targeted.jsonl",  40,  "dagger_v17"),
    (FB / "dagger_v15_targeted.jsonl",  40,  "dagger_v15"),
    (FB / "v17_success_replay.jsonl",   30,  "replay_v17"),
    (FB / "v14b_success_replay.jsonl",  20,  "replay_v14b"),
]

OUT = FB / "v17_train.jsonl"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Loading sources...")
    all_rows = []
    breakdown = []

    for path, mult, label in SOURCES:
        if not path.exists():
            sys.exit(f"\nERROR: missing file — {path}\nRun the corresponding gen_* script first.")
        rows = load(path)
        effective = len(rows) * mult
        all_rows.extend(rows * mult)
        breakdown.append((label, len(rows), mult, effective))
        print(f"  {label:<20} {len(rows):>6} raw  ×{mult:<3} = {effective:>7}")

    random.seed(42)
    random.shuffle(all_rows)

    total = len(all_rows)
    print(f"\n  {'-'*52}")
    print(f"  {'TOTAL':<20} {'':>6}       {'':>4}   {total:>7}")
    print(f"\nMix breakdown:")
    for label, raw, mult, eff in breakdown:
        print(f"  {label:<20} {eff/total*100:>5.1f}%")

    # Sanity check: verify all rows have expected schema
    bad = [i for i, r in enumerate(all_rows) if "messages" not in r or len(r["messages"]) != 3]
    if bad:
        print(f"\nWARNING: {len(bad)} rows have unexpected schema (first: row {bad[0]})")
    else:
        print(f"\nSchema check: OK — all {total} rows have 3-message format")

    print(f"\nWriting {OUT.name}...")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Saved {total:,} rows to {OUT}")
    print(f"\nUpload to Colab Drive before training:")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
