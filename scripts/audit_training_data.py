"""
Audit existing TreeGen training data quality.

Usage:
    python scripts/audit_training_data.py
    python scripts/audit_training_data.py --dirs data/training data/financebench/trees
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent


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
        s, e = n.get("start_index", 0), n.get("end_index", 0)
        if s > e:
            issues += 1
        issues += check_page_ranges(n.get("nodes", []))
    return issues


def audit_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"file": path.name, "error": str(e)}

    tree = data.get("tree", {})
    structure = tree.get("structure", [])

    nodes = count_nodes(structure)
    depth = max_depth(structure)
    summaries = collect_summaries(structure)
    unique_summaries = len(set(summaries))
    page_issues = check_page_ranges(structure)
    provider = data.get("metadata", {}).get("provider", "unknown")
    model = data.get("metadata", {}).get("model", "unknown")
    page_count = data.get("page_count", 0)

    dup_ratio = 1.0 - (unique_summaries / len(summaries)) if summaries else 0.0

    return {
        "file": path.name,
        "nodes": nodes,
        "depth": depth,
        "page_count": page_count,
        "summaries": len(summaries),
        "unique_summaries": unique_summaries,
        "dup_ratio": round(dup_ratio, 2),
        "page_issues": page_issues,
        "provider": provider,
        "model": model,
        "flat": depth == 0 and page_count > 5,
        "too_few_nodes": nodes < 3,
        "has_summaries": len(summaries) > 0,
        "bad_dups": dup_ratio > 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit TreeGen training data quality")
    parser.add_argument(
        "--dirs", nargs="+",
        default=["data/training", "data/financebench/trees"],
        help="Directories to audit",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "treegen_audit_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    all_results = []
    for d in args.dirs:
        dir_path = ROOT / d
        if not dir_path.exists():
            print(f"[skip] {dir_path} does not exist")
            continue
        files = sorted(dir_path.glob("*.json"))
        print(f"\nAuditing {len(files)} files in {dir_path} ...")
        for f in files:
            all_results.append({**audit_file(f), "source_dir": d})

    if not all_results:
        print("No files found.")
        return

    # ── Summary stats ──────────────────────────────────────────────────────────
    total = len(all_results)
    errors = [r for r in all_results if "error" in r]
    valid = [r for r in all_results if "error" not in r]

    flat = sum(1 for r in valid if r["flat"])
    too_few = sum(1 for r in valid if r["too_few_nodes"])
    has_summaries = sum(1 for r in valid if r["has_summaries"])
    bad_dups = sum(1 for r in valid if r["bad_dups"])
    page_issues = sum(1 for r in valid if r["page_issues"] > 0)
    providers = Counter(r["provider"] for r in valid)

    avg_nodes = sum(r["nodes"] for r in valid) / len(valid) if valid else 0
    avg_depth = sum(r["depth"] for r in valid) / len(valid) if valid else 0

    print(f"\n{'='*60}")
    print(f"  TREEGEN DATA AUDIT REPORT")
    print(f"{'='*60}")
    print(f"  Total files:          {total}")
    print(f"  Parse errors:         {len(errors)}")
    print(f"  Valid files:          {len(valid)}")
    print(f"")
    print(f"  Avg nodes per tree:   {avg_nodes:.1f}")
    print(f"  Avg depth:            {avg_depth:.1f}")
    print(f"")
    print(f"  Quality issues:")
    print(f"    Flat trees (depth=0, >5 pages): {flat} ({flat/len(valid)*100:.0f}%)")
    print(f"    Too few nodes (<3):             {too_few} ({too_few/len(valid)*100:.0f}%)")
    print(f"    Has summaries:                  {has_summaries} ({has_summaries/len(valid)*100:.0f}%)")
    print(f"    Duplicate summaries >50%:       {bad_dups} ({bad_dups/len(valid)*100:.0f}%)")
    print(f"    Page range issues:              {page_issues} ({page_issues/len(valid)*100:.0f}%)")
    print(f"")
    print(f"  Provider distribution:")
    for prov, cnt in providers.most_common():
        print(f"    {prov}: {cnt}")
    print(f"{'='*60}")

    # Identify good vs bad files
    good = [r for r in valid if not r["flat"] and not r["too_few_nodes"] and not r["bad_dups"]]
    bad = [r for r in valid if r["flat"] or r["too_few_nodes"] or r["bad_dups"]]
    print(f"\n  USABLE for training:  {len(good)} / {len(valid)}")
    print(f"  Need regeneration:    {len(bad)} / {len(valid)}")

    if bad:
        print(f"\n  Files needing regeneration:")
        for r in bad[:20]:
            reasons = []
            if r["flat"]: reasons.append("flat")
            if r["too_few_nodes"]: reasons.append(f"only {r['nodes']} nodes")
            if r["bad_dups"]: reasons.append(f"dup_ratio={r['dup_ratio']}")
            print(f"    {r['file']}: {', '.join(reasons)}")
        if len(bad) > 20:
            print(f"    ... and {len(bad)-20} more")

    # Save report
    report = {
        "summary": {
            "total": total,
            "errors": len(errors),
            "valid": len(valid),
            "usable": len(good),
            "needs_regen": len(bad),
            "avg_nodes": round(avg_nodes, 1),
            "avg_depth": round(avg_depth, 1),
            "flat": flat,
            "too_few_nodes": too_few,
            "has_summaries": has_summaries,
            "bad_dups": bad_dups,
            "page_issues": page_issues,
            "providers": dict(providers),
        },
        "files": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
