"""
Diagnose why structure extraction fell to page chunking.
Shows exactly what each strategy sees in the PDF.

Usage:
    python scripts/debug_structure.py --pdf data/financebench/pdfs/3M_2018_10K.pdf
"""
import argparse, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True)
args = parser.parse_args()

doc = pymupdf.open(args.pdf)
total = len(doc)
print(f"\nPDF: {Path(args.pdf).name} | {total} pages\n")

# ── Strategy 1: embedded bookmarks ───────────────────────────────────────────
print("="*60)
print("STRATEGY 1: get_toc() — embedded bookmarks")
print("="*60)
toc = doc.get_toc(simple=True)
if toc:
    print(f"Found {len(toc)} entries:")
    for entry in toc[:20]:
        print(f"  Level {entry[0]} | {entry[1][:60]} | page {entry[2]}")
    if len(toc) > 20:
        print(f"  ... and {len(toc)-20} more")
else:
    print("  EMPTY — no embedded bookmarks in this PDF")

# ── Strategy 2: font sizes ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY 2: font-size heading detection")
print("="*60)
sizes = []
for page in doc:
    for block in page.get_text("dict", flags=0).get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0:
                    sizes.append(round(size, 1))

if sizes:
    counter = Counter(sizes)
    body_size = counter.most_common(1)[0][0]
    threshold = body_size * 1.20
    print(f"  Body text size (most common): {body_size}pt")
    print(f"  Heading threshold (1.20x):    {threshold:.1f}pt")
    print(f"  Top 10 font sizes found: {counter.most_common(10)}")

    # Show what would be detected as headings
    headings_found = []
    seen = set()
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("dict", flags=0).get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if size >= threshold and len(text) > 3 and page_num not in seen:
                        headings_found.append((size, text[:60], page_num))
                        seen.add(page_num)
    print(f"\n  Headings detected ({len(headings_found)} total):")
    for size, text, pg in headings_found[:20]:
        print(f"    p{pg:3d} | {size:.1f}pt | {text}")
else:
    print("  No text spans found")

# ── Strategy 3: raw text of first 5 pages ─────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY 3: raw text — first 5 pages (for regex TOC check)")
print("="*60)
for i in range(min(5, total)):
    page = doc[i]
    text = page.get_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    print(f"\n  --- Page {i+1} ({len(lines)} lines) ---")
    for line in lines[:30]:
        print(f"    {repr(line)}")
    if len(lines) > 30:
        print(f"    ... ({len(lines)-30} more lines)")

doc.close()
