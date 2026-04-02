"""
Collect PDFs from 7 missing sectors for Arbor training data expansion.

Sources used:
  - arXiv         : research papers in each domain (same API as collect_pdfs.py)
  - SEC EDGAR     : 10-K filings for energy, insurance, real estate, automotive
  - PubMed Central: open-access healthcare/clinical papers

Usage:
    python scripts/collect_domain_pdfs.py                    # all sectors, 30 each
    python scripts/collect_domain_pdfs.py --sector legal     # one sector only
    python scripts/collect_domain_pdfs.py --count 50         # 50 per sector
    python scripts/collect_domain_pdfs.py --resume           # skip already downloaded
    python scripts/collect_domain_pdfs.py --dry-run          # show counts only

Output:
    data/domain_pdfs/<sector>/          one subfolder per sector
    data/domain_metadata.json           manifest of all downloads
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
OUT_DIR    = DATA_DIR / "domain_pdfs"
META_FILE  = DATA_DIR / "domain_metadata.json"

MAX_PAGES  = 60    # skip PDFs longer than this
MIN_PAGES  = 5     # skip PDFs shorter than this
RATE_LIMIT = 2.0   # seconds between requests

# SEC EDGAR requires email in User-Agent or returns 403
EDGAR_UA   = "ArborResearch/1.0 arbor.research.ai@gmail.com"
ARXIV_UA   = "ArborResearch/1.0 (academic)"

# ── Sector definitions ────────────────────────────────────────────────────────

# arXiv: (category, keyword) pairs — keyword is a single term OR empty for no filter
# Using single keywords (not phrases) avoids over-restrictive AND logic
ARXIV_SECTORS = {
    "legal": [
        ("cs.CY",  "legal"),
        ("cs.AI",  "legal"),
        ("cs.CL",  "contract"),
    ],
    "healthcare": [
        ("cs.LG",   "clinical"),
        ("q-bio.QM",""),
        ("cs.AI",   "medical"),
    ],
    "government": [
        ("cs.CY",   ""),          # whole category — cs.CY is Computers and Society/Policy
        ("econ.GN", "policy"),
        ("cs.AI",   "policy"),
    ],
    "energy": [
        ("eess.SY", "energy"),
        ("cs.LG",   "energy"),
        ("econ.GN", "energy"),
    ],
    "insurance": [
        ("q-fin.RM", ""),
        ("q-fin.MF", ""),
        ("econ.GN",  "insurance"),
    ],
    "real_estate": [
        ("econ.GN",  "housing"),
        ("q-fin.GN", ""),
        ("cs.LG",    "housing"),
    ],
    "automotive": [
        ("cs.RO",   ""),          # whole Robotics category — covers autonomous vehicles
        ("eess.SY", "vehicle"),
        ("cs.CV",   "driving"),
    ],
}

# SEC EDGAR SIC codes per sector (for 10-K filings)
EDGAR_SECTORS = {
    "energy": [
        ("1311", "Crude Petroleum and Natural Gas"),
        ("4911", "Electric Services"),
        ("4922", "Natural Gas Transmission"),
        ("1381", "Drilling Oil and Gas Wells"),
    ],
    "insurance": [
        ("6311", "Life Insurance"),
        ("6321", "Accident and Health Insurance"),
        ("6331", "Fire, Marine and Casualty Insurance"),
        ("6411", "Insurance Agents, Brokers and Service"),
    ],
    "real_estate": [
        ("6512", "Operators of Apartment Buildings"),
        ("6798", "Real Estate Investment Trusts"),
        ("6552", "Land Subdividers and Developers"),
    ],
    "automotive": [
        ("3711", "Motor Vehicles and Passenger Car Bodies"),
        ("5511", "Motor Vehicle Dealers (New and Used)"),
        ("3714", "Motor Vehicle Parts and Accessories"),
    ],
}

# PubMed Central search terms per sector
PMC_SECTORS = {
    "healthcare": [
        "clinical trial randomized controlled",
        "drug efficacy safety pharmacology",
        "medical device evaluation",
        "cancer treatment outcomes",
    ],
    "legal": [
        "legal compliance regulation healthcare",
        "medical malpractice liability",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_page_count(pdf_path: Path) -> int:
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            return len(PyPDF2.PdfReader(f).pages)
    except Exception:
        return 0


def download_pdf(url: str, dest: Path, source: str = "") -> bool:
    try:
        r = requests.get(
            url, timeout=45,
            headers={"User-Agent": ARXIV_UA},
            allow_redirects=True,
        )
        r.raise_for_status()
        if b"%PDF" not in r.content[:16]:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


# ── arXiv collector ───────────────────────────────────────────────────────────

def fetch_arxiv(sector: str, count: int, existing_ids: set, dry_run: bool) -> list[dict]:
    import feedparser

    ARXIV_API = "http://export.arxiv.org/api/query"
    sector_dir = OUT_DIR / sector
    sector_dir.mkdir(parents=True, exist_ok=True)

    configs  = ARXIV_SECTORS.get(sector, [])
    per_conf = max(1, count // max(len(configs), 1))
    results  = []

    for cat, keywords in configs:
        if len(results) >= count:
            break

        query = f"cat:{cat}"
        if keywords:
            # Single keyword — simple AND with category, no over-restrictive phrase matching
            query = f"cat:{cat}+AND+all:{keywords}"

        params = f"search_query={query}&max_results={per_conf * 3}&sortBy=submittedDate&sortOrder=descending"
        url    = f"{ARXIV_API}?{params}"

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"    [warn] arXiv fetch failed for {cat}/{keywords}: {e}", flush=True)
            time.sleep(RATE_LIMIT)
            continue

        for entry in feed.entries:
            if len(results) >= count:
                break

            arxiv_id = entry.id.split("/abs/")[-1].replace("/", "_")
            uid      = f"arxiv_{arxiv_id}"
            if uid in existing_ids:
                continue

            pdf_url = next(
                (l.href for l in entry.links if l.get("type") == "application/pdf"),
                f"https://arxiv.org/pdf/{arxiv_id}",
            )

            if dry_run:
                results.append({"uid": uid, "source": "arxiv", "sector": sector})
                continue

            dest = sector_dir / f"{arxiv_id}.pdf"
            if dest.exists():
                existing_ids.add(uid)
                continue

            time.sleep(RATE_LIMIT)
            ok = download_pdf(pdf_url, dest)
            if not ok:
                continue

            pages = get_page_count(dest)
            if pages < MIN_PAGES or pages > MAX_PAGES:
                dest.unlink()
                continue

            record = {
                "uid":      uid,
                "arxiv_id": arxiv_id,
                "title":    entry.title.replace("\n", " ").strip(),
                "sector":   sector,
                "source":   "arxiv",
                "category": cat,
                "pages":    pages,
                "path":     str(dest),
            }
            results.append(record)
            existing_ids.add(uid)
            print(f"    [arxiv] {arxiv_id}.pdf ({pages}p) — {record['title'][:60]}", flush=True)

        time.sleep(RATE_LIMIT)

    return results


# ── SEC EDGAR collector ───────────────────────────────────────────────────────

def fetch_edgar(sector: str, count: int, existing_ids: set, dry_run: bool) -> list[dict]:
    if sector not in EDGAR_SECTORS:
        return []

    sector_dir = OUT_DIR / sector
    sector_dir.mkdir(parents=True, exist_ok=True)

    EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    FILING_BASE  = "https://www.sec.gov"

    sic_codes = EDGAR_SECTORS[sector]
    per_sic   = max(1, count // len(sic_codes))
    results   = []

    for sic, sic_name in sic_codes:
        if len(results) >= count:
            break

        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&type=10-K&dateb=&owner=include"
            f"&count={per_sic * 3}&search_text=&SIC={sic}&output=atom"
        )

        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": EDGAR_UA})
            r.raise_for_status()
            root = ET.fromstring(r.content)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
        except Exception as e:
            print(f"    [warn] EDGAR SIC={sic} failed: {e}", flush=True)
            time.sleep(RATE_LIMIT)
            continue

        for entry in entries:
            if len(results) >= count:
                break

            filing_index = entry.find("atom:link", ns)
            if filing_index is None:
                continue

            filing_url = filing_index.get("href", "")
            uid        = f"edgar_{sic}_{filing_url.split('/')[-2] if '/' in filing_url else filing_url[-20:]}"

            if uid in existing_ids:
                continue

            if dry_run:
                results.append({"uid": uid, "source": "edgar", "sector": sector})
                continue

            # Get the filing index page to find the 10-K document
            time.sleep(RATE_LIMIT)
            try:
                idx_r = requests.get(
                    filing_url, timeout=30,
                    headers={"User-Agent": EDGAR_UA},
                )
                idx_r.raise_for_status()
                # Find the 10-K document link
                import re
                pdf_match = re.search(
                    r'href="(/Archives/edgar/data/[^"]+\.pdf)"',
                    idx_r.text, re.IGNORECASE
                )
                if not pdf_match:
                    # Try htm version
                    htm_match = re.search(
                        r'href="(/Archives/edgar/data/[^"]+10[kK][^"]*\.htm)"',
                        idx_r.text, re.IGNORECASE
                    )
                    if not htm_match:
                        continue
                    # Skip HTM — we only want PDFs
                    continue

                pdf_url  = FILING_BASE + pdf_match.group(1)
                filename = pdf_match.group(1).replace("/", "_").lstrip("_") + ".pdf"
                dest     = sector_dir / filename

                if dest.exists():
                    existing_ids.add(uid)
                    continue

                ok = download_pdf(pdf_url, dest)
                if not ok:
                    continue

                pages = get_page_count(dest)
                if pages < MIN_PAGES or pages > MAX_PAGES:
                    dest.unlink()
                    continue

                company = entry.find("atom:company-name", ns)
                company_name = company.text if company is not None else sic_name

                record = {
                    "uid":     uid,
                    "company": company_name,
                    "sic":     sic,
                    "sector":  sector,
                    "source":  "edgar",
                    "pages":   pages,
                    "path":    str(dest),
                }
                results.append(record)
                existing_ids.add(uid)
                print(f"    [edgar] {filename} ({pages}p) — {company_name[:50]}", flush=True)

            except Exception as e:
                print(f"    [warn] EDGAR doc fetch failed: {e}", flush=True)

        time.sleep(RATE_LIMIT)

    return results


# ── PubMed Central collector ──────────────────────────────────────────────────

def fetch_pubmed(sector: str, count: int, existing_ids: set, dry_run: bool) -> list[dict]:
    if sector not in PMC_SECTORS:
        return []

    sector_dir = OUT_DIR / sector
    sector_dir.mkdir(parents=True, exist_ok=True)

    PMC_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PMC_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PMC_PDF    = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

    queries  = PMC_SECTORS[sector]
    per_q    = max(1, count // len(queries))
    results  = []

    for query in queries:
        if len(results) >= count:
            break

        try:
            search_r = requests.get(PMC_SEARCH, params={
                "db":      "pmc",
                "term":    query + "[Title/Abstract] AND open access[filter]",
                "retmax":  per_q * 3,
                "retmode": "json",
                "sort":    "relevance",
            }, timeout=30, headers={"User-Agent": EDGAR_UA})
            search_r.raise_for_status()
            ids = search_r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"    [warn] PubMed search failed for '{query}': {e}", flush=True)
            time.sleep(RATE_LIMIT)
            continue

        for pmcid_raw in ids:
            if len(results) >= count:
                break

            pmcid = f"PMC{pmcid_raw}"
            uid   = f"pubmed_{pmcid}"

            if uid in existing_ids:
                continue

            if dry_run:
                results.append({"uid": uid, "source": "pubmed", "sector": sector})
                continue

            pdf_url = PMC_PDF.format(pmcid=pmcid)
            dest    = sector_dir / f"{pmcid}.pdf"

            if dest.exists():
                existing_ids.add(uid)
                continue

            time.sleep(RATE_LIMIT)
            ok = download_pdf(pdf_url, dest)
            if not ok:
                continue

            pages = get_page_count(dest)
            if pages < MIN_PAGES or pages > MAX_PAGES:
                dest.unlink()
                continue

            record = {
                "uid":    uid,
                "pmcid":  pmcid,
                "sector": sector,
                "source": "pubmed",
                "query":  query,
                "pages":  pages,
                "path":   str(dest),
            }
            results.append(record)
            existing_ids.add(uid)
            print(f"    [pubmed] {pmcid}.pdf ({pages}p) — {query[:50]}", flush=True)

        time.sleep(RATE_LIMIT)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_SECTORS = list(ARXIV_SECTORS.keys())


def main():
    parser = argparse.ArgumentParser(description="Collect domain PDFs for Arbor training")
    parser.add_argument("--sector",  default="all", choices=ALL_SECTORS + ["all"],
                        help="Which sector to collect (default: all)")
    parser.add_argument("--count",   type=int, default=30,
                        help="Target PDFs per sector (default: 30)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip already-downloaded files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count available PDFs without downloading")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metadata
    existing_meta: list[dict] = []
    existing_ids:  set[str]   = set()
    if args.resume and META_FILE.exists():
        existing_meta = json.loads(META_FILE.read_text())
        existing_ids  = {e["uid"] for e in existing_meta}

    sectors = ALL_SECTORS if args.sector == "all" else [args.sector]

    print(f"Collecting {args.count} PDFs x {len(sectors)} sectors")
    print(f"Output: {OUT_DIR}\n")

    all_new = []

    for sector in sectors:
        print(f"[{sector.upper()}]", flush=True)

        new_docs = []

        # arXiv (all sectors)
        arxiv_results = fetch_arxiv(sector, args.count, existing_ids, args.dry_run)
        new_docs.extend(arxiv_results)

        # EDGAR (energy, insurance, real_estate, automotive)
        remaining = max(0, args.count - len(new_docs))
        if remaining > 0:
            edgar_results = fetch_edgar(sector, remaining, existing_ids, args.dry_run)
            new_docs.extend(edgar_results)

        # PubMed (healthcare, legal)
        remaining = max(0, args.count - len(new_docs))
        if remaining > 0:
            pubmed_results = fetch_pubmed(sector, remaining, existing_ids, args.dry_run)
            new_docs.extend(pubmed_results)

        all_new.extend(new_docs)
        print(f"  -> {len(new_docs)} PDFs collected for {sector}\n", flush=True)

    if not args.dry_run:
        META_FILE.write_text(json.dumps(existing_meta + all_new, indent=2))

    # Summary
    by_sector = {}
    for doc in all_new:
        by_sector.setdefault(doc["sector"], []).append(doc)

    print("=" * 50)
    print(f"{'SECTOR':<20} {'COUNT':>6}  {'SOURCES'}")
    print("=" * 50)
    total = 0
    for sector in sectors:
        docs    = by_sector.get(sector, [])
        sources = {}
        for d in docs:
            sources[d["source"]] = sources.get(d["source"], 0) + 1
        src_str = ", ".join(f"{v} {k}" for k, v in sources.items())
        print(f"  {sector:<18} {len(docs):>6}  {src_str}")
        total += len(docs)
    print("=" * 50)
    print(f"  {'TOTAL':<18} {total:>6}")
    if args.dry_run:
        print("\n[dry-run] No files downloaded.")
    else:
        print(f"\nMetadata saved to {META_FILE}")


if __name__ == "__main__":
    main()
