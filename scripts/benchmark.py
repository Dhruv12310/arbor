"""
Benchmark Arbor across providers.

Usage:
    python scripts/benchmark.py data/pdfs/2301.12345.pdf
"""

import argparse, asyncio, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import arbor
from arbor.utils.tree_utils import count_nodes
from arbor.extraction.pdf_extractor import get_page_contents

ROOT = Path(__file__).parent.parent
BENCH_DIR = ROOT / "data" / "benchmarks"
QUESTIONS = ["What is the main contribution?", "What methodology is used?", "What are the key results?"]
COSTS = {"groq": (0.05, 0.10), "openai": (0.15, 0.60), "anthropic": (0.25, 1.25), "ollama": (0, 0)}


def tree_depth(nodes, d=1):
    return max((d + tree_depth(n.nodes, d) for n in nodes), default=0)

def est_cost(name, inp, out):
    r, w = next((v for k, v in COSTS.items() if k in name), (0, 0))
    c = (inp * r + out * w) / 1_000_000
    return f"${c:.4f}" if c else "free"

def get_providers():
    out = []
    if os.environ.get("GROQ_API_KEY"):      out.append(("groq",      arbor.GroqProvider()))
    if os.environ.get("OPENAI_API_KEY"):    out.append(("openai",    arbor.OpenAIProvider(model="gpt-4o-mini")))
    if os.environ.get("ANTHROPIC_API_KEY"): out.append(("anthropic", arbor.AnthropicProvider(model="haiku")))
    try:
        p = arbor.OllamaProvider()
        if asyncio.run(p.is_available()): out.append(("ollama", p))
    except Exception: pass
    return out

async def bench(pdf, name, provider):
    config = arbor.ArborConfig(add_node_ids=True, add_summaries=True, add_node_text=False)
    pages = get_page_contents(pdf)
    inp_tokens = sum(p.token_count for p in pages)

    t0 = time.monotonic()
    tree = await arbor.generate_tree(pdf, provider, config)
    gen_time = round(time.monotonic() - t0, 1)

    t1 = time.monotonic()
    for q in QUESTIONS:
        await arbor.search_tree(tree, q, provider)
    avg_search = round((time.monotonic() - t1) / len(QUESTIONS), 2)

    nodes = count_nodes(tree.structure)
    return {
        "provider": name, "model": provider.name,
        "gen_time_s": gen_time, "nodes": nodes, "depth": tree_depth(tree.structure),
        "avg_search_s": avg_search, "est_cost": est_cost(name, inp_tokens, nodes * 80),
    }

def print_table(results):
    print(f"\n{'Provider':<12} {'Model':<35} {'Time':>6} {'Nodes':>6} {'Depth':>6} {'Search':>8} {'Cost':>10}")
    print("-" * 88)
    for r in results:
        print(f"{r['provider']:<12} {r['model']:<35} {r['gen_time_s']:>5}s "
              f"{r['nodes']:>6} {r['depth']:>6} {r['avg_search_s']:>7}s {r['est_cost']:>10}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Arbor across providers")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")

    providers = get_providers()
    if not providers:
        raise SystemExit("No providers available. Set GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama.")

    print(f"Benchmarking: {args.pdf.name}  ({len(providers)} provider(s))")
    results = []
    for name, provider in providers:
        print(f"  Running {name}...", flush=True)
        try:
            results.append(asyncio.run(bench(str(args.pdf), name, provider)))
        except Exception as e:
            print(f"  [error] {name}: {e}")

    print_table(results)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = BENCH_DIR / f"{args.pdf.stem}_{ts}.json"
    out.write_text(json.dumps({"pdf": str(args.pdf), "results": results}, indent=2))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
