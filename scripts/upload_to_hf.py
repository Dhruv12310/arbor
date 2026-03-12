"""Upload fine-tuned Arbor model to HuggingFace Hub.
Usage:
    pip install huggingface_hub && huggingface-cli login
    python scripts/upload_to_hf.py models/treegen-merged    arbor-ai/treegen-7b
    python scripts/upload_to_hf.py models/treesearch-merged  arbor-ai/treesearch-3b
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def build_card(repo_id: str) -> str:
    s = "search" in repo_id.lower()
    base  = "Qwen/Qwen2.5-3B-Instruct" if s else "Qwen/Qwen2.5-7B-Instruct"
    vram  = "~3 GB" if s else "~6 GB"
    task  = "tree search (retrieval)" if s else "tree generation (indexing)"
    usage = ('result = await arbor.search_tree(tree, "What are the conclusions?", provider)'
             if s else 'tree = await arbor.generate_tree("paper.pdf", provider)')
    bench = ""
    for f in sorted((ROOT / "data" / "benchmarks").glob("*.json"), reverse=True)[:1]:
        rows = json.loads(f.read_text()).get("results", [])
        bench = "## Benchmarks\n| Provider | Time | Nodes | Cost |\n|---|---|---|---|\n" + \
            "\n".join(f"| {r['provider']} | {r['gen_time_s']}s | {r['nodes']} | {r['est_cost']} |" for r in rows)
    return "\n".join([
        "---", f"license: mit", f"base_model: {base}", "tags: [arbor, rag, qlora]", "---",
        f"# {repo_id}",
        f"Fine-tuned for **{task}** in [Arbor](https://github.com/Dhruv12310/arbor) vectorless RAG.",
        f"**Base:** {base} | **QLoRA 4-bit** | **VRAM:** {vram} | **Data:** 200 arXiv papers",
        bench,
        "## Usage with Arbor",
        "```python", "import arbor",
        f'provider = arbor.ArborFineTunedProvider("{repo_id}")', usage, "```",
        "## Usage with transformers",
        "```python", "from transformers import AutoModelForCausalLM, AutoTokenizer",
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", device_map="auto")',
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")', "```",
        "## License", "MIT",
    ])


def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned Arbor model to HuggingFace Hub")
    parser.add_argument("model_path", type=Path, help="Local merged model directory")
    parser.add_argument("repo_id", help="e.g. arbor-ai/treegen-7b")
    args = parser.parse_args()
    if not args.model_path.exists():
        raise SystemExit(f"Not found: {args.model_path}")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, exist_ok=True)
    print(f"Uploading to https://huggingface.co/{args.repo_id}")
    api.upload_file(path_or_fileobj=build_card(args.repo_id).encode(), path_in_repo="README.md", repo_id=args.repo_id)
    api.upload_folder(folder_path=str(args.model_path), repo_id=args.repo_id, ignore_patterns=["*.tmp"]); print("Done.")

if __name__ == "__main__":
    main()
