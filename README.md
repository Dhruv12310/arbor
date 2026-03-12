# Arbor

**Vectorless RAG for PDFs — tree-structured document indexing with LLM reasoning.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

No embeddings. No vector databases. Arbor builds a semantic tree of your document and uses an LLM to navigate it — the same technique used by [PageIndex](https://github.com/VectifyAI/PageIndex).

---

## Install

```bash
pip install arbor-rag
```

---

## 5-line quickstart

```python
import asyncio, arbor

provider = arbor.GroqProvider()   # free tier at console.groq.com

async def main():
    response = await arbor.query("paper.pdf", "What are the conclusions?", provider)
    print(response.answer)

asyncio.run(main())
```

Set `GROQ_API_KEY` env var and run. That's it.

---

## Providers

Arbor works with any LLM — pick whichever fits your budget and privacy requirements.

```python
# Groq — free tier, fastest to start
provider = arbor.GroqProvider()                          # needs GROQ_API_KEY

# Ollama — 100% local, zero cost, zero data sharing
provider = arbor.OllamaProvider(model="qwen2.5:7b")      # needs: ollama pull qwen2.5:7b

# OpenAI
provider = arbor.OpenAIProvider(model="gpt-4o-mini")     # needs OPENAI_API_KEY

# Claude
provider = arbor.AnthropicProvider(model="haiku")        # needs ANTHROPIC_API_KEY

# Any OpenAI-compatible endpoint
provider = arbor.OpenAICompatibleProvider(
    base_url="https://your-endpoint/v1",
    api_key="your-key",
    model="your-model",
)
```

---

## Fine-Tuned Models (Coming Soon)

Arbor includes purpose-built models for tree generation and tree search,
fine-tuned from Qwen 2.5. These run locally and are 75x cheaper than GPT-4.

### With Ollama (recommended)
```bash
ollama run arbor-ai/treegen-7b
ollama run arbor-ai/treesearch-3b
```
```python
from arbor import OllamaProvider, generate_tree, search_tree

# Use fine-tuned tree generator
tree_provider = OllamaProvider(model="arbor-ai/treegen-7b")
tree = await generate_tree("document.pdf", provider=tree_provider)

# Use fine-tuned tree searcher
search_provider = OllamaProvider(model="arbor-ai/treesearch-3b")
result = await search_tree(tree, "What are the conclusions?", provider=search_provider)
```

### With HuggingFace
```python
from arbor import ArborFineTunedProvider, generate_tree

provider = ArborFineTunedProvider("arbor-ai/treegen-7b")
tree = await generate_tree("document.pdf", provider=provider)
```

### Cost Comparison
| Method | Tree Generation | Per Query | Local |
|--------|----------------|-----------|-------|
| GPT-4 (PageIndex) | ~$1.00 | ~$0.10 | No |
| Groq Free Tier | $0.00 | $0.00 | No |
| Arbor Fine-Tuned | $0.00 | $0.00 | Yes |

---

## How it works

Arbor indexes a PDF in three stages:

```
PDF → Page Extraction → Tree Generation → Search & Answer
```

**1. Page Extraction** — Each page is extracted as tagged text:
```
<physical_index_5>
...page content...
<physical_index_5>
```

**2. Tree Generation** — Arbor detects or generates a table of contents, then builds a hierarchical tree of `TreeNode` objects. Each node has a title, page range, and LLM-generated summary. Three fallback modes handle any document:

| Mode | Trigger | Method |
|------|---------|--------|
| `TOC_WITH_PAGES` | TOC with page numbers found | Extract → verify → fix |
| `TOC_NO_PAGES` | TOC without page numbers | Extract → locate pages via LLM |
| `NO_TOC` | No TOC | Generate structure from content |

**3. Search & Answer** — The LLM navigates the tree using summaries (not full text), selects the relevant nodes, extracts their page ranges, and generates a grounded answer.

```python
# Reuse a pre-built tree across multiple questions
tree = await arbor.generate_tree("paper.pdf", provider=provider)

result = await arbor.search_tree(tree, "What is the methodology?", provider)
# result.nodes → list of relevant TreeNode objects

response = await arbor.query("paper.pdf", "What are the results?", provider, tree=tree)
# response.answer, response.citations, response.context
```

---

## Comparison

| | Arbor | PageIndex | Vector RAG |
|---|---|---|---|
| Indexing | Tree (LLM) | Tree (LLM) | Chunks + embeddings |
| Search | LLM navigation | LLM navigation | ANN similarity |
| Vector DB | None | None | Required |
| Works offline | Yes (Ollama) | No | Depends |
| Understands structure | Yes | Yes | No |
| Open source | Yes | Yes | Varies |
| Provider-agnostic | Yes | No (OpenAI only) | Varies |
| Free tier | Yes (Groq) | No | No |

Arbor is a clean-room, provider-agnostic reimplementation of the PageIndex algorithm with a simple Python API.

---

## Advanced usage

```python
from arbor import ArborConfig

config = ArborConfig(
    toc_check_pages=20,         # pages to scan for TOC
    max_pages_per_node=10,      # max pages per leaf node before subdivision
    max_tokens_per_node=20000,  # max tokens per chunk
    add_summaries=True,         # generate LLM summaries for each node
    add_node_ids=True,          # assign depth-first node IDs (0001, 0002, ...)
    add_node_text=False,        # include raw page text in tree output
    max_concurrent_llm_calls=5, # rate-limit parallel LLM calls
)

tree = await arbor.generate_tree("paper.pdf", provider=provider, config=config)
```

The returned `DocumentTree`:
```python
tree.doc_name        # str — document filename without extension
tree.structure       # list[TreeNode] — top-level nodes
tree.description     # str | None — one-sentence document summary
```

Each `TreeNode`:
```python
node.node_id         # "0001" — depth-first pre-order index
node.title           # "3.2 Multi-Head Attention"
node.start_index     # 5 — first page (1-indexed)
node.end_index       # 6 — last page (inclusive)
node.summary         # "Describes how multi-head attention..."
node.nodes           # list[TreeNode] — children
```

---

## Scripts

Utility scripts for data collection, training, and benchmarking.

| Script | Description |
|--------|-------------|
| `scripts/collect_pdfs.py` | Download public domain PDFs from arXiv (4 categories, up to 200 papers) |
| `scripts/generate_training_pair.py` | Process a single PDF through Arbor and save as training data |
| `scripts/run_batch.py` | Batch-process all PDFs in `data/pdfs/` with resume support |
| `scripts/format_training_data.py` | Convert training pairs to HuggingFace-compatible JSONL datasets |
| `scripts/finetune_treegen.py` | QLoRA fine-tune Qwen2.5-7B for tree generation |
| `scripts/finetune_treesearch.py` | QLoRA fine-tune Qwen2.5-3B for tree search |
| `scripts/upload_to_hf.py` | Upload fine-tuned models to HuggingFace Hub |
| `scripts/benchmark.py` | Benchmark Arbor across providers (time, nodes, cost) |

```bash
# Collect 200 arXiv PDFs
python scripts/collect_pdfs.py --count 50

# Generate training data (needs GROQ_API_KEY)
python scripts/run_batch.py

# Format for fine-tuning
python scripts/format_training_data.py

# Fine-tune (needs GPU)
python scripts/finetune_treegen.py
python scripts/finetune_treesearch.py

# Benchmark providers
python scripts/benchmark.py data/pdfs/2301.12345.pdf
```

---

## Contributing

Contributions welcome. Please open an issue before submitting large changes.

```bash
git clone https://github.com/yourusername/arbor
cd arbor
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
