# Arbor

**Vectorless RAG for PDFs — tree-structured document navigation with fine-tuned LLMs.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

No embeddings. No vector databases. Arbor parses any PDF into a hierarchical JSON tree, then navigates that tree level-by-level to find answers — the same technique used by [PageIndex](https://github.com/VectifyAI/PageIndex), rebuilt as open-source with fine-tuned models that run locally.

**TreeSearch v9 results:** F1 = 0.911 | Exact Match = 87.0% | 4.0× improvement over the baseline

---

## Install

```bash
pip install arbor-rag
```

---

## Quickstart

```python
import asyncio, arbor

provider = arbor.GroqProvider()   # free tier — set GROQ_API_KEY

async def main():
    response = await arbor.query("report.pdf", "What are the conclusions?", provider)
    print(response.answer)

asyncio.run(main())
```

### Streaming (real-time progress)

```python
async def main():
    async for event in arbor.query_stream("report.pdf", "What is the revenue?", provider):
        if isinstance(event, arbor.NavigatingEvent):
            print(f"Level {event.level}: {event.section_titles}")
        elif isinstance(event, arbor.NodeFoundEvent):
            print(f"Found: {event.title} (pages {event.page_range})")
        elif isinstance(event, arbor.AnswerEvent):
            print(f"\nAnswer: {event.text}")
        elif isinstance(event, arbor.ErrorEvent):
            print(f"Error: {event.message}")
```

---

## Providers

```python
# Groq — free tier, fastest to start
provider = arbor.GroqProvider()                            # needs GROQ_API_KEY

# Gemini — fast, cheap
provider = arbor.GeminiProvider()                          # needs GEMINI_API_KEY

# Ollama — 100% local, zero cost, zero data sharing
provider = arbor.OllamaProvider(model="qwen2.5:7b")        # needs: ollama pull qwen2.5:7b

# OpenAI
provider = arbor.OpenAIProvider(model="gpt-4o-mini")       # needs OPENAI_API_KEY

# Claude
provider = arbor.AnthropicProvider(model="claude-haiku-4-5-20251001")  # needs ANTHROPIC_API_KEY

# Any OpenAI-compatible endpoint
provider = arbor.OpenAICompatibleProvider(
    base_url="https://your-endpoint/v1",
    api_key="your-key",
    model="your-model",
)
```

---

## Budget Controls

Set hard limits on every query so costs and latency are predictable:

```python
config = arbor.ArborConfig(
    max_hops=3,             # stop navigating after 3 tree levels
    max_nodes_searched=50,  # stop after examining 50 nodes total
    max_cost_usd=0.10,      # hard stop at $0.10 estimated spend
    timeout_sec=30.0,       # hard stop after 30 seconds wall-clock
    max_retries_on_bad_json=2,  # retry malformed model output up to 2 times
)

try:
    response = await arbor.query("report.pdf", "What is revenue?", provider, config=config)
except arbor.BudgetExceededError as e:
    print(f"Hit limit: {e}")
    print(f"Partial results: {e.partial_nodes}")
```

---

## Fine-Tuned Models

Arbor ships with two purpose-built models fine-tuned from Qwen2.5:

| Model | Task | Base | Size | Accuracy |
|-------|------|------|------|----------|
| `arbor-treegen` | PDF → JSON tree | Qwen2.5-7B | 7B | — |
| `arbor-treesearch` | Tree navigation | Qwen2.5-3B | 3B | F1=0.911 |

### With Ollama (recommended for local use)

```bash
# Start Ollama with memory optimizations (reduces TreeGen RAM by ~10 GB)
OLLAMA_FLASH_ATTENTION=true OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve

# Register the models
ollama create arbor-treegen    -f models/Modelfile.treegen
ollama create arbor-treesearch -f models/Modelfile.treesearch
```

```python
# Tree generation with fine-tuned TreeGen
tree_provider = arbor.OllamaProvider(model="arbor-treegen")
tree = await arbor.generate_tree("document.pdf", provider=tree_provider)

# Multi-hop search with fine-tuned TreeSearch v9
search_provider = arbor.OllamaProvider(model="arbor-treesearch")
result = await arbor.search_tree(
    tree, "What are the key findings?",
    provider=search_provider,
    multihop=True,
)
```

**TreeGen RAM usage at 16K context:**

| Setup | VRAM |
|-------|------|
| Default | ~15 GB |
| + Flash attention | ~7 GB |
| + Flash attention + Q8 KV cache | ~5 GB |

---

## MCP Server (Claude Code / Cursor / Claude Desktop)

Arbor exposes its pipeline as MCP tools so any MCP-compatible client can call it directly — no integration code needed.

### Setup

```bash
pip install mcp
```

Add `.mcp.json` to your project root (already included in this repo):

```json
{
  "mcpServers": {
    "arbor": {
      "command": "python",
      "args": ["-m", "arbor.mcp_server"],
      "env": { "GEMINI_API_KEY": "${GEMINI_API_KEY}" }
    }
  }
}
```

Restart Claude Code. The `arbor` server will appear in `/mcp`.

### Tools available

| Tool | What it does |
|------|-------------|
| `query_document` | Full pipeline: PDF → answer with citations |
| `generate_tree` | PDF → hierarchical JSON tree |
| `search_tree` | Tree + question → relevant node IDs |

### Usage in Claude Code

```
> Use the arbor tool to find what 3M's capital expenditure was in FY2022

→ Level 1: exploring ['Business Overview', 'Financial Statements', 'Legal Proceedings']
✓ Found: Capital Expenditures (pages 48-50)

Answer: 3M's FY2022 capital expenditure was $1.7 billion...
```

### HTTP/SSE mode (for web clients)

```bash
python -m arbor.mcp_server --http --port 8080
```

---

## How It Works

```
PDF
 ↓
[TreeGen — Qwen2.5-7B fine-tune]
 ↓
Hierarchical JSON tree (chapters → sections → subsections, with page ranges and node IDs)
 ↓
[TreeSearch — Qwen2.5-3B fine-tune, multi-hop]
 Each hop: sees 10-20 sections, picks which branch to explore → recurse in parallel
 ↓
Relevant leaf nodes (the specific pages containing the answer)
 ↓
[Answer LLM] — reads only those pages, generates grounded answer with citations
```

### Multi-hop navigation example

```
Question: "What is the encoder architecture?"

Hop 1  →  [Introduction] [Methods] [Results] [Appendix]
           Model picks: Methods

Hop 2  →  [2.1 Data] [2.2 Model Architecture] [2.3 Training]
           Model picks: Model Architecture

Hop 3  →  [2.2.1 Encoder] [2.2.2 Decoder]
           Model picks: Encoder  ← leaf node, done

Answer pulled from pages 26–32.
```

Each hop sees ≤20 sections — fits in ~500 tokens. Zero truncation possible. Works on documents with thousands of nodes.

### Three tree-generation modes

| Mode | Trigger | Strategy |
|------|---------|----------|
| `TOC_WITH_PAGES` | TOC with page numbers found | Extract → verify → fix mismatches |
| `TOC_NO_PAGES` | TOC without page numbers | Extract → locate pages via LLM scan |
| `NO_TOC` | No TOC detected | Generate full structure from content |

---

## Configuration

```python
config = arbor.ArborConfig(
    # Tree generation
    toc_check_pages=20,          # pages to scan for TOC
    max_pages_per_node=10,       # max pages per leaf before subdivision
    max_tokens_per_node=20000,   # max tokens per chunk
    add_summaries=True,          # LLM-generated node summaries
    add_node_ids=True,           # assign 4-digit node IDs (0001, 0002, ...)
    add_node_text=False,         # include raw page text in output (large)
    add_doc_description=False,   # one-sentence document description
    max_concurrent_llm_calls=5,  # asyncio.Semaphore limit
    overlap_pages=1,             # pages of overlap between chunks

    # Budget controls (new in v2)
    max_hops=5,                  # max tree levels to navigate
    max_nodes_searched=100,      # hard stop on nodes examined (0 = no limit)
    max_cost_usd=0.50,           # estimated API spend limit (0 = no limit)
    timeout_sec=120.0,           # wall-clock timeout in seconds (0 = no limit)
    max_retries_on_bad_json=2,   # retries on malformed TreeSearch output
)
```

---

## Streaming Events

`query_stream()` yields typed event objects in this order:

```python
TreeLoadedEvent(node_count=47, page_count=120)
NavigatingEvent(level=1, exploring_ids=["0001","0002","0003"], section_titles=[...])
NavigatingEvent(level=2, ...)
NodeFoundEvent(node_id="0015", title="Capital Expenditures", page_range="48-50")
AnswerEvent(text="...", citations=[...], nodes_examined=12)

# On budget exceeded:
ErrorEvent(message="Timeout after 30.0s", partial_nodes=["0015"])
```

---

## Comparison

| | Arbor | PageIndex | Vector RAG |
|---|---|---|---|
| Indexing | Tree (LLM/fine-tuned) | Tree (GPT-4) | Chunks + embeddings |
| Search | Multi-hop tree navigation | LLM navigation | ANN similarity |
| Vector DB | None | None | Required |
| Works offline | Yes (Ollama) | No | Depends |
| Understands structure | Yes | Yes | No |
| Open source | Yes | Yes | Varies |
| Provider-agnostic | Yes | No (OpenAI only) | Varies |
| MCP server | Yes | No | No |
| Budget controls | Yes | No | No |
| Streaming | Yes | No | Varies |
| F1 (retrieval) | 0.911 | N/A (closed) | ~0.60–0.75 |

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/generate_training_pair.py` | Process a single PDF → training example |
| `scripts/build_v6_dataset.py` | Build v6 training dataset from arXiv PDFs |
| `scripts/build_multihop_dataset.py` | Convert trees + Q&A into multi-hop training examples |
| `scripts/check_multihop_quality.py` | Validate training data quality before training |
| `scripts/collect_domain_pdfs.py` | Collect PDFs from 7 new domains (legal, healthcare, etc.) |
| `scripts/generate_domain_trees.py` | Batch-generate trees for domain PDFs |
| `scripts/finetune_treesearch_v7.py` | QLoRA fine-tune Qwen2.5-3B for TreeSearch (Colab-ready) |
| `scripts/finetune_treegen_v2.py` | QLoRA fine-tune Qwen2.5-7B for TreeGen |
| `scripts/colab_treegen_vllm_v2.py` | vLLM inference pipeline for TreeGen on Colab |
| `scripts/evaluate_treegen.py` | Benchmark TreeGen quality vs baseline |
| `scripts/audit_training_data.py` | Audit training data for quality issues |
| `scripts/normalize_tree.py` | Normalize tree JSON format |
| `scripts/postprocess_trees.py` | Postprocess generated trees |
| `scripts/format_treegen_v2.py` | Format TreeGen training data |

---

## Training History

Arbor's TreeSearch model was built from scratch and iterated 8 times to reach production accuracy.

### The regression and fix (v1–v7)

TreeSearch v1 achieved F1=0.454. Versions v2–v6 consistently regressed to F1=0.23 despite more data and training time. Root cause:

> FinanceBench 10-K filings have 3,000+ nodes. The full tree exceeded the 8,192-token context window. **52% of training examples were silently truncated** — the model learned from broken, cut-off data. Loss curves looked normal.

The fix was a complete task reformulation: instead of showing the full tree at once (one-shot), navigate level-by-level with 10–20 nodes per call (multi-hop). Maximum input per call: ~500 tokens. Zero truncation possible.

### Results

| Version | F1 | Exact Match | Training Examples | Key Change |
|---------|-----|-------------|-------------------|------------|
| v1 | 0.454 | — | ~500 | Initial one-shot approach |
| v2–v6 | 0.23 | — | 2,700 | More data, same broken format |
| **v7** | **0.851** | **80.9%** | **1,217** | Multi-hop reformulation |
| **v8** | **0.875** | **83.5%** | **1,353** | FinanceBench real Q&A added |
| **v9** | **0.911** | **87.0%** | **3,063** | 7 domain sectors added (legal, healthcare, government, energy, insurance, real estate, automotive) |

### Domain coverage (v9)

Training data spans 2 primary domains + 7 expansion domains (191 trees):

| Domain | Source | Trees |
|--------|--------|-------|
| AI/ML research | arXiv | 141 |
| Finance (10-K filings) | EDGAR/FinanceBench | 7 + real Q&A |
| Legal | arXiv + EDGAR | 27 |
| Healthcare/Clinical | PubMed | 25 |
| Government/Policy | Public sources | 26 |
| Energy | arXiv + EDGAR | 28 |
| Insurance | EDGAR | 29 |
| Real Estate | EDGAR | 28 |
| Automotive | arXiv | 28 |

### Fine-tuning setup (reproducible on Google Colab Pro)

- **Base model:** Qwen2.5-3B-Instruct
- **Method:** QLoRA, 4-bit quantization (Unsloth)
- **LoRA config:** r=32, alpha=64, RSLoRA=True, target all attention + MLP layers
- **Training:** 3 epochs, batch size 16 (4 × 4 grad accum), LR=2e-4, cosine schedule, warmup_steps=57, max_grad_norm=0.3
- **Hardware:** NVIDIA A100 40GB (Google Colab Pro)
- **Training examples:** 3,063 train / 341 eval
- **Trainable params:** 59.8M of 3.1B (1.9%)
- **Context window:** 2,048 tokens (all examples fit, zero truncation)

---

## Key Insights

1. **Task formulation matters more than model size.** Switching from one-shot full-tree to multi-hop level-by-level navigation (same 3B model, same data volume) produced a 3.7× F1 improvement.

2. **Silent truncation is a silent killer.** Training pipelines that silently truncate look normal — loss decreases, no errors — but the model learns from broken data. Always verify examples fit within the context window.

3. **Small models with bounded tasks outperform large models with unbounded ones.** A 3B model deciding "which of these 10–20 sections is relevant?" outperforms general-purpose retrieval at a fraction of the cost.

4. **Data quality > data quantity.** 1,217 verified examples produced F1=0.851. Previous versions used 2,700+ examples but achieved F1=0.23 due to truncation.

---

## Documentation

| File | Contents |
|------|----------|
| [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) | Deep-dive explanation of all 5 production features for learners |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Technical spec for the 5-feature production upgrade |
| [`PORTFOLIO.md`](PORTFOLIO.md) | Project writeup with full training history and benchmarks |
| [`ACCURACY_IMPROVEMENT_PLAN.md`](ACCURACY_IMPROVEMENT_PLAN.md) | Plan for reaching F1=0.90+ with v9 |
| [`CHUNKED_TREEGEN_ARCHITECTURE.md`](CHUNKED_TREEGEN_ARCHITECTURE.md) | Architecture for chunked tree generation on very large documents |

---

## Contributing

Contributions welcome. Open an issue before submitting large changes.

```bash
git clone https://github.com/Dhruv12310/arbor
cd arbor
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — Dhruv Bhatt, 2026
