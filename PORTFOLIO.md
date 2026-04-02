# Arbor — Vectorless RAG with Fine-Tuned Document Navigation Models

## Overview

Arbor is an open-source document intelligence system that replaces traditional vector database retrieval with LLM-driven hierarchical tree navigation. Instead of embedding documents and doing similarity search, Arbor parses any PDF into a structured JSON tree and then navigates that tree level-by-level to find answers — no vector database required.

The project involved building two custom fine-tuned models from scratch, diagnosing and fixing a critical training regression that dropped accuracy by 54%, and ultimately achieving **F1 = 0.851 and Exact Match = 80.9%** on a held-out evaluation set.

---

## The Problem Being Solved

Traditional RAG (Retrieval-Augmented Generation) pipelines work by:
1. Chunking documents into fixed-size text windows
2. Embedding those chunks into a vector space
3. Running cosine similarity search at query time

This approach has known failure modes: semantic similarity does not always correlate with relevance, chunks lose document structure context, and vector databases add infrastructure overhead.

**PageIndex** (the inspiration for Arbor) proved a different approach works — use an LLM to navigate a hierarchical document tree rather than doing embedding search. The result is more precise retrieval that respects document structure (chapters, sections, subsections). However, PageIndex was:

- OpenAI-only (locked to GPT-4)
- Expensive: $2–10 per document
- Slow: 3–15 minutes for a 200-page document
- Non-open-source, 20+ sequential LLM calls per document

**Arbor's goal:** reproduce and improve this approach with open-source fine-tuned models that run locally, for near-zero cost.

---

## Architecture

The system has two components running in a pipeline:

```
PDF → [TreeGen] → JSON Tree → [TreeSearch] → Relevant Node IDs → Answer
```

### TreeGen (Qwen2.5-7B fine-tune)
Converts a raw PDF into a hierarchical JSON tree. Each node contains:
- A section title
- Page range (start/end page)
- Node ID
- Child nodes (subsections)

Example output:
```json
{
  "structure": [
    {
      "node_id": "0001",
      "title": "Introduction",
      "start_index": 1,
      "end_index": 3,
      "nodes": [
        {"node_id": "0002", "title": "Background", ...},
        {"node_id": "0003", "title": "Contributions", ...}
      ]
    },
    ...
  ]
}
```

### TreeSearch (Qwen2.5-3B fine-tune)
Given a question and a document tree, navigates the tree level-by-level to find the relevant nodes. At each level, the model sees the current set of sections (10–20 at a time) and decides which branches to explore next.

Training format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a document tree navigator..."},
    {"role": "user", "content": "Question: What is the capital expenditure for FY2022?\n\nSections at this level:\n[0001] Executive Summary (pages 1-3)\n[0002] Financial Statements (pages 45-89) [has sub-sections]\n..."},
    {"role": "assistant", "content": "{\"thinking\": \"The answer is likely in: Financial Statements\", \"navigate_to\": [\"0002\"]}"}
  ]
}
```

---

## Technical Achievements

### 1. Root Cause Diagnosis — The v1–v6 Regression

The first version of TreeSearch (v1) achieved **F1 = 0.454**. Subsequent versions (v2 through v6) consistently regressed, with the final version hitting **F1 = 0.23** — a 49% drop from v1 despite using more data and training time.

After analysis, the root cause was identified:

> **FinanceBench 10-K filings have 3,000+ nodes with a median of ~8,700 tokens per tree. The model's training context window was 8,192 tokens. 52% of training examples were being silently truncated — the model never saw a complete example and learned broken patterns.**

This is a subtle but devastating failure mode: the loss function continued to decrease, training looked normal, but the model was learning from incomplete, cut-off data.

### 2. The Multi-Hop Navigation Fix (v7)

The fix required rethinking the entire task formulation:

**Before (one-shot):**
- Input: Full document tree (3,000+ nodes) + question
- Output: All relevant node IDs at once
- Problem: Input exceeded context window for 52% of examples

**After (multi-hop):**
- Input: Question + 10–20 nodes at current tree level (~500 tokens max)
- Output: Which nodes to explore next (`navigate_to`)
- Navigate level-by-level until leaf nodes are reached
- Zero truncation possible at any point

This is architecturally identical to how Claude Code's QueryEngine works — a think→call→observe loop that decomposes a complex task into small, bounded decisions.

### 3. Training Data Pipeline

Built an end-to-end data generation pipeline using Gemini 2.5 Flash Lite:

1. **Source documents:** 141 arXiv research papers + 7 FinanceBench 10-K filings + 150 real financial Q&A pairs
2. **Question generation:** Gemini generates 5 domain-specific questions per document
3. **Ground truth labeling:** Gemini identifies which tree nodes contain the answer
4. **Multi-hop conversion:** Each (tree, question, answer_nodes) tuple is converted into 3–4 training examples — one per tree level traversed

**Final dataset:**
- 1,217 training examples
- 136 evaluation examples
- Token distribution: p50 = 218 tokens, p90 = 384 tokens, max = 4,273 tokens
- Zero truncation issues (all examples fit within 2,048 token context)
- Cost to generate: ~$0.15
- Time to generate: ~10 minutes

Quality validation (automated):
- All node IDs in `navigate_to` verified to exist in the presented sections
- Valid JSON in all 1,353 examples
- Non-empty reasoning (`thinking` field) in all examples

### 4. Fine-Tuning Setup

- **Base model:** Qwen2.5-3B-Instruct (fresh base, not the regressed v1–v6 weights)
- **Method:** QLoRA with 4-bit quantization (Unsloth)
- **LoRA config:** r=32, alpha=64, RSLoRA=True
- **Training:** 3 epochs, effective batch size 16, LR=2e-4, cosine schedule
- **Hardware:** NVIDIA A100 40GB (Google Colab Pro)
- **Training time:** 8 minutes
- **Trainable parameters:** 59.8M of 3.1B (1.9%)

Training loss curve:

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 50   | 0.553     | 0.524    |
| 100  | 0.256     | 0.334    |
| 150  | 0.169     | 0.224    |
| 200  | 0.097     | 0.204    |

Clean convergence with no overfitting.

### 5. Evaluation Results

Evaluated on 136 held-out examples (never seen during training):

| Metric | Score |
|--------|-------|
| Precision | 0.854 |
| Recall | 0.857 |
| **F1** | **0.851** |
| **Exact Match** | **0.809 (80.9%)** |

**Improvement over baseline:** F1 0.23 → 0.851 — a **3.7× improvement** over the best previous version.

The model generalizes across document types:
- arXiv research papers (ML, NLP, AI, information retrieval)
- Financial 10-K filings (3M, Activision Blizzard, Adobe, and others)

Example correct predictions:

```
Q: What is the FY2018 capital expenditure for 3M?
Ground truth: ['0013']   Predicted: ['0013']  ✓
Thinking: "The answer is likely in: ITEM 8. Other sections cover unrelated content."

Q: What are the limitations of the current approach (section [0028])?
Ground truth: ['0028']   Predicted: ['0028']  ✓
Thinking: "The answer is likely in: Limitations and future work."

Q: How does the two-stage training method contribute to the overall agent?
Ground truth: ['0003']   Predicted: ['0003']  ✓
Thinking: "The answer is likely in: PROBLEM DEFINITION."
```

---

## Cost and Performance Comparison

| | PageIndex (original) | Arbor v7 |
|--|---------------------|----------|
| Cost per document | $2–10 | ~$0 (local) |
| Tree generation time | 3–15 minutes | 30–60 seconds |
| TreeSearch inference | ~$0.10/query | ~$0 (local Ollama) |
| Model size | GPT-4 (closed) | 3B params (open) |
| Runs locally | No | Yes |
| Context window risk | High (full tree) | None (10–20 nodes/call) |
| F1 Score | N/A (closed) | 0.851 |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Fine-tuning | Unsloth + QLoRA + TRL |
| Base models | Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct |
| Data generation | Google Gemini 2.5 Flash Lite API |
| Local inference | Ollama |
| PDF extraction | PyPDF2 + custom page extractor |
| Training hardware | NVIDIA A100 40GB (Google Colab Pro) |
| Model hosting | HuggingFace Hub (private) |
| Language | Python 3.13 |

---

## Domain Coverage

The v7 model was trained on arXiv papers and financial documents. Active work is expanding coverage to:

- **Legal** — contracts, case law, regulatory filings
- **Healthcare/Clinical** — drug trials, clinical guidelines, medical device docs
- **Government/Policy** — legislation, regulatory frameworks, government reports
- **Energy** — oil & gas technical reports, grid infrastructure
- **Insurance** — policy documents, actuarial reports
- **Real Estate** — lease agreements, property reports
- **Automotive** — engineering specs, safety documentation

191 domain-specific trees have been generated across these 7 sectors (avg 37 nodes/tree) to train the next version.

---

## Key Insights

1. **Task formulation matters more than model size.** Switching from one-shot full-tree selection to multi-hop level-by-level navigation (same 3B model, same data volume) produced a 3.7× accuracy improvement. The architecture of the task is the lever, not scale.

2. **Silent truncation is a silent killer.** Training pipelines that silently truncate inputs look normal — loss goes down, metrics look fine during training — but the model is learning from broken data. Always verify that your training examples fit within the context window.

3. **Small models with focused tasks outperform large models with unfocused tasks.** A 3B model given a precise, bounded task (which of these 10-20 sections leads to the answer?) outperforms general-purpose retrieval at a fraction of the cost.

4. **Data quality over data quantity.** 1,217 high-quality, verified training examples produced F1=0.851. Previous versions used 2,700+ examples but achieved F1=0.23 because those examples were truncated and therefore incorrect.

---

## Links

- GitHub: [github.com/dhruvbhatt/arbor](https://github.com/dhruvbhatt/arbor)
- Model: HuggingFace (private — contact for access)

---

*Built by Dhruv Bhatt — April 2026*
