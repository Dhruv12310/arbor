# Arbor TreeSearch Accuracy Improvement — Full Context & Plan Request

## What is Arbor?
Arbor is a **vectorless RAG system** that uses tree-structured document indexing instead of embeddings.
Instead of vector search, a fine-tuned LLM (TreeSearch model) reads the document's tree structure and selects which nodes (sections) contain the answer to a question.

The pipeline has two models:
- **TreeGen**: Reads a PDF and generates a hierarchical tree structure (titles, page ranges, node IDs)
- **TreeSearch**: Given the tree + a question, outputs `{"thinking": "...", "node_list": ["0001", "0042"]}` — the relevant node IDs

**This document is about improving the TreeSearch model.**

---

## The Model
- **Base**: Qwen/Qwen2.5-3B-Instruct
- **Task**: Given a JSON tree structure + question, output JSON with relevant node IDs
- **Current best**: `TStark12310/arbor-treesearch-3b` (v1) — 43% exact match, F1=0.454
- **Training framework**: QLoRA (4-bit), LoRA r=16, alpha=32
- **Hardware**: Google Colab Pro A100 (40GB VRAM)

---

## Training Data
- **Source 1**: 2716 arXiv scientific paper PDFs — tree generated, QA pairs extracted
- **Source 2**: 6 FinanceBench 10-K/10-Q financial documents — tree generated, QA pairs extracted
- **Total training**: 2722 examples in `treesearch_train.jsonl`
- **Eval set**: 301 examples in `treesearch_eval.jsonl`
- **Format**: Each example has `messages` with system + user (tree + question) + assistant (JSON node_list)

### Token length distribution of training data (estimated):
| Max Length | % of data that fits |
|------------|---------------------|
| 2048       | 1%                  |
| 4096       | 11%                 |
| 8192       | 48%                 |
| 16384      | 85%                 |

The median training example is ~8700 tokens. This is critical.

---

## Full History of Training Attempts

### Phase 0 — Early experiments (expensive)
- Used Anthropic Claude + Groq (Llama) to generate training data
- Cost was very high (~$3+ per run), unsustainable
- Switched to Gemini 2.5 Flash Lite (free tier, 4M TPM) for tree generation

### Phase 1 — First proper training (v1)
- Trained on arXiv papers only (~2716 pairs)
- Used standard transformers + PEFT (no Unsloth)
- MAX_LEN = 2048 (but this was during data generation, not training — training was done differently)
- **Result: 43% exact match, F1 = 0.454** ← best result so far
- Model pushed to: `TStark12310/arbor-treesearch-3b`

### Phase 2 — Added FinanceBench data, continued training (v2) ← MISTAKE
- Added 6 FinanceBench pairs to arXiv data = 2722 total
- Continued fine-tuning from v1 (`arbor-treesearch-3b`)
- Used base `Trainer` from transformers
- **MAX_LEN = 2048** ← critical mistake: only 1% of data fits in 2048 tokens
- 99% of examples were truncated mid-prompt — model learned from corrupted data
- **Result: F1 dropped to ~0.17** — significantly worse than v1

### Phase 3 — Fixed MAX_LEN, switched to Unsloth (v3) ← current
- Went back to base model `arbor-treesearch-3b` (v1)
- Used Unsloth (Flash Attention 2, 2x memory efficient)
- Attempted MAX_LEN=16384 → OOM
- Attempted MAX_LEN=8192 with base Trainer → OOM (needed 9.27GB, had 9.21GB)
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + Unsloth → 8192 fits
- BATCH_SIZE=1, GRAD_ACCUM=16, LR=1e-4, EPOCHS=1
- packing=True (Unsloth)
- **Result so far: F1 ~0.23–0.32 (eval in progress, trending downward)**
- Still bad — 52% of data still truncated at 8192

---

## Current Problems

### 1. Context length mismatch (most critical)
The training data has a median length of ~8700 tokens. Even with MAX_LEN=8192:
- 52% of examples are still truncated
- The tree structure gets cut off mid-way → model sees incomplete input
- Model can't learn proper node selection from truncated trees

### 2. Training data quality
- 2716/2722 examples are from arXiv (scientific papers) — small, well-structured trees
- Only 6 examples from FinanceBench (financial documents) — large, deep trees (400-3000 nodes)
- The eval set likely contains more complex documents than training data

### 3. Only 1 epoch
- 1 epoch may not be enough to properly learn from 2722 examples
- But 3 epochs risks overfitting on small dataset

### 4. Loss was high at training start (1.91 vs 0.24 in v2)
- Unsloth's packing changes loss calculation (more tokens per step)
- May indicate the model is being asked to generalize too hard

### 5. v1 accuracy baseline is unclear
- v1 was eval'd on 1508 examples (different eval set?)
- Current eval is on 301 examples
- Hard to do apples-to-apples comparison

---

## What We Want
- Beat v1's 43% exact match / F1=0.454
- Ideally get to 60%+ exact match
- The model needs to work on both small (arXiv) and large (FinanceBench) document trees
- Must fit on A100 40GB with Unsloth

---

## Constraints
- Hardware: Google Colab Pro A100 (40GB VRAM)
- Framework preference: Unsloth (for memory efficiency)
- Budget: Low — prefer free/cheap options
- Model size: 3B params (can't go larger, needs to run on edge)
- Training data: 2722 examples (generating more is expensive with LLM APIs)

---

## Questions for Claude
1. Given the token length distribution (median 8700 tokens, 85% fits in 16384), and that 16384 OOMs on A100 40GB with Unsloth — what is the best strategy? Options:
   - Filter training data to only keep examples ≤8192 tokens (loses 52% of data)
   - Generate shorter training examples (smaller trees)
   - Use a different training approach that handles long sequences better
   - Use gradient checkpointing more aggressively to fit 16384

2. Is 2722 training examples enough for this task? What's the minimum needed?

3. The model needs to handle trees with 50 nodes (arXiv) to 3000 nodes (FinanceBench 10-K). Should we train separate models or one unified model?

4. What hyperparameters should we try? (LR, epochs, LoRA rank, etc.)

5. Is there a way to restructure the training data so trees fit in smaller context windows? E.g., only include the tree structure (titles + node IDs, no page ranges) to reduce token count?

6. Should we use a different base model? Qwen2.5-7B would be more capable but uses more memory.

7. What's a realistic accuracy target for this task given the data size?

---

## Current Eval Metrics (v3, in progress)
- Progress 10/100: F1 = 0.317
- Progress 20/100: F1 = 0.258
- Progress 30/100: F1 = 0.228
- Progress 40/100: F1 = 0.232
- Final result unknown yet but trending ~0.23–0.25

**v1 baseline**: Exact match 43%, F1 = 0.454
**v2**: F1 ~0.17 (broken by MAX_LEN=2048 truncation)
**v3**: F1 ~0.23–0.25 (better than v2 but worse than v1)

---

## The Ask
Please analyze everything above and give me a **concrete, step-by-step plan** to get from ~0.23 F1 to 0.45+ F1 (matching or beating v1). Include:
- Exactly what changes to make to training config
- Whether to generate more/different training data
- How to handle the context length problem
- What to try first (highest impact, lowest risk)
- What NOT to waste time on
