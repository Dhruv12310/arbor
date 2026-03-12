# Arbor Fine-Tuned Models

Two QLoRA-adapted models for running Arbor's pipeline fully locally.

| Model | Base | Task | VRAM |
|-------|------|------|------|
| `treegen-adapter` | Qwen2.5-7B-Instruct | Tree generation | ~6 GB |
| `treesearch-adapter` | Qwen2.5-3B-Instruct | Tree search | ~3 GB |

---

## Option 1: Ollama

Requires the adapter weights in `models/treegen-adapter/` and `models/treesearch-adapter/`.

```bash
# Build and register
ollama create arbor-treegen   -f models/Modelfile.treegen
ollama create arbor-treesearch -f models/Modelfile.treesearch

# Use with Arbor
```

```python
import arbor

tree = await arbor.generate_tree(
    "paper.pdf",
    provider=arbor.OllamaProvider(model="arbor-treegen"),
)
result = await arbor.search_tree(
    tree, "What are the conclusions?",
    provider=arbor.OllamaProvider(model="arbor-treesearch"),
)
```

---

## Option 2: HuggingFace (merged weights)

After running `finetune_treegen.py` / `finetune_treesearch.py`, the merged models are in
`models/treegen-merged/` and `models/treesearch-merged/`. Upload with:

```bash
huggingface-cli upload your-username/arbor-treegen   models/treegen-merged/
huggingface-cli upload your-username/arbor-treesearch models/treesearch-merged/
```

Use from HuggingFace Hub:

```python
provider = arbor.ArborFineTunedProvider("your-username/arbor-treegen")
```

---

## Option 3: ArborFineTunedProvider (local path)

```python
import arbor

# Loads with 4-bit quantization automatically
gen_provider    = arbor.ArborFineTunedProvider("models/treegen-merged")
search_provider = arbor.ArborFineTunedProvider("models/treesearch-merged")

tree = await arbor.generate_tree("paper.pdf", provider=gen_provider)
result = await arbor.search_tree(tree, "What is the methodology?", provider=search_provider)
```

Requires: `pip install transformers bitsandbytes accelerate`

---

## Fine-tuning your own

```bash
# 1. Collect PDFs
python scripts/collect_pdfs.py --count 50

# 2. Generate training data (needs GROQ_API_KEY)
python scripts/run_batch.py

# 3. Format datasets
python scripts/format_training_data.py

# 4. Fine-tune (needs GPU)
python scripts/finetune_treegen.py
python scripts/finetune_treesearch.py
```
