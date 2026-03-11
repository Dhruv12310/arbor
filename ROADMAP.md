# TreeReason — Open-Source Vectorless RAG Engine
## Complete Cost Breakdown + Roadmap (Broke-Founder Edition)

---

## HONEST COST ANALYSIS

### What You Already Have (Free)
- Claude Code Max subscription (you're already paying for this)
- GitHub account (free)
- HuggingFace account (free)
- The PageIndex open-source repo (MIT license — free to study and learn from)
- Your existing ExamPrep codebase (reusable as a demo app)

### Phase 0: Research & Skill Building — $0

| Item | Cost | Why Free |
|------|------|----------|
| Skill Seekers tool | $0 | Open-source, pip install |
| PageIndex repo analysis | $0 | LOCAL mode uses your Claude Code Max |
| Reading PageIndex source code | $0 | It's on GitHub, MIT license |
| Understanding the algorithms | $0 | Your brain + Claude |
| Writing the core library code | $0 | Claude Code Max |

### Phase 1: Build the Library (API-backed) — $0-50

This is the critical insight: you DON'T need to fine-tune models on day one.
Start by building the open-source library that uses existing LLM APIs.

| Item | Cost | Notes |
|------|------|-------|
| Core library development | $0 | You + Claude Code |
| Testing with Claude API | $5-20 | Anthropic API free credits on signup, then pay-as-you-go |
| Testing with free LLM APIs | $0 | HuggingFace Inference API has free tier, Groq has free tier |
| GitHub repo + docs | $0 | GitHub free |
| npm/PyPI package publishing | $0 | Both free |

**Key strategy:** Build the library so it works with ANY LLM backend —
Claude API, OpenAI, Groq (free tier), Ollama (local), HuggingFace Inference.
This way you can test with free APIs during development.

### Phase 2: Generate Training Data — $20-100

Before fine-tuning, you need training data: document→tree pairs.

| Item | Cost | Notes |
|------|------|-------|
| Public domain PDFs | $0 | arXiv papers, Project Gutenberg, government docs |
| Generate trees using your library + cheap API | $20-80 | Use Groq (free tier for Llama 3.1 70B) or Claude Haiku ($0.25/M input tokens) |
| 1,000-2,000 document→tree pairs | ~$50 | This is enough for initial fine-tuning |
| Data cleaning + formatting | $0 | Python scripts |

**Cost optimization:** Use Groq's free tier (Llama 3.1 70B) to generate
most of the training pairs. It's free up to rate limits. You can generate
~50-100 pairs/day on the free tier. In 2-3 weeks you'd have enough data.

### Phase 3: Fine-Tune the Models — $5-50

This is where people think it costs thousands. It doesn't.

| Item | Cost | Notes |
|------|------|-------|
| Google Colab Pro | $10/month | T4/A100 GPU access, enough for QLoRA |
| OR: Kaggle notebooks | $0 | Free 30hrs/week of GPU (T4/P100) |
| OR: HuggingFace ZeroGPU | $0 (free) or $9/mo (Pro) | Free tier gets some GPU quota |
| QLoRA fine-tuning a 7B model | $0-10 | 2-4 hours on a T4 GPU via Colab/Kaggle |
| QLoRA fine-tuning a 3B model | $0-5 | 1-2 hours on a T4 GPU |
| Model hosting on HuggingFace Hub | $0 | Free for public models |

**The QLoRA trick:** You load the base model (Qwen 2.5 7B or Llama 3.1 8B)
in 4-bit precision. This means a 7B model needs only ~8-10GB VRAM.
A free Kaggle T4 (16GB VRAM) handles this easily. Google Colab free tier
has T4 too, though with usage limits. Colab Pro ($10/mo) removes limits.

**Realistic fine-tuning cost: $0-10 per training run.**

### Phase 4: Deployment & Distribution — $0-25/month

| Item | Cost | Notes |
|------|------|-------|
| Library on npm + PyPI | $0 | Free publishing |
| HuggingFace model hosting | $0 | Free for public models |
| Ollama model distribution | $0 | Users run locally |
| Demo app (ExamPrep on Vercel) | $0-20/mo | Vercel free tier or Pro |
| Domain name | $10/year | Optional, but professional |
| Documentation site | $0 | GitHub Pages or Docusaurus |

### Phase 5: Hosted Cloud Service (Optional, for revenue) — $50-100/month

Only do this once you have users and ideally some revenue.

| Item | Cost | Notes |
|------|------|-------|
| GPU server (RunPod/Vast.ai) | $30-80/mo | RTX 3090 instance, runs the fine-tuned model |
| Supabase (metadata + auth) | $25/mo | Pro plan |
| Vercel (API gateway) | $20/mo | Pro plan |

---

## TOTAL COST SUMMARY

```
Phase 0 (Research):           $0
Phase 1 (Build Library):      $0-50        (mostly free API testing)
Phase 2 (Training Data):      $20-100      (can be $0 with Groq free tier + patience)
Phase 3 (Fine-Tuning):        $0-20        (Kaggle free or Colab $10/mo)
Phase 4 (Distribution):       $0-25/month
─────────────────────────────────────────
TOTAL TO LAUNCH:              $20-195

Conservative budget:          ~$100 total to have a working, published
                              open-source vectorless RAG engine with
                              fine-tuned models on HuggingFace
```

Compare this to PageIndex: they charge API credits, require enterprise
pricing for on-prem, and their cloud service has usage-based billing.
You'd be offering the same capability for FREE (self-hosted) or cheap (cloud).

---

## FULL ROADMAP

### PHASE 0 — Research & Deep Analysis (Week 1)
**Cost: $0 | Tools: Skill Seekers + Claude Code**

- [ ] Install Skill Seekers: `pip install skill-seekers`
- [ ] Analyze PageIndex repo: `skill-seekers github --repo VectifyAI/PageIndex --enhance-local`
- [ ] Install the generated skill into Claude Code
- [ ] Use Claude Code (with PageIndex skill) to deeply understand:
  - How `run_pageindex.py` generates trees (the exact prompts, the chunking)
  - What the tree JSON structure looks like (node format, metadata)
  - How tree search works during retrieval (the reasoning prompts)
  - How they handle different document types (PDF, markdown, HTML)
  - What their OCR pipeline does
  - The PageIndex cookbook notebooks (vectorless_rag, vision_rag)
- [ ] Study the RAPTOR paper (PageIndex cites it as inspiration)
- [ ] Document all findings in a RESEARCH.md file

**Deliverable:** A deep technical understanding of exactly how vectorless
RAG works, the specific prompts, data structures, and algorithms.

### PHASE 1 — Build the Core Library (Week 2-4)
**Cost: $0-50 | Tools: Claude Code, TypeScript/Python**

The library has 3 core modules:

#### Module 1: Tree Generator
```
Input:  Document text (string or pages array)
Output: Hierarchical JSON tree with summaries

Algorithm:
1. If document fits in context window → single-pass tree generation
2. If document exceeds context → chunked processing:
   a. Split into overlapping page groups (e.g., 10 pages each)
   b. Generate sub-trees for each group
   c. Merge sub-trees into a unified tree
   d. Generate top-level summaries
3. Output standardized TreeNode[] JSON
```

#### Module 2: Tree Searcher (Reasoner)
```
Input:  Tree JSON + User query
Output: Ranked list of relevant node IDs + content

Algorithm:
1. Present top-level tree to LLM with the query
2. LLM reasons about which branches are relevant
3. Expand selected branches, present deeper nodes
4. LLM selects leaf nodes
5. Return content from selected nodes with confidence scores
```

#### Module 3: RAG Pipeline (Orchestrator)
```
Input:  Document(s) + User query
Output: Answer with citations

Pipeline:
1. Check if tree exists for document → if not, generate
2. Run tree search with query → get relevant sections
3. Build prompt: system instructions + retrieved context + query
4. Call answer LLM → return response with page citations
```

#### LLM Provider Interface (Pluggable)
```typescript
interface LLMProvider {
  complete(prompt: string, options?: CompletionOptions): Promise<string>;
  name: string;
}

// Implementations:
class ClaudeProvider implements LLMProvider { ... }
class OpenAIProvider implements LLMProvider { ... }
class GroqProvider implements LLMProvider { ... }    // FREE tier
class OllamaProvider implements LLMProvider { ... }  // Local, FREE
class HuggingFaceProvider implements LLMProvider { ... }
class CustomProvider implements LLMProvider { ... }   // Any OpenAI-compatible API
```

#### Key Design Decisions
- **Dual language:** Build in Python first (ML ecosystem), then TypeScript port
- **LLM-agnostic:** Works with any provider (critical for cost flexibility)
- **Cacheable:** Trees are JSON, store anywhere (file, DB, S3)
- **Streamable:** Tree search can stream intermediate results
- **Benchmarkable:** Built-in eval against PageIndex on same documents

**Deliverables:**
- `treereason` Python package on PyPI
- `treereason` npm package on npm
- GitHub repo with docs, examples, benchmarks
- Works with Groq free tier out of the box (for zero-cost usage)

### PHASE 2 — Training Data Generation (Week 5-6)
**Cost: $20-100 | Tools: Groq API (free), your library**

- [ ] Collect 2,000+ diverse public domain PDFs:
  - arXiv papers (CS, physics, biology, economics) — API available
  - SEC filings (EDGAR database, free)
  - Wikipedia featured articles (dump available)
  - Project Gutenberg books (free)
  - Government reports (public domain)
  - Open-access textbooks (OpenStax, free)
- [ ] Run your library (Phase 1) with GPT-4/Claude to generate "gold standard" trees
- [ ] Create training pairs: `{ input: document_text, output: tree_json }`
- [ ] Create search training pairs: `{ input: { tree, question }, output: relevant_node_ids }`
- [ ] For search pairs, generate questions using Claude, then label correct nodes
- [ ] Clean and validate all data
- [ ] Publish dataset on HuggingFace Datasets (builds credibility + community)

**Data format:**
```json
{
  "document_text": "full text of the document...",
  "document_metadata": { "type": "academic_paper", "pages": 12 },
  "tree": { "title": "...", "children": [...] },
  "qa_pairs": [
    {
      "question": "What methodology was used?",
      "relevant_nodes": ["node_1_2", "node_1_3"],
      "answer": "The study used..."
    }
  ]
}
```

### PHASE 3 — Fine-Tune Specialized Models (Week 7-8)
**Cost: $0-20 | Tools: Kaggle/Colab, HuggingFace, QLoRA**

#### Model A: TreeGen-7B (Tree Generation)
- Base model: Qwen 2.5 7B Instruct (Apache 2.0 license, commercially free)
- Fine-tune task: Given document text → output tree JSON
- QLoRA config: rank=16, alpha=32, target all linear layers
- Training: ~2,000 examples, 3 epochs, ~3-4 hours on T4
- Expected output: A model that generates document trees 10-50x cheaper than GPT-4

#### Model B: TreeSearch-3B (Tree Navigation)
- Base model: Qwen 2.5 3B Instruct or Phi-3.5 Mini (3.8B)
- Fine-tune task: Given tree + question → output relevant node IDs
- This is a simpler task, so a smaller model works
- QLoRA: rank=8, alpha=16
- Training: ~5,000 examples, 5 epochs, ~1-2 hours on T4

#### Publishing
- Upload both models to HuggingFace Hub (free, public)
- Include model cards with benchmarks
- Provide Ollama Modelfile for easy local deployment:
  `ollama run treereason/treegen-7b`
  `ollama run treereason/treesearch-3b`

### PHASE 4 — Integration & Demo (Week 9-10)
**Cost: $0-25/mo | Tools: Existing ExamPrep codebase**

- [ ] Update the `treereason` library to support the fine-tuned models as providers
- [ ] Add Ollama integration (users run models locally for free)
- [ ] Rebuild ExamPrep to use `treereason` instead of PageIndex API:
  - Replace `lib/pageindex.ts` with `treereason` client
  - Tree generation happens locally or via your models
  - Chat uses tree search + any LLM for answer generation
  - Zero external API dependency (except the answer LLM)
- [ ] Create a standalone demo: `npx treereason-demo` that:
  - Spins up a local web UI
  - Upload a PDF, generates tree, ask questions
  - Works 100% offline with Ollama
- [ ] Write comprehensive documentation
- [ ] Create a "Getting Started in 5 Minutes" guide
- [ ] Record a demo video

### PHASE 5 — Launch & Community (Week 11-12)
**Cost: $10 (domain) | Tools: GitHub, Twitter, Reddit, HN**

- [ ] Polish the GitHub repo (README with badges, GIF demos, architecture diagrams)
- [ ] Publish blog post: "We Made Vectorless RAG 75x Cheaper and Open Source"
- [ ] Post to:
  - Hacker News (Show HN)
  - r/MachineLearning, r/LocalLLaMA, r/selfhosted
  - Twitter/X AI community
  - HuggingFace community
  - Dev.to / Medium
- [ ] Submit to Product Hunt
- [ ] HuggingFace Spaces demo (free hosted demo)

### PHASE 6 — Monetization (Month 3+)
**Cost: $50-100/mo for infrastructure | Revenue: hopefully > cost**

Once there's community traction:
- **Hosted API:** `api.treereason.dev` — pay-per-document pricing
  - Tree generation: $0.05/document (vs PageIndex's higher pricing)
  - Tree search: $0.005/query
  - This is profitable because your fine-tuned models are 75x cheaper to run
- **Cloud dashboard:** Upload docs, manage trees, query — $9-29/month plans
- **Enterprise:** On-prem deployment support, custom fine-tuning
- **ExamPrep:** Relaunch as a demo product built on TreeReason

---

## WEEKLY BUDGET PLANNER (Worst Case)

```
Week 1:   $0    (research, Skill Seekers, reading code)
Week 2:   $0    (writing library code)
Week 3:   $0    (writing library code)
Week 4:   $10   (API testing — use free tiers first, Claude Haiku for overflow)
Week 5:   $10   (Colab Pro for data generation at scale)
Week 6:   $20   (more data generation, validating quality)
Week 7:   $0    (fine-tuning on Kaggle free GPUs)
Week 8:   $0    (fine-tuning iteration, evaluation)
Week 9:   $0    (integration, demo building)
Week 10:  $0    (documentation, testing)
Week 11:  $10   (domain name, final polish)
Week 12:  $0    (launch)
──────────────────
TOTAL:    ~$50   (conservative real-world spend)
```

---

## SKILL SEEKERS — EXACT COMMANDS TO RUN

### Step 1: Install Skill Seekers
```bash
pip install skill-seekers
```

### Step 2: Configure GitHub access (for private repo rate limits)
```bash
skill-seekers config --github
# Enter your GitHub personal access token
# This gives you 5000 API requests/hour instead of 60
```

### Step 3: Analyze the PageIndex repository
```bash
# Full analysis with LOCAL enhancement (uses Claude Code Max, no API cost)
skill-seekers github --repo VectifyAI/PageIndex --enhance-local

# This will:
# 1. Clone and analyze the entire repo structure
# 2. Parse all Python files, understand the algorithms
# 3. Extract code patterns, data structures, prompts
# 4. Generate a comprehensive SKILL.md
# 5. Enhance it using your Claude Code Max (free)
```

### Step 4: Install the skill into Claude Code
```bash
# Copy the generated skill to Claude Code's skill directory
cp -r output/pageindex ~/.claude/skills/

# Verify it's installed
ls ~/.claude/skills/pageindex/SKILL.md
```

### Step 5: Also analyze the PageIndex documentation site
```bash
skill-seekers scrape --url https://docs.pageindex.ai --enhance-local
cp -r output/pageindex-docs ~/.claude/skills/
```

### Step 6: Use Claude Code with the skills loaded
Now when you open Claude Code in the PageIndex project directory,
Claude will have deep knowledge of:
- How run_pageindex.py works
- The tree generation algorithm and prompts
- The tree search / retrieval logic
- The data structures (TreeNode, Document, etc.)
- The OCR pipeline
- The API endpoints
- The cookbook examples and best practices

You can then ask Claude Code things like:
- "Show me exactly how PageIndex generates the tree structure from a PDF"
- "What prompts does PageIndex use for tree search?"
- "How does the tree merging work for long documents?"
- "Rebuild this algorithm using Qwen 2.5 7B via Ollama instead of their API"

---

## WHY THIS CAN BE BIG — THE MARKET THESIS

1. **PageIndex proved vectorless RAG works** — 98.7% on FinanceBench
2. **But it's expensive and proprietary** — API credits, enterprise pricing
3. **Nobody has built the open-source alternative yet** — first mover advantage
4. **The timing is perfect:**
   - Small LLMs (3B-8B) are now good enough for structured tasks
   - QLoRA makes fine-tuning accessible on consumer hardware
   - Ollama makes local model deployment trivial
   - The developer community is hungry for vector DB alternatives
5. **The moat:** Once you have fine-tuned models + training data + benchmarks,
   it's very hard for someone else to catch up quickly
6. **Revenue path is clear:** Free self-hosted → Paid cloud API → Enterprise

The equivalent in other domains:
- Supabase is to Firebase what TreeReason is to PageIndex
- Ollama is to OpenAI what TreeReason is to PageIndex
- Next.js is to Vercel what TreeReason's library is to its cloud service

---

## NAMING IDEAS

The project needs a strong name. Some options:
- **TreeReason** — describes exactly what it does (tree-based reasoning retrieval)
- **Dendrite** — tree-like neural connections (scientific, memorable)
- **BranchSearch** — intuitive, describes the tree search
- **ReasonRAG** — reasoning-based RAG
- **OpenPageIndex** — directly positions against PageIndex (risky legally)
- **TreeDex** — short, catchy
- **Arbor** — Latin for tree (elegant, simple)

Pick one that feels right to you. The name should work as:
- A GitHub repo: `github.com/yourname/treereason`
- A Python package: `pip install treereason`
- A npm package: `npm install treereason`
- A domain: `treereason.dev`
- A HuggingFace org: `huggingface.co/treereason`
