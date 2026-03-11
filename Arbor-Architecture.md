# Arbor — Library Architecture & Phase 1 Implementation Plan

> Based on RESEARCH.md (1994-line deep analysis of PageIndex internals)
> This file is the blueprint for Claude Code to build the Arbor library.

---

## What We Learned from PageIndex

PageIndex's core is surprisingly straightforward — 14 LLM prompts + ~500 lines of orchestration logic.
The brilliance is in the design, not the complexity. Here's what Arbor needs to replicate and improve:

### The Algorithm (3 sentences)
1. Extract text from each PDF page, tag them with `<physical_index_N>` markers
2. Use an LLM to produce a flat list of `{ structure: "1.2.3", title, physical_index }` items (via TOC detection or from-scratch generation), then convert dot-notation to nested tree via `list_to_tree()`
3. For retrieval, present the tree (with summaries) to an LLM and ask "which node_ids contain the answer?" — single prompt, returns `{ thinking, node_list }`

### What Arbor Does Differently
- **LLM-agnostic**: Works with Claude, OpenAI, Groq (free), Ollama (local), HuggingFace, any OpenAI-compatible API
- **Dual language**: Python (primary) + TypeScript port (Phase 2)
- **No vendor lock-in**: PageIndex hardcodes OpenAI SDK; Arbor uses a provider interface
- **Cheaper defaults**: Uses Groq's free Llama 3.1 70B as default provider (zero cost)
- **Better chunking**: Same algorithm but configurable overlap and token counting per provider
- **Async-first**: All LLM calls are async with configurable concurrency limits
- **Streaming tree search**: Stream intermediate reasoning during retrieval

---

## Project Structure

```
arbor/
├── README.md                        # Project overview + quickstart
├── ROADMAP.md                       # Full project roadmap
├── RESEARCH.md                      # PageIndex deep analysis
├── LICENSE                          # MIT License
├── pyproject.toml                   # Python package config (setuptools/hatch)
├── setup.py                         # Fallback setup
│
├── arbor/                           # Python package
│   ├── __init__.py                  # Public API exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tree_generator.py        # Main tree generation pipeline
│   │   ├── tree_searcher.py         # Tree search / retrieval
│   │   ├── rag_pipeline.py          # Full RAG orchestrator (generate + search + answer)
│   │   └── types.py                 # Dataclasses: TreeNode, Document, SearchResult, etc.
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py         # PDF text extraction (PyPDF2 + PyMuPDF)
│   │   ├── markdown_extractor.py    # Markdown header-based extraction (regex, no LLM)
│   │   └── text_utils.py            # Page tagging, token counting, chunking
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── tree_generation.py       # All tree generation prompts (14 prompts)
│   │   ├── tree_search.py           # Tree search prompts
│   │   ├── summarization.py         # Node summary + doc description prompts
│   │   └── verification.py          # TOC verification + fix prompts
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract LLMProvider interface
│   │   ├── openai_provider.py       # OpenAI / OpenAI-compatible (Groq, Together, etc.)
│   │   ├── anthropic_provider.py    # Claude API
│   │   ├── ollama_provider.py       # Local Ollama models
│   │   └── huggingface_provider.py  # HuggingFace Inference API
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── toc_detector.py          # TOC detection in first N pages
│   │   ├── toc_processor.py         # 3-mode processing (with-pages, no-pages, no-toc)
│   │   ├── tree_builder.py          # list_to_tree() + post_processing()
│   │   ├── node_subdivision.py      # Recursive large-node splitting
│   │   ├── verification.py          # verify_toc() + fix_incorrect entries
│   │   └── json_utils.py            # extract_json(), continuation logic
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tree_utils.py            # write_node_id, create_node_mapping, print_tree, etc.
│       ├── token_counter.py         # Token counting (tiktoken or approximation)
│       └── async_helpers.py         # Concurrency limiter, retry logic
│
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_tree_generator.py
│   ├── test_tree_searcher.py
│   ├── test_tree_builder.py
│   ├── test_prompts.py
│   └── fixtures/                    # Test PDFs + expected outputs
│       ├── sample.pdf
│       └── expected_tree.json
│
├── examples/
│   ├── quickstart.py                # 10-line example
│   ├── custom_provider.py           # Using Ollama locally
│   ├── full_rag.py                  # Complete RAG pipeline
│   └── benchmark.py                 # Compare Arbor vs PageIndex
│
└── docs/
    ├── getting-started.md
    ├── providers.md
    ├── api-reference.md
    └── architecture.md
```

---

## Core Data Types (`arbor/core/types.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ProcessingMode(Enum):
    TOC_WITH_PAGES = "toc_with_pages"
    TOC_NO_PAGES = "toc_no_pages"
    NO_TOC = "no_toc"

@dataclass
class TreeNode:
    title: str
    start_index: int                          # 1-based page number
    end_index: int                            # 1-based, inclusive
    node_id: Optional[str] = None             # Zero-padded 4-digit: "0000"
    summary: Optional[str] = None
    text: Optional[str] = None
    nodes: list['TreeNode'] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (recursive)"""
        d = {"title": self.title, "start_index": self.start_index, "end_index": self.end_index}
        if self.node_id: d["node_id"] = self.node_id
        if self.summary: d["summary"] = self.summary
        if self.text: d["text"] = self.text
        if self.nodes: d["nodes"] = [n.to_dict() for n in self.nodes]
        return d

@dataclass
class DocumentTree:
    doc_name: str
    structure: list[TreeNode]
    doc_description: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"doc_name": self.doc_name, "structure": [n.to_dict() for n in self.structure]}
        if self.doc_description: d["doc_description"] = self.doc_description
        return d

@dataclass
class PageContent:
    text: str
    token_count: int
    page_number: int                          # 1-based

@dataclass
class SearchResult:
    thinking: str                             # LLM's reasoning about relevance
    node_ids: list[str]                       # Retrieved node IDs
    nodes: list[TreeNode] = field(default_factory=list)

@dataclass
class RAGResponse:
    answer: str
    search_result: SearchResult
    context: str                              # The text that was sent to the LLM
    citations: list[dict] = field(default_factory=list)  # [{page, text}]

@dataclass
class ArborConfig:
    model: str = "llama-3.1-70b-versatile"    # Default: Groq free tier
    toc_check_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20000
    add_node_ids: bool = True
    add_summaries: bool = True
    add_doc_description: bool = False
    add_node_text: bool = False
    max_concurrent_llm_calls: int = 5
    overlap_pages: int = 1                    # Overlap between chunks
```

---

## LLM Provider Interface (`arbor/providers/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional

class LLMProvider(ABC):
    """Abstract interface for LLM backends. Arbor works with any provider."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """Send a prompt and return the completion text."""
        ...

    @abstractmethod
    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        """Returns (content, finish_reason). finish_reason = 'stop' | 'length'"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    def count_tokens(self, text: str) -> int:
        """Approximate token count. Override for exact counting."""
        return len(text) // 4  # Rough 4-chars-per-token approximation
```

### Provider Implementations

**GroqProvider** (default — FREE):
```python
class GroqProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
```

**OllamaProvider** (local — FREE, no API key):
```python
class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=f"{base_url}/v1"
        )
```

**AnthropicProvider**:
```python
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        # Uses anthropic SDK directly
```

**OpenAIProvider**:
```python
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        # Standard OpenAI SDK
```

---

## Public API (`arbor/__init__.py`)

```python
from arbor.core.tree_generator import generate_tree
from arbor.core.tree_searcher import search_tree
from arbor.core.rag_pipeline import query
from arbor.core.types import (
    TreeNode, DocumentTree, SearchResult, RAGResponse, ArborConfig,
    PageContent, ProcessingMode
)
from arbor.providers.base import LLMProvider
from arbor.providers.openai_provider import OpenAIProvider, GroqProvider
from arbor.providers.anthropic_provider import AnthropicProvider
from arbor.providers.ollama_provider import OllamaProvider

__version__ = "0.1.0"

# Convenience: one-line usage
async def arbor(
    document: str,              # PDF path or markdown path
    provider: LLMProvider = None,
    config: ArborConfig = None,
) -> DocumentTree:
    """Generate a document tree. This is the main entry point."""
    config = config or ArborConfig()
    provider = provider or GroqProvider()
    return await generate_tree(document, provider, config)
```

### Usage Examples

```python
# === Quickstart (3 lines, FREE with Groq) ===
import arbor
tree = await arbor.arbor("textbook.pdf")
print(tree.to_dict())

# === With Ollama (100% local, zero cost) ===
provider = arbor.OllamaProvider(model="qwen2.5:7b")
tree = await arbor.arbor("report.pdf", provider=provider)

# === Full RAG pipeline ===
provider = arbor.AnthropicProvider()
response = await arbor.query(
    document="textbook.pdf",
    question="What is backpropagation?",
    provider=provider,
)
print(response.answer)
print(response.citations)

# === Custom config ===
config = arbor.ArborConfig(
    toc_check_pages=30,
    max_pages_per_node=15,
    add_summaries=True,
)
tree = await arbor.arbor("long-report.pdf", config=config)

# === Use pre-generated tree for search ===
result = await arbor.search_tree(
    tree=tree,
    question="What are the Q3 revenue numbers?",
    provider=provider,
)
for node_id in result.node_ids:
    print(f"Node {node_id}: {result.nodes[node_id].title}")
```

---

## Implementation Priority (for Claude Code)

Build in this exact order. Each step should be tested before moving to the next.

### Step 1: Types + Provider Interface
- `arbor/core/types.py` — All dataclasses above
- `arbor/providers/base.py` — Abstract LLMProvider
- `arbor/providers/openai_provider.py` — OpenAI + Groq (same SDK, different base_url)
- `arbor/providers/ollama_provider.py` — Ollama local
- Test: verify a provider can complete a simple prompt

### Step 2: PDF Extraction + Text Utils
- `arbor/extraction/pdf_extractor.py` — `get_page_contents(pdf_path) -> list[PageContent]`
  - Use PyPDF2 (lighter) with PyMuPDF fallback
  - Returns list of (text, token_count, page_number) per page
- `arbor/extraction/text_utils.py`:
  - `tag_pages(pages) -> list[str]` — wraps each page in `<physical_index_N>` tags
  - `group_pages(pages, max_tokens, overlap) -> list[str]` — the chunking algorithm:
    `target = ceil((total/N + max_tokens) / 2)` with overlap
  - `count_tokens(text) -> int` — tiktoken if available, else len(text)//4
- `arbor/extraction/markdown_extractor.py` — regex header extraction (no LLM needed)
- Test: extract pages from a real PDF, verify token counts

### Step 3: All Prompts
- `arbor/prompts/tree_generation.py` — All 14 prompts from RESEARCH.md:
  1. `toc_detector_prompt(content) -> str`
  2. `check_toc_complete_prompt(content, toc) -> str`
  3. `check_toc_transform_complete_prompt(content, toc) -> str`
  4. `extract_toc_prompt(content) -> str`
  5. `detect_page_index_prompt(toc_content) -> str`
  6. `toc_transformer_prompt(toc_content) -> str`
  7. `toc_index_extractor_prompt(toc, content) -> str`
  8. `add_page_number_prompt(part, structure) -> str`
  9. `generate_toc_init_prompt(part) -> str`
  10. `generate_toc_continue_prompt(toc_content, part) -> str`
  11. `check_title_appearance_prompt(title, page_text) -> str`
  12. `check_title_at_start_prompt(title, page_text) -> str`
  13. `fix_toc_entry_prompt(section_title, content) -> str`
  14. `generate_summary_prompt(text) -> str`
- `arbor/prompts/tree_search.py`:
  1. `tree_search_prompt(query, tree_json) -> str`
  2. `tree_search_with_preference_prompt(query, tree_json, preference) -> str`
  3. `answer_generation_prompt(query, context) -> str`
- `arbor/prompts/verification.py` — verification prompts
- Test: verify all prompts render correctly with sample data

### Step 4: TOC Processing Pipeline
- `arbor/processing/toc_detector.py`:
  - `check_toc(pages, config) -> {toc_content, toc_page_list, page_index_given}`
  - Scans first N pages, calls toc_detector prompt, extracts TOC
- `arbor/processing/toc_processor.py`:
  - `process_toc_with_pages(toc_content, toc_page_list, pages, provider)`
  - `process_toc_no_pages(toc_content, toc_page_list, pages, provider)`
  - `process_no_toc(pages, start_index, provider)`
  - Each returns flat list of `{structure, title, physical_index}`
- `arbor/processing/json_utils.py`:
  - `extract_json(content) -> dict` — handles ```json blocks, trailing commas, None→null
  - `continue_truncated_output(provider, prompt, partial_response) -> str`
- Test: process a real PDF through each mode

### Step 5: Tree Building
- `arbor/processing/tree_builder.py`:
  - `list_to_tree(flat_items) -> list[TreeNode]` — dot-notation hierarchy
  - `post_processing(items, total_pages) -> list[TreeNode]` — compute end_index, build tree
  - `add_preface_if_needed(items)` — prepend untitled content before first section
- `arbor/utils/tree_utils.py`:
  - `write_node_ids(tree) -> None` — depth-first pre-order, zero-padded 4-digit
  - `create_node_mapping(tree) -> dict[str, TreeNode]` — node_id → node lookup
  - `add_node_text(tree, pages)` — inject page text into nodes
  - `remove_node_text(tree)` — strip text field
  - `print_tree(tree, indent=0)` — pretty print
- Test: convert sample flat list to tree, verify against expected output

### Step 6: Verification + Error Correction
- `arbor/processing/verification.py`:
  - `verify_toc(pages, toc_items, provider) -> (accuracy, incorrect_items)`
  - `fix_incorrect_entries(toc_items, pages, incorrect, provider, max_retries=3)`
- Test: deliberately introduce errors, verify correction

### Step 7: Node Subdivision
- `arbor/processing/node_subdivision.py`:
  - `process_large_nodes(tree, pages, config, provider)` — recursive splitting
  - Threshold: `end - start > max_pages AND tokens > max_tokens`
  - Re-processes large node's pages with `process_no_toc`, merges as children
- Test: create a node spanning 30 pages, verify subdivision

### Step 8: Main Tree Generator (orchestrator)
- `arbor/core/tree_generator.py`:
  - `generate_tree(document, provider, config) -> DocumentTree`
  - Orchestrates: extract pages → detect TOC → process (3-mode fallback) → verify → build tree → subdivide → add summaries → return
  - The meta_processor fallback cascade: toc_with_pages → toc_no_pages → no_toc (on accuracy < 0.6)
- Test: end-to-end on real PDF

### Step 9: Tree Search + RAG
- `arbor/core/tree_searcher.py`:
  - `search_tree(tree, question, provider) -> SearchResult`
  - Builds tree JSON (without text), sends search prompt, parses response
- `arbor/core/rag_pipeline.py`:
  - `query(document, question, provider, config, tree=None) -> RAGResponse`
  - If no tree, generate one first; then search; then extract context; then answer
- Test: end-to-end query on a real document

### Step 10: Package + Publish
- `pyproject.toml` with all metadata
- `README.md` with badges, installation, quickstart
- Publish to PyPI: `pip install arbor-rag`
- Push to GitHub

---

## Critical Implementation Notes

### 1. JSON Extraction (must be robust)
PageIndex has issues with LLM JSON responses (trailing commas, None instead of null,
markdown code blocks). Port their `extract_json()` logic:
```python
def extract_json(content: str) -> any:
    # Strip ```json ... ``` wrapper
    # Replace Python None with null
    # Remove trailing commas before } and ]
    # Try json.loads, fall back to ast.literal_eval
```

### 2. Page Tagging Format (must be exact)
```python
def tag_page(text: str, page_number: int) -> str:
    return f"<physical_index_{page_number}>\n{text}\n<physical_index_{page_number}>\n\n"
```
Both opening and closing tags are identical (not open/close). This is how PageIndex does it.

### 3. Token Counting Strategy
- If tiktoken is installed, use it (exact, but requires the package)
- Fallback: `len(text) // 4` (rough approximation, good enough for chunking)
- Make tiktoken an optional dependency: `pip install arbor-rag[tiktoken]`

### 4. Continuation for Truncated Outputs
When `finish_reason == "length"`, continue via chat history:
```python
chat_history = [
    {"role": "user", "content": original_prompt},
    {"role": "assistant", "content": partial_response},
]
continuation = await provider.complete(
    "Please continue the generation. Output only the remaining part.",
    chat_history=chat_history
)
full_response = partial_response + continuation
```
Loop up to 5 times. This is critical for long TOCs.

### 5. Concurrency Control
Use asyncio.Semaphore to limit concurrent LLM calls:
```python
semaphore = asyncio.Semaphore(config.max_concurrent_llm_calls)
async def limited_call(prompt):
    async with semaphore:
        return await provider.complete(prompt)
```
Important for Groq (rate-limited) and Ollama (single model instance).

### 6. Physical Index Parsing
TOC items return `physical_index: "<physical_index_5>"` as a string.
Must extract the integer:
```python
def parse_physical_index(value: str) -> int:
    if isinstance(value, int): return value
    match = re.search(r'physical_index_(\d+)', str(value))
    return int(match.group(1)) if match else None
```

---

## Dependencies (minimal)

### Required
```
PyPDF2>=3.0.0          # PDF text extraction
openai>=1.0.0          # OpenAI-compatible API client (works for Groq, Ollama too)
```

### Optional
```
pymupdf>=1.20.0        # Better PDF extraction (fallback)
tiktoken>=0.5.0        # Exact token counting
anthropic>=0.20.0      # Claude API support
```

### pyproject.toml
```toml
[project]
name = "arbor-rag"
version = "0.1.0"
description = "Open-source vectorless RAG engine. Tree-structured document indexing with LLM reasoning."
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "PyPDF2>=3.0.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
tiktoken = ["tiktoken>=0.5.0"]
pymupdf = ["pymupdf>=1.20.0"]
anthropic = ["anthropic>=0.20.0"]
all = ["tiktoken>=0.5.0", "pymupdf>=1.20.0", "anthropic>=0.20.0"]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/arbor"
Documentation = "https://github.com/YOUR_USERNAME/arbor#readme"
```
