# Arbor — Production Upgrade Implementation Plan

> Reference document for building the next 5 production features.
> Written after deep analysis of Arbor source + Claude Code source.

---

## Overview

| # | Feature | Files Touched | Effort |
|---|---------|--------------|--------|
| 1 | MCP Server | `arbor/mcp_server.py` (new) | 4 hours |
| 2 | AsyncGenerator Streaming | `arbor/core/rag_pipeline.py`, `arbor/__init__.py` | 3 hours |
| 3 | Budget Controls | `arbor/types.py`, `arbor/core/tree_searcher.py`, `arbor/core/rag_pipeline.py` | 1 hour |
| 4 | Schema Enforcement | `arbor/core/tree_searcher.py` | 2 hours |
| 5 | Parallel Branch Navigation | `arbor/core/tree_searcher.py` | 1 session |

Build order: **3 → 4 → 5 → 2 → 1**
Reason: budget + schema + parallel are changes to existing code. Streaming wraps them. MCP wraps streaming.

---

## Feature 3 — Budget Controls

### Why first
Budget controls are a safety net for every subsequent feature. Parallel execution (feature 5) without a budget cap could run 5x the calls. Streaming (feature 2) without a timeout could hang forever. Do this first.

### What to add to `ArborConfig` (in `arbor/types.py`)

```python
@dataclass
class ArborConfig:
    # ... existing fields unchanged ...
    model: str = "llama-3.3-70b-versatile"
    toc_check_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20000
    add_node_ids: bool = True
    add_summaries: bool = True
    add_doc_description: bool = False
    add_node_text: bool = False
    max_concurrent_llm_calls: int = 5
    overlap_pages: int = 1

    # NEW: budget controls
    max_hops: int = 5               # Max tree levels to navigate (already in search_tree sig, now in config)
    max_nodes_searched: int = 100   # Hard stop: total nodes examined across all levels
    max_cost_usd: float = 0.50      # Hard stop on estimated API spend (0 = no limit)
    timeout_sec: float = 120.0      # Wall clock timeout for full query() call (0 = no limit)
    max_retries_on_bad_json: int = 2 # How many times to retry malformed TreeSearch output
```

### New exception class (add to `arbor/types.py`)

```python
class BudgetExceededError(Exception):
    """Raised when a query exceeds configured budget limits."""
    def __init__(self, reason: str, partial_nodes: list[str] = None):
        super().__init__(reason)
        self.partial_nodes = partial_nodes or []
```

### Budget tracker (add to `arbor/core/tree_searcher.py`)

```python
@dataclass
class _BudgetTracker:
    max_nodes: int
    max_cost_usd: float
    nodes_examined: int = 0
    estimated_cost_usd: float = 0.0

    def charge_node(self, node_text_tokens: int, model_cost_per_1k: float = 0.000075):
        self.nodes_examined += 1
        self.estimated_cost_usd += (node_text_tokens / 1000) * model_cost_per_1k
        if self.max_nodes > 0 and self.nodes_examined >= self.max_nodes:
            raise BudgetExceededError(
                f"max_nodes_searched={self.max_nodes} reached",
                partial_nodes=[]
            )
        if self.max_cost_usd > 0 and self.estimated_cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"max_cost_usd=${self.max_cost_usd:.2f} exceeded "
                f"(estimated ${self.estimated_cost_usd:.4f})"
            )
```

### Wire into `search_tree()` in `arbor/core/tree_searcher.py`

Current signature:
```python
async def search_tree(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    preference: Optional[str] = None,
    multihop: bool = False,
    max_hops: int = 5,
) -> SearchResult:
```

New signature:
```python
async def search_tree(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    preference: Optional[str] = None,
    multihop: bool = False,
    max_hops: int = 5,
    config: Optional[ArborConfig] = None,  # ADD THIS
) -> SearchResult:
```

Inside `_search_multihop()`, create tracker and pass it to `navigate_level()`:
```python
tracker = _BudgetTracker(
    max_nodes=config.max_nodes_searched if config else 100,
    max_cost_usd=config.max_cost_usd if config else 0,
)
```

### Wire into `query()` in `arbor/core/rag_pipeline.py`

Wrap the full query in `asyncio.wait_for`:
```python
async def query(...) -> RAGResponse:
    timeout = config.timeout_sec if config else 120.0
    try:
        return await asyncio.wait_for(
            _query_impl(...),
            timeout=timeout if timeout > 0 else None
        )
    except asyncio.TimeoutError:
        raise BudgetExceededError(f"Query timed out after {timeout}s")
```

---

## Feature 4 — Schema Enforcement on TreeSearch Output

### The problem
`_search_multihop()` calls `provider.complete()` and then `_parse_navigate_response()`. If the model returns malformed JSON (missing `navigate_to`, non-list value, hallucinated field names), `_parse_navigate_response()` silently returns empty results. The navigation stops without error. User gets no answer.

### Current `_parse_navigate_response()` behavior (in `tree_searcher.py`)
- Tries `json.loads(response)`
- Falls back to regex `r'\b(\d{4})\b'` to extract node IDs
- Returns empty list on total failure

### New: retry loop with correction prompt

```python
_NAVIGATE_SCHEMA = {
    "type": "object",
    "required": ["thinking", "navigate_to"],
    "properties": {
        "thinking": {"type": "string", "minLength": 1},
        "navigate_to": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^\d{4}$"}
        }
    }
}

_CORRECTION_PROMPT_TEMPLATE = """\
Your previous response was not valid JSON matching the required schema.

Required format:
{{"thinking": "brief reasoning", "navigate_to": ["0001", "0002"]}}

Rules:
- navigate_to must be a JSON array of strings
- Each string must be a 4-digit node ID exactly as shown in the section list
- If no sections are relevant, use: {{"thinking": "none relevant", "navigate_to": []}}

Your previous response was:
{bad_response}

Valid node IDs in this level are: {valid_ids}

Please respond with valid JSON only:"""


async def _enforce_navigate_schema(
    response: str,
    valid_node_ids: set[str],
    provider: LLMProvider,
    original_prompt: str,
    max_retries: int = 2,
) -> dict:
    """
    Validate response against navigate schema.
    Retry with correction prompt up to max_retries times.
    Returns parsed dict with 'thinking' and 'navigate_to' keys.
    On total failure, returns {"thinking": "parse_failed", "navigate_to": []}.
    """
    for attempt in range(max_retries + 1):
        parsed = _try_parse(response, valid_node_ids)
        if parsed is not None:
            return parsed

        if attempt == max_retries:
            break

        # Build correction prompt
        correction = _CORRECTION_PROMPT_TEMPLATE.format(
            bad_response=response[:500],
            valid_ids=", ".join(sorted(valid_node_ids)[:20])
        )
        response = await provider.complete(correction, temperature=0.0)

    # Total failure — return empty navigation (model will be asked at next level)
    return {"thinking": "schema_enforcement_failed", "navigate_to": []}


def _try_parse(response: str, valid_node_ids: set[str]) -> dict | None:
    """Parse and validate. Returns None if invalid."""
    text = response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON object with regex
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    # Validate structure
    if not isinstance(parsed, dict):
        return None
    if "navigate_to" not in parsed:
        return None
    nav = parsed["navigate_to"]
    if not isinstance(nav, list):
        return None
    # Filter to only valid IDs (prevents hallucinated node IDs)
    parsed["navigate_to"] = [
        str(n) for n in nav
        if str(n) in valid_node_ids
    ]
    if "thinking" not in parsed or not parsed["thinking"]:
        parsed["thinking"] = "(no reasoning provided)"

    return parsed
```

### Where to call this in `_search_multihop()`

Replace the existing parse call:
```python
# BEFORE (current):
response = await provider.complete(prompt)
result = _parse_navigate_response(response)
navigate_to = result.get("navigate_to", [])

# AFTER (with enforcement):
valid_ids = {node.node_id for node in window}
result = await _enforce_navigate_schema(
    response=await provider.complete(prompt),
    valid_node_ids=valid_ids,
    provider=provider,
    original_prompt=prompt,
    max_retries=config.max_retries_on_bad_json if config else 2,
)
navigate_to = result.get("navigate_to", [])
```

---

## Feature 5 — Parallel Branch Navigation

### Current behavior in `_search_multihop()`

The `navigate_level()` coroutine iterates over `navigate_to` **sequentially**:
```python
for node_id in navigate_to:
    node = get_node_by_id(structure, node_id)
    if node and node.nodes:
        await navigate_level(node.nodes, depth + 1)  # SEQUENTIAL
    else:
        final_node_ids.append(node_id)
```

If `navigate_to = ["0003", "0007", "0012"]`, this makes 3 sequential LLM calls at the next level.

### New behavior: parallel gather

Each branch navigation is **independently safe** — branch A's result doesn't affect branch B. Replace sequential loop with `asyncio.gather`:

```python
async def navigate_level(nodes: list, depth: int) -> None:
    if depth >= max_hops:
        final_node_ids.extend(n.node_id for n in nodes)
        return

    # Window chunking — unchanged
    windows = _chunk_windows(nodes, _MAX_NODES_PER_WINDOW)

    # Collect all navigate_to from all windows
    all_navigate_to: list[str] = []

    for window in windows:
        valid_ids = {n.node_id for n in window}
        prompt = _build_navigate_prompt(question, window)
        response = await provider.complete(prompt, temperature=0.0)
        result = await _enforce_navigate_schema(response, valid_ids, provider, prompt)
        all_navigate_to.extend(result.get("navigate_to", []))

    if not all_navigate_to:
        return

    # Recurse into children IN PARALLEL
    async def recurse_into(node_id: str) -> None:
        node = get_node_by_id(structure, node_id)
        if node is None:
            return
        if node.nodes:
            await navigate_level(node.nodes, depth + 1)
        else:
            final_node_ids.append(node_id)

    # This is the key change: gather instead of sequential for-loop
    await asyncio.gather(*[recurse_into(nid) for nid in all_navigate_to])
```

### Concurrency safety guarantee
Each `recurse_into()` call:
- Has its own `node_id` and subtree — no shared state
- Appends to `final_node_ids` (thread-safe via asyncio single-thread event loop)
- Uses separate provider calls (the provider's `asyncio.Semaphore` already limits concurrency via `max_concurrent_llm_calls`)

### Expected speedup
- Single-branch document (research paper): no change (only 1 branch at each level)
- 2-branch document: ~1.8x faster
- 3-branch document (financial 10-K): ~2.7x faster
- The fine-tuned v8 model often selects 2-3 branches on ambiguous questions

---

## Feature 2 — AsyncGenerator Streaming

### Design goal
Keep `query()` as the simple non-streaming entry point (backward compatible).
Add `query_stream()` as the new streaming entry point.

### Event types

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class TreeLoadedEvent:
    type: Literal["tree_loaded"] = "tree_loaded"
    node_count: int = 0
    page_count: int = 0

@dataclass
class NavigatingEvent:
    type: Literal["navigating"] = "navigating"
    level: int = 0
    exploring: list[str] = None      # node_ids being explored
    section_titles: list[str] = None  # human-readable titles

@dataclass
class NodeFoundEvent:
    type: Literal["node_found"] = "node_found"
    node_id: str = ""
    title: str = ""
    page_range: str = ""

@dataclass
class AnswerEvent:
    type: Literal["answer"] = "answer"
    text: str = ""
    citations: list[str] = None
    nodes_examined: int = 0
    estimated_cost_usd: float = 0.0

@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    message: str = ""
    partial_nodes: list[str] = None

ArborEvent = TreeLoadedEvent | NavigatingEvent | NodeFoundEvent | AnswerEvent | ErrorEvent
```

### New `query_stream()` function (in `arbor/core/rag_pipeline.py`)

```python
async def query_stream(
    document: Union[str, BytesIO],
    question: str,
    provider: LLMProvider,
    config: Optional[ArborConfig] = None,
    tree: Optional[DocumentTree] = None,
    preference: Optional[str] = None,
) -> AsyncGenerator[ArborEvent, None]:
    """
    Streaming version of query(). Yields ArborEvent objects as the pipeline progresses.
    Use this for real-time UIs, APIs, and demos.

    Example:
        async for event in arbor.query_stream(pdf_path, question, provider):
            if event.type == "navigating":
                print(f"Exploring: {event.section_titles}")
            elif event.type == "answer":
                print(f"Answer: {event.text}")
    """
    if config is None:
        config = ArborConfig()

    try:
        # Step 1: Tree generation
        if tree is None:
            tree = await generate_tree(document, provider, config)

        node_count = _count_nodes(tree.structure)
        page_count = getattr(tree, 'page_count', 0)
        yield TreeLoadedEvent(node_count=node_count, page_count=page_count)

        # Step 2: Streaming tree search
        # Pass a callback to tree_searcher that yields navigation events
        navigation_events: asyncio.Queue[ArborEvent] = asyncio.Queue()

        async def on_navigate(level: int, node_ids: list[str], nodes: list):
            titles = [n.title for n in nodes if n.node_id in node_ids]
            await navigation_events.put(
                NavigatingEvent(level=level, exploring=node_ids, section_titles=titles)
            )

        async def on_node_found(node):
            pages = f"{node.start_index}-{node.end_index}"
            await navigation_events.put(
                NodeFoundEvent(node_id=node.node_id, title=node.title, page_range=pages)
            )

        # Run search_tree in background task, drain event queue
        search_task = asyncio.create_task(
            search_tree(
                tree, question, provider,
                preference=preference,
                multihop=True,
                config=config,
                on_navigate=on_navigate,       # NEW callback param
                on_node_found=on_node_found,   # NEW callback param
            )
        )

        # Yield events as they arrive, until search_task completes
        while not search_task.done():
            try:
                event = navigation_events.get_nowait()
                yield event
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)

        # Drain remaining events
        while not navigation_events.empty():
            yield navigation_events.get_nowait()

        search_result = await search_task

        # Step 3: Answer generation
        pages = get_page_contents(document)
        add_node_text(tree.structure, pages)
        node_map = create_node_mapping(tree.structure)

        context_parts = []
        citations = []
        for node_id in search_result.node_ids:
            node = node_map.get(node_id)
            if node and node.text:
                context_parts.append(f"[{node.title}]\n{node.text}")
                citations.append(f"{node.title} (pages {node.start_index}-{node.end_index})")

        if context_parts:
            prompt = answer_generation_prompt(question, "\n\n".join(context_parts))
            answer = await provider.complete_with_retry(prompt)
        else:
            answer = "No relevant sections found."

        yield AnswerEvent(
            text=answer,
            citations=citations,
            nodes_examined=getattr(search_result, 'nodes_examined', len(search_result.node_ids)),
        )

    except BudgetExceededError as e:
        yield ErrorEvent(message=str(e), partial_nodes=e.partial_nodes)
    except Exception as e:
        yield ErrorEvent(message=f"Pipeline error: {e}")
```

### Backward-compatible `query()` wrapper

```python
async def query(
    document: Union[str, BytesIO],
    question: str,
    provider: LLMProvider,
    config: Optional[ArborConfig] = None,
    tree: Optional[DocumentTree] = None,
    preference: Optional[str] = None,
) -> RAGResponse:
    """Original blocking interface. Internally uses query_stream() and collects final answer."""
    answer_event = None
    search_result_nodes = []
    citations = []

    async for event in query_stream(document, question, provider, config, tree, preference):
        if isinstance(event, NodeFoundEvent):
            search_result_nodes.append(event.node_id)
        elif isinstance(event, AnswerEvent):
            answer_event = event
            citations = event.citations or []
        elif isinstance(event, ErrorEvent):
            raise BudgetExceededError(event.message, event.partial_nodes or [])

    if answer_event is None:
        return RAGResponse(answer="No answer generated.", search_result=None, context="", citations=[])

    return RAGResponse(
        answer=answer_event.text,
        search_result=SearchResult(node_ids=search_result_nodes, thinking="", nodes=[]),
        context="",
        citations=citations,
    )
```

### Export `query_stream` from `arbor/__init__.py`

Add to `__all__`:
```python
"query_stream",
"TreeLoadedEvent", "NavigatingEvent", "NodeFoundEvent", "AnswerEvent", "ErrorEvent", "ArborEvent",
"BudgetExceededError",
```

---

## Feature 1 — MCP Server

### What it is
An MCP (Model Context Protocol) server that wraps Arbor's pipeline as tools. Once deployed, any MCP-compatible client (Claude Code, Cursor, Windsurf, Claude Desktop) can call Arbor directly — no integration code, no API keys needed by the client.

### Dependencies
```
pip install mcp
```
The `mcp` Python package is the official Python SDK at `github.com/modelcontextprotocol/python-sdk`.

### File: `arbor/mcp_server.py`

```python
"""
Arbor MCP Server — exposes Arbor pipeline as MCP tools.

Usage (STDIO — for Claude Code, Cursor, Claude Desktop):
    python -m arbor.mcp_server

Usage (HTTP — for web clients):
    python -m arbor.mcp_server --http --port 8080

Add to Claude Code (.mcp.json in project root):
    {
      "mcpServers": {
        "arbor": {
          "command": "python",
          "args": ["-m", "arbor.mcp_server"],
          "env": {"GEMINI_API_KEY": "your_key"}
        }
      }
    }
"""

import argparse
import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

import arbor
from arbor.providers.gemini_provider import GeminiProvider


def _get_default_provider() -> arbor.LLMProvider:
    """Build provider from environment variables."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY environment variable."
    )


def create_server() -> Server:
    server = Server("arbor")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="query_document",
                description=(
                    "Query a PDF document using Arbor's vectorless RAG pipeline. "
                    "Parses the PDF into a hierarchical tree, navigates the tree to find "
                    "relevant sections, and generates a grounded answer with citations. "
                    "No vector database required."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["pdf_path", "question"],
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Absolute path to the PDF file to query",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer from the document",
                        },
                        "max_hops": {
                            "type": "integer",
                            "description": "Max tree levels to navigate (default: 5)",
                            "default": 5,
                        },
                        "max_cost_usd": {
                            "type": "number",
                            "description": "Max API spend in USD (default: 0.50, 0 = unlimited)",
                            "default": 0.50,
                        },
                    },
                },
            ),
            Tool(
                name="generate_tree",
                description=(
                    "Parse a PDF into a hierarchical JSON tree structure. "
                    "Returns the tree with section titles, page ranges, node IDs, and summaries. "
                    "Use this to inspect document structure before querying."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["pdf_path"],
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Absolute path to the PDF file",
                        },
                        "add_summaries": {
                            "type": "boolean",
                            "description": "Generate LLM summaries for each node (default: true)",
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="search_tree",
                description=(
                    "Search a pre-generated Arbor document tree for nodes relevant to a question. "
                    "Uses multi-hop navigation: navigates level-by-level, 10-20 sections at a time. "
                    "Returns relevant node IDs, titles, page ranges, and navigation reasoning."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["tree_json", "question"],
                    "properties": {
                        "tree_json": {
                            "type": "string",
                            "description": "JSON string of a DocumentTree (from generate_tree output)",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to find relevant sections for",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        provider = _get_default_provider()

        if name == "query_document":
            pdf_path = arguments["pdf_path"]
            question = arguments["question"]
            config = arbor.ArborConfig(
                max_hops=arguments.get("max_hops", 5),
                max_cost_usd=arguments.get("max_cost_usd", 0.50),
                add_summaries=True,
            )

            # Collect streaming events for rich output
            events = []
            answer_text = ""
            citations = []

            async for event in arbor.query_stream(pdf_path, question, provider, config):
                if isinstance(event, arbor.NavigatingEvent):
                    events.append(f"→ Level {event.level}: exploring {event.section_titles}")
                elif isinstance(event, arbor.NodeFoundEvent):
                    events.append(f"✓ Found: {event.title} (pages {event.page_range})")
                elif isinstance(event, arbor.AnswerEvent):
                    answer_text = event.text
                    citations = event.citations or []
                elif isinstance(event, arbor.ErrorEvent):
                    return [TextContent(type="text", text=f"Error: {event.message}")]

            output = []
            output.append(f"**Answer:** {answer_text}")
            if citations:
                output.append(f"\n**Sources:**")
                for c in citations:
                    output.append(f"  - {c}")
            if events:
                output.append(f"\n**Navigation path:**")
                for e in events:
                    output.append(f"  {e}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "generate_tree":
            pdf_path = arguments["pdf_path"]
            config = arbor.ArborConfig(
                add_summaries=arguments.get("add_summaries", True),
                add_node_ids=True,
            )
            tree = await arbor.generate_tree(pdf_path, provider, config)
            return [TextContent(type="text", text=json.dumps(tree.to_dict(), indent=2))]

        elif name == "search_tree":
            tree_data = json.loads(arguments["tree_json"])
            question = arguments["question"]
            # Reconstruct DocumentTree from dict
            from arbor.types import DocumentTree
            tree = DocumentTree.from_dict(tree_data)
            result = await arbor.search_tree(tree, question, provider, multihop=True)
            output = {
                "node_ids": result.node_ids,
                "thinking": result.thinking,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "title": n.title,
                        "pages": f"{n.start_index}-{n.end_index}",
                    }
                    for n in (result.nodes or [])
                ],
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _run_stdio():
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="Arbor MCP Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP server instead of STDIO")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args()

    if args.http:
        # HTTP/SSE transport for web clients
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        import uvicorn

        server = create_server()
        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse.handle_post_message),
        ])
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
```

### `.mcp.json` config file for users (place in project root)

```json
{
  "mcpServers": {
    "arbor": {
      "command": "python",
      "args": ["-m", "arbor.mcp_server"],
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

### How users will use it

**In Claude Code:**
```
> Use the arbor tool to find what the capital expenditure was in this 10-K
[Claude Code calls arbor.query_document automatically]
→ Level 1: exploring ['Financial Statements', 'Business Overview']
✓ Found: Capital Expenditure Summary (pages 45-47)
Answer: The FY2023 capital expenditure was $2.4 billion...
```

**In any MCP client:**
- Install: `pip install arbor-rag`
- Add `.mcp.json` to project
- No other setup

---

## Build Order & Testing Checklist

### Step 1: Feature 3 — Budget Controls
- [ ] Add fields to `ArborConfig` in `arbor/types.py`
- [ ] Add `BudgetExceededError` to `arbor/types.py`
- [ ] Add `_BudgetTracker` to `arbor/core/tree_searcher.py`
- [ ] Wire tracker into `_search_multihop()`
- [ ] Wire `asyncio.wait_for` into `query()`
- [ ] Export `BudgetExceededError` from `arbor/__init__.py`
- [ ] Test: `config = ArborConfig(max_hops=1)` should stop after 1 level
- [ ] Test: `config = ArborConfig(max_nodes_searched=5)` should stop after 5 nodes

### Step 2: Feature 4 — Schema Enforcement
- [ ] Add `_enforce_navigate_schema()` to `arbor/core/tree_searcher.py`
- [ ] Add `_try_parse()` helper
- [ ] Replace `_parse_navigate_response()` calls with `_enforce_navigate_schema()`
- [ ] Test: simulate bad JSON response → should retry → should return empty nav on failure
- [ ] Test: simulate hallucinated node ID → should be filtered out

### Step 3: Feature 5 — Parallel Navigation
- [ ] Refactor `navigate_level()` to use `asyncio.gather` for recursion
- [ ] Confirm `final_node_ids.append()` is safe (asyncio single-thread = no race condition)
- [ ] Test: document with 3-branch navigation → all 3 branches explored
- [ ] Benchmark: measure wall-clock time before/after on a 5-level document

### Step 4: Feature 2 — Streaming
- [ ] Add event dataclasses to `arbor/types.py`
- [ ] Add `on_navigate` and `on_node_found` callback params to `search_tree()`
- [ ] Implement `query_stream()` in `arbor/core/rag_pipeline.py`
- [ ] Refactor `query()` to wrap `query_stream()`
- [ ] Export new symbols from `arbor/__init__.py`
- [ ] Test: streaming emits `TreeLoadedEvent` first
- [ ] Test: streaming emits at least 1 `NavigatingEvent`
- [ ] Test: streaming emits `AnswerEvent` last
- [ ] Test: `query()` still returns `RAGResponse` (backward compat)

### Step 5: Feature 1 — MCP Server
- [ ] `pip install mcp` (add to requirements.txt / pyproject.toml)
- [ ] Create `arbor/mcp_server.py`
- [ ] Add `__main__` block so `python -m arbor.mcp_server` works
- [ ] Create `.mcp.json` in repo root
- [ ] Test STDIO: `echo '{"method":"tools/list"}' | python -m arbor.mcp_server`
- [ ] Test in Claude Code: add `.mcp.json`, restart, `/mcp` command shows arbor server
- [ ] Test `query_document` tool end-to-end with a real PDF
- [ ] Test `generate_tree` returns valid JSON
- [ ] Update README with MCP setup instructions

---

## Key implementation notes

**asyncio safety in parallel navigation**
`final_node_ids.append()` is safe without locks in asyncio because Python's event loop is single-threaded. Two coroutines cannot append simultaneously — only one runs at a time between await points.

**Provider semaphore still applies**
The existing `asyncio.Semaphore(config.max_concurrent_llm_calls)` in each provider limits total concurrent API calls even with parallel navigation. Default is 5. This is the correct throttle.

**DocumentTree.from_dict() may not exist**
The MCP server's `search_tree` tool deserializes a tree from JSON. Check if `DocumentTree.from_dict()` exists — if not, build a simple reconstructor or serialize/deserialize through the existing `to_dict()` format. The tree's structure is just nested dicts that map directly to `TreeNode` objects.

**`query_stream()` callback approach vs queue**
The plan uses an `asyncio.Queue` to bridge the `search_tree()` coroutine (which needs to run to completion) with the async generator (which yields). An alternative is to pass callbacks directly into `search_tree()` and `yield` from those callbacks — but generators can't yield from callbacks in Python. The queue approach is correct.

**MCP `mcp` package version**
The Claude Code MCP server uses `@modelcontextprotocol/sdk ^1.12.1`. The Python equivalent is `mcp>=1.0.0`. The API surface: `Server`, `stdio_server`, `Tool`, `TextContent`, `CallToolResult` are stable.
