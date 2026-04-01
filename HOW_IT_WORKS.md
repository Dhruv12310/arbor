# How Arbor Works — A Beginner's Deep Dive

> Written for a curious learner who wants to understand every decision, not just run the code.
> No prior RAG or ML systems knowledge assumed — just Python and a willingness to think carefully.

---

## Table of Contents

1. [The Big Picture — What Arbor Actually Does](#1-the-big-picture)
2. [Feature 3 — Budget Controls (Safety Rails)](#2-feature-3--budget-controls)
3. [Feature 4 — Schema Enforcement (Sanity Checking the AI)](#3-feature-4--schema-enforcement)
4. [Feature 5 — Parallel Branch Navigation (Going Faster)](#4-feature-5--parallel-branch-navigation)
5. [Feature 2 — AsyncGenerator Streaming (Watching in Real Time)](#5-feature-2--asyncgenerator-streaming)
6. [Feature 1 — MCP Server (Talking to Claude Code)](#6-feature-1--mcp-server)
7. [How All 5 Features Fit Together](#7-how-all-5-features-fit-together)

---

## 1. The Big Picture

Before understanding any single feature, you need a mental model of what Arbor does end-to-end.

### The problem with traditional RAG

The standard way to answer questions about documents is called RAG (Retrieval-Augmented Generation). It works like this:

1. Split your document into chunks (e.g., every 500 words)
2. Convert each chunk into a vector (a list of ~1500 numbers) using an embedding model
3. When a question arrives, convert the question into a vector too
4. Find the chunks whose vectors are "close" to the question vector (cosine similarity)
5. Feed those chunks to an LLM and ask it to answer

This seems elegant but fails in ways that are hard to debug:

- **Semantic drift**: "What is the capital expenditure?" and "What did they spend on buildings?" might not have close vectors even though they mean the same thing
- **Context loss**: A chunk that says "It decreased by 12%" is meaningless without knowing what "it" refers to — the chunk lost that context
- **Infrastructure cost**: You need to run an embedding model AND store a vector database

### The Arbor approach: navigate a tree instead

Arbor treats a document like a filing cabinet. It doesn't randomly sample drawers — it reads the labels on the drawers, picks the right one, opens it, and reads. The process:

```
PDF
 ↓
[TreeGen model] — reads the document, creates an outline like a table of contents
 ↓
JSON Tree (hierarchical structure: chapters → sections → subsections)
 ↓
[TreeSearch model] — given your question, navigates the tree level-by-level
                     like a librarian who knows which shelf to check
 ↓
Relevant nodes (the specific pages/sections that contain the answer)
 ↓
[Answer LLM] — reads only those pages and generates the final answer
```

### What a "tree" looks like

Imagine a 200-page textbook. After TreeGen runs, you get a JSON structure like this:

```
Chapter 1: Introduction (pages 1–15)
├── 1.1 Background (pages 1–5)
├── 1.2 Problem Statement (pages 6–10)
└── 1.3 Our Contributions (pages 11–15)
Chapter 2: Methods (pages 16–60)
├── 2.1 Data Collection (pages 16–25)
├── 2.2 Model Architecture (pages 26–45)
│   ├── 2.2.1 Encoder (pages 26–32)
│   └── 2.2.2 Decoder (pages 33–45)
└── 2.3 Training Details (pages 46–60)
...
```

Each box in this tree is called a **node**. Each node has:
- A title (what it's about)
- A page range (where it lives in the PDF)
- A node_id (like "0003" — a 4-digit ID used to refer to it)
- Children (sub-sections, if any)

### What "multihop navigation" means

When you ask "What is the encoder architecture?", the TreeSearch model doesn't see all 300 nodes at once (that would overflow its context window). Instead, it navigates level-by-level:

**Hop 1** — sees only top-level chapters:
```
[0001] Chapter 1: Introduction (pages 1-15) [has sub-sections]
[0002] Chapter 2: Methods (pages 16-60) [has sub-sections]
[0003] Chapter 3: Results (pages 61-90) [has sub-sections]
...
Model says: navigate_to ["0002"]  ← "Methods probably has architecture info"
```

**Hop 2** — zooms into Chapter 2's children:
```
[0004] 2.1 Data Collection (pages 16-25)
[0005] 2.2 Model Architecture (pages 26-45) [has sub-sections]
[0006] 2.3 Training Details (pages 46-60)
Model says: navigate_to ["0005"]  ← "Model Architecture is obviously right"
```

**Hop 3** — zooms into 2.2's children:
```
[0007] 2.2.1 Encoder (pages 26-32)
[0008] 2.2.2 Decoder (pages 33-45)
Model says: navigate_to ["0007"]  ← "Encoder specifically"
```

Node 0007 is a leaf (no children) → answer is on pages 26–32. Done. The model made 3 small decisions instead of 1 giant one. Each decision fits comfortably in the context window.

---

## 2. Feature 3 — Budget Controls

### Why this needs to exist

Imagine you deploy Arbor in a product. A user uploads a 10,000-page legal archive and asks a vague question. Without any limits:

- The model might explore 500 nodes × 3 LLM calls each = 1500 LLM API calls
- That might cost $5 per query
- It might take 10 minutes
- If something loops, it might never stop

Budget controls are **safety rails** — they let you promise users "this will answer in under 30 seconds and cost under $0.50."

### What gets controlled

Four independent limits, any of which can fire:

| Limit | What it controls | Default |
|-------|-----------------|---------|
| `max_hops` | How many levels deep we navigate | 5 |
| `max_nodes_searched` | Total nodes examined across all levels | 100 |
| `max_cost_usd` | Estimated API spend | $0.50 |
| `timeout_sec` | Wall-clock time for the whole query | 120 seconds |

Users set these in `ArborConfig`:

```python
config = ArborConfig(
    max_hops=3,            # don't go deeper than 3 levels
    max_nodes_searched=50, # stop after examining 50 nodes
    timeout_sec=30.0,      # hard stop after 30 seconds
)
response = await arbor.query(pdf_path, question, provider, config=config)
```

### How it's implemented: the `_BudgetTracker`

Inside `tree_searcher.py`, there's an internal dataclass that tracks consumption:

```python
@dataclass
class _BudgetTracker:
    max_hops: int = 5
    max_nodes: int = 0         # 0 means "no limit"
    timeout: float = 0.0       # 0 means "no limit"
    start_time: float = ...    # set to current time when tracker is created
    nodes_examined: int = 0    # increases as we explore
```

At each step during navigation, we call `budget.check()`:

```python
def check(self, partial_ids):
    # Has the clock run out?
    if self.timeout > 0 and (time.monotonic() - self.start_time) >= self.timeout:
        raise BudgetExceededError(f"Timeout after {self.timeout}s", partial_ids)
    
    # Have we looked at too many nodes?
    if self.max_nodes > 0 and self.nodes_examined >= self.max_nodes:
        raise BudgetExceededError(f"Exceeded max_nodes_searched={self.max_nodes}", partial_ids)
```

`time.monotonic()` is a system clock that only goes forward — it's immune to daylight saving changes or clock adjustments.

### Why `partial_ids` matters

When a budget limit fires, we might have already found some useful nodes. The `BudgetExceededError` carries `partial_nodes` — the node IDs we found before the cutoff. In streaming mode, these get surfaced to the user with an `ErrorEvent` that says "hit the limit, but here's what we found so far." The user still gets a partial answer rather than nothing.

### Hop depth checking

Before entering any level of navigation, we check depth:

```python
def check_hops(self, depth, partial_ids):
    if depth > self.max_hops:
        raise BudgetExceededError(f"Exceeded max_hops={self.max_hops}", partial_ids)
```

If `max_hops=3` and we're about to enter depth 4, we stop and return whatever we found at depth 3. This prevents runaway recursion in deeply-nested documents.

### The timeout wrapper in `query_stream()`

Individual `budget.check()` calls catch timeouts during navigation, but what about tree generation or answer writing? For those, we wrap the entire pipeline in an `asyncio.wait_for`:

```python
timeout = config.timeout_sec if config.timeout_sec > 0 else None
task = asyncio.create_task(
    asyncio.wait_for(_pipeline(), timeout=timeout)
)
```

`asyncio.wait_for(coro, timeout=N)` cancels the coroutine if it hasn't finished in N seconds. It raises `asyncio.TimeoutError`, which we catch and convert to an `ErrorEvent`.

---

## 3. Feature 4 — Schema Enforcement

### The problem this solves

TreeSearch is a fine-tuned language model. Language models don't always follow instructions perfectly. When Arbor asks "which of these sections should I explore?", the model is supposed to reply with exactly:

```json
{"thinking": "The answer is in Financial Statements", "navigate_to": ["0003", "0007"]}
```

But sometimes it might return:

```json
{"reasoning": "Financial Statements", "sections": ["0003"]}   ← wrong field names
```

```
I think you should look at nodes 0003 and 0007.                ← prose instead of JSON
```

```json
{"thinking": "...", "navigate_to": ["0099", "1234"]}          ← IDs that don't exist at this level
```

Without schema enforcement, Arbor silently gets zero results at this level and returns a bad answer. With schema enforcement, Arbor retries with a corrective prompt.

### The three-layer parsing approach

**Layer 1 — JSON parse**: Try `json.loads(response)`. This handles the clean case.

**Layer 2 — Regex extraction**: If JSON parsing fails (model wrapped it in markdown code fences, added a preamble), use a regex to find the JSON object:
```python
match = re.search(r'\{.*\}', text, re.DOTALL)
```
`re.DOTALL` makes `.` match newlines, so it captures multi-line JSON.

**Layer 3 — Last resort**: Extract any 4-digit numbers from the text and treat them as node IDs.

### ID filtering (not rejection)

Once we have a parsed dict, we validate `navigate_to`. The key design decision: **filter invalid IDs rather than rejecting the whole response**.

```python
# Filter out hallucinated IDs rather than rejecting the whole response
data["navigate_to"] = [str(n) for n in nav if str(n) in valid_ids]
```

If the model returns `["0003", "9999"]` and `9999` doesn't exist at this level, we accept `["0003"]` and move on. We don't penalize the model for one hallucinated ID by discarding the valid ones.

This matters because language models trained with LoRA on small datasets sometimes confuse node IDs from different documents seen during training. Filtering makes the system resilient to this.

### The retry loop

If the response is completely unparseable (not a dict, no `navigate_to` key at all), we retry with a correction prompt:

```python
for attempt in range(max_retries + 1):
    response = await provider.complete_with_retry(prompt)
    data = _parse_navigate_response(response)
    
    if _enforce_navigate_schema(data, valid_ids):
        return data  # success
    
    # Failed — build a corrective prompt explaining what went wrong
    correction = (
        f"Your previous response was not valid JSON or contained unknown node IDs. "
        f"Valid IDs are: {sorted(valid_ids)}. "
        f'Respond ONLY with: {{"thinking": "...", "navigate_to": ["id1"]}}'
    )
    prompt = correction_prompt  # retry with this
```

`max_retries` comes from `config.max_retries_on_bad_json` (default: 2). So the model gets up to 3 total attempts before we give up and return an empty navigation result for this level.

### Why this is architecturally important

Schema enforcement is what makes the difference between a research demo and a production system. Research demos fail gracefully — "sometimes it gives weird output." Production systems fail predictably — "if it fails, here is exactly what happens and here is the recovery path."

---

## 4. Feature 5 — Parallel Branch Navigation

### The sequential problem

Imagine the TreeSearch model navigates hop 1 and selects two branches:
```
navigate_to: ["0002", "0008"]
```

Both "Chapter 2: Methods" and "Chapter 8: Appendix" might contain relevant information. In the original sequential code:

```python
for node in selected:
    if node.nodes:
        await navigate_level(node.nodes, depth + 1)  # wait for this to finish...
                                                      # THEN start the next one
```

We explore Chapter 2 completely, THEN start exploring Chapter 8. If each branch takes 2 seconds of LLM calls, the total is 4 seconds.

### The parallel solution

These two explorations are **completely independent** — Chapter 2's result doesn't influence Chapter 8's navigation. They can run at the same time:

```python
recurse_tasks = []
for node in selected:
    if node.nodes:
        recurse_tasks.append(navigate_level(node.nodes, depth + 1))

if recurse_tasks:
    await asyncio.gather(*recurse_tasks)  # launch all at once, wait for all to finish
```

`asyncio.gather(*tasks)` takes a list of coroutines and runs them concurrently. Because Python's async uses a single-threaded event loop (no true parallelism like threads), they don't actually run simultaneously at the CPU level — but while one is waiting for an LLM API response (network I/O), the other can send its request. This is exactly where async shines: **I/O-bound tasks that spend most of their time waiting**.

With 3 branches each taking 2 seconds: sequential = 6 seconds, parallel = ~2 seconds (limited by the slowest branch).

### Thread safety concern — and why it doesn't apply here

When multiple things write to the same list simultaneously, you usually get race conditions. But here:

```python
final_node_ids.append(node.node_id)  # called from multiple branches
```

This is safe because Python's `asyncio` is **single-threaded**. There is no actual simultaneous execution. The event loop runs one coroutine at a time, switching between them only at `await` points. Two coroutines can never execute the same line of code at the exact same millisecond. So `list.append()` is safe without locks.

This is one of the most important properties of asyncio to internalize: concurrent ≠ parallel. Concurrent means "making progress on multiple things by interleaving," not "running at the exact same instant."

### The provider semaphore still applies

Even though branches navigate in parallel, the number of concurrent LLM API calls is already capped. Each provider has an `asyncio.Semaphore(config.max_concurrent_llm_calls)` (default: 5). A semaphore is a counter that decrements when you enter a block and increments when you leave. If the count hits 0, new callers wait. This means even if you have 10 branches in parallel, at most 5 LLM calls are in-flight at once.

---

## 5. Feature 2 — AsyncGenerator Streaming

### What "streaming" means and why it matters

Without streaming, `query()` works like this:
1. Call it
2. Wait (maybe 30 seconds)
3. Get the answer

To a user, this looks like the application is frozen. There's no feedback about what's happening.

With streaming, `query_stream()` works like this:
1. Call it
2. Immediately start receiving events:
   - "Tree loaded: 47 nodes, 120 pages"
   - "Level 1: exploring ['Introduction', 'Methods', 'Results']"
   - "Level 2: exploring ['Data Collection', 'Model Architecture']"
   - "Found: Section 2.2 Model Architecture (pages 26–45)"
   - "Answer: The encoder uses a transformer with 12 heads..."
3. Total time same as before, but user sees progress the whole time

This is how Claude Code shows you "Searching files... Reading file... Writing code..." instead of a spinning cursor for 30 seconds.

### The AsyncGenerator type

An `AsyncGenerator` is a function that uses `async def` AND `yield`. It produces values one at a time and can be iterated with `async for`:

```python
async for event in arbor.query_stream(path, question, provider):
    if isinstance(event, arbor.NavigatingEvent):
        print(f"Navigating level {event.level}...")
    elif isinstance(event, arbor.AnswerEvent):
        print(f"Answer: {event.text}")
```

Each call to `async for` resumes the generator until the next `yield`. The generator pauses at each `yield`, suspends control back to the caller, and only resumes when the caller asks for the next item.

### The bridge problem

Here's a tricky design challenge. We want:
- `query_stream()` to be an async generator that `yield`s events
- `search_tree()` to emit events via a callback during navigation

But **you can't `yield` from inside a callback in Python**. If `search_tree()` calls `event_cb(NavigatingEvent(...))`, that callback can't reach up and yield from the outer generator.

The solution is a **queue bridge**:

```
query_stream()                           _pipeline()
[async generator]                        [background task]
      |                                       |
      |   ← queue.get() ←   queue   ← queue.put(event) ← event_cb
      |                                       |
      yield event                        search_tree(event_cb=_push)
```

Step by step:

1. `query_stream()` creates an `asyncio.Queue`
2. Defines `_push(event)` which puts events into the queue
3. Launches `_pipeline()` as a background `asyncio.Task` — this runs concurrently
4. `_pipeline()` calls `search_tree(event_cb=_push)` — as navigation happens, events go into the queue
5. Meanwhile, `query_stream()` loops: `evt = await queue.get()` — this suspends until something is in the queue
6. When an event arrives, `yield evt` — this passes it to the caller
7. When `_pipeline()` finishes, it puts `None` into the queue (a sentinel)
8. The generator sees `None`, breaks the loop, cleans up

```python
task = asyncio.create_task(_pipeline())  # launch in background

while True:
    evt = await queue.get()  # suspend until next event
    if evt is None:           # sentinel = pipeline done
        break
    yield evt                 # give event to caller
```

### The five event types

Every event is a dataclass (a Python class that's really just a named tuple with default values):

```python
@dataclass
class TreeLoadedEvent:
    type: str = "tree_loaded"
    node_count: int = 0    # total nodes in the tree
    page_count: int = 0    # total pages in the document

@dataclass
class NavigatingEvent:
    type: str = "navigating"
    level: int = 0                     # which hop (1, 2, 3...)
    exploring_ids: list = []           # node IDs being looked at
    section_titles: list = []          # human-readable section names

@dataclass
class NodeFoundEvent:
    type: str = "node_found"
    node_id: str = ""       # the specific node ID confirmed as an answer location
    title: str = ""         # section title
    page_range: str = ""    # e.g. "26-45"

@dataclass
class AnswerEvent:
    type: str = "answer"
    text: str = ""              # the final answer text
    citations: list = []        # page/section references used
    nodes_examined: int = 0     # how many nodes were looked at total

@dataclass
class ErrorEvent:
    type: str = "error"
    message: str = ""         # what went wrong
    partial_nodes: list = []  # nodes found before the error
```

### How `query()` uses `query_stream()` internally

`query()` is the backwards-compatible blocking interface. It just collects events from `query_stream()` and packages them into a `RAGResponse`:

```python
async def query(...) -> RAGResponse:
    answer_event = None
    found_node_ids = []
    
    async for event in query_stream(document, question, provider, config, tree, preference):
        if isinstance(event, NodeFoundEvent):
            found_node_ids.append(event.node_id)
        elif isinstance(event, AnswerEvent):
            answer_event = event
        elif isinstance(event, ErrorEvent):
            raise BudgetExceededError(event.message, event.partial_nodes)
    
    return RAGResponse(
        answer=answer_event.text,
        ...
    )
```

This is called the "wrapper pattern." `query()` gets the timeout, budget controls, and streaming internals for free because `query_stream()` already handles all of that. There's only one place where the pipeline logic lives — less code to maintain, fewer bugs.

---

## 6. Feature 1 — MCP Server

### What MCP is

MCP stands for **Model Context Protocol**. It's an open standard invented by Anthropic that defines how LLM-based applications (like Claude Code, Cursor, Windsurf) can discover and call external tools.

Think of it like USB for AI tools. Before USB, every device needed its own proprietary connector. MCP is like USB-C — any MCP-compatible client can talk to any MCP-compatible server using the same protocol, without custom integration code.

Without MCP, to use Arbor inside Claude Code you would need to:
- Write a Claude Code extension
- Register custom slash commands
- Handle auth, argument passing, response formatting yourself

With MCP, you write one server, and any MCP client (Claude Code, Cursor, Claude Desktop, your own web app) can use it immediately.

### The transport layer: STDIO vs HTTP/SSE

MCP supports two transports:

**STDIO** (Standard Input/Output): The simplest. The client launches `python -m arbor.mcp_server` as a subprocess. Messages are JSON sent over stdin/stdout. This is what Claude Code and Claude Desktop use — zero network setup, zero ports.

```
Claude Code ──(stdin/stdout)──> python -m arbor.mcp_server
```

**HTTP/SSE** (Server-Sent Events): For web clients. Arbor runs as an HTTP server. The client connects to `/sse` and receives a stream of events. This is what you'd use in a web app or API.

```
Web Browser ──(HTTP GET /sse)──> arbor server on localhost:8080
```

### The three tools exposed

Arbor's MCP server exposes three tools that any MCP client can discover and call:

**`query_document`** — The full pipeline: PDF → answer with citations.
```
Input: { "pdf_path": "/path/to/report.pdf", "question": "What was revenue in Q3?" }
Output: Answer text + sources + navigation path
```

**`generate_tree`** — Just the first stage: PDF → JSON tree.
```
Input: { "pdf_path": "/path/to/report.pdf", "add_summaries": true }
Output: Full JSON tree (useful for inspecting document structure)
```

**`search_tree`** — Just the second stage: tree + question → relevant nodes.
```
Input: { "tree_json": "...", "question": "What was revenue in Q3?" }
Output: { "node_ids": ["0013"], "thinking": "...", "nodes": [...] }
```

Exposing all three separately means a user can cache the tree (tree generation is the slow part) and reuse it for multiple questions.

### How MCP tools are defined in code

Each tool needs two things: a JSON Schema describing its inputs, and a handler function.

```python
Tool(
    name="query_document",
    description="Query a PDF...",
    inputSchema={
        "type": "object",
        "required": ["pdf_path", "question"],
        "properties": {
            "pdf_path": {"type": "string", "description": "Absolute path to PDF"},
            "question": {"type": "string", "description": "Question to answer"},
        }
    }
)
```

The `inputSchema` is standard JSON Schema. The MCP client uses this to validate arguments before sending them, and to show the user what fields are available.

The handler is decorated with `@server.call_tool()`:
```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_document":
        # ... run the pipeline, return results
        return [TextContent(type="text", text="Answer: ...")]
```

`TextContent` is the MCP type for a text response. Everything the tool returns must be wrapped in one of MCP's content types.

### The `.mcp.json` config file

To tell Claude Code about the Arbor MCP server, you add a `.mcp.json` file to your project root:

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

Claude Code reads this file, sees "there's a server called `arbor`", launches the subprocess, and makes the three tools available. The `${GEMINI_API_KEY}` means "read from the environment variable of the same name" — API keys stay in your shell environment, not in source control.

### What it looks like in practice

Once configured, you can just talk to Claude Code naturally:

```
You:    Use the arbor tool to find what 3M's capital expenditure was in FY2022

Claude: [calls arbor.query_document with the 10-K PDF path and the question]

        Navigation path:
        → Level 1: exploring ['Business Overview', 'Financial Statements', 'Legal Proceedings']
        ✓ Found: Capital Expenditures (pages 48-50)
        
        Answer: 3M's capital expenditure for FY2022 was $1.7 billion, representing
        a decrease of 8% from FY2021's $1.85 billion, primarily due to reduced
        spending on manufacturing capacity expansion.
        
        Sources:
          - Capital Expenditures (pages 48-50)
```

No copy-pasting PDFs. No writing prompt templates. Claude Code calls Arbor, Arbor navigates the document, Claude presents the answer.

---

## 7. How All 5 Features Fit Together

This is the complete picture of what happens when you call `query_stream()`:

```
arbor.query_stream(pdf_path, question, provider, config)
│
├─ [config: max_hops=3, max_nodes=50, timeout=30s, max_retries=2]
│
├─ Step 1: generate_tree(pdf_path)
│   └─ Emits: TreeLoadedEvent(node_count=47, page_count=120)
│
├─ Step 2: search_tree(tree, question, multihop=True, config=config)
│   │
│   └─ _search_multihop()
│       │
│       ├─ _BudgetTracker created ──────────────────── [Feature 3]
│       │   starts timer, tracks nodes examined
│       │
│       ├─ navigate_level(depth=1)
│       │   ├─ budget.check_hops(1) → OK ──────────── [Feature 3]
│       │   ├─ budget.check() → nodes=0, OK ─────────── [Feature 3]
│       │   ├─ Emits: NavigatingEvent(level=1, ...) ── [Feature 2]
│       │   │
│       │   ├─ LLM call → response
│       │   ├─ _try_parse_with_retry() ──────────────── [Feature 4]
│       │   │   ├─ _parse_navigate_response() → parse JSON
│       │   │   ├─ _enforce_navigate_schema() → filter invalid IDs
│       │   │   └─ retry up to 2x if completely unparseable
│       │   │
│       │   ├─ navigate_to: ["0002", "0008"]
│       │   │
│       │   └─ asyncio.gather( ───────────────────────── [Feature 5]
│       │       navigate_level(node_0002.children, depth=2),  ← parallel
│       │       navigate_level(node_0008.children, depth=2),  ← parallel
│       │      )
│       │       │
│       │       ├─ (branch A finishes, finds node 0015)
│       │       │   └─ Emits: NodeFoundEvent(node_id="0015", ...) [Feature 2]
│       │       │
│       │       └─ (branch B finishes, finds node 0041)
│       │           └─ Emits: NodeFoundEvent(node_id="0041", ...) [Feature 2]
│       │
│       └─ Returns SearchResult(node_ids=["0015", "0041"])
│
├─ Step 3: extract text from nodes 0015 and 0041
│
├─ Step 4: LLM generates answer from extracted text
│
└─ Emits: AnswerEvent(text="...", citations=[...]) ───── [Feature 2]
```

And at any point, if budget.check() fires → `BudgetExceededError` →
caught in `_pipeline()` → `ErrorEvent` pushed to queue → yielded to caller.

### The dependency order explains the build order

The plan built these in order: **3 → 4 → 5 → 2 → 1**. Here's why that order was necessary:

- **Feature 3 first** because budget controls protect Features 4 and 5. Running Feature 5 (parallel) without caps could explode into hundreds of API calls.
- **Feature 4 before Feature 5** because parallel navigation makes retries more complex. Get single-branch retries right before parallelizing.
- **Feature 5 before Feature 2** because the streaming events come FROM the navigation, so the navigation needs to work correctly first.
- **Feature 2 before Feature 1** because the MCP server uses `query_stream()` to get rich navigation events for its output. If streaming doesn't exist, the MCP server can only return a final answer.
- **Feature 1 last** because it's purely additive — it wraps everything else and doesn't change any core logic.

---

## Key Concepts Quick Reference

| Concept | What it is | Where it appears |
|---------|-----------|-----------------|
| `asyncio` | Python's async framework — runs coroutines on a single thread by interleaving them at `await` points | Everywhere |
| `await` | Suspend this coroutine until the thing on the right is ready | Every LLM call |
| `async for` | Iterate over an async generator, suspending between items | `query_stream()` loop |
| `asyncio.gather()` | Run multiple coroutines concurrently, wait for all | Feature 5 |
| `asyncio.Queue` | Thread-safe (and async-safe) FIFO queue for passing data between coroutines | Feature 2 bridge |
| `asyncio.create_task()` | Start a coroutine running in the background immediately | Feature 2 |
| `asyncio.wait_for()` | Run a coroutine with a timeout | Features 2, 3 |
| `@dataclass` | Python decorator that auto-generates `__init__`, `__repr__` from field annotations | All event types |
| `asyncio.Semaphore` | A counter-based lock that limits how many coroutines can be in a section at once | Provider concurrency cap |
| `json.loads()` | Parse a JSON string into a Python dict | Feature 4 |
| `re.DOTALL` | Regex flag that makes `.` match newlines (for multi-line JSON extraction) | Feature 4 |

---

*Written by Claude Code — April 2026*
