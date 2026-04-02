"""
Full RAG pipeline: generate tree → search → extract context → answer.

Two entry points:
  query()        — blocking, returns RAGResponse
  query_stream() — AsyncGenerator, yields event objects as the pipeline runs
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import AsyncGenerator, Optional, Union

from arbor.core.tree_generator import generate_tree
from arbor.core.tree_searcher import search_tree
from arbor.core.types import (
    AnswerEvent,
    ArborConfig,
    BudgetExceededError,
    DocumentTree,
    ErrorEvent,
    NavigatingEvent,
    NodeFoundEvent,
    RAGResponse,
    SearchResult,
    TreeLoadedEvent,
)
from arbor.prompts.tree_search import answer_generation_prompt
from arbor.providers.base import LLMProvider
from arbor.utils.tree_utils import add_node_text, create_node_mapping
from arbor.extraction.pdf_extractor import get_page_contents


async def query(
    document: Union[str, BytesIO],
    question: str,
    provider: LLMProvider,
    config: Optional[ArborConfig] = None,
    tree: Optional[DocumentTree] = None,
    preference: Optional[str] = None,
) -> RAGResponse:
    """
    Answer a question about a document using vectorless RAG.

    Internally calls query_stream() and collects the final AnswerEvent.
    Budget controls (timeout, max_hops, max_nodes_searched) from config apply.

    Raises:
        BudgetExceededError: if any budget limit is hit.
    """
    config = config or ArborConfig()
    answer_event: Optional[AnswerEvent] = None
    found_node_ids: list[str] = []

    async for event in query_stream(document, question, provider, config, tree, preference):
        if isinstance(event, NodeFoundEvent):
            found_node_ids.append(event.node_id)
        elif isinstance(event, AnswerEvent):
            answer_event = event
        elif isinstance(event, ErrorEvent):
            raise BudgetExceededError(event.message, list(event.partial_nodes or []))

    if answer_event is None:
        return RAGResponse(
            answer="No answer generated.",
            search_result=SearchResult(thinking="", node_ids=[], nodes=[]),
            context="",
            citations=[],
        )

    return RAGResponse(
        answer=answer_event.text,
        search_result=SearchResult(
            thinking="",
            node_ids=found_node_ids,
            nodes=[],
        ),
        context="",
        citations=list(answer_event.citations or []),
    )


async def query_stream(
    document: Union[str, BytesIO],
    question: str,
    provider: LLMProvider,
    config: Optional[ArborConfig] = None,
    tree: Optional[DocumentTree] = None,
    preference: Optional[str] = None,
) -> AsyncGenerator:
    """
    Streaming version of query(). Yields event objects as the pipeline runs.

    Event sequence:
        TreeLoadedEvent
        NavigatingEvent  (one per tree level explored)
        NodeFoundEvent   (one per confirmed leaf node)
        AnswerEvent      (final, contains full answer text)

    On budget/timeout exceeded:
        ErrorEvent       (with partial_nodes if any were found before cutoff)

    Example::

        async for event in query_stream(path, question, provider):
            if isinstance(event, NavigatingEvent):
                print(f"Level {event.level}: exploring {event.section_titles}")
            elif isinstance(event, AnswerEvent):
                print(event.text)
            elif isinstance(event, ErrorEvent):
                print("Error:", event.message)
    """
    config = config or ArborConfig()
    queue: asyncio.Queue = asyncio.Queue()

    async def _push(evt: object) -> None:
        await queue.put(evt)

    async def _pipeline() -> None:
        try:
            nonlocal tree
            # Step 1: Tree generation
            if tree is None:
                tree = await generate_tree(document, provider, config)

            node_count = _count_nodes(tree.structure)
            page_count = _max_page(tree.structure)
            await _push(TreeLoadedEvent(node_count=node_count, page_count=page_count))

            # Step 2: Search with event callback
            search_result = await search_tree(
                tree, question, provider,
                preference=preference,
                multihop=True,
                config=config,
                event_cb=_push,
            )

            # Step 3: Build context
            pages = get_page_contents(document)
            add_node_text(tree.structure, pages)

            node_map = create_node_mapping(tree.structure)
            context_parts: list[str] = []
            citations: list[dict] = []

            for node_id in search_result.node_ids:
                node = node_map.get(node_id)
                if node and node.text:
                    context_parts.append(f"[{node.title}]\n{node.text}")
                    citations.append({
                        "node_id": node_id,
                        "title": node.title,
                        "start_page": node.start_index,
                        "end_page": node.end_index,
                    })

            context = "\n\n---\n\n".join(context_parts)

            # Step 4: Generate answer
            if context:
                prompt = answer_generation_prompt(question, context)
                answer = await provider.complete_with_retry(prompt)
            else:
                answer = "I could not find relevant information in this document to answer your question."

            await _push(AnswerEvent(
                text=answer,
                citations=citations,
                nodes_examined=len(search_result.node_ids),
            ))

        except BudgetExceededError as exc:
            await _push(ErrorEvent(
                message=str(exc),
                partial_nodes=exc.partial_nodes,
            ))
        except Exception as exc:
            await _push(ErrorEvent(message=str(exc)))
        finally:
            await queue.put(None)  # sentinel — generator stops here

    # Wrap pipeline in timeout if configured
    timeout = config.timeout_sec if config.timeout_sec > 0 else None
    if timeout:
        task = asyncio.create_task(
            asyncio.wait_for(_pipeline(), timeout=timeout)
        )
    else:
        task = asyncio.create_task(_pipeline())

    while True:
        evt = await queue.get()
        if evt is None:
            break
        yield evt

    # Surface any unhandled exception from the task
    try:
        await task
    except asyncio.TimeoutError:
        yield ErrorEvent(message=f"Pipeline timed out after {timeout:.1f}s")
    except Exception:
        pass  # already surfaced via ErrorEvent in _pipeline


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_nodes(nodes: list) -> int:
    total = 0
    for n in nodes:
        total += 1
        if n.nodes:
            total += _count_nodes(n.nodes)
    return total


def _max_page(nodes: list) -> int:
    best = 0
    for n in nodes:
        if n.end_index > best:
            best = n.end_index
        if n.nodes:
            child_max = _max_page(n.nodes)
            if child_max > best:
                best = child_max
    return best
