"""
Full RAG pipeline: generate tree → search → extract context → answer.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Union

from arbor.core.tree_generator import generate_tree
from arbor.core.tree_searcher import search_tree
from arbor.core.types import ArborConfig, DocumentTree, RAGResponse, SearchResult
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

    Args:
        document: Path to PDF/Markdown, or BytesIO.
        question: The user's question.
        provider: LLM provider for all AI calls.
        config: Arbor config. Defaults used if None.
        tree: Pre-generated DocumentTree. If provided, skips generation.
        preference: Optional expert guidance for tree search.

    Returns:
        RAGResponse with answer, search_result, context, and citations.
    """
    config = config or ArborConfig()

    # Step 1: Generate tree if not provided
    if tree is None:
        tree = await generate_tree(document, provider, config)

    # Step 2: Search tree for relevant nodes
    search_result = await search_tree(tree, question, provider, preference=preference)

    # Step 3: Extract text from retrieved nodes
    # Need pages with text to build context
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
    answer = ""
    if context:
        prompt = answer_generation_prompt(question, context)
        answer = await provider.complete_with_retry(prompt)
    else:
        answer = "I could not find relevant information in this document to answer your question."

    return RAGResponse(
        answer=answer,
        search_result=search_result,
        context=context,
        citations=citations,
    )
