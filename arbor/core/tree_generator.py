"""
Main tree generation pipeline — the orchestrator.

generate_tree() ties everything together:
    1. Extract pages from PDF
    2. Detect TOC
    3. Process pages (3-mode fallback: toc_with_pages → toc_no_pages → no_toc)
    4. Verify and fix TOC entries
    5. Build tree (list_to_tree + post_processing)
    6. Subdivide large nodes recursively
    7. Add text, summaries, node IDs
    8. Return DocumentTree
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from typing import Optional, Union

from arbor.core.types import ArborConfig, DocumentTree, ProcessingMode
from arbor.extraction.pdf_extractor import get_page_contents, get_pdf_name
from arbor.extraction.markdown_extractor import extract_from_markdown
from arbor.processing.toc_detector import check_toc
from arbor.processing.toc_processor import meta_processor
from arbor.processing.tree_builder import post_processing, add_preface_if_needed, validate_and_clamp_indices
from arbor.processing.node_subdivision import process_large_nodes
from arbor.processing.verification import check_appear_at_start
from arbor.processing.json_utils import safe_extract_json
from arbor.prompts.tree_generation import generate_summary_prompt, generate_doc_description_prompt
from arbor.providers.base import LLMProvider
from arbor.utils.tree_utils import write_node_ids, add_node_text, remove_node_text, create_node_mapping
from arbor.utils.async_helpers import make_semaphore, semaphore_gather


async def generate_tree(
    document: Union[str, BytesIO],
    provider: LLMProvider,
    config: Optional[ArborConfig] = None,
) -> DocumentTree:
    """
    Generate a hierarchical tree structure from a document.

    Args:
        document: Path to a PDF/Markdown file, or a BytesIO object (PDF).
        provider: LLM provider to use for all AI calls.
        config: Arbor configuration. Uses defaults if None.

    Returns:
        DocumentTree with nested TreeNode structure.

    Raises:
        ValueError: If document type is unsupported.
        ImportError: If required extraction library is missing.
    """
    config = config or ArborConfig()

    # ── Markdown fast path (no LLM for structure extraction) ──────────────────
    if isinstance(document, str) and document.lower().endswith(".md"):
        with open(document, encoding="utf-8") as f:
            content = f.read()
        doc_name = os.path.splitext(os.path.basename(document))[0]
        tree = extract_from_markdown(content, doc_name=doc_name, add_node_ids=config.add_node_ids)
        if config.add_summaries:
            await _generate_summaries(tree.structure, provider, config)
        return tree

    # ── PDF path ──────────────────────────────────────────────────────────────
    doc_name = get_pdf_name(document)
    pages = get_page_contents(document)

    # Step 1: Detect TOC
    toc_result = await check_toc(pages, provider, config)

    has_toc = bool(toc_result.get("toc_content") and toc_result["toc_content"].strip())
    has_page_numbers = toc_result.get("page_index_given_in_toc") == "yes"

    if has_toc and has_page_numbers:
        initial_mode = ProcessingMode.TOC_WITH_PAGES
    elif has_toc:
        initial_mode = ProcessingMode.TOC_NO_PAGES
    else:
        initial_mode = ProcessingMode.NO_TOC

    # Step 2: Process TOC (with fallback cascade)
    flat_items = await meta_processor(
        pages, provider, config,
        mode=initial_mode,
        toc_content=toc_result.get("toc_content"),
        toc_page_list=toc_result.get("toc_page_list", []),
        start_index=1,
    )

    # Step 3: Check which sections start at the top of their page
    flat_items = await check_appear_at_start(flat_items, pages, provider, start_index=1)

    # Step 4: Add preface if content precedes first section
    flat_items = add_preface_if_needed(flat_items, start_index=1)

    # Step 5: Build nested tree
    tree_nodes = post_processing(flat_items, len(pages), start_index=1)

    # Step 6: Subdivide large nodes recursively
    await process_large_nodes(tree_nodes, pages, provider, config)

    # Step 7: Assign node IDs (depth-first pre-order)
    if config.add_node_ids:
        write_node_ids(tree_nodes)

    # Step 8: Add text content to nodes (needed for summaries)
    needs_text = config.add_summaries or config.add_node_text
    if needs_text:
        add_node_text(tree_nodes, pages)

    # Step 9: Generate summaries
    if config.add_summaries:
        await _generate_summaries(tree_nodes, provider, config)

    # Step 10: Remove text if not requested in output
    if not config.add_node_text:
        remove_node_text(tree_nodes)

    # Step 11: Optionally generate document description
    doc_description = None
    if config.add_doc_description:
        clean_structure = [n.to_dict() for n in tree_nodes]
        # Strip text from description input (keep it small)
        for n in clean_structure:
            n.pop("text", None)
        prompt = generate_doc_description_prompt(clean_structure)
        try:
            doc_description = await provider.complete_with_retry(prompt)
        except Exception:
            pass

    return DocumentTree(
        doc_name=doc_name,
        structure=tree_nodes,
        doc_description=doc_description,
    )


async def _generate_summaries(
    nodes: list,
    provider: LLMProvider,
    config: ArborConfig,
) -> None:
    """Generate summaries for all nodes that have text content."""
    semaphore = make_semaphore(config.max_concurrent_llm_calls)

    async def summarize_node(node) -> None:
        if node.text:
            prompt = generate_summary_prompt(node.text)
            try:
                async with semaphore:
                    node.summary = await provider.complete_with_retry(prompt)
            except Exception:
                pass
        if node.nodes:
            await asyncio.gather(*[summarize_node(child) for child in node.nodes])

    await asyncio.gather(*[summarize_node(n) for n in nodes])
