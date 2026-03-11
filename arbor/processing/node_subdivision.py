"""
Recursive large-node subdivision.

Nodes that exceed both max_pages_per_node AND max_tokens_per_node get
re-processed with process_no_toc() to generate sub-structure as children.
"""

from __future__ import annotations

import asyncio

from arbor.core.types import ArborConfig, PageContent, TreeNode
from arbor.processing.tree_builder import post_processing
from arbor.processing.toc_processor import process_no_toc
from arbor.processing.verification import check_appear_at_start
from arbor.providers.base import LLMProvider
from arbor.utils.async_helpers import make_semaphore


async def process_large_nodes(
    nodes: list[TreeNode],
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
) -> None:
    """
    Recursively subdivide any node that is too large.

    Threshold: node spans > max_pages_per_node pages AND > max_tokens_per_node tokens.
    Both conditions must be true (short but dense nodes are OK; long but sparse nodes too).

    Modifies nodes in-place by populating node.nodes with children.
    """
    semaphore = make_semaphore(config.max_concurrent_llm_calls)
    tasks = [
        _process_node_recursive(node, pages, provider, config, semaphore)
        for node in nodes
    ]
    await asyncio.gather(*tasks)


async def _process_node_recursive(
    node: TreeNode,
    pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    """Subdivide a single node if it's too large, then recurse into children."""
    page_count = node.end_index - node.start_index + 1
    node_pages = pages[node.start_index - 1 : node.end_index]
    token_count = sum(p.token_count for p in node_pages)

    if page_count > config.max_pages_per_node and token_count >= config.max_tokens_per_node:
        await _subdivide_node(node, node_pages, pages, provider, config, semaphore)

    # Recurse into children (whether just created or pre-existing)
    if node.nodes:
        child_tasks = [
            _process_node_recursive(child, pages, provider, config, semaphore)
            for child in node.nodes
        ]
        await asyncio.gather(*child_tasks)


async def _subdivide_node(
    node: TreeNode,
    node_pages: list[PageContent],
    all_pages: list[PageContent],
    provider: LLMProvider,
    config: ArborConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Run process_no_toc on a large node's pages to generate sub-structure.

    If the first generated item matches the node's own title, skip it
    (it's just re-detecting the parent) and use the rest as children.
    """
    async with semaphore:
        sub_items = await process_no_toc(
            node_pages, provider, config, start_index=node.start_index
        )

    if not sub_items:
        return

    # Check whether each sub-item starts at the top of its page
    sub_items = await check_appear_at_start(
        sub_items, all_pages, provider, start_index=node.start_index
    )

    # If first item's title matches the node itself, skip it
    if sub_items and sub_items[0].get("title", "").strip() == node.title.strip():
        sub_items = sub_items[1:]
        # Adjust node's own end_index to just before the second sub-item
        if sub_items:
            first_child_start = sub_items[0].get("physical_index", node.start_index)
            node.end_index = first_child_start - 1 if sub_items[0].get("appear_start") == "yes" else first_child_start

    if sub_items:
        node.nodes = post_processing(sub_items, node.end_index, start_index=node.start_index)
