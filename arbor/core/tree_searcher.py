"""
Tree search — find the nodes most likely to answer a question.
"""

from __future__ import annotations

from typing import Optional

from arbor.core.types import DocumentTree, SearchResult, TreeNode
from arbor.processing.json_utils import safe_extract_json
from arbor.prompts.tree_search import tree_search_prompt, tree_search_with_preference_prompt
from arbor.providers.base import LLMProvider
from arbor.utils.tree_utils import create_node_mapping, tree_to_search_dict


async def search_tree(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    preference: Optional[str] = None,
) -> SearchResult:
    """
    Find tree nodes that likely contain the answer to a question.

    Sends the tree (with summaries, without text) to the LLM along with
    the question. The LLM reasons about which node_ids are relevant.

    Args:
        tree: The DocumentTree to search.
        question: The user's question.
        provider: LLM provider for reasoning.
        preference: Optional expert guidance (e.g. "Check Item 7 for financials").

    Returns:
        SearchResult with thinking, node_ids, and resolved TreeNode objects.
    """
    # Build search-friendly tree dict (summaries only, no full text)
    search_dict = tree_to_search_dict(tree.structure)

    if preference:
        prompt = tree_search_with_preference_prompt(question, search_dict, preference)
    else:
        prompt = tree_search_prompt(question, search_dict)

    response = await provider.complete_with_retry(prompt)
    data = safe_extract_json(response, {})

    thinking = str(data.get("thinking", ""))
    node_ids: list[str] = []
    raw_list = data.get("node_list", [])
    if isinstance(raw_list, list):
        node_ids = [str(nid) for nid in raw_list if nid]

    # Resolve node IDs to actual TreeNode objects
    node_map = create_node_mapping(tree.structure)
    nodes = [node_map[nid] for nid in node_ids if nid in node_map]

    return SearchResult(
        thinking=thinking,
        node_ids=node_ids,
        nodes=nodes,
    )
