"""
Prompts for tree search and answer generation.
"""

from __future__ import annotations

import json


def tree_search_prompt(query: str, tree_json: dict | list | str) -> str:
    """
    Find nodes in the document tree that likely contain the answer to a query.

    Returns JSON: {"thinking": str, "node_list": [node_id, ...]}
    """
    tree_str = json.dumps(tree_json, indent=2) if not isinstance(tree_json, str) else tree_json
    return f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{tree_str}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else."""


def tree_search_with_preference_prompt(
    query: str,
    tree_json: dict | list | str,
    preference: str,
) -> str:
    """
    Tree search with expert knowledge / domain preference hints.

    preference: domain-specific guidance (e.g. "For EBITDA questions, check Item 7 MD&A").

    Returns JSON: {"thinking": str, "node_list": [node_id, ...]}
    """
    tree_str = json.dumps(tree_json, indent=2) if not isinstance(tree_json, str) else tree_json
    return f"""You are given a question and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {tree_str}

Expert Knowledge of relevant sections: {preference}

Reply in the following JSON format:
{{
    "thinking": <reasoning about which nodes are relevant>,
    "node_list": [node_id1, node_id2, ...]
}}
Directly return the final JSON structure. Do not output anything else."""


def answer_generation_prompt(query: str, context: str) -> str:
    """
    Generate an answer based on retrieved context.

    Returns: plain text answer.
    """
    return f"""Answer the question based on the context:

Question: {query}
Context: {context}

Provide a clear, concise answer based only on the context provided."""
