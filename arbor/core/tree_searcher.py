"""
Tree search — find the nodes most likely to answer a question.

Two modes:
  oneshot  (default) — sends the full tree in one LLM call, returns node_list.
                       Works well with GPT-4o, Claude, Gemini on small trees.
  multihop           — navigates the tree level-by-level (10-20 nodes per call).
                       Required for the v8 fine-tuned model and large documents.
                       Use multihop=True with ArborFineTunedProvider("treesearch-v8").
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from arbor.core.types import (
    ArborConfig,
    BudgetExceededError,
    DocumentTree,
    NavigatingEvent,
    NavigationDecisionEvent,
    NodeFoundEvent,
    SearchResult,
    TreeNode,
)
from arbor.processing.json_utils import safe_extract_json
from arbor.prompts.tree_search import tree_search_prompt, tree_search_with_preference_prompt
from arbor.providers.base import LLMProvider
from arbor.utils.tree_utils import create_node_mapping, tree_to_search_dict

# Maximum nodes to show at one level in multihop mode
_MAX_NODES_PER_WINDOW = 20

# System prompt used by the v8 fine-tuned model (must match training exactly)
_MULTIHOP_SYSTEM = (
    "You are a document tree navigator. "
    "Given a question and a list of document sections at the current level, "
    "select which sections to explore next to find the answer.\n\n"
    "Always reply with valid JSON:\n"
    '{"thinking": "brief reasoning", "navigate_to": ["node_id1", "node_id2"]}'
)


@dataclass
class _BudgetTracker:
    """Tracks resource consumption during a multihop search."""
    max_hops: int = 5
    max_nodes: int = 0         # 0 = no limit
    max_cost: float = 0.0      # 0 = no limit
    timeout: float = 0.0       # 0 = no limit
    start_time: float = field(default_factory=time.monotonic)
    nodes_examined: int = 0

    def check(self, partial_ids: list[str]) -> None:
        """Raise BudgetExceededError if any limit is hit."""
        if self.timeout > 0 and (time.monotonic() - self.start_time) >= self.timeout:
            raise BudgetExceededError(
                f"Timeout after {self.timeout:.1f}s", partial_ids
            )
        if self.max_nodes > 0 and self.nodes_examined >= self.max_nodes:
            raise BudgetExceededError(
                f"Exceeded max_nodes_searched={self.max_nodes}", partial_ids
            )

    def check_hops(self, depth: int, partial_ids: list[str]) -> None:
        if depth > self.max_hops:
            raise BudgetExceededError(
                f"Exceeded max_hops={self.max_hops}", partial_ids
            )


async def search_tree(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    preference: Optional[str] = None,
    multihop: bool = False,
    max_hops: int = 5,
    config: Optional[ArborConfig] = None,
    event_cb: Optional[Callable] = None,
    question_hint: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> SearchResult:
    """
    Find tree nodes that likely contain the answer to a question.

    Args:
        tree:           The DocumentTree to search.
        question:       The user's question.
        provider:       LLM provider for reasoning.
        preference:     Optional domain guidance. Only used in oneshot mode.
        multihop:       If True, navigate level-by-level (required for fine-tuned model).
        max_hops:       Maximum depth. Auto-scaled by tree size if not set via config.
        config:         ArborConfig with budget controls and retry settings.
        event_cb:       Optional async or sync callback receiving streaming event objects.
        question_hint:  Optional one-line question-type hint prepended at depth 1.
                        e.g. "METADATA — answer is in the title page or first section."
        doc_type:       Optional document type string prepended to system context.
                        e.g. "10-K annual report" or "academic research paper"

    Returns:
        SearchResult with thinking, node_ids, and resolved TreeNode objects.
    """
    if multihop:
        # Adaptive max_hops: always scale with tree size.
        # config.max_hops acts as a floor, not a ceiling — large docs need more hops
        # regardless of what the caller requested.
        total_nodes_for_hops = _count_tree_nodes(tree.structure)
        if total_nodes_for_hops > 200:
            adaptive = 12
        elif total_nodes_for_hops > 100:
            adaptive = 10
        else:
            adaptive = max_hops
        if config and config.max_hops:
            effective_max_hops = max(config.max_hops, adaptive)
        else:
            effective_max_hops = adaptive
        return await _search_multihop(
            tree, question, provider, effective_max_hops, config, event_cb,
            question_hint=question_hint, doc_type=doc_type,
        )
    return await _search_oneshot(tree, question, provider, preference)


# ── One-shot search (existing behavior) ──────────────────────────────────────

async def _search_oneshot(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    preference: Optional[str],
) -> SearchResult:
    """Send full tree to LLM, parse node_list from response."""
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

    node_map = create_node_mapping(tree.structure)
    nodes = [node_map[nid] for nid in node_ids if nid in node_map]

    return SearchResult(thinking=thinking, node_ids=node_ids, nodes=nodes)


# ── Multi-hop search (v8 fine-tuned model) ───────────────────────────────────

async def _search_multihop(
    tree: DocumentTree,
    question: str,
    provider: LLMProvider,
    max_hops: int,
    config: Optional[ArborConfig],
    event_cb: Optional[Callable],
    question_hint: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> SearchResult:
    """
    Navigate the tree level-by-level.

    At each level:
    1. Show up to _MAX_NODES_PER_WINDOW sections to the model
    2. Model returns navigate_to (which ones to explore)
    3. Recurse into children of selected nodes in parallel (asyncio.gather)
    4. Collect all final selected nodes (leaf or max-depth reached)

    Wide levels (>_MAX_NODES_PER_WINDOW) are processed in windows.
    The model only ever sees ≤20 sections per call → no truncation.
    """
    node_map = create_node_mapping(tree.structure)
    all_thinking: list[str] = []
    final_node_ids: list[str] = []
    retries = config.max_retries_on_bad_json if config else 2
    visited: set[str] = set()  # tracks all node IDs explored across hops

    # Pre-compute doc-level stats once (used in every navigate_level call)
    _total_nodes = _count_tree_nodes(tree.structure)
    _total_pages = max((n.end_index for n in tree.structure), default=0) if tree.structure else 0
    _doc_stats = f"~{_total_pages} pages, {_total_nodes} sections total"

    # Build doc-type aware system prompt
    system_prompt = _MULTIHOP_SYSTEM
    if doc_type:
        system_prompt = f"Document type: {doc_type} ({_doc_stats})\n\n{_MULTIHOP_SYSTEM}"

    budget = _BudgetTracker(
        max_hops=max_hops,
        max_nodes=config.max_nodes_searched if config else 0,
        max_cost=config.max_cost_usd if config else 0.0,
        timeout=config.timeout_sec if config else 0.0,
    )

    async def _emit(evt: object) -> None:
        if event_cb is None:
            return
        if asyncio.iscoroutinefunction(event_cb):
            await event_cb(evt)
        else:
            event_cb(evt)

    async def navigate_level(
        nodes: list[TreeNode],
        depth: int,
        current_path: Optional[list[str]] = None,
    ) -> None:
        if current_path is None:
            current_path = []

        # Budget checks
        budget.check_hops(depth, final_node_ids[:])
        budget.check(final_node_ids[:])

        if not nodes:
            return

        await _emit(NavigatingEvent(
            level=depth,
            exploring_ids=[n.node_id for n in nodes if n.node_id],
            section_titles=[n.title for n in nodes],
        ))

        # Chunk wide levels into windows
        windows = _chunk_nodes(nodes, _MAX_NODES_PER_WINDOW)
        total_at_level = len(nodes)

        for win_idx, window in enumerate(windows):
            budget.nodes_examined += len(window)
            budget.check(final_node_ids[:])

            valid_ids = {n.node_id for n in window if n.node_id}

            # Doc context: shown at every call so model knows document scale
            doc_context_line = f"Document: {_doc_stats}\n"

            # Section count annotation: tells model it may be seeing a fraction of this level
            if len(windows) > 1:
                start_num = win_idx * _MAX_NODES_PER_WINDOW + 1
                end_num   = min((win_idx + 1) * _MAX_NODES_PER_WINDOW, total_at_level)
                count_line = f"(showing sections {start_num}–{end_num} of {total_at_level} at this level)\n"
            else:
                count_line = ""

            # Breadcrumb: tell the model where in the document it currently is
            location_line = ""
            if current_path:
                location_line = f"Current location: {' > '.join(current_path)}\n\n"

            # Visited: show nodes already explored so the model avoids re-picking them
            window_visited = [n for n in window if n.node_id and n.node_id in visited]
            visited_line = ""
            if window_visited:
                visited_line = (
                    "\n\nAlready explored: "
                    + ", ".join(f"[{n.node_id}] {n.title}" for n in window_visited)
                )

            # Question-type hint: only shown at root level (depth=1, no parent path yet)
            hint_line = ""
            if question_hint and not current_path:
                hint_line = f"Question type: {question_hint}\n\n"

            user_content = (
                f"{doc_context_line}"
                f"Question: {question}\n\n"
                f"{hint_line}"
                f"{location_line}"
                f"Sections at this level:\n"
                f"{count_line}"
                f"{_format_sections(window)}"
                f"{visited_line}\n\n"
                f"Which sections should we explore next?"
            )
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Feature 4: schema enforcement with retry loop
            data = await _try_parse_with_retry(provider, prompt, valid_ids, retries)

            thinking = data.get("thinking", "")
            if thinking:
                all_thinking.append(f"[depth {depth}] {thinking}")

            navigate_to: list[str] = data.get("navigate_to", [])
            selected = [n for n in window if n.node_id in set(navigate_to)]

            await _emit(NavigationDecisionEvent(
                level=depth,
                shown_ids=[n.node_id for n in window if n.node_id],
                selected_ids=[n.node_id for n in selected if n.node_id],
                current_path=list(current_path),
            ))

            # Mark selected nodes as visited, then recurse or collect
            recurse_tasks = []
            for node in selected:
                if node.node_id:
                    visited.add(node.node_id)
                if node.nodes:
                    recurse_tasks.append(
                        navigate_level(node.nodes, depth + 1, current_path + [node.title])
                    )
                else:
                    if node.node_id:
                        final_node_ids.append(node.node_id)
                        await _emit(NodeFoundEvent(
                            node_id=node.node_id,
                            title=node.title,
                            page_range=f"{node.start_index}-{node.end_index}",
                        ))

            if recurse_tasks:
                await asyncio.gather(*recurse_tasks)

    await navigate_level(tree.structure, depth=1, current_path=[])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_ids = []
    for nid in final_node_ids:
        if nid not in seen:
            seen.add(nid)
            unique_ids.append(nid)

    nodes_out = [node_map[nid] for nid in unique_ids if nid in node_map]
    combined_thinking = " | ".join(all_thinking) if all_thinking else ""

    return SearchResult(
        thinking=combined_thinking,
        node_ids=unique_ids,
        nodes=nodes_out,
    )


# ── Schema enforcement (Feature 4) ───────────────────────────────────────────

async def _try_parse_with_retry(
    provider: LLMProvider,
    prompt: str,
    valid_ids: set[str],
    max_retries: int,
) -> dict:
    """
    Call provider and parse navigate response.
    Retries up to max_retries times on bad JSON or IDs that don't exist in the window.
    Returns best-effort result on exhaustion.
    """
    last_response = ""
    data: dict = {}

    for attempt in range(max_retries + 1):
        if attempt == 0:
            response = await provider.complete_with_retry(prompt)
        else:
            correction = (
                f"{prompt}{last_response}<|im_end|>\n"
                f"<|im_start|>user\nYour previous response was not valid JSON or contained "
                f"unknown node IDs. Valid IDs: {sorted(valid_ids)}. "
                f'Respond ONLY with: {{"thinking": "...", "navigate_to": ["id1"]}}'
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            response = await provider.complete_with_retry(correction)

        last_response = response
        data = _parse_navigate_response(response)
        if _enforce_navigate_schema(data, valid_ids):
            return data

    return data


def _enforce_navigate_schema(data: dict, valid_ids: set[str]) -> bool:
    """
    Validates and fixes a navigate response in-place:
    - Filters navigate_to to only IDs that exist in valid_ids (hallucinated IDs are removed)
    - Adds placeholder thinking if missing
    Returns True if the response is structurally valid (has navigate_to list).
    Returns False only if the response is unparseable (not a dict, no navigate_to key, wrong type).
    """
    if not isinstance(data, dict):
        return False
    nav = data.get("navigate_to")
    if not isinstance(nav, list):
        return False
    # Filter out hallucinated IDs rather than rejecting the whole response
    data["navigate_to"] = [str(n) for n in nav if str(n) in valid_ids]
    if not data.get("thinking"):
        data["thinking"] = "(no reasoning provided)"
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_tree_nodes(nodes: list[TreeNode]) -> int:
    return sum(1 + _count_tree_nodes(n.nodes) for n in nodes)


def _chunk_nodes(nodes: list[TreeNode], size: int) -> list[list[TreeNode]]:
    return [nodes[i : i + size] for i in range(0, len(nodes), size)]


def _format_sections(nodes: list[TreeNode]) -> str:
    lines = []
    for n in nodes:
        sub = " [has sub-sections]" if n.nodes else ""
        lines.append(
            f"[{n.node_id}] {n.title} "
            f"(pages {n.start_index}-{n.end_index}){sub}"
        )
    return "\n".join(lines)


def _parse_navigate_response(text: str) -> dict:
    """Parse navigate_to from model response. Robust to markdown wrappers."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        # Last resort: extract node IDs directly
        node_ids = re.findall(r"\b(\d{4})\b", text)
        return {"navigate_to": node_ids, "thinking": ""}
