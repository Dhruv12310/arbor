"""
Steps 4–9 tests: JSON utils, Tree Building, Tree Utils, Search, RAG.

All LLM calls are mocked — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbor.core.types import (
    ArborConfig, DocumentTree, PageContent, SearchResult, TreeNode,
)
from arbor.processing.json_utils import (
    extract_json, safe_extract_json,
    continue_if_truncated, complete_with_continuation,
)
from arbor.processing.tree_builder import (
    list_to_tree, post_processing, add_preface_if_needed,
    validate_and_clamp_indices,
)
from arbor.utils.tree_utils import (
    write_node_ids, create_node_mapping, add_node_text,
    remove_node_text, tree_to_search_dict, remove_fields, count_nodes,
)
from arbor.utils.async_helpers import semaphore_gather, make_semaphore


# ─── JSON Utils ───────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_dict(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_code_fence_json(self):
        result = extract_json('```json\n{"key": "val"}\n```')
        assert result == {"key": "val"}

    def test_code_fence_no_lang(self):
        result = extract_json('```\n{"x": 1}\n```')
        assert result == {"x": 1}

    def test_none_to_null(self):
        result = extract_json('{"val": None}')
        assert result == {"val": None}

    def test_true_false(self):
        result = extract_json('{"a": True, "b": False}')
        assert result == {"a": True, "b": False}

    def test_trailing_comma_dict(self):
        result = extract_json('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_trailing_comma_list(self):
        result = extract_json('[1, 2, 3,]')
        assert result == [1, 2, 3]

    def test_list(self):
        result = extract_json('[{"title": "A"}, {"title": "B"}]')
        assert len(result) == 2

    def test_nested(self):
        data = '{"nodes": [{"id": "0001", "children": []}]}'
        result = extract_json(data)
        assert result["nodes"][0]["id"] == "0001"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_json("this is not json at all!!!")

    def test_safe_returns_default(self):
        result = safe_extract_json("bad json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_safe_returns_none_on_failure(self):
        result = safe_extract_json("bad json")
        assert result is None

    def test_embedded_json_extraction(self):
        text = 'Here is the result: {"thinking": "yes", "answer": "no"} done.'
        result = extract_json(text)
        assert result["answer"] == "no"


class TestContinuation:
    @pytest.mark.asyncio
    async def test_no_continuation_on_stop(self):
        provider = MagicMock()
        result = await continue_if_truncated(
            provider, "prompt", "response", "stop"
        )
        assert result == "response"
        provider.complete_with_finish_reason.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_on_length(self):
        provider = MagicMock()
        provider.complete_with_finish_reason = AsyncMock(return_value=(" continued", "stop"))
        result = await continue_if_truncated(
            provider, "prompt", "partial", "length"
        )
        assert result == "partial continued"

    @pytest.mark.asyncio
    async def test_multiple_continuations(self):
        provider = MagicMock()
        responses = [(" part2", "length"), (" part3", "stop")]
        provider.complete_with_finish_reason = AsyncMock(side_effect=responses)
        result = await continue_if_truncated(
            provider, "prompt", "part1", "length", max_continuations=5
        )
        assert "part1" in result
        assert "part2" in result
        assert "part3" in result

    @pytest.mark.asyncio
    async def test_complete_with_continuation_stop(self):
        provider = MagicMock()
        provider.complete_with_finish_reason = AsyncMock(return_value=("done", "stop"))
        result = await complete_with_continuation(provider, "prompt")
        assert result == "done"


# ─── Tree Builder ─────────────────────────────────────────────────────────────

class TestListToTree:
    def _make_item(self, structure, title, start=1, end=5):
        return {"structure": structure, "title": title, "start_index": start, "end_index": end}

    def test_flat_roots(self):
        items = [
            self._make_item("1", "Chapter 1", 1, 10),
            self._make_item("2", "Chapter 2", 11, 20),
        ]
        tree = list_to_tree(items)
        assert len(tree) == 2
        assert tree[0].title == "Chapter 1"
        assert tree[1].title == "Chapter 2"

    def test_nested_children(self):
        items = [
            self._make_item("1", "Chapter 1", 1, 10),
            self._make_item("1.1", "Section 1.1", 2, 5),
            self._make_item("1.2", "Section 1.2", 6, 10),
            self._make_item("2", "Chapter 2", 11, 20),
        ]
        tree = list_to_tree(items)
        assert len(tree) == 2
        assert len(tree[0].nodes) == 2
        assert tree[0].nodes[0].title == "Section 1.1"

    def test_deep_nesting(self):
        items = [
            self._make_item("1", "A", 1, 20),
            self._make_item("1.1", "B", 2, 10),
            self._make_item("1.1.1", "C", 3, 5),
        ]
        tree = list_to_tree(items)
        assert tree[0].nodes[0].nodes[0].title == "C"

    def test_orphan_promoted_to_root(self):
        # 1.2 without 1 — should be promoted to root
        items = [
            self._make_item("1.2", "Orphan", 1, 5),
        ]
        tree = list_to_tree(items)
        assert len(tree) == 1
        assert tree[0].title == "Orphan"

    def test_empty_items(self):
        assert list_to_tree([]) == []


class TestPostProcessing:
    def test_basic(self):
        items = [
            {"structure": "1", "title": "Intro", "physical_index": 1, "appear_start": "yes"},
            {"structure": "2", "title": "Body", "physical_index": 5, "appear_start": "yes"},
        ]
        tree = post_processing(items, total_pages=10)
        assert tree[0].start_index == 1
        assert tree[0].end_index == 4   # next.start - 1 (because appear_start=yes)
        assert tree[1].start_index == 5
        assert tree[1].end_index == 10  # last item gets total_pages

    def test_appear_start_no(self):
        items = [
            {"structure": "1", "title": "A", "physical_index": 1, "appear_start": "yes"},
            {"structure": "2", "title": "B", "physical_index": 5, "appear_start": "no"},
        ]
        tree = post_processing(items, total_pages=10)
        # When appear_start="no", end_index = next.start (shared boundary)
        assert tree[0].end_index == 5

    def test_none_physical_index_skipped(self):
        items = [
            {"structure": "1", "title": "A", "physical_index": None},
            {"structure": "2", "title": "B", "physical_index": 3},
        ]
        tree = post_processing(items, total_pages=10)
        assert len(tree) == 1
        assert tree[0].title == "B"

    def test_tag_string_physical_index(self):
        items = [
            {"structure": "1", "title": "A", "physical_index": "<physical_index_2>"},
        ]
        tree = post_processing(items, total_pages=5)
        assert tree[0].start_index == 2


class TestAddPreface:
    def test_no_preface_needed(self):
        items = [{"structure": "1", "title": "Chapter 1", "physical_index": 1}]
        result = add_preface_if_needed(items, start_index=1)
        assert len(result) == 1
        assert result[0]["title"] == "Chapter 1"

    def test_preface_added(self):
        items = [{"structure": "1", "title": "Chapter 1", "physical_index": 5}]
        result = add_preface_if_needed(items, start_index=1)
        assert len(result) == 2
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1

    def test_empty_items(self):
        assert add_preface_if_needed([], start_index=1) == []


class TestValidateClamp:
    def test_removes_none_index(self):
        items = [
            {"structure": "1", "title": "A", "physical_index": None},
            {"structure": "2", "title": "B", "physical_index": 3},
        ]
        result = validate_and_clamp_indices(items, total_pages=10, start_index=1)
        assert len(result) == 1
        assert result[0]["title"] == "B"

    def test_removes_out_of_range(self):
        items = [
            {"structure": "1", "title": "A", "physical_index": 0},   # below start
            {"structure": "2", "title": "B", "physical_index": 5},   # valid
            {"structure": "3", "title": "C", "physical_index": 20},  # above max
        ]
        result = validate_and_clamp_indices(items, total_pages=10, start_index=1)
        assert len(result) == 1
        assert result[0]["title"] == "B"


# ─── Tree Utils ───────────────────────────────────────────────────────────────

class TestWriteNodeIds:
    def test_sequential_ids(self):
        nodes = [
            TreeNode("A", 1, 5, nodes=[
                TreeNode("B", 2, 3),
                TreeNode("C", 4, 5),
            ]),
            TreeNode("D", 6, 10),
        ]
        write_node_ids(nodes)
        assert nodes[0].node_id == "0001"
        assert nodes[0].nodes[0].node_id == "0002"
        assert nodes[0].nodes[1].node_id == "0003"
        assert nodes[1].node_id == "0004"

    def test_zero_padded(self):
        nodes = [TreeNode(f"Node {i}", i, i) for i in range(10)]
        write_node_ids(nodes)
        assert nodes[0].node_id == "0001"
        assert nodes[9].node_id == "0010"

    def test_deep_nesting(self):
        root = TreeNode("Root", 1, 10, nodes=[
            TreeNode("Child", 2, 5, nodes=[
                TreeNode("Grandchild", 3, 4)
            ])
        ])
        write_node_ids([root])
        assert root.node_id == "0001"
        assert root.nodes[0].node_id == "0002"
        assert root.nodes[0].nodes[0].node_id == "0003"


class TestCreateNodeMapping:
    def test_basic_mapping(self):
        nodes = [
            TreeNode("A", 1, 5, node_id="0001"),
            TreeNode("B", 6, 10, node_id="0002"),
        ]
        mapping = create_node_mapping(nodes)
        assert mapping["0001"].title == "A"
        assert mapping["0002"].title == "B"

    def test_recursive_mapping(self):
        nodes = [
            TreeNode("Parent", 1, 10, node_id="0001", nodes=[
                TreeNode("Child", 2, 5, node_id="0002"),
            ])
        ]
        mapping = create_node_mapping(nodes)
        assert "0001" in mapping
        assert "0002" in mapping

    def test_excludes_nodes_without_id(self):
        nodes = [TreeNode("No ID", 1, 5)]
        mapping = create_node_mapping(nodes)
        assert len(mapping) == 0


class TestAddRemoveNodeText:
    def test_add_node_text(self):
        pages = [
            PageContent("Page 1 text", 10, 1),
            PageContent("Page 2 text", 10, 2),
            PageContent("Page 3 text", 10, 3),
        ]
        nodes = [TreeNode("Chapter", 1, 3)]
        add_node_text(nodes, pages)
        assert "Page 1 text" in nodes[0].text
        assert "Page 3 text" in nodes[0].text

    def test_remove_node_text(self):
        nodes = [
            TreeNode("A", 1, 1, text="some text", nodes=[
                TreeNode("B", 1, 1, text="child text")
            ])
        ]
        remove_node_text(nodes)
        assert nodes[0].text is None
        assert nodes[0].nodes[0].text is None


class TestTreeToSearchDict:
    def test_basic(self):
        nodes = [TreeNode("A", 1, 5, node_id="0001", summary="Summary A")]
        result = tree_to_search_dict(nodes)
        assert len(result) == 1
        assert result[0]["node_id"] == "0001"
        assert result[0]["summary"] == "Summary A"
        assert "text" not in result[0]

    def test_text_excluded(self):
        nodes = [TreeNode("A", 1, 5, node_id="0001", text="lots of text")]
        result = tree_to_search_dict(nodes)
        assert "text" not in result[0]

    def test_nested(self):
        nodes = [TreeNode("Parent", 1, 10, node_id="0001", nodes=[
            TreeNode("Child", 2, 5, node_id="0002")
        ])]
        result = tree_to_search_dict(nodes)
        assert "nodes" in result[0]
        assert result[0]["nodes"][0]["node_id"] == "0002"


class TestRemoveFields:
    def test_remove_from_dict(self):
        d = {"a": 1, "b": 2, "text": "long text"}
        result = remove_fields(d, ["text"])
        assert "text" not in result
        assert result["a"] == 1

    def test_remove_from_list(self):
        lst = [{"a": 1, "text": "x"}, {"b": 2, "text": "y"}]
        result = remove_fields(lst, ["text"])
        assert all("text" not in item for item in result)

    def test_recursive(self):
        d = {"nodes": [{"title": "A", "text": "big text"}]}
        result = remove_fields(d, ["text"])
        assert "text" not in result["nodes"][0]
        assert result["nodes"][0]["title"] == "A"


class TestCountNodes:
    def test_flat(self):
        nodes = [TreeNode("A", 1, 1), TreeNode("B", 2, 2)]
        assert count_nodes(nodes) == 2

    def test_nested(self):
        nodes = [TreeNode("Root", 1, 10, nodes=[
            TreeNode("Child1", 2, 5),
            TreeNode("Child2", 6, 10),
        ])]
        assert count_nodes(nodes) == 3


# ─── Async Helpers ────────────────────────────────────────────────────────────

class TestSemaphoreGather:
    @pytest.mark.asyncio
    async def test_no_semaphore(self):
        async def work(x): return x * 2
        results = await semaphore_gather([work(i) for i in range(5)])
        assert sorted(results) == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_with_semaphore(self):
        sem = make_semaphore(2)
        async def work(x): return x + 1
        results = await semaphore_gather([work(i) for i in range(5)], semaphore=sem)
        assert sorted(results) == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_empty(self):
        results = await semaphore_gather([])
        assert results == []


# ─── Tree Searcher (mocked) ───────────────────────────────────────────────────

class TestTreeSearcher:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        from arbor.core.tree_searcher import search_tree

        # Build a simple tree
        tree = DocumentTree(
            doc_name="test",
            structure=[
                TreeNode("Chapter 1", 1, 5, node_id="0001", summary="About intro"),
                TreeNode("Chapter 2", 6, 10, node_id="0002", summary="About methods"),
            ]
        )

        # Mock provider returning valid search result
        provider = MagicMock()
        provider.complete_with_retry = AsyncMock(return_value=json.dumps({
            "thinking": "Chapter 1 is about the intro which is relevant",
            "node_list": ["0001"],
        }))

        result = await search_tree(tree, "What is the introduction about?", provider)
        assert result.thinking != ""
        assert "0001" in result.node_ids
        assert len(result.nodes) == 1
        assert result.nodes[0].title == "Chapter 1"

    @pytest.mark.asyncio
    async def test_unknown_node_id_ignored(self):
        from arbor.core.tree_searcher import search_tree

        tree = DocumentTree(
            doc_name="test",
            structure=[TreeNode("Chapter 1", 1, 5, node_id="0001")]
        )
        provider = MagicMock()
        provider.complete_with_retry = AsyncMock(return_value=json.dumps({
            "thinking": "relevant",
            "node_list": ["0001", "9999"],  # 9999 doesn't exist
        }))

        result = await search_tree(tree, "query", provider)
        assert "0001" in result.node_ids
        assert len(result.nodes) == 1  # 9999 silently dropped

    @pytest.mark.asyncio
    async def test_empty_node_list(self):
        from arbor.core.tree_searcher import search_tree

        tree = DocumentTree(doc_name="t", structure=[TreeNode("A", 1, 1, node_id="0001")])
        provider = MagicMock()
        provider.complete_with_retry = AsyncMock(return_value=json.dumps({
            "thinking": "nothing relevant",
            "node_list": [],
        }))

        result = await search_tree(tree, "unrelated query", provider)
        assert result.node_ids == []
        assert result.nodes == []


# ─── Full Public API Import ───────────────────────────────────────────────────

class TestPublicAPIComplete:
    def test_generate_tree_importable(self):
        import arbor
        assert callable(arbor.generate_tree)

    def test_search_tree_importable(self):
        import arbor
        assert callable(arbor.search_tree)

    def test_query_importable(self):
        import arbor
        assert callable(arbor.query)
