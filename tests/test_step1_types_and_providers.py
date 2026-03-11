"""
Step 1 tests: Types + Provider Interface.

Tests that:
1. All dataclasses are correctly defined and serializable
2. TreeNode to_dict / from_dict round-trips work
3. DocumentTree serialization works
4. FlatTOCItem works
5. Provider base class enforces abstract methods
6. OllamaProvider, OpenAIProvider, GroqProvider, AnthropicProvider instantiate correctly
7. Provider.complete_with_retry works (using a mock)

Does NOT make real API calls — all tested via mocks or offline.
"""

import asyncio
import json
import pytest
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import arbor
from arbor.core.types import (
    TreeNode, DocumentTree, PageContent, SearchResult,
    RAGResponse, ArborConfig, ProcessingMode, FlatTOCItem,
)
from arbor.providers.base import LLMProvider
from arbor.providers.openai_provider import OpenAIProvider, GroqProvider, OpenAICompatibleProvider
from arbor.providers.ollama_provider import OllamaProvider
from arbor.providers.anthropic_provider import AnthropicProvider


# ─── TreeNode Tests ───────────────────────────────────────────────────────────

class TestTreeNode:
    def test_basic_creation(self):
        node = TreeNode(title="Introduction", start_index=1, end_index=5)
        assert node.title == "Introduction"
        assert node.start_index == 1
        assert node.end_index == 5
        assert node.node_id is None
        assert node.summary is None
        assert node.text is None
        assert node.nodes == []

    def test_full_creation(self):
        child = TreeNode(title="1.1 Background", start_index=2, end_index=3, node_id="0002")
        node = TreeNode(
            title="1. Introduction",
            start_index=1,
            end_index=5,
            node_id="0001",
            summary="This section introduces the topic.",
            text="Full text here...",
            nodes=[child],
        )
        assert node.node_id == "0001"
        assert len(node.nodes) == 1
        assert node.nodes[0].title == "1.1 Background"

    def test_to_dict_minimal(self):
        node = TreeNode(title="Section 1", start_index=1, end_index=3)
        d = node.to_dict()
        assert d == {"title": "Section 1", "start_index": 1, "end_index": 3}
        # None fields should not be in dict
        assert "node_id" not in d
        assert "summary" not in d
        assert "text" not in d
        assert "nodes" not in d

    def test_to_dict_full(self):
        child = TreeNode(title="Child", start_index=2, end_index=2, node_id="0002")
        node = TreeNode(
            title="Parent",
            start_index=1,
            end_index=5,
            node_id="0001",
            summary="Summary text",
            nodes=[child],
        )
        d = node.to_dict()
        assert d["node_id"] == "0001"
        assert d["summary"] == "Summary text"
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["title"] == "Child"

    def test_from_dict_round_trip(self):
        original = TreeNode(
            title="Section",
            start_index=1,
            end_index=10,
            node_id="0001",
            summary="A summary",
            nodes=[
                TreeNode(title="Subsection", start_index=2, end_index=5, node_id="0002")
            ],
        )
        d = original.to_dict()
        restored = TreeNode.from_dict(d)
        assert restored.title == original.title
        assert restored.start_index == original.start_index
        assert restored.end_index == original.end_index
        assert restored.node_id == original.node_id
        assert restored.summary == original.summary
        assert len(restored.nodes) == 1
        assert restored.nodes[0].title == "Subsection"

    def test_json_serializable(self):
        node = TreeNode(
            title="Test",
            start_index=1,
            end_index=3,
            node_id="0001",
            summary="Test summary",
        )
        # Must be JSON-serializable without errors
        json_str = json.dumps(node.to_dict())
        assert "Test" in json_str

    def test_page_count(self):
        node = TreeNode(title="X", start_index=3, end_index=7)
        assert node.page_count == 5

    def test_is_leaf(self):
        leaf = TreeNode(title="Leaf", start_index=1, end_index=1)
        assert leaf.is_leaf()

        parent = TreeNode(
            title="Parent",
            start_index=1,
            end_index=5,
            nodes=[leaf],
        )
        assert not parent.is_leaf()


# ─── DocumentTree Tests ───────────────────────────────────────────────────────

class TestDocumentTree:
    def test_basic_creation(self):
        tree = DocumentTree(
            doc_name="test_document",
            structure=[TreeNode(title="Chapter 1", start_index=1, end_index=10)],
        )
        assert tree.doc_name == "test_document"
        assert len(tree.structure) == 1
        assert tree.doc_description is None

    def test_to_dict(self):
        tree = DocumentTree(
            doc_name="report",
            structure=[
                TreeNode(title="Intro", start_index=1, end_index=2, node_id="0001"),
                TreeNode(title="Body", start_index=3, end_index=8, node_id="0002"),
            ],
            doc_description="A financial report.",
        )
        d = tree.to_dict()
        assert d["doc_name"] == "report"
        assert d["doc_description"] == "A financial report."
        assert len(d["structure"]) == 2

    def test_to_dict_no_description(self):
        tree = DocumentTree(doc_name="x", structure=[])
        d = tree.to_dict()
        assert "doc_description" not in d

    def test_from_dict_round_trip(self):
        original = DocumentTree(
            doc_name="paper",
            structure=[
                TreeNode(title="Abstract", start_index=1, end_index=1, node_id="0001"),
                TreeNode(title="Introduction", start_index=2, end_index=4, node_id="0002"),
            ],
        )
        restored = DocumentTree.from_dict(original.to_dict())
        assert restored.doc_name == original.doc_name
        assert len(restored.structure) == 2
        assert restored.structure[0].title == "Abstract"


# ─── PageContent Tests ────────────────────────────────────────────────────────

class TestPageContent:
    def test_creation(self):
        page = PageContent(text="Hello world", token_count=2, page_number=1)
        assert page.text == "Hello world"
        assert page.token_count == 2
        assert page.page_number == 1


# ─── ArborConfig Tests ────────────────────────────────────────────────────────

class TestArborConfig:
    def test_defaults(self):
        config = ArborConfig()
        assert config.model == "llama-3.1-70b-versatile"
        assert config.toc_check_pages == 20
        assert config.max_pages_per_node == 10
        assert config.max_tokens_per_node == 20000
        assert config.add_node_ids is True
        assert config.add_summaries is True
        assert config.add_doc_description is False
        assert config.add_node_text is False
        assert config.max_concurrent_llm_calls == 5
        assert config.overlap_pages == 1

    def test_custom(self):
        config = ArborConfig(
            model="gpt-4o-mini",
            toc_check_pages=30,
            add_summaries=False,
        )
        assert config.model == "gpt-4o-mini"
        assert config.toc_check_pages == 30
        assert config.add_summaries is False


# ─── FlatTOCItem Tests ────────────────────────────────────────────────────────

class TestFlatTOCItem:
    def test_creation(self):
        item = FlatTOCItem(structure="1.2", title="Background", physical_index=5)
        assert item.structure == "1.2"
        assert item.title == "Background"
        assert item.physical_index == 5

    def test_to_dict_minimal(self):
        item = FlatTOCItem(structure="1", title="Introduction")
        d = item.to_dict()
        assert d == {"structure": "1", "title": "Introduction"}
        assert "physical_index" not in d

    def test_to_dict_with_index(self):
        item = FlatTOCItem(structure="2.1", title="Methods", physical_index=10)
        d = item.to_dict()
        assert d["physical_index"] == 10


# ─── ProcessingMode Tests ─────────────────────────────────────────────────────

class TestProcessingMode:
    def test_values(self):
        assert ProcessingMode.TOC_WITH_PAGES.value == "toc_with_pages"
        assert ProcessingMode.TOC_NO_PAGES.value == "toc_no_pages"
        assert ProcessingMode.NO_TOC.value == "no_toc"


# ─── Provider Interface Tests ─────────────────────────────────────────────────

class TestLLMProviderInterface:
    def test_cannot_instantiate_abstract(self):
        """LLMProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_must_implement_methods(self):
        """Partial implementation raises TypeError on instantiation."""
        class IncompleteProvider(LLMProvider):
            @property
            def name(self) -> str:
                return "incomplete"
            # Missing complete() and complete_with_finish_reason()

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_default_token_count(self):
        """Default token counting is len(text) // 4."""
        class MinimalProvider(LLMProvider):
            @property
            def name(self): return "minimal"
            async def complete(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
                return "ok"
            async def complete_with_finish_reason(self, prompt, temperature=0.0, max_tokens=None, chat_history=None):
                return "ok", "stop"

        p = MinimalProvider()
        assert p.count_tokens("hello world") == len("hello world") // 4
        assert p.count_tokens("x" * 400) == 100

    @pytest.mark.asyncio
    async def test_complete_with_retry_succeeds_on_first_try(self):
        """retry wrapper passes through when no exception."""
        class SuccessProvider(LLMProvider):
            @property
            def name(self): return "success"
            async def complete(self, prompt, **kwargs): return "answer"
            async def complete_with_finish_reason(self, prompt, **kwargs): return "answer", "stop"

        p = SuccessProvider()
        result = await p.complete_with_retry("hello")
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_complete_with_retry_retries_on_failure(self):
        """retry wrapper retries the specified number of times."""
        call_count = 0

        class FlakyProvider(LLMProvider):
            @property
            def name(self): return "flaky"
            async def complete_with_finish_reason(self, prompt, **kwargs):
                return "x", "stop"
            async def complete(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("transient error")
                return "finally succeeded"

        p = FlakyProvider()
        result = await p.complete_with_retry("hello", max_retries=5, retry_delay=0.0)
        assert result == "finally succeeded"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_complete_with_retry_raises_after_max(self):
        """retry wrapper raises after max_retries exhausted."""
        class AlwaysFailsProvider(LLMProvider):
            @property
            def name(self): return "fails"
            async def complete_with_finish_reason(self, prompt, **kwargs):
                return "x", "stop"
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("always fails")

        p = AlwaysFailsProvider()
        with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
            await p.complete_with_retry("hello", max_retries=3, retry_delay=0.0)


# ─── OpenAI Provider Tests ────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_instantiation(self):
        """OpenAIProvider instantiates with a fake key."""
        p = OpenAIProvider(api_key="sk-fake", model="gpt-4o-mini")
        assert p.name == "openai/gpt-4o-mini"
        assert p.model == "gpt-4o-mini"

    def test_name_property(self):
        p = OpenAIProvider(api_key="sk-fake", model="gpt-4o")
        assert p.name == "openai/gpt-4o"

    def test_token_count_fallback(self):
        """Without tiktoken, falls back to len//4."""
        p = OpenAIProvider(api_key="sk-fake")
        # With or without tiktoken installed, should return a positive int
        count = p.count_tokens("hello world this is a test")
        assert count > 0

    @pytest.mark.asyncio
    async def test_complete_mocked(self):
        """Verify complete() calls the OpenAI client correctly."""
        p = OpenAIProvider(api_key="sk-fake")

        # Mock the underlying client
        mock_choice = MagicMock()
        mock_choice.message.content = "  The answer is 42.  "
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        p._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await p.complete("What is 6 * 7?")
        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_complete_with_finish_reason_length(self):
        """Returns 'length' when finish_reason is not 'stop'."""
        p = OpenAIProvider(api_key="sk-fake")

        mock_choice = MagicMock()
        mock_choice.message.content = "partial output..."
        mock_choice.finish_reason = "length"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        p._client.chat.completions.create = AsyncMock(return_value=mock_response)

        content, finish = await p.complete_with_finish_reason("prompt")
        assert content == "partial output..."
        assert finish == "length"

    @pytest.mark.asyncio
    async def test_chat_history_passed(self):
        """chat_history is prepended to messages."""
        p = OpenAIProvider(api_key="sk-fake")

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        create_mock = AsyncMock(return_value=mock_response)
        p._client.chat.completions.create = create_mock

        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        await p.complete("follow-up question", chat_history=history)

        call_kwargs = create_mock.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "previous question"
        assert messages[2]["content"] == "follow-up question"


# ─── Groq Provider Tests ──────────────────────────────────────────────────────

class TestGroqProvider:
    def test_instantiation(self):
        p = GroqProvider(api_key="gsk_fake")
        assert p.name == "groq/llama-3.1-70b-versatile"

    def test_custom_model(self):
        p = GroqProvider(api_key="gsk_fake", model="mixtral-8x7b-32768")
        assert p.name == "groq/mixtral-8x7b-32768"

    def test_no_api_key_raises(self):
        import os
        original = os.environ.pop("GROQ_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Groq API key required"):
                GroqProvider()
        finally:
            if original:
                os.environ["GROQ_API_KEY"] = original

    def test_env_var_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
        p = GroqProvider()
        assert p.name == "groq/llama-3.1-70b-versatile"

    @pytest.mark.asyncio
    async def test_complete_mocked(self):
        p = GroqProvider(api_key="gsk_fake")

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq response"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        p._client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await p.complete("hello")
        assert result == "Groq response"


# ─── Ollama Provider Tests ────────────────────────────────────────────────────

class TestOllamaProvider:
    def test_instantiation(self):
        p = OllamaProvider(model="qwen2.5:7b")
        assert p.name == "ollama/qwen2.5:7b"
        assert "localhost:11434" in p._base_url

    def test_custom_base_url(self):
        p = OllamaProvider(model="llama3.1:8b", base_url="http://192.168.1.100:11434")
        assert "192.168.1.100" in p._base_url

    def test_token_count(self):
        p = OllamaProvider()
        assert p.count_tokens("hello") > 0

    @pytest.mark.asyncio
    async def test_complete_mocked(self):
        p = OllamaProvider(model="qwen2.5:7b")

        mock_choice = MagicMock()
        mock_choice.message.content = "Local model response"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        p._client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await p.complete("hello")
        assert result == "Local model response"


# ─── Anthropic Provider Tests ─────────────────────────────────────────────────

class TestAnthropicProvider:
    def test_instantiation(self):
        p = AnthropicProvider(api_key="sk-ant-fake")
        assert "anthropic" in p.name
        assert "haiku" in p.name or "claude" in p.name

    def test_model_alias_haiku(self):
        p = AnthropicProvider(api_key="sk-ant-fake", model="haiku")
        assert p.model == "claude-haiku-4-5-20251001"

    def test_model_alias_sonnet(self):
        p = AnthropicProvider(api_key="sk-ant-fake", model="sonnet")
        assert p.model == "claude-sonnet-4-6"

    def test_model_alias_opus(self):
        p = AnthropicProvider(api_key="sk-ant-fake", model="opus")
        assert p.model == "claude-opus-4-6"

    def test_full_model_id_passthrough(self):
        p = AnthropicProvider(api_key="sk-ant-fake", model="claude-haiku-4-5-20251001")
        assert p.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_complete_mocked(self):
        p = AnthropicProvider(api_key="sk-ant-fake")

        mock_block = MagicMock()
        mock_block.text = "Claude response"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"

        p._client.messages.create = AsyncMock(return_value=mock_response)
        result = await p.complete("hello")
        assert result == "Claude response"

    @pytest.mark.asyncio
    async def test_finish_reason_max_tokens(self):
        p = AnthropicProvider(api_key="sk-ant-fake")

        mock_block = MagicMock()
        mock_block.text = "truncated output"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "max_tokens"

        p._client.messages.create = AsyncMock(return_value=mock_response)
        content, finish = await p.complete_with_finish_reason("prompt")
        assert content == "truncated output"
        assert finish == "length"


# ─── Public API / __init__ Tests ──────────────────────────────────────────────

class TestPublicAPI:
    def test_version(self):
        assert arbor.__version__ == "0.1.0"

    def test_all_types_importable(self):
        assert arbor.TreeNode is TreeNode
        assert arbor.DocumentTree is DocumentTree
        assert arbor.ArborConfig is ArborConfig
        assert arbor.SearchResult is SearchResult
        assert arbor.RAGResponse is RAGResponse

    def test_all_providers_importable(self):
        assert arbor.LLMProvider is LLMProvider
        assert arbor.OpenAIProvider is OpenAIProvider
        assert arbor.GroqProvider is GroqProvider
        assert arbor.OllamaProvider is OllamaProvider
        assert arbor.AnthropicProvider is AnthropicProvider


# ─── OpenAICompatibleProvider Tests ──────────────────────────────────────────

class TestOpenAICompatibleProvider:
    def test_instantiation(self):
        p = OpenAICompatibleProvider(
            api_key="fake",
            base_url="http://localhost:8000/v1",
            model="mistral-7b",
        )
        assert p.name == "openai-compat/mistral-7b"

    @pytest.mark.asyncio
    async def test_complete_mocked(self):
        p = OpenAICompatibleProvider(
            api_key="fake",
            base_url="http://localhost:8000/v1",
            model="mistral-7b",
        )
        mock_choice = MagicMock()
        mock_choice.message.content = "compatible response"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        p._client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await p.complete("test")
        assert result == "compatible response"
