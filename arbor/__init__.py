"""
Arbor — Open-source vectorless RAG engine.

Tree-structured document indexing with LLM reasoning.
Works with any LLM: Claude, OpenAI, Groq (free), Ollama (local), and more.

Quickstart:
    import arbor

    # With Groq (free tier)
    provider = arbor.GroqProvider()
    tree = await arbor.generate_tree("document.pdf", provider=provider)

    # With Ollama (100% local, zero cost)
    provider = arbor.OllamaProvider(model="qwen2.5:7b")
    tree = await arbor.generate_tree("document.pdf", provider=provider)

    # Full RAG pipeline
    response = await arbor.query(
        document="document.pdf",
        question="What are the conclusions?",
        provider=provider,
    )
    print(response.answer)
"""

__version__ = "0.1.0"

# Core types
from arbor.core.types import (
    TreeNode,
    DocumentTree,
    PageContent,
    SearchResult,
    RAGResponse,
    ArborConfig,
    ProcessingMode,
    FlatTOCItem,
)

# Providers
from arbor.providers.base import LLMProvider
from arbor.providers.openai_provider import OpenAIProvider, GroqProvider, OpenAICompatibleProvider
from arbor.providers.ollama_provider import OllamaProvider
from arbor.providers.anthropic_provider import AnthropicProvider
try:
    from arbor.providers.finetuned_provider import ArborFineTunedProvider
except Exception:  # bitsandbytes/transformers not installed or broken env
    ArborFineTunedProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "__version__",
    # Types
    "TreeNode",
    "DocumentTree",
    "PageContent",
    "SearchResult",
    "RAGResponse",
    "ArborConfig",
    "ProcessingMode",
    "FlatTOCItem",
    # Pipeline
    "generate_tree",
    "search_tree",
    "query",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "GroqProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "ArborFineTunedProvider",
]

# Core pipeline functions
from arbor.core.tree_generator import generate_tree
from arbor.core.tree_searcher import search_tree
from arbor.core.rag_pipeline import query
