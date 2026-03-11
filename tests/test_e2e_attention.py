"""
End-to-end integration test using "Attention Is All You Need" (1706.03762).

Uses a scripted mock provider that returns pre-built realistic responses
for each type of prompt. This proves the full pipeline works without needing
a live API key.

Run:
    python -m pytest tests/test_e2e_attention.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Optional

import pytest

PDF_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "attention.pdf")


# ─── Scripted Mock Provider ───────────────────────────────────────────────────

class AttentionPaperMock:
    """
    Deterministic mock provider pre-programmed with realistic responses
    for "Attention Is All You Need".

    Routes each call to the right response by inspecting the prompt content.
    """

    def __init__(self):
        self.call_log: list[str] = []

    @property
    def name(self) -> str:
        return "mock/attention-paper"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        content, _ = await self.complete_with_finish_reason(
            prompt, temperature, max_tokens, chat_history
        )
        return content

    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        response = self._route(prompt)
        self.call_log.append(f"[{self._classify(prompt)}]")
        return response, "stop"

    async def complete_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> str:
        return await self.complete(prompt, temperature, max_tokens, chat_history)

    def _classify(self, prompt: str) -> str:
        p = prompt.lower()
        if "table of content" in p and "detect" in p or "toc_detected" in p:
            return "toc_detect"
        if "extract the full table of contents" in p:
            return "toc_extract"
        if "page numbers/indices" in p:
            return "detect_page_index"
        if "physical_index" in p and "add the physical_index" in p:
            return "toc_index_extractor"
        if "continue the tree structure from the previous part" in p:
            return "toc_continue"
        if "hierarchical tree structure" in p and "generate the tree structure of the document" in p:
            return "toc_init"
        if "appears or starts in the given page_text" in p and "start_begin" not in p:
            return "check_appear"
        if "starts in the beginning" in p:
            return "check_at_start"
        if "find the physical index of the start page" in p:
            return "fix_entry"
        if "description of the partial document" in p:
            return "summarize"
        if "Answer the question based on the context" in prompt:
            return "answer"
        if "node_list" in p and "tree structure" in p:
            return "tree_search"
        return "other"

    def _route(self, prompt: str) -> str:
        kind = self._classify(prompt)

        if kind == "toc_detect":
            # Page 1 of attention paper has no TOC — return no
            return json.dumps({"thinking": "No table of contents visible", "toc_detected": "no"})

        if kind == "toc_extract":
            return "No table of contents found"

        if kind == "detect_page_index":
            return json.dumps({"thinking": "No page numbers", "page_index_given_in_toc": "no"})

        if kind == "toc_init":
            # Generate realistic tree structure from first pass
            return json.dumps([
                {"structure": "1", "title": "Introduction", "physical_index": "<physical_index_2>"},
                {"structure": "2", "title": "Background", "physical_index": "<physical_index_3>"},
                {"structure": "3", "title": "Model Architecture", "physical_index": "<physical_index_4>"},
                {"structure": "3.1", "title": "Encoder and Decoder Stacks", "physical_index": "<physical_index_4>"},
                {"structure": "3.2", "title": "Attention", "physical_index": "<physical_index_5>"},
                {"structure": "3.2.1", "title": "Scaled Dot-Product Attention", "physical_index": "<physical_index_5>"},
                {"structure": "3.2.2", "title": "Multi-Head Attention", "physical_index": "<physical_index_5>"},
                {"structure": "3.2.3", "title": "Applications of Attention in our Model", "physical_index": "<physical_index_6>"},
                {"structure": "3.3", "title": "Position-wise Feed-Forward Networks", "physical_index": "<physical_index_6>"},
                {"structure": "3.4", "title": "Embeddings and Softmax", "physical_index": "<physical_index_7>"},
                {"structure": "3.5", "title": "Positional Encoding", "physical_index": "<physical_index_7>"},
                {"structure": "4", "title": "Why Self-Attention", "physical_index": "<physical_index_7>"},
                {"structure": "5", "title": "Training", "physical_index": "<physical_index_8>"},
                {"structure": "5.1", "title": "Training Data and Batching", "physical_index": "<physical_index_8>"},
                {"structure": "5.2", "title": "Hardware and Schedule", "physical_index": "<physical_index_8>"},
                {"structure": "5.3", "title": "Optimizer", "physical_index": "<physical_index_9>"},
                {"structure": "5.4", "title": "Regularization", "physical_index": "<physical_index_9>"},
                {"structure": "6", "title": "Results", "physical_index": "<physical_index_9>"},
                {"structure": "6.1", "title": "Machine Translation", "physical_index": "<physical_index_9>"},
                {"structure": "6.2", "title": "Model Variations", "physical_index": "<physical_index_10>"},
                {"structure": "6.3", "title": "English Constituency Parsing", "physical_index": "<physical_index_11>"},
                {"structure": "7", "title": "Conclusion", "physical_index": "<physical_index_12>"},
                {"structure": "8", "title": "References", "physical_index": "<physical_index_13>"},
            ])

        if kind == "toc_continue":
            return json.dumps([])  # Nothing new in later chunks

        if kind == "check_appear":
            # All sections appear on their pages
            return json.dumps({"thinking": "Section title found in page", "answer": "yes"})

        if kind == "check_at_start":
            # Most sections don't start at the very top (shared pages)
            title = re.search(r'section title is (.+?)\.', prompt)
            if title:
                t = title.group(1).lower()
                if any(x in t for x in ["introduction", "background", "conclusion", "references"]):
                    return json.dumps({"thinking": "Starts at top", "start_begin": "yes"})
            return json.dumps({"thinking": "Other content precedes it", "start_begin": "no"})

        if kind == "fix_entry":
            return json.dumps({"thinking": "Found on this page", "physical_index": "<physical_index_4>"})

        if kind == "summarize":
            # Generate plausible summaries based on section title hints in prompt
            p_lower = prompt.lower()
            if "introduction" in p_lower:
                return "Introduces the Transformer model, a novel architecture based solely on attention mechanisms, eliminating recurrence and convolutions."
            elif "model architecture" in p_lower or "encoder" in p_lower:
                return "Describes the Transformer architecture with encoder-decoder stacks, each using multi-head self-attention and feed-forward layers."
            elif "attention" in p_lower and "scaled" in p_lower:
                return "Defines Scaled Dot-Product Attention: softmax(QK^T/sqrt(d_k))V, which scales dot products to prevent vanishing gradients."
            elif "multi-head" in p_lower:
                return "Multi-Head Attention projects queries, keys, values h times to run attention in parallel, capturing different representation subspaces."
            elif "self-attention" in p_lower or "why" in p_lower:
                return "Analyzes computational complexity of self-attention vs recurrence, showing O(1) sequential operations and O(n^2 * d) total complexity."
            elif "training" in p_lower:
                return "Trained on WMT 2014 EN-DE (4.5M pairs) and EN-FR (36M pairs) with Adam optimizer and custom learning rate schedule."
            elif "results" in p_lower or "translation" in p_lower:
                return "Achieves 28.4 BLEU on EN-DE and 41.0 on EN-FR, outperforming all prior models at a fraction of the training cost."
            elif "conclusion" in p_lower:
                return "Concludes that Transformers, based entirely on attention, achieve state-of-the-art performance and plan to apply to images, audio, and video."
            else:
                return "This section presents key technical details about the Transformer model architecture and training methodology."

        if kind == "tree_search":
            # For the architecture question, return the Model Architecture nodes
            return json.dumps({
                "thinking": (
                    "The question asks about the Transformer architecture. "
                    "The most relevant nodes are '3. Model Architecture', '3.1 Encoder and Decoder Stacks', "
                    "'3.2 Attention' and its subsections, and '3.3 Position-wise Feed-Forward Networks'. "
                    "The introduction also mentions the architecture at a high level."
                ),
                "node_list": ["0001", "0003", "0004", "0005", "0006", "0007", "0008"]
            })

        if kind == "answer":
            return (
                "The Transformer is a novel neural network architecture introduced in 'Attention Is All You Need' "
                "that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. "
                "It consists of an encoder-decoder structure where both encoder and decoder are composed of "
                "stacked identical layers. The encoder maps input sequences to continuous representations; "
                "the decoder then generates output sequences auto-regressively. Each layer uses Multi-Head "
                "Self-Attention, which computes Scaled Dot-Product Attention (softmax(QK^T/sqrt(d_k))V) in "
                "parallel across h attention heads, and Position-wise Feed-Forward Networks. Positional "
                "encodings are added to input embeddings to inject sequence order information. This design "
                "achieves O(1) sequential operations (vs O(n) for RNNs) and enables significantly more "
                "parallelization during training, reaching 28.4 BLEU on EN-DE translation."
            )

        return json.dumps({"thinking": "ok", "result": "ok"})


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(PDF_PATH),
    reason="attention.pdf not found — run: curl -o tests/fixtures/attention.pdf https://arxiv.org/pdf/1706.03762"
)
class TestAttentionPaperE2E:

    @pytest.mark.asyncio
    async def test_page_extraction(self):
        """Verify we can extract all 15 pages with reasonable token counts."""
        from arbor.extraction.pdf_extractor import get_page_contents
        pages = get_page_contents(PDF_PATH)

        print(f"\n{'='*60}")
        print(f"PDF: Attention Is All You Need (1706.03762)")
        print(f"{'='*60}")
        print(f"Pages extracted: {len(pages)}")
        total_tokens = sum(p.token_count for p in pages)
        print(f"Total tokens: {total_tokens:,}")
        print(f"\nPer-page breakdown:")
        for p in pages:
            bar = "#" * (p.token_count // 100)
            print(f"  Page {p.page_number:2d}: {p.token_count:4d} tokens  {bar}")

        assert len(pages) == 15
        assert total_tokens > 5000
        assert all(p.page_number == i + 1 for i, p in enumerate(pages))
        assert all(p.token_count >= 0 for p in pages)

    @pytest.mark.asyncio
    async def test_generate_tree(self):
        """Generate full tree structure using mock provider."""
        from arbor.core.tree_generator import generate_tree
        from arbor.core.types import ArborConfig

        provider = AttentionPaperMock()
        config = ArborConfig(
            add_node_ids=True,
            add_summaries=True,
            add_doc_description=False,
            add_node_text=False,
            max_concurrent_llm_calls=3,
        )

        tree = await generate_tree(PDF_PATH, provider, config)

        print(f"\n{'='*60}")
        print(f"Generated Tree: {tree.doc_name}")
        print(f"{'='*60}")
        from arbor.utils.tree_utils import print_tree, count_nodes
        print_tree(tree.structure)
        print(f"\nTotal nodes: {count_nodes(tree.structure)}")
        print(f"LLM calls made: {len(provider.call_log)}")
        print(f"Call types: {', '.join(set(provider.call_log))}")

        assert tree.doc_name == "attention"
        assert len(tree.structure) > 0
        total_nodes = count_nodes(tree.structure)
        assert total_nodes >= 5, f"Expected at least 5 nodes, got {total_nodes}"

        # Verify node IDs are assigned
        from arbor.utils.tree_utils import create_node_mapping
        node_map = create_node_mapping(tree.structure)
        assert len(node_map) > 0
        assert "0001" in node_map

        # Verify summaries were generated
        def has_summaries(nodes):
            return any(n.summary is not None or has_summaries(n.nodes) for n in nodes)
        assert has_summaries(tree.structure)

        # Verify no text leaked into output (add_node_text=False)
        def has_text(nodes):
            return any(n.text is not None or has_text(n.nodes) for n in nodes)
        assert not has_text(tree.structure), "text should be stripped from output"

        return tree

    @pytest.mark.asyncio
    async def test_tree_search(self):
        """Search the generated tree for the Transformer architecture."""
        from arbor.core.tree_generator import generate_tree
        from arbor.core.tree_searcher import search_tree
        from arbor.core.types import ArborConfig

        provider = AttentionPaperMock()
        config = ArborConfig(add_node_ids=True, add_summaries=True, add_node_text=False)

        tree = await generate_tree(PDF_PATH, provider, config)

        question = "What is the Transformer architecture?"
        result = await search_tree(tree, question, provider)

        print(f"\n{'='*60}")
        print(f"Tree Search: {question!r}")
        print(f"{'='*60}")
        print(f"\nReasoning:")
        print(f"  {result.thinking[:300]}...")
        print(f"\nRetrieved nodes ({len(result.node_ids)}):")
        for node in result.nodes:
            print(f"  [{node.node_id}] {node.title} (pp. {node.start_index}-{node.end_index})")

        assert len(result.node_ids) > 0
        assert result.thinking != ""
        assert len(result.nodes) > 0
        # The architecture nodes should be retrieved
        titles = [n.title for n in result.nodes]
        arch_found = any("architecture" in t.lower() or "attention" in t.lower()
                         or "encoder" in t.lower() for t in titles)
        assert arch_found, f"Expected architecture nodes in results, got: {titles}"

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self):
        """Full end-to-end: PDF → tree → search → answer."""
        from arbor.core.rag_pipeline import query
        from arbor.core.types import ArborConfig

        provider = AttentionPaperMock()
        config = ArborConfig(add_node_ids=True, add_summaries=True, add_node_text=False)

        question = "What is the Transformer architecture?"
        response = await query(
            document=PDF_PATH,
            question=question,
            provider=provider,
            config=config,
        )

        print(f"\n{'='*60}")
        print(f"Full RAG Pipeline")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}")
        print(f"\nAnswer:")
        print(f"  {response.answer}")
        print(f"\nCitations ({len(response.citations)}):")
        for c in response.citations:
            print(f"  [{c['node_id']}] {c['title']} (pp. {c['start_page']}-{c['end_page']})")
        print(f"\nContext length: {len(response.context):,} chars")
        print(f"\nSearch reasoning (excerpt):")
        print(f"  {response.search_result.thinking[:200]}...")

        assert response.answer != ""
        assert "Transformer" in response.answer or "attention" in response.answer.lower()
        assert len(response.citations) > 0
        assert response.context != ""
        assert response.search_result.thinking != ""


# ─── Standalone runner ────────────────────────────────────────────────────────

async def _run_demo():
    """Standalone demo that prints a full end-to-end run."""
    from arbor.core.tree_generator import generate_tree
    from arbor.core.tree_searcher import search_tree
    from arbor.core.rag_pipeline import query
    from arbor.core.types import ArborConfig
    from arbor.extraction.pdf_extractor import get_page_contents
    from arbor.utils.tree_utils import print_tree, count_nodes

    print("\n" + "="*60)
    print("Arbor End-to-End Demo — Attention Is All You Need")
    print("="*60)

    # Step 1: Extract pages
    pages = get_page_contents(PDF_PATH)
    print(f"\n[1/4] PDF Extraction")
    print(f"  Pages: {len(pages)}")
    print(f"  Total tokens: {sum(p.token_count for p in pages):,}")

    # Step 2: Generate tree
    provider = AttentionPaperMock()
    config = ArborConfig(add_node_ids=True, add_summaries=True, add_node_text=False)
    tree = await generate_tree(PDF_PATH, provider, config)

    print(f"\n[2/4] Tree Generation — {tree.doc_name}")
    print(f"  Total nodes: {count_nodes(tree.structure)}")
    print(f"  Tree structure:")
    print_tree(tree.structure)

    # Step 3: Tree search
    question = "What is the Transformer architecture?"
    result = await search_tree(tree, question, provider)

    print(f"\n[3/4] Tree Search")
    print(f"  Question: {question!r}")
    print(f"  Retrieved {len(result.nodes)} node(s):")
    for node in result.nodes[:5]:
        print(f"    [{node.node_id}] {node.title}")

    # Step 4: Full RAG
    response = await query(PDF_PATH, question, provider, config)
    print(f"\n[4/4] Answer")
    print(f"  {response.answer[:500]}")
    print(f"\n  Citations:")
    for c in response.citations[:4]:
        print(f"    p.{c['start_page']}-{c['end_page']}: {c['title']}")

    print("\n" + "="*60)
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(_run_demo())
