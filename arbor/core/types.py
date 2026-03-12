"""
Core data types for Arbor.

All public types users interact with — TreeNode, DocumentTree, SearchResult, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProcessingMode(Enum):
    TOC_WITH_PAGES = "toc_with_pages"
    TOC_NO_PAGES = "toc_no_pages"
    NO_TOC = "no_toc"


@dataclass
class TreeNode:
    """A single node in the document tree."""

    title: str
    start_index: int          # 1-based page number where section starts
    end_index: int            # 1-based page number where section ends (inclusive)
    node_id: Optional[str] = None    # Zero-padded 4-digit string, e.g. "0001"
    summary: Optional[str] = None   # LLM-generated summary of this section
    text: Optional[str] = None      # Raw extracted text (optional, large)
    nodes: list[TreeNode] = field(default_factory=list)  # Child nodes

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (recursive, omits None/empty fields)."""
        d: dict = {
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }
        if self.node_id is not None:
            d["node_id"] = self.node_id
        if self.summary is not None:
            d["summary"] = self.summary
        if self.text is not None:
            d["text"] = self.text
        if self.nodes:
            d["nodes"] = [n.to_dict() for n in self.nodes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        """Reconstruct a TreeNode from a dict (recursive)."""
        nodes = [cls.from_dict(child) for child in d.get("nodes", [])]
        return cls(
            title=d["title"],
            start_index=d["start_index"],
            end_index=d["end_index"],
            node_id=d.get("node_id"),
            summary=d.get("summary"),
            text=d.get("text"),
            nodes=nodes,
        )

    @property
    def page_count(self) -> int:
        return self.end_index - self.start_index + 1

    def is_leaf(self) -> bool:
        return len(self.nodes) == 0


@dataclass
class DocumentTree:
    """The complete tree structure for a document."""

    doc_name: str
    structure: list[TreeNode]
    doc_description: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "doc_name": self.doc_name,
            "structure": [n.to_dict() for n in self.structure],
        }
        if self.doc_description is not None:
            d["doc_description"] = self.doc_description
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DocumentTree:
        return cls(
            doc_name=d["doc_name"],
            structure=[TreeNode.from_dict(n) for n in d.get("structure", [])],
            doc_description=d.get("doc_description"),
        )


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""

    text: str
    token_count: int
    page_number: int   # 1-based


@dataclass
class SearchResult:
    """Result of a tree search operation."""

    thinking: str            # LLM's reasoning about which nodes are relevant
    node_ids: list[str]      # The retrieved node IDs, in order of relevance
    nodes: list[TreeNode] = field(default_factory=list)  # Resolved TreeNode objects


@dataclass
class RAGResponse:
    """Full RAG pipeline response: search + answer + citations."""

    answer: str
    search_result: SearchResult
    context: str                              # The text sent as context to the answer LLM
    citations: list[dict] = field(default_factory=list)  # [{"page": int, "title": str}]


@dataclass
class ArborConfig:
    """Configuration for the Arbor pipeline."""

    model: str = "llama-3.3-70b-versatile"   # Default: Groq free tier
    toc_check_pages: int = 20                 # First N pages to scan for TOC
    max_pages_per_node: int = 10              # Nodes larger than this get subdivided
    max_tokens_per_node: int = 20000          # Nodes with more tokens get subdivided
    add_node_ids: bool = True                 # Add zero-padded node_id to each node
    add_summaries: bool = True                # Generate LLM summary for each node
    add_doc_description: bool = False         # Generate one-sentence doc description
    add_node_text: bool = False               # Include raw page text in output (large)
    max_concurrent_llm_calls: int = 5         # asyncio.Semaphore limit
    overlap_pages: int = 1                    # Pages of overlap between chunks


# Intermediate type used during TOC processing (before tree conversion)
@dataclass
class FlatTOCItem:
    """A single flat TOC item before being converted to a TreeNode."""

    structure: str           # Dot-notation: "1", "1.2", "1.2.3"
    title: str
    physical_index: Optional[int] = None    # 1-based page number
    appear_start: Optional[str] = None      # "yes" | "no" — starts at top of page?
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def to_dict(self) -> dict:
        d: dict = {"structure": self.structure, "title": self.title}
        if self.physical_index is not None:
            d["physical_index"] = self.physical_index
        if self.appear_start is not None:
            d["appear_start"] = self.appear_start
        return d
