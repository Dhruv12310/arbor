"""
PDF text extraction.

Tries PyPDF2 first (lighter dependency). Falls back to PyMuPDF (fitz)
if PyPDF2 fails or returns empty text — PyMuPDF handles more PDF variants.

Returns a list of PageContent objects, one per page, 1-based.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Union

from arbor.core.types import PageContent
from arbor.utils.token_counter import count_tokens

# Both are optional — we try in order
try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False

try:
    import pymupdf  # PyMuPDF exposes itself as "pymupdf" (>=1.24) or "fitz" (legacy)
    _PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf  # legacy import name
        _PYMUPDF_AVAILABLE = True
    except ImportError:
        _PYMUPDF_AVAILABLE = False


def get_page_contents(
    source: Union[str, BytesIO],
    prefer_pymupdf: bool = False,
) -> list[PageContent]:
    """
    Extract text from every page of a PDF.

    Args:
        source: Path to a PDF file, or a BytesIO object.
        prefer_pymupdf: If True, try PyMuPDF first. Default: PyPDF2 first.

    Returns:
        List of PageContent, one per page, with 1-based page_number.

    Raises:
        ValueError: If source is not a valid PDF path or BytesIO.
        ImportError: If no PDF library is installed.
    """
    if not _PYPDF2_AVAILABLE and not _PYMUPDF_AVAILABLE:
        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install PyPDF2           (lightweight)\n"
            "  pip install pymupdf          (better extraction)\n"
            "  pip install arbor-rag[pymupdf]"
        )

    _validate_source(source)

    if prefer_pymupdf and _PYMUPDF_AVAILABLE:
        pages = _extract_with_pymupdf(source)
    elif _PYPDF2_AVAILABLE:
        pages = _extract_with_pypdf2(source)
        # If PyPDF2 returned mostly empty pages, try PyMuPDF as fallback
        if _PYMUPDF_AVAILABLE and _is_mostly_empty(pages):
            pages = _extract_with_pymupdf(source)
    else:
        pages = _extract_with_pymupdf(source)

    return pages


def _validate_source(source: Union[str, BytesIO]) -> None:
    if isinstance(source, str):
        if not source.lower().endswith(".pdf"):
            raise ValueError(f"Expected a .pdf file, got: {source!r}")
        if not os.path.isfile(source):
            raise ValueError(f"PDF file not found: {source!r}")
    elif not isinstance(source, BytesIO):
        raise ValueError(
            f"source must be a PDF file path (str) or BytesIO, got {type(source).__name__}"
        )


def _extract_with_pypdf2(source: Union[str, BytesIO]) -> list[PageContent]:
    """Extract text using PyPDF2."""
    if isinstance(source, BytesIO):
        source.seek(0)
        reader = PyPDF2.PdfReader(source)
    else:
        reader = PyPDF2.PdfReader(source)

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageContent(
            text=text,
            token_count=count_tokens(text),
            page_number=i + 1,  # 1-based
        ))
    return pages


def _extract_with_pymupdf(source: Union[str, BytesIO]) -> list[PageContent]:
    """Extract text using PyMuPDF (better for complex layouts, tables, columns)."""
    if isinstance(source, BytesIO):
        source.seek(0)
        doc = pymupdf.open(stream=source.read(), filetype="pdf")
    else:
        doc = pymupdf.open(source)

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        pages.append(PageContent(
            text=text,
            token_count=count_tokens(text),
            page_number=i + 1,  # 1-based
        ))
    doc.close()
    return pages


def _is_mostly_empty(pages: list[PageContent], threshold: float = 0.8) -> bool:
    """Return True if more than `threshold` fraction of pages are empty/near-empty."""
    if not pages:
        return True
    empty = sum(1 for p in pages if len(p.text.strip()) < 10)
    return (empty / len(pages)) >= threshold


def get_pdf_name(source: Union[str, BytesIO]) -> str:
    """Return the document name (filename without extension) for a PDF source."""
    if isinstance(source, str):
        return os.path.splitext(os.path.basename(source))[0]
    return "document"


def render_page_images(
    source: Union[str, BytesIO],
    output_dir: str = "pdf_images",
    zoom: float = 2.0,
    fmt: str = "jpeg",
) -> dict[int, str]:
    """
    Render PDF pages as images (for vision-RAG pipeline).

    Requires PyMuPDF. Uses 2x zoom for better quality (same as PageIndex cookbook).

    Returns:
        Dict mapping 1-based page_number → image file path.
    """
    if not _PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF required for image rendering. "
            "Install with: pip install pymupdf  or  pip install arbor-rag[pymupdf]"
        )

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(source, BytesIO):
        source.seek(0)
        doc = pymupdf.open(stream=source.read(), filetype="pdf")
    else:
        doc = pymupdf.open(source)

    page_images: dict[int, str] = {}
    matrix = pymupdf.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes(fmt)
        ext = "jpg" if fmt == "jpeg" else fmt
        img_path = os.path.join(output_dir, f"page_{i + 1}.{ext}")
        with open(img_path, "wb") as f:
            f.write(img_data)
        page_images[i + 1] = img_path

    doc.close()
    return page_images
