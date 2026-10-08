"""A PDF file as a document, and pages or chunks of positioned text.

The document-level half needed for bounded page recovery: it opens a file with
the bundled parser and turns a page's content stream into ordered, merged lines.
The stream-level half is `content`; grouping runs into lines is `lines`.

This package reads text only. It decodes no images, renders nothing, and runs no
external process, OCR, or PDF tool.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission.
"""

from __future__ import annotations

from .content import extract_page
from .geometry import page_size
from .lines import Line, group_lines, merge_line_text
from .objects import Document

__all__ = [
    "Line",
    "lines_from_chunks",
    "open_doc",
    "page_chunks",
    "page_lines",
    "page_size",
]


def open_doc(path):
    """Open a PDF and index its object streams.

    The extra pass matters for files whose cross-reference table omits objects
    that live inside an /ObjStm.
    """
    document = Document(str(path))
    document._index_objstms()
    return document


def page_chunks(document, page, number):
    """One page's positioned text runs, tagged with the physical page number."""
    chunks = extract_page(document, page)
    for chunk in chunks:
        chunk.page = number
    return chunks


def lines_from_chunks(chunks):
    """Group positioned runs into lines and merge each line's marked text."""
    lines = group_lines(chunks)
    for line in lines:
        line.text = merge_line_text(line.chunks)
    return [line for line in lines if line.text.strip()]


def page_lines(document, page, number):
    """One page's non-blank lines, in reading order."""
    return lines_from_chunks(page_chunks(document, page, number))
