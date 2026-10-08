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
from .lines import (
    Line,
    detect_column_gutter,
    detect_column_x,
    group_lines,
    merge_line_text,
    order_page_lines,
    split_column_chunks,
)
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


def _merged_lines(chunks):
    lines = group_lines(chunks)
    for line in lines:
        line.text = merge_line_text(line.chunks)
    lines = [line for line in lines if line.text.strip()]
    return sorted(lines, key=lambda line: (-round(line.y, 2), line.x))


def lines_from_chunks(chunks, page_w=None):
    """Group positioned runs into lines and merge each line's marked text.

    With ``page_w`` the page's width in points, two-column pages are read
    column-major. The columns are separated before their runs are merged, so two
    columns that share a baseline are never welded into one line: the boundary is
    the line-start cluster `detect_column_x` finds, or, when the columns share
    baselines and that test cannot fire, the vertical gutter
    `detect_column_gutter` finds. Without a width, or on a page the geometry
    cannot split, the lines keep top-down order.
    """
    lines = _merged_lines(chunks)
    if page_w is None or not lines:
        return lines
    boundary = detect_column_x(lines, page_w)
    if boundary is None:
        boundary = detect_column_gutter(chunks, page_w)
    if boundary is None:
        return order_page_lines(lines, page_w)
    left, right = split_column_chunks(chunks, boundary)
    if not left or not right:
        return order_page_lines(lines, page_w)
    return [*_merged_lines(left), *_merged_lines(right)]


def page_lines(document, page, number):
    """One page's non-blank lines, in reading order, columns split."""
    width, _height = page_size(document, page)
    return lines_from_chunks(page_chunks(document, page, number), width)
