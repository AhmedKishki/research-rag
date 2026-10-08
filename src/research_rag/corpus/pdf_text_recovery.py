"""Bounded native text recovery for unhealthy blocks on one PDF page.

The helper reads one page with the bundled pure-python PDF reader in
`pdf_native` and reassembles only the requested block boxes. It reads the text
layer directly: no external process, OCR, rendering, or image decoding, and it
never writes a file or edits a source. The reader in `pdf_native` is adapted
from the user's own AhmedKishki/pdf-tools project.

Caller boxes use PyMuPDF's unrotated top-left page coordinates. The bundled
reader reports PDF user-space coordinates with the origin at the page box's
bottom left, so each run's centre is converted with the page box before it is
matched. A page whose /Rotate is not zero is refused rather than mapped wrongly.
A parser failure is returned as a bounded reason code; it never raises and never
rejects the source.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .pdf_native.extract import lines_from_chunks, open_doc, page_chunks
from .pdf_native.geometry import chunk_bounds
from .pdf_native.objects import Document

# Why the caller got no text. Both are bounded: they carry no page text and no
# parser detail. The caller decides whether to keep the page's existing text.
REASON_UNAVAILABLE = "pdf_text_recovery_unavailable"
REASON_FAILED = "pdf_text_recovery_failed"
REASON_UNSUPPORTED_ROTATION = "pdf_text_recovery_unsupported_rotation"

_DEFAULT_PAGE_BOX = (0.0, 0.0, 612.0, 792.0)

_Box = tuple[float, float, float, float]
PdfRecoveryCache = dict[Path, tuple[Document, list[dict[str, Any]]] | None]


def _valid_page_number(page_number: object) -> bool:
    return (
        isinstance(page_number, int)
        and not isinstance(page_number, bool)
        and page_number > 0
    )


def _valid_coordinate(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _valid_boxes(boxes: object) -> dict[int, _Box] | None:
    """Return finite boxes keyed by block id, or ``None`` when the caller is wrong."""

    if not isinstance(boxes, dict):
        return None
    checked: dict[int, _Box] = {}
    for key, box in boxes.items():
        if isinstance(key, bool) or not isinstance(key, int):
            return None
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            return None
        if not all(_valid_coordinate(item) for item in box):
            return None
        checked[key] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return checked


def _page_rectangle(document, page) -> _Box:
    """The page's CropBox or MediaBox as a normalised rectangle, or US Letter."""

    for key in ("CropBox", "MediaBox"):
        box = document.resolve(page.get(key))
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x0, y0, x1, y1 = (float(document.resolve(v)) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return _DEFAULT_PAGE_BOX


def _page_rotation(document, page) -> int:
    rotate = document.resolve(page.get("Rotate"))
    if isinstance(rotate, bool) or not isinstance(rotate, (int, float)):
        return 0
    return int(rotate) % 360


def _chunk_top_centre(chunk, rectangle: _Box) -> tuple[float, float]:
    """A run's centre in PyMuPDF top-left page coordinates."""

    x0, y0, x1, y1 = chunk_bounds(chunk)
    centre_x = (x0 + x1) / 2.0
    centre_y = (y0 + y1) / 2.0
    return centre_x - rectangle[0], rectangle[3] - centre_y


def _select_chunks(chunks, boxes: dict[int, _Box], rectangle: _Box):
    """Place each run in the smallest box whose rectangle holds its centre, once."""

    assignments: dict[int, list] = {key: [] for key in boxes}
    for chunk in chunks:
        centre_x, centre_y = _chunk_top_centre(chunk, rectangle)
        winner: int | None = None
        winner_area = 0.0
        for key, (x0, y0, x1, y1) in boxes.items():
            if not (x0 <= centre_x <= x1 and y0 <= centre_y <= y1):
                continue
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if winner is None or (area, key) < (winner_area, winner):
                winner = key
                winner_area = area
        if winner is not None:
            assignments[winner].append(chunk)
    return assignments


def recover_pdf_blocks(
    path: Path,
    page_number: int,
    boxes: dict[int, tuple[float, float, float, float]],
    cache: PdfRecoveryCache | None = None,
) -> tuple[dict[int, str], str | None]:
    """Recover text for the given block boxes on one physical PDF page.

    Returns recovered text per requested block id, and ``None`` or a reason code.
    A block whose recovered text carries no alphanumeric character is omitted.
    The helper never raises for a bad caller argument or an unreadable file.
    The optional cache belongs to one source scan, never a process or generation.
    It avoids reopening and reparsing the PDF for each unhealthy page.
    """

    checked = _valid_boxes(boxes)
    if checked is None or not _valid_page_number(page_number):
        return {}, REASON_FAILED
    if not checked:
        return {}, None

    source = Path(path)
    if not source.is_file():
        return {}, REASON_UNAVAILABLE

    try:
        if cache is not None and source in cache:
            cached = cache[source]
            if cached is None:
                return {}, REASON_FAILED
            document, pages = cached
        else:
            try:
                document = open_doc(source)
                pages = document.pages()
            except Exception:
                if cache is not None:
                    cache[source] = None
                raise
            if cache is not None:
                cache[source] = (document, pages)
        if page_number > len(pages):
            return {}, REASON_FAILED
        page = pages[page_number - 1]
        if _page_rotation(document, page) != 0:
            return {}, REASON_UNSUPPORTED_ROTATION
        rectangle = _page_rectangle(document, page)
        chunks = page_chunks(document, page, page_number)
        recovered: dict[int, str] = {}
        for key, members in _select_chunks(chunks, checked, rectangle).items():
            text = "\n".join(
                line.text.strip()
                for line in lines_from_chunks(members)
                if line.text.strip()
            )
            if text and any(character.isalnum() for character in text):
                recovered[key] = text
        return recovered, None
    except Exception:  # noqa: BLE001 - any parser failure is one bounded outcome.
        return {}, REASON_FAILED
