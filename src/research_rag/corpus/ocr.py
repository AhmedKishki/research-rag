"""Give a scanned PDF a text layer, as a new file, on request and never by itself.

`research-rag ocr` is the only caller. Ingestion never runs it, no agent tool and
no workspace action reaches it, and it never edits the file it reads: the result
is a copy with the recognised words laid invisibly over each scanned page, and the
reader decides whether it replaces the original in the sources directory.

The recogniser is an optional dependency, named when it is missing, so a project
that only indexes text layers installs and imports none of it. Pages that already
carry text are copied untouched unless the caller asks for every page, because
recognising a page that has a good text layer can only make it worse.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pymupdf

# Pixels per inch a page is rendered at for recognition. Small print needs more;
# beyond this the cost grows faster than the accuracy.
DEFAULT_DPI = 200
# A page whose text layer holds fewer characters than this counts as having none.
MINIMUM_TEXT_CHARACTERS = 25
# A recognised line below this confidence is dropped rather than laid over the page.
MINIMUM_CONFIDENCE = 0.5

INSTALL_HINT = (
    "OCR needs the optional recognition backend: install it with "
    "`uv sync --extra ocr` in a checkout, or `uv tool install --with "
    "rapidocr-onnxruntime research-rag`. Nothing was written."
)


class OcrError(RuntimeError):
    """OCR could not run, or its output could not be written safely."""


@dataclass(frozen=True, slots=True)
class RecognisedLine:
    """One line of text and the box it was found in, in pixels of the rendered page."""

    left: float
    top: float
    right: float
    bottom: float
    text: str
    confidence: float


class Recogniser(Protocol):
    def __call__(self, image: Any) -> Sequence[RecognisedLine]: ...


def load_recogniser() -> Recogniser:
    """The RapidOCR recogniser, or the sentence that says how to get it."""

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OcrError(INSTALL_HINT) from exc
    engine = RapidOCR()

    def recognise(image: Any) -> list[RecognisedLine]:
        found, _timings = engine(image)
        lines: list[RecognisedLine] = []
        for box, text, confidence in found or []:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(
                RecognisedLine(
                    min(xs), min(ys), max(xs), max(ys), str(text), float(confidence)
                )
            )
        return lines

    return recognise


def _page_image(page: pymupdf.Page, dpi: int) -> tuple[Any, float]:
    """The page as an array, and the factor from its pixels back to points."""

    import numpy as np

    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return array, 72.0 / dpi


def _lay_text(
    page: pymupdf.Page, lines: Sequence[RecognisedLine], scale: float
) -> tuple[int, float]:
    """Write the recognised lines invisibly where they were found.

    Returns how many lines were laid and their mean confidence. Render mode 3 is
    the PDF text mode that draws nothing, so the page looks as it did and its
    words can be selected, searched, and extracted.
    """

    laid = 0
    confidence = 0.0
    for line in lines:
        text = line.text.strip()
        if not text or line.confidence < MINIMUM_CONFIDENCE:
            continue
        rect = pymupdf.Rect(
            line.left * scale, line.top * scale, line.right * scale, line.bottom * scale
        )
        if rect.is_empty or rect.height < 2:
            continue
        # A line's height bounds its font size; the box is widened a little so a
        # font that runs wider than the original does not drop its last word.
        size = max(4.0, min(rect.height * 0.8, 48.0))
        page.insert_textbox(
            pymupdf.Rect(rect.x0, rect.y0, rect.x1 + rect.width * 0.25, rect.y1 + size),
            text,
            fontsize=size,
            fontname="helv",
            render_mode=3,
        )
        laid += 1
        confidence += line.confidence
    return laid, (confidence / laid if laid else 0.0)


def check_output_path(
    source: Path, output: Path, *, sources_root: Path, force: bool
) -> None:
    """Refuse an output that could overwrite or join the originals.

    The sources directory is where the originals are the quotation authority, so
    nothing is written into it, and the file being read is never the file written.
    """

    resolved = output.resolve()
    if resolved == source.resolve():
        raise OcrError(
            f"The output is the file being read, which is never edited: {output}"
        )
    root = sources_root.resolve()
    if resolved == root or root in resolved.parents:
        raise OcrError(
            f"The output {output} is inside the sources directory, where nothing "
            "is written. Write it elsewhere, then move it in yourself if it "
            "should replace the original."
        )
    if output.exists() and not force:
        raise OcrError(f"{output} exists. Pass --force to replace it.")
    if not output.parent.is_dir():
        raise OcrError(f"The directory {output.parent} does not exist.")


def ocr_pdf(
    source: Path,
    output: Path,
    *,
    recogniser: Recogniser | None = None,
    dpi: int = DEFAULT_DPI,
    every_page: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write a copy of `source` whose scanned pages carry a recognised text layer."""

    try:
        document = pymupdf.open(source)
    except Exception as exc:
        raise OcrError(f"Cannot open PDF: {source}") from exc
    try:
        if document.needs_pass:
            raise OcrError(f"The PDF is password protected: {source}")
        recognise = recogniser or load_recogniser()
        started = time.perf_counter()
        recognised_pages = 0
        skipped_pages = 0
        lines_laid = 0
        confidences: list[float] = []
        for index, page in enumerate(document):
            if (
                not every_page
                and len(page.get_text("text").strip()) >= MINIMUM_TEXT_CHARACTERS
            ):
                skipped_pages += 1
            else:
                image, scale = _page_image(page, dpi)
                laid, confidence = _lay_text(page, recognise(image), scale)
                recognised_pages += 1
                lines_laid += laid
                if laid:
                    confidences.append(confidence)
            if progress is not None:
                progress(index + 1, document.page_count)
        temporary = output.with_name(f".{output.name}.partial")
        try:
            document.save(temporary, garbage=3, deflate=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "source": str(source),
            "output": str(output),
            "pages": document.page_count,
            "recognised_pages": recognised_pages,
            "pages_with_text_already": skipped_pages,
            "lines_recognised": lines_laid,
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 3) if confidences else None
            ),
            "seconds": round(time.perf_counter() - started, 1),
        }
    finally:
        document.close()
