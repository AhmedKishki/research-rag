"""OCR is a copy a person asks for, and nothing here edits or writes into sources."""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

from research_rag.corpus import ocr
from research_rag.corpus.ocr import (
    OcrError,
    RecognisedLine,
    check_output_path,
    ocr_pdf,
)
from research_rag.project.config import resolve_config
from research_rag.surfaces import cli
from tests.conftest import write_pdf

SENTENCE = "The commodity is, in the first place, an object outside us."


def _scan(path: Path, *, text_pages: int = 0, pages: int = 2) -> None:
    """A PDF whose pages are pictures of text, like a scan."""

    scan = pymupdf.open()
    for index in range(pages):
        page = pymupdf.open().new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 200), SENTENCE, fontsize=14)
        pixmap = page.get_pixmap(dpi=150)
        target = scan.new_page(width=page.rect.width, height=page.rect.height)
        target.insert_image(target.rect, stream=pixmap.tobytes("png"))
        if index < text_pages:
            target.insert_textbox(
                pymupdf.Rect(72, 300, 540, 400),
                "A page that already carries a text layer of its own.",
                fontsize=11,
            )
    scan.save(path)
    scan.close()


def _recogniser(image: object) -> list[RecognisedLine]:
    return [RecognisedLine(200, 200, 1180, 260, SENTENCE, 0.98)]


def test_a_scanned_page_gains_a_text_layer_and_the_original_is_untouched(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scan.pdf"
    _scan(source, pages=2)
    before = source.read_bytes()
    output = tmp_path / "out" / "scan.ocr.pdf"
    output.parent.mkdir()

    result = ocr_pdf(source, output, recogniser=_recogniser)

    assert source.read_bytes() == before
    assert result["pages"] == 2
    assert result["recognised_pages"] == 2
    assert result["lines_recognised"] == 2
    assert result["mean_confidence"] == 0.98
    assert source.stat().st_size == len(before)
    with pymupdf.open(output) as copy:
        for page in copy:
            assert SENTENCE.split(",")[0] in page.get_text("text")
    with pymupdf.open(source) as original:
        assert all(not page.get_text("text").strip() for page in original)


def test_a_page_that_already_has_text_is_copied_unless_every_page_is_asked_for(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed.pdf"
    _scan(source, text_pages=1, pages=2)

    kept = ocr_pdf(source, tmp_path / "kept.pdf", recogniser=_recogniser)
    every = ocr_pdf(
        source, tmp_path / "every.pdf", recogniser=_recogniser, every_page=True
    )

    assert (kept["recognised_pages"], kept["pages_with_text_already"]) == (1, 1)
    assert (every["recognised_pages"], every["pages_with_text_already"]) == (2, 0)


def test_a_low_confidence_line_is_not_laid_over_the_page(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    _scan(source, pages=1)

    result = ocr_pdf(
        source,
        tmp_path / "out.pdf",
        recogniser=lambda _image: [RecognisedLine(200, 200, 1180, 260, "noise", 0.2)],
    )

    assert result["lines_recognised"] == 0
    assert result["mean_confidence"] is None


def test_the_output_may_not_be_the_original_inside_sources_or_an_existing_file(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "scan.pdf"
    _scan(source, pages=1)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    existing = elsewhere / "scan.ocr.pdf"
    existing.write_bytes(b"x")

    with pytest.raises(OcrError, match="never edited"):
        check_output_path(source, source, sources_root=sources, force=True)
    with pytest.raises(OcrError, match="inside the sources directory"):
        check_output_path(
            source, sources / "scan.ocr.pdf", sources_root=sources, force=True
        )
    with pytest.raises(OcrError, match="--force"):
        check_output_path(source, existing, sources_root=sources, force=False)
    check_output_path(source, existing, sources_root=sources, force=True)
    with pytest.raises(OcrError, match="does not exist"):
        check_output_path(
            source, tmp_path / "missing" / "x.pdf", sources_root=sources, force=False
        )


def test_a_missing_backend_says_how_to_install_it_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scan.pdf"
    _scan(source, pages=1)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    with pytest.raises(OcrError, match="uv sync --extra ocr"):
        ocr_pdf(source, tmp_path / "out.pdf")
    assert not (tmp_path / "out.pdf").exists()


def test_the_command_reads_a_source_and_never_writes_into_the_sources(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scan(project / "sources" / "scan.pdf", pages=1)
    write_pdf(project / "sources" / "notes.pdf", ["Plain text."], title="Notes")
    monkeypatch.setattr(ocr, "load_recogniser", lambda: _recogniser)
    config = resolve_config(project, vanilla_executable=sys.executable)
    monkeypatch.chdir(tmp_path)

    class Args:
        source = "scan.pdf"
        output = None
        force = False
        all_pages = False
        dpi = 200

    written = cli._ocr(Args(), config)  # type: ignore[arg-type]

    assert written["output"] == str(tmp_path / "scan.ocr.pdf")
    assert (tmp_path / "scan.ocr.pdf").is_file()
    assert "The original was not touched" in written["next"]
    assert sorted(path.name for path in (project / "sources").iterdir()) == [
        "notes.pdf",
        "scan.pdf",
    ]

    Args.source = "../escape.pdf"
    with pytest.raises(Exception, match="escapes the sources"):
        cli._ocr(Args(), config)  # type: ignore[arg-type]
    Args.source = "scan.pdf"
    Args.output = str(project / "sources" / "scan.ocr.pdf")
    with pytest.raises(Exception, match="inside the sources directory"):
        cli._ocr(Args(), config)  # type: ignore[arg-type]


def test_the_real_recogniser_reads_a_scanned_page(tmp_path: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    source = tmp_path / "scan.pdf"
    _scan(source, pages=1)

    result = ocr_pdf(source, tmp_path / "out.pdf")

    assert result["lines_recognised"] >= 1
    with pymupdf.open(tmp_path / "out.pdf") as copy:
        assert "commodity" in copy[0].get_text("text")
