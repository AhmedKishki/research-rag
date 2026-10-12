"""PDF cleaning reaches ingestion, diagnostics, search, and reuse.

The bundled reader is retired: every PDF is read by the Docling worker. These
tests replace the worker with a recorded page mapping, so they drive the real
staged build, the shared cleaning gate, and the loss budget without importing
Docling, loading a model, or touching the network.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from research_rag.core.service import ResearchService
from research_rag.corpus import extraction
from research_rag.corpus.pdf_backend import DoclingConversionError
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from research_rag.storage.records import read_json, read_jsonl
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

_HEALTHY = "Readable cobalt evidence about the council and the quay. " * 20


def _page_payload(text: str, page_no: int) -> list[dict[str, Any]]:
    return [
        {
            "contents": text,
            "content_kind": "prose",
            "provenance": [
                {
                    "page_no": page_no,
                    "charspan": [0, len(text)],
                    "bbox": {"l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
                }
            ],
        }
    ]


def _install_pages(
    monkeypatch: pytest.MonkeyPatch,
    by_source: dict[str, dict[int, str]],
    *,
    fail: tuple[str, ...] = (),
) -> list[str]:
    """Record the worker result per source name; optionally fail named sources."""

    calls: list[str] = []

    def fake_run(
        *,
        source_path: Path,
        start: int,
        end: int,
        offline: bool,
        model_cache_root: Path | None,
        docling_env: Any,
        timeout: float = 180,
    ) -> dict[str, Any]:
        del offline, model_cache_root, docling_env, timeout
        name = Path(source_path).name
        calls.append(name)
        if name in fail:
            raise DoclingConversionError(f"worker could not read {name}")
        pages = by_source.get(name, {})
        emitted = {
            str(page_no): _page_payload(text, page_no)
            for page_no, text in pages.items()
            if start <= page_no <= end and text
        }
        return {
            "version": "2.135.0",
            "options": {"ocr": False, "vlm": False, "table_structure": False},
            "boundary": {"memory.max": str(2 * 1024**3)},
            "pages": emitted,
            "diagnostics": {
                "retained_item_count": len(emitted),
                "excluded_furniture_item_count": 0,
                "excluded_picture_child_item_count": 0,
            },
            "pdf_metadata": {"title": "Docling Doc", "author": "Ada Example"},
            "page_labels": {str(page_no): str(page_no) for page_no in pages},
        }

    monkeypatch.setattr(extraction, "run_docling_batch", fake_run)
    return calls


def _service(config: Any) -> ResearchService:
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )


def test_docling_pdf_is_read_with_a_physical_page_and_never_edited(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Readable opening page.", "Native amber evidence survives."])
    original = source.read_bytes()
    _install_pages(
        monkeypatch,
        {
            "article.pdf": {
                1: "Readable opening page.",
                2: "Native amber evidence survives.",
            }
        },
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    async def verify() -> None:
        result = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert result["status"] == "ready"
        assert result["recovered_pdf_blocks"] == 0
        assert result["pdf_text_recovery_pages"] == 0
        assert result["excluded_corrupt_passage_count"] == 0
        manifest = read_json(Path(result["generation_root"]) / "manifest.json")
        assert manifest["extraction_backend"]["backend"] == "docling"
        assert manifest["documents"][0]["extraction_backend"]["backend"] == "docling"
        hits = await _service(config).search("amber", retrieval_method="bm25")
        assert hits["hits"][0]["text"] == "Native amber evidence survives."
        assert hits["hits"][0]["locator"]["page"] == 2
        assert source.read_bytes() == original

    asyncio.run(verify())


def test_clean_docling_pdf_has_no_cleaning(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "clean.pdf", ["Clean cobalt evidence."])

    def unexpected_repair(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a clean passage must not be repaired")

    monkeypatch.setattr(extraction, "clean_unclean_passage", unexpected_repair)
    _install_pages(monkeypatch, {"clean.pdf": {1: "Clean cobalt evidence."}})
    config = resolve_config(project, vanilla_executable=sys.executable)
    result = asyncio.run(_service(config).ingest(chunk_size=50, chunk_overlap=10))
    assert result["pdf_text_recovery_pages"] == 0
    assert result["cleaned_passage_count"] == 0
    assert result["discarded_corrupt_character_count"] == 0


def test_docling_corruption_within_budget_is_cleaned_not_left(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small damaged span is dropped to a gap inside the project's budget."""

    write_pdf(project / "sources" / "article.pdf", ["page one", "page two"])
    _install_pages(
        monkeypatch,
        {"article.pdf": {1: "Small \ufffd passage.", 2: _HEALTHY}},
    )
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["ingestion.maximum_unclean_percent=2.0"],
    )

    async def verify() -> None:
        built = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert built["status"] == "ready"
        assert built["excluded_corrupt_passage_count"] == 0
        assert built["partially_cleaned_passage_count"] == 1
        assert built["cleaned_corrupt_span_count"] == 1
        assert built["discarded_corrupt_character_count"] == 1
        assert 0 < built["unclean_character_rate"] < 0.02
        assert all(
            "\ufffd" not in item["contents"]
            for item in read_jsonl(
                Path(built["generation_root"]) / "chunks/chunks.jsonl"
            )
        )
        hits = await _service(config).search("cobalt", retrieval_method="bm25")
        assert hits["hits"]

    asyncio.run(verify())


def test_docling_source_over_budget_is_omitted_without_activation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source over its loss budget is omitted, not a reason to abort the build.

    The damaged source loses almost all of its substantive text, and a readable
    neighbour keeps the build worth validating. The bad source is skipped, the
    partial generation is retained, and nothing is selected.
    """

    source = project / "sources" / "bad.pdf"
    write_pdf(source, ["page one"])
    original = source.read_bytes()
    write_pdf(
        project / "sources" / "neighbour.pdf",
        ["Readable cobalt evidence about the council and the quay."],
    )
    _install_pages(
        monkeypatch,
        {
            "bad.pdf": {1: "\ufffd" * 5000},
            "neighbour.pdf": {
                1: "Readable cobalt evidence about the council and the quay."
            },
        },
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    async def verify() -> None:
        result = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert result["status"] == "partial"
        assert result["document_count"] == 1
        skipped = result["skipped_sources"]
        assert [item["source_relative_path"] for item in skipped] == ["bad.pdf"]
        assert "bad.pdf" in skipped[0]["reason"]
        assert "ingest" in skipped[0]["remedy"]
        assert not config.current_path.exists()
        manifest = read_json(Path(result["generation_root"]) / "manifest.json")
        assert [
            document["source_relative_path"] for document in manifest["documents"]
        ] == ["neighbour.pdf"]
        assert source.read_bytes() == original

    asyncio.run(verify())


def test_every_docling_source_over_budget_refuses_the_generation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build with no source under its budget has nothing to validate."""

    write_pdf(project / "sources" / "bad.pdf", ["page one"])
    _install_pages(monkeypatch, {"bad.pdf": {1: "Small \ufffd passage."}})
    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(ResearchError, match="No clean sources remain"):
        asyncio.run(_service(config).ingest(chunk_size=50, chunk_overlap=10))

    assert not config.current_path.exists()


def test_docling_conversion_failure_omits_the_source(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source-local worker failure is an omission, never a fatal build error."""

    write_pdf(project / "sources" / "bad.pdf", ["page one"])
    write_pdf(project / "sources" / "good.pdf", ["page one"])
    _install_pages(
        monkeypatch,
        {"good.pdf": {1: "Readable cobalt evidence about the council."}},
        fail=("bad.pdf",),
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    result = asyncio.run(_service(config).ingest(chunk_size=50, chunk_overlap=10))

    assert result["status"] == "partial"
    assert [item["source_relative_path"] for item in result["skipped_sources"]] == [
        "bad.pdf"
    ]
    assert not config.current_path.exists()


def test_docling_quality_threshold_changes_invalidate_reuse(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget change rebuilds the document without leaving the damaged glyph."""

    write_pdf(project / "sources" / "article.pdf", ["page one", "page two"])
    _install_pages(
        monkeypatch,
        {"article.pdf": {1: "Small \ufffd passage.", 2: _HEALTHY}},
    )
    base = resolve_config(project, vanilla_executable=sys.executable)
    accepted = replace(
        base,
        settings=replace(base.settings, maximum_unclean_percent=2.0),
    )

    async def verify() -> None:
        built = await _service(accepted).ingest(chunk_size=50, chunk_overlap=10)
        assert built["status"] == "ready"
        assert built["partially_cleaned_passage_count"] == 1
        changed = replace(
            accepted,
            settings=replace(accepted.settings, maximum_unclean_percent=100.0),
        )
        status = await _service(changed).status()
        assert status["generation_upgrade_required"] is False
        assert (
            "passage_cleaning" in status["generation_diagnostics"]["policy_differences"]
        )
        rebuilt = await _service(changed).ingest(chunk_size=50, chunk_overlap=10)
        assert rebuilt["generation_id"] != built["generation_id"]
        assert rebuilt["reused_document_count"] == 0
        assert all(
            "\ufffd" not in item["contents"]
            for item in read_jsonl(
                Path(rebuilt["generation_root"]) / "chunks/chunks.jsonl"
            )
        )
        manifest = read_json(Path(rebuilt["generation_root"]) / "manifest.json")
        assert manifest["passage_cleaning"]["maximum_unclean_percent"] == 100.0
        unchanged = await _service(changed).ingest(chunk_size=50, chunk_overlap=10)
        assert unchanged["status"] == "unchanged"

    asyncio.run(verify())
