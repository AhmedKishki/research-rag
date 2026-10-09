"""The production Docling staging path, driven end to end with a fake worker.

No Docling import, model load, or network: ``run_docling_batch`` is replaced by a
recorded mapping, and the retrieval stack by the shared fakes. This exercises the
real ``ResearchService.ingest`` -> ``_advance_ingestion`` state machine, not the
direct extraction helper.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import research_rag.generations.ingestion as ingestion_module
from research_rag.core.service import ResearchService
from research_rag.corpus import extraction
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.pdf_backend import DoclingConversionError
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from research_rag.storage.records import read_json
from tests.conftest import write_epub, write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

_PAGES = {
    1: "Page one amber evidence argues the case.",
    2: "Page two cobalt evidence continues.",
}

_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={"docling": "2.135.0", "rapidocr": "3.9.1", "pymupdf": "1.26.0"},
)


def _install_fake_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_module, "ensure_docling_env", lambda *a, **k: _FAKE_ENV
    )
    monkeypatch.setattr(
        ingestion_module, "resolve_docling_env", lambda *a, **k: _FAKE_ENV
    )


def _install_fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[int, str] | None = None,
    *,
    fail: bool = False,
) -> None:
    full = dict(pages or _PAGES)

    def fake_run(
        *,
        source_path: Path,
        start: int,
        end: int,
        offline: bool,
        model_cache_root: Path | None,
        docling_env: DoclingEnv,
        timeout: float = 180,
    ) -> dict[str, Any]:
        if fail:
            raise DoclingConversionError("worker could not read the page")
        emitted: dict[str, list[dict[str, Any]]] = {}
        for page_no, text in full.items():
            if start <= page_no <= end:
                emitted[str(page_no)] = [
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
        return {
            "version": "2.135.0",
            "options": {"ocr": False, "vlm": False, "table_structure": False},
            "boundary": {"memory.max": str(2 * 1024**3)},
            "pages": emitted,
            "diagnostics": {
                "retained_item_count": len(emitted),
                "excluded_furniture_item_count": 1,
                "excluded_picture_child_item_count": 0,
            },
            "pdf_metadata": {"title": "Docling Doc", "author": "Ada Example"},
            "page_labels": {str(page): str(page) for page in full},
        }

    monkeypatch.setattr(extraction, "run_docling_batch", fake_run)


def _config(project: Path):
    return resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=[
            "ingestion.pdf_backend=docling",
            "ingestion.maximum_unclean_percent=2.0",
        ],
    )


def _drive(
    service: ResearchService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real ingest loop one checkpointed slice at a time, capturing staging."""

    captured: dict[str, Any] = {"limits": [], "document": None, "batch_files": []}
    original_advance = service._advance_ingestion
    original_screen = ingestion_module.screen_source_units

    async def wrapped(**kwargs: Any) -> dict[str, Any]:
        result = await original_advance(**kwargs)
        sources_dir = kwargs["staging_root"] / "work" / "sources"
        if sources_dir.exists():
            for document_path in sources_dir.glob("*/document.json"):
                captured["document"] = read_json(document_path)
                captured["batch_files"] = sorted(
                    path.name
                    for path in document_path.parent.glob("unit-batches/*.jsonl")
                )
        return result

    def screen_spy(*args: Any, **kwargs: Any):
        captured["limits"].append(kwargs.get("maximum_unclean_percent"))
        return original_screen(*args, **kwargs)

    monkeypatch.setattr(service, "_advance_ingestion", wrapped)
    monkeypatch.setattr(ingestion_module, "screen_source_units", screen_spy)

    async def run() -> dict[str, Any] | ResearchError:
        for _ in range(80):
            try:
                result = await service.ingest(
                    chunk_size=50,
                    chunk_overlap=10,
                    force_recompute=force,
                    work_budget_seconds=0,
                )
            except ResearchError as exc:
                return exc
            if result["status"] != "in_progress":
                return result
        raise AssertionError("ingest did not finish within the step budget")

    return asyncio.run(run()), captured


def test_staged_docling_build_records_artifacts_and_uses_the_project_budget(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(
        project / "sources" / "doc.pdf",
        ["first", "second"],
        title="Docling Doc",
    )
    config = _config(project)
    _install_fake_environment(monkeypatch)
    _install_fake_worker(monkeypatch)
    caller_thread = threading.get_ident()
    install_threads = []
    original_ensure = ingestion_module.ensure_docling_env

    def ensure(config):
        install_threads.append(threading.get_ident())
        return original_ensure(config)

    monkeypatch.setattr(ingestion_module, "ensure_docling_env", ensure)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )

    result, captured = _drive(service, monkeypatch)

    assert result["status"] == "ready"
    assert install_threads
    assert all(thread != caller_thread for thread in install_threads)
    # The shared gate ran once for the PDF with the project's configured 2%.
    assert captured["limits"] == [2.0]
    # document.json was written before finalize, and every requested page has a
    # checkpoint file, including a blank one.
    assert captured["document"]["extraction_backend"]["backend"] == "docling"
    assert (
        captured["document"]["docling_diagnostics"]["excluded_furniture_item_count"]
        == 1
    )
    assert captured["batch_files"] == ["00000000.jsonl", "00000001.jsonl"]

    manifest = read_json(Path(result["generation_root"]) / "manifest.json")
    assert manifest["extraction_backend"]["backend"] == "docling"
    assert manifest["passage_cleaning"]["maximum_unclean_percent"] == 2.0
    assert manifest["passage_cleaning"]["effective_maximum_unclean_percent"] == 2.0
    assert manifest["passage_cleaning"]["pdf_backend"] == "docling"


def test_docling_single_source_failure_preserves_the_selected_generation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "doc.pdf", ["first", "second"], title="Docling Doc")
    config = _config(project)
    _install_fake_environment(monkeypatch)
    _install_fake_worker(monkeypatch)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )

    first, _captured = _drive(service, monkeypatch)
    assert first["status"] == "ready"
    selected_before = read_json(config.current_path)["generation_id"]

    def failing_run(**_kwargs: Any) -> dict[str, Any]:
        raise DoclingConversionError("worker could not read the page")

    monkeypatch.setattr(extraction, "run_docling_batch", failing_run)
    outcome, _captured = _drive(service, monkeypatch, force=True)

    # Whether the build is refused or reported partial, the previously selected
    # generation stays selected: a source-local failure never swaps it.
    if isinstance(outcome, ResearchError):
        pass
    else:
        assert outcome["status"] in {"ready", "in_progress"}
    assert read_json(config.current_path)["generation_id"] == selected_before


def test_backend_switch_reuses_epub_and_reextracts_pdf(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "doc.pdf", ["Amber PDF evidence."], title="P")
    write_epub(project / "sources" / "book.epub", "Cobalt EPUB evidence.", title="B")

    custom = resolve_config(project, vanilla_executable=sys.executable)
    first = asyncio.run(
        ResearchService(
            custom,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"

    _install_fake_environment(monkeypatch)
    _install_fake_worker(monkeypatch)
    docling_batches: list[int] = []
    original_run = extraction.run_docling_batch

    def counting_run(**kwargs: Any) -> dict[str, Any]:
        docling_batches.append(1)
        return original_run(**kwargs)

    monkeypatch.setattr(extraction, "run_docling_batch", counting_run)
    epub_calls: list[int] = []
    original_epub = extraction.extract_epub_spine_item

    def counting_epub(*args: Any, **kwargs: Any):
        epub_calls.append(1)
        return original_epub(*args, **kwargs)

    monkeypatch.setattr(extraction, "extract_epub_spine_item", counting_epub)

    docling = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["ingestion.pdf_backend=docling"],
    )
    second = asyncio.run(
        ResearchService(
            docling,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )

    assert second["status"] == "ready"
    assert second["generation_id"] != first["generation_id"]
    # The PDF is re-extracted under the new backend; the EPUB is reused as-is.
    assert docling_batches == [1]
    assert epub_calls == []
