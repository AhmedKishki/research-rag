"""The production Docling staging path, driven end to end with a fake worker.

No Docling import, model load, or network: ``run_docling_batch`` is replaced by a
recorded mapping, and the retrieval stack by the shared fakes. This exercises the
real ``ResearchService.ingest`` -> ``_advance_ingestion`` state machine, not the
direct extraction helper.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import research_rag.generations.ingestion as ingestion_module
from research_rag.core.service import ResearchService
from research_rag.corpus import extraction
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.pdf_backend import (
    DoclingConversionError,
    docling_backend_record_from_env,
)
from research_rag.project.config import resolve_config
from research_rag.project.policy import (
    REFLOWABLE_EXTRACTION_POLICY_VERSION,
    ResearchError,
)
from research_rag.storage.durable_io import read_json, read_jsonl
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


def test_backend_or_adapter_drift_alone_does_not_rebuild(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded backend is provenance, not a reason to rebuild an unchanged corpus."""

    write_pdf(project / "sources" / "doc.pdf", ["Amber PDF evidence."], title="P")
    config = _config(project)
    _install_fake_worker(monkeypatch)
    first = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"

    # Simulate an adapter-policy move: the recorded backend fingerprint changes,
    # but every source byte, the extraction/cleaning policy, and the embedding
    # identity stay the same.
    manifest_path = Path(first["generation_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["extraction_backend"] = {
        **manifest["extraction_backend"],
        "adapter_policy_version": 999,
        "fingerprint": "pdf-backend:docling:drift",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_worker(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an unchanged PDF must not be converted again")

    monkeypatch.setattr(extraction, "run_docling_batch", unexpected_worker)
    second = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )

    assert second["status"] == "unchanged"
    assert second["generation_id"] == first["generation_id"]


def test_incremental_build_passes_unchanged_legacy_pdf_through(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new source rebuilds the corpus without re-reading an unchanged PDF.

    The selected generation is made to look legacy: its document carries no
    per-document backend. The new generation must pass the unchanged PDF through
    whole, stamp it as the retired bundled reader, never provision Docling for it,
    and extract only the newly added source.
    """

    write_pdf(project / "sources" / "doc.pdf", ["Amber PDF evidence."], title="P")
    config = _config(project)
    _install_fake_worker(monkeypatch)
    first = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"

    manifest_path = Path(first["generation_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    for document in manifest["documents"]:
        document.pop("extraction_backend", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    write_epub(project / "sources" / "book.epub", "Cobalt EPUB evidence.", title="B")

    def unexpected_worker(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the unchanged PDF must not be converted")

    monkeypatch.setattr(extraction, "run_docling_batch", unexpected_worker)
    second = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )

    assert second["status"] == "ready"
    assert second["reused_document_count"] == 1
    assert second["rebuilt_document_count"] == 1
    new_manifest = read_json(Path(second["generation_root"]) / "manifest.json")
    by_path = {
        document["source_relative_path"]: document
        for document in new_manifest["documents"]
    }
    # The unchanged PDF keeps its provenance, inferred as the retired reader
    # rather than claimed as Docling output.
    assert by_path["doc.pdf"]["extraction_backend"]["backend"] == "custom"
    assert "extraction_backend" not in by_path["book.epub"]


def test_older_epub_rules_reread_the_epub_and_pass_the_pdf_through(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EPUB read under earlier HTML rules is read again; an unchanged PDF is not.

    No source changes, so without the per-document version the ingest would be a
    no-op and the EPUB would keep the text the earlier rules produced.
    """

    write_pdf(project / "sources" / "doc.pdf", ["Amber PDF evidence."], title="P")
    write_epub(project / "sources" / "book.epub", "Cobalt EPUB evidence.", title="B")
    config = _config(project)
    _install_fake_worker(monkeypatch)

    def ingest() -> dict[str, Any]:
        return asyncio.run(
            ResearchService(
                config,
                FakeUltraRAG(),
                dense=FakeDenseBackend(),  # type: ignore[arg-type]
            ).ingest(chunk_size=50, chunk_overlap=10)
        )

    first = ingest()
    assert first["status"] == "ready"
    manifest_path = Path(first["generation_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    for document in manifest["documents"]:
        assert ("reflowable_extraction_policy_version" in document) is (
            document["format"] == "epub"
        )
        document.pop("reflowable_extraction_policy_version", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_worker(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the unchanged PDF must not be converted")

    monkeypatch.setattr(extraction, "run_docling_batch", unexpected_worker)
    second = ingest()

    assert second["status"] == "ready"
    assert second["reused_document_count"] == 1
    assert second["rebuilt_document_count"] == 1
    by_path = {
        document["source_relative_path"]: document
        for document in read_json(Path(second["generation_root"]) / "manifest.json")[
            "documents"
        ]
    }
    assert (
        by_path["book.epub"]["reflowable_extraction_policy_version"]
        == REFLOWABLE_EXTRACTION_POLICY_VERSION
    )
    assert "reflowable_extraction_policy_version" not in by_path["doc.pdf"]
    assert ingest()["status"] == "unchanged"


def test_a_changed_pdf_is_reextracted_with_docling(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed bytes are re-read by the operational reader, not passed through."""

    source = project / "sources" / "doc.pdf"
    write_pdf(source, ["Amber PDF evidence."], title="P")
    config = _config(project)
    _install_fake_worker(monkeypatch)
    first = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"

    write_pdf(source, ["Amber PDF evidence, revised."], title="P")
    calls: list[int] = []
    original_run = extraction.run_docling_batch

    def counting_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return original_run(**kwargs)

    monkeypatch.setattr(extraction, "run_docling_batch", counting_run)
    second = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )

    assert second["status"] == "ready"
    assert calls
    assert second["rebuilt_document_count"] == 1
    assert second["reused_document_count"] == 0


def test_a_no_op_does_not_resolve_the_docling_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged corpus returns before Docling is ever provisioned."""

    write_pdf(project / "sources" / "doc.pdf", ["Amber PDF evidence."], title="P")
    config = _config(project)
    _install_fake_worker(monkeypatch)
    provisions: list[int] = []

    def ensure(*_args: Any, **_kwargs: Any) -> Any:
        provisions.append(1)
        return _FAKE_ENV

    monkeypatch.setattr(ingestion_module, "ensure_docling_env", ensure)
    first = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"
    # The first build did extract a PDF, so it resolved the environment once.
    assert provisions == [1]

    # A fresh service has no cached environment; a no-op must not resolve one.
    second = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert second["status"] == "unchanged"
    assert provisions == [1]


def test_a_mixed_generation_records_each_documents_real_backend(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused legacy PDF and a freshly converted PDF carry their own provenance."""

    write_pdf(project / "sources" / "a.pdf", ["Amber A evidence."], title="A")
    write_pdf(project / "sources" / "b.pdf", ["Cobalt B evidence."], title="B")
    config = _config(project)
    _install_fake_worker(monkeypatch)
    first = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )
    assert first["status"] == "ready"

    manifest_path = Path(first["generation_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    for document in manifest["documents"]:
        if document["source_relative_path"] == "a.pdf":
            document.pop("extraction_backend", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    write_pdf(project / "sources" / "b.pdf", ["Cobalt B evidence, revised."], title="B")
    second = asyncio.run(
        ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        ).ingest(chunk_size=50, chunk_overlap=10)
    )

    assert second["status"] == "ready"
    assert second["reused_document_count"] == 1
    assert second["rebuilt_document_count"] == 1
    new_manifest = read_json(Path(second["generation_root"]) / "manifest.json")
    by_path = {
        document["source_relative_path"]: document
        for document in new_manifest["documents"]
    }
    assert by_path["a.pdf"]["extraction_backend"]["backend"] == "custom"
    assert by_path["b.pdf"]["extraction_backend"]["backend"] == "docling"
    # The manifest's own backend is only the target for new work.
    assert new_manifest["extraction_backend"]["backend"] == "docling"


def test_the_backend_identity_is_stable_across_first_provision(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint identity ignores the installed environment's components."""

    import research_rag.corpus.pdf_backend as pdf_backend_module

    config = _config(project)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )
    absent = service._pdf_backend_fingerprint
    ready = DoclingEnv(
        root=Path("/nonexistent/ready"),
        spec_fingerprint="test-spec",
        versions={"docling": "9.9.9"},
    )
    # The two environments really do have different actual fingerprints...
    assert (
        docling_backend_record_from_env(ready)["fingerprint"]
        != docling_backend_record_from_env(_FAKE_ENV)["fingerprint"]
    )
    # ...but the checkpoint identity, which uses the intended spec, does not move.
    monkeypatch.setattr(
        pdf_backend_module,
        "resolve_docling_env",
        lambda *args, **kwargs: ready,
    )
    assert service._pdf_backend_fingerprint == absent


def test_resumed_pdf_build_rejects_pages_from_a_changed_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume under a changed actual environment never mixes provenance.

    An interrupted two-page build leaves one page staged by environment A. Resuming
    with environment B (same intended spec, different installed component version)
    must discard A's unfinished page and convert the source again from the first
    page, so the document carries only B's fingerprint. The previously selected
    generation is untouched.
    """

    write_pdf(
        project / "sources" / "doc.pdf",
        ["Page one amber evidence.", "Page two cobalt evidence."],
        title="P",
    )
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["ingestion.pdf_page_batch_size=1"],
    )
    _install_fake_worker(monkeypatch)
    env_a = DoclingEnv(
        root=Path("/nonexistent/a"),
        spec_fingerprint="test-spec",
        versions={"docling": "2.135.0"},
    )
    env_b = DoclingEnv(
        root=Path("/nonexistent/b"),
        spec_fingerprint="test-spec",
        versions={"docling": "2.136.0"},
    )
    assert (
        docling_backend_record_from_env(env_a)["fingerprint"]
        != docling_backend_record_from_env(env_b)["fingerprint"]
    )

    async def exercise() -> None:
        monkeypatch.setattr(
            ingestion_module, "ensure_docling_env", lambda *a, **k: env_a
        )
        baseline = await ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        ).ingest(chunk_size=50, chunk_overlap=10)
        assert baseline["status"] == "ready"
        baseline_manifest = Path(baseline["generation_root"]) / "manifest.json"
        baseline_bytes = baseline_manifest.read_bytes()

        # Force a rebuild under A and stop once one of two page batches is staged.
        partial = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        state_path: Path | None = None
        for _ in range(40):
            result = await partial.ingest(
                chunk_size=50,
                chunk_overlap=10,
                force_recompute=True,
                work_budget_seconds=0,
            )
            assert result["status"] == "in_progress"
            candidates = list(config.staging_root.glob("*/work/sources/*/state.json"))
            if candidates:
                state = read_json(candidates[0])
                if (
                    state.get("extraction_stage") == "docling_pages"
                    and state.get("next_index") == 1
                ):
                    state_path = candidates[0]
                    break
        assert state_path is not None
        staged = read_json(state_path.parent / "document.json")
        assert (
            staged["extraction_backend"]["fingerprint"]
            == docling_backend_record_from_env(env_a)["fingerprint"]
        )

        # Resume under B: the staged A page must not be carried into the document.
        monkeypatch.setattr(
            ingestion_module, "ensure_docling_env", lambda *a, **k: env_b
        )
        resumed = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        final = await resumed.ingest(
            chunk_size=50,
            chunk_overlap=10,
            force_recompute=True,
            work_budget_seconds=300,
        )

        assert final["status"] == "ready"
        assert final["resumed"] is True
        assert final["generation_id"] != baseline["generation_id"]
        manifest = read_json(Path(final["generation_root"]) / "manifest.json")
        assert (
            manifest["documents"][0]["extraction_backend"]["fingerprint"]
            == docling_backend_record_from_env(env_b)["fingerprint"]
        )
        units = read_jsonl(
            Path(final["generation_root"]) / "corpus/extracted-units.jsonl"
        )
        assert [unit["locator"]["page"] for unit in units] == [1, 2]

        # The environment change never mutated the previously selected generation.
        assert baseline_manifest.read_bytes() == baseline_bytes

    asyncio.run(exercise())
