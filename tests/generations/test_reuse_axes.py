"""Independent reuse axes, driven through the real ingest state machine.

The gate split lives in ``generations/generation.py`` and is unit-tested in
``test_generation.py``. Here ``_advance_ingestion`` is exercised with the shared
retrieval fakes, so the decisions that only appear end to end are covered:

- a model-only change reuses extraction and chunks while recomputing every vector;
- ``force_recompute`` disables every reuse axis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.embeddings import resolve_embedding_model
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

_OTHER_EMBEDDING = resolve_embedding_model("BAAI/bge-base-en-v1.5")


def _service(config: object) -> ResearchService:
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )


def test_a_model_change_reuses_text_and_recomputes_vectors(project: Path) -> None:
    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Stable cobalt evidence about commodity fetishism."],
            title="Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        first = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"
        assert first["chunk_count"] >= 1

        # Rewrite the selected generation's manifest so its vectors belong to a
        # different embedding model. The sources, extraction, and chunking are
        # untouched, which is exactly a model-only change.
        manifest_path = Path(first["generation_root"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["retrieval"]["dense"] = {
            "embedding_model": _OTHER_EMBEDDING.name,
            "embedding_model_revision": _OTHER_EMBEDDING.revision,
            "embedding_dimension": _OTHER_EMBEDDING.dimension,
            "embedding_model_repository": _OTHER_EMBEDDING.repository,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        second = await _service(config).ingest(chunk_size=50, chunk_overlap=10)

        assert second["status"] == "ready"
        assert second["generation_changed"] is True
        # Extraction and chunk text are reused; every vector is recomputed because
        # the recorded model identity no longer matches.
        assert second["rebuilt_document_count"] == 0
        assert second["reused_document_count"] == second["document_count"]
        assert second["rebuilt_chunk_count"] == 0
        assert second["reused_chunk_count"] == second["chunk_count"]
        assert second["reused_vector_count"] == 0
        assert second["created_vector_count"] == second["chunk_count"]

    asyncio.run(exercise())


def test_force_recompute_disables_every_reuse_axis(project: Path) -> None:
    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Stable cobalt evidence about commodity fetishism."],
            title="Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        first = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"

        forced = await _service(config).ingest(
            chunk_size=50,
            chunk_overlap=10,
            force_recompute=True,
        )

        assert forced["status"] == "ready"
        assert forced["reused_document_count"] == 0
        assert forced["rebuilt_document_count"] == forced["document_count"]
        assert forced["reused_chunk_count"] == 0
        assert forced["reused_vector_count"] == 0
        assert forced["created_vector_count"] == forced["chunk_count"]

    asyncio.run(exercise())


@pytest.mark.parametrize("old_headers", [False, True])
def test_a_header_change_reuses_units_and_rebuilds_chunks(
    project: Path, old_headers: bool
) -> None:
    """A header edit keeps extraction units but re-derives every chunk."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Stable cobalt evidence about commodity fetishism."],
            title="Article",
        )
        base = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=[f"chunking.headers={str(old_headers).lower()}"],
        )
        first = await _service(base).ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"
        unchanged = await _service(base).ingest(chunk_size=50, chunk_overlap=10)
        assert unchanged["status"] == "unchanged"
        assert unchanged["generation_id"] == first["generation_id"]

        headed_config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=[f"chunking.headers={str(not old_headers).lower()}"],
        )
        headed = await _service(headed_config).ingest(chunk_size=50, chunk_overlap=10)

        assert headed["status"] == "ready"
        assert headed["generation_changed"] is True
        # The units come from the same extraction, so they are reused; the chunker
        # changed what each vector covers, so no chunk or vector is reusable.
        assert headed["rebuilt_document_count"] == 0
        assert headed["reused_document_count"] == headed["document_count"]
        assert headed["rebuilt_chunk_count"] == headed["chunk_count"]
        assert headed["reused_chunk_count"] == 0
        assert headed["created_vector_count"] == headed["chunk_count"]

    asyncio.run(exercise())


def test_header_change_resumes_with_units_reused_and_chunks_rebuilt(
    project: Path,
) -> None:
    """The staged ``reuse_chunks`` flag survives a checkpointed resume."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Stable cobalt evidence about commodity fetishism."],
            title="Article",
        )
        base = resolve_config(project, vanilla_executable=sys.executable)
        first = await _service(base).ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"

        headed_config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["chunking.headers=true"],
        )
        service = _service(headed_config)
        result: dict[str, object] | None = None
        for _ in range(80):
            result = await service.ingest(
                chunk_size=50,
                chunk_overlap=10,
                work_budget_seconds=0,
            )
            if result["status"] != "in_progress":
                break
        assert result is not None
        assert result["status"] == "ready"
        assert result["reused_document_count"] == result["document_count"]
        assert result["rebuilt_chunk_count"] == result["chunk_count"]
        assert result["reused_chunk_count"] == 0

    asyncio.run(exercise())


def test_a_model_change_recomputes_the_embedding_token_audit(project: Path) -> None:
    """Reused chunks carry the new model's token audit, never the old model's."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Stable cobalt evidence about commodity fetishism."],
            title="Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        first_backend = FakeDenseBackend()
        first_backend.token_count_override = 7
        first = await ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=first_backend
        ).ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"

        manifest_path = Path(first["generation_root"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["retrieval"]["dense"] = {
            "embedding_model": _OTHER_EMBEDDING.name,
            "embedding_model_revision": _OTHER_EMBEDDING.revision,
            "embedding_dimension": _OTHER_EMBEDDING.dimension,
            "embedding_model_repository": _OTHER_EMBEDDING.repository,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        second_backend = FakeDenseBackend()
        second_backend.token_count_override = 99
        second = await ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=second_backend
        ).ingest(chunk_size=50, chunk_overlap=10)
        assert second["status"] == "ready"
        assert second["reused_chunk_count"] == second["chunk_count"]

        new_manifest = json.loads(
            (Path(second["generation_root"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        chunks_path = Path(second["generation_root"]) / new_manifest["files"]["chunks"]
        counts = {
            json.loads(line)["embedding_token_count"]
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line
        }
        assert counts == {99}

    asyncio.run(exercise())


def test_ingest_logs_bounded_build_progress_without_corpus_text(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sentinel = "cobalt-untraceable-sentinel"

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            [f"Stable evidence {sentinel}."],
            title="Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        with caplog.at_level(
            logging.DEBUG,
            logger="research_rag.generations.ingestion",
        ):
            result = await _service(config).ingest(chunk_size=50, chunk_overlap=10)
        assert result["status"] == "ready"

    asyncio.run(exercise())

    text = caplog.text
    assert "ingest: build" in text
    assert "starting phase" in text
    assert "complete" in text
    # Only identifiers, phases, and counts are logged, never corpus text.
    assert sentinel not in text
