"""MOBI uses the shared inventory, staged ingestion, review and search contract."""

import asyncio
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.storage.records import read_jsonl
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG
from tests.mobi_fixtures import install_small_pipe_worker, write_mobi


def test_mobi_ingestion_search_noop_reuse_and_partial_failure(project):
    async def exercise():
        path = project / "sources" / "evidence.mobi"
        write_mobi(path)
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(config, FakeUltraRAG(), dense=FakeDenseBackend())
        first = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"
        assert first["document_count"] == 1
        root = Path(first["generation_root"])
        documents = (await service.list_sources())["sources"]
        assert documents[0]["format"] == "mobi"
        chunks = read_jsonl(root / "chunks" / "chunks.jsonl")
        assert all(chunk["locator"]["type"] == "mobi_section" for chunk in chunks)
        found = await service.find_source("Synthetic Evidence")
        assert found["matches"][0]["source_relative_path"] == "evidence.mobi"
        searched = await service.search(
            "cobalt institutions", retrieval_method="hybrid", rerank=False
        )
        assert searched["hits"]
        assert searched["hits"][0]["source_path"] == "sources/evidence.mobi"
        chunk_id = searched["hits"][0]["chunk_id"]
        artifact_before = (root / "chunks" / "chunks.jsonl").read_bytes()
        await service.set_source_metadata(
            {
                "title": "Reviewed MOBI",
                "authors": ["Reviewed Author"],
                "language": ["de"],
            },
            source_path="evidence.mobi",
        )
        reviewed = await service.search("cobalt institutions", languages_any=["de"])
        assert reviewed["hits"][0]["title"] == "Reviewed MOBI"
        assert (await service.find_source("Reviewed Author"))["match_count"] == 1
        await service.set_chunk_inclusion(
            chunk_id, included=False, reason="Reviewed omission"
        )
        hidden = await service.search("cobalt institutions")
        assert chunk_id not in {hit["chunk_id"] for hit in hidden["hits"]}
        await service.set_chunk_inclusion(chunk_id, included=True)
        await service.set_source_inclusion(
            source_path="evidence.mobi", included=False, reason="Reviewed source"
        )
        assert not (await service.search("cobalt institutions"))["hits"]
        await service.set_source_inclusion(source_path="evidence.mobi", included=True)
        assert (root / "chunks" / "chunks.jsonl").read_bytes() == artifact_before
        unchanged = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert unchanged["generation_changed"] is False
        selected = config.current_path.read_bytes()
        write_mobi(project / "sources" / "protected.mobi", drm=2)
        partial = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert partial["status"] == "partial"
        assert partial["generation_changed"] is False
        assert partial["reused_document_count"] == 1
        assert "DRM/encrypted" in partial["skipped_sources"][0]["reason"]
        assert config.current_path.read_bytes() == selected

    asyncio.run(exercise())


@pytest.mark.parametrize("budget", ["tags", "ipc"])
def test_mobi_budget_failure_omits_only_failed_source(project, monkeypatch, budget):
    from research_rag.corpus import extraction, mobi_reader

    async def exercise():
        write_mobi(project / "sources" / "clean.mobi")
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(config, FakeUltraRAG(), dense=FakeDenseBackend())
        first = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert first["status"] == "ready"
        selected = config.current_path.read_bytes()
        if budget == "tags":
            write_mobi(project / "sources" / "dense.mobi", "<p>x</p>" * 100_000)
        else:
            write_mobi(project / "sources" / "large-output.mobi")
            monkeypatch.setattr(mobi_reader, "MAX_MOBI_IPC_BYTES", 128)

        def forbidden(*args, **kwargs):
            raise AssertionError("MOBI HTML reached the serving parent")

        monkeypatch.setattr(extraction, "BeautifulSoup", forbidden)
        partial = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert partial["status"] == "partial"
        assert partial["document_count"] == 1
        assert partial["reused_document_count"] == 1
        assert partial["skipped_source_count"] == 1
        assert "safety limit" in partial["skipped_sources"][0]["reason"]
        assert config.current_path.read_bytes() == selected

    asyncio.run(exercise())


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux pipe-size control"
)
def test_early_worker_pipe_refusal_remains_source_local_and_preserves_selected(
    project, monkeypatch
):
    from research_rag.corpus import mobi_reader

    async def exercise():
        write_mobi(project / "sources" / "clean.mobi")
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(config, FakeUltraRAG(), dense=FakeDenseBackend())
        assert (await service.ingest(chunk_size=50, chunk_overlap=10))[
            "status"
        ] == "ready"
        selected = config.current_path.read_bytes()
        write_mobi(project / "sources" / "refused.mobi")
        original = mobi_reader._worker_payload

        def large_request(path, request=None):
            return original(path, {**request, "padding": "x" * 60_000})

        monkeypatch.setattr(mobi_reader, "_worker_payload", large_request)
        processes = install_small_pipe_worker(
            monkeypatch,
            "import os,sys; os.close(0); sys.stderr.write('refused: corrupt source\\n'); sys.exit(1)",
        )
        partial = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert partial["status"] == "partial"
        assert partial["document_count"] == 1
        assert partial["reused_document_count"] == 1
        assert partial["skipped_source_count"] == 1
        assert "refused: corrupt source" in partial["skipped_sources"][0]["reason"]
        assert config.current_path.read_bytes() == selected
        assert len(processes) == 1
        assert processes[0].poll() == 1

    asyncio.run(exercise())
