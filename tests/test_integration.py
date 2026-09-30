"""End-to-end coverage against the real vanilla UltraRAG gateway.

The app answers in process, so there is no handshake, no tool schema, and no
resource to read: what this file proves is that the workspace reaches the same
generations, indexes, review state, and gateway as the terminal does, and that a
gateway that cannot start is reported by the operation that needed it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import write_pdf

from research_rag.config import resolve_config
from research_rag.service import ResearchService
from research_rag.ultrarag import LazyGateway, VanillaUltraRAG

# Answer-level keys the workspace always carries, because it always returns the
# complete payload rather than a projection an agent reads.
FULL_ANSWER_KEYS = (
    "allowed_formats",
    "available_retrieval_methods",
    "categories",
    "ignored_extensions",
    "projects",
    "retrieval",
    "source_exclusion_revision",
    "ui_launcher",
    "version",
)

# The same rule one level down, for a returned passage.
FULL_HIT_KEYS = (
    "citation",
    "component_ranks",
    "component_scores",
    "direct_quote_safe",
    "doi",
    "document_id",
    "fusion_score",
    "metadata_provenance",
    "rank",
    "rerank_score",
    "source_id",
    "source_path",
    "text_fidelity",
    "text_notes",
    "title",
    "year",
)


class Research:
    """One `ResearchService` over a lazily opened gateway, closed with the test.

    The gateway opens on the first call that needs it, which is how the terminal
    and the workspace both work, so a test can prove that a project reads fine
    against a gateway that could never start.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.gateway = LazyGateway(config)
        self.service: ResearchService | None = None

    async def __aenter__(self) -> ResearchService:
        self.service = ResearchService(
            self.config, VanillaUltraRAG(self.gateway, self.config)
        )
        return self.service

    async def __aexit__(self, *_: object) -> None:
        await self.gateway.aclose()


# The gateway is a separate product with its own console script, so the app
# resolves it beside the running interpreter rather than assuming the module.
VANILLA_EXECUTABLE = Path(sys.executable).parent / "vanilla-ultra-rag-mcp"


def _config(project: Path, **options: Any) -> Any:
    options.setdefault("vanilla_executable", VANILLA_EXECUTABLE)
    return resolve_config(project, **options)


async def _ingest_until_complete(service: ResearchService) -> dict[str, Any]:
    while True:
        ingestion = await service.ingest()
        if ingestion["status"] != "in_progress":
            return ingestion


def _search(
    service: ResearchService, query: str, top_k: int = 1, **filters: Any
) -> dict[str, Any]:
    return asyncio.run(
        service.search(
            query, top_k=top_k, retrieval_method="hybrid", rerank=True, **filters
        )
    )


@pytest.mark.integration
def test_the_real_vanilla_research_flow(project: Path) -> None:
    write_pdf(
        project / "sources" / "evidence.pdf",
        [
            (
                "The cobalt heron is evidence for a page-aware research result. "
                "Its habitat is the amber marsh, where labour and ecology meet."
            )
        ],
        title="Citable Evidence",
    )
    (project / "sources" / "notes.md").write_text(
        "cobalt heron derived notes must not be indexed",
        encoding="utf-8",
    )

    async def scenario() -> None:
        async with Research(_config(project)) as service:
            initial = await service.status()
            # Nothing can be served yet, and that is the one readiness fact worth
            # saying before a build.
            assert initial["ready"] is False

            # Reviewed metadata is a hand-edited review-state file; it applies at
            # read time, so it is written directly and checked through the service.
            review_state = project / ".research-rag" / "source-metadata.json"
            review_state.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": {
                            "evidence.pdf": {
                                "categories": ["research"],
                                "keywords": ["wetland"],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            ingested = await _ingest_until_complete(service)
            assert ingested["status"] == "ready"
            assert ingested["generation_changed"] is True
            assert ingested["document_count"] == 1

            current = json.loads(
                (project / ".research-rag" / "runtime" / "current.json").read_text(
                    encoding="utf-8"
                )
            )
            generation_root = (
                project
                / ".research-rag"
                / "runtime"
                / "generations"
                / current["generation_id"]
            )
            manifest = json.loads(
                (generation_root / "manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["retrieval"]["dense"]["point_count"] == 1
            # The default `auto` backend is the exact scan below the corpus
            # threshold, and the manifest records which backend owns the index.
            assert (
                manifest["retrieval"]["dense"]["dense_backend"]
                == "portable-exact-vectors"
            )
            assert manifest["files"]["dense_index"] == "indexes/vectors"
            assert (generation_root / "indexes" / "vectors" / "index.json").is_file()

            ready = await service.status()
            assert ready["ready"] is True
            assert ready["hybrid_ready"] is True
            assert ready["generation_upgrade_required"] is False
            assert ready["retained_generation_count"] >= 1
            assert ready["retained_generation_bytes"] > 0
            for key in FULL_ANSWER_KEYS:
                assert key in ready, key

            # A stale status counts what the corpus gained and names the sources
            # that went missing.
            added_source = project / "sources" / "added-later.pdf"
            write_pdf(added_source, ["Amber marsh evidence added after the build."])
            added_status = await service.status()
            assert added_status["stale"] is True
            assert added_status["changes"]["added"] == ["added-later.pdf"]
            assert added_status["changes"]["modified"] == []
            assert added_status["changes"]["removed"] == []

            removed_source = project / "sources" / "evidence.pdf"
            removed_bytes = removed_source.read_bytes()
            removed_source.unlink()
            missing_status = await service.status()
            assert missing_status["changes"]["removed"] == ["evidence.pdf"]
            removed_source.write_bytes(removed_bytes)
            added_source.unlink()
            settled_status = await service.status()
            assert settled_status["stale"] is False
            assert settled_status["changes"]["added"] == []
            assert settled_status["changes"]["modified"] == []

            result = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                retrieval_method="hybrid",
                rerank=True,
            )
            initial_hit = result["hits"][0]
            assert initial_hit["source_relative_path"] == "evidence.pdf"
            assert initial_hit["locator"]["page"] == 1
            assert "cobalt heron" in initial_hit["text"].lower()
            assert "notes" not in initial_hit["text"].lower()
            for key in FULL_HIT_KEYS:
                assert key in initial_hit, key
            assert initial_hit["direct_quote_safe"] is False
            assert initial_hit["rerank_score"] is not None
            assert result["retrieval_method"] == "hybrid"

            manifest_before_metadata = (generation_root / "manifest.json").read_bytes()
            chunks_before_metadata = (
                generation_root / "chunks" / "chunks.jsonl"
            ).read_bytes()
            review_state.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": {
                            "evidence.pdf": {
                                "title": "Reviewed Marsh Evidence",
                                "authors": ["Field Researcher"],
                                "year": 2025,
                                "categories": ["corrected"],
                                "keywords": ["heron"],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            # A read-time overlay never rewrites a generation.
            assert (
                generation_root / "manifest.json"
            ).read_bytes() == manifest_before_metadata
            assert (
                generation_root / "chunks" / "chunks.jsonl"
            ).read_bytes() == chunks_before_metadata

            metadata_status = await service.status()
            assert metadata_status["stale"] is False
            assert metadata_status["metadata_overlay_active"] is True

            inventory = await service.list_sources()
            source_record = inventory["sources"][0]
            assert source_record["source_id"].startswith("src_")
            assert source_record["title"] == "Reviewed Marsh Evidence"

            corrected = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                categories_any=["corrected"],
                keywords=["heron"],
                retrieval_method="hybrid",
                rerank=True,
            )
            hit = corrected["hits"][0]
            assert hit["chunk_id"] == initial_hit["chunk_id"]
            assert hit["authors"] == ["Field Researcher"]

            context = await service.get_passage(hit["chunk_id"])
            passage = context["context"][0]
            assert passage["source_relative_path"] == "evidence.pdf"
            assert passage["authors"] == ["Field Researcher"]

            filtered_out = await service.search(
                "wetland bird",
                top_k=1,
                categories_any=["unrelated"],
                retrieval_method="hybrid",
                rerank=True,
            )
            assert filtered_out["hits"] == []

            # A reviewed author and title are what a name filter matches, and a
            # substring is enough for either.
            by_name = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                authors_any=["field researcher"],
                titles_any=["marsh evidence"],
                retrieval_method="hybrid",
                rerank=True,
            )
            assert by_name["hits"][0]["chunk_id"] == hit["chunk_id"]

            # A name no source carries is a filtered answer rather than a silent
            # corpus, so the answer names the filter it applied.
            unmatched_name = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                authors_any=["Nobody At All"],
                retrieval_method="hybrid",
                rerank=True,
            )
            assert unmatched_name["hits"] == []
            assert unmatched_name["filters"]["authors_any"] == ["nobody at all"]

            excluded = await service.set_source_inclusion(
                source_path="evidence.pdf",
                included=False,
                reason="Reviewed duplicate representation test.",
            )
            assert excluded["effective_immediately"] is True
            assert (project / "sources" / "evidence.pdf").is_file()
            after_exclusion = await service.search(
                "cobalt heron",
                top_k=1,
                retrieval_method="hybrid",
                rerank=True,
            )
            assert after_exclusion["hits"] == []

            restored = await service.set_source_inclusion(
                source_path="evidence.pdf", included=True
            )
            assert restored["effective_immediately"] is True
            after_restore = await service.search(
                "cobalt heron",
                top_k=1,
                retrieval_method="hybrid",
                rerank=True,
            )
            assert len(after_restore["hits"]) == 1

    asyncio.run(scenario())

    # A second process starts from the same caches with nothing to download.
    async def offline_scenario() -> None:
        async with Research(_config(project, offline=True)) as service:
            result = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                retrieval_method="hybrid",
                rerank=True,
            )
            assert result["retrieval_method"] == "hybrid"
            assert result["hits"][0]["rerank_score"] is not None
            assert result["filters"]["active_document_count"] == 1
            assert result["withheld_candidates"]["total"] == 0
            assert result["hits"][0]["text_fidelity"] == "cleaned_semantic_text"
            status = await service.status()
            assert status["generation_upgrade_required"] is False
            assert status["retrieval"]["available_methods"] == [
                "bm25",
                "dense",
                "hybrid",
            ]

    asyncio.run(offline_scenario())


@pytest.mark.integration
def test_a_gateway_that_cannot_start_is_reported_by_the_operation_that_needed_it(
    project: Path,
) -> None:
    """The reason is the gateway's own log tail, not a connection that closed.

    Nothing here opens a gateway until an operation needs one, so a status read
    succeeds against a gateway that cannot start, and the failure arrives with
    the reason the gateway wrote to its own log plus the paths a reader has to
    open to see the rest.
    """

    write_pdf(project / "sources" / "evidence.pdf", ["The cobalt heron is evidence."])
    broken = project / "broken-vanilla-gateway"
    broken.write_text(
        "#!/bin/sh\necho 'no runtime at /nowhere' >&2\nexit 3\n", encoding="utf-8"
    )
    broken.chmod(0o755)
    config = _config(project, vanilla_executable=str(broken))

    async def scenario() -> None:
        async with Research(config) as service:
            assert (await service.status())["ready"] is False
            # A search refuses a project with no generation before it reaches for
            # anything, so the build is the first operation that needs the gateway.
            with pytest.raises(Exception) as failure:
                await service.ingest()

        message = str(failure.value)
        assert "The UltraRAG gateway could not start" in message
        # The reason is the line the gateway itself printed, not the symptom a
        # caller would otherwise be left with.
        assert "no runtime at /nowhere" in message
        runtime_root = project / ".research-rag" / "runtime"
        assert str(runtime_root / "logs" / "vanilla-gateway-stderr.log") in message
        assert str(runtime_root / "ultrarag-runtime" / "logs") in message

    asyncio.run(scenario())


@pytest.mark.integration
def test_the_terminal_and_the_workspace_answer_the_same_payload(project: Path) -> None:
    """One engine, one payload: a CLI-shaped call and a workspace call agree."""

    write_pdf(project / "sources" / "evidence.pdf", ["The cobalt heron is evidence."])

    async def scenario() -> None:
        async with Research(_config(project)) as service:
            await _ingest_until_complete(service)
            direct = await service.search(
                "cobalt heron", top_k=2, retrieval_method="hybrid", rerank=True
            )
            from research_rag.ui import ResearchUIAdapter

            adapter = ResearchUIAdapter(service.config, service)
            through_workspace = await adapter.call(
                "search", {"query": "cobalt heron", "top_k": 2}
            )

            assert direct == dict(through_workspace)

    asyncio.run(scenario())


@pytest.mark.integration
def test_a_neighbouring_markdown_file_is_not_indexed(project: Path) -> None:
    write_pdf(project / "sources" / "evidence.pdf", ["The cobalt heron is evidence."])
    (project / "sources" / "notes.md").write_text(
        "cobalt heron derived notes", encoding="utf-8"
    )

    async def scenario() -> None:
        async with Research(_config(project)) as service:
            ingested = await _ingest_until_complete(service)
            assert ingested["document_count"] == 1
            inventory = await service.list_sources()
            assert [item["source_relative_path"] for item in inventory["sources"]] == [
                "evidence.pdf"
            ]

    asyncio.run(scenario())
