"""The direct backend serves one app process, in process.

The workspace and the terminal reach the same generation, both indexes, review
state, and direct retrieval backend, so there is no handshake, tool schema, or
child process to start. Offline mode repeats a hybrid reranked search from the
caches the build left behind.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.retrieval.direct import DirectRetrieval
from tests.conftest import write_pdf

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
    "version",
)

# The same rule one level down, for a returned passage.
FULL_HIT_KEYS = (
    "citation",
    "component_ranks",
    "component_scores",
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
    """One direct backend and the service that speaks through it, closed together."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.backend = DirectRetrieval(config)
        self.service: ResearchService | None = None

    async def __aenter__(self) -> ResearchService:
        self.service = ResearchService(self.config, self.backend)
        return self.service

    async def __aexit__(self, *_: object) -> None:
        await self.backend.aclose()


def _config(project: Path, **options: Any) -> Any:
    return resolve_config(project, **options)


async def _ingest_until_complete(service: ResearchService) -> dict[str, Any]:
    while True:
        ingestion = await service.ingest()
        if ingestion["status"] != "in_progress":
            return ingestion


@pytest.mark.integration
def test_the_real_direct_research_flow(project: Path) -> None:
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
            assert initial["ready"] is False

            # Reviewed metadata applies at read time, so it is written directly
            # and checked through the service.
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
            inventory = await service.list_sources()
            assert [item["source_relative_path"] for item in inventory["sources"]] == [
                "evidence.pdf"
            ]

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
            # Both indexes exist and are recorded: the lexical BM25 index and the
            # portable dense vectors.
            assert (generation_root / manifest["files"]["bm25_index"]).is_dir()
            assert (generation_root / "indexes" / "vectors" / "index.json").is_file()
            assert manifest["retrieval"]["dense"]["point_count"] == 1
            assert (
                manifest["retrieval"]["dense"]["dense_backend"]
                == "portable-exact-vectors"
            )
            assert manifest["files"]["dense_index"] == "indexes/vectors"

            ready = await service.status()
            assert ready["ready"] is True
            assert ready["hybrid_ready"] is True
            assert ready["generation_upgrade_required"] is False
            assert ready["retained_generation_count"] >= 1
            assert ready["retained_generation_bytes"] > 0
            for key in FULL_ANSWER_KEYS:
                assert key in ready, key

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
            assert "source_relative_path" not in initial_hit
            assert initial_hit["locator"]["page"] == 1
            assert "cobalt heron" in initial_hit["text"].lower()
            assert "notes" not in initial_hit["text"].lower()
            for key in FULL_HIT_KEYS:
                assert key in initial_hit, key
            assert initial_hit["rerank_score"] is not None
            assert result["retrieval_method"] == "hybrid"

            # The dense half alone answers the same known passage.
            dense = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                retrieval_method="dense",
                rerank=True,
            )
            assert dense["retrieval_method"] == "dense"
            assert dense["hits"][0]["chunk_id"] == initial_hit["chunk_id"]

            from research_rag.surfaces.ui import ResearchUIAdapter

            adapter = ResearchUIAdapter(service.config, service)
            through_workspace = await adapter.call(
                "search", {"query": "cobalt heron amber marsh", "top_k": 1}
            )
            assert result == dict(through_workspace)

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
            assert "source_relative_path" not in passage
            assert passage["authors"] == ["Field Researcher"]

            filtered_out = await service.search(
                "wetland bird",
                top_k=1,
                categories_any=["unrelated"],
                retrieval_method="hybrid",
                rerank=True,
            )
            assert filtered_out["hits"] == []

            by_name = await service.search(
                "cobalt heron amber marsh",
                top_k=1,
                authors_any=["field researcher"],
                titles_any=["marsh evidence"],
                retrieval_method="hybrid",
                rerank=True,
            )
            assert by_name["hits"][0]["chunk_id"] == hit["chunk_id"]

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
def test_a_neighbour_exclusion_is_held_by_the_real_backend(project: Path) -> None:
    """A passage decision applies to the corpus that has neighbours to withhold from.

    Neither retrieval half may answer with the excluded passage, and a reader that
    opens the passage beside it still sees the decision on that neighbour.
    """

    write_pdf(
        project / "sources" / "wide.pdf",
        [
            "Cobalt labour evidence of the northern mine. "
            + "detail " * 40
            + "end one.",
            "Cobalt labour evidence of the southern mine. "
            + "detail " * 40
            + "end two.",
        ],
    )

    async def scenario() -> None:
        async with Research(_config(project)) as service:
            await _ingest_until_complete(service)
            hits = (
                await service.search("cobalt labour evidence", top_k=10, rerank=False)
            )["hits"]
            assert len(hits) >= 2
            target = hits[0]["chunk_id"]
            context = await service.get_passage(target, context_chunks=1)
            neighbours = [
                passage["chunk_id"]
                for passage in context["context"]
                if passage["chunk_id"] != target
            ]
            assert neighbours
            neighbour = neighbours[0]

            decision = await service.set_chunk_inclusion(
                neighbour, included=False, reason="Reviewed fragment."
            )
            assert decision["in_current_generation"] is True
            assert decision["effective_immediately"] is True

            for method in ("bm25", "dense", "hybrid"):
                answer = await service.search(
                    "cobalt labour evidence",
                    top_k=10,
                    rerank=False,
                    retrieval_method=method,
                )
                assert neighbour not in [hit["chunk_id"] for hit in answer["hits"]], (
                    method
                )

            beside = await service.get_passage(target, context_chunks=1)
            marked = {passage["chunk_id"]: passage for passage in beside["context"]}
            assert neighbour in marked
            assert marked[neighbour]["excluded_from_search"] is True
            assert "excluded_from_search" not in marked[target]

    asyncio.run(scenario())
