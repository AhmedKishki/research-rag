"""Search counts by rank, and the corpus facts the stats answer reads for free."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.state_files import SEARCH_STATS_FILE
from research_rag.storage.search_stats import read_search_stats, record_search
from tests.conftest import write_pdf, write_reviewed_metadata
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


def _service(project: Path, *, record_searches: bool = True) -> ResearchService:
    config = resolve_config(project, vanilla_executable=sys.executable)
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
        record_searches=record_searches,
    )


def test_a_search_counts_its_first_ranks_and_never_its_query(project: Path) -> None:
    """Each source and passage is counted at rank one and in the top five.

    A measurement's search is not a reader's, so a traced search and a service
    built not to record count nothing. The query is never written down.
    """

    async def exercise() -> dict[str, object]:
        write_pdf(
            project / "sources" / "cobalt.pdf",
            ["Cobalt evidence about labour and artificial intelligence."],
            title="Cobalt Article",
        )
        write_pdf(
            project / "sources" / "quartz.pdf",
            ["Quartz material about mining and its labour."],
            title="Quartz Article",
        )
        write_pdf(
            project / "sources" / "unread.pdf",
            ["Silence about anything a query here asks for."],
            title="Unread Article",
        )
        service = _service(project)
        write_reviewed_metadata(
            service.config, "cobalt.pdf", {"authors": ["One"], "year": 2021}
        )
        await service.ingest(chunk_size=100, chunk_overlap=10)
        await service.search("cobalt labour", top_k=2)
        await service.search("cobalt evidence", top_k=2)
        await service.search("quartz mining", top_k=1)
        await service.search("cobalt", top_k=1, evaluation_trace=True)
        await _service(project, record_searches=False).search("cobalt", top_k=1)
        return await service.search_stats()

    stats = asyncio.run(exercise())

    searches = stats["searches"]
    assert searches["search_count"] == 3
    assert searches["zero_result_count"] == 0
    assert searches["searches_by_day"][0]["count"] == 3
    assert searches["median_elapsed_ms"] is not None
    sources = {entry["title"]: entry for entry in stats["sources"]}
    assert sources["Cobalt Article"]["rank_one"] == 2
    assert sources["Quartz Article"]["rank_one"] == 1
    assert sum(entry["top_five"] for entry in stats["sources"]) == sum(
        entry["top_five"] for entry in stats["passages"]
    )
    assert all(entry["in_current_generation"] for entry in stats["passages"])
    assert stats["unreached_source_count"] == 1
    assert stats["unreached_sources"][0]["title"] == "Unread Article"

    corpus = stats["corpus"]
    assert corpus["source_count"] == 3
    assert corpus["formats"] == [{"value": "pdf", "count": 3}]
    assert corpus["missing_metadata"] == {"authors": 0, "year": 2, "categories": 3}
    assert corpus["passages_per_source"]["minimum"] >= 1
    assert stats["last_build"]["created_vector_count"] is not None
    assert stats["generations"]["count"] == 1

    stored = (project / ".research-rag" / "runtime" / SEARCH_STATS_FILE).read_bytes()
    assert b"cobalt labour" not in stored
    assert b"quartz mining" not in stored


def test_counts_start_at_zero_and_track_only_the_first_five(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / SEARCH_STATS_FILE

    assert read_search_stats(path)["search_count"] == 0
    record_search(
        path,
        searched_at="2026-10-06T09:00:00Z",
        generation_id="g1",
        requested_top_k=8,
        result_count=8,
        elapsed_ms=12.0,
        ranked=[
            (rank, f"c{rank}", "src_a" if rank == 1 else "src_b")
            for rank in range(1, 9)
        ],
    )
    record_search(
        path,
        searched_at="2026-10-07T09:00:00Z",
        generation_id="g1",
        requested_top_k=8,
        result_count=0,
        elapsed_ms=30.0,
        ranked=[],
    )
    stats = read_search_stats(path)

    assert stats["search_count"] == 2
    assert stats["zero_result_count"] == 1
    assert stats["mean_result_count"] == 4.0
    assert [day["day"] for day in stats["searches_by_day"]] == [
        "2026-10-07",
        "2026-10-06",
    ]
    assert stats["sources"] == [
        {"source_id": "src_b", "top_five": 4, "rank_one": 0},
        {"source_id": "src_a", "top_five": 1, "rank_one": 1},
    ]
    assert len(stats["passages"]) == 5
    assert stats["median_elapsed_ms"] == 21.0
