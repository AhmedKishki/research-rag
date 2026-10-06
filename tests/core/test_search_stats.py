"""Search counts by rank, and the corpus facts the stats answer reads for free."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from research_rag.project.state_files import SEARCH_STATS_FILE
from research_rag.storage.search_stats import (
    read_appearances,
    read_history,
    read_search_stats,
    record_search,
)
from tests.conftest import write_epub, write_pdf, write_reviewed_metadata
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


def _service(
    project: Path, *, record_searches: bool = True, history: bool = True
) -> ResearchService:
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=[f"runtime.search_history={'true' if history else 'false'}"],
    )
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
        record_searches=record_searches,
    )


def test_a_search_counts_its_first_ranks_and_keeps_its_question_in_history(
    project: Path,
) -> None:
    """Each source and passage is counted at rank one and in the top five.

    A measurement's search is not a reader's, so a traced search and a service
    built not to record count nothing. The question is kept, with its filters, so
    it can be run again, and clearing the history leaves the counts.
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
            service.config,
            "cobalt.pdf",
            {"authors": ["One"], "year": 2021, "categories": ["political economy"]},
        )
        await service.ingest(chunk_size=100, chunk_overlap=10)
        await service.search(
            "cobalt labour", top_k=2, categories_any=["political economy"]
        )
        await service.search("cobalt evidence", top_k=2)
        await service.search("quartz mining", top_k=1)
        await service.search("cobalt", top_k=1, evaluation_trace=True)
        await _service(project, record_searches=False).search("cobalt", top_k=1)
        stats = await service.search_stats()
        history = await service.search_history(limit=10)
        cleared = await service.clear_search_history()
        after = await service.search_history()
        counted_after = await service.search_stats()
        return {
            **stats,
            "history": history,
            "cleared": cleared,
            "history_after": after,
            "search_count_after": counted_after["searches"]["search_count"],
        }

    stats = asyncio.run(exercise())
    history = stats["history"]
    assert [entry["query"] for entry in history["searches"]] == [
        "quartz mining",
        "cobalt evidence",
        "cobalt labour",
    ]
    assert history["searches"][2]["filters"] == {
        "categories_any": ["political economy"]
    }
    assert history["searches"][2]["requested_top_k"] == 2
    assert history["searches"][0]["caller"] == "unknown"
    assert history["recording"] is True
    assert stats["cleared"]["cleared"] == 3
    assert stats["history_after"]["searches"] == []
    assert stats["search_count_after"] == 3

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
    assert corpus["missing_metadata"] == {"authors": 0, "year": 2, "categories": 2}
    assert corpus["passages_per_source"]["minimum"] >= 1
    assert stats["last_build"]["created_vector_count"] is not None
    assert stats["generations"]["count"] == 1


def test_with_history_off_only_the_counts_are_kept(project: Path) -> None:
    async def exercise() -> tuple[int, bytes]:
        write_pdf(
            project / "sources" / "cobalt.pdf",
            ["Cobalt evidence about labour."],
            title="Cobalt Article",
        )
        service = _service(project, history=False)
        await service.ingest(chunk_size=100, chunk_overlap=10)
        await service.search("cobalt labour", top_k=1)
        history = await service.search_history()
        stored = (
            project / ".research-rag" / "runtime" / SEARCH_STATS_FILE
        ).read_bytes()
        return history["count"], stored

    kept, stored = asyncio.run(exercise())
    assert kept == 0
    assert b"cobalt labour" not in stored


def test_a_window_counts_only_recent_searches(tmp_path: Path) -> None:
    path = tmp_path / SEARCH_STATS_FILE
    for stamp, source in (
        ("2020-01-01T00:00:00Z", "old"),
        ("2999-01-01T00:00:00Z", "new"),
    ):
        record_search(
            path,
            searched_at=stamp,
            generation_id="g",
            requested_top_k=5,
            result_count=1,
            elapsed_ms=1.0,
            ranked=[(1, f"chunk_{source}", f"src_{source}")],
            query=source,
        )

    everything = read_search_stats(path)
    recent = read_search_stats(path, since_days=1)

    assert everything["search_count"] == 2
    assert recent["search_count"] == 1
    assert [entry["source_id"] for entry in recent["sources"]] == ["src_new"]
    assert read_appearances(path, chunk_ids=["chunk_old", "chunk_new"]) == {
        "chunk_old": (1, 1),
        "chunk_new": (1, 1),
    }
    assert read_history(path, since_days=1)["searches"][0]["query"] == "new"


def test_a_version_one_counts_file_gains_history_and_keeps_its_rows(
    tmp_path: Path,
) -> None:
    import sqlite3

    path = tmp_path / SEARCH_STATS_FILE
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE searches (id INTEGER PRIMARY KEY, searched_at TEXT NOT NULL,
          generation_id TEXT NOT NULL, requested_top_k INTEGER NOT NULL,
          result_count INTEGER NOT NULL, elapsed_ms REAL NOT NULL);
        CREATE TABLE appearances (search_id INTEGER NOT NULL, rank INTEGER NOT NULL,
          chunk_id TEXT NOT NULL, source_id TEXT NOT NULL,
          PRIMARY KEY (search_id, rank)) WITHOUT ROWID;
        INSERT INTO searches VALUES (1, '2026-10-01T00:00:00Z', 'g', 5, 1, 9.0);
        INSERT INTO appearances VALUES (1, 1, 'c', 's');
        PRAGMA user_version = 1;
        """
    )
    connection.commit()
    connection.close()

    assert read_search_stats(path)["search_count"] == 1
    assert read_history(path)["searches"] == []
    record_search(
        path,
        searched_at="2026-10-02T00:00:00Z",
        generation_id="g",
        requested_top_k=5,
        result_count=0,
        elapsed_ms=1.0,
        ranked=[],
        query="after the upgrade",
    )
    assert read_history(path)["searches"][0]["query"] == "after the upgrade"


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


def test_a_source_lists_its_passages_with_how_often_each_was_returned(
    project: Path,
) -> None:
    async def exercise() -> dict[str, object]:
        write_pdf(
            project / "sources" / "cobalt.pdf",
            [
                f"Cobalt evidence about labour, page {page}. " * 30
                for page in range(1, 6)
            ],
            title="Cobalt Article",
        )
        service = _service(project)
        await service.ingest(chunk_size=60, chunk_overlap=10)
        found = await service.search("cobalt labour", top_k=3)
        source_id = found["hits"][0]["source_id"]
        first = await service.source_chunks(source_id=source_id, page_size=2)
        last = await service.source_chunks(
            source_path="cobalt.pdf", page=99, page_size=2
        )
        with pytest.raises(ResearchError, match="no such source"):
            await service.source_chunks(source_id="src_unknown")
        with pytest.raises(ResearchError, match="page must be"):
            await service.source_chunks(source_id=source_id, page=0)
        return {"first": first, "last": last, "top": found["hits"][0]["chunk_id"]}

    answered = asyncio.run(exercise())
    first, last = answered["first"], answered["last"]  # type: ignore[assignment]

    assert first["source"]["title"] == "Cobalt Article"
    assert first["source"]["passage_count"] >= 4
    assert first["pages"] == -(-first["source"]["passage_count"] // 2)
    assert [chunk["ordinal"] for chunk in first["chunks"]] == [1, 2]
    assert first["source"]["top_five"] >= 1
    assert first["source"]["rank_one"] == 1
    assert all(chunk["text"] for chunk in first["chunks"])
    assert all(chunk["locator"]["type"] == "pdf_page" for chunk in first["chunks"])
    # A page past the end is the last page, and its numbering continues.
    assert last["page"] == last["pages"]
    assert last["chunks"][0]["ordinal"] == (last["pages"] - 1) * 2 + 1
    counted = {
        chunk["chunk_id"]: chunk["top_five"]
        for page in (first, last)
        for chunk in page["chunks"]
    }
    assert counted.get(answered["top"], 0) >= 1


def test_the_largest_sources_are_ranked_by_what_the_reader_asks(project: Path) -> None:
    async def exercise() -> dict[str, object]:
        for name, pages in (("short", 1), ("long", 3), ("middle", 2)):
            write_pdf(
                project / "sources" / f"{name}.pdf",
                [
                    f"Evidence about {name} labour, page {n}. " * 12
                    for n in range(pages)
                ],
                title=name.title(),
            )
        write_epub(
            project / "sources" / "book.epub",
            "A whole book of argument about labour and its fetishism. " * 400,
            title="Book",
        )
        service = _service(project)
        write_reviewed_metadata(
            service.config,
            "long.pdf",
            {"authors": ["Ada"], "categories": ["labour", "theory"]},
        )
        write_reviewed_metadata(
            service.config, "short.pdf", {"authors": ["Ada"], "categories": ["labour"]}
        )
        await service.ingest(chunk_size=60, chunk_overlap=10)
        by_size = await service.search_stats(largest_by="size", top=2)
        by_passages = await service.search_stats(largest_by="passages", top=3)
        with pytest.raises(ResearchError, match="largest_by"):
            await service.search_stats(largest_by="words")
        return {"size": by_size, "passages": by_passages}

    answered = asyncio.run(exercise())
    by_size, passages = answered["size"], answered["passages"]  # type: ignore[assignment]

    # An EPUB has no pages, so size is the measure every format shares: the book
    # ranks first on its text, with no page count to put it last.
    largest = by_size["corpus"]["largest_sources"]
    assert by_size["corpus"]["largest_by"] == "size"
    assert [entry["source_relative_path"] for entry in largest] == [
        "book.epub",
        "long.pdf",
    ]
    assert largest[0]["physical_pages"] is None
    assert largest[0]["text_bytes"] > largest[1]["text_bytes"] > 0
    assert all(entry["source_id"] for entry in largest)
    assert len(passages["corpus"]["largest_sources"]) == 3
    counts = passages["corpus"]
    assert counts["categories"][0] == {"value": "labour", "count": 2}
    assert {"value": "Ada", "count": 2} in counts["authors"]
    assert counts["formats"] == [
        {"value": "pdf", "count": 3},
        {"value": "epub", "count": 1},
    ]
