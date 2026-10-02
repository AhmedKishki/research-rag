"""A `chunk_id` is derived from content, so a decision outlives a rebuild of unchanged bytes.

The retrieval filter enforces it, because it sees the same identifier whatever the
generation holds, and the report names the generation it was recorded in, because
that is the only thing telling a reader whether it withholds anything now.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchError, ResearchService
from research_rag.project.config import resolve_config
from research_rag.storage.records import (
    StorageError,
    load_chunk_exclusions,
    write_chunk_exclusions,
)
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


async def _service(
    project: Path,
    *,
    pages: int = 1,
) -> tuple[ResearchService, list[dict[str, str]]]:
    """Build a project whose passages all answer one query, and search it once."""

    write_pdf(
        project / "sources" / "wide.pdf",
        [
            "Cobalt labour evidence of the northern mine, "
            + ("detail " * 20 + f"section {number}.")
            for number in range(1, pages + 1)
        ],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=60, chunk_overlap=10)
    found = await service.search("labour evidence", top_k=10, rerank=False)
    return service, found["hits"]


def _write_exclusion_file(project: Path, value: object) -> Path:
    """Write the chunk-exclusion file the way a person would: by hand."""

    path = project / ".research-rag" / "chunk-exclusions.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_a_decision_about_one_passage_withholds_it_from_every_retrieval_half(
    project: Path,
) -> None:
    """Neither BM25 nor the dense index may answer with the passage.

    They rank the same candidates and merge them afterwards, so a filter added to only
    one half would show up as a passage reappearing whenever the other half wins.
    """

    async def exercise() -> None:
        service, hits = await _service(project)
        assert len(hits) == 1
        target = hits[0]["chunk_id"]

        for method in ("bm25", "dense", "hybrid"):
            await service.set_chunk_inclusion(
                target, included=False, reason="Misread extraction."
            )
            answer = await service.search(
                "labour evidence",
                top_k=10,
                rerank=False,
                retrieval_method=method,
            )
            assert answer["hits"] == [], method
            assert answer["result_count"] == 0, method
            assert answer["excluded_chunk_count"] == 1, method
            await service.set_chunk_inclusion(target, included=True)

    asyncio.run(exercise())


def test_an_excluded_passage_is_refused_by_name_and_kept_as_a_neighbour(
    project: Path,
) -> None:
    """Hiding a neighbour would remove the shape of the argument to a decision about one
    passage inside it, and a reader who opened a passage to see what surrounds it asked a
    question the exclusion never answered.
    """

    async def exercise() -> None:
        service, hits = await _service(project, pages=3)
        # Which two passages are neighbours is asked of the corpus, because a ranking
        # says which passage is strongest and not what sits beside it.
        adjacent = [
            (passage["chunk_id"], hit["chunk_id"])
            for hit in hits
            for passage in (
                await service.get_passage(hit["chunk_id"], context_chunks=1)
            )["context"]
            if passage["chunk_id"] != hit["chunk_id"]
        ]
        assert adjacent
        target, reader = adjacent[0]

        await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )

        with pytest.raises(ResearchError, match="chunk is currently excluded"):
            await service.get_passage(target)

        context = await service.get_passage(reader, context_chunks=1)
        assert target in [passage["chunk_id"] for passage in context["context"]]
        # The passage the reader asked for is still the passage it named.
        assert context["requested_chunk_id"] == reader

    asyncio.run(exercise())


def test_a_decision_is_reversible_and_the_answer_names_what_it_did(
    project: Path,
) -> None:
    """Excluding twice with one reason changes nothing the second time."""

    async def exercise() -> None:
        service, hits = await _service(project)
        target = hits[0]["chunk_id"]

        excluded = await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )
        assert excluded["status"] == "changed"
        assert excluded["source_relative_path"] == "wide.pdf"
        assert excluded["locator"].startswith("p. ")
        assert excluded["in_current_generation"] is True
        assert excluded["effective_immediately"] is True
        # A filter applies the decision to the generation on screen and to every one built
        # after it, so nothing has to be rebuilt for it to hold.
        assert excluded["generation_rebuild_recommended"] is False
        assert "no ingestion is needed" in excluded["message"]

        repeated = await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )
        assert repeated["status"] == "unchanged"
        assert "already excluded" in repeated["message"]

        with pytest.raises(ResearchError, match="non-empty reason"):
            await service.set_chunk_inclusion(target, included=False)

        restored = await service.set_chunk_inclusion(target, included=True)
        assert restored["status"] == "changed"
        assert restored["reason"] is None
        assert load_chunk_exclusions(service.config.chunk_exclusions_path) == {}
        assert (await service.search("labour evidence", rerank=False))["hits"] == hits

        assert (await service.set_chunk_inclusion(target, included=True))[
            "status"
        ] == "unchanged"

    asyncio.run(exercise())


def test_a_decision_about_a_passage_this_generation_does_not_hold_is_reported(
    project: Path,
) -> None:
    """A removed chunk cannot come back, so the answer says so instead of failing.

    `stale` names `ingest` as the call that closes the gap, and no ingestion brings a
    removed chunk back.
    """

    async def exercise() -> None:
        service, hits = await _service(project)
        target = hits[0]["chunk_id"]
        await service.set_chunk_inclusion(target, included=False, reason="Reviewed.")
        generation_id = str(service._load_current_optional()[1]["generation_id"])

        await service.set_chunk_inclusion(
            "chk_deadbeefdeadbeef", included=False, reason="Copied from an old search."
        )
        write_chunk_exclusions(
            service.config.chunk_exclusions_path,
            {
                "chk_deadbeefdeadbeef": {
                    "reason": "Copied from an old search.",
                    "excluded_at": "2026-09-30T19:12:35.104Z",
                    "generation_id": "20260101T000000Z-abcdef",
                }
            },
        )

        listed = await service.list_chunk_exclusions()
        assert listed["generation_id"] == generation_id
        assert [entry["chunk_id"] for entry in listed["exclusions"]] == [
            "chk_deadbeefdeadbeef"
        ]
        assert listed["exclusions"][0]["in_current_generation"] is False
        # Nothing on disk can say where that passage was, so nothing is invented
        # for the row a reader has to delete by hand.
        assert listed["exclusions"][0]["source_relative_path"] == ""
        assert listed["exclusions"][0]["locator"] == ""
        assert "no ingestion restores a removed chunk" in listed["message"]

        status = await service.status()
        assert status["excluded_chunk_count"] == 1
        assert status["chunk_exclusion_other_generation_count"] == 1
        assert status["stale"] is False
        assert (await service.status())["ready"] is True
        assert "another generation" in status["message"]
        # An entry naming a chunk this generation does not hold removes nothing
        # from the passages a reader can reach.
        assert len((await service.search("labour evidence", rerank=False))["hits"]) == 1

    asyncio.run(exercise())


def test_a_project_with_no_generation_refuses_to_record_a_passage_decision(
    project: Path,
) -> None:
    """A chunk id belongs to a generation, so there is nothing honest to record."""

    async def exercise() -> None:
        write_pdf(project / "sources" / "wide.pdf", ["Cobalt labour evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )

        with pytest.raises(ResearchError, match="no chunk can be named"):
            await service.set_chunk_inclusion(
                "chk_1a2b3c", included=False, reason="Reviewed."
            )
        assert not service.config.chunk_exclusions_path.exists()
        assert (await service.status())["excluded_chunk_count"] == 0

    asyncio.run(exercise())


def test_top_k_counts_the_passages_a_reader_kept(project: Path, monkeypatch) -> None:
    """The lexical window widens while a filter is in force, so an exclusion must not shrink
    the answer, or empty it.

    Without that, a search answers with fewer passages than the corpus still offers, and
    with none while the whole window is excluded.
    """

    monkeypatch.setenv("RESEARCH_RAG_RETRIEVAL_MINIMUM_CANDIDATES", "2")

    async def exercise() -> None:
        service, _hits = await _service(project, pages=12)
        ranked = await service.search(
            "cobalt labour evidence",
            top_k=20,
            rerank=False,
            retrieval_method="bm25",
        )
        assert len(ranked["hits"]) == 12

        # One window holds four candidates per passage the search is asked for, so ten
        # exclusions leave the last two passages of the corpus outside it.
        for hit in ranked["hits"][:10]:
            await service.set_chunk_inclusion(
                hit["chunk_id"], included=False, reason="Reviewed fragment."
            )

        kept = await service.search(
            "cobalt labour evidence",
            top_k=2,
            rerank=False,
            retrieval_method="bm25",
        )
        assert len(kept["hits"]) == 2
        assert {hit["chunk_id"] for hit in kept["hits"]} == {
            hit["chunk_id"] for hit in ranked["hits"][10:]
        }
        assert kept["excluded_chunk_count"] == 10

    asyncio.run(exercise())


def test_the_dense_window_is_narrowed_by_the_candidates_a_query_will_drop(
    project: Path,
) -> None:
    """A post-query drop must not be paid for out of the answer.

    The dense backend filters by document only, so an excluded chunk is still returned and
    dropped here. The window it is asked for is therefore one narrower than the active
    chunk count, the same compensation the lexical half gets from widening.
    """

    async def exercise() -> None:
        service, hits = await _service(project)
        before = await service.search(
            "labour evidence", top_k=10, rerank=False, retrieval_method="dense"
        )

        await service.set_chunk_inclusion(
            hits[0]["chunk_id"], included=False, reason="Reviewed fragment."
        )
        after = await service.search(
            "labour evidence", top_k=10, rerank=False, retrieval_method="dense"
        )

        assert after["candidate_depth"] == before["candidate_depth"] - 1
        assert after["hits"] == []

    asyncio.run(exercise())


def test_the_review_file_refuses_a_field_it_does_not_name(project: Path) -> None:
    """The schema is closed, so a hand edit that meant something else fails loudly."""

    async def exercise() -> None:
        service, hits = await _service(project)
        target = hits[0]["chunk_id"]
        path = service.config.chunk_exclusions_path

        _write_exclusion_file(
            project,
            {
                "schema_version": 1,
                "chunks": {
                    target: {
                        "reason": "Reviewed.",
                        "excluded_at": "2026-09-30T19:12:35.104Z",
                        "generation_id": "20261001T000000Z-abcdef",
                        "included": False,
                    }
                },
            },
        )
        with pytest.raises(ResearchError, match="Unsupported chunk exclusion fields"):
            await service.status()

        _write_exclusion_file(
            project,
            {
                "schema_version": 1,
                "chunks": {target: {"reason": "Reviewed."}},
            },
        )
        with pytest.raises(ResearchError, match="requires excluded_at"):
            await service.status()

        _write_exclusion_file(
            project,
            {
                "schema_version": 1,
                "chunks": {
                    target: {
                        "reason": "  ",
                        "excluded_at": "2026-09-30T19:12:35.104Z",
                        "generation_id": "20261001T000000Z-abcdef",
                    }
                },
            },
        )
        with pytest.raises(ResearchError, match="non-empty reason"):
            await service.status()

        _write_exclusion_file(project, {"schema_version": 2, "chunks": {}})
        with pytest.raises(ResearchError, match="Unsupported chunk-exclusion file"):
            await service.status()

        _write_exclusion_file(project, {"schema_version": 1, "chunks": []})
        with pytest.raises(ResearchError, match="Invalid chunk-exclusion mapping"):
            await service.status()

        # A project with no file at all is the ordinary case.
        path.unlink()
        assert load_chunk_exclusions(path) == {}
        assert (await service.status())["excluded_chunk_count"] == 0

    asyncio.run(exercise())


def test_an_entry_edited_by_hand_survives_a_service_write(project: Path) -> None:
    """The service rewrites the same plain JSON, so another reader's entry stays."""

    async def exercise() -> None:
        service, hits = await _service(project, pages=2)
        target, other = (hit["chunk_id"] for hit in hits)
        _write_exclusion_file(
            project,
            {
                "schema_version": 1,
                "chunks": {
                    other: {
                        "reason": "Written by hand.",
                        "excluded_at": "2026-09-30T19:12:35.104Z",
                        "generation_id": str(
                            service._load_current_optional()[1]["generation_id"]
                        ),
                    }
                },
            },
        )

        await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )

        stored = load_chunk_exclusions(service.config.chunk_exclusions_path)
        assert set(stored) == {other, target}
        assert stored[other]["reason"] == "Written by hand."
        assert stored[target]["reason"] == "Reviewed fragment."

    asyncio.run(exercise())


def test_a_chunk_id_is_not_permanent_across_a_rebuild(project: Path) -> None:
    """Unchanged text keeps the id, and the decision with it, for the next build.

    Content that did change gets a new id, and the old entry then names a chunk this
    generation does not hold.
    """

    async def exercise() -> None:
        service, hits = await _service(project)
        target = hits[0]["chunk_id"]
        await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )
        first = str(service._load_current_optional()[1]["generation_id"])

        await service.ingest(chunk_size=60, chunk_overlap=10, force_recompute=True)
        second = str(service._load_current_optional()[1]["generation_id"])
        assert second != first

        assert (await service.search("labour evidence", rerank=False))["hits"] == []
        stored = load_chunk_exclusions(service.config.chunk_exclusions_path)
        assert stored[target]["generation_id"] == first
        listed = await service.list_chunk_exclusions()
        # The generation changed, but the passage is still the same text, so the decision
        # still withholds it and the row says so.
        assert listed["exclusions"][0]["in_current_generation"] is True
        status = await service.status()
        assert status["chunk_exclusion_other_generation_count"] == 1
        assert status["stale"] is False

    asyncio.run(exercise())


def test_the_serialiser_writes_only_the_fields_the_schema_names(
    project: Path,
) -> None:
    """The file stays a hand-editable document, sorted and indented like the rest."""

    path = project / ".research-rag" / "chunk-exclusions.json"
    write_chunk_exclusions(
        path,
        {
            "chk_b": {
                "reason": "Second.",
                "excluded_at": "2026-09-30T19:12:35.104Z",
                "generation_id": "g1",
            },
            "chk_a": {
                "reason": "First.",
                "excluded_at": "2026-09-30T19:12:35.104Z",
                "generation_id": "g1",
            },
        },
    )

    text = path.read_text(encoding="utf-8")
    assert list(load_chunk_exclusions(path)) == ["chk_a", "chk_b"]
    assert json.loads(text) == {
        "schema_version": 1,
        "chunks": {
            "chk_a": {
                "reason": "First.",
                "excluded_at": "2026-09-30T19:12:35.104Z",
                "generation_id": "g1",
            },
            "chk_b": {
                "reason": "Second.",
                "excluded_at": "2026-09-30T19:12:35.104Z",
                "generation_id": "g1",
            },
        },
    }
    assert text.endswith("}\n")


def test_an_exclusion_is_a_decision_about_evidence_and_not_about_quoting(
    project: Path,
) -> None:
    """An exclusion is a decision about whether a passage is evidence, not about what may be
    quoted from it, so `direct_quote_safe` keeps its one meaning and no per-passage field
    is added for it.
    """

    async def exercise() -> None:
        service, hits = await _service(project, pages=2)
        target, neighbour = (hit["chunk_id"] for hit in hits)
        assert {hit["direct_quote_safe"] for hit in hits} == {False}

        await service.set_chunk_inclusion(
            target, included=False, reason="Reviewed fragment."
        )
        context = await service.get_passage(neighbour, context_chunks=1)
        returned = {
            passage["chunk_id"]: passage["direct_quote_safe"]
            for passage in context["context"]
        }
        assert returned[target] is False

    asyncio.run(exercise())


def test_a_decision_named_by_hand_outside_a_generation_is_refused_loudly(
    project: Path,
) -> None:
    """A file whose record is a list rather than a mapping is not read on a guess."""

    path = project / ".research-rag" / "chunk-exclusions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": 1, "chunks": {"": {}}}\n', encoding="utf-8")

    with pytest.raises(StorageError, match="Invalid excluded chunk id"):
        load_chunk_exclusions(path)
