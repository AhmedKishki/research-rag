"""Human-edited review state: the plain JSON files stay authoritative."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchError, ResearchService
from research_rag.project.config import resolve_config
from research_rag.storage.records import load_metadata_overrides
from tests.conftest import write_pdf, write_reviewed_metadata
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


def _write_metadata_file(project: Path, sources: dict[str, dict[str, object]]) -> Path:
    """Write the reviewed-metadata file the way a person would: by hand."""

    path = project / ".research-rag" / "source-metadata.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


async def _service(project: Path) -> tuple[ResearchService, str]:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence for the hand-edited review state."],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)
    return service, "article.pdf"


def test_hand_written_metadata_file_is_honoured(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(
            project,
            {
                relative: {
                    "title": "Hand-edited title",
                    "authors": ["Hand Editor", "Second Author"],
                    "year": 2025,
                    "doi": "10.1000/hand",
                    "categories": ["marxism", "waste studies"],
                    "keywords": ["fetishism", "use value"],
                    "project": ["ai-and-fetishism"],
                }
            },
        )

        listed = await service.list_sources()
        source = listed["sources"][0]
        assert source["title"] == "Hand-edited title"
        assert source["authors"] == ["Hand Editor", "Second Author"]
        assert source["year"] == 2025
        assert source["doi"] == "10.1000/hand"
        assert source["categories"] == ["marxism", "waste studies"]
        assert source["keywords"] == ["fetishism", "use value"]
        assert source["project"] == ["ai-and-fetishism"]

        status = await service.status()
        assert (status["categories"], status["projects"]) == (
            [
                {"category": "marxism", "searchable_source_count": 1},
                {"category": "waste studies", "searchable_source_count": 1},
            ],
            [{"project": "ai-and-fetishism", "searchable_source_count": 1}],
        )
        assert status["metadata_overlay_active"] is True

        filtered = await service.search(
            "cobalt evidence",
            top_k=5,
            rerank=False,
            projects_any=["ai-and-fetishism"],
            categories_any=["marxism"],
            keywords=["use value"],
        )
        assert filtered["filters"]["projects_any"] == ["ai-and-fetishism"]
        assert len(filtered["hits"]) == 1
        assert filtered["hits"][0]["title"] == "Hand-edited title"
        assert filtered["hits"][0]["project"] == ["ai-and-fetishism"]

    asyncio.run(exercise())


def test_a_chunk_exclusion_written_by_hand_is_honoured_and_refused_loudly(
    project: Path,
) -> None:
    """A decision a reader makes without this app is the decision this app would have written.

    A mistake in it is named rather than read as intent.
    """

    async def exercise() -> None:
        service, _name = await _service(project)
        found = await service.search("cobalt evidence", top_k=10, rerank=False)
        target = found["hits"][0]["chunk_id"]
        generation_id = json.loads(
            (project / ".research-rag" / "runtime" / "current.json").read_text(
                encoding="utf-8"
            )
        )["generation_id"]
        path = project / ".research-rag" / "chunk-exclusions.json"

        entry = {
            "reason": "Table header, not a passage.",
            "excluded_at": "2026-09-30T19:12:35.104Z",
            "generation_id": generation_id,
        }
        path.write_text(
            json.dumps({"schema_version": 1, "chunks": {target: entry}}, indent=2)
            + "\n",
            encoding="utf-8",
        )

        # The filter enforces the decision, so the passage is gone from the next search the
        # moment the file says so, with no rebuild.
        result = await service.search("cobalt evidence", top_k=10, rerank=False)
        assert result["hits"] == []
        assert result["excluded_chunk_count"] == 1

        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "chunks": {target: {**entry, "included": False}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ResearchError, match="Unsupported chunk exclusion fields"):
            await service.status()

    asyncio.run(exercise())


def test_hand_written_metadata_typo_fails_loudly(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(project, {relative: {"titel": "Misspelled field"}})

        with pytest.raises(ResearchError, match="Unsupported metadata fields: titel"):
            await service.status()

    asyncio.run(exercise())


def test_hand_written_metadata_rejects_a_bad_value_or_path(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(project, {relative: {"year": "2025"}})
        with pytest.raises(ResearchError, match="'year' must be null or an integer"):
            await service.status()

        _write_metadata_file(project, {"../escape.pdf": {"title": "Outside"}})
        with pytest.raises(ResearchError, match="normalized and relative"):
            await service.status()

        (project / ".research-rag" / "source-metadata.json").write_text(
            json.dumps({"schema_version": 2, "sources": {}}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ResearchError, match="Unsupported metadata file"):
            await service.status()

    asyncio.run(exercise())


def test_a_service_write_applies_without_re_ingesting(project: Path) -> None:
    """The review is read-time state, so saving it needs no rebuild."""

    async def exercise() -> None:
        service, relative = await _service(project)

        saved = await service.set_source_metadata(
            {"title": "Saved title", "authors": ["Saver"], "year": 2026},
            source_path=relative,
        )

        assert saved["status"] == "changed"
        assert saved["effective_immediately"] is True
        assert saved["source_file_changed"] is False
        assert saved["generation_rebuild_recommended"] is False
        listed = await service.list_sources()
        source = listed["sources"][0]
        assert source["title"] == "Saved title"
        assert source["authors"] == ["Saver"]
        assert source["year"] == 2026

    asyncio.run(exercise())


def test_a_service_write_keeps_an_entry_edited_by_hand(project: Path) -> None:
    """A hand edit to another source survives a service write: same file, no clobber."""

    async def exercise() -> None:
        service, relative = await _service(project)
        write_pdf(
            project / "sources" / "hand.pdf",
            ["A second source, reviewed by hand."],
        )
        write_reviewed_metadata(
            service.config,
            "hand.pdf",
            {"title": "Hand title", "keywords": ["by hand"]},
        )

        await service.set_source_metadata(
            {"title": "Service title"}, source_path=relative
        )

        overrides = load_metadata_overrides(service.config.metadata_path)
        assert overrides["hand.pdf"] == {"title": "Hand title", "keywords": ["by hand"]}
        assert overrides[relative] == {"title": "Service title"}

    asyncio.run(exercise())


def test_an_empty_review_restores_automatic_metadata(project: Path) -> None:
    """Clearing every field removes the entry, so extraction metadata applies again."""

    async def exercise() -> None:
        service, relative = await _service(project)
        await service.set_source_metadata(
            {"title": "Reviewed title"}, source_path=relative
        )

        cleared = await service.set_source_metadata(
            {"title": "", "authors": [], "year": None}, source_path=relative
        )

        assert cleared["status"] == "changed"
        assert cleared["metadata"] == {}
        assert cleared["effective_immediately"] is False
        assert load_metadata_overrides(service.config.metadata_path) == {}
        listed = await service.list_sources()
        assert listed["sources"][0]["title"] == "Test PDF"

    asyncio.run(exercise())


def test_an_unknown_metadata_field_is_refused(project: Path) -> None:
    """The reviewed schema is closed, so a typo fails loudly instead of persisting."""

    async def exercise() -> None:
        service, relative = await _service(project)
        with pytest.raises(ResearchError, match="Unsupported metadata fields"):
            await service.set_source_metadata({"titel": "typo"}, source_path=relative)
        assert load_metadata_overrides(service.config.metadata_path) == {}

    asyncio.run(exercise())


def test_metadata_written_by_a_later_version_is_refused_by_cause(project: Path) -> None:
    """A file whose writer understood a field this build does not is refused before any entry
    is read, so the answer says which version is needed instead of listing fields the
    caller never typed.
    """

    path = project / ".research-rag" / "source-metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "written_with": {"metadata_fields": ["title", "language", "subject"]},
                "sources": {"article.pdf": {"title": "Test PDF"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    from research_rag.storage.records import StorageError

    with pytest.raises(StorageError) as raised:
        load_metadata_overrides(path)

    message = str(raised.value)
    assert "subject" in message
    assert "Update the server" in message
    # The known field must not be named.
    assert "title" not in message


def test_metadata_written_before_the_record_existed_still_loads(project: Path) -> None:
    """Existing projects carry no writer record and must keep working."""

    (project / ".research-rag").mkdir(parents=True, exist_ok=True)
    path = _write_metadata_file(project, {"article.pdf": {"title": "Test PDF"}})

    assert load_metadata_overrides(path) == {"article.pdf": {"title": "Test PDF"}}


def test_a_writer_record_of_the_known_fields_is_accepted(project: Path) -> None:
    """The common case must not raise, or the record would break every project."""

    async def exercise() -> None:
        service, relative = await _service(project)
        await service.set_source_metadata({"title": "Kept"}, source_path=relative)

        written = json.loads(service.config.metadata_path.read_text(encoding="utf-8"))
        assert "language" in written["written_with"]["metadata_fields"]
        assert (
            load_metadata_overrides(service.config.metadata_path)["article.pdf"][
                "title"
            ]
            == "Kept"
        )

    asyncio.run(exercise())
