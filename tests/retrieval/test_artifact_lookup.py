from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from research_rag.corpus.text_quality import (
    CHUNK_FLAG_APPARATUS,
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    CHUNK_FLAGS_UNREADABLE,
    chunk_health_flags,
)
from research_rag.retrieval.artifact_lookup import (
    LOOKUP_APPARATUS_CONTEXT_KEY,
    LOOKUP_HEALTH_FLAGS_KEY,
    LOOKUP_SCHEMA_VERSION,
    ArtifactLookup,
    ArtifactLookupError,
    DocumentArtifactMapping,
    VectorByContentsMapping,
    build_artifact_lookup,
    ensure_artifact_lookup,
    validate_artifact_lookup,
)
from research_rag.retrieval.search import _candidate_flags
from research_rag.storage.durable_io import write_jsonl


def _artifacts(root: Path) -> tuple[Path, Path, Path]:
    chunks_path = root / "chunks.jsonl"
    units_path = root / "units.jsonl"
    lookup_path = root / "artifact-lookup.sqlite3"
    write_jsonl(
        chunks_path,
        [
            {
                "id": "chunk-1",
                "chunk_id": "chunk-1",
                "document_id": "document-a",
                "source_id": "source-a",
                "unit_id": "unit-a",
                "document_chunk_index": 0,
                "contents": "Café evidence with an em dash — and 漢字.",
            },
            {
                "id": "chunk-2",
                "chunk_id": "chunk-2",
                "document_id": "document-a",
                "source_id": "source-a",
                "unit_id": "unit-a",
                "document_chunk_index": 1,
                "contents": "Repeated exact passage.",
            },
            {
                "id": "chunk-3",
                "chunk_id": "chunk-3",
                "document_id": "document-b",
                "source_id": "source-b",
                "unit_id": "unit-b",
                "document_chunk_index": 0,
                "contents": "Repeated exact passage.",
            },
        ],
    )
    write_jsonl(
        units_path,
        [
            {
                "id": "unit-a",
                "document_id": "document-a",
                "source_id": "source-a",
                "contents": "Large unit text A that stays outside SQLite.",
            },
            {
                "id": "unit-b",
                "document_id": "document-b",
                "source_id": "source-b",
                "contents": "Large unit text B that stays outside SQLite.",
            },
        ],
    )
    return chunks_path, units_path, lookup_path


def test_lookup_reads_utf8_offsets_duplicates_documents_and_vectors(
    tmp_path: Path,
) -> None:
    chunks_path, units_path, lookup_path = _artifacts(tmp_path)
    build_artifact_lookup(chunks_path, units_path, lookup_path)
    lookup = ArtifactLookup(chunks_path, units_path, lookup_path)

    assert lookup.chunk_count() == 3
    assert lookup.chunk_count({"document-a"}) == 2
    assert lookup.chunk_count(set()) == 0
    assert lookup.chunks_by_ids(["chunk-1"])["chunk-1"]["contents"].startswith("Café")
    assert [
        item["chunk_id"]
        for item in lookup.chunks_by_contents(["Repeated exact passage."])[
            "Repeated exact passage."
        ]
    ] == ["chunk-2", "chunk-3"]
    assert [item["chunk_id"] for item in lookup.chunks_for_document("document-a")] == [
        "chunk-1",
        "chunk-2",
    ]
    assert lookup.units_for_document("document-b")[0]["id"] == "unit-b"

    chunks_by_document = DocumentArtifactMapping(lookup, "chunks")
    assert list(chunks_by_document) == ["document-a", "document-b"]
    assert chunks_by_document["document-b"][0]["chunk_id"] == "chunk-3"
    canonical_chunks = [
        record
        for record in map(json.loads, chunks_path.read_text().splitlines())
        if record["document_id"] == "document-a"
    ]
    assert chunks_by_document["document-a"] == canonical_chunks

    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    reusable = VectorByContentsMapping(lookup, vectors)
    resolved = reusable.get_many(
        ["Repeated exact passage.", "Café evidence with an em dash — and 漢字."]
    )
    assert resolved["Repeated exact passage."].tolist() == vectors[1].tolist()
    assert resolved["Café evidence with an em dash — and 漢字."].tolist() == (
        vectors[0].tolist()
    )

    lookup_bytes = lookup_path.read_bytes()
    assert b"Repeated exact passage" not in lookup_bytes
    assert b"Large unit text" not in lookup_bytes
    assert validate_artifact_lookup(
        chunks_path,
        units_path,
        lookup_path,
        expected_chunk_count=3,
        expected_unit_count=2,
    )


def test_content_digest_candidates_still_require_exact_text(tmp_path: Path) -> None:
    chunks_path, units_path, lookup_path = _artifacts(tmp_path)
    build_artifact_lookup(chunks_path, units_path, lookup_path)
    target = "Repeated exact passage."
    with sqlite3.connect(lookup_path) as connection:
        connection.execute(
            "UPDATE chunks SET content_sha256 = ? WHERE chunk_id = ?",
            (hashlib.sha256(target.encode("utf-8")).digest(), "chunk-1"),
        )

    lookup = ArtifactLookup(chunks_path, units_path, lookup_path)
    assert [
        item["chunk_id"] for item in lookup.chunks_by_contents([target])[target]
    ] == ["chunk-2", "chunk-3"]


def test_lookup_stores_the_chunk_rejection_verdict(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    lookup_path = tmp_path / "artifact-lookup.sqlite3"
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "chunk-healthy",
                "document_id": "document-a",
                "document_chunk_index": 0,
                "contents": "Cobalt heron evidence from the amber marsh.",
            },
            {
                "chunk_id": "chunk-symbol-only",
                "document_id": "document-a",
                "document_chunk_index": 1,
                "contents": "\u2014 \u2022 \u220e",
            },
            {
                "chunk_id": "chunk-flagged",
                "document_id": "document-a",
                "document_chunk_index": 2,
                "contents": "Amber marsh evidence.",
                "quality_flags": ["extraction_artifact"],
            },
        ],
    )
    write_jsonl(
        units_path,
        [
            {
                "id": "unit-a",
                "document_id": "document-a",
                "contents": "Cobalt heron evidence.",
            }
        ],
    )

    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    records = lookup.chunks_by_ids(
        ["chunk-healthy", "chunk-symbol-only", "chunk-flagged"]
    )

    assert records["chunk-healthy"][LOOKUP_HEALTH_FLAGS_KEY] == 0
    assert (
        records["chunk-symbol-only"][LOOKUP_HEALTH_FLAGS_KEY]
        & CHUNK_FLAG_EXTRACTION_ARTIFACT
    )
    assert (
        records["chunk-flagged"][LOOKUP_HEALTH_FLAGS_KEY]
        & CHUNK_FLAG_EXTRACTION_ARTIFACT
    )
    # The verdict never appears in a canonical record or a public projection.
    assert "health_flags" not in json.loads(
        chunks_path.read_text(encoding="utf-8").splitlines()[0]
    )


def test_lookup_verdict_matches_a_query_time_scan(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    lookup_path = tmp_path / "artifact-lookup.sqlite3"
    samples = [
        "Cobalt heron evidence from the amber marsh.",
        "\u2014 \u2022 \u220e",
        "Amber marsh evidence.",
        "",
        "\ufffd evidence with a replacement character",
    ]
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": f"chunk-{index}",
                "document_id": "document-a",
                "document_chunk_index": index,
                "contents": text,
            }
            for index, text in enumerate(samples)
            if text
        ],
    )
    write_jsonl(
        units_path,
        [
            {
                "id": "unit-a",
                "document_id": "document-a",
                "contents": "Cobalt heron evidence.",
            }
        ],
    )

    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)

    for record in lookup.chunks_by_ids(
        [f"chunk-{index}" for index, text in enumerate(samples) if text]
    ).values():
        stored = record[LOOKUP_HEALTH_FLAGS_KEY]
        recomputed = chunk_health_flags(record["contents"])
        assert stored == recomputed


def test_ensure_rebuilds_missing_corrupt_and_stale_lookup(tmp_path: Path) -> None:
    chunks_path, units_path, lookup_path = _artifacts(tmp_path)
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    assert lookup.chunk_count() == 3

    lookup_path.write_bytes(b"not sqlite")
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    assert lookup.chunk_count() == 3

    chunks = list(lookup.chunks_by_ids(["chunk-1", "chunk-2"]).values())
    write_jsonl(chunks_path, chunks)
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    assert lookup.chunk_count() == 2
    assert validate_artifact_lookup(
        chunks_path,
        units_path,
        lookup_path,
        expected_chunk_count=2,
        expected_unit_count=2,
    )


def test_contextual_verdicts_refresh_without_changing_canonical_evidence(
    tmp_path: Path,
) -> None:
    samples = [
        ("Emergence", "section_header", None, True),
        ("Emergence is irreducible", "section_header", None, False),
        ("Index", "section_header", None, False),
        ("emergence 97–8", "text", None, True),
        ("emergence 17,142,144", "text", None, True),
        ("Growth, 17, 142, 144", "table", None, False),
        ("Growth, 17, 142, 144", "unknown_label", None, False),
        ("Conclusion", "section_header", None, False),
        ("emergence 97–8", "text", None, False),
        # A recorded EPUB section title is also positive Index context.
        ("emergence 97–8", None, "Index of Subjects", True),
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    lookup_path = tmp_path / "artifact-lookup.sqlite3"
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": f"c{i}",
                "unit_id": f"u{i}",
                "document_id": "doc",
                "document_chunk_index": i,
                "contents": text,
            }
            for i, (text, _label, _section, _flagged) in enumerate(samples)
        ],
    )
    write_jsonl(
        units_path,
        [
            {
                "id": f"u{i}",
                "document_id": "doc",
                "contents": text,
                **({"docling_label": label} if label is not None else {}),
                **(
                    {"locator": {"section_title": section}}
                    if section is not None
                    else {}
                ),
            }
            for i, (text, label, section, _flagged) in enumerate(samples)
        ],
    )
    canonical = (chunks_path.read_bytes(), units_path.read_bytes())
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    # Simulate the old classifier's cached zero verdicts, not a serving-schema change.
    with sqlite3.connect(lookup_path) as connection:
        connection.execute("UPDATE chunks SET health_flags = 0")
        connection.execute(f"PRAGMA user_version = {LOOKUP_SCHEMA_VERSION - 1}")
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    records = lookup.chunks_by_ids([f"c{i}" for i in range(len(samples))])
    for i, (_text, _label, _section, flagged) in enumerate(samples):
        record = records[f"c{i}"]
        stored = record.pop(LOOKUP_HEALTH_FLAGS_KEY)
        assert bool(stored & CHUNK_FLAG_APPARATUS) is flagged
        assert not stored & CHUNK_FLAGS_UNREADABLE
        assert _candidate_flags(record) == stored
        assert (
            chunk_health_flags(
                record["contents"],
                apparatus_context=record[LOOKUP_APPARATUS_CONTEXT_KEY],
            )
            == stored
        )
    # A heading remains a readable neighbour even though opt-in search excludes it.
    assert [item["chunk_id"] for item in lookup.passage_window("doc", 1, radius=1)] == [
        "c0",
        "c1",
        "c2",
    ]
    assert lookup.apparatus_chunk_count() == 4
    assert canonical == (chunks_path.read_bytes(), units_path.read_bytes())


def test_lookup_rejects_duplicate_document_positions_and_unsafe_artifacts(
    tmp_path: Path,
) -> None:
    chunks_path, units_path, lookup_path = _artifacts(tmp_path)
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    chunks = lookup.chunks_for_document("document-a")
    chunks[1]["document_chunk_index"] = 0
    write_jsonl(chunks_path, chunks)
    with pytest.raises(ArtifactLookupError, match="document position"):
        build_artifact_lookup(chunks_path, units_path, lookup_path)

    chunks_path.unlink()
    chunks_path.symlink_to(units_path)
    with pytest.raises(ArtifactLookupError, match="missing or unsafe"):
        build_artifact_lookup(chunks_path, units_path, lookup_path)


def test_a_passage_window_skips_unretrievable_neighbours(tmp_path: Path) -> None:
    """`get_passage` reads its neighbours from the index, not the whole document.

    A neighbour whose stored verdict carries a flag is passed over, so the window
    still reaches the nearest retrievable passage on each side.
    """

    texts = [
        "Opening evidence.",
        "First argument.",
        ". . . . . .",
        "The target passage.",
        "Closing argument.",
        "Late evidence.",
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": f"c{index}",
                "document_id": "document-a",
                "document_chunk_index": index,
                "contents": text,
            }
            for index, text in enumerate(texts)
        ],
    )
    write_jsonl(units_path, [{"id": "unit-a", "document_id": "document-a"}])
    lookup = ensure_artifact_lookup(
        chunks_path, units_path, tmp_path / "artifact-lookup.sqlite3"
    )

    window = lookup.passage_window("document-a", 3, radius=1)
    assert [item["chunk_id"] for item in window] == ["c1", "c3", "c4"]
    assert [
        item["chunk_id"] for item in lookup.passage_window("document-a", 0, radius=2)
    ] == [
        "c0",
        "c1",
        "c3",
    ]


def test_a_lookup_connection_is_closed_when_its_block_ends(tmp_path: Path) -> None:
    """A sqlite3 connection's own context manager ends a transaction, not the file."""

    chunks_path, units_path, lookup_path = _artifacts(tmp_path)
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)

    with lookup._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
