from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from research_rag.corpus.text_quality import (
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    CHUNK_FLAG_NON_EVIDENCE,
    chunk_health_flags,
)
from research_rag.retrieval.artifact_lookup import (
    LOOKUP_HEALTH_FLAGS_KEY,
    ArtifactLookup,
    ArtifactLookupError,
    DocumentArtifactMapping,
    VectorByContentsMapping,
    build_artifact_lookup,
    ensure_artifact_lookup,
    validate_artifact_lookup,
)
from research_rag.storage.records import write_jsonl


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


def test_mixed_parent_preserves_claim_children_and_navigation_tables(
    tmp_path: Path,
) -> None:
    references = "\n".join(
        f"Writer, {initial}. ({year}). River accounts. Ledger, pp. 10-20."
        for initial, year in [("A", 2020), ("B", 2021), ("C", 2022)]
    )
    claim = "The estimates exclude unpaid labour."
    navigation_table = "First category 3\nSecond category 17\nThird category 42"
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    lookup_path = tmp_path / "artifact-lookup.sqlite3"
    write_jsonl(
        units_path,
        [
            {
                "id": "mixed",
                "document_id": "doc",
                "contents": references + "\n" + claim,
            },
            {
                "id": "heading",
                "document_id": "doc",
                "contents": "Contents",
                "locator": {"type": "pdf_page", "page": 1},
            },
            {
                "id": "table",
                "document_id": "doc",
                "contents": navigation_table,
                "content_kind": "table",
                "locator": {"type": "pdf_page", "page": 1},
            },
        ],
    )
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "claim",
                "document_id": "doc",
                "unit_id": "mixed",
                "document_chunk_index": 0,
                "contents": claim,
            },
            {
                "chunk_id": "references",
                "document_id": "doc",
                "unit_id": "mixed",
                "document_chunk_index": 1,
                "contents": references,
            },
            {
                "chunk_id": "table",
                "document_id": "doc",
                "unit_id": "table",
                "document_chunk_index": 2,
                "contents": navigation_table,
                "content_kind": "table",
            },
        ],
    )
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    records = lookup.chunks_by_ids(["claim", "references", "table"])
    assert records["claim"][LOOKUP_HEALTH_FLAGS_KEY] == 0
    assert records["table"][LOOKUP_HEALTH_FLAGS_KEY] == 0
    assert records["references"][LOOKUP_HEALTH_FLAGS_KEY] & CHUNK_FLAG_NON_EVIDENCE


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


def test_legacy_lookup_recomputes_structural_unit_context_without_artifact_changes(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    units_path = tmp_path / "units.jsonl"
    lookup_path = tmp_path / "artifact-lookup.sqlite3"
    entries = "Labour and value .... 12\nCapital and machinery .... 34\nEcology and society .... 56"
    prose = (
        "Labour and value are contested categories. Smith (2020, p. 12) argues "
        "that machinery changes the organisation of work rather than removing "
        "the need for labour. This note qualifies the argument with evidence."
    )
    units = [
        {"id": "heading", "contents": "Contents", "page": 1},
        {"id": "entries", "contents": entries, "page": 1},
        {"id": "prose", "contents": prose, "page": 2},
    ]
    for unit in units:
        unit["document_id"] = "document-a"
        unit["locator"] = {"type": "pdf_page", "page": unit.pop("page")}
    write_jsonl(units_path, units)
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": f"chunk-{index}",
                "document_id": "document-a",
                "unit_id": unit_id,
                "document_chunk_index": index,
                "contents": text,
                "quality_flags": [],
            }
            for index, (unit_id, text) in enumerate(
                [("entries", "Labour and value"), ("prose", prose)]
            )
        ],
    )
    canonical = (chunks_path.read_bytes(), units_path.read_bytes())
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    with sqlite3.connect(lookup_path) as connection:
        connection.execute("PRAGMA user_version = 3")
        connection.execute("UPDATE chunks SET health_flags = 0")
    lookup = ensure_artifact_lookup(chunks_path, units_path, lookup_path)
    records = lookup.chunks_by_ids(["chunk-0", "chunk-1"])
    assert records["chunk-0"][LOOKUP_HEALTH_FLAGS_KEY] & CHUNK_FLAG_NON_EVIDENCE
    assert records["chunk-1"][LOOKUP_HEALTH_FLAGS_KEY] == 0
    assert lookup.ordinal_for_contents("Labour and value") == 0
    assert lookup.ordinal_for_contents(prose) == 1
    assert (chunks_path.read_bytes(), units_path.read_bytes()) == canonical
    assert prose.encode() not in lookup_path.read_bytes()
