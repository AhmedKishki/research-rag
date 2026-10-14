from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research_rag.generations.generation import (
    ReuseSnapshot,
    chunks_are_reusable,
    load_reuse_snapshot,
    source_set_matches,
    units_are_reusable,
    vectors_are_reusable,
)
from research_rag.generations.generation_inventory import generation_inventory
from research_rag.project.config import resolve_config
from research_rag.project.support import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    SCHEMA_VERSION,
)
from research_rag.retrieval.embeddings import EmbeddingModel, resolve_embedding_model
from research_rag.storage.durable_io import atomic_write_json, write_jsonl

_EMBEDDING = resolve_embedding_model("BAAI/bge-small-en-v1.5")
_OTHER_EMBEDDING = resolve_embedding_model("BAAI/bge-base-en-v1.5")
_PROJECT_ID = "proj_test"
_CHUNK_SIZE = 100
_CHUNK_OVERLAP = 10


def _dense_manifest(model: EmbeddingModel) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "cleaning_policy_version": CLEANING_POLICY_VERSION,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "project_id": _PROJECT_ID,
        "chunking": {
            "backend": "UltraRAG token chunker",
            "tokenizer": "gpt2",
            "chunk_size": _CHUNK_SIZE,
            "chunk_overlap": _CHUNK_OVERLAP,
            "headers": False,
        },
        "retrieval": {
            "dense": {
                "embedding_model": model.name,
                "embedding_model_revision": model.revision,
                "embedding_dimension": model.dimension,
                "embedding_model_repository": model.repository,
            }
        },
    }


def _is_reusable(manifest: dict[str, Any], model: EmbeddingModel) -> bool:
    return (
        units_are_reusable(
            manifest,
            schema_version=SCHEMA_VERSION,
            extraction_policy_version=EXTRACTION_POLICY_VERSION,
            cleaning_policy_version=CLEANING_POLICY_VERSION,
            artifact_policy_version=ARTIFACT_POLICY_VERSION,
            project_id=_PROJECT_ID,
        )
        and chunks_are_reusable(
            manifest,
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            chunk_headers=False,
        )
        and vectors_are_reusable(manifest, embedding=model)
    )


def _generation_manifest(model: EmbeddingModel) -> dict[str, Any]:
    """A manifest whose artifacts a ``load_reuse_snapshot`` call can read."""

    manifest = _dense_manifest(model)
    manifest["files"] = {
        "extracted_units": "corpus/extracted-units.jsonl",
        "chunks": "chunks/chunks.jsonl",
        "portable_embeddings": "portable/embeddings.npy",
        "bm25_index": "indexes/bm25",
        "dense_index": "indexes/qdrant",
    }
    manifest["documents"] = [
        {
            "source_relative_path": "article.pdf",
            "document_id": "doc_1",
            "source_id": "src_1",
        }
    ]
    manifest["source_files"] = [
        {
            "source_relative_path": "article.pdf",
            "source_id": "src_1",
            "sha256": "digest",
            "included": True,
        }
    ]
    return manifest


def _write_generation(
    root: Path,
    *,
    contents: str = "alpha beta",
    vector_value: float = 1.5,
    units_extra: dict[str, Any] | None = None,
) -> None:
    """Write canonical units, chunks, and a one-row vector matrix under ``root``.

    The records carry the fields the current builder writes, so an older
    artifact-policy version with these records is a structurally readable layout.
    ``units_extra`` overrides unit fields to model an incompatible older record.
    """

    unit = {
        "id": "unit_1",
        "document_id": "doc_1",
        "source_id": "src_1",
        "title": "Article",
        "contents": contents,
        "locator": {},
        "content_kind": "prose",
    }
    unit.update(units_extra or {})
    write_jsonl(root / "corpus" / "extracted-units.jsonl", [unit])
    write_jsonl(
        root / "chunks" / "chunks.jsonl",
        [
            {
                "chunk_id": "chk_1",
                "id": "chk_1",
                "document_id": "doc_1",
                "source_id": "src_1",
                "unit_id": "unit_1",
                "document_chunk_index": 0,
                "contents": contents,
                "locator": {},
                "quality_flags": [],
            }
        ],
    )
    (root / "portable").mkdir(parents=True, exist_ok=True)
    with (root / "portable" / "embeddings.npy").open("wb") as handle:
        np.save(
            handle,
            np.full((1, _EMBEDDING.dimension), vector_value, dtype=np.float32),
            allow_pickle=False,
        )


def _load(
    root: Path,
    manifest: dict[str, Any],
    *,
    model: EmbeddingModel = _EMBEDDING,
    extraction_policy_version: int = EXTRACTION_POLICY_VERSION,
    cleaning_policy_version: int = CLEANING_POLICY_VERSION,
) -> ReuseSnapshot | None:
    return load_reuse_snapshot(
        (root, manifest),
        schema_version=SCHEMA_VERSION,
        extraction_policy_version=extraction_policy_version,
        cleaning_policy_version=cleaning_policy_version,
        artifact_policy_version=ARTIFACT_POLICY_VERSION,
        project_id=_PROJECT_ID,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        chunk_headers=False,
        embedding=model,
    )


@pytest.mark.parametrize(
    "backend", ["UltraRAG token chunker", "Chonkie token chunker (UltraRAG-compatible)"]
)
def test_compatible_direct_and_historical_chunk_provenance_reuse(backend: str) -> None:
    manifest = _dense_manifest(_EMBEDDING)
    manifest["chunking"]["backend"] = backend
    assert _is_reusable(manifest, _EMBEDDING)
    assert manifest["chunking"]["backend"] == backend


def test_unrecognized_chunk_backend_does_not_reuse() -> None:
    manifest = _dense_manifest(_EMBEDDING)
    manifest["chunking"]["backend"] = "different semantics"
    assert not _is_reusable(manifest, _EMBEDDING)


def _snapshot(
    root: Path,
    *,
    metadata_revision: str = "old",
    metadata_storage_policy: str = "automatic-only-v1",
) -> ReuseSnapshot:
    (root / "indexes" / "bm25").mkdir(parents=True)
    (root / "indexes" / "qdrant").mkdir(parents=True)
    return ReuseSnapshot(
        root=root,
        manifest={
            "metadata_revision": metadata_revision,
            "metadata_storage_policy": metadata_storage_policy,
            "source_exclusion_revision": "exclusions-v1",
            "retrieval_policy_fingerprint": "retrieval-v1",
            "files": {
                "bm25_index": "indexes/bm25",
                "dense_index": "indexes/qdrant",
            },
        },
        source_files={
            "article.pdf": {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        },
        documents={},
        units_by_document={},
        chunks_by_document={},
        vectors_by_text={},
    )


def test_source_set_match_ignores_reviewed_metadata_revision(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "generation", metadata_revision="stale-value")

    assert source_set_matches(
        snapshot,
        [
            {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        ],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_source_set_match_ignores_ranking_policy_drift(tmp_path: Path) -> None:
    """A ranking edit changes what a search returns, not what a generation stores."""

    snapshot = _snapshot(tmp_path / "generation")
    snapshot.manifest["retrieval_policy_fingerprint"] = "an-older-ranking-policy"

    assert source_set_matches(
        snapshot,
        [
            {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        ],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_source_set_match_still_rejects_generation_affecting_changes(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path / "generation")
    records = [
        {
            "source_relative_path": "article.pdf",
            "sha256": "source-digest",
            "included": True,
        }
    ]

    assert not source_set_matches(
        snapshot,
        records,
        exclusion_revision="exclusions-v2",
        metadata_storage_policy="automatic-only-v1",
    )
    assert not source_set_matches(
        snapshot,
        [{**records[0], "sha256": "changed-source"}],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_source_set_match_requires_both_index_directories(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "generation")
    (snapshot.root / "indexes" / "qdrant").rmdir()

    assert not source_set_matches(
        snapshot,
        [
            {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        ],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_source_set_match_rejects_legacy_metadata_storage_policy(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "generation",
        metadata_storage_policy="reviewed-metadata-baked-into-generation",
    )

    assert not source_set_matches(
        snapshot,
        [
            {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        ],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_generation_is_reusable_requires_matching_dense_identity() -> None:
    model = _EMBEDDING
    assert _is_reusable(_dense_manifest(model), model) is True

    for field, wrong in (
        ("embedding_model", "BAAI/bge-base-en-v1.5"),
        ("embedding_model_revision", "0" * 40),
        ("embedding_dimension", model.dimension + 1),
        ("embedding_model_repository", "some/other-mirror"),
    ):
        manifest = _dense_manifest(model)
        manifest["retrieval"]["dense"][field] = wrong
        assert _is_reusable(manifest, model) is False, field


def _units_axis(manifest: dict[str, Any]) -> bool:
    return units_are_reusable(
        manifest,
        schema_version=SCHEMA_VERSION,
        extraction_policy_version=EXTRACTION_POLICY_VERSION,
        cleaning_policy_version=CLEANING_POLICY_VERSION,
        artifact_policy_version=ARTIFACT_POLICY_VERSION,
        project_id=_PROJECT_ID,
    )


def _chunks_axis(manifest: dict[str, Any]) -> bool:
    return chunks_are_reusable(
        manifest,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        chunk_headers=False,
    )


def test_units_axis_ignores_the_chunker_parameters() -> None:
    """A header or chunk-size change keeps the extraction units reusable."""

    manifest = _dense_manifest(_EMBEDDING)
    manifest["chunking"]["headers"] = True
    manifest["chunking"]["chunk_size"] = _CHUNK_SIZE + 1

    assert _units_axis(manifest) is True
    assert _chunks_axis(manifest) is False


def test_processing_mismatch_still_reuses_text_identical_vectors(
    tmp_path: Path,
) -> None:
    """Vectors load independently when only the processing policy moved."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["extraction_policy_version"] = EXTRACTION_POLICY_VERSION - 1
    manifest["cleaning_policy_version"] = CLEANING_POLICY_VERSION - 1
    _write_generation(root, contents="alpha beta", vector_value=2.5)

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is False
    assert snapshot.chunks_reusable is False
    assert snapshot.vectors_reusable is True
    # The processing axis is off, so units and chunks stay unloaded.
    assert dict(snapshot.units_by_document) == {}
    assert dict(snapshot.chunks_by_document) == {}
    # The vector axis is on, and the canonical text still keys a valid vector.
    resolved = snapshot.vectors_for_texts(["alpha beta"])
    assert set(resolved) == {"alpha beta"}
    assert np.allclose(resolved["alpha beta"], 2.5)


def test_model_change_reuses_extraction_and_chunks_without_vectors(
    tmp_path: Path,
) -> None:
    """A different embedding model reuses the text and drops only vectors."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_OTHER_EMBEDDING)
    _write_generation(root, contents="alpha beta", vector_value=2.5)

    snapshot = _load(root, manifest, model=_EMBEDDING)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is True
    assert snapshot.vectors_reusable is False
    assert [unit["id"] for unit in snapshot.units_by_document["doc_1"]] == ["unit_1"]
    assert [chunk["contents"] for chunk in snapshot.chunks_by_document["doc_1"]] == [
        "alpha beta"
    ]
    assert dict(snapshot.vectors_by_text) == {}
    assert dict(snapshot.vectors_for_texts(["alpha beta"])) == {}


def test_header_change_reuses_units_and_drops_chunks(tmp_path: Path) -> None:
    """A header-only change reuses units, rebuilds chunks, and drops vectors."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["chunking"]["headers"] = True
    _write_generation(root, contents="alpha beta", vector_value=2.5)

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is False
    assert snapshot.vectors_reusable is True
    assert dict(snapshot.units_by_document) != {}
    # The chunks belong to the old chunker, so none are offered for reuse.
    assert dict(snapshot.chunks_by_document) == {}


def test_structural_mismatch_loads_no_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["artifact_policy_version"] = ARTIFACT_POLICY_VERSION + 1
    _write_generation(root)

    assert _load(root, manifest) is None


def test_future_schema_version_loads_no_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["schema_version"] = SCHEMA_VERSION + 1
    _write_generation(root)

    assert _load(root, manifest) is None


def test_foreign_project_loads_no_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["project_id"] = "someone-else"
    _write_generation(root)

    assert _load(root, manifest) is None


def test_older_schema_and_artifact_versions_stay_readable(tmp_path: Path) -> None:
    """An older version number is not proof of an unreadable layout."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["schema_version"] = SCHEMA_VERSION - 1
    manifest["artifact_policy_version"] = ARTIFACT_POLICY_VERSION - 1
    _write_generation(root, contents="alpha beta", vector_value=2.5)

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is True
    assert snapshot.vectors_reusable is True
    assert [chunk["contents"] for chunk in snapshot.chunks_by_document["doc_1"]] == [
        "alpha beta"
    ]


def test_older_artifact_policy_with_incompatible_records_drops_processing(
    tmp_path: Path,
) -> None:
    """An older artifact layout missing a required field falls back cleanly."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["artifact_policy_version"] = ARTIFACT_POLICY_VERSION - 1
    # The old record carries no source identity, which the current builder writes
    # and activation requires, so the processing axes are not offered.
    _write_generation(
        root,
        contents="alpha beta",
        vector_value=2.5,
        units_extra={"source_id": ""},
    )

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is False
    assert snapshot.chunks_reusable is False
    assert dict(snapshot.units_by_document) == {}
    assert dict(snapshot.chunks_by_document) == {}
    # The vector axis is still keyed by canonical text, so it survives.
    assert snapshot.vectors_reusable is True
    assert dict(snapshot.vectors_for_texts(["alpha beta"])) != {}


@pytest.mark.parametrize("escape", ["../outside.jsonl", "/etc/passwd", "a/../../b"])
def test_escaping_artifact_path_loads_no_snapshot(tmp_path: Path, escape: str) -> None:
    """A recorded artifact path that leaves the generation root fails closed."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["files"]["chunks"] = escape
    _write_generation(root)

    assert _load(root, manifest) is None


def test_escaping_index_path_refuses_source_set_match(tmp_path: Path) -> None:
    """A recorded index path that leaves the generation root fails closed."""

    snapshot = _snapshot(tmp_path / "generation")
    snapshot.manifest["files"]["dense_index"] = "../outside"

    assert not source_set_matches(
        snapshot,
        [
            {
                "source_relative_path": "article.pdf",
                "sha256": "source-digest",
                "included": True,
            }
        ],
        exclusion_revision="exclusions-v1",
        metadata_storage_policy="automatic-only-v1",
    )


def test_older_artifact_policy_missing_unit_locator_drops_processing(
    tmp_path: Path,
) -> None:
    """A legacy unit without a locator cannot seed a current chunk."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["artifact_policy_version"] = ARTIFACT_POLICY_VERSION - 1
    _write_generation(root, units_extra={"locator": "not-a-locator"})

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is False
    assert snapshot.chunks_reusable is False
    assert snapshot.vectors_reusable is True


def test_manifest_without_required_artifacts_loads_no_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["files"].pop("chunks")
    _write_generation(root)

    assert _load(root, manifest) is None


def _vector_path(root: Path) -> Path:
    return root / "portable" / "embeddings.npy"


def _valid_vector_bytes(root: Path) -> bytes:
    return _vector_path(root).read_bytes()


def test_missing_vector_file_keeps_extraction(tmp_path: Path) -> None:
    """A missing vector matrix disables only the vector axis."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    _write_generation(root, contents="alpha beta")
    _vector_path(root).unlink()

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is True
    assert snapshot.chunks_by_document["doc_1"] != []
    assert dict(snapshot.vectors_by_text) == {}


def test_truncated_vector_file_keeps_extraction(tmp_path: Path) -> None:
    """A truncated vector matrix disables only the vector axis."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    _write_generation(root, contents="alpha beta")
    payload = _valid_vector_bytes(root)
    _vector_path(root).write_bytes(payload[: len(payload) // 2])

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is True
    assert snapshot.chunks_by_document["doc_1"] != []
    assert dict(snapshot.vectors_by_text) == {}


def test_corrupt_vector_file_keeps_extraction(tmp_path: Path) -> None:
    """A non-array vector file disables only the vector axis."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    _write_generation(root, contents="alpha beta")
    _vector_path(root).write_bytes(b"this is not a numpy array")

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_by_document["doc_1"] != []
    assert dict(snapshot.vectors_by_text) == {}


@pytest.mark.parametrize("mode", ["dtype", "shape"])
def test_wrong_vector_matrix_keeps_extraction(tmp_path: Path, mode: str) -> None:
    """A vector matrix of the wrong dtype or shape disables only the vector axis."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    _write_generation(root, contents="alpha beta")
    if mode == "dtype":
        array = np.full((1, _EMBEDDING.dimension), 1.5, dtype=np.float64)
    else:
        array = np.full((2, _EMBEDDING.dimension), 1.5, dtype=np.float32)
    with _vector_path(root).open("wb") as handle:
        np.save(handle, array, allow_pickle=False)

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_by_document["doc_1"] != []
    assert dict(snapshot.vectors_by_text) == {}


def test_absent_portable_vector_path_keeps_extraction(tmp_path: Path) -> None:
    """A generation with no recorded vector path disables only the vector axis."""

    root = tmp_path / "generation"
    manifest = _generation_manifest(_EMBEDDING)
    manifest["files"].pop("portable_embeddings")
    _write_generation(root, contents="alpha beta")

    snapshot = _load(root, manifest)

    assert snapshot is not None
    assert snapshot.units_reusable is True
    assert snapshot.chunks_reusable is True
    assert snapshot.chunks_by_document["doc_1"] != []
    assert dict(snapshot.vectors_by_text) == {}


def test_inventory_reports_recorded_configuration_not_active_settings(
    project: Path,
) -> None:
    config = resolve_config(project)
    generation_id = "20260101T000000Z-aabbccdd"
    root = config.generations_root / generation_id
    root.mkdir(parents=True)
    manifest = _dense_manifest(_EMBEDDING)
    manifest.update(
        generation_id=generation_id,
        retrieval_policy_fingerprint="historical-retrieval-policy",
        documents=[{"source_relative_path": "private-source.pdf"}],
    )
    manifest["retrieval"].update(
        bm25={"language": "german"},
        reranker={"model": "historical-reranker", "model_revision": "old-revision"},
        fusion={"rrf_k": 71, "bm25_weight": 0.25, "dense_weight": 0.75},
    )
    path = root / "manifest.json"
    atomic_write_json(path, manifest)
    before = path.read_bytes()

    inventory = generation_inventory(config, generation_id)

    record = inventory["generations"][0]
    assert record["is_current"] is True
    for key in (
        "chunking",
        "retrieval",
        "retrieval_policy_fingerprint",
        "extraction_policy_version",
        "cleaning_policy_version",
        "artifact_policy_version",
    ):
        assert record[key] == manifest[key]
    assert record["retrieval"]["bm25"]["language"] == "german"
    assert record["retrieval"]["reranker"]["model"] == "historical-reranker"
    assert "documents" not in record
    assert path.read_bytes() == before
    assert not config.current_path.exists()


@pytest.mark.parametrize("block", [None, "not-an-object", [], 42])
def test_inventory_does_not_invent_missing_or_malformed_configuration(
    project: Path, block: Any
) -> None:
    config = resolve_config(project)
    root = config.generations_root / "20260101T000000Z-aabbccdd"
    root.mkdir(parents=True)
    manifest = {} if block is None else {"chunking": block, "retrieval": block}
    atomic_write_json(root / "manifest.json", manifest)

    record = generation_inventory(config, None)["generations"][0]

    assert record["chunking"] is None
    assert record["retrieval"] is None
    assert record["retrieval_policy_fingerprint"] is None
    assert "manifest_error" not in record


@pytest.mark.parametrize("content", [None, "{not-json", "[]", "null"])
def test_inventory_marks_unreadable_manifest_configuration_unknown(
    project: Path, content: str | None
) -> None:
    config = resolve_config(project)
    root = config.generations_root / "20260101T000000Z-aabbccdd"
    root.mkdir(parents=True)
    if content is not None:
        (root / "manifest.json").write_text(content, encoding="utf-8")

    record = generation_inventory(config, None)["generations"][0]

    assert record["manifest_error"]
    for key in (
        "chunking",
        "retrieval",
        "retrieval_policy_fingerprint",
        "extraction_policy_version",
        "cleaning_policy_version",
        "artifact_policy_version",
    ):
        assert record[key] is None
