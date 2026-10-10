from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_rag.generations.generation import (
    ReuseSnapshot,
    generation_is_reusable,
    source_set_matches,
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
from research_rag.storage.records import atomic_write_json

_EMBEDDING = resolve_embedding_model("BAAI/bge-small-en-v1.5")
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
    return generation_is_reusable(
        manifest,
        schema_version=SCHEMA_VERSION,
        extraction_policy_version=EXTRACTION_POLICY_VERSION,
        cleaning_policy_version=CLEANING_POLICY_VERSION,
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
        retrieval_policy_fingerprint="retrieval-v1",
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
        retrieval_policy_fingerprint="retrieval-v1",
        metadata_storage_policy="automatic-only-v1",
    )
    assert not source_set_matches(
        snapshot,
        records,
        exclusion_revision="exclusions-v1",
        retrieval_policy_fingerprint="retrieval-v2",
        metadata_storage_policy="automatic-only-v1",
    )
    assert not source_set_matches(
        snapshot,
        [{**records[0], "sha256": "changed-source"}],
        exclusion_revision="exclusions-v1",
        retrieval_policy_fingerprint="retrieval-v1",
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
        retrieval_policy_fingerprint="retrieval-v1",
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
        retrieval_policy_fingerprint="retrieval-v1",
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


def test_generation_without_recorded_repository_still_matches() -> None:
    model = _EMBEDDING
    manifest = _dense_manifest(model)
    del manifest["retrieval"]["dense"]["embedding_model_repository"]

    assert _is_reusable(manifest, model) is True


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
