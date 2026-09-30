from __future__ import annotations

from pathlib import Path

from research_rag.generation import ReuseSnapshot, source_set_matches


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
