"""Compatibility checks and reusable state for immutable generations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .artifact_lookup import (
    LOOKUP_RELATIVE_PATH,
    ArtifactLookupError,
    DocumentArtifactMapping,
    VectorByContentsMapping,
    ensure_artifact_lookup,
    validate_artifact_lookup,
)
from .embeddings import EmbeddingModel
from .storage import StorageError, iter_jsonl


def value_fingerprint(value: Any) -> str:
    """Return a stable fingerprint for one JSON-compatible policy value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_is_reusable(
    manifest: dict[str, Any],
    *,
    schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    project_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_headers: bool,
    embedding: EmbeddingModel,
) -> bool:
    """Require exact processing and model compatibility before any reuse."""

    chunking = manifest.get("chunking", {})
    dense = manifest.get("retrieval", {}).get("dense", {})
    return bool(
        manifest.get("schema_version") == schema_version
        and manifest.get("extraction_policy_version") == extraction_policy_version
        and manifest.get("cleaning_policy_version") == cleaning_policy_version
        and manifest.get("artifact_policy_version") == artifact_policy_version
        and manifest.get("project_id") == project_id
        and chunking.get("backend") == "UltraRAG token chunker"
        and chunking.get("tokenizer") == "gpt2"
        and chunking.get("chunk_size") == chunk_size
        and chunking.get("chunk_overlap") == chunk_overlap
        # A generation built before contextual headers existed means no headers,
        # which is also the shipped default.
        and chunking.get("headers", False) == chunk_headers
        and dense.get("embedding_model") == embedding.name
        and dense.get("embedding_model_revision") == embedding.revision
        and dense.get("embedding_dimension") == embedding.dimension
    )


@dataclass(frozen=True, slots=True)
class ReuseSnapshot:
    """Validated reusable records from the selected generation."""

    root: Path
    manifest: dict[str, Any]
    source_files: dict[str, dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    units_by_document: Mapping[str, list[dict[str, Any]]]
    chunks_by_document: Mapping[str, list[dict[str, Any]]]
    vectors_by_text: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]]

    def vectors_for_texts(
        self,
        texts: Sequence[str],
    ) -> Mapping[str, np.ndarray[Any, np.dtype[np.float32]]]:
        """Resolve an embedding batch without materializing the prior corpus."""

        batch_loader = getattr(self.vectors_by_text, "get_many", None)
        if callable(batch_loader):
            resolved = batch_loader(texts)
        else:
            resolved = {
                text: vector
                for text in dict.fromkeys(texts)
                if (vector := self.vectors_by_text.get(text)) is not None
            }
        return {
            text: vector
            for text, vector in resolved.items()
            if np.isfinite(vector).all()
        }


def load_reuse_snapshot(
    current: tuple[Path, dict[str, Any]] | None,
    *,
    schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    project_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_headers: bool,
    embedding: EmbeddingModel,
    load_units: bool = True,
    load_chunks: bool = True,
    load_vectors: bool = True,
) -> ReuseSnapshot | None:
    """Load reusable records, or return None for any incompatible/corrupt state."""

    if current is None:
        return None
    root, manifest = current
    if not generation_is_reusable(
        manifest,
        schema_version=schema_version,
        extraction_policy_version=extraction_policy_version,
        cleaning_policy_version=cleaning_policy_version,
        artifact_policy_version=artifact_policy_version,
        project_id=project_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_headers=chunk_headers,
        embedding=embedding,
    ):
        return None
    try:
        files = manifest["files"]
        units_path = root / str(files["extracted_units"])
        chunks_path = root / str(files["chunks"])
        lookup = (
            ensure_artifact_lookup(
                chunks_path,
                units_path,
                root / LOOKUP_RELATIVE_PATH,
            )
            if load_units or load_chunks or load_vectors
            else None
        )
        vectors = (
            np.load(
                root / str(files["portable_embeddings"]),
                allow_pickle=False,
                mmap_mode="r",
            )
            if load_vectors
            else None
        )
        chunk_count = lookup.chunk_count() if lookup is not None else 0
    except (
        ArtifactLookupError,
        EOFError,
        KeyError,
        OSError,
        ValueError,
        StorageError,
    ):
        return None
    if load_vectors and (
        vectors is None
        or vectors.dtype != np.float32
        or vectors.shape != (chunk_count, embedding.dimension)
    ):
        return None

    source_files = {
        str(item.get("source_relative_path")): item
        for item in manifest.get("source_files", [])
        if isinstance(item, dict) and item.get("source_relative_path")
    }
    documents = {
        str(item.get("source_relative_path")): item
        for item in manifest.get("documents", [])
        if isinstance(item, dict) and item.get("source_relative_path")
    }
    units_by_document: Mapping[str, list[dict[str, Any]]] = (
        DocumentArtifactMapping(lookup, "units")
        if load_units and lookup is not None
        else {}
    )
    chunks_by_document: Mapping[str, list[dict[str, Any]]] = (
        DocumentArtifactMapping(lookup, "chunks")
        if load_chunks and lookup is not None
        else {}
    )
    vectors_by_text: Mapping[
        str,
        np.ndarray[Any, np.dtype[np.float32]],
    ] = (
        VectorByContentsMapping(lookup, vectors)
        if vectors is not None and lookup is not None
        else {}
    )
    return ReuseSnapshot(
        root=root,
        manifest=manifest,
        source_files=source_files,
        documents=documents,
        units_by_document=units_by_document,
        chunks_by_document=chunks_by_document,
        vectors_by_text=vectors_by_text,
    )


def source_set_matches(
    snapshot: ReuseSnapshot,
    source_records: list[dict[str, Any]],
    *,
    exclusion_revision: str,
    retrieval_policy_fingerprint: str,
    metadata_storage_policy: str,
) -> bool:
    """Compare source bytes and generation-affecting portable state.

    Reviewed metadata is a portable read-time overlay. It does not affect
    extraction text, chunk identity, embeddings, or either retrieval index, so
    a metadata-only correction must not force a new immutable generation.
    """

    if snapshot.manifest.get("metadata_storage_policy") != metadata_storage_policy:
        return False
    if snapshot.manifest.get("source_exclusion_revision") != exclusion_revision:
        return False
    if (
        snapshot.manifest.get("retrieval_policy_fingerprint")
        != retrieval_policy_fingerprint
    ):
        return False
    files = snapshot.manifest.get("files", {})
    for field in ("bm25_index", "dense_index"):
        relative = files.get(field)
        if not isinstance(relative, str) or not relative:
            return False
        if not (snapshot.root / relative).is_dir():
            return False
    expected = {
        str(item["source_relative_path"]): (
            str(item["sha256"]),
            bool(item["included"]),
        )
        for item in source_records
    }
    existing = {
        path: (str(item.get("sha256") or ""), bool(item.get("included")))
        for path, item in snapshot.source_files.items()
    }
    return expected == existing


def generation_artifacts_are_valid(
    root: Path,
    manifest: dict[str, Any],
    *,
    embedding: EmbeddingModel,
) -> bool:
    """Validate immutable portable artifacts without retaining passage text."""

    try:
        files = manifest["files"]
        if not isinstance(files, dict):
            return False
        documents = manifest["documents"]
        source_files = manifest["source_files"]
        if not isinstance(documents, list) or not isinstance(source_files, list):
            return False
        if (
            len(documents) != manifest["document_count"]
            or len(source_files) != manifest["source_file_count"]
        ):
            return False

        source_by_path: dict[str, dict[str, Any]] = {}
        for source in source_files:
            if not isinstance(source, dict):
                return False
            relative = str(source.get("source_relative_path") or "")
            source_id = str(source.get("source_id") or "")
            if not relative or not source_id or relative in source_by_path:
                return False
            source_by_path[relative] = source

        document_sources: dict[str, str] = {}
        for document in documents:
            if not isinstance(document, dict):
                return False
            document_id = str(document.get("document_id") or "")
            source_id = str(document.get("source_id") or "")
            relative = str(document.get("source_relative_path") or "")
            source = source_by_path.get(relative)
            if (
                not document_id
                or not source_id
                or document_id in document_sources
                or source is None
                or source.get("source_id") != source_id
                or source.get("sha256") != document.get("sha256")
            ):
                return False
            document_sources[document_id] = source_id

        def artifact_path(field: str) -> Path:
            value = str(files[field])
            relative = PurePosixPath(value)
            if (
                value in {"", "."}
                or "\\" in value
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != value
            ):
                raise ValueError("Invalid generation artifact path")
            return root.joinpath(*relative.parts)

        extracted_path = artifact_path("extracted_units")
        chunks_path = artifact_path("chunks")
        vectors_path = artifact_path("portable_embeddings")
        index_paths = [
            artifact_path("bm25_index"),
            artifact_path("dense_index"),
        ]
        if (
            any(path.is_symlink() or not path.is_dir() for path in index_paths)
            or extracted_path.is_symlink()
            or chunks_path.is_symlink()
            or vectors_path.is_symlink()
        ):
            return False

        unit_ids: set[str] = set()
        unit_count = 0
        for unit in iter_jsonl(extracted_path):
            unit_id = str(unit.get("id") or "")
            document_id = str(unit.get("document_id") or "")
            if (
                not unit_id
                or unit_id in unit_ids
                or document_id not in document_sources
                or unit.get("source_id") != document_sources[document_id]
            ):
                return False
            unit_ids.add(unit_id)
            unit_count += 1
        if unit_count != manifest["extraction_unit_count"]:
            return False

        chunk_ids: set[str] = set()
        chunk_count = 0
        for chunk in iter_jsonl(chunks_path):
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            if (
                not chunk_id
                or chunk_id in chunk_ids
                or chunk.get("id") != chunk_id
                or document_id not in document_sources
                or chunk.get("source_id") != document_sources[document_id]
                or chunk.get("unit_id") not in unit_ids
                or not isinstance(chunk.get("contents"), str)
                or not str(chunk["contents"]).strip()
            ):
                return False
            chunk_ids.add(chunk_id)
            chunk_count += 1
        if chunk_count != manifest["chunk_count"]:
            return False

        vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
        if vectors.dtype != np.float32 or vectors.shape != (
            chunk_count,
            embedding.dimension,
        ):
            return False
        for offset in range(0, chunk_count, 4096):
            if not np.isfinite(vectors[offset : offset + 4096]).all():
                return False
        lookup_path = root / LOOKUP_RELATIVE_PATH
        return not lookup_path.exists() or validate_artifact_lookup(
            chunks_path,
            extracted_path,
            lookup_path,
            expected_chunk_count=chunk_count,
            expected_unit_count=unit_count,
        )
    except (
        ArtifactLookupError,
        EOFError,
        KeyError,
        OSError,
        StorageError,
        TypeError,
        ValueError,
    ):
        return False
