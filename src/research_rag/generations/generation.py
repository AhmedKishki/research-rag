from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..project.normalized_paths import normalized_relative_path
from ..retrieval.artifact_lookup import (
    LOOKUP_RELATIVE_PATH,
    ArtifactLookupError,
    DocumentArtifactMapping,
    VectorByContentsMapping,
    ensure_artifact_lookup,
    validate_artifact_lookup,
)
from ..retrieval.embeddings import EmbeddingModel
from ..storage.durable_io import StorageError, iter_jsonl

LOGGER = logging.getLogger(__name__)

# The chunker provenances whose passage boundaries this app can reuse. A recorded
# backend outside this set is unknown text provenance and is never reused.
_CHUNK_BACKENDS = frozenset(
    {"UltraRAG token chunker", "Chonkie token chunker (UltraRAG-compatible)"}
)

# The earliest value these version fields have ever carried. A value below either
# is not a record this build wrote, so its layout is unknown.
MIN_READABLE_SCHEMA_VERSION = 1
MIN_READABLE_ARTIFACT_POLICY_VERSION = 1


def generation_is_structurally_readable(
    manifest: dict[str, Any],
    *,
    schema_version: int,
    artifact_policy_version: int,
    project_id: str,
) -> bool:
    """Whether a generation's manifest and artifacts may be read at all.

    A *newer* schema or artifact-policy version names a layout this build cannot
    know, so it is refused. An older version is not refused by its number alone:
    the manifest must carry the fields a reader needs and the artifact records
    must validate, and that structure is what actually proves compatibility. A
    version below the earliest this build wrote, a non-integer version, and a
    foreign project identity are all refused.
    """

    recorded_schema = manifest.get("schema_version")
    recorded_artifact = manifest.get("artifact_policy_version")
    return bool(
        manifest.get("project_id") == project_id
        and isinstance(recorded_schema, int)
        and not isinstance(recorded_schema, bool)
        and MIN_READABLE_SCHEMA_VERSION <= recorded_schema <= schema_version
        and isinstance(recorded_artifact, int)
        and not isinstance(recorded_artifact, bool)
        and MIN_READABLE_ARTIFACT_POLICY_VERSION
        <= recorded_artifact
        <= artifact_policy_version
    )


def units_are_reusable(
    manifest: dict[str, Any],
    *,
    schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    project_id: str,
    maximum_unclean_percent: float = 1.0,
) -> bool:
    """Whether existing extraction units may be reused.

    This axis decides the text a source yields and is independent of both the
    chunker and the embedding model. It covers the extraction and cleaning
    policies and the PDF loss budget. The PDF backend is deliberately absent: it
    decides how a PDF is re-extracted, per source, not whether an EPUB or
    unrelated artifact is reusable.
    """

    passage_cleaning = manifest.get("passage_cleaning")
    recorded_budget = (
        passage_cleaning.get("maximum_unclean_percent", 1.0)
        if isinstance(passage_cleaning, dict)
        else 1.0
    )
    return bool(
        generation_is_structurally_readable(
            manifest,
            schema_version=schema_version,
            artifact_policy_version=artifact_policy_version,
            project_id=project_id,
        )
        and manifest.get("extraction_policy_version") == extraction_policy_version
        and manifest.get("cleaning_policy_version") == cleaning_policy_version
        and recorded_budget == maximum_unclean_percent
    )


def chunks_are_reusable(
    manifest: dict[str, Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_headers: bool,
) -> bool:
    """Whether existing chunks may be reused for this chunker.

    Chunk text is derived from extraction units, so this axis is independent of
    the embedding model but not of the extraction policy: a caller must pair it
    with :func:`units_are_reusable`. It covers the chunker provenance and the
    parameters that decide passage boundaries and the text a vector covers.
    """

    chunking = manifest.get("chunking")
    if not isinstance(chunking, dict):
        return False
    return bool(
        chunking.get("backend") in _CHUNK_BACKENDS
        and chunking.get("tokenizer") == "gpt2"
        and chunking.get("chunk_size") == chunk_size
        and chunking.get("chunk_overlap") == chunk_overlap
        # A generation built before contextual headers existed means no headers,
        # which is also the shipped default.
        and chunking.get("headers", False) == chunk_headers
    )


def vectors_are_reusable(
    manifest: dict[str, Any],
    *,
    embedding: EmbeddingModel,
) -> bool:
    """Whether stored vectors may be reused for this embedding model.

    Vector reuse requires the identical model identity: name, revision,
    dimension, and repository when the generation recorded one. It is independent
    of the extraction and chunking policies, so a processing change that leaves a
    passage's canonical text identical still reuses that passage's vector. The
    exact-text match itself is enforced at reuse time by the embedder.
    """

    retrieval = manifest.get("retrieval")
    dense = retrieval.get("dense") if isinstance(retrieval, dict) else None
    return bool(isinstance(dense, dict) and embedding.matches_dense_metadata(dense))


def _artifact_files(manifest: dict[str, Any]) -> dict[str, PurePosixPath] | None:
    """The artifact paths a reader needs, or None when the manifest omits them.

    Every path is validated with the one path rule this package uses, so a
    hand-edited ``files`` mapping that is absolute, escapes the generation root,
    or is not written as a normalized relative path fails closed instead of being
    joined into a filesystem location. Only the two canonical JSONL artifacts are
    required: a generation with no dense half records no portable vector path, and
    that absence disables the vector axis alone rather than the whole snapshot.
    """

    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    resolved: dict[str, PurePosixPath] = {}
    for key in ("extracted_units", "chunks"):
        relative = normalized_relative_path(files.get(key))
        if relative is None:
            return None
        resolved[key] = relative
    portable = normalized_relative_path(files.get("portable_embeddings"))
    if portable is not None:
        resolved["portable_embeddings"] = portable
    return resolved


def _records_by_source_path(
    manifest: dict[str, Any],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Index one manifest record list by normalized source path, skipping junk."""

    items = manifest.get(key)
    if not isinstance(items, list):
        return {}
    return {
        str(item["source_relative_path"]): item
        for item in items
        if isinstance(item, dict) and item.get("source_relative_path")
    }


def _legacy_artifact_records_are_current(
    chunks_path: Path,
    units_path: Path,
) -> tuple[bool, bool]:
    """Whether canonical records carry the fields the current builder requires.

    A generation recording an older artifact policy cannot be certified by its
    version alone, because the reuse path copies these records verbatim into a new
    generation that activation then validates. Scanning the canonical files once
    here turns a genuinely older record shape into a clean full recompute instead
    of a later build failure. Only the fields the builder and activation require
    are checked.

    Returns ``(units_ok, chunks_ok)``; an unreadable units file fails both,
    because chunk validity is measured against the unit identifiers.
    """

    unit_ids: set[str] = set()
    document_sources: dict[str, str] = {}
    try:
        for unit in iter_jsonl(units_path):
            unit_id = str(unit.get("id") or "")
            document_id = str(unit.get("document_id") or "")
            source_id = str(unit.get("source_id") or "")
            if (
                not unit_id
                or not document_id
                or not source_id
                or not isinstance(unit.get("locator"), dict)
                or not isinstance(unit.get("contents"), str)
                or not str(unit["contents"]).strip()
            ):
                return False, False
            unit_ids.add(unit_id)
            document_sources[document_id] = source_id
    except (OSError, StorageError, ValueError):
        return False, False
    try:
        for chunk in iter_jsonl(chunks_path):
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            document_chunk_index = chunk.get("document_chunk_index")
            if (
                not chunk_id
                or chunk.get("id") != chunk_id
                or not document_id
                or chunk.get("unit_id") not in unit_ids
                or str(chunk.get("source_id") or "")
                != document_sources.get(document_id)
                or not isinstance(chunk.get("locator"), dict)
                or not isinstance(chunk.get("contents"), str)
                or not str(chunk["contents"]).strip()
                or isinstance(document_chunk_index, bool)
                or not isinstance(document_chunk_index, int)
                or document_chunk_index < 0
            ):
                return True, False
    except (OSError, StorageError, ValueError):
        return True, False
    return True, True


@dataclass(frozen=True, slots=True)
class ReuseSnapshot:
    root: Path
    manifest: dict[str, Any]
    source_files: dict[str, dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    units_by_document: Mapping[str, list[dict[str, Any]]]
    chunks_by_document: Mapping[str, list[dict[str, Any]]]
    vectors_by_text: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]]
    # Which independent reuse axes the selected generation satisfied. A snapshot
    # may be present while an axis is false, so a caller decides per axis instead
    # of treating the whole generation as reusable or not.
    units_reusable: bool = False
    chunks_reusable: bool = False
    vectors_reusable: bool = False

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
    maximum_unclean_percent: float = 1.0,
    load_units: bool = True,
    load_chunks: bool = True,
    load_vectors: bool = True,
) -> ReuseSnapshot | None:
    """Load reusable records, or return None when the generation cannot be read.

    Each reuse axis is decided independently. Only a structural incompatibility
    (a future or unrecorded version, a foreign project, or a manifest that does not
    name the artifacts a reader needs) makes the whole generation unreadable, so
    None is returned. Otherwise a snapshot is returned carrying which axes matched.
    A failing units or chunk axis leaves those mappings empty; a missing, corrupt,
    or wrongly shaped vector file disables only the vector axis.
    """

    if current is None:
        return None
    root, manifest = current
    if not generation_is_structurally_readable(
        manifest,
        schema_version=schema_version,
        artifact_policy_version=artifact_policy_version,
        project_id=project_id,
    ):
        LOGGER.debug("generation reuse: structural mismatch; nothing is reusable")
        return None
    files = _artifact_files(manifest)
    if files is None:
        LOGGER.debug("generation reuse: manifest names no readable artifacts")
        return None
    units_policy_ok = units_are_reusable(
        manifest,
        schema_version=schema_version,
        extraction_policy_version=extraction_policy_version,
        cleaning_policy_version=cleaning_policy_version,
        artifact_policy_version=artifact_policy_version,
        project_id=project_id,
        maximum_unclean_percent=maximum_unclean_percent,
    )
    chunks_policy_ok = chunks_are_reusable(
        manifest,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_headers=chunk_headers,
    )
    recorded_artifact = manifest.get("artifact_policy_version")
    if (
        (units_policy_ok or chunks_policy_ok)
        and isinstance(recorded_artifact, int)
        and not isinstance(recorded_artifact, bool)
        and recorded_artifact < artifact_policy_version
    ):
        # An older artifact policy cannot be certified by its number alone, so the
        # canonical records are checked against the fields the current builder
        # requires before either axis is offered for reuse.
        legacy_units_ok, legacy_chunks_ok = _legacy_artifact_records_are_current(
            root / files["chunks"],
            root / files["extracted_units"],
        )
        units_ok = units_policy_ok and legacy_units_ok
        chunks_ok = units_ok and chunks_policy_ok and legacy_chunks_ok
    else:
        units_ok = units_policy_ok
        chunks_ok = units_ok and chunks_policy_ok
    vectors_ok = vectors_are_reusable(manifest, embedding=embedding)
    want_units = load_units and units_ok
    want_chunks = load_chunks and chunks_ok
    want_vectors = load_vectors and vectors_ok
    recorded_chunking = manifest.get("chunking") or {}
    if want_vectors and recorded_chunking.get("headers", False):
        # The lookup keys vectors by canonical passage text, not the exact input
        # including contextual headers. A new headerless chunk must never reuse
        # an old headered vector just because their canonical passages match.
        # Keep the model-axis verdict for no-op validation, but offer no vectors
        # from this mapping without a provable exact embedding-input identity.
        LOGGER.debug(
            "generation reuse: recorded contextual headers prevent canonical-key "
            "vector reuse; unchanged generation remains eligible for a no-op"
        )
        want_vectors = False
    LOGGER.debug(
        "generation reuse: units_reusable=%s chunks_reusable=%s "
        "vectors_reusable=%s load units=%s chunks=%s vectors=%s",
        units_ok,
        chunks_ok,
        vectors_ok,
        want_units,
        want_chunks,
        want_vectors,
    )
    try:
        units_path = root / files["extracted_units"]
        chunks_path = root / files["chunks"]
        lookup = (
            ensure_artifact_lookup(
                chunks_path,
                units_path,
                root / LOOKUP_RELATIVE_PATH,
            )
            if want_units or want_chunks or want_vectors
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
    vectors: Any = None
    if want_vectors:
        # The vector file is a separable axis. A missing path, a truncated, or an
        # otherwise unreadable matrix disables vector reuse alone: the extraction
        # units and chunk text this snapshot also carries stay reusable.
        relative = files.get("portable_embeddings")
        if relative is None:
            LOGGER.debug(
                "generation reuse: no portable vector path; dropping vector reuse only"
            )
            want_vectors = False
        else:
            try:
                vectors = np.load(
                    root / relative,
                    allow_pickle=False,
                    mmap_mode="r",
                )
            except (EOFError, OSError, ValueError):
                vectors = None
            if (
                vectors is None
                or vectors.dtype != np.float32
                or vectors.shape != (chunk_count, embedding.dimension)
            ):
                LOGGER.debug(
                    "generation reuse: vector artifact unusable; "
                    "dropping vector reuse only"
                )
                vectors = None
                want_vectors = False

    source_files = _records_by_source_path(manifest, "source_files")
    documents = _records_by_source_path(manifest, "documents")
    units_by_document: Mapping[str, list[dict[str, Any]]] = (
        DocumentArtifactMapping(lookup, "units")
        if want_units and lookup is not None
        else {}
    )
    chunks_by_document: Mapping[str, list[dict[str, Any]]] = (
        DocumentArtifactMapping(lookup, "chunks")
        if want_chunks and lookup is not None
        else {}
    )
    vectors_by_text: Mapping[
        str,
        np.ndarray[Any, np.dtype[np.float32]],
    ] = (
        VectorByContentsMapping(lookup, vectors)
        if want_vectors and vectors is not None and lookup is not None
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
        units_reusable=units_ok,
        chunks_reusable=chunks_ok,
        vectors_reusable=vectors_ok,
    )


def source_set_matches(
    snapshot: ReuseSnapshot,
    source_records: list[dict[str, Any]],
    *,
    exclusion_revision: str,
    metadata_storage_policy: str,
) -> bool:
    """Compare source bytes and generation-affecting portable state.

    Reviewed metadata does not affect extraction text, chunk identity,
    embeddings, or either retrieval index, so a metadata-only correction must not
    force a new immutable generation. The ranking policy is a runtime decision
    that changes what a search returns, not what a generation stores, so a
    ranking edit never forces a new generation either.
    """

    if snapshot.manifest.get("metadata_storage_policy") != metadata_storage_policy:
        return False
    if snapshot.manifest.get("source_exclusion_revision") != exclusion_revision:
        return False
    files = snapshot.manifest.get("files")
    if not isinstance(files, dict):
        return False
    for field in ("bm25_index", "dense_index"):
        relative = normalized_relative_path(files.get(field))
        if relative is None or not snapshot.root.joinpath(*relative.parts).is_dir():
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
            relative = normalized_relative_path(str(files[field]))
            if relative is None:
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
