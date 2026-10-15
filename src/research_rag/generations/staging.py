"""What a build derives before it becomes a generation.

The identity a staged build resumes from, the inventory of its sources, and
the chunk records made from their extracted units.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..corpus.sources import SourceScan, sha256_file
from ..corpus.text_normalization import normalize_inline_text, normalize_reading_text
from ..corpus.text_quality import (
    has_searchable_alphanumeric_content,
    text_corruption_reasons,
)
from ..project.embeddings import EmbeddingModel
from ..project.policy import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    DEFAULT_PDF_BACKEND_FINGERPRINT,
    EXTRACTION_POLICY_VERSION,
    INGESTION_IDENTITY_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
    ResearchError,
    value_fingerprint,
)
from ..retrieval.artifact_lookup import chunk_text
from ..retrieval.dense import DenseTokenAuditUnavailable
from ..storage.durable_io import file_identity


def _generation_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _source_work_key(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def _pdf_batch_count(page_count: int, batch_size: int) -> int:
    if page_count < 0 or batch_size <= 0:
        raise ValueError("PDF page and batch counts must be valid")
    return (page_count + batch_size - 1) // batch_size


def _hash_with_stable_stat(path: Path) -> tuple[str, dict[str, int]]:
    before = file_identity(path)
    digest = sha256_file(path)
    after = file_identity(path)
    if before != after:
        raise OSError("Source changed while it was being hashed")
    return digest, after


def _source_inventory(scan: SourceScan) -> list[dict[str, Any]]:
    return [
        {
            "source_id": source.source_id,
            "source_path": source.project_relative_path,
            "source_relative_path": source.source_relative_path,
            "format": source.extension.removeprefix("."),
            "size": source.size,
            "mtime_ns": source.mtime_ns,
        }
        for source in scan.selected
    ]


def _checkpoint_identity(
    *,
    project_id: str,
    inventory: list[dict[str, Any]],
    exclusion_revision: str,
    baseline_generation_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
    chunk_headers: bool,
    force_recompute: bool,
    embedding: EmbeddingModel,
    maximum_unclean_percent: float = 1.0,
    pdf_backend_fingerprint: str = DEFAULT_PDF_BACKEND_FINGERPRINT,
    selection: dict[str, Any] | None = None,
) -> str:
    """Fingerprint the inputs a staged build may resume from.

    A build of named sources, or one that takes sources out, carries its
    selection; a build of every source carries none, so its identity is the one
    a build checkpointed before selections existed computed.

    The ranking policy is deliberately absent: it decides how a generation is
    *searched*, not what its artifacts contain, so a ranking edit reuses every
    chunk and vector instead of rebuilding the corpus. The policy still reaches the
    manifest at publish time. Contextual chunk headers are not a ranking policy,
    because they decide the text a vector covers.
    """

    return value_fingerprint(
        {
            **({"selection": selection} if selection is not None else {}),
            "project_id": project_id,
            "inventory": inventory,
            "exclusion_revision": exclusion_revision,
            "baseline_generation_id": baseline_generation_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_headers": chunk_headers,
            "maximum_unclean_percent": maximum_unclean_percent,
            "pdf_backend_fingerprint": pdf_backend_fingerprint,
            "force_recompute": force_recompute,
            "generation_schema_version": SCHEMA_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "cleaning_policy_version": CLEANING_POLICY_VERSION,
            "artifact_policy_version": ARTIFACT_POLICY_VERSION,
            "embedding_model": embedding.name,
            "embedding_model_revision": embedding.revision,
            "embedding_dimension": embedding.dimension,
            "ingestion_identity_policy_version": (INGESTION_IDENTITY_POLICY_VERSION),
            "metadata_storage_policy": METADATA_STORAGE_POLICY,
        }
    )


def _embedding_text(chunk: dict[str, Any]) -> str:
    """Return the text a chunk is embedded from, contextual header included.

    A header is prepended to what the dense half embeds and never to what a search
    returns, so a returned passage stays quotable. A generation built without
    headers has no separate embedding text.
    """

    return str(chunk.get("embedding_text") or chunk_text(chunk))


def _chunk_header(document: dict[str, Any], locator: dict[str, Any]) -> str:
    """Return the context line prepended to a chunk's embedding text.

    The parts are what a passage cannot say about itself. A PDF locator carries a
    page rather than a section, so a PDF chunk is headed by its title alone.
    """
    title = normalize_inline_text(str(document.get("title") or ""))
    section = normalize_inline_text(str(locator.get("section_title") or ""))
    parts = [part for part in (title, section) if part]
    if len(parts) == 2 and parts[0].casefold() == parts[1].casefold():
        parts = parts[:1]
    return " — ".join(parts)


def _enrich_chunks(
    raw_chunks: list[dict[str, Any]],
    units: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    headers: bool = False,
) -> tuple[list[dict[str, Any]], int, int, int]:
    units_by_id = {str(item["id"]): item for item in units}
    documents_by_id = {str(item["document_id"]): item for item in documents}
    document_ordinals: defaultdict[str, int] = defaultdict(int)
    enriched: list[dict[str, Any]] = []
    discarded_empty_chunks = 0
    discarded_symbol_only_chunks = 0
    discarded_corrupt_chunks = 0

    for raw in raw_chunks:
        unit_id = str(raw.get("doc_id") or "")
        unit = units_by_id.get(unit_id)
        if unit is None:
            raise ResearchError(
                f"Chunker returned an unknown extraction unit: {unit_id}"
            )
        document_id = str(unit["document_id"])
        document = documents_by_id[document_id]
        text = normalize_reading_text(str(raw.get("contents") or ""))
        if not text:
            discarded_empty_chunks += 1
            continue
        if not has_searchable_alphanumeric_content(text):
            discarded_symbol_only_chunks += 1
            continue
        if text_corruption_reasons(text):
            discarded_corrupt_chunks += 1
            continue

        ordinal = document_ordinals[document_id]
        document_ordinals[document_id] += 1
        identity = f"{unit_id}\0{ordinal}\0{text}".encode()
        chunk_id = f"chk_{hashlib.sha256(identity).hexdigest()[:24]}"
        locator = dict(unit["locator"])
        record: dict[str, Any] = {
            "id": chunk_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_id": document["source_id"],
            "document_chunk_index": ordinal,
            "unit_id": unit_id,
            "locator": locator,
            "contents": text,
            "content_kind": str(unit.get("content_kind") or "prose"),
            "annotations": list(unit.get("annotations") or []),
            "quality_flags": list(unit.get("quality_flags") or []),
        }
        if headers and (header := _chunk_header(document, locator)):
            record["embedding_text"] = f"{header}\n\n{text}"
        enriched.append(record)

    represented = {item["document_id"] for item in enriched}
    missing = [
        item["source_path"]
        for item in documents
        if item["document_id"] not in represented
    ]
    if missing:
        raise ResearchError(
            "No searchable chunks were produced for: " + ", ".join(missing)
        )
    return (
        enriched,
        discarded_empty_chunks,
        discarded_symbol_only_chunks,
        discarded_corrupt_chunks,
    )


def _record_embedding_token_counts(
    chunks: list[dict[str, Any]],
    count_tokens: Any,
    *,
    maximum_tokens: int,
) -> bool:
    """Record each built chunk's embedding token count and truncation flag.

    FastEmbed silently truncates input past the embedding model's limit, so a
    chunk's dense vector covers only a prefix while BM25 indexes the whole text.
    False means the tokenizer could not be inspected, so the fields stay absent.
    """

    if not chunks:
        return True
    texts = [_embedding_text(chunk) for chunk in chunks]
    try:
        counts = count_tokens(texts)
    except DenseTokenAuditUnavailable:
        return False
    if len(counts) != len(chunks):
        raise ResearchError(
            "The embedding tokenizer returned a different count than chunks"
        )
    for chunk, count in zip(chunks, counts, strict=True):
        chunk["embedding_token_count"] = int(count)
        chunk["dense_truncated"] = int(count) > maximum_tokens
    return True
