"""Pure helpers shared by every layer.

Imports nothing from the service, the MCP surface, or an entry point, so it stays
a leaf.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_lookup import (
    LOOKUP_HEALTH_FLAGS_KEY,
)
from .dense import (
    DenseTokenAuditUnavailable,
)
from .embeddings import EmbeddingModel
from .generation import (
    value_fingerprint,
)
from .rerankers import resolve_reranker_model
from .settings import EffectiveSettings
from .sources import (
    SourceScan,
    normalize_metadata,
    sha256_file,
    stable_source_id,
)
from .text_normalization import (
    normalize_inline_text,
    normalize_reading_text,
)
from .text_quality import (
    EXTRACTION_ARTIFACT_TOKEN,
    chunk_health_flags,
    has_searchable_alphanumeric_content,
    text_corruption_reasons,
    text_health_reasons,
    text_script_notes,
)


def retrieval_policy_fingerprint(settings: EffectiveSettings) -> str:
    """Return the identity of the ranking policy these settings describe.

    The fusion constants and gates decide what a search returns, so they are part
    of what a generation *is*, and a generation recording a different policy is
    not reusable.

    A value stays out when it only reorders what those already return: the
    source-diversity penalty is taken at the final top_k pick and changes no stored
    artifact, so fingerprinting it would report a byte-identical rebuild.
    """

    return value_fingerprint(
        {
            "default_method": DEFAULT_RETRIEVAL_METHOD,
            "available_methods": sorted(RETRIEVAL_METHODS),
            "bm25": {
                "language": settings.bm25_stopwords_language,
                "tokenizer": "default",
            },
            "fusion": {
                "method": "weighted_reciprocal_rank_fusion",
                "rrf_k": settings.rrf_k,
                "bm25_weight": settings.bm25_weight,
                "dense_weight": settings.dense_weight,
                "minimum_candidates": settings.minimum_candidates,
                "maximum_candidates": settings.maximum_candidates,
            },
            "relevance_gates": {
                "bm25_requires_query_token_overlap": True,
                "dense_minimum_cosine_similarity": (
                    settings.dense_minimum_cosine_similarity
                ),
            },
        }
    )


def _selection_relevance(
    ordered_ids: Sequence[str],
    scores: Mapping[str, float],
) -> dict[str, float]:
    """Normalize the score an order was built from onto ``0.0..1.0``.

    The best-scored candidate is 1.0 and the worst is 0.0, so the diversity
    penalty is a share of this ranking's confidence rather than of the pool's
    depth. An unscored candidate -- the unranked tail after a reranked window --
    is 0.0, and a flat score band leaves every scored candidate at 1.0.
    """

    relevance = dict.fromkeys(ordered_ids, 0.0)
    scored = [scores[chunk_id] for chunk_id in ordered_ids if chunk_id in scores]
    if not scored:
        return relevance
    low = min(scored)
    high = max(scored)
    if high <= low:
        # A flat score band says nothing about relative order, so every scored
        # candidate counts as equally relevant and the order itself stands.
        for chunk_id in ordered_ids:
            if chunk_id in scores:
                relevance[chunk_id] = 1.0
        return relevance
    span = high - low
    for chunk_id in ordered_ids:
        if chunk_id in scores:
            relevance[chunk_id] = (scores[chunk_id] - low) / span
    return relevance


def _passage_equality_key(text: str) -> str:
    """Return the words of a passage, so two copies of it read the same.

    Two copies differ only in case, punctuation, and spacing, which changes no
    meaning.
    """

    return " ".join(_WORD.findall(str(text).casefold()))


def _source_diverse_selection(
    ordered_ids: Sequence[str],
    *,
    source_id_by_chunk: Mapping[str, str],
    scores: Mapping[str, float],
    top_k: int,
    penalty: float,
) -> list[str]:
    """Pick ``top_k`` candidates, charging a source for each of its repeats.

    Greedy maximal-marginal-relevance over a fused order: a candidate's adjusted
    score is its relevance minus ``penalty`` per candidate already taken from the
    same source, with fused position then chunk ID settling ties.

    A penalty at or below zero, a pool no deeper than the request, or a pool with
    no scored candidate returns the plain slice, because an unreranked ranking
    offers no relevance to charge against, so the penalty alone would make it a
    round-robin over sources.
    """

    if penalty <= 0.0 or len(ordered_ids) <= top_k or not scores:
        return list(ordered_ids[:top_k])
    relevance = _selection_relevance(ordered_ids, scores)
    base_rank = {chunk_id: index for index, chunk_id in enumerate(ordered_ids)}
    remaining = list(ordered_ids)
    selected: list[str] = []
    repeats: Counter[str] = Counter()

    def ordering_key(chunk_id: str) -> tuple[float, int, str]:
        source = source_id_by_chunk.get(chunk_id, chunk_id)
        adjusted = relevance.get(chunk_id, 0.0) - penalty * repeats[source]
        return (-adjusted, base_rank[chunk_id], chunk_id)

    while remaining and len(selected) < top_k:
        chosen = min(remaining, key=ordering_key)
        remaining.remove(chosen)
        selected.append(chosen)
        repeats[source_id_by_chunk.get(chosen, chosen)] += 1
    return selected


async def _atomic_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Let a write-side thread finish its atomic unit before propagating cancel."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generation_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _source_work_key(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def _pdf_batch_count(page_count: int, batch_size: int) -> int:
    if page_count < 0 or batch_size <= 0:
        raise ValueError("PDF page and batch counts must be valid")
    return (page_count + batch_size - 1) // batch_size


def _source_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _hash_with_stable_stat(path: Path) -> tuple[str, dict[str, int]]:
    before = _source_stat_identity(path)
    digest = sha256_file(path)
    after = _source_stat_identity(path)
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
) -> str:
    """Fingerprint the inputs a staged build may resume from.

    The ranking policy is deliberately absent: it decides how a generation is
    *searched*, not what its artifacts contain, so a ranking edit reuses every
    chunk and vector instead of rebuilding the corpus. The policy still reaches the
    manifest at publish time. Contextual chunk headers are not a ranking policy,
    because they decide the text a vector covers.
    """

    return value_fingerprint(
        {
            "project_id": project_id,
            "inventory": inventory,
            "exclusion_revision": exclusion_revision,
            "baseline_generation_id": baseline_generation_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_headers": chunk_headers,
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


def _locator_place(locator: Mapping[str, Any]) -> str:
    """Where a passage sits in the original, as the fragment a citation ends with.

    A reader who is told only this is reading a position, not a claim about the
    work, so the printed page label is carried only where it differs from the
    physical page and a section is named rather than numbered when it has a name.
    """

    if locator.get("type") == "pdf_page":
        return f"p. {locator.get('page_label') or locator.get('page')}"
    section = (
        locator.get("href_with_fragment")
        or locator.get("section_title")
        or locator.get("href")
    )
    if section:
        return f"section {section}"
    return f"EPUB section {locator.get('section_index')}"


def _citation(document: dict[str, Any], locator: dict[str, Any]) -> str:
    authors = document.get("authors") or []
    creator = "; ".join(str(item) for item in authors) if authors else ""
    title = str(document.get("title") or document.get("source_path") or "Source")
    year = document.get("year")
    lead = creator or title
    if creator and title:
        lead = f"{creator}, {title}"
    if year:
        lead = f"{lead} ({year})"
    doi = normalize_inline_text(str(document.get("doi") or ""))
    if doi:
        lead = f"{lead}, doi:{doi.removeprefix('doi:')}"

    return f"{lead}, {_locator_place(locator)}"


def _content_tokens(value: str, stopwords: frozenset[str]) -> set[str]:
    """Return meaningful Unicode word tokens used for lexical abstention.

    ``stopwords`` is the gate's resolved function-word set, which stops at least
    what the index stops.
    """

    return {
        token
        for match in _WORD.finditer(value.casefold())
        if (token := match.group(0)) not in stopwords
    }


def document_frequencies(
    texts: Iterable[str],
    stopwords: frozenset[str],
) -> Counter[str]:
    """Count, for each content token, how many of these texts contain it.

    Distinct tokens per text, so a repeated word does not look common. A feedback
    rule weights against this table and it costs a full corpus pass, so build it
    once.
    """

    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(_content_tokens(text, stopwords))
    return frequencies


def _inverse_document_frequency(
    frequencies: Mapping[str, int] | None,
    corpus_size: int,
    term: str,
) -> float:
    """Return a term's rarity weight, smoothed so a ubiquitous term stays positive.

    Without a table the weight is 1, so a caller that has none gets the
    unweighted ranking.
    """

    if not frequencies or corpus_size <= 0:
        return 1.0
    return math.log((corpus_size + 1) / (frequencies.get(term, 0) + 1)) + 1.0


def _pseudo_relevance_terms(
    *,
    query: str,
    texts: list[str],
    maximum_terms: int,
    stopwords: frozenset[str],
    document_frequencies: Mapping[str, int] | None = None,
    corpus_size: int = 0,
) -> list[str]:
    """Mine expansion terms from the first-pass lexical leaders.

    A candidate is scored by leader support times rarity, because support alone
    mines the words that appear everywhere and discriminate nothing.

    Ordered by score then term, so one query expands the same way on every run.
    Terms the query already holds are skipped: they would change nothing and hide
    whether the expansion did anything.
    """

    if maximum_terms <= 0:
        return []
    query_tokens = _content_tokens(query, stopwords)
    support: Counter[str] = Counter()
    for text in texts:
        support.update(_content_tokens(text, stopwords) - query_tokens)

    def rank(item: tuple[str, int]) -> tuple[float, int, str]:
        term, count = item
        weight = _inverse_document_frequency(document_frequencies, corpus_size, term)
        return (-count * weight, -count, term)

    return [term for term, _count in sorted(support.items(), key=rank)[:maximum_terms]]


def _normalized_filter(values: list[str] | None) -> set[str]:
    return {
        normalized.casefold()
        for value in values or []
        if (normalized := normalize_inline_text(str(value)))
    }


def _requested_ids(values: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _normalized_scalar(value: Any) -> set[str]:
    """Normalize one scalar metadata value, or nothing when it is empty.

    `title` is a string where `authors` and the other list fields are lists, and
    iterating a string yields its characters.
    """

    text = normalize_inline_text(str(value or ""))
    return {text.casefold()} if text else set()


def _matches_name_filter(supplied: set[str], values: set[str]) -> bool:
    """Whether one of the supplied phrases appears inside one of these values.

    Titles and author names are phrases, not controlled tags, so a substring is the
    right comparison. Both sides arrive casefolded.
    """

    if not supplied:
        return True
    return any(
        any(name in document_value for document_value in values) for name in supplied
    )


def _document_matches_metadata(
    document: dict[str, Any],
    *,
    keywords: set[str],
    categories_any: set[str] = frozenset(),
    projects_any: set[str] = frozenset(),
    languages_any: set[str] = frozenset(),
    authors_any: set[str] = frozenset(),
    titles_any: set[str] = frozenset(),
) -> bool:
    """Match one document against the reviewed-metadata filter layers.

    `keywords` is all-of; `project`, `categories`, `language` and `authors` are
    any-of. The project layer is a passthrough inside a one-project server and
    matters when a corpus is copied or shared.

    `title` and `authors` are names rather than tags, so they match by
    case-insensitive substring. Reviewed values override extracted ones, so these
    filters read the effective document the query path already holds.
    """

    document_categories = _normalized_filter(document.get("categories"))
    document_keywords = _normalized_filter(document.get("keywords"))
    document_projects = _normalized_filter(document.get("project"))
    document_languages = _normalized_filter(document.get("language"))
    document_authors = _normalized_filter(document.get("authors"))
    document_titles = _normalized_scalar(document.get("title"))
    return (
        keywords.issubset(document_keywords)
        and (not categories_any or not categories_any.isdisjoint(document_categories))
        and (not projects_any or not projects_any.isdisjoint(document_projects))
        and (not languages_any or not languages_any.isdisjoint(document_languages))
        and _matches_name_filter(authors_any, document_authors)
        and _matches_name_filter(titles_any, document_titles)
    )


def _metadata_inventory(
    documents: dict[str, dict[str, Any]],
    excluded_document_ids: set[str],
    *,
    field: str,
    label: str,
) -> list[dict[str, Any]]:
    """Count searchable sources per reviewed value of one list-valued field.

    Reviewed values are free strings, so this is the inventory an agent uses before
    searching. A source counts once per value it carries, and reviewed exclusions
    are not counted.
    """

    display: dict[str, str] = {}
    counts: dict[str, int] = {}
    for document_id, document in documents.items():
        if document_id in excluded_document_ids:
            continue
        normalized = _normalized_filter(document.get(field))
        if not normalized:
            continue
        for raw in document.get(field) or []:
            value = normalize_inline_text(str(raw))
            if value and value.casefold() in normalized:
                display.setdefault(value.casefold(), value)
        for name in normalized:
            counts[name] = counts.get(name, 0) + 1
    return [
        {label: display.get(name, name), "searchable_source_count": counts[name]}
        for name in sorted(counts)
    ]


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata without extraction-related line wrapping."""

    result = dict(document)
    # These fields existed in older immutable generations and are internal
    # implementation details, not reviewed metadata or useful diagnostics.
    result.pop("metadata_confidence", None)
    result.pop("metadata_override_revision", None)
    for field in ("title", "doi"):
        result[field] = normalize_inline_text(str(result.get(field) or ""))
    for field in ("authors", "categories", "keywords", "language", "project"):
        result[field] = [
            normalized
            for value in result.get(field) or []
            if (normalized := normalize_inline_text(str(value)))
        ]
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = list(result.get("metadata_warnings") or [])
    if provenance.get("title") != "reviewed_override" and text_health_reasons(
        result["title"]
    ):
        result["title"] = Path(str(result.get("source_path") or "source")).stem
        warnings.append("corrupt_extracted_title")
    if provenance.get("authors") != "reviewed_override":
        clean_authors = [
            author for author in result["authors"] if not text_health_reasons(author)
        ]
        if len(clean_authors) != len(result["authors"]):
            warnings.append("corrupt_extracted_authors")
        result["authors"] = clean_authors
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    return result


def _reranker_revision(model: str) -> str:
    return resolve_reranker_model(model)[1]


def _canonical_metadata_override(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize one override while preserving explicit empty reviewed values."""

    normalized = normalize_metadata(value)
    result: dict[str, Any] = {}
    for field in ("title", "doi"):
        if field in normalized:
            result[field] = normalize_inline_text(str(normalized[field]))
    for field in ("authors", "categories", "keywords", "language", "project"):
        if field not in normalized:
            continue
        items: list[str] = []
        seen: set[str] = set()
        for raw in normalized[field]:
            item = normalize_inline_text(str(raw))
            key = item.casefold()
            if item and key not in seen:
                items.append(item)
                seen.add(key)
        result[field] = items
    if "year" in normalized:
        result["year"] = normalized["year"]
    return result


def _metadata_snapshot_changed(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> bool:
    stored_revision = manifest.get("metadata_revision")
    if stored_revision is None:
        return bool(metadata)
    return stored_revision != value_fingerprint(metadata)


def _effective_document_metadata(
    document: dict[str, Any],
    override: dict[str, Any],
    *,
    unknown_legacy_snapshot_mismatch: bool = False,
) -> dict[str, Any]:
    """Overlay current reviewed metadata without mutating generation artifacts.

    Generation documents retain the metadata snapshot used while building their
    indexes; the portable reviewed-metadata file is authoritative at read time.

    An older generation does not retain the automatic value a reviewed override hid,
    so removing one uses a deterministic fallback.
    """

    normalized_override = _canonical_metadata_override(override)
    override_revision = value_fingerprint(normalized_override)
    stored_override_revision = document.get("metadata_override_revision")
    if stored_override_revision == override_revision:
        result = dict(document)
        result.pop("metadata_confidence", None)
        return result

    result = dict(document)
    result.pop("metadata_confidence", None)
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = [
        str(item)
        for item in result.get("metadata_warnings") or []
        if item
        not in {
            "authors_missing",
            "title_from_filename",
            "automatic_metadata_unavailable_after_override_removal",
        }
    ]
    removed_reviewed_value = False
    unknown_legacy_snapshot = (
        unknown_legacy_snapshot_mismatch
        and not provenance
        and stored_override_revision != override_revision
    )

    fallbacks: dict[str, Any] = {
        "title": Path(str(result.get("source_path") or "source")).stem,
        "authors": [],
        "year": None,
        "doi": "",
        "language": [],
    }
    for field in ("title", "authors", "year", "doi", "language"):
        if field in normalized_override:
            result[field] = normalized_override[field]
            provenance[field] = "reviewed_override"
        elif provenance.get(field) == "reviewed_override" or unknown_legacy_snapshot:
            result[field] = fallbacks[field]
            provenance[field] = "filename" if field == "title" else "missing"
            removed_reviewed_value = True

    # Categories, keywords, and the project tag have no automatic extraction
    # source, so the current reviewed lists are represented exactly even on
    # old generations.
    for field in ("categories", "keywords", "project"):
        result[field] = list(normalized_override.get(field, []))
        provenance[field] = (
            "reviewed_override" if field in normalized_override else "missing"
        )

    if not result.get("authors"):
        warnings.append("authors_missing")
    if provenance.get("title") == "filename":
        warnings.append("title_from_filename")
    if removed_reviewed_value:
        warnings.append("automatic_metadata_unavailable_after_override_removal")

    result["metadata_provenance"] = provenance
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    result["metadata_override_revision"] = override_revision
    return result


def _effective_documents(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    legacy_snapshot_mismatch = manifest.get(
        "metadata_storage_policy"
    ) != METADATA_STORAGE_POLICY and _metadata_snapshot_changed(manifest, metadata)
    project_id = str(manifest.get("project_id") or "")
    result: dict[str, dict[str, Any]] = {}
    for stored in manifest.get("documents", []):
        relative = str(stored.get("source_relative_path") or "")
        document = _effective_document_metadata(
            stored,
            metadata.get(relative, {}),
            unknown_legacy_snapshot_mismatch=legacy_snapshot_mismatch,
        )
        if not document.get("source_id") and project_id and relative:
            document["source_id"] = stable_source_id(project_id, relative)
        result[str(document["document_id"])] = document
    return result


def _document_for_chunk(
    chunk: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(chunk.get("document_id") or "")
    document = documents_by_id.get(document_id)
    if document is None:
        raise ResearchError(
            f"Current generation chunk references unknown document: {document_id}"
        )
    return document


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Read canonical passage text with compatibility for older generations."""

    return str(
        chunk.get("contents") or chunk.get("text") or chunk.get("embedding_text") or ""
    )


# The chunker counts a chunk in GPT-2 tokens and the generation records that
# name, so a length rule stated as a fraction of the chunk size and a token count
# are one unit. The counter is imported and loaded on first use, so a query
# that applies no token floor never pays for the dependency.
TOKENIZER_REPOSITORIES = {"gpt2": "openai-community/gpt2"}

_TOKENIZERS: dict[str, Any] = {}


def passage_token_count(text: str, tokenizer: str) -> int:
    """Count a passage in the tokens the generation was chunked by."""

    counter = _TOKENIZERS.get(tokenizer)
    if counter is None:
        repository = TOKENIZER_REPOSITORIES.get(tokenizer)
        if repository is None:
            raise ResearchError(
                "The current generation records an unknown chunker tokenizer: "
                f"{tokenizer!r}. Known names: "
                f"{', '.join(sorted(TOKENIZER_REPOSITORIES))}."
            )
        try:
            from tokie import Tokenizer
        except ImportError as exc:  # pragma: no cover - a packaging failure
            raise ResearchError(
                f"The {tokenizer} tokenizer is unavailable: {exc}"
            ) from exc
        try:
            counter = Tokenizer.from_pretrained(repository)
        except Exception as exc:
            raise ResearchError(
                f"The {tokenizer} tokenizer could not be loaded: {exc}"
            ) from exc
        _TOKENIZERS[tokenizer] = counter
    return int(counter.count_tokens(text))


def _embedding_text(chunk: dict[str, Any]) -> str:
    """Return the text a chunk is embedded from, contextual header included.

    A header is prepended to what the dense half embeds and never to what a search
    returns, so a returned passage stays quotable. A generation built without
    headers has no separate embedding text.
    """

    return str(chunk.get("embedding_text") or _chunk_text(chunk))


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


def _public_passage(
    chunk: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    public_document = _public_document(document)
    locator = dict(chunk.get("locator") or {})
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_id": public_document["source_id"],
        "source_path": public_document["source_path"],
        "source_relative_path": public_document.get("source_relative_path"),
        "title": public_document["title"],
        "authors": public_document["authors"],
        "year": public_document.get("year"),
        "doi": public_document["doi"],
        "language": public_document["language"],
        "categories": public_document["categories"],
        "keywords": public_document["keywords"],
        "project": public_document["project"],
        "locator": locator,
        "citation": normalize_inline_text(_citation(public_document, locator)),
        "text": normalize_reading_text(_chunk_text(chunk)),
        "text_fidelity": "cleaned_semantic_text",
        "direct_quote_safe": False,
        "text_notes": text_script_notes(_chunk_text(chunk)),
        "embedding_token_count": chunk.get("embedding_token_count"),
        "dense_truncated": chunk.get("dense_truncated"),
        "content_kind": str(chunk.get("content_kind") or "prose"),
        "annotations": list(chunk.get("annotations") or []),
        "quality_flags": list(chunk.get("quality_flags") or []),
        "metadata_provenance": dict(public_document.get("metadata_provenance") or {}),
        "metadata_warnings": list(public_document.get("metadata_warnings") or []),
    }


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
                f"UltraRAG returned an unknown extraction unit: {unit_id}"
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


def _is_extraction_artifact(chunk: dict[str, Any]) -> bool:
    quality_flags = {str(item) for item in chunk.get("quality_flags", [])}
    return EXTRACTION_ARTIFACT_TOKEN in quality_flags or not (
        has_searchable_alphanumeric_content(_chunk_text(chunk))
    )


def _candidate_flags(chunk: dict[str, Any]) -> int:
    """Return the stored retrieval-rejection verdict for one candidate.

    The artifact lookup computes this when it is built, so a query does not rescan
    chunk text. A lookup predating it falls back to computing the same flags here.
    """

    stored = chunk.get(LOOKUP_HEALTH_FLAGS_KEY)
    if isinstance(stored, int):
        return stored
    return chunk_health_flags(
        _chunk_text(chunk),
        quality_flags=chunk.get("quality_flags"),
    )


def _record_withheld(
    withheld: dict[str, dict[str, Any]],
    chunk: dict[str, Any],
    reasons: Sequence[str],
    *,
    limit: int,
) -> None:
    for reason in reasons:
        entry = withheld.setdefault(reason, {"count": 0, "example_chunk_ids": []})
        entry["count"] = int(entry["count"]) + 1
        examples = entry["example_chunk_ids"]
        if len(examples) < limit:
            examples.append(str(chunk["chunk_id"]))


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


DEFAULT_RETRIEVAL_METHOD = "hybrid"

RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})

SCHEMA_VERSION = 5

EXTRACTION_POLICY_VERSION = 7

CLEANING_POLICY_VERSION = 3

ARTIFACT_POLICY_VERSION = 3

# The ranking policy is not part of this identity: it never affected artifacts,
# and fingerprinting it made a ranking edit discard a build in progress.
INGESTION_IDENTITY_POLICY_VERSION = 3

METADATA_STORAGE_POLICY = "automatic_only_runtime_overlay_v1"

# The shape the builder writes for a build identifier, and the shape nothing
# else may supply. A generation directory is named by a caller in `generations`,
# `use_generation`, and `remove_generation`, so those three check the name against
# this rather than joining it to a path: a name that does not look like a build
# identifier never reaches the filesystem as one.
GENERATION_ID_PATTERN = r"\d{8}T\d{6}Z-[0-9a-f]{8}"

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class ResearchError(RuntimeError):
    """User-facing research workflow failure."""
