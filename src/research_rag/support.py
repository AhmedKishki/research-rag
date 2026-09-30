"""Pure helpers: values, identities, fingerprints, and the shared exceptions.

Nothing here imports the service, the MCP surface, or an entry point, so this
module is a leaf. It exists so the parts of `service.py` that only compute a
value can be read and tested without the orchestration around them.
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
from .extraction import (
    chunk_health_flags,
    has_searchable_alphanumeric_content,
    normalize_inline_text,
    normalize_reading_text,
    text_corruption_reasons,
    text_health_reasons,
    text_script_notes,
)
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


def retrieval_policy_fingerprint(settings: EffectiveSettings) -> str:
    """Return the identity of the ranking policy these settings describe.

    The fusion constants and the relevance gates decide what a search returns,
    so their values are part of what a generation *is*: a generation that
    recorded a different policy than this process runs is not reusable, and the
    next ingestion builds a new generation instead of mixing two.

    A value stays out when it only reorders what those already return. The
    source-diversity penalty is taken at the final top_k pick over candidates
    fusion and reranking ranked first, so it changes no stored artifact, and
    fingerprinting it would report every existing generation as needing a
    rebuild that reproduces the same index byte for byte.
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

    The best-scored candidate is 1.0 and the worst-scored is 0.0, so the
    diversity penalty is a share of this ranking's own confidence instead of a
    share of the candidate list, which would grow with however deep the pool
    happens to be. A candidate the order carries without a score -- the
    unranked tail appended after a reranked window -- is 0.0, because it is
    already ranked last and yields to every scored candidate first.
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

    Case, punctuation and spacing are what two copies of one passage differ by —
    one file's text extraction and another's do not agree on them — and none of
    that is a difference in what the passage says.
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

    Greedy maximal-marginal-relevance selection over a fused order: a
    candidate's adjusted score is its relevance minus ``penalty`` for every
    candidate already taken from the same source. The best adjusted score wins,
    with the fused position and then the chunk ID settling ties, so one query
    always yields one order.

    A penalty at or below zero, a pool no deeper than the request, or a pool
    where no candidate carries a score returns the plain slice and leaves the
    ranking exactly as it was. Unscored means an unreranked BM25 or dense
    ranking, which offers no relevance to charge against: the penalty would
    otherwise be the only signal left and would replace that ranking with a
    round-robin over sources. A source that is the only one with relevant
    candidates still fills the answer: its repeats are charged like any other,
    but nothing else is left to outrank them.
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

    The ranking policy is deliberately absent. It decides how a generation is
    *searched*, not what its artifacts contain: extraction, chunk boundaries, and
    vectors are identical whatever the weights, gates, or window are, so a
    ranking edit must not discard a build in progress. The policy still reaches
    the manifest at publish time, so the record stays truthful; measured on the
    reference corpus, a ranking edit reuses every chunk and vector and costs about
    two minutes instead of a full rebuild.

    Contextual chunk headers are not a ranking policy: they decide the text a
    vector covers, so they belong here and a resume cannot mix the two.
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

    if locator.get("type") == "pdf_page":
        location = f"p. {locator.get('page_label') or locator.get('page')}"
    else:
        section = (
            locator.get("href_with_fragment")
            or locator.get("section_title")
            or locator.get("href")
        )
        location = (
            f"section {section}"
            if section
            else f"EPUB section {locator.get('section_index')}"
        )
    return f"{lead}, {location}"


def _content_tokens(value: str, stopwords: frozenset[str]) -> set[str]:
    """Return meaningful Unicode word tokens used for lexical abstention.

    ``stopwords`` is the gate's resolved function-word set, which stops at
    least what the index stops and also the question and do-support words no
    query is anchored by.
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

    Distinct tokens per text, so a passage that repeats a word does not make it
    look common. This is the table a feedback rule weights against, and it costs
    a full pass over the corpus, so a caller builds it once and keeps it.
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

    Without a table the weight is 1, which leaves a caller that has none with the
    unweighted ranking rather than a wrong one.
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

    A candidate is scored by how many leaders use it times how rare it is across
    the generation. Leader support alone mines the words that appear everywhere —
    *about*, *between*, *have* — because those are what leaders share, and they
    discriminate nothing; the rarity weight is what makes a term the author
    chose beat a term the language supplies.

    Ordered by score then term, so the same query expands the same way on every
    run: a feedback rule that is not reproducible cannot be measured. Terms the
    query already contains are skipped, because they would change nothing and
    would hide whether the expansion did anything at all.
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
    """Return caller-supplied IDs in order, without blanks or repeats."""

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

    `title` is a single string where `authors`, `categories`, and the other list
    fields are lists, and iterating a string yields its characters, so the two
    shapes cannot share one normalizer.
    """

    text = normalize_inline_text(str(value or ""))
    return {text.casefold()} if text else set()


def _matches_name_filter(supplied: set[str], values: set[str]) -> bool:
    """Whether one of the supplied phrases appears inside one of these values.

    Titles and author names are phrases, not controlled tags, so a substring is
    the honest comparison: a surname finds the author without the caller
    reproducing a bibliography's punctuation, and a remembered title fragment
    finds the work without reproducing its subtitle. Both sides arrive casefolded
    from `_normalized_filter`.
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

    `project` records which project a source was gathered for, `categories` the
    branches it belongs to, `keywords` the terms that identify it, and `language`
    what it is written in. A source normally carries one project, so that layer is
    a passthrough inside a one-project server and becomes meaningful when a corpus
    is copied or shared.

    `title` and `authors` are read as names rather than as tags, so they match by
    case-insensitive substring through `_matches_name_filter`. Reviewed values
    override the extracted ones wherever they exist, which is why these filters
    read the effective document the query path already holds.
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

    Reviewed values are free strings, so this is the inventory an agent uses to
    see a corpus partition (categories) or its project tags before searching. A
    source is counted once per value it carries, and reviewed exclusions are not
    counted.
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
    # These fields existed in older immutable generations but are internal
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
    """Return the pinned revision of one supported reranker model."""

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
    """Compare portable metadata with a generation's observational snapshot."""

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
    immutable indexes.  The portable reviewed-metadata file is authoritative at
    read time, so corrections can take effect without rebuilding those indexes.

    Older generations do not retain the automatic value hidden by a reviewed
    bibliographic override.  If such an override is later removed, use a safe
    deterministic fallback instead of silently retaining the value the user
    removed.  A later ingestion can recover automatic metadata from the source.
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
    # source, so the current reviewed lists can be represented exactly even on
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
    """Return current read-time document metadata keyed by stable document ID."""

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
# are the same unit. The counter is imported and loaded on first use, so a query
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

    A header is prepended to what the dense half embeds and never to what a
    search returns, so a returned passage stays quotable as it stands. A
    generation built without headers has no separate embedding text, and the
    canonical passage text is what was embedded.
    """

    return str(chunk.get("embedding_text") or _chunk_text(chunk))


def _chunk_header(document: dict[str, Any], locator: dict[str, Any]) -> str:
    """Return the context line prepended to a chunk's embedding text.

    The parts are what a passage cannot say about itself: the source it came from
    and the section it sits in. A PDF locator carries a page and a page label
    rather than a section, so a PDF chunk is headed by its title alone instead of
    by a fabricated section.
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
    """Project one immutable chunk and current document metadata for clients."""

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
    return "extraction_artifact" in quality_flags or not (
        has_searchable_alphanumeric_content(_chunk_text(chunk))
    )


def _candidate_flags(chunk: dict[str, Any]) -> int:
    """Return the stored retrieval-rejection verdict for one candidate.

    The artifact lookup computes this when it is built, so a query does not
    rescan chunk text. A generation whose lookup predates the stored verdict
    falls back to computing the same flags here.
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
    """Record why a candidate was withheld so the response can disclose it."""

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

    FastEmbed silently truncates input that exceeds the embedding model's limit,
    so the dense vector of such a chunk covers only a prefix while BM25 indexes
    the whole text. The audit makes that visible per chunk. Returning False means
    the tokenizer could not be inspected; the fields stay absent and the build
    metrics report the audit as unavailable rather than inventing a value.
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

# 3: the ranking policy left this identity. It never affected artifacts, and
# keeping it made a ranking edit discard a build in progress.
INGESTION_IDENTITY_POLICY_VERSION = 3

METADATA_STORAGE_POLICY = "automatic_only_runtime_overlay_v1"

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class ResearchError(RuntimeError):
    """User-facing research workflow failure."""
