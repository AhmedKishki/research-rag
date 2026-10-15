from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus.text_quality import (
    CHUNK_FLAG_APPARATUS,
    CHUNK_FLAG_CORRUPT_TEXT,
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    chunk_health_flags,
    text_corruption_reasons,
)
from ..project.policy import DEFAULT_RETRIEVAL_METHOD, RETRIEVAL_METHODS, ResearchError
from ..project.rerankers import _reranker_revision
from ..storage.durable_io import _atomic_to_thread, iter_jsonl
from .artifact_lookup import (
    LOOKUP_HEALTH_FLAGS_KEY,
    ArtifactLookup,
    VectorByContentsMapping,
    chunk_text,
)
from .dense import DenseSearchHit, RerankerUnavailable
from .documents import (
    _document_for_chunk,
    _document_matches_metadata,
    _normalized_filter,
    _public_document,
    _public_passage,
)
from .ranking import (
    _content_tokens,
    _passage_equality_key,
    _pseudo_relevance_terms,
    _source_diverse_selection,
    document_frequencies,
    passage_token_count,
)

#: The most candidate identifiers one stage of an evaluation trace may carry. The
#: trace is a bounded read of a pipeline, not the pipeline's own state, so a stage
#: deeper than this says it was cut rather than growing the answer without limit.
EVALUATION_TRACE_CANDIDATE_BUDGET = 256


def _trace_stage(chunk_ids: Sequence[str]) -> dict[str, Any]:
    """One pipeline stage: how many identifiers it held, and a bounded list of them.

    The count is the stage's own size, so a measurement reads what the pipeline
    held even where the list was cut. ``truncated`` is what keeps the two apart:
    an identifier absent from a truncated list was not necessarily absent from
    the stage, and a measurement that cannot tell has not measured it.
    """

    listed = [
        str(chunk_id) for chunk_id in chunk_ids[:EVALUATION_TRACE_CANDIDATE_BUDGET]
    ]
    return {
        "count": len(chunk_ids),
        "chunk_ids": listed,
        "truncated": len(listed) < len(chunk_ids),
    }


def _collapsed_by_reason(collapsed: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in collapsed:
        reason = str(entry.get("collapsed_by") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _evaluation_trace(
    *,
    retrieval_method: str,
    candidate_depth: int,
    bm25: dict[str, Any] | None,
    dense: dict[str, Any] | None,
    rerank: dict[str, Any],
    stages: dict[str, dict[str, Any] | None],
    collapsed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The pipeline's own account of one search: identifiers, counts, no text.

    A judged passage can be dropped by the cosine gate, the fusion, the reranked
    window, or the repetition collapse, and the answer reports only what survived
    all four. A measurement cannot see the drop from outside, so each stage names
    the identifiers it held and how many, and where a stage did not run the entry
    is ``None`` rather than an empty stage: a branch that never ran the gate has
    not admitted zero candidates through it.
    """

    listed = list(collapsed[:EVALUATION_TRACE_CANDIDATE_BUDGET])
    return {
        "retrieval_method": retrieval_method,
        "candidate_depth": candidate_depth,
        "candidate_budget": EVALUATION_TRACE_CANDIDATE_BUDGET,
        "stages": stages,
        "bm25": bm25,
        "dense": dense,
        "rerank": rerank,
        "collapsed": {
            "count": len(collapsed),
            "by_reason": _collapsed_by_reason(collapsed),
            "discarded": listed,
            "truncated": len(listed) < len(collapsed),
        },
    }


def _collapse_repetitions(
    ordered_ids: Sequence[str],
    *,
    chunks_by_id: Mapping[str, dict[str, Any]],
    documents_by_id: Mapping[str, dict[str, Any]],
    vectors: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The ranked passages with near-repeats of them removed.

    The words first, which need no vector, then the cosine for the passage that says the
    same thing in other words. A survivor is compared only against survivors, so the
    best-ranked of a repeated pair is kept. Both copies are reported with their sources.
    """

    def normalised(
        vector: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """The vector at unit length, so the dot product is a cosine.

        A store whose vectors are not at unit length would otherwise score two
        unrelated passages above any threshold this setting allows.
        """
        unit = np.asarray(vector, dtype=np.float32)
        length = float(np.linalg.norm(unit))
        if not np.isfinite(length) or length == 0.0:
            return np.zeros_like(unit)
        return unit / length

    def described(chunk_id: str, prefix: str = "") -> dict[str, Any]:
        chunk = chunks_by_id[chunk_id]
        document = _document_for_chunk(chunk, documents_by_id)
        return {
            f"{prefix}chunk_id": chunk_id,
            f"{prefix}source_id": str(document["source_id"]),
            f"{prefix}source_relative_path": str(
                document.get("source_relative_path") or ""
            ),
        }

    kept: list[str] = []
    kept_words: dict[str, str] = {}
    kept_vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    collapsed: list[dict[str, Any]] = []
    for chunk_id in ordered_ids:
        text = chunk_text(chunks_by_id[chunk_id])
        words = _passage_equality_key(text)
        # A passage with no words is not a repetition of another with no words:
        # an empty key would make every such passage match every other, and there
        # is nothing in it to say they are the same.
        repeated = kept_words.get(words) if words else None
        if repeated is not None:
            collapsed.append(
                {
                    **described(chunk_id),
                    **described(repeated, prefix="repeated_"),
                    "collapsed_by": "same_words",
                }
            )
            continue
        vector = vectors.get(text)
        unit = None if vector is None else normalised(vector)
        if unit is not None and kept_vectors:
            scores = np.asarray(kept_vectors) @ unit
            best = int(np.argmax(scores))
            score = float(scores[best])
            if score >= threshold:
                collapsed.append(
                    {
                        **described(chunk_id),
                        **described(kept[best], prefix="repeated_"),
                        "collapsed_by": "same_meaning",
                        "similarity": round(score, 6),
                    }
                )
                continue
        kept.append(chunk_id)
        if words:
            kept_words[words] = chunk_id
        if unit is not None:
            kept_vectors.append(unit)
    return kept, collapsed


#: What the lexical gates drop a candidate for, in the order the answer reports them.
_LEXICAL_REJECTIONS = (
    "no_query_token_overlap",
    "extraction_artifact",
    "corrupt_text",
    "too_short",
    "apparatus",
)


def _requested_ids(values: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _candidate_flags(chunk: dict[str, Any]) -> int:
    """Return the stored retrieval-rejection verdict for one candidate.

    The artifact lookup computes this when it is built, so a query does not rescan
    chunk text. A lookup predating it falls back to computing the same flags here.
    """

    stored = chunk.get(LOOKUP_HEALTH_FLAGS_KEY)
    if isinstance(stored, int):
        return stored
    return chunk_health_flags(
        chunk_text(chunk),
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


@dataclass
class _SearchContext:
    """What every stage of one search reads, and the withheld candidates they add to.

    ``filters`` holds `SearchWorkflow._matches_filters`'s keywords. The dense filter
    is the document set the dense index is asked for, or None when nothing narrows
    it.
    """

    generation_root: Path
    manifest: dict[str, Any]
    lookup: ArtifactLookup
    total_chunk_count: int
    documents_by_id: dict[str, dict[str, Any]]
    method: str
    filters: dict[str, set[str]]
    dense_document_filter: set[str] | None
    candidate_depth: int
    stopwords: frozenset[str]
    token_policy: dict[str, Any]
    withheld: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def use_bm25(self) -> bool:
        return self.method in {"bm25", "hybrid"}

    @property
    def use_dense(self) -> bool:
        return self.method in {"dense", "hybrid"}


@dataclass
class _LexicalRanking:
    """One BM25 ranking: what survived the gates, and what the gates dropped.

    ``window`` is how deep the index was asked, and ``returned`` how many passages
    it answered with: a window wider than the corpus is answered with everything
    the corpus holds, so the two differ.
    """

    window: int = 0
    returned: int = 0
    ranking: list[str] = field(default_factory=list)
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_LEXICAL_REJECTIONS, 0)
    )
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prf_terms: list[str] = field(default_factory=list)


@dataclass
class _DenseGate:
    """The cosine gate over what the dense index returned, and every count it reports."""

    returned_ids: list[str] = field(default_factory=list)
    eligible_ids: list[str] = field(default_factory=list)
    admitted: list[DenseSearchHit] = field(default_factory=list)
    best_score: float | None = None
    admitted_below_floor: int = 0
    rejected_below_floor: int = 0
    extraction_artifact: int = 0
    corrupt_text: int = 0
    too_short: int = 0
    apparatus: int = 0
    filtered_out: int = 0
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def counts(self) -> dict[str, Any]:
        """The gate's arithmetic, as the answer and the evaluation trace report it.

        Every eligible candidate was admitted above the floor, admitted below it by
        the margin, or rejected below it, and nothing else happened to it. The three
        counts are what makes the gate's share computable.
        """

        above_floor = len(self.admitted) - self.admitted_below_floor
        return {
            "eligible_total": len(self.eligible_ids),
            "admitted_above_floor": above_floor,
            "admitted_below_floor": self.admitted_below_floor,
            "rejected_below_floor": self.rejected_below_floor,
            "conserved": len(self.eligible_ids)
            == above_floor + self.admitted_below_floor + self.rejected_below_floor,
            "excluded_before_gate": {
                "returned_by_index": len(self.returned_ids),
                "extraction_artifact": self.extraction_artifact,
                "corrupt_text": self.corrupt_text,
                "too_short": self.too_short,
                "apparatus": self.apparatus,
                "filtered_out": self.filtered_out,
            },
        }


@dataclass
class _Rerank:
    """The ranking after the cross-encoder read its window.

    ``scored_ids`` names the window the model scored. A fallback leaves it empty
    rather than naming candidates no model read, which is what tells a scored
    window from one that was not.
    """

    ordered_ids: list[str]
    window: int = 0
    window_ids: list[str] = field(default_factory=list)
    scored_ids: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    fallback: dict[str, Any] | None = None


def _served_method(
    requested: str | None,
    manifest: dict[str, Any],
    upgrade_reasons: Sequence[str],
) -> tuple[str, dict[str, Any] | None]:
    """The method that answers this search, and the disclosure when it is not the one asked.

    No surface offers an agent a retrieval mode, so a request that names none is
    the one the app's own surfaces make, and it may be answered with another
    method when the generation cannot serve the default.

    The dense vectors carry the identity of the model that embedded them. A
    generation built with a different model cannot be scored with the current one,
    so its dense and hybrid methods are not served even where the manifest records
    them. The reasons are computed from the manifest and the resolved settings
    only, so a caller that skips the staleness walk cannot bypass this guard.
    """

    method = DEFAULT_RETRIEVAL_METHOD if requested is None else requested
    embedding_incompatible = "embedding_model" in upgrade_reasons
    available = (
        {"bm25"}
        if embedding_incompatible
        else set(manifest.get("retrieval", {}).get("available_methods") or ["bm25"])
    )
    if method in available:
        return method, None
    if requested is None and not embedding_incompatible and "bm25" in available:
        # A readable lexical generation must answer the normal agent search
        # surface without requiring an optional dense build.
        return "bm25", None
    if requested is None and embedding_incompatible and method in {"dense", "hybrid"}:
        # BM25 needs no query embeddings. A default request keeps answering and
        # discloses why the dense half was skipped.
        return "bm25", {
            "reason": "embedding_model_mismatch",
            "requested_method": method,
            "served_method": "bm25",
            "message": (
                "The selected generation's dense index was built with "
                "a different embedding model, so it cannot be scored "
                "with the current one. BM25 results are returned. Restore "
                "the recorded embedding model to use existing vectors, "
                "or explicitly ingest to build matching vectors."
            ),
            "effect": "bm25_results_returned",
        }
    raise ResearchError(
        f"Current generation does not support {method!r}; "
        f"available methods: {', '.join(sorted(available))}. "
        "Choose an available method; adding another method is optional."
    )


class SearchWorkflow:
    async def _collapse_repetitions(
        self,
        ordered_ids: list[str],
        context: _SearchContext,
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """The ranked passages with near-repeats of them removed.

        The vectors come from the generation's own portable matrix, read by passage
        text through the sidecar that maps a text to its row. The matrix is mapped,
        not read, so a large generation need not be resident.
        """

        if len(ordered_ids) < 2:
            return ordered_ids, []
        vectors_path = context.generation_root / str(
            context.manifest.get("files", {}).get("portable_embeddings")
            or "portable/embeddings.npy"
        )

        def resolve() -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
            if not vectors_path.is_file() or vectors_path.is_symlink():
                return {}
            matrix = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
            if matrix.ndim != 2 or not matrix.shape[0]:
                return {}
            texts = [chunk_text(chunks_by_id[item]) for item in ordered_ids]
            return VectorByContentsMapping(context.lookup, matrix).get_many(texts)

        vectors = await asyncio.to_thread(resolve)
        return _collapse_repetitions(
            ordered_ids,
            chunks_by_id=chunks_by_id,
            documents_by_id=context.documents_by_id,
            vectors=vectors,
            threshold=float(self.config.settings.duplicate_cosine),
        )

    async def _ensure_loaded(
        self,
        generation_root: Path,
        manifest: dict[str, Any],
    ) -> None:
        generation_id = str(manifest["generation_id"])
        if self._loaded_generation == generation_id:
            return
        await self.lexical.initialize_bm25(
            generation_root / manifest["files"]["chunks"],
            generation_root / manifest["files"]["bm25_index"],
            language=self.config.settings.bm25_stopwords_language,
        )
        self._loaded_generation = generation_id

    @staticmethod
    def _matches_filters(
        chunk: dict[str, Any],
        documents_by_id: dict[str, dict[str, Any]],
        *,
        categories_any: set[str],
        keywords: set[str],
        projects_any: set[str],
        languages_any: set[str],
        authors_any: set[str],
        titles_any: set[str],
        document_filter: set[str],
        excluded_document_ids: set[str],
        excluded_chunk_ids: set[str],
    ) -> bool:
        document = _document_for_chunk(chunk, documents_by_id)
        return not (
            str(chunk["chunk_id"]) in excluded_chunk_ids
            or chunk["document_id"] in excluded_document_ids
            or (document_filter and chunk["document_id"] not in document_filter)
            or not _document_matches_metadata(
                document,
                keywords=keywords,
                categories_any=categories_any,
                projects_any=projects_any,
                languages_any=languages_any,
                authors_any=authors_any,
                titles_any=titles_any,
            )
        )

    def _passage_token_policy(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """The token floor this query applies, and what it was read from.

        A fraction of the chunk size the generation was built to, read from the
        generation rather than the current settings, so the rule follows the corpus it
        filters. The tokenizer is the one the chunker counted with.
        """

        fraction = self.config.settings.minimum_passage_token_fraction
        if fraction <= 0:
            return {"minimum_tokens": 0, "tokenizer": None, "chunk_size": None}
        chunking = manifest.get("chunking")
        if not isinstance(chunking, dict):
            raise ResearchError(
                "Current generation records no chunking block, so "
                "retrieval.minimum_passage_token_fraction has no chunk size to "
                "measure against. Set it to 0, or ingest to build a generation "
                "that records one."
            )
        chunk_size = chunking.get("chunk_size")
        tokenizer = chunking.get("tokenizer")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ResearchError(
                "Current generation records no chunk size, so "
                "retrieval.minimum_passage_token_fraction has nothing to measure "
                "against. Set it to 0, or ingest to build a generation that "
                "records one."
            )
        if not isinstance(tokenizer, str) or not tokenizer:
            raise ResearchError(
                "Current generation records no chunker tokenizer, so "
                "retrieval.minimum_passage_token_fraction cannot count in the "
                "unit the chunks were built in. Set it to 0, or ingest to build a "
                "generation that records one."
            )
        return {
            "minimum_tokens": max(1, int(fraction * chunk_size)),
            "tokenizer": tokenizer,
            "chunk_size": chunk_size,
        }

    def _passage_too_short(
        self,
        chunk: dict[str, Any],
        *,
        token_policy: dict[str, Any] | None = None,
    ) -> bool:
        """True when a candidate is too short to be cited as evidence."""

        text = chunk_text(chunk)
        minimum = self.config.settings.minimum_passage_words
        if minimum > 0 and len(text.split()) < minimum:
            return True
        if not token_policy:
            return False
        floor = int(token_policy.get("minimum_tokens") or 0)
        tokenizer = token_policy.get("tokenizer")
        if floor <= 0 or not isinstance(tokenizer, str):
            return False
        return (
            passage_token_count(
                text,
                tokenizer,
                offline=self.config.offline,
                cache_root=self.config.model_cache_root,
            )
            < floor
        )

    async def _passage_too_short_async(
        self,
        chunk: dict[str, Any],
        *,
        token_policy: dict[str, Any] | None = None,
    ) -> bool:
        """Move tokenizer-backed checks off the serving event loop."""
        if token_policy:
            return await asyncio.to_thread(
                self._passage_too_short, chunk, token_policy=token_policy
            )
        return self._passage_too_short(chunk)

    def _record_rejected_candidate(
        self,
        examples: dict[str, list[dict[str, Any]]],
        *,
        reason: str,
        chunk: dict[str, Any],
        documents_by_id: dict[str, dict[str, Any]],
        cosine_similarity: float | None = None,
    ) -> None:
        """Name a dropped candidate's source, up to the example limit."""

        bucket = examples.setdefault(reason, [])
        if len(bucket) >= self.config.settings.maximum_withheld_examples:
            return
        document = _document_for_chunk(chunk, documents_by_id)
        entry: dict[str, Any] = {
            "chunk_id": str(chunk["chunk_id"]),
            "source_relative_path": _public_document(document).get(
                "source_relative_path"
            ),
        }
        if cosine_similarity is not None:
            entry["cosine_similarity"] = cosine_similarity
        bucket.append(entry)

    async def _bm25_ranking(
        self, query: str, context: _SearchContext
    ) -> _LexicalRanking:
        """Rank lexically with the lexical retriever held for this generation."""

        async with self._retriever(context.generation_root, context.manifest):
            limit = context.candidate_depth
            if limit <= 0:
                return _LexicalRanking()
            settings = self.config.settings
            total = context.total_chunk_count
            # An exclusion is a candidate this window can no longer use, so the
            # window widens past candidates the filters drop instead of returning
            # fewer passages than asked for. It stops at the candidate ceiling the
            # settings already declare, which is the same bound the dense depth
            # takes: past it no ranking decision has been found, and reaching for
            # it costs a second full scan of the corpus per doubling. Filters
            # narrow what a ranking returns rather than what it ranks, so a narrow
            # filter can still find nothing inside the ceiling; the response says
            # so rather than reporting an empty window as an empty corpus.
            filtered = any(context.filters.values())
            ceiling = min(total, settings.maximum_candidates) if filtered else total
            requested = min(total, max(limit * 4, settings.minimum_candidates))
            query_tokens = _content_tokens(query, context.stopwords)
            by_contents: dict[str, list[dict[str, Any]]] = {}
            loaded_contents: set[str] = set()
            # A candidate's verdict depends only on the chunk, so a repeat in a
            # widening iteration reuses it instead of rescanning the text.
            flags_cache: dict[str, int] = {}
            tokens_cache: dict[str, frozenset[str]] = {}
            while True:
                passages = await self.lexical.search_bm25(query, requested)
                missing_contents = [
                    passage for passage in passages if passage not in loaded_contents
                ]
                if missing_contents:
                    by_contents.update(
                        await asyncio.to_thread(
                            context.lookup.chunks_by_contents,
                            missing_contents,
                        )
                    )
                    loaded_contents.update(missing_contents)

                result = _LexicalRanking(window=requested, returned=len(passages))
                used: set[str] = set()
                for passage in passages:
                    candidates = by_contents.get(passage)
                    if not candidates:
                        raise ResearchError(
                            "Lexical backend returned a passage absent from the current chunk store"
                        )
                    chunk = next(
                        (
                            item
                            for item in candidates
                            if str(item["chunk_id"]) not in used
                            and self._matches_filters(
                                item, context.documents_by_id, **context.filters
                            )
                        ),
                        None,
                    )
                    if chunk is None:
                        continue
                    chunk_id = str(chunk["chunk_id"])
                    used.add(chunk_id)
                    result.chunks[chunk_id] = chunk
                    flags = flags_cache.get(chunk_id)
                    if flags is None:
                        flags = flags_cache[chunk_id] = _candidate_flags(chunk)
                    if flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                        result.rejected["extraction_artifact"] += 1
                        continue
                    if flags & CHUNK_FLAG_CORRUPT_TEXT:
                        result.rejected["corrupt_text"] += 1
                        # Reason codes are recomputed only here, because the
                        # response discloses them and they are not stored.
                        _record_withheld(
                            context.withheld,
                            chunk,
                            text_corruption_reasons(chunk_text(chunk)),
                            limit=settings.maximum_withheld_examples,
                        )
                        continue
                    if flags & CHUNK_FLAG_APPARATUS and settings.exclude_apparatus:
                        result.rejected["apparatus"] += 1
                        continue
                    tokens = tokens_cache.get(chunk_id)
                    if tokens is None:
                        tokens = tokens_cache[chunk_id] = frozenset(
                            _content_tokens(chunk_text(chunk), context.stopwords)
                        )
                    if not query_tokens.intersection(tokens):
                        result.rejected["no_query_token_overlap"] += 1
                        continue
                    if await self._passage_too_short_async(
                        chunk, token_policy=context.token_policy
                    ):
                        result.rejected["too_short"] += 1
                        self._record_rejected_candidate(
                            result.examples,
                            reason="bm25_too_short",
                            chunk=chunk,
                            documents_by_id=context.documents_by_id,
                        )
                        continue
                    result.ranking.append(chunk_id)
                    if len(result.ranking) == limit:
                        break

                if (
                    len(result.ranking) == limit
                    or not filtered
                    or requested >= ceiling
                    or len(passages) < requested
                ):
                    return result
                requested = min(ceiling, requested * 2)

    async def _lexical_ranking(
        self, query: str, context: _SearchContext
    ) -> _LexicalRanking:
        """BM25, then a second pass widened by pseudo-relevance feedback.

        The lexical leaders of the first pass name the vocabulary the author
        actually used, so a question asked in other words can still reach those
        passages. Terms are mined from the first-pass ranking only, and the second
        pass replaces the lexical ranking that the fusion and the answer see.
        """

        if not context.use_bm25:
            return _LexicalRanking()
        first = await self._bm25_ranking(query, context)
        settings = self.config.settings
        if not first.ranking or not settings.prf:
            return first
        frequencies = await self._generation_document_frequencies(
            str(context.manifest["generation_id"]),
            context.generation_root / str(context.manifest["files"]["chunks"]),
            context.stopwords,
        )
        terms = _pseudo_relevance_terms(
            query=query,
            texts=[
                chunk_text(first.chunks[chunk_id])
                for chunk_id in first.ranking[: settings.prf_documents]
            ],
            maximum_terms=settings.prf_terms,
            stopwords=context.stopwords,
            document_frequencies=frequencies,
            corpus_size=context.total_chunk_count,
        )
        if not terms:
            return first
        second = await self._bm25_ranking(f"{query} {' '.join(terms)}", context)
        # The feedback pass searched a second window, so the answer reports the
        # wider of the two rather than the first.
        second.window = max(first.window, second.window)
        second.chunks = {**first.chunks, **second.chunks}
        second.prf_terms = terms
        return second

    @staticmethod
    def _fuse_rankings(
        bm25_ranking: list[str],
        dense_ranking: list[str],
        *,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        maximum_candidates: int,
    ) -> tuple[list[str], dict[str, float]]:
        scores: defaultdict[str, float] = defaultdict(float)
        component_ranks = (
            {chunk_id: rank for rank, chunk_id in enumerate(ranking, 1)}
            for ranking in (bm25_ranking, dense_ranking)
        )
        bm25_ranks, dense_ranks = component_ranks
        for chunk_id, rank in bm25_ranks.items():
            scores[chunk_id] += bm25_weight / (rrf_k + rank)
        for chunk_id, rank in dense_ranks.items():
            scores[chunk_id] += dense_weight / (rrf_k + rank)
        absent = maximum_candidates + 1
        ordered = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                min(
                    bm25_ranks.get(chunk_id, absent),
                    dense_ranks.get(chunk_id, absent),
                ),
                chunk_id,
            ),
        )
        return ordered, dict(scores)

    async def _generation_document_frequencies(
        self,
        generation_id: str,
        chunks_path: Path,
        stopwords: frozenset[str],
    ) -> dict[str, int]:
        """This generation's term frequencies, built once and kept.

        A feedback term must be rare, not merely frequent, which is what the table
        gives. It costs a full pass, so it is built on the first search that asks.
        """

        cached = self._document_frequencies
        if cached is not None and cached[0] == generation_id and cached[1] == stopwords:
            return cached[2]
        frequencies = await _atomic_to_thread(
            document_frequencies,
            (chunk_text(record) for record in iter_jsonl(chunks_path)),
            stopwords,
        )
        self._document_frequencies = (generation_id, stopwords, frequencies)
        return frequencies

    async def _candidate_depth(
        self,
        lookup: ArtifactLookup,
        top_k: int,
        active_document_ids: set[str],
        *,
        narrowed: bool,
        excluded_chunk_ids: set[str],
    ) -> int:
        """How many candidates each half ranks: what the filters leave, within bounds."""

        active = await asyncio.to_thread(
            lookup.chunk_count, active_document_ids if narrowed else None
        )
        # The count above filters by document, so it still holds every excluded
        # chunk inside the active documents. The dense backend filters by document
        # too and cannot be asked for less, so the candidates it will drop are
        # taken off the depth here instead: a post-query drop must not cost the
        # reader a passage.
        if excluded_chunk_ids:
            excluded = await asyncio.to_thread(
                lookup.chunks_by_ids, sorted(excluded_chunk_ids)
            )
            active = max(
                0,
                active
                - sum(
                    str(chunk["document_id"]) in active_document_ids
                    for chunk in excluded.values()
                ),
            )
        settings = self.config.settings
        return min(
            active,
            settings.maximum_candidates,
            max(settings.minimum_candidates, top_k * 4),
        )

    async def _dense_search(
        self, query: str, context: _SearchContext
    ) -> list[DenseSearchHit]:
        """The dense index's nearest passages inside the documents the filters leave."""

        if (
            not context.use_dense
            or context.candidate_depth == 0
            or context.dense_document_filter == set()
        ):
            return []
        return await asyncio.to_thread(
            self._dense_for(context.manifest).search,
            context.generation_root / context.manifest["files"]["dense_index"],
            query,
            context.candidate_depth,
            # Keep Qdrant payloads lean. Translate current reviewed metadata
            # filters to document IDs at query time so edits remain exact without
            # rebuilding the dense index.
            document_ids=sorted(context.dense_document_filter or []),
            excluded_document_ids=sorted(context.filters["excluded_document_ids"]),
        )

    async def _dense_gate(
        self,
        hits: list[DenseSearchHit],
        context: _SearchContext,
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> _DenseGate:
        """Admit dense candidates by score, after every filter that is not the score.

        The floor is absolute, and a query whose best passage still scores below it
        can have its whole candidate list packed into a band under it, so the floor
        drops candidates that matched it about as closely. The margin admits that
        band, but only once something has cleared the floor, so a query the corpus
        cannot support still abstains rather than returning its least-bad passage.
        """

        gate = _DenseGate(returned_ids=[hit.chunk_id for hit in hits])
        eligible: list[tuple[DenseSearchHit, dict[str, Any]]] = []
        for hit in hits:
            chunk = chunks_by_id.get(hit.chunk_id)
            if chunk is None:
                raise ResearchError(
                    "The dense index returned a chunk absent from the current "
                    f"chunk store: {hit.chunk_id}"
                )
            flags = _candidate_flags(chunk)
            if flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                gate.extraction_artifact += 1
            elif flags & CHUNK_FLAG_CORRUPT_TEXT:
                gate.corrupt_text += 1
                _record_withheld(
                    context.withheld,
                    chunk,
                    text_corruption_reasons(chunk_text(chunk)),
                    limit=self.config.settings.maximum_withheld_examples,
                )
            elif (
                flags & CHUNK_FLAG_APPARATUS and self.config.settings.exclude_apparatus
            ):
                gate.apparatus += 1
            elif not self._matches_filters(
                chunk, context.documents_by_id, **context.filters
            ):
                gate.filtered_out += 1
            elif await self._passage_too_short_async(
                chunk, token_policy=context.token_policy
            ):
                gate.too_short += 1
                self._record_rejected_candidate(
                    gate.examples,
                    reason="dense_too_short",
                    chunk=chunk,
                    documents_by_id=context.documents_by_id,
                )
            else:
                eligible.append((hit, chunk))
        gate.eligible_ids = [hit.chunk_id for hit, _chunk in eligible]
        gate.best_score = max((hit.score for hit, _chunk in eligible), default=None)
        floor = self.config.settings.dense_minimum_cosine_similarity
        margin = self.config.settings.dense_relative_similarity_margin
        admission_floor = floor
        if margin > 0.0 and gate.best_score is not None and gate.best_score >= floor:
            admission_floor = min(floor, gate.best_score - margin)
        for hit, chunk in eligible:
            if hit.score >= admission_floor:
                if hit.score < floor:
                    gate.admitted_below_floor += 1
                gate.admitted.append(hit)
                continue
            gate.rejected_below_floor += 1
            self._record_rejected_candidate(
                gate.examples,
                reason="dense_below_threshold",
                chunk=chunk,
                documents_by_id=context.documents_by_id,
                cosine_similarity=round(hit.score, 4),
            )
        return gate

    async def _rerank(
        self,
        query: str,
        ordered_ids: list[str],
        base_ranks: dict[str, int],
        *,
        top_k: int,
        model: str | None,
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> _Rerank:
        """The cross-encoder's order over the head of the ranking; the tail keeps its place.

        Requested reranking cannot run without its model, so an unloadable one
        leaves the candidate order unchanged and the answer says why instead of
        failing the search.
        """

        if not ordered_ids:
            return _Rerank(ordered_ids)
        settings = self.config.settings
        window = min(
            len(ordered_ids),
            settings.rerank_max_candidates,
            max(top_k * settings.rerank_window_multiple, settings.rerank_window_floor),
        )
        head = ordered_ids[:window]
        try:
            # A backend that answers tool calls keeps its configured model, so the
            # keyword is passed only when this call names another.
            scores = await asyncio.to_thread(
                self.dense.rerank,
                query,
                [chunk_text(chunks_by_id[item]) for item in head],
                **({"model": model} if model else {}),
            )
        except RerankerUnavailable as exc:
            return _Rerank(
                ordered_ids,
                window=window,
                window_ids=head,
                fallback={
                    "reason": "reranker_model_unavailable",
                    "message": str(exc),
                    "effect": "unranked_candidate_order_returned",
                },
            )
        by_id = dict(zip(head, scores, strict=True))
        reordered = sorted(
            head,
            key=lambda chunk_id: (-by_id[chunk_id], base_ranks[chunk_id], chunk_id),
        )
        return _Rerank(
            [*reordered, *ordered_ids[window:]],
            window=window,
            window_ids=reordered,
            scored_ids=head,
            scores=by_id,
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        categories_any: list[str] | None = None,
        projects_any: list[str] | None = None,
        keywords: list[str] | None = None,
        languages_any: list[str] | None = None,
        authors_any: list[str] | None = None,
        titles_any: list[str] | None = None,
        source_ids: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
        retrieval_method: str | None = None,
        rerank: bool = False,
        rerank_model: str | None = None,
        include_staleness: bool = True,
        evaluation_trace: bool = False,
    ) -> dict[str, Any]:
        """Retrieve evidence.

        A caller that names no ``retrieval_method`` asks for the engine's default.
        A generation whose recorded dense identity does not match the current
        embedding model cannot serve dense or hybrid, so a default search is
        answered with BM25 and a ``retrieval_fallback`` disclosure, while an
        explicit ``dense`` or ``hybrid`` request is refused before any model or
        backend is used.

        The public MCP tool defaults ``rerank`` to true, the largest quality gain the judged
        set showed (``MEASUREMENTS.md``); this API keeps the neutral default. An unloadable
        reranker model still returns a search, in unranked candidate order with
        ``rerank_fallback``.

        ``rerank_model`` names a reranker for this call alone, so one process can
        compare models against the same generation.

        ``evaluation_trace`` adds ``evaluation_trace`` to the answer: the bounded
        identifier sets of each ranking stage, the gate's own conservation
        arithmetic, and what the repetition collapse discarded. It is what a
        measurement reads to see where a passage went, no surface asks for it,
        and no surface projects it into an agent's answer.
        """
        started = time.perf_counter()
        query = query.strip()
        if not query:
            raise ResearchError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ResearchError("top_k must be between 1 and 50")
        if retrieval_method is not None:
            retrieval_method = retrieval_method.casefold().strip()
            if retrieval_method not in RETRIEVAL_METHODS:
                raise ResearchError(
                    "retrieval_method must be one of: bm25, dense, hybrid"
                )
        if rerank_model is not None and not rerank:
            raise ResearchError("rerank_model requires rerank=True")
        applied_reranker = rerank_model or self.config.reranker_model
        try:
            applied_reranker_revision = _reranker_revision(applied_reranker)
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc
        settings = self.config.settings

        async with self._search_read() as lease:
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lease.hold(str(manifest["generation_id"]))
            guard_upgrade_reasons = tuple(self._generation_upgrade_reasons(manifest))
            if "project_identity" in guard_upgrade_reasons:
                raise ResearchError(
                    "The selected generation belongs to a different project. "
                    "Select a generation belonging to this project; build one only "
                    "if no compatible generation is retained."
                )
            token_policy = self._passage_token_policy(manifest)
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            total_chunk_count = await asyncio.to_thread(lookup.chunk_count)
            if not total_chunk_count:
                raise ResearchError("The current generation has no chunks")
            documents_by_id = self._reviewed_documents(manifest)
            exclusions = self._source_exclusions()
            excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
            # A chunk id is derived from content, so the same passage keeps it
            # across a rebuild of unchanged bytes: the decision is enforced by
            # every id it names, and which of them this generation holds is
            # reported by the review state rather than guessed here.
            excluded_chunk_ids = set(self._chunk_exclusions())
            method, retrieval_fallback = _served_method(
                retrieval_method, manifest, guard_upgrade_reasons
            )

            metadata_filters = {
                "categories_any": _normalized_filter(categories_any),
                "projects_any": _normalized_filter(projects_any),
                "keywords": _normalized_filter(keywords),
                "languages_any": _normalized_filter(languages_any),
                "authors_any": _normalized_filter(authors_any),
                "titles_any": _normalized_filter(titles_any),
            }
            requested_source_ids = _requested_ids(source_ids)
            requested_exclude_source_ids = _requested_ids(exclude_source_ids)
            document_filter, unknown_source_ids = self._document_ids_for_source_ids(
                manifest, requested_source_ids
            )
            excluded_on_request, unknown_exclude_source_ids = (
                self._document_ids_for_source_ids(
                    manifest, requested_exclude_source_ids
                )
            )
            if requested_source_ids and not document_filter:
                raise ResearchError(
                    "source_ids matched no document in the current generation: "
                    f"{', '.join(unknown_source_ids)}. Use find_source to resolve "
                    "the current IDs; a renamed or moved source receives a new "
                    "source_id."
                )
            # Reviewed exclusions always win over a search-level exclusion, and a
            # search-level include can never re-admit an excluded source.
            excluded_document_ids |= excluded_on_request
            metadata_filtered = any(metadata_filters.values())
            matching_document_ids = {
                document_id
                for document_id, document in documents_by_id.items()
                if _document_matches_metadata(document, **metadata_filters)
            }
            dense_document_filter = matching_document_ids if metadata_filtered else None
            if document_filter:
                dense_document_filter = (
                    document_filter
                    if dense_document_filter is None
                    else dense_document_filter & document_filter
                )
            active_document_ids = {
                document_id
                for document_id in matching_document_ids
                if document_id not in excluded_document_ids
                and (not document_filter or document_id in document_filter)
            }
            context = _SearchContext(
                generation_root=generation_root,
                manifest=manifest,
                lookup=lookup,
                total_chunk_count=total_chunk_count,
                documents_by_id=documents_by_id,
                method=method,
                filters={
                    **metadata_filters,
                    "document_filter": document_filter,
                    "excluded_document_ids": excluded_document_ids,
                    "excluded_chunk_ids": excluded_chunk_ids,
                },
                dense_document_filter=dense_document_filter,
                candidate_depth=await self._candidate_depth(
                    lookup,
                    top_k,
                    active_document_ids,
                    narrowed=bool(
                        metadata_filtered or document_filter or excluded_document_ids
                    ),
                    excluded_chunk_ids=excluded_chunk_ids,
                ),
                stopwords=settings.gate_stopwords,
                token_policy=token_policy,
            )

            lexical, dense_hits = await asyncio.gather(
                self._lexical_ranking(query, context),
                self._dense_search(query, context),
            )
            chunks_by_id = {
                **lexical.chunks,
                **await asyncio.to_thread(
                    lookup.chunks_by_ids, [hit.chunk_id for hit in dense_hits]
                ),
            }
            gate = await self._dense_gate(dense_hits, context, chunks_by_id)
            bm25_ranking = lexical.ranking
            dense_ranking = [hit.chunk_id for hit in gate.admitted]
            if method == "hybrid":
                fused_ids, fusion_scores = self._fuse_rankings(
                    bm25_ranking,
                    dense_ranking,
                    rrf_k=settings.rrf_k,
                    bm25_weight=settings.bm25_weight,
                    dense_weight=settings.dense_weight,
                    maximum_candidates=settings.maximum_candidates,
                )
            else:
                fused_ids = bm25_ranking if method == "bm25" else dense_ranking
                fusion_scores = {}
            unknown_ids = [item for item in fused_ids if item not in chunks_by_id]
            if unknown_ids:
                raise ResearchError(
                    "The retrieval index returned chunk IDs absent from the current "
                    f"chunk store: {unknown_ids[:3]}"
                )
            base_ranks = {chunk_id: rank for rank, chunk_id in enumerate(fused_ids, 1)}
            reranked = (
                await self._rerank(
                    query,
                    fused_ids,
                    base_ranks,
                    top_k=top_k,
                    model=rerank_model,
                    chunks_by_id=chunks_by_id,
                )
                if rerank
                else _Rerank(fused_ids)
            )
            reranked_applied = bool(reranked.scores)

            # A repeated passage is settled here, on the candidates this search
            # already holds, rather than by dropping a chunk at build time. Two
            # copies of one passage stay in the corpus and independently citable,
            # and the answer shows one of them.
            candidate_ids, collapsed = await self._collapse_repetitions(
                reranked.ordered_ids, context, chunks_by_id
            )
            source_id_by_chunk = {
                item: str(
                    _document_for_chunk(chunks_by_id[item], documents_by_id)[
                        "source_id"
                    ]
                )
                for item in candidate_ids
            }
            selected_ids = _source_diverse_selection(
                candidate_ids,
                source_id_by_chunk=source_id_by_chunk,
                scores=reranked.scores or fusion_scores,
                top_k=top_k,
                penalty=settings.source_diversity_penalty,
            )

            bm25_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(bm25_ranking, 1)
            }
            dense_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(dense_ranking, 1)
            }
            dense_scores = {hit.chunk_id: hit.score for hit in gate.admitted}
            hits: list[dict[str, Any]] = []
            for rank, chunk_id in enumerate(selected_ids, 1):
                chunk = chunks_by_id[chunk_id]
                bm25_rank = bm25_ranks.get(chunk_id)
                dense_rank = dense_ranks.get(chunk_id)
                hits.append(
                    {
                        "rank": rank,
                        "retrieval_rank": base_ranks[chunk_id],
                        **_public_passage(
                            chunk, _document_for_chunk(chunk, documents_by_id)
                        ),
                        "match_kind": (
                            ("hybrid" if dense_rank else "lexical")
                            if bm25_rank
                            else "semantic"
                        ),
                        "retrieval_method": method,
                        "component_ranks": {"bm25": bm25_rank, "dense": dense_rank},
                        "component_scores": {
                            "dense_cosine_similarity": dense_scores.get(chunk_id),
                            "bm25": None,
                        },
                        "fusion_score": fusion_scores.get(chunk_id),
                        "rerank_score": reranked.scores.get(chunk_id),
                    }
                )

            repetition_disclosure: dict[str, Any] = {}
            if collapsed:
                # A lean answer states the rule once rather than repeating it per
                # passage, so this is a count, and the pairs are for the caller
                # that asks for the full detail: the actionable part of an
                # overlap is which two files it came from.
                by_words = sum(
                    1 for item in collapsed if item["collapsed_by"] == "same_words"
                )
                repetition_disclosure = {
                    "repetitions_collapsed": len(collapsed),
                    "collapsed_by_same_words": by_words,
                    "collapsed_by_same_meaning": len(collapsed) - by_words,
                    "note": (
                        "One passage repeated another and only the better-ranked "
                        "copy is shown. Both remain in the corpus and can be "
                        "retrieved by name."
                    ),
                }

            if include_staleness:
                # Walking the source tree is the only per-request work here that
                # grows with the collection, so a caller that does not need a
                # freshness verdict can skip it.
                status = await asyncio.to_thread(self._status, current)
                stale: bool | None = status["stale"]
                upgrade_reasons = list(status["upgrade_reasons"])
            else:
                stale = None
                upgrade_reasons = list(guard_upgrade_reasons)
            window = lexical.window or context.candidate_depth
            payload = {
                "query": query,
                "generation_id": manifest["generation_id"],
                "stale": stale,
                "staleness_checked": include_staleness,
                "generation_upgrade_required": bool(upgrade_reasons),
                "excluded_source_count": len(exclusions),
                "excluded_chunk_count": len(excluded_chunk_ids),
                "filters": {
                    "categories_any": sorted(metadata_filters["categories_any"]),
                    "projects_any": sorted(metadata_filters["projects_any"]),
                    "keywords_all": sorted(metadata_filters["keywords"]),
                    "languages_any": sorted(metadata_filters["languages_any"]),
                    "authors_any": sorted(metadata_filters["authors_any"]),
                    "titles_any": sorted(metadata_filters["titles_any"]),
                    "source_document_ids": sorted(document_filter),
                    "source_ids": requested_source_ids,
                    "exclude_source_ids": requested_exclude_source_ids,
                    "unknown_source_ids": unknown_source_ids,
                    "unknown_exclude_source_ids": unknown_exclude_source_ids,
                    "active_document_count": len(active_document_ids),
                    "corpus_chunk_count": total_chunk_count,
                    "window_chunk_count": window,
                    "window_is_whole_corpus": window >= total_chunk_count,
                    "note": (
                        "Filters drop passages after the corpus is ranked, so an "
                        "empty result reads as 'nothing the window reached matches "
                        "these filters' rather than 'no source carries them'. "
                        "window_chunk_count is how much of the corpus the ranking "
                        "reached; when it is below corpus_chunk_count a narrow "
                        "filter can find nothing it did not reach. Reviewed source "
                        "exclusions always win: a "
                        "source_ids entry for an excluded source stays excluded. "
                        "Unresolved IDs are reported in unknown_source_ids and "
                        "unknown_exclude_source_ids; an include list that resolves "
                        "to nothing is an error rather than an unfiltered result. "
                        "authors_any and titles_any match reviewed values by "
                        "case-insensitive substring."
                    ),
                },
                "retrieval_method": method,
                "reranked": reranked_applied,
                "rerank_requested": rerank,
                "rerank_fallback": reranked.fallback,
                "retrieval_fallback": retrieval_fallback,
                "rerank_window": reranked.window,
                "prf_requested": settings.prf,
                "prf_terms": lexical.prf_terms,
                "candidate_depth": context.candidate_depth,
                "candidate_count": len(candidate_ids),
                "candidate_distinct_reference_count": len(
                    set(source_id_by_chunk.values())
                ),
                "requested_top_k": top_k,
                "fusion": (
                    {
                        "method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": settings.rrf_k,
                        "bm25_weight": settings.bm25_weight,
                        "dense_weight": settings.dense_weight,
                    }
                    if method == "hybrid"
                    else None
                ),
                "relevance_policy": {
                    "bm25_requires_query_token_overlap": True,
                    "dense_minimum_cosine_similarity": (
                        settings.dense_minimum_cosine_similarity
                    ),
                },
                "selection_policy": {
                    "method": "greedy_source_diversity",
                    "source_diversity_penalty": settings.source_diversity_penalty,
                },
                "dense_gate": {
                    "minimum_cosine_similarity": (
                        settings.dense_minimum_cosine_similarity
                    ),
                    "relative_margin": settings.dense_relative_similarity_margin,
                    "best_cosine_similarity": (
                        None if gate.best_score is None else round(gate.best_score, 4)
                    ),
                    # The denominator, so the share the gate removed is a
                    # division rather than an estimate.
                    **(
                        gate.counts()
                        if context.use_dense
                        else dict.fromkeys(gate.counts())
                    ),
                    "note": (
                        "Dense candidates are admitted by score. A candidate below "
                        "the floor is admitted when the query's best candidate "
                        "cleared the floor and this one is within the margin of it; "
                        "when nothing clears the floor the query is left with "
                        "nothing rather than its least-bad passage. A query whose "
                        "whole candidate list sits in a narrow band under the floor "
                        "is why the margin exists. The three counts above account "
                        "for every eligible candidate between them; excluded_before_gate "
                        "counts the ones dropped before the score decision for a "
                        "reason that is not the score, and those are also counted "
                        "per reason in rejected_candidates. A method that never runs "
                        "the dense half reports null here rather than a zero, because "
                        "zero candidates through a gate that did not open is not a "
                        "measurement."
                    ),
                },
                "passage_length_policy": {
                    "minimum_words": settings.minimum_passage_words,
                    "minimum_tokens": token_policy["minimum_tokens"],
                    "token_fraction": settings.minimum_passage_token_fraction,
                    "chunk_size": token_policy["chunk_size"],
                    "tokenizer": token_policy["tokenizer"],
                    "note": (
                        "A candidate below either floor is dropped before fusion: "
                        "fewer words than minimum_words, or fewer tokens than "
                        "token_fraction of the generation's chunk size. Chunks never "
                        "span extraction units, so an index line, a heading, or a "
                        "copyright line becomes a chunk that matches a query about "
                        "its own words while carrying no prose to cite. The token "
                        "floor is counted over the returned text with the tokenizer "
                        "the generation was chunked by, so a header cannot make a "
                        "fragment look long."
                    ),
                },
                "apparatus_policy": {
                    "exclude_apparatus": settings.exclude_apparatus,
                    "note": (
                        "With exclude_apparatus on, a candidate whose text is a "
                        "reference list, citation-only notes, a back-of-book index, "
                        "or a copyright page is dropped before fusion, and each half "
                        "counts it in rejected_candidates. The verdict is read from "
                        "the passage's own text when the artifact lookup is built. "
                        "Nothing is deleted, and turning the setting off returns "
                        "every passage."
                    ),
                },
                "rejected_candidates": {
                    **{
                        f"bm25_{reason}": count
                        for reason, count in lexical.rejected.items()
                    },
                    "dense_below_threshold": gate.rejected_below_floor,
                    "dense_extraction_artifact": gate.extraction_artifact,
                    "dense_corrupt_text": gate.corrupt_text,
                    "dense_too_short": gate.too_short,
                    "dense_apparatus": gate.apparatus,
                },
                "withheld_candidates": {
                    "policy": "corruption_evidence_only",
                    "note": (
                        "Candidates withheld from this result for corruption "
                        "evidence. Script notes such as non_latin_dominant or "
                        "mixed_script_text appear per hit in text_notes and never "
                        "withhold a passage."
                    ),
                    "total": sum(
                        int(entry["count"]) for entry in context.withheld.values()
                    ),
                    "reasons": {
                        reason: {
                            "count": int(entry["count"]),
                            "example_chunk_ids": list(entry["example_chunk_ids"]),
                        }
                        for reason, entry in sorted(context.withheld.items())
                    },
                    "flagged_passages_returned": sum(
                        1 for hit in hits if hit.get("text_notes")
                    ),
                },
                "collapsed_repetitions": {**repetition_disclosure, "pairs": collapsed},
                "rejected_candidate_examples": {
                    "policy": "bounded_examples",
                    "limit_per_reason": settings.maximum_withheld_examples,
                    "note": (
                        "The gate's count says how many candidates it removed; "
                        "these name the sources they came from, so a thin answer "
                        "can be read as the corpus being thinned rather than "
                        "silent."
                    ),
                    "reasons": {
                        "bm25_too_short": lexical.examples.get("bm25_too_short", []),
                        "dense_below_threshold": gate.examples.get(
                            "dense_below_threshold", []
                        ),
                        "dense_too_short": gate.examples.get("dense_too_short", []),
                    },
                },
                "dense_fidelity": {
                    "embedding_maximum_tokens": settings.embedding_maximum_tokens,
                    "audited_passages_returned": sum(
                        1
                        for hit in hits
                        if isinstance(hit.get("embedding_token_count"), int)
                    ),
                    "truncated_passages_returned": sum(
                        1 for hit in hits if hit.get("dense_truncated")
                    ),
                    "note": (
                        "FastEmbed truncates text beyond the embedding model "
                        "limit, so such a chunk is matched lexically but only "
                        "partly semantically. A null embedding_token_count means "
                        "the generation predates the ingestion audit."
                    ),
                },
                "embedding_model": (
                    settings.embedding_model if context.use_dense else None
                ),
                "embedding_model_revision": (
                    settings.embedding_model_revision if context.use_dense else None
                ),
                "reranker_model": applied_reranker if reranked_applied else None,
                "reranker_model_revision": (
                    applied_reranker_revision if reranked_applied else None
                ),
                "result_count": len(hits),
                "distinct_reference_count": len(
                    {str(hit["source_id"]) for hit in hits}
                ),
                "relevance_limited": len(candidate_ids) < top_k,
                "hits": hits,
            }
            if evaluation_trace:
                payload["evaluation_trace"] = _evaluation_trace(
                    retrieval_method=method,
                    candidate_depth=context.candidate_depth,
                    bm25=(
                        {
                            "index_window": lexical.window,
                            "returned_by_index": lexical.returned,
                            "after_gates": len(bm25_ranking),
                            "rejected": dict(sorted(lexical.rejected.items())),
                        }
                        if context.use_bm25
                        else None
                    ),
                    dense=(
                        {"returned_by_index": len(gate.returned_ids), **gate.counts()}
                        if context.use_dense
                        else None
                    ),
                    rerank={
                        "requested": bool(rerank),
                        "applied": reranked_applied,
                        "fallback": reranked.fallback,
                        "window": reranked.window,
                        "scored_ids": reranked.scored_ids[
                            :EVALUATION_TRACE_CANDIDATE_BUDGET
                        ],
                        "scored_count": len(reranked.scored_ids),
                        "scored_truncated": (
                            len(reranked.scored_ids) > EVALUATION_TRACE_CANDIDATE_BUDGET
                        ),
                    },
                    stages={
                        "dense_before_filters": (
                            _trace_stage(gate.returned_ids)
                            if context.use_dense
                            else None
                        ),
                        "dense_eligible": (
                            _trace_stage(gate.eligible_ids)
                            if context.use_dense
                            else None
                        ),
                        "dense_admitted": (
                            _trace_stage(dense_ranking) if context.use_dense else None
                        ),
                        "bm25_after_gates": (
                            _trace_stage(bm25_ranking) if context.use_bm25 else None
                        ),
                        "fused_pre_rerank": _trace_stage(fused_ids),
                        "reranked": (
                            _trace_stage(reranked.window_ids)
                            if reranked.window
                            else None
                        ),
                        "post_collapse": _trace_stage(candidate_ids),
                        "final": _trace_stage(selected_ids),
                    },
                    collapsed=collapsed,
                )
            # A measurement's search is not a reader's, so it is not counted.
            if not evaluation_trace:
                await self._record_search(
                    payload,
                    requested_top_k=top_k,
                    started=started,
                    query=query,
                    filters={
                        "categories_any": categories_any or [],
                        "projects_any": projects_any or [],
                        "keywords": keywords or [],
                        "languages_any": languages_any or [],
                        "authors_any": authors_any or [],
                        "titles_any": titles_any or [],
                        "source_ids": source_ids or [],
                        "exclude_source_ids": exclude_source_ids or [],
                    },
                )
            return payload

    async def get_passage(
        self,
        chunk_id: str,
        *,
        context_chunks: int = 1,
    ) -> dict[str, Any]:
        """One passage and the neighbours around it, from the selected generation.

        An excluded passage is refused here the way an excluded source is, because
        a caller asking for it by name is asking for it as evidence. A neighbour
        is context rather than a claim, so an excluded one is still returned: a
        reader who opened a passage to see what surrounds it would otherwise lose
        the shape of the argument to a decision about one passage inside it.
        """

        if not 0 <= context_chunks <= 5:
            raise ResearchError("context_chunks must be between 0 and 5")
        async with self._read() as lease:
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lease.hold(str(manifest["generation_id"]))
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            documents_by_id = self._reviewed_documents(manifest)
            target = (await asyncio.to_thread(lookup.chunks_by_ids, [chunk_id])).get(
                chunk_id
            )
            if target is None:
                raise ResearchError(f"Unknown chunk_id: {chunk_id}")
            exclusions = self._source_exclusions()
            excluded_document_ids = self._excluded_document_ids(
                manifest,
                exclusions,
            )
            if target["document_id"] in excluded_document_ids:
                raise ResearchError(
                    "The source for this chunk is currently excluded from retrieval; "
                    "include the source before requesting its passage"
                )
            excluded_chunk_ids = set(self._chunk_exclusions())
            if chunk_id in excluded_chunk_ids:
                raise ResearchError(
                    "The requested chunk is currently excluded from retrieval; "
                    "include the chunk before requesting its passage"
                )
            flags = _candidate_flags(target)
            if flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                raise ResearchError(
                    "The requested chunk is an extraction artifact and is not "
                    "available; re-ingest to remove it from the generation"
                )
            if flags & CHUNK_FLAG_CORRUPT_TEXT:
                raise ResearchError(
                    "The requested chunk contains corrupt extracted text and is "
                    "not available; re-ingest to remove it from the generation"
                )
            window = await asyncio.to_thread(
                lookup.passage_window,
                str(target["document_id"]),
                int(target["document_chunk_index"]),
                radius=context_chunks,
            )
            context = []
            document = _document_for_chunk(target, documents_by_id)
            for item in window:
                passage = _public_passage(item, document)
                if item["chunk_id"] in excluded_chunk_ids:
                    passage["excluded_from_search"] = True
                context.append(passage)
            return {
                "generation_id": manifest["generation_id"],
                "requested_chunk_id": chunk_id,
                "context": context,
            }
