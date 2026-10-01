"""Ranked retrieval: BM25, fusion, reranking, and evidence assembly."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_lookup import ArtifactLookup, VectorByContentsMapping
from .dense import DenseSearchHit, RerankerUnavailable
from .extraction import (
    CHUNK_FLAG_CORRUPT_TEXT,
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    text_corruption_reasons,
)
from .storage import iter_jsonl
from .support import (
    DEFAULT_RETRIEVAL_METHOD,
    RETRIEVAL_METHODS,
    ResearchError,
    _atomic_to_thread,
    _candidate_flags,
    _chunk_text,
    _content_tokens,
    _document_for_chunk,
    _document_matches_metadata,
    _effective_documents,
    _is_extraction_artifact,
    _normalized_filter,
    _passage_equality_key,
    _pseudo_relevance_terms,
    _public_document,
    _public_passage,
    _record_withheld,
    _requested_ids,
    _reranker_revision,
    _source_diverse_selection,
    document_frequencies,
    passage_token_count,
)


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
        text = _chunk_text(chunks_by_id[chunk_id])
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


class SearchWorkflow:
    async def _collapse_repetitions(
        self,
        ordered_ids: list[str],
        *,
        generation_root: Path,
        manifest: dict[str, Any],
        lookup: ArtifactLookup,
        chunks_by_id: dict[str, dict[str, Any]],
        documents_by_id: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """The ranked passages with near-repeats of them removed.

        The vectors come from the generation's own portable matrix, read by passage
        text through the sidecar that maps a text to its row. The matrix is mapped,
        not read, so a large generation need not be resident.
        """

        if len(ordered_ids) < 2:
            return ordered_ids, []
        vectors_path = generation_root / str(
            manifest.get("files", {}).get("portable_embeddings")
            or "portable/embeddings.npy"
        )

        def resolve() -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
            if not vectors_path.is_file() or vectors_path.is_symlink():
                return {}
            matrix = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
            if matrix.ndim != 2 or not matrix.shape[0]:
                return {}
            texts = [_chunk_text(chunks_by_id[item]) for item in ordered_ids]
            return VectorByContentsMapping(lookup, matrix).get_many(texts)

        vectors = await asyncio.to_thread(resolve)
        return _collapse_repetitions(
            ordered_ids,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
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
        await self.ultrarag.initialize_bm25(
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

        text = _chunk_text(chunk)
        minimum = self.config.settings.minimum_passage_words
        if minimum > 0 and len(text.split()) < minimum:
            return True
        if not token_policy:
            return False
        floor = int(token_policy.get("minimum_tokens") or 0)
        tokenizer = token_policy.get("tokenizer")
        if floor <= 0 or not isinstance(tokenizer, str):
            return False
        return passage_token_count(text, tokenizer) < floor

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
        self,
        query: str,
        lookup: ArtifactLookup,
        total_chunk_count: int,
        documents_by_id: dict[str, dict[str, Any]],
        limit: int,
        *,
        categories_any: set[str],
        projects_any: set[str],
        keywords: set[str],
        languages_any: set[str],
        authors_any: set[str],
        titles_any: set[str],
        document_filter: set[str],
        excluded_document_ids: set[str],
        excluded_chunk_ids: set[str],
        withheld: dict[str, dict[str, Any]],
        stopwords: frozenset[str],
        token_policy: dict[str, Any] | None = None,
    ) -> tuple[
        list[str],
        dict[str, int],
        dict[str, dict[str, Any]],
        dict[str, list[dict[str, Any]]],
        int,
    ]:
        if limit <= 0:
            return (
                [],
                {
                    "no_query_token_overlap": 0,
                    "extraction_artifact": 0,
                    "corrupt_text": 0,
                    "too_short": 0,
                },
                {},
                {},
                0,
            )
        # An exclusion is a candidate this window can no longer use, so the
        # window widens past it instead of returning fewer passages than asked
        # for. Without this clause a top_k of ten answers with nine the moment
        # one chunk is excluded.
        filtered = bool(
            categories_any
            or keywords
            or projects_any
            or languages_any
            or authors_any
            or titles_any
            or document_filter
            or excluded_document_ids
            or excluded_chunk_ids
        )
        # The window widens past candidates the filters will drop, so an exclusion
        # does not shrink top_k. It stops at the candidate ceiling the settings
        # already declare, which is the same bound the dense depth takes: past it
        # no measurement has found a ranking decision, and reaching for it costs a
        # second full scan of the corpus per doubling. Filters narrow what a
        # ranking returns rather than what it ranks, so a narrow filter can still
        # find nothing inside the ceiling; the response says so rather than
        # reporting an empty window as an empty corpus.
        ceiling = (
            min(total_chunk_count, self.config.settings.maximum_candidates)
            if filtered
            else total_chunk_count
        )
        requested = min(
            total_chunk_count,
            max(limit * 4, self.config.settings.minimum_candidates),
        )
        query_tokens = _content_tokens(query, stopwords)
        by_contents: dict[str, list[dict[str, Any]]] = {}
        loaded_contents: set[str] = set()
        # A candidate's verdict depends only on the chunk, so a repeat in a
        # widening iteration reuses it instead of rescanning the text.
        flags_cache: dict[str, int] = {}
        tokens_cache: dict[str, frozenset[str]] = {}
        while True:
            passages = await self.ultrarag.search_bm25(query, requested)
            missing_contents = [
                passage for passage in passages if passage not in loaded_contents
            ]
            if missing_contents:
                by_contents.update(
                    await asyncio.to_thread(
                        lookup.chunks_by_contents,
                        missing_contents,
                    )
                )
                loaded_contents.update(missing_contents)

            ranking: list[str] = []
            used: set[str] = set()
            resolved: dict[str, dict[str, Any]] = {}
            rejected = {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
                "corrupt_text": 0,
                "too_short": 0,
            }
            rejected_examples: dict[str, list[dict[str, Any]]] = {}
            for passage in passages:
                candidates = by_contents.get(passage)
                if not candidates:
                    raise ResearchError(
                        "UltraRAG returned a passage absent from the current chunk store"
                    )
                chunk = next(
                    (
                        item
                        for item in candidates
                        if str(item["chunk_id"]) not in used
                        and self._matches_filters(
                            item,
                            documents_by_id,
                            categories_any=categories_any,
                            keywords=keywords,
                            projects_any=projects_any,
                            languages_any=languages_any,
                            authors_any=authors_any,
                            titles_any=titles_any,
                            document_filter=document_filter,
                            excluded_document_ids=excluded_document_ids,
                            excluded_chunk_ids=excluded_chunk_ids,
                        )
                    ),
                    None,
                )
                if chunk is None:
                    continue
                chunk_id = str(chunk["chunk_id"])
                used.add(chunk_id)
                resolved[chunk_id] = chunk
                flags = flags_cache.get(chunk_id)
                if flags is None:
                    flags = _candidate_flags(chunk)
                    flags_cache[chunk_id] = flags
                if flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                    rejected["extraction_artifact"] += 1
                    continue
                if flags & CHUNK_FLAG_CORRUPT_TEXT:
                    rejected["corrupt_text"] += 1
                    # Reason codes are recomputed only here, because the
                    # response discloses them and they are not stored.
                    _record_withheld(
                        withheld,
                        chunk,
                        text_corruption_reasons(_chunk_text(chunk)),
                        limit=self.config.settings.maximum_withheld_examples,
                    )
                    continue
                tokens = tokens_cache.get(chunk_id)
                if tokens is None:
                    tokens = frozenset(_content_tokens(_chunk_text(chunk), stopwords))
                    tokens_cache[chunk_id] = tokens
                if not query_tokens.intersection(tokens):
                    rejected["no_query_token_overlap"] += 1
                    continue
                if self._passage_too_short(chunk, token_policy=token_policy):
                    rejected["too_short"] += 1
                    self._record_rejected_candidate(
                        rejected_examples,
                        reason="bm25_too_short",
                        chunk=chunk,
                        documents_by_id=documents_by_id,
                    )
                    continue
                ranking.append(chunk_id)
                if len(ranking) == limit:
                    break

            if (
                len(ranking) == limit
                or not filtered
                or requested >= ceiling
                or len(passages) < requested
            ):
                return ranking, rejected, resolved, rejected_examples, requested
            requested = min(ceiling, requested * 2)

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
            (_chunk_text(record) for record in iter_jsonl(chunks_path)),
            stopwords,
        )
        self._document_frequencies = (generation_id, stopwords, frequencies)
        return frequencies

    async def search(  # noqa: C901 — complexity 42; the threshold is not a refactor schedule
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
        retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
        rerank: bool = False,
        rerank_model: str | None = None,
        include_staleness: bool = True,
    ) -> dict[str, Any]:
        """Retrieve evidence.

        The public MCP tool defaults ``rerank`` to true, the largest measured quality
        gain (``MEASUREMENTS.md``); this API keeps the neutral default. An unloadable
        reranker model still returns a search, in unranked candidate order with
        ``rerank_fallback``.

        ``rerank_model`` names a reranker for this call alone, so one process can
        compare models against the same generation.
        """
        query = query.strip()
        if not query:
            raise ResearchError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ResearchError("top_k must be between 1 and 50")
        retrieval_method = retrieval_method.casefold().strip()
        if retrieval_method not in RETRIEVAL_METHODS:
            raise ResearchError("retrieval_method must be one of: bm25, dense, hybrid")
        if rerank_model is not None and not rerank:
            raise ResearchError("rerank_model requires rerank=True")
        applied_reranker = rerank_model or self.config.reranker_model
        try:
            applied_reranker_revision = _reranker_revision(applied_reranker)
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc

        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            passage_token_policy = self._passage_token_policy(manifest)
            gate_stopwords = self.config.settings.gate_stopwords
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            total_chunk_count = await asyncio.to_thread(lookup.chunk_count)
            if not total_chunk_count:
                raise ResearchError("The current generation has no chunks")
            metadata = self._metadata()
            documents_by_id = _effective_documents(manifest, metadata)
            chunks_by_id: dict[str, dict[str, Any]] = {}
            exclusions = self._source_exclusions()
            excluded_document_ids = self._excluded_document_ids(
                manifest,
                exclusions,
            )
            # A chunk id is derived from content, so the same passage keeps it
            # across a rebuild of unchanged bytes: the decision is enforced by
            # every id it names, and which of them this generation holds is
            # reported by the review state rather than guessed here.
            excluded_chunk_ids = set(self._chunk_exclusions())

            retrieval = manifest.get("retrieval", {})
            available_methods = set(retrieval.get("available_methods") or ["bm25"])
            if retrieval_method not in available_methods:
                raise ResearchError(
                    f"Current generation does not support {retrieval_method!r}; "
                    f"available methods: {', '.join(sorted(available_methods))}. "
                    "Run ingest to build a hybrid generation."
                )

            requested_source_ids = _requested_ids(source_ids)
            requested_exclude_source_ids = _requested_ids(exclude_source_ids)
            category_any_filter = _normalized_filter(categories_any)
            project_any_filter = _normalized_filter(projects_any)
            keyword_filter = _normalized_filter(keywords)
            language_any_filter = _normalized_filter(languages_any)
            author_any_filter = _normalized_filter(authors_any)
            title_any_filter = _normalized_filter(titles_any)
            source_include_document_ids, unknown_source_ids = (
                self._document_ids_for_source_ids(manifest, requested_source_ids)
            )
            source_exclude_document_ids, unknown_exclude_source_ids = (
                self._document_ids_for_source_ids(
                    manifest,
                    requested_exclude_source_ids,
                )
            )
            if requested_source_ids and not source_include_document_ids:
                raise ResearchError(
                    "source_ids matched no document in the current generation: "
                    f"{', '.join(unknown_source_ids)}. Use find_source to resolve "
                    "the current IDs; a renamed or moved source receives a new "
                    "source_id."
                )
            # Reviewed exclusions always win over a search-level exclusion, and a
            # search-level include can never re-admit an excluded source.
            excluded_document_ids = excluded_document_ids | source_exclude_document_ids
            document_filter = set(source_include_document_ids)

            dense_document_filter: set[str] | None = None
            metadata_filter_active = bool(
                category_any_filter
                or project_any_filter
                or keyword_filter
                or language_any_filter
                or author_any_filter
                or title_any_filter
            )
            if metadata_filter_active:
                dense_document_filter = {
                    document_id
                    for document_id, document in documents_by_id.items()
                    if _document_matches_metadata(
                        document,
                        keywords=keyword_filter,
                        categories_any=category_any_filter,
                        projects_any=project_any_filter,
                        languages_any=language_any_filter,
                        authors_any=author_any_filter,
                        titles_any=title_any_filter,
                    )
                }
            if document_filter:
                dense_document_filter = (
                    document_filter
                    if dense_document_filter is None
                    else dense_document_filter & document_filter
                )
            active_document_ids = {
                document_id
                for document_id, document in documents_by_id.items()
                if document_id not in excluded_document_ids
                and _document_matches_metadata(
                    document,
                    keywords=keyword_filter,
                    categories_any=category_any_filter,
                    projects_any=project_any_filter,
                    languages_any=language_any_filter,
                    authors_any=author_any_filter,
                    titles_any=title_any_filter,
                )
                and (not document_filter or document_id in document_filter)
            }
            active_chunk_count = await asyncio.to_thread(
                lookup.chunk_count,
                (
                    active_document_ids
                    if metadata_filter_active
                    or document_filter
                    or excluded_document_ids
                    else None
                ),
            )
            # The count above filters by document, so it still holds every
            # excluded chunk inside the active documents. The dense backend
            # filters by document too and cannot be asked for less, so the
            # candidates it will drop are taken off the depth here instead:
            # a post-query drop must not cost the reader a passage.
            if excluded_chunk_ids:
                excluded_chunks = await asyncio.to_thread(
                    lookup.chunks_by_ids,
                    sorted(excluded_chunk_ids),
                )
                active_chunk_count = max(
                    0,
                    active_chunk_count
                    - sum(
                        1
                        for chunk in excluded_chunks.values()
                        if str(chunk["document_id"]) in active_document_ids
                    ),
                )
            candidate_depth = min(
                active_chunk_count,
                self.config.settings.maximum_candidates,
                max(self.config.settings.minimum_candidates, top_k * 4),
            )

            use_bm25 = retrieval_method in {"bm25", "hybrid"}
            use_dense = retrieval_method in {"dense", "hybrid"}
            if use_bm25:
                await self._ensure_loaded(generation_root, manifest)

            bm25_ranking: list[str] = []
            bm25_window = 0
            dense_hits: list[DenseSearchHit] = []
            withheld: dict[str, dict[str, Any]] = {}
            bm25_examples: dict[str, list[dict[str, Any]]] = {}
            bm25_rejected = {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
                "corrupt_text": 0,
                "too_short": 0,
            }

            async def search_dense() -> list[DenseSearchHit]:
                if candidate_depth == 0 or dense_document_filter == set():
                    return []
                return await asyncio.to_thread(
                    self._dense_for(manifest).search,
                    generation_root / manifest["files"]["dense_index"],
                    query,
                    candidate_depth,
                    # Keep Qdrant payloads lean. Translate current reviewed
                    # metadata filters to document IDs at query time so edits
                    # remain exact without rebuilding the dense index.
                    document_ids=sorted(dense_document_filter or []),
                    excluded_document_ids=sorted(excluded_document_ids),
                )

            if use_bm25 and use_dense:
                bm25_result, dense_hits = await asyncio.gather(
                    self._bm25_ranking(
                        query,
                        lookup,
                        total_chunk_count,
                        documents_by_id,
                        candidate_depth,
                        categories_any=category_any_filter,
                        keywords=keyword_filter,
                        projects_any=project_any_filter,
                        languages_any=language_any_filter,
                        authors_any=author_any_filter,
                        titles_any=title_any_filter,
                        document_filter=document_filter,
                        excluded_document_ids=excluded_document_ids,
                        excluded_chunk_ids=excluded_chunk_ids,
                        withheld=withheld,
                        stopwords=gate_stopwords,
                        token_policy=passage_token_policy,
                    ),
                    search_dense(),
                )
                (
                    bm25_ranking,
                    bm25_rejected,
                    bm25_chunks,
                    bm25_examples,
                    bm25_window,
                ) = bm25_result
                chunks_by_id.update(bm25_chunks)
            elif use_bm25:
                (
                    bm25_ranking,
                    bm25_rejected,
                    bm25_chunks,
                    bm25_examples,
                    bm25_window,
                ) = await self._bm25_ranking(
                    query,
                    lookup,
                    total_chunk_count,
                    documents_by_id,
                    candidate_depth,
                    categories_any=category_any_filter,
                    keywords=keyword_filter,
                    projects_any=project_any_filter,
                    languages_any=language_any_filter,
                    authors_any=author_any_filter,
                    titles_any=title_any_filter,
                    document_filter=document_filter,
                    excluded_document_ids=excluded_document_ids,
                    excluded_chunk_ids=excluded_chunk_ids,
                    withheld=withheld,
                    stopwords=gate_stopwords,
                    token_policy=passage_token_policy,
                )
                chunks_by_id.update(bm25_chunks)
            else:
                dense_hits = await search_dense()

            # Pseudo-relevance feedback: the lexical leaders of the first pass
            # name the vocabulary the author actually used, so a question asked
            # in other words can still reach those passages. Terms are mined from
            # the first-pass ranking only, and the second pass replaces the
            # lexical ranking that the fusion and the payload see.
            prf_terms: list[str] = []
            if use_bm25 and bm25_ranking and self.config.settings.prf:
                frequencies = await self._generation_document_frequencies(
                    str(manifest["generation_id"]),
                    generation_root / str(manifest["files"]["chunks"]),
                    gate_stopwords,
                )
                prf_terms = _pseudo_relevance_terms(
                    query=query,
                    texts=[
                        _chunk_text(chunks_by_id[chunk_id])
                        for chunk_id in bm25_ranking[
                            : self.config.settings.prf_documents
                        ]
                        if chunk_id in chunks_by_id
                    ],
                    maximum_terms=self.config.settings.prf_terms,
                    stopwords=gate_stopwords,
                    document_frequencies=frequencies,
                    corpus_size=total_chunk_count,
                )
                if prf_terms:
                    (
                        bm25_ranking,
                        bm25_rejected,
                        bm25_chunks,
                        bm25_examples,
                        prf_window,
                    ) = await self._bm25_ranking(
                        f"{query} {' '.join(prf_terms)}",
                        lookup,
                        total_chunk_count,
                        documents_by_id,
                        candidate_depth,
                        categories_any=category_any_filter,
                        keywords=keyword_filter,
                        projects_any=project_any_filter,
                        languages_any=language_any_filter,
                        authors_any=author_any_filter,
                        titles_any=title_any_filter,
                        document_filter=document_filter,
                        excluded_document_ids=excluded_document_ids,
                        excluded_chunk_ids=excluded_chunk_ids,
                        withheld=withheld,
                        stopwords=gate_stopwords,
                        token_policy=passage_token_policy,
                    )
                    chunks_by_id.update(bm25_chunks)
                    # The feedback pass searched a second window, so the answer
                    # reports the wider of the two rather than the first.
                    bm25_window = max(bm25_window, prf_window)

            dense_chunks = await asyncio.to_thread(
                lookup.chunks_by_ids,
                [hit.chunk_id for hit in dense_hits],
            )
            chunks_by_id.update(dense_chunks)
            accepted_dense_hits: list[DenseSearchHit] = []
            dense_margin = self.config.settings.dense_relative_similarity_margin
            dense_floor = self.config.settings.dense_minimum_cosine_similarity
            dense_below_threshold = 0
            dense_below_threshold_rescued = 0
            dense_quality_rejected = 0
            dense_corrupt_text_rejected = 0
            dense_too_short = 0
            dense_rejected_examples: dict[str, list[dict[str, Any]]] = {}
            # Every non-score filter runs first, so the score decision below sees
            # only candidates the corpus can actually offer.
            eligible_dense_hits: list[tuple[DenseSearchHit, dict[str, Any]]] = []
            for dense_hit in dense_hits:
                chunk = chunks_by_id.get(dense_hit.chunk_id)
                if chunk is None:
                    raise ResearchError(
                        "The dense index returned a chunk absent from the current "
                        f"chunk store: {dense_hit.chunk_id}"
                    )
                dense_flags = _candidate_flags(chunk)
                if dense_flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                    dense_quality_rejected += 1
                    continue
                if dense_flags & CHUNK_FLAG_CORRUPT_TEXT:
                    dense_corrupt_text_rejected += 1
                    _record_withheld(
                        withheld,
                        chunk,
                        text_corruption_reasons(_chunk_text(chunk)),
                        limit=self.config.settings.maximum_withheld_examples,
                    )
                    continue
                if not self._matches_filters(
                    chunk,
                    documents_by_id,
                    categories_any=category_any_filter,
                    keywords=keyword_filter,
                    projects_any=project_any_filter,
                    languages_any=language_any_filter,
                    authors_any=author_any_filter,
                    titles_any=title_any_filter,
                    document_filter=document_filter,
                    excluded_document_ids=excluded_document_ids,
                    excluded_chunk_ids=excluded_chunk_ids,
                ):
                    continue
                if self._passage_too_short(chunk, token_policy=passage_token_policy):
                    dense_too_short += 1
                    self._record_rejected_candidate(
                        dense_rejected_examples,
                        reason="dense_too_short",
                        chunk=chunk,
                        documents_by_id=documents_by_id,
                    )
                    continue
                eligible_dense_hits.append((dense_hit, chunk))
            # The floor is absolute, and a query whose best passage still scores
            # below it can have its whole candidate list packed into a band under
            # it: on the reference corpus a one-word query spanned 0.681 to 0.742
            # across fifty candidates, so a 0.72 floor kept six of them even though
            # the rest matched it about as closely. The margin admits that band, but
            # only once something has cleared the floor, so a query the corpus
            # cannot support still abstains rather than returning its least-bad
            # passage.
            best_dense_score: float | None = None
            for dense_hit, _chunk in eligible_dense_hits:
                if best_dense_score is None or dense_hit.score > best_dense_score:
                    best_dense_score = dense_hit.score
            admission_floor = dense_floor
            if (
                dense_margin > 0.0
                and best_dense_score is not None
                and best_dense_score >= dense_floor
            ):
                admission_floor = min(dense_floor, best_dense_score - dense_margin)
            for dense_hit, chunk in eligible_dense_hits:
                if dense_hit.score >= admission_floor:
                    if dense_hit.score < dense_floor:
                        dense_below_threshold_rescued += 1
                    accepted_dense_hits.append(dense_hit)
                    continue
                dense_below_threshold += 1
                self._record_rejected_candidate(
                    dense_rejected_examples,
                    reason="dense_below_threshold",
                    chunk=chunk,
                    documents_by_id=documents_by_id,
                    cosine_similarity=round(dense_hit.score, 4),
                )
            dense_hits = accepted_dense_hits
            dense_ranking = [hit.chunk_id for hit in dense_hits]
            dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
            bm25_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(bm25_ranking, 1)
            }
            dense_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(dense_ranking, 1)
            }
            if retrieval_method == "hybrid":
                ordered_ids, fusion_scores = self._fuse_rankings(
                    bm25_ranking,
                    dense_ranking,
                    rrf_k=self.config.settings.rrf_k,
                    bm25_weight=self.config.settings.bm25_weight,
                    dense_weight=self.config.settings.dense_weight,
                    maximum_candidates=self.config.settings.maximum_candidates,
                )
            elif retrieval_method == "bm25":
                ordered_ids = bm25_ranking
                fusion_scores = {}
            else:
                ordered_ids = dense_ranking
                fusion_scores = {}

            unknown_ids = [item for item in ordered_ids if item not in chunks_by_id]
            if unknown_ids:
                raise ResearchError(
                    "The retrieval index returned chunk IDs absent from the current "
                    f"chunk store: {unknown_ids[:3]}"
                )

            base_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(ordered_ids, 1)
            }
            rerank_scores: dict[str, float] = {}
            rerank_fallback: dict[str, Any] | None = None
            rerank_count = 0
            if rerank and ordered_ids:
                rerank_count = min(
                    len(ordered_ids),
                    self.config.settings.rerank_max_candidates,
                    max(
                        top_k * self.config.settings.rerank_window_multiple,
                        self.config.settings.rerank_window_floor,
                    ),
                )
                rerank_ids = ordered_ids[:rerank_count]
                rerank_tail = ordered_ids[rerank_count:]
                # A backend that answers tool calls keeps its configured model,
                # so the keyword is passed only when this call names another.
                rerank_kwargs = {"model": rerank_model} if rerank_model else {}
                try:
                    scores = await asyncio.to_thread(
                        self.dense.rerank,
                        query,
                        [_chunk_text(chunks_by_id[item]) for item in rerank_ids],
                        **rerank_kwargs,
                    )
                except RerankerUnavailable as exc:
                    # Requested reranking cannot run without its model, so the
                    # unranked candidate order is returned unchanged and the
                    # response discloses why instead of failing the search.
                    rerank_fallback = {
                        "reason": "reranker_model_unavailable",
                        "message": str(exc),
                        "effect": "unranked_candidate_order_returned",
                    }
                else:
                    rerank_scores = dict(zip(rerank_ids, scores, strict=True))
                    ordered_ids = (
                        sorted(
                            rerank_ids,
                            key=lambda chunk_id: (
                                -rerank_scores[chunk_id],
                                base_ranks[chunk_id],
                                chunk_id,
                            ),
                        )
                        + rerank_tail
                    )
            reranked_applied = bool(rerank_scores)

            # A repeated passage is settled here, on the candidates this search
            # already holds, rather than by dropping a chunk at build time. Two
            # copies of one passage stay in the corpus and independently citable,
            # and the answer shows one of them.
            ordered_ids, collapsed = await self._collapse_repetitions(
                ordered_ids,
                generation_root=generation_root,
                manifest=manifest,
                lookup=lookup,
                chunks_by_id=chunks_by_id,
                documents_by_id=documents_by_id,
            )
            candidate_count = len(ordered_ids)
            source_id_by_chunk = {
                item: str(
                    _document_for_chunk(chunks_by_id[item], documents_by_id)[
                        "source_id"
                    ]
                )
                for item in ordered_ids
            }
            candidate_distinct_reference_count = len(set(source_id_by_chunk.values()))
            selected_ids = _source_diverse_selection(
                ordered_ids,
                source_id_by_chunk=source_id_by_chunk,
                scores=rerank_scores or fusion_scores,
                top_k=top_k,
                penalty=self.config.settings.source_diversity_penalty,
            )

            hits: list[dict[str, Any]] = []
            for rank, chunk_id in enumerate(selected_ids, 1):
                chunk = chunks_by_id[chunk_id]
                document = _document_for_chunk(chunk, documents_by_id)
                component_ranks = {
                    "bm25": bm25_ranks.get(chunk_id),
                    "dense": dense_ranks.get(chunk_id),
                }
                if (
                    component_ranks["bm25"] is not None
                    and component_ranks["dense"] is not None
                ):
                    match_kind = "hybrid"
                elif component_ranks["bm25"] is not None:
                    match_kind = "lexical"
                else:
                    match_kind = "semantic"
                hits.append(
                    {
                        "rank": rank,
                        "retrieval_rank": base_ranks[chunk_id],
                        **_public_passage(chunk, document),
                        "match_kind": match_kind,
                        "retrieval_method": retrieval_method,
                        "component_ranks": component_ranks,
                        "component_scores": {
                            "dense_cosine_similarity": dense_scores.get(chunk_id),
                            "bm25": None,
                        },
                        "fusion_score": fusion_scores.get(chunk_id),
                        "rerank_score": rerank_scores.get(chunk_id),
                    }
                )

            distinct_reference_count = len({str(hit["source_id"]) for hit in hits})
            relevance_limited = candidate_count < top_k
            repetition_disclosure: dict[str, Any] = {}
            if collapsed:
                # A lean answer states the rule once rather than repeating it per
                # passage, so this is a count, and the pairs are for the caller
                # that asks for the full detail: the actionable part of an
                # overlap is which two files it came from.
                by_words = sum(
                    1 for item in collapsed if item["collapsed_by"] == "same_words"
                )
                by_meaning = len(collapsed) - by_words
                repetition_disclosure = {
                    "repetitions_collapsed": len(collapsed),
                    "collapsed_by_same_words": by_words,
                    "collapsed_by_same_meaning": by_meaning,
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
                upgrade_reasons = self._generation_upgrade_reasons(manifest)
            return {
                "query": query,
                "generation_id": manifest["generation_id"],
                "stale": stale,
                "staleness_checked": include_staleness,
                "generation_upgrade_required": bool(upgrade_reasons),
                "excluded_source_count": len(exclusions),
                "excluded_chunk_count": len(excluded_chunk_ids),
                "filters": {
                    "categories_any": sorted(category_any_filter),
                    "projects_any": sorted(project_any_filter),
                    "keywords_all": sorted(keyword_filter),
                    "languages_any": sorted(language_any_filter),
                    "authors_any": sorted(author_any_filter),
                    "titles_any": sorted(title_any_filter),
                    "source_document_ids": sorted(document_filter),
                    "source_ids": requested_source_ids,
                    "exclude_source_ids": requested_exclude_source_ids,
                    "unknown_source_ids": unknown_source_ids,
                    "unknown_exclude_source_ids": unknown_exclude_source_ids,
                    "active_document_count": len(active_document_ids),
                    "corpus_chunk_count": total_chunk_count,
                    "window_chunk_count": bm25_window or candidate_depth,
                    "window_is_whole_corpus": (bm25_window or candidate_depth)
                    >= total_chunk_count,
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
                "retrieval_method": retrieval_method,
                "reranked": reranked_applied,
                "rerank_requested": rerank,
                "rerank_fallback": rerank_fallback,
                "rerank_window": rerank_count,
                "prf_requested": self.config.settings.prf,
                "prf_terms": prf_terms,
                "candidate_depth": candidate_depth,
                "candidate_count": candidate_count,
                "candidate_distinct_reference_count": (
                    candidate_distinct_reference_count
                ),
                "requested_top_k": top_k,
                "fusion": (
                    {
                        "method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": self.config.settings.rrf_k,
                        "bm25_weight": self.config.settings.bm25_weight,
                        "dense_weight": self.config.settings.dense_weight,
                    }
                    if retrieval_method == "hybrid"
                    else None
                ),
                "relevance_policy": {
                    "bm25_requires_query_token_overlap": True,
                    "dense_minimum_cosine_similarity": (
                        self.config.settings.dense_minimum_cosine_similarity
                    ),
                },
                "selection_policy": {
                    "method": "greedy_source_diversity",
                    "source_diversity_penalty": (
                        self.config.settings.source_diversity_penalty
                    ),
                },
                "dense_gate": {
                    "minimum_cosine_similarity": dense_floor,
                    "relative_margin": dense_margin,
                    "best_cosine_similarity": (
                        None if best_dense_score is None else round(best_dense_score, 4)
                    ),
                    "admitted_below_floor": dense_below_threshold_rescued,
                    "rejected_below_floor": dense_below_threshold,
                    "note": (
                        "Dense candidates are admitted by score. A candidate below "
                        "the floor is admitted when the query's best candidate "
                        "cleared the floor and this one is within the margin of it; "
                        "when nothing clears the floor the query is left with "
                        "nothing rather than its least-bad passage. A query whose "
                        "whole candidate list sits in a narrow band under the floor "
                        "is why the margin exists."
                    ),
                },
                "passage_length_policy": {
                    "minimum_words": self.config.settings.minimum_passage_words,
                    "minimum_tokens": passage_token_policy["minimum_tokens"],
                    "token_fraction": self.config.settings.minimum_passage_token_fraction,
                    "chunk_size": passage_token_policy["chunk_size"],
                    "tokenizer": passage_token_policy["tokenizer"],
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
                "rejected_candidates": {
                    "bm25_no_query_token_overlap": bm25_rejected[
                        "no_query_token_overlap"
                    ],
                    "bm25_extraction_artifact": bm25_rejected["extraction_artifact"],
                    "bm25_corrupt_text": bm25_rejected["corrupt_text"],
                    "bm25_too_short": bm25_rejected["too_short"],
                    "dense_below_threshold": dense_below_threshold,
                    "dense_extraction_artifact": dense_quality_rejected,
                    "dense_corrupt_text": dense_corrupt_text_rejected,
                    "dense_too_short": dense_too_short,
                },
                "withheld_candidates": {
                    "policy": "corruption_evidence_only",
                    "note": (
                        "Candidates withheld from this result for corruption "
                        "evidence. Script notes such as non_latin_dominant or "
                        "mixed_script_text appear per hit in text_notes and never "
                        "withhold a passage."
                    ),
                    "total": sum(int(entry["count"]) for entry in withheld.values()),
                    "reasons": {
                        reason: {
                            "count": int(entry["count"]),
                            "example_chunk_ids": list(entry["example_chunk_ids"]),
                        }
                        for reason, entry in sorted(withheld.items())
                    },
                    "flagged_passages_returned": sum(
                        1 for hit in hits if hit.get("text_notes")
                    ),
                },
                "collapsed_repetitions": {
                    **repetition_disclosure,
                    "pairs": collapsed,
                },
                "rejected_candidate_examples": {
                    "policy": "bounded_examples",
                    "limit_per_reason": self.config.settings.maximum_withheld_examples,
                    "note": (
                        "The gate's count says how many candidates it removed; "
                        "these name the sources they came from, so a thin answer "
                        "can be read as the corpus being thinned rather than "
                        "silent."
                    ),
                    "reasons": {
                        "bm25_too_short": bm25_examples.get("bm25_too_short", []),
                        "dense_below_threshold": dense_rejected_examples.get(
                            "dense_below_threshold", []
                        ),
                        "dense_too_short": dense_rejected_examples.get(
                            "dense_too_short", []
                        ),
                    },
                },
                "dense_fidelity": {
                    "embedding_maximum_tokens": self.config.settings.embedding_maximum_tokens,
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
                "embedding_model": self.config.settings.embedding_model
                if use_dense
                else None,
                "embedding_model_revision": (
                    self.config.settings.embedding_model_revision if use_dense else None
                ),
                "reranker_model": applied_reranker if reranked_applied else None,
                "reranker_model_revision": (
                    applied_reranker_revision if reranked_applied else None
                ),
                "result_count": len(hits),
                "distinct_reference_count": distinct_reference_count,
                "relevance_limited": relevance_limited,
                "hits": hits,
            }

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
        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            documents_by_id = _effective_documents(manifest, self._metadata())
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
            if chunk_id in self._chunk_exclusions():
                raise ResearchError(
                    "The requested chunk is currently excluded from retrieval; "
                    "include the chunk before requesting its passage"
                )
            if _is_extraction_artifact(target):
                raise ResearchError(
                    "The requested chunk is an extraction artifact and is not "
                    "available; re-ingest to remove it from the generation"
                )
            if text_corruption_reasons(_chunk_text(target)):
                raise ResearchError(
                    "The requested chunk contains corrupt extracted text and is "
                    "not available; re-ingest to remove it from the generation"
                )
            document_chunks = await asyncio.to_thread(
                lookup.chunks_for_document,
                str(target["document_id"]),
            )
            same_document = [
                item
                for item in document_chunks
                if not _is_extraction_artifact(item)
                and not text_corruption_reasons(_chunk_text(item))
            ]
            target_position = next(
                index
                for index, item in enumerate(same_document)
                if item["chunk_id"] == chunk_id
            )
            start = max(0, target_position - context_chunks)
            end = min(len(same_document), target_position + context_chunks + 1)
            context = []
            document = _document_for_chunk(target, documents_by_id)
            for item in same_document[start:end]:
                context.append(_public_passage(item, document))
            return {
                "generation_id": manifest["generation_id"],
                "requested_chunk_id": chunk_id,
                "context": context,
            }
