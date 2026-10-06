"""What the corpus answered with, and what it holds.

`StatsWorkflow` records each search's first five ranks into the machine-local
counts in `storage/search_stats.py`, and answers `search_stats` from those counts
and from facts the selected generation already carries: its documents, their
reviewed metadata, its build metrics, and its passage index. Nothing here runs a
search, reads a source file, or writes anything but the counts.

A search a measurement runs is not counted: `evaluation_trace` marks one, and a
service built with `record_searches=False` counts none.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..corpus.text_normalization import normalize_reading_text
from ..generations.generation_inventory import generation_inventory
from ..project.policy import ResearchError
from ..project.state_files import SEARCH_STATS_FILE
from ..project.support import (
    _chunk_text,
    _effective_documents,
    _public_document,
)
from ..storage.search_stats import (
    TRACKED_RANKS,
    SearchStatsError,
    clear_history,
    read_appearances,
    read_history,
    read_search_stats,
    record_search,
)
from .admission import current_caller

LOGGER = logging.getLogger(__name__)

# What the largest sources are ranked by.
LARGEST_BY = ("passages", "size")

# How much of a passage a list of them shows; the whole is `get_passage`.
PREVIEW_CHARACTERS = 420

# How many sources and passages each ranked list in the answer names.
TOP_ENTRIES = 20


def caller_kind(caller: str) -> str:
    """What kind of caller a fairness key names, for a list a reader scans."""

    if caller in {"workspace", "command line"}:
        return caller
    return "unknown" if caller in {"", "anonymous"} else "agent"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _decade(year: Any) -> str | None:
    try:
        value = int(year)
    except (TypeError, ValueError):
        return None
    return f"{value // 10 * 10}s" if 1 <= value <= 9999 else None


def _counted(values: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ]


class StatsWorkflow:
    @property
    def _search_stats_path(self) -> Path:
        return self.config.state_root / SEARCH_STATS_FILE

    async def _record_search(
        self,
        payload: dict[str, Any],
        *,
        requested_top_k: int,
        started: float,
        query: str = "",
        filters: dict[str, list[str]] | None = None,
    ) -> None:
        """Count one answered search. A failure is logged, never raised.

        The question and its filters are kept only while `runtime.search_history`
        is on, and only under the kind of caller that asked.
        """

        if not self._records_searches:
            return
        ranked = [
            (int(hit["rank"]), str(hit["chunk_id"]), str(hit.get("source_id") or ""))
            for hit in payload.get("hits", [])
            if int(hit.get("rank") or 0) <= TRACKED_RANKS
        ]
        try:
            await asyncio.to_thread(
                record_search,
                self._search_stats_path,
                searched_at=_utc_now(),
                generation_id=str(payload.get("generation_id") or ""),
                requested_top_k=requested_top_k,
                result_count=int(payload.get("result_count") or 0),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                ranked=ranked,
                query=query if self.config.settings.search_history else None,
                filters=(
                    {key: value for key, value in (filters or {}).items() if value}
                    if self.config.settings.search_history
                    else None
                ),
                caller=caller_kind(current_caller()),
            )
        except (SearchStatsError, OSError) as exc:
            LOGGER.warning("search counts not recorded: %s", exc)

    async def search_stats(
        self,
        *,
        since_days: float | None = None,
        top: int = TOP_ENTRIES,
        largest_by: str = "passages",
    ) -> dict[str, Any]:
        """Search counts by rank, and the facts the selected generation carries.

        `since_days` bounds the counts to recent searches and `top` the length of
        the ranked lists, including the largest sources, which `largest_by` ranks by
        `passages` or `size` (the extracted text, which every format has). The corpus and build facts are the generation's own
        and do not move with the time window.
        """

        if largest_by not in LARGEST_BY:
            raise ResearchError(f"largest_by must be one of: {', '.join(LARGEST_BY)}")
        if since_days is not None and since_days <= 0:
            raise ResearchError("since_days must be a positive number of days")
        if not 1 <= top <= 100:
            raise ResearchError("top must be between 1 and 100")

        async with self._read() as lease:
            current = self._load_current_optional()
            try:
                counts = await asyncio.to_thread(
                    read_search_stats,
                    self._search_stats_path,
                    top=top,
                    days=None,
                    since_days=since_days,
                )
            except SearchStatsError as exc:
                raise ResearchError(str(exc)) from exc
            answer: dict[str, Any] = {
                "counts_path": str(self._search_stats_path),
                "tracked_ranks": TRACKED_RANKS,
                "searches": {
                    key: counts[key]
                    for key in (
                        "search_count",
                        "zero_result_count",
                        "mean_result_count",
                        "first_search_at",
                        "last_search_at",
                        "median_elapsed_ms",
                        "p95_elapsed_ms",
                        "searches_by_day",
                    )
                },
            }
            builds = await asyncio.to_thread(
                generation_inventory,
                self.config,
                None if current is None else str(current[1].get("generation_id")),
            )
            answer["generations"] = {
                "count": builds.get("retained_generation_count", 0),
                "bytes": builds.get("retained_generation_bytes", 0),
            }
            if current is None:
                answer.update(
                    ready=False,
                    sources=[],
                    passages=[],
                    unreached_source_count=0,
                    unreached_sources=[],
                    corpus=None,
                    last_build=None,
                )
                return answer
            generation_root, manifest = current
            lease.hold(str(manifest["generation_id"]))
            documents = _effective_documents(manifest, self._metadata())
            excluded = set(self._source_exclusions())
            by_source = {
                str(document.get("source_id")): document
                for document in documents.values()
            }
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            passages_by_document = await asyncio.to_thread(
                lookup.chunk_counts_by_document
            )
            text_bytes = await asyncio.to_thread(lookup.text_bytes_by_document)
            ranked_chunks = await asyncio.to_thread(
                lookup.chunks_by_ids,
                [entry["chunk_id"] for entry in counts["passages"]],
            )

        answer["ready"] = True
        answer["generation_id"] = manifest["generation_id"]
        answer["sources"] = [
            {**entry, **self._source_label(by_source.get(entry["source_id"]))}
            for entry in counts["sources"]
        ]
        answer["passages"] = []
        for entry in counts["passages"]:
            chunk = ranked_chunks.get(entry["chunk_id"])
            answer["passages"].append(
                {
                    **entry,
                    **self._source_label(by_source.get(entry["source_id"])),
                    "in_current_generation": chunk is not None,
                    "locator": dict(chunk.get("locator") or {}) if chunk else None,
                }
            )
        appeared = set(counts["appeared_source_ids"])
        searchable = [
            document
            for document in documents.values()
            if str(document.get("source_relative_path")) not in excluded
        ]
        unreached = [
            document
            for document in searchable
            if str(document.get("source_id")) not in appeared
        ]
        answer["unreached_source_count"] = (
            len(unreached) if counts["search_count"] else 0
        )
        answer["unreached_sources"] = (
            [
                self._source_label(document)
                for document in sorted(
                    unreached,
                    key=lambda item: str(item.get("title") or "").casefold(),
                )[:top]
            ]
            if counts["search_count"]
            else []
        )
        answer["corpus"] = self._corpus_facts(
            searchable,
            passages_by_document,
            text_bytes=text_bytes,
            top=top,
            largest_by=largest_by,
        )
        answer["last_build"] = self._build_facts(manifest)
        return answer

    async def search_history(
        self, *, limit: int = 20, since_days: float | None = None
    ) -> dict[str, Any]:
        """Recent searches that kept their question, newest first, to run again."""

        if not 1 <= limit <= 200:
            raise ResearchError("limit must be between 1 and 200")
        try:
            history = await asyncio.to_thread(
                read_history,
                self._search_stats_path,
                limit=limit,
                since_days=since_days,
            )
        except SearchStatsError as exc:
            raise ResearchError(str(exc)) from exc
        history["recording"] = bool(self.config.settings.search_history)
        return history

    async def clear_search_history(self) -> dict[str, Any]:
        """Forget every kept question and filter. The counts stay."""

        async with self._operation():
            try:
                cleared = await asyncio.to_thread(
                    clear_history, self._search_stats_path
                )
            except SearchStatsError as exc:
                raise ResearchError(str(exc)) from exc
        return {
            "cleared": cleared,
            "message": (
                f"Forgot {cleared} kept question{'' if cleared == 1 else 's'}. "
                "The counts are unchanged."
            ),
        }

    async def source_chunks(
        self,
        *,
        source_id: str | None = None,
        source_path: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """One source and a page of its passages, each with how often it was returned.

        The source is named by its stable id or its path relative to the sources
        directory. Passages come in reading order from the selected generation's
        index, with the cleaned text trimmed for a list; the whole passage and its
        neighbours are `get_passage`.
        """

        if not (source_id or source_path):
            raise ResearchError("Name the source by source_id or source_path")
        if page < 1 or not 1 <= page_size <= 50:
            raise ResearchError("page must be 1 or more, and page_size 1 to 50")
        async with self._read() as lease:
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lease.hold(str(manifest["generation_id"]))
            documents = _effective_documents(manifest, self._metadata())
            document = next(
                (
                    item
                    for item in documents.values()
                    if (source_id and str(item.get("source_id")) == source_id)
                    or (
                        source_path
                        and str(item.get("source_relative_path")) == source_path
                    )
                ),
                None,
            )
            if document is None:
                raise ResearchError(
                    "The selected generation holds no such source; it may have been "
                    "added since the build, or renamed."
                )
            document_id = str(document["document_id"])
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            total = await asyncio.to_thread(lookup.chunk_count, {document_id})
            pages = max(1, -(-total // page_size))
            page = min(page, pages)
            offset = (page - 1) * page_size
            chunks = await asyncio.to_thread(
                lookup.chunks_for_document_page,
                document_id,
                offset=offset,
                limit=page_size,
            )
            excluded_chunks = set(self._chunk_exclusions())
            excluded_source = str(document.get("source_relative_path")) in set(
                self._source_exclusions()
            )
        appearances = await asyncio.to_thread(
            read_appearances,
            self._search_stats_path,
            chunk_ids=[str(chunk["chunk_id"]) for chunk in chunks],
            source_id=str(document["source_id"]),
        )
        public = _public_document(document)
        stored = next(
            (
                item
                for item in manifest.get("documents", [])
                if str(item.get("document_id")) == document_id
            ),
            {},
        )
        top_five, rank_one = appearances.get(str(document["source_id"]), (0, 0))
        entries = []
        for position, chunk in enumerate(chunks):
            chunk_id = str(chunk["chunk_id"])
            text = _chunk_text(chunk)
            counted = appearances.get(chunk_id, (0, 0))
            entries.append(
                {
                    "chunk_id": chunk_id,
                    "ordinal": offset + position + 1,
                    "locator": dict(chunk.get("locator") or {}),
                    "text": normalize_reading_text(text[:PREVIEW_CHARACTERS]),
                    "truncated": len(text) > PREVIEW_CHARACTERS,
                    "characters": len(text),
                    "embedding_token_count": chunk.get("embedding_token_count"),
                    "dense_truncated": chunk.get("dense_truncated"),
                    "content_kind": str(chunk.get("content_kind") or "prose"),
                    "quality_flags": list(chunk.get("quality_flags") or []),
                    "excluded": chunk_id in excluded_chunks,
                    "top_five": counted[0],
                    "rank_one": counted[1],
                }
            )
        return {
            "generation_id": manifest["generation_id"],
            "source": {
                "source_id": public["source_id"],
                "title": public["title"],
                "authors": public["authors"],
                "year": public.get("year"),
                "doi": public["doi"],
                "language": public["language"],
                "categories": public["categories"],
                "keywords": public["keywords"],
                "project": public["project"],
                "source_path": public["source_path"],
                "source_relative_path": public.get("source_relative_path"),
                "format": stored.get("format"),
                "excluded": excluded_source,
                "passage_count": total,
                "physical_pages": stored.get("physical_pages"),
                "extracted_units": stored.get("extracted_units"),
                "withheld_units": stored.get("excluded_corrupt_unit_count"),
                "unclean_character_rate": stored.get("unclean_character_rate"),
                "metadata_warnings": list(stored.get("metadata_warnings") or []),
                "top_five": top_five,
                "rank_one": rank_one,
            },
            "page": page,
            "pages": pages,
            "page_size": page_size,
            "chunks": entries,
        }

    @staticmethod
    def _source_label(document: dict[str, Any] | None) -> dict[str, Any]:
        if document is None:
            return {"title": None, "source_relative_path": None, "in_corpus": False}
        return {
            "source_id": document.get("source_id"),
            "title": document.get("title") or document.get("source_relative_path"),
            "source_relative_path": document.get("source_relative_path"),
            "in_corpus": True,
        }

    @staticmethod
    def _corpus_facts(
        documents: list[dict[str, Any]],
        passages_by_document: dict[str, int],
        *,
        text_bytes: dict[str, int] | None = None,
        top: int = TOP_ENTRIES,
        largest_by: str = "passages",
    ) -> dict[str, Any]:
        """Composition the manifest already records, counted over searchable sources.

        Every distribution is complete and a reader's card shortens it; only the
        largest sources are cut here, because ranking them is the question asked.
        """

        formats: Counter[str] = Counter()
        languages: Counter[str] = Counter()
        decades: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        authors: Counter[str] = Counter()
        missing: Counter[str] = Counter()
        pages = 0
        sizes = []
        for document in documents:
            formats[str(document.get("format") or "unknown")] += 1
            for language in document.get("language") or ["unknown"]:
                languages[str(language)] += 1
            decade = _decade(document.get("year"))
            decades[decade or "no year"] += 1
            for category in document.get("categories") or []:
                categories[str(category)] += 1
            for author in document.get("authors") or []:
                authors[str(author)] += 1
            for field in ("authors", "year", "categories"):
                if not document.get(field):
                    missing[field] += 1
            pages += int(document.get("physical_pages") or 0)
            sizes.append(
                (
                    passages_by_document.get(str(document.get("document_id")), 0),
                    document,
                )
            )
        counts = sorted(count for count, _ in sizes)
        sizes_in_bytes = text_bytes or {}

        def stored(item: tuple[int, dict[str, Any]]) -> int:
            return sizes_in_bytes.get(str(item[1].get("document_id")), 0)

        if largest_by == "size":
            ranked = sorted(sizes, key=lambda item: (-stored(item), -item[0]))
        else:
            ranked = sorted(sizes, key=lambda item: -item[0])
        largest = ranked[:top]
        return {
            "source_count": len(documents),
            "passage_count": sum(counts),
            "pdf_page_count": pages,
            "passages_per_source": {
                "minimum": counts[0] if counts else None,
                "median": statistics.median(counts) if counts else None,
                "maximum": counts[-1] if counts else None,
            },
            "largest_by": largest_by,
            "largest_sources": [
                {
                    "passage_count": count,
                    "text_bytes": stored((count, document)),
                    "physical_pages": document.get("physical_pages"),
                    **StatsWorkflow._source_label(document),
                }
                for count, document in largest
            ],
            "formats": _counted(formats),
            "languages": _counted(languages),
            "decades": _counted(decades),
            "categories": _counted(categories),
            "authors": _counted(authors),
            "missing_metadata": {
                field: missing.get(field, 0)
                for field in ("authors", "year", "categories")
            },
        }

    @staticmethod
    def _build_facts(manifest: dict[str, Any]) -> dict[str, Any]:
        """The build that wrote the selected generation, from its own manifest."""

        metrics = manifest.get("build_metrics") or {}
        phases = metrics.get("phase_timings_seconds") or {}
        return {
            "created_at": manifest.get("created_at"),
            "seconds": round(sum(float(value) for value in phases.values()), 1)
            if phases
            else None,
            "phase_seconds": {
                str(name): round(float(value), 1)
                for name, value in sorted(
                    phases.items(), key=lambda item: -float(item[1])
                )
            },
            "reused_vector_count": metrics.get("reused_vector_count"),
            "created_vector_count": metrics.get("created_vector_count"),
            "reused_document_count": metrics.get("reused_document_count"),
            "rebuilt_document_count": metrics.get("rebuilt_document_count"),
            "excluded_corrupt_unit_count": metrics.get("excluded_corrupt_unit_count"),
            "dense_truncated_chunk_count": metrics.get("dense_truncated_chunk_count"),
        }
