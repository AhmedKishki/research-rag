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

from ..generations.generation_inventory import generation_inventory
from ..project.policy import ResearchError
from ..project.state_files import SEARCH_STATS_FILE
from ..project.support import _effective_documents
from ..storage.search_stats import (
    TRACKED_RANKS,
    SearchStatsError,
    read_search_stats,
    record_search,
)

LOGGER = logging.getLogger(__name__)

# How many sources and passages each ranked list in the answer names.
TOP_ENTRIES = 20
# How many sources the answer names as never having reached a top five.
UNREACHED_ENTRIES = 20


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
    ) -> None:
        """Count one answered search. A failure is logged, never raised."""

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
            )
        except (SearchStatsError, OSError) as exc:
            LOGGER.warning("search counts not recorded: %s", exc)

    async def search_stats(self) -> dict[str, Any]:
        """Search counts by rank, and the facts the selected generation carries."""

        async with self._read() as lease:
            current = self._load_current_optional()
            try:
                counts = await asyncio.to_thread(
                    read_search_stats, self._search_stats_path, top=TOP_ENTRIES
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
                )[:UNREACHED_ENTRIES]
            ]
            if counts["search_count"]
            else []
        )
        answer["corpus"] = self._corpus_facts(searchable, passages_by_document)
        answer["last_build"] = self._build_facts(manifest)
        return answer

    @staticmethod
    def _source_label(document: dict[str, Any] | None) -> dict[str, Any]:
        if document is None:
            return {"title": None, "source_relative_path": None, "in_corpus": False}
        return {
            "title": document.get("title") or document.get("source_relative_path"),
            "source_relative_path": document.get("source_relative_path"),
            "in_corpus": True,
        }

    @staticmethod
    def _corpus_facts(
        documents: list[dict[str, Any]],
        passages_by_document: dict[str, int],
    ) -> dict[str, Any]:
        """Composition the manifest already records, counted over searchable sources."""

        formats: Counter[str] = Counter()
        languages: Counter[str] = Counter()
        decades: Counter[str] = Counter()
        missing: Counter[str] = Counter()
        pages = 0
        sizes = []
        for document in documents:
            formats[str(document.get("format") or "unknown")] += 1
            for language in document.get("language") or ["unknown"]:
                languages[str(language)] += 1
            decade = _decade(document.get("year"))
            decades[decade or "no year"] += 1
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
        largest = sorted(sizes, key=lambda item: -item[0])[:5]
        return {
            "source_count": len(documents),
            "passage_count": sum(counts),
            "pdf_page_count": pages,
            "passages_per_source": {
                "minimum": counts[0] if counts else None,
                "median": statistics.median(counts) if counts else None,
                "maximum": counts[-1] if counts else None,
            },
            "largest_sources": [
                {"passage_count": count, **StatsWorkflow._source_label(document)}
                for count, document in largest
            ],
            "formats": _counted(formats),
            "languages": _counted(languages),
            "decades": _counted(decades),
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
