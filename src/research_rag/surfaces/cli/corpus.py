"""Corpus commands: the app's operations, answered by the running app over
the control API or in this process when no app is up, and the progress a
build reports while it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import os
import shutil
import sys
import time
import unicodedata
from collections.abc import AsyncIterator
from contextlib import (
    asynccontextmanager,
)
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    # The service is reached when a command needs a corpus operation. A command
    # that only reports -- `--version`, `help`, `config`, `doctor`, `install`, and
    # `update` -- must not import the retrieval stack to get there, which is the
    # rule `AGENTS.md` states.
    from ...core.service import ResearchService

from ...core.operations import run
from ...project.config import (
    ResearchConfig,
)
from ...project.policy import ResearchError
from ...runtime.control import Control, connect


@asynccontextmanager
async def _service(config: ResearchConfig) -> AsyncIterator[ResearchService]:
    """Yield the service over the lazy, in-process compatible retrieval backend."""

    from ...core.service import ResearchService
    from ...retrieval.direct import DirectRetrieval

    backend = DirectRetrieval(config)
    try:
        yield ResearchService(config, backend)
    finally:
        await backend.aclose()


def _metadata_body(args: argparse.Namespace) -> dict[str, Any]:
    """Build the review body, refusing a request that asks for two things at once."""

    supplied: dict[str, Any] = {
        "title": args.title,
        "authors": args.author,
        "year": args.year,
        "doi": args.doi,
        "language": args.language,
        "categories": args.category,
        "keywords": args.keyword,
        "project": args.project,
    }
    if args.clear and any(value is not None for value in supplied.values()):
        raise ResearchError(
            "--clear cannot be combined with a field: it removes the review so "
            "automatic metadata applies again."
        )
    if args.clear:
        return {}
    return {key: value for key, value in supplied.items() if value is not None}


class Local:
    """Corpus operations answered in this process, for a project with no app up.

    When the app *is* up, the same command goes to it, so there is never a second
    service holding the project.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._context = _service(config)
        self.service: ResearchService | None = None

    async def __aenter__(self) -> Self:
        self.service = await self._context.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._context.__aexit__(*exc)

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        if self.service is None:
            raise ResearchError("The local service is not open")
        return await run(self.service, name, arguments, command_line=True)


class Remote:
    """The running app's operations, awaited.

    `Control` speaks HTTP and is synchronous, so each call runs in a thread and a
    build's progress can be read while the build is outstanding.
    """

    def __init__(self, control: Control) -> None:
        self.control = control

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.control.close()

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        # The running app answers a search with the method and reranking the
        # engine chooses; `TODO.md` holds the decision about the two flags.
        arguments.pop("retrieval_method", None)
        arguments.pop("rerank", None)
        return await asyncio.to_thread(self.control.operation, name, arguments)


INGEST_PROGRESS_INTERVAL = 2.0


INGEST_HEARTBEAT_INTERVAL = 15.0


PASSAGE_ETA_PHASES = frozenset({"embedding", "dense_indexing", "qdrant_indexing"})


class _PhaseETA:
    """Measure current-phase throughput, with bounded display updates."""

    def __init__(self) -> None:
        self.key: tuple[Any, ...] | None = None
        self.start = 0.0
        self.initial = 0
        self.previous = 0
        self.advances = 0
        self.last_progress = 0.0
        self.displayed: float | None = None
        self.display_time = 0.0
        self.stalled = False
        self.sources = False
        self.activity: tuple[Any, ...] | None = None

    def observe(self, progress: dict[str, Any], now: float) -> float | None:
        activity = _ingestion_progress_key({"ingestion_progress": progress})
        if activity != self.activity:
            self.activity = activity
            self.last_progress = now
        self.sources = progress.get("phase") in {"extraction", "chunking"}
        counts = progress.get("overall_progress" if self.sources else "progress")
        if (
            (not self.sources and progress.get("phase") not in PASSAGE_ETA_PHASES)
            or not isinstance(counts, dict)
            or counts.get("unit") != ("sources" if self.sources else "chunks")
            or not math.isfinite(now)
        ):
            self.key = None
            return None
        completed, total = counts.get("completed"), counts.get("total")
        if (
            type(completed) is not int
            or type(total) is not int
            or not 0 <= completed <= total
            or total <= 0
        ):
            self.key = None
            return None
        key = (
            progress.get("build_id"),
            progress.get("phase"),
            total,
            counts.get("unit"),
        )
        if key != self.key or completed < self.previous or now < self.start:
            self.key = key
            self.start = now
            self.initial = completed
            self.previous = completed
            self.advances = 0
            self.last_progress = now
            self.displayed = None
            self.display_time = now
            self.stalled = False
            return None
        if completed > self.previous:
            self.advances += 1
            self.last_progress = now
        self.previous = completed
        self.stalled = now - self.last_progress >= 60
        sampled = completed - self.initial >= 3 if self.sources else self.advances >= 2
        if not sampled or now <= self.start or completed == total:
            self.displayed = None
            return None
        return (total - completed) * (now - self.start) / (completed - self.initial)

    def label(self, estimate: float | None, now: float) -> str:
        if estimate is None:
            return "ETA —"
        if self.sources:
            # Power-of-two minute ranges are intentionally coarse for unequal books.
            upper = 2 ** max(1, math.ceil(math.log2(max(1, estimate / 60))))
            candidate = float(upper * 60)
        else:
            bucket = 60 if estimate >= 60 else 10
            candidate = float(max(bucket, math.ceil(estimate / bucket) * bucket))
        if self.displayed is None or (
            now - self.display_time >= 60
            and (candidate >= self.displayed * 1.5 or candidate <= self.displayed / 1.5)
        ):
            self.displayed = candidate
            self.display_time = now
        minutes = int(self.displayed / 60)
        if self.sources:
            return f"ETA {minutes // 2}–{minutes}m"
        return f"ETA ~{minutes}m" if minutes else f"ETA ~{int(self.displayed)}s"


def _ingestion_progress_line(
    status: dict[str, Any], estimator: _PhaseETA | None = None
) -> str | None:
    """Render a compact human-readable snapshot, not the backend payload."""
    progress = status.get("ingestion_progress")
    if not isinstance(progress, dict) or not progress:
        if estimator is not None:
            estimator.key = None
        return None
    now = time.monotonic()
    estimate = estimator.observe(progress, now) if estimator is not None else None
    phase = str(progress.get("phase") or "unknown")
    labels = {
        "source_hashing": "Hashing sources",
        "source_revalidation": "Rechecking sources",
        "dense_indexing": "Building dense index",
        "qdrant_indexing": "Building dense index",
        "bm25_indexing": "Building search index",
    }
    fields = [labels.get(phase, phase.replace("_", " ").capitalize())]

    def count(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        completed, total = value.get("completed"), value.get("total")
        if type(completed) is not int or type(total) is not int or total <= 0:
            return None
        unit = str(value.get("unit") or "items")
        if unit == "phase":
            return None
        units = {
            "pdf_page_batches_or_epub_sections": "extraction batches/sections",
            "extraction_units": "extraction units",
        }.get(unit, unit.replace("_", " "))
        return f"{completed:,}/{total:,} {units}"

    overall = progress.get("overall_progress")
    aggregate = count(overall) or count(progress.get("progress"))
    if aggregate:
        fields.append(aggregate)
    source = progress.get("source")
    if isinstance(source, str) and source:
        # Source names are data, not terminal commands or multiline log records.
        safe_source = "".join(char if char.isprintable() else " " for char in source)
        fields.append(safe_source)
    # Backend/source deadlines cannot describe measured global passage work.
    counts = progress.get("progress")
    passage_counts = (
        phase in PASSAGE_ETA_PHASES
        and isinstance(counts, dict)
        and counts.get("unit") == "chunks"
        and type(counts.get("completed")) is int
        and type(counts.get("total")) is int
        and 0 <= counts["completed"] <= counts["total"]
        and counts["total"] > 0
    )
    fields.append(estimator.label(estimate, now) if estimator else "ETA —")
    if passage_counts and counts["completed"] == counts["total"]:
        fields.append("Finalizing")
    if estimator is not None and estimator.stalled and estimator.key is not None:
        fields.append("Stalled: no progress for ≥60s")
    return " | ".join(fields)


def _ingestion_progress_key(status: dict[str, Any]) -> tuple[Any, ...] | None:
    progress = status.get("ingestion_progress")
    if not isinstance(progress, dict):
        return None
    counters = []
    for name in ("progress", "overall_progress", "source_progress"):
        value = progress.get(name)
        counters.append(
            (value.get("completed"), value.get("total"), value.get("unit"))
            if isinstance(value, dict)
            else None
        )
    return (
        progress.get("build_id"),
        progress.get("phase"),
        progress.get("source"),
        *counters,
    )


def _terminal_progress_line(line: str, columns: int) -> str:
    """Clip by terminal cell width, reserving one cell against line wrapping."""
    remaining = max(0, columns - 1)
    result: list[str] = []
    for char in line:
        width = (
            0
            if unicodedata.combining(char)
            else (2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1)
        )
        if width > remaining:
            break
        result.append(char)
        remaining -= width
    return "".join(result)


def _ingestion_progress_frame(status: dict[str, Any], line: str, columns: int) -> str:
    """Draw progress across the current phase, never within a source."""
    progress = status["ingestion_progress"]
    fields = line.split(" | ")
    header = fields[0]
    overall = progress.get("overall_progress")
    if isinstance(overall, dict) and overall.get("unit") == "sources":
        header += f" | {overall.get('completed', 0)}/{overall.get('total', 0)} sources"
    source = progress.get("source")
    if isinstance(source, str) and source:
        header += " | " + "".join(
            char if char.isprintable() else " " for char in source
        )
    counts = (
        overall
        if isinstance(overall, dict) and overall.get("unit") == "sources"
        else progress.get("progress")
    )
    completed = counts.get("completed") if isinstance(counts, dict) else None
    total = counts.get("total") if isinstance(counts, dict) else None
    valid = (
        type(completed) is int
        and type(total) is int
        and total > 0
        and 0 <= completed <= total
        and counts.get("unit") != "phase"
    )
    percentage = round(100 * completed / total) if valid else None
    eta = next(
        (field for field in fields if field.startswith("ETA ")),
        "",
    )
    suffix = f" {percentage:3d}%" if percentage is not None else "  --%"
    if eta:
        suffix += f" | {eta}"
    for field in fields:
        if field.startswith(("Stalled:", "Finalizing")):
            suffix += f" | {field}"
    bar_width = max(0, min(30, columns - len(suffix) - 5))
    filled = round(bar_width * completed / total) if valid else 0
    bar = f"[{'#' * filled}{'-' * (bar_width - filled)}]{suffix}"
    return (
        _terminal_progress_line(header, columns)
        + "\n"
        + _terminal_progress_line(bar, columns)
    )


async def _ingest_with_progress(
    operations: Local | Remote, operation: str, **arguments: Any
) -> dict[str, Any]:
    """Poll read-only status while the single build request is outstanding."""
    ingest = asyncio.create_task(operations.call(operation, **arguments))
    status_task: asyncio.Task[dict[str, Any]] | None = None
    estimator = _PhaseETA()
    previous_key: tuple[Any, ...] | None = None
    last_printed = 0.0
    redraw = (
        bool(getattr(sys.stderr, "isatty", lambda: False)())
        and os.environ.get("TERM") != "dumb"
    )
    drawn = False
    try:
        while not ingest.done():
            done, _ = await asyncio.wait({ingest}, timeout=INGEST_PROGRESS_INTERVAL)
            if done:
                break
            status_task = asyncio.create_task(operations.call("status"))
            await asyncio.wait(
                {ingest, status_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if ingest.done():
                break
            # Progress is advisory; a failed read must not abandon a build.
            status_result = (await asyncio.gather(status_task, return_exceptions=True))[
                0
            ]
            line = (
                _ingestion_progress_line(status_result, estimator)
                if isinstance(status_result, dict)
                else None
            )
            key = (
                _ingestion_progress_key(status_result)
                if isinstance(status_result, dict)
                else None
            )
            now = time.monotonic()
            changed = key != previous_key
            if line is not None and (
                changed or now - last_printed >= INGEST_HEARTBEAT_INTERVAL
            ):
                # Only an unchanged heartbeat claims the status endpoint answered;
                # it does not claim that the worker made progress.
                if redraw:
                    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
                    frame = _ingestion_progress_frame(status_result, line, columns)
                    sys.stderr.write(
                        ("\r\x1b[1A" if drawn else "")
                        + "\r\x1b[2K"
                        + frame.replace("\n", "\n\r\x1b[2K", 1)
                    )
                    sys.stderr.flush()
                    drawn = True
                else:
                    print(
                        line
                        if changed
                        else f"{line} | status checked {time.strftime('%H:%M:%S')}",
                        file=sys.stderr,
                        flush=True,
                    )
                previous_key = key
                last_printed = now
            status_task = None
        return await ingest
    finally:
        if drawn:
            sys.stderr.write("\n")
            sys.stderr.flush()
        # Drain an already-started HTTP read before Remote closes its client.
        # Do not start a final status request or print after ingestion returns.
        if status_task is not None:
            await asyncio.gather(status_task, return_exceptions=True)
        if not ingest.done():
            ingest.cancel()
            await asyncio.gather(ingest, return_exceptions=True)


async def _operate(
    args: argparse.Namespace,
    operations: Local | Remote,
) -> dict[str, Any]:
    """Call the one operation this command names, and return its payload."""

    command = args.command
    if command == "status":
        payload = await operations.call("status")
        # The terminal is an agent's second bounded reader: the same projection,
        # so the two cannot answer differently. `--verbose` is where a person
        # goes for the complete payload the workspace reads.
        from ...core.tool_views import lean_status

        return dict(payload) if args.verbose else lean_status(payload)
    if command == "ingest":
        return await _ingest_with_progress(
            operations,
            "ingest",
            force_recompute=args.force_recompute,
            **({"sources": args.sources} if args.sources else {}),
        )
    if command == "uningest":
        return await _ingest_with_progress(operations, "uningest", sources=args.sources)
    if command == "search":
        search: dict[str, Any] = {
            "top_k": args.top_k,
            "categories_any": args.category,
            "projects_any": args.project,
            "keywords": args.keyword,
            "languages_any": args.language,
            "authors_any": args.author,
            "titles_any": args.title,
            "source_ids": args.source_id,
            "exclude_source_ids": args.exclude_source_id,
            "rerank": not args.no_rerank,
        }
        # A named --method is a diagnostic choice. Leaving it off asks the engine
        # for the method the generation can serve, which is BM25 with a
        # disclosure when only the dense half is unavailable.
        if args.method is not None:
            search["retrieval_method"] = args.method
        return await operations.call("search", query=args.query, **search)
    if command == "sources":
        return await operations.call("list_sources")
    if command == "passage":
        return await operations.call(
            "get_passage", chunk_id=args.chunk_id, context_chunks=args.context_chunks
        )
    if command in {"include", "exclude"}:
        # One decision about retrievable evidence, two subjects. Each refusal
        # names the subject it is missing or the pair that contradict each other,
        # because a command that silently ignored one of them would record a
        # decision the reader did not make.
        if args.chunk and (args.source or args.source_id):
            raise ResearchError(
                "--chunk decides about one passage and a path or --source-id "
                "about a whole source; name one of them."
            )
        if args.chunk:
            return await operations.call(
                "set_chunk_inclusion",
                chunk_id=args.chunk,
                included=command == "include",
                reason=args.reason,
            )
        if not args.source and not args.source_id:
            raise ResearchError(
                "Name the passage with --chunk CHUNK_ID, or the source with a "
                "path or --source-id."
            )
        return await operations.call(
            "set_source_inclusion",
            source_path=args.source,
            source_id=args.source_id,
            included=command == "include",
            reason=args.reason,
        )
    if command == "metadata":
        return await operations.call(
            "set_source_metadata",
            source_path=args.source,
            source_id=args.source_id,
            metadata=_metadata_body(args),
        )
    if command == "stats":
        return await operations.call(
            "search_stats", days=args.days, top=args.top, largest_by=args.largest_by
        )
    if command == "history":
        if args.clear:
            return await operations.call("clear_search_history")
        return await operations.call("search_history", limit=args.limit, days=args.days)
    if command == "chunks":
        by_id = args.source.startswith("src_")
        return await operations.call(
            "source_chunks",
            source_id=args.source if by_id else None,
            source_path=None if by_id else args.source,
            page=args.page,
            page_size=args.page_size,
        )
    if command == "generations":
        if args.use_generation:
            return await operations.call(
                "use_generation", generation_id=args.use_generation
            )
        return await operations.call("list_generations")
    if command == "remove-generation":
        return await operations.call(
            "remove_generation", generation_id=args.generation_id, confirm=args.confirm
        )
    raise ResearchError(f"Unknown command: {command}")


@contextlib.asynccontextmanager
async def _operations(config: ResearchConfig) -> AsyncIterator[Local | Remote]:
    """Reach the app when it is up, and build the one service when it is not."""

    handle = connect(config)
    if handle is not None:
        async with Remote(handle) as remote:
            yield remote
        return
    async with Local(config) as local:
        yield local
