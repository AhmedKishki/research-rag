"""What searches returned, counted on this machine, and what was asked.

A search records when it ran, which generation answered it, how many passages it
asked for and returned, how long it took, who asked, and which passage and source
held each of its first five ranks. It also records the query and its filters,
unless `runtime.search_history` is off, so a search can be run again from a list.
The history is the one place a question is kept: clearing it removes every query
and filter and leaves the counts.

The file is machine-local state under the runtime directory. Deleting it resets
every count and the history, and loses nothing else. Writes are one short
transaction per search, so a reader and a writer in two threads only wait on each
other briefly.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
# The ranks a search records. The top five is what a reader reads, and rank one
# is what an agent quotes first.
TRACKED_RANKS = 5
# How many recent searches the latency figures are taken over, so an old slow
# build does not set today's median.
LATENCY_WINDOW = 1000
# Wait this long for another thread's transaction before giving up on a write.
BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY,
    searched_at TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    requested_top_k INTEGER NOT NULL,
    result_count INTEGER NOT NULL,
    elapsed_ms REAL NOT NULL,
    query TEXT,
    filters TEXT,
    caller TEXT
);
CREATE TABLE IF NOT EXISTS appearances (
    search_id INTEGER NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (search_id, rank)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS appearances_source ON appearances(source_id);
CREATE INDEX IF NOT EXISTS appearances_chunk ON appearances(chunk_id);
"""


class SearchStatsError(RuntimeError):
    """The counts file exists and cannot be read as one."""


@contextmanager
def _connected(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)) as connection:
        connection.row_factory = sqlite3.Row
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, 1, SCHEMA_VERSION):
            raise SearchStatsError(
                f"{path} has schema {version}, not {SCHEMA_VERSION}; delete it to "
                "start the counts again."
            )
        if version == 0:
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version == 1:
            # Version 1 counted ranks and kept no question; its rows stay as they
            # were, with nothing recorded for who asked or what.
            with connection:
                for column in ("query", "filters", "caller"):
                    connection.execute(f"ALTER TABLE searches ADD COLUMN {column} TEXT")
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        yield connection


def record_search(
    path: Path,
    *,
    searched_at: str,
    generation_id: str,
    requested_top_k: int,
    result_count: int,
    elapsed_ms: float,
    ranked: Iterable[tuple[int, str, str]],
    query: str | None = None,
    filters: dict[str, list[str]] | None = None,
    caller: str | None = None,
) -> None:
    """Record one search and the (rank, chunk id, source id) of its first ranks.

    `query` and `filters` are what the caller asked; a caller that must keep no
    question passes neither.
    """

    try:
        with _connected(path) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO searches (searched_at, generation_id, requested_top_k, "
                "result_count, elapsed_ms, query, filters, caller) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    searched_at,
                    generation_id,
                    requested_top_k,
                    result_count,
                    elapsed_ms,
                    query,
                    json.dumps(filters, ensure_ascii=False, sort_keys=True)
                    if filters
                    else None,
                    caller,
                ),
            )
            connection.executemany(
                "INSERT INTO appearances (search_id, rank, chunk_id, source_id) "
                "VALUES (?, ?, ?, ?)",
                [
                    (cursor.lastrowid, rank, chunk_id, source_id)
                    for rank, chunk_id, source_id in ranked
                    if 1 <= rank <= TRACKED_RANKS
                ],
            )
    except sqlite3.Error as exc:
        raise SearchStatsError(f"{path} could not be written: {exc}") from exc


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 1)


def _cutoff(since_days: float | None) -> str | None:
    """The oldest timestamp inside a window of days, or None for all time."""

    if since_days is None:
        return None
    moment = datetime.now(UTC) - timedelta(days=float(since_days))
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _window(since_days: float | None) -> tuple[str, tuple[str, ...]]:
    """The WHERE clause that keeps a search inside the window, and its parameter."""

    cutoff = _cutoff(since_days)
    return ("WHERE searched_at >= ?", (cutoff,)) if cutoff else ("", ())


def read_search_stats(
    path: Path,
    *,
    top: int = 20,
    days: int = 30,
    since_days: float | None = None,
) -> dict[str, Any]:
    """The counts the workspace and `research-rag stats` report.

    `since_days` bounds every figure to the searches of that many recent days;
    `None` counts all of them. A file that does not exist yet is a project nobody
    has searched on this machine, which is an answer of zeros rather than an error.
    """

    empty: dict[str, Any] = {
        "search_count": 0,
        "zero_result_count": 0,
        "mean_result_count": None,
        "first_search_at": None,
        "last_search_at": None,
        "median_elapsed_ms": None,
        "p95_elapsed_ms": None,
        "searches_by_day": [],
        "sources": [],
        "passages": [],
        "appeared_source_ids": [],
    }
    if not path.is_file():
        return empty
    where, parameters = _window(since_days)
    inside = f"search_id IN (SELECT id FROM searches {where})"
    try:
        with _connected(path) as connection:
            totals = connection.execute(
                "SELECT COUNT(*) AS searches, "
                "COALESCE(SUM(result_count = 0), 0) AS zero_results, "
                "AVG(result_count) AS mean_results, "
                f"MIN(searched_at) AS first_at, MAX(searched_at) AS last_at FROM searches {where}",
                parameters,
            ).fetchone()
            if not totals["searches"]:
                return empty
            elapsed = [
                float(row[0])
                for row in connection.execute(
                    f"SELECT elapsed_ms FROM searches {where} ORDER BY id DESC LIMIT ?",
                    (*parameters, LATENCY_WINDOW),
                )
            ]
            by_day = [
                {"day": row["day"], "count": int(row["count"])}
                for row in connection.execute(
                    "SELECT substr(searched_at, 1, 10) AS day, COUNT(*) AS count "
                    f"FROM searches {where} GROUP BY day ORDER BY day DESC LIMIT ?",
                    (*parameters, days),
                )
            ]
            sources = [
                {
                    "source_id": row["source_id"],
                    "top_five": int(row["top_five"]),
                    "rank_one": int(row["rank_one"]),
                }
                for row in connection.execute(
                    "SELECT source_id, COUNT(*) AS top_five, "
                    f"SUM(rank = 1) AS rank_one FROM appearances WHERE {inside} "
                    "GROUP BY source_id ORDER BY top_five DESC, rank_one DESC, "
                    "source_id LIMIT ?",
                    (*parameters, top),
                )
            ]
            passages = [
                {
                    "chunk_id": row["chunk_id"],
                    "source_id": row["source_id"],
                    "top_five": int(row["top_five"]),
                    "rank_one": int(row["rank_one"]),
                }
                for row in connection.execute(
                    "SELECT chunk_id, MIN(source_id) AS source_id, "
                    "COUNT(*) AS top_five, SUM(rank = 1) AS rank_one "
                    f"FROM appearances WHERE {inside} GROUP BY chunk_id "
                    "ORDER BY top_five DESC, rank_one DESC, chunk_id LIMIT ?",
                    (*parameters, top),
                )
            ]
            appeared = [
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT source_id FROM appearances WHERE {inside}",
                    parameters,
                )
            ]
    except sqlite3.Error as exc:
        raise SearchStatsError(f"{path} could not be read: {exc}") from exc
    mean = totals["mean_results"]
    return {
        "search_count": int(totals["searches"]),
        "zero_result_count": int(totals["zero_results"]),
        "mean_result_count": None if mean is None else round(float(mean), 2),
        "first_search_at": totals["first_at"],
        "last_search_at": totals["last_at"],
        "median_elapsed_ms": (
            round(statistics.median(elapsed), 1) if elapsed else None
        ),
        "p95_elapsed_ms": _percentile(elapsed, 0.95),
        "searches_by_day": by_day,
        "sources": sources,
        "passages": passages,
        "appeared_source_ids": appeared,
    }


def read_history(
    path: Path, *, limit: int = 20, since_days: float | None = None
) -> dict[str, Any]:
    """The most recent searches that kept their question, newest first.

    A search recorded while history was off, or cleared since, carries no question
    and is left out: there is nothing to run again.
    """

    if not path.is_file():
        return {"searches": [], "count": 0}
    where, parameters = _window(since_days)
    asked = "query IS NOT NULL" + (" AND " + where[6:] if where else "")
    try:
        with _connected(path) as connection:
            rows = connection.execute(
                "SELECT searched_at, query, filters, requested_top_k, result_count, "
                f"elapsed_ms, caller FROM searches WHERE {asked} "
                "ORDER BY id DESC LIMIT ?",
                (*parameters, limit),
            ).fetchall()
            count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM searches WHERE {asked}", parameters
                ).fetchone()[0]
            )
    except sqlite3.Error as exc:
        raise SearchStatsError(f"{path} could not be read: {exc}") from exc
    return {
        "count": count,
        "searches": [
            {
                "searched_at": row["searched_at"],
                "query": row["query"],
                "filters": json.loads(row["filters"]) if row["filters"] else {},
                "requested_top_k": int(row["requested_top_k"]),
                "result_count": int(row["result_count"]),
                "elapsed_ms": float(row["elapsed_ms"]),
                "caller": row["caller"] or "",
            }
            for row in rows
        ],
    }


def clear_history(path: Path) -> int:
    """Forget every question and filter, and keep the counts. Returns how many."""

    if not path.is_file():
        return 0
    try:
        with _connected(path) as connection, connection:
            cleared = connection.execute(
                "UPDATE searches SET query = NULL, filters = NULL "
                "WHERE query IS NOT NULL OR filters IS NOT NULL"
            ).rowcount
    except sqlite3.Error as exc:
        raise SearchStatsError(f"{path} could not be written: {exc}") from exc
    return int(cleared)


def read_appearances(
    path: Path,
    *,
    chunk_ids: Sequence[str] = (),
    source_id: str | None = None,
) -> dict[str, tuple[int, int]]:
    """How often each named passage, or one source as a whole, reached the top five.

    Keys are chunk ids, and `source_id` when it was asked for. Each value is
    (top five, rank one). A passage nothing returned is absent.
    """

    found: dict[str, tuple[int, int]] = {}
    if not path.is_file():
        return found
    try:
        with _connected(path) as connection:
            for start in range(0, len(chunk_ids), 500):
                batch = list(chunk_ids[start : start + 500])
                marks = ",".join("?" for _ in batch)
                for row in connection.execute(
                    "SELECT chunk_id, COUNT(*), SUM(rank = 1) FROM appearances "
                    f"WHERE chunk_id IN ({marks}) GROUP BY chunk_id",
                    batch,
                ):
                    found[str(row[0])] = (int(row[1]), int(row[2]))
            if source_id:
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(rank = 1), 0) FROM appearances "
                    "WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                if row[0]:
                    found[source_id] = (int(row[0]), int(row[1]))
    except sqlite3.Error as exc:
        raise SearchStatsError(f"{path} could not be read: {exc}") from exc
    return found
