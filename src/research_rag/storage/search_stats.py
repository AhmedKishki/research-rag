"""What searches returned, counted on this machine.

A search records when it ran, which generation answered it, how many passages it
asked for and returned, how long it took, and which passage and source held each
of its first five ranks. The query text is never stored: these are counts of what
the corpus answered with, not a log of what was asked.

The file is machine-local derived state under the runtime directory. Deleting it
resets every count and loses nothing else. Writes are one short transaction per
search, so a reader and a writer in two threads only wait on each other briefly.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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
    elapsed_ms REAL NOT NULL
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
        if version not in (0, SCHEMA_VERSION):
            raise SearchStatsError(
                f"{path} has schema {version}, not {SCHEMA_VERSION}; delete it to "
                "start the counts again."
            )
        if version == 0:
            connection.executescript(_SCHEMA)
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
) -> None:
    """Record one search and the (rank, chunk id, source id) of its first ranks."""

    try:
        with _connected(path) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO searches (searched_at, generation_id, requested_top_k, "
                "result_count, elapsed_ms) VALUES (?, ?, ?, ?, ?)",
                (searched_at, generation_id, requested_top_k, result_count, elapsed_ms),
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


def read_search_stats(path: Path, *, top: int = 20, days: int = 30) -> dict[str, Any]:
    """The counts the workspace and `research-rag stats` report.

    A file that does not exist yet is a project nobody has searched on this
    machine, which is an answer of zeros rather than an error.
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
    try:
        with _connected(path) as connection:
            totals = connection.execute(
                "SELECT COUNT(*) AS searches, "
                "COALESCE(SUM(result_count = 0), 0) AS zero_results, "
                "AVG(result_count) AS mean_results, "
                "MIN(searched_at) AS first_at, MAX(searched_at) AS last_at "
                "FROM searches"
            ).fetchone()
            if not totals["searches"]:
                return empty
            elapsed = [
                float(row[0])
                for row in connection.execute(
                    "SELECT elapsed_ms FROM searches ORDER BY id DESC LIMIT ?",
                    (LATENCY_WINDOW,),
                )
            ]
            by_day = [
                {"day": row["day"], "count": int(row["count"])}
                for row in connection.execute(
                    "SELECT substr(searched_at, 1, 10) AS day, COUNT(*) AS count "
                    "FROM searches GROUP BY day ORDER BY day DESC LIMIT ?",
                    (days,),
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
                    "SUM(rank = 1) AS rank_one FROM appearances "
                    "GROUP BY source_id ORDER BY top_five DESC, rank_one DESC, "
                    "source_id LIMIT ?",
                    (top,),
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
                    "FROM appearances GROUP BY chunk_id "
                    "ORDER BY top_five DESC, rank_one DESC, chunk_id LIMIT ?",
                    (top,),
                )
            ]
            appeared = [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT source_id FROM appearances"
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
