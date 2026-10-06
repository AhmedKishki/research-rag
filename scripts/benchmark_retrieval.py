"""Measure retrieval latency for one condition, and aggregate worker reports.

The app owns the numbers; a design owns the experiment. This harness is one
worker: it opens the project it is pointed at, runs the frozen query set it is
given through ``hybrid`` with reranking, and writes one JSON report. It starts no
app, registers no client, and stops nothing, and it never inspects a process
tree. The stdio gateway it opens is a child of the worker that opened it, and the
worker closes it before it reports, so the reaped children's CPU is that child's
and nothing else.

No relevance is measured here. A row says how long one search took and what it
returned; judging whether that return was right belongs to
``scripts/evaluate_retrieval.py`` and its judged set. No report carries a label, a
grade, or a relevance figure, and the report says so in its own ``notice``.

Cold and warm are named and never blended. The first query of every worker pays
the index and model load, so it is measured alone as ``first_query_cold``; its
result is the first warm-up, ``--warmup-count`` further searches follow outside
every measured window, and only then is the complete shuffled query set timed
warm. Each worker is a fresh interpreter, so every worker's cold row is a genuine
first query rather than a repeat.

Nothing is flushed and nothing is cleared. The model binaries, the index files,
and the generation are read from the caches and the disposable project as they
lie, so a "cold" row here means a cold *process* reading possibly warm files, and
every report says exactly that. A figure that claims a cold disk would be a claim
this harness cannot support.

    uv run python scripts/benchmark_retrieval.py --project /tmp/rr-latency \\
        --queries blocks/block-0.json --report reports/block-0.json

    uv run python scripts/benchmark_retrieval.py --summarize manifest.json \\
        --report reports/summary.json

Report version 1. ``aggregate_reports`` is the summarizer: it groups worker
reports by condition, reports cold and warm separately, pairs a condition against
a baseline only where a block and its query list match, and names a determinism
mismatch rather than dropping the row that carries it. It computes no test
statistic and no winner.
"""

from __future__ import annotations

import time

#: When this interpreter began the run it is about to report on. Everything the
#: report calls startup — the import of ``fastmcp``, of the app, and of the
#: retrieval stack — happens below this line, so ``service_ready_elapsed`` and
#: ``startup_to_first_result_elapsed`` are costs a reader pays in production and
#: not costs this script pays to measure itself.
_START = time.perf_counter()

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import Client

from research_rag.core.service import ResearchService
from research_rag.project.config import (
    ConfigurationError,
    configured_source_directory,
    resolve_config,
)
from research_rag.project.support import ResearchError
from research_rag.retrieval.ultrarag import (
    VanillaUltraRAG,
    create_vanilla_transport,
)

#: The protocol this harness speaks. A worker report and the summary over worker
#: reports each carry their own version, because a figure measured by report
#: version 0 is not comparable with one measured by version 1.
REPORT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
#: The only path a worker measures. ``hybrid`` with reranking is what the public
#: search tool does, so this is the latency a caller waits for.
PROBE_METHOD = "hybrid"
PROBE_RERANK = True
#: How much of one vanilla call and one handshake a worker may wait. A cold
#: first query loads the indexes inside the first call, so the call budget is the
#: same one the app's own transport uses rather than a latency probe's guess.
CLIENT_TIMEOUT_SECONDS = 1800
CLIENT_INIT_TIMEOUT_SECONDS = 300
DEFAULT_TOP_K = 10
DEFAULT_WARMUP_COUNT = 3
DEFAULT_SEED = 0
#: The query-set schema this harness reads: a frozen set for one condition and one
#: balanced block, with no judgments in it.
QUERY_SET_SCHEMA_VERSION = 1
#: The fixed candidate-depth and rerank-window points the experiment compares.
#: A cap is not a measured window, so the summary reports the point each condition
#: observed beside these three and names any point outside them rather than
#: averaging across points.
PROTOCOL_POINTS: tuple[tuple[int, int], ...] = ((40, 20), (40, 50), (80, 20))
#: The files whose digests a worker compares across its own run. The runtime
#: directory is deliberately absent: the gateway writes its logs and handoff
#: files there, so hashing it would report this harness's own traffic as a change
#: to the project's data. Nothing else in the project is written either, and this
#: probe writes only its own report, outside the project.
DATA_STATE_FILES: tuple[str, ...] = (
    "project.json",
    "config.toml",
    "source-metadata.json",
    "source-exclusions.json",
    "chunk-exclusions.json",
    "source-catalog.json",
)

#: How a reader is told that a run's own two kinds of CPU are not the same
#: quantity. A gateway child burns most of the CPU a search costs, and a worker
#: that reported one blended number would be describing a machine it does not own.
CPU_ACCOUNTING_NOTE = (
    "self_cpu_seconds is this worker's own user plus system time. "
    "children_cpu_seconds_total_reaped is the same sum for the stdio gateway this "
    "worker started, counted only once the operating system has reaped it, so a "
    "gateway still running is absent from it. The two totals are the whole cost of "
    "one worker; a per-query row carries the self figure alone and never the "
    "gateway's."
)

#: Reported beside every worker so a reader does not mistake a cold process for a
#: cold disk.
CACHE_DISCLOSURE = (
    "No page cache was dropped, no index was reloaded, and no model file was "
    "evicted. Model binaries are read from the shared user cache and the index is "
    "read from the disposable project as both already lie. Every cold row in this "
    "report is therefore a cold process reading possibly warm files, and the "
    "startup figures include reading those files for the first time in this "
    "process."
)

#: Reported beside the reports a worker writes, so a reader knows what the file is
#: not.
PROBE_NOTICE = (
    "Latency and the pipeline's own budget fields only. No relevance, grade, "
    "label, or judgment is computed here: what a row returned is recorded as "
    "identifiers so a repeat can be compared for determinism, and nothing is said "
    "about whether those identifiers answer the query. Searches run through "
    "ResearchService.search in process, which is the engine the three surfaces "
    "share, with include_staleness=false and without the evaluation trace, so a "
    "row times the path a reader waits for rather than a diagnostic that no "
    "surface requests. The project is read-only to this harness: no ingestion runs, "
    "no setting is written, and the digests under data_state say so."
)


class ProbeError(RuntimeError):
    """Raised when a latency probe cannot be measured, or cannot be written."""


# --------------------------------------------------------------------------
# Small shared helpers. Nearest-rank percentiles, because every figure they
# produce is an observation this harness actually made.
# --------------------------------------------------------------------------


def _utc_now() -> str:
    """The current instant in UTC, in one format every report uses.

    A timestamp that carries no offset cannot be ordered against another one, so
    every date in every report is written in UTC with an explicit ``Z``.
    """

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _seconds(value: float) -> float:
    """A duration at the precision a report prints, and no more."""

    return round(float(value), 6)


def _finite(value: Any) -> float | None:
    """A usable number, or ``None``.

    A boolean is not a measurement, and neither is a NaN or an infinity: a
    percentile over them would be a figure nothing observed.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ranks(values: Sequence[Any]) -> list[float]:
    """The usable durations in ascending order, and nothing else."""

    return sorted(
        number for number in (_finite(value) for value in values) if number is not None
    )


def latency_stats(values: Sequence[Any]) -> dict[str, float | int | None]:
    """The distribution of measured durations, nearest-rank.

    Nearest-rank rather than interpolated, so each figure is a duration this
    harness observed. ``p95_seconds`` is reported beside ``max_seconds`` and is
    not the slowest: over twenty observations it is the nineteenth, and calling
    that the slowest would overstate it by whatever the sample size decides. An
    empty set reports ``None`` rather than zero, because a latency that was not
    measured is not a latency of zero.
    """

    measured = _ranks(values)
    if not measured:
        return {
            "count": 0,
            "min_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
        }
    return {
        "count": len(measured),
        "min_seconds": measured[0],
        "p50_seconds": measured[max(math.ceil(0.50 * len(measured)) - 1, 0)],
        "p95_seconds": measured[max(math.ceil(0.95 * len(measured)) - 1, 0)],
        "max_seconds": measured[-1],
        "mean_seconds": statistics.fmean(measured),
    }


def _ratio_stats(values: Sequence[Any]) -> dict[str, float | int | None]:
    """The same nearest-rank ranks over dimensionless ratios.

    Kept separate from :func:`latency_stats` because a key reading
    ``p50_seconds`` over a ratio would name a unit the number does not have.
    """

    measured = _ranks(values)
    if not measured:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(measured),
        "min": measured[0],
        "p50": measured[max(math.ceil(0.50 * len(measured)) - 1, 0)],
        "p95": measured[max(math.ceil(0.95 * len(measured)) - 1, 0)],
        "max": measured[-1],
        "mean": statistics.fmean(measured),
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def harness_provenance() -> dict[str, Any]:
    """Which code produced this report: its digest, its revision, its interpreter.

    Two latency figures are comparable only when the code that wrote them is the
    code that was meant to write them, and a digest says that where a version
    number cannot: an edited script and its tag share a version. The revision is
    read from git when the checkout can answer and is ``None`` when it cannot,
    because a missing revision is not a clean tree.
    """

    digest = _sha256_file(Path(__file__).resolve())
    revision = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent,
            check=False,
        )
        if completed.returncode == 0:
            revision = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "script": Path(__file__).name,
        "script_sha256": digest,
        "git_revision": revision,
        "python": platform.python_version(),
    }


def environment_snapshot() -> dict[str, Any]:
    """The machine and its background load, as read at one moment.

    The load average is the only background-load figure this harness reports, and
    it is the operating system's own one-minute view: a mean over the processes
    runnable or in uninterruptible sleep, not this worker's own cost. It is
    recorded before and after the run rather than interpreted, because a harness
    that judged whether the machine was quiet enough would be selecting the runs
    it keeps.
    """

    loadavg: list[float] | None
    try:
        loadavg = [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        loadavg = None
    return {
        "captured_at": _utc_now(),
        "loadavg": loadavg,
        "loadavg_note": (
            "the operating system's one, five, and fifteen minute averages; "
            "recorded, not judged"
        ),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


def _cpu_seconds(usage: Any) -> float:
    """One rusage record's user plus system time."""

    return float(getattr(usage, "ru_utime", 0.0)) + float(
        getattr(usage, "ru_stime", 0.0)
    )


def _peak_rss_mib(usage: Any) -> float | None:
    """One rusage record's peak resident set size, in MiB.

    ``ru_maxrss`` is kibibytes on Linux and bytes on Darwin. The two peaks of one
    worker — its own and its reaped gateway's — are reported separately and never
    added: they are maxima over different windows, and their sum would be a
    figure describing no moment that ever existed on this machine.
    """

    raw = getattr(usage, "ru_maxrss", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(float(raw) / divisor, 3)


# --------------------------------------------------------------------------
# The frozen query set. It carries no judgment, so nothing here can be scored.
# --------------------------------------------------------------------------


def load_query_set(path: Path) -> tuple[dict[str, Any], str]:
    """Read and structurally validate one frozen query set, with its digest.

    The set is frozen before the run, so its digest is the identity of the
    measurement: two reports over the same queries carry the same digest, and a
    report whose digest differs from the next worker's is measuring something
    else. An empty set is refused because a latency distribution over no query is
    not a latency measurement.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Cannot read the query set: {path}") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"Unsupported query set in {path}: not an object")
    if payload.get("schema_version") != QUERY_SET_SCHEMA_VERSION:
        raise ProbeError(
            f"Unsupported query-set schema in {path}: "
            f"{payload.get('schema_version')!r}, expected "
            f"{QUERY_SET_SCHEMA_VERSION}"
        )
    if not str(payload.get("condition_id") or "").strip():
        raise ProbeError(f"The query set names no condition_id: {path}")
    if not str(payload.get("generation_id") or "").strip():
        raise ProbeError(f"The query set names no generation_id: {path}")
    block = payload.get("block")
    if isinstance(block, bool) or not isinstance(block, int):
        raise ProbeError(f"The query set names no integer block: {path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ProbeError(f"The query set's metadata is not an object: {path}")
    entries = sorted(_query_rows(payload), key=lambda row: row["query_id"])
    # A trial's condition/block/configuration is provenance, not different query
    # work. Hash the frozen cohort separately from the input-file receipt.
    digest = _sha256_bytes(
        json.dumps(
            entries,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    payload["input_file_sha256"] = _sha256_file(path)
    payload["queries"] = entries
    return payload, digest


def _query_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The set's queries as ``query_id``, text, and declared metadata.

    A query carries an identifier and text, and nothing else is required of it:
    no target, no class, no judgment. The declared metadata travels with the row
    so a reader can see what the query was asked to be, including whether it was
    declared out of the corpus's physical scope.
    """

    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ProbeError("The query set has no queries")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if not isinstance(query, dict):
            raise ProbeError(f"Query is not an object: {query!r}")
        query_id = str(query.get("query_id") or "").strip()
        text = str(query.get("query") or "").strip()
        metadata = query.get("metadata", {})
        if not query_id or not text:
            raise ProbeError(f"Query has no query_id or text: {query!r}")
        if query_id in seen:
            raise ProbeError(f"Query id repeats in the set: {query_id!r}")
        if not isinstance(metadata, dict):
            raise ProbeError(f"Query metadata is not an object: {query_id!r}")
        seen.add(query_id)
        rows.append({"query_id": query_id, "query": text, "metadata": dict(metadata)})
    return rows


def _order_for(
    rows: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """The executed order: the complete set, shuffled from one seed.

    The shuffle is seeded rather than random so a repeated worker runs the same
    order and a difference between two workers is not a difference between two
    orders. The complete set is timed warm, including the query that was the cold
    one, so every query has a warm measurement and none has a cold measurement
    only.
    """

    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return ordered, [str(item["query_id"]) for item in ordered]


# --------------------------------------------------------------------------
# The project's own files, read-only. Digests before and after the run are how a
# reader sees that a probe changed nothing but its own report.
# --------------------------------------------------------------------------


def data_state_digests(
    portable_root: Path, current_path: Path, generation_root: Path | None
) -> dict[str, str | None]:
    """A digest for every file whose contents this harness must not change.

    The project's portable state and its selected generation are listed by name
    rather than by walking the project, because the runtime directory holds the
    gateway's logs and handoff files: a directory digest would report this
    harness's own traffic as a change to the project's data. A file that is absent
    reads ``None``, which is an absence and not a digest of nothing, and a
    generation that is not known yet contributes no generation digest at all.
    """

    digests: dict[str, str | None] = {
        name: _sha256_file(portable_root / name) for name in DATA_STATE_FILES
    }
    digests["runtime/current.json"] = _sha256_file(current_path)
    if generation_root is not None:
        digests["generation/manifest.json"] = _sha256_file(
            generation_root / "manifest.json"
        )
        digests["generation/chunks/chunks.jsonl"] = _sha256_file(
            generation_root / "chunks" / "chunks.jsonl"
        )
    return digests


def _changed_files(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> list[str]:
    """The names whose digest is not the same on both sides of the run."""

    return sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )


# --------------------------------------------------------------------------
# The search payload, read as the app's own search method returns it. Every field
# below is one the service puts in the answer; none is recomputed here.
# --------------------------------------------------------------------------


def rerank_fallback_reason(
    entry: Mapping[str, Any], payload: Mapping[str, Any]
) -> str | None:
    """Why this search is not the hybrid+rerank path the probe names, or ``None``.

    A search that asked for reranking and did not rerank returned an unranked
    candidate order, so a latency row measured on it would time a path this report
    does not claim. The run refuses rather than recording a wrong row.

    One case is not a refusal. A query the frozen set declares
    ``physically_unsupported_scope`` is one the corpus cannot hold at all, so an
    empty result is the expected answer and an empty result has no window for the
    cross-encoder to reorder. That declaration is the set's, it is recorded in the
    row, and it excuses only an empty result beside a fallback: a declared
    out-of-scope query that returned passages and still did not rerank is refused
    like any other.
    """

    if not bool(payload.get("rerank_requested")) or bool(payload.get("reranked")):
        return None
    hits = payload.get("hits")
    empty = isinstance(hits, list) and not hits
    declared = entry.get("metadata", {}).get("physically_unsupported_scope") is True
    if empty and declared:
        return None
    fallback = payload.get("rerank_fallback")
    reason = (
        str(fallback.get("reason") or "no reason recorded")
        if isinstance(fallback, Mapping)
        else "no reason recorded"
    )
    return f"{reason}, empty_result={empty}"


def _require_rerank_applied(
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    phase: str,
    position: int,
) -> None:
    reason = rerank_fallback_reason(entry, payload)
    if reason is not None:
        raise ProbeError(
            f"{phase} search {position} for {entry['query_id']} requested reranking "
            f"and did not rerank ({reason}). A latency row timed on that path would "
            "report a configuration this run does not claim. Fix the model cache or "
            "the project, then run the condition again."
        )


async def _search(
    service: ResearchService, entry: Mapping[str, Any], top_k: int
) -> dict[str, Any]:
    """One search through the service method the three surfaces share.

    ``include_staleness`` is false and the evaluation trace is not asked for, so
    the row times the path a reader waits for. The payload's own budget fields
    come back either way, so nothing is measured at the cost of the measurement.
    """

    payload = await service.search(
        str(entry["query"]),
        top_k=top_k,
        retrieval_method=PROBE_METHOD,
        rerank=PROBE_RERANK,
        include_staleness=False,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        raise ProbeError(
            f"search returned an unexpected payload for {entry['query_id']}"
        )
    return payload


def _row(
    payload: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    phase: str,
    position: int,
    top_k: int,
    elapsed: float,
    self_cpu: float,
) -> dict[str, Any]:
    """One measured search, with what it returned and what it asked for.

    ``returned_chunk_ids`` is recorded in rank order so a repeat can be compared
    with this one, and the actual candidate depth, candidate count, and rerank
    window are recorded beside it so two reports over the same budget can be told
    apart from two reports over different ones. A cap the settings asked for is
    not what the engine did, and only these are the latter.
    """

    hits = payload.get("hits") or []
    return {
        "query_id": str(entry["query_id"]),
        "phase": phase,
        "position": position,
        "requested_top_k": top_k,
        "elapsed_seconds": _seconds(elapsed),
        "self_cpu_seconds": _seconds(self_cpu),
        "returned_chunk_ids": [
            str(hit.get("chunk_id")) for hit in hits if isinstance(hit, Mapping)
        ],
        "result_count": int(payload.get("result_count") or 0),
        "retrieval_method": payload.get("retrieval_method"),
        "candidate_depth": payload.get("candidate_depth"),
        "candidate_count": payload.get("candidate_count"),
        "rerank_window": payload.get("rerank_window"),
        "rerank_requested": bool(payload.get("rerank_requested")),
        "reranked": bool(payload.get("reranked")),
        "rerank_fallback": payload.get("rerank_fallback"),
        "metadata": dict(entry.get("metadata") or {}),
    }


# One worker: open the project, serve the condition, close the gateway, report.
# --------------------------------------------------------------------------


def _client_scope(config: Any) -> AbstractAsyncContextManager[Any]:
    """The stdio gateway this worker starts, as a context manager.

    The gateway is opened here and nowhere else, is a child of this worker, and
    is closed by the ``async with`` that ends the measurement. Nothing is
    registered, no daemon is contacted, and no existing app is signalled.
    """

    return Client(
        create_vanilla_transport(config),
        timeout=CLIENT_TIMEOUT_SECONDS,
        init_timeout=CLIENT_INIT_TIMEOUT_SECONDS,
    )


def _service_for(config: Any, client: Any) -> ResearchService:
    """The app's own service, over that gateway."""

    return ResearchService(
        config, VanillaUltraRAG(client, config), record_searches=False
    )


def _require_servable(
    status: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[Path, str]:
    """Refuse a project this probe cannot measure, naming what it needed.

    Two refusals matter. A generation that is not hybrid-ready has no dense half
    to time, and a project serving a generation other than the one the frozen set
    names would have its rows attributed to work this run did not do. Both are
    refused rather than reported as a surprising number.
    """

    if not isinstance(status, Mapping):
        raise ProbeError("status returned an unexpected payload")
    if status.get("hybrid_ready") is not True:
        raise ProbeError(
            "This project's current generation is not hybrid-ready, so a "
            "hybrid+rerank row would time a path this corpus does not serve. "
            "Build a generation the project can serve hybrid retrieval from, or "
            "point the probe at a project that has one."
        )
    generation_root = str(status.get("generation_root") or "")
    if not generation_root:
        raise ProbeError("The project has no selected generation to probe")
    expected = str(payload["generation_id"])
    observed = str(status.get("generation_id") or "")
    if observed != expected:
        raise ProbeError(
            f"The query set names generation {expected!r} and the project serves "
            f"{observed!r}. Re-freeze the query set against the generation the "
            "project serves, then run the condition again."
        )
    return Path(generation_root), observed


def _report_destination(value: Path) -> Path:
    """The report path, refused when it would overwrite or follow a symlink.

    A report is a record of one run. Replacing an earlier one destroys the
    evidence that it was a different run, and writing through a symlink can land
    the record somewhere nobody will look for it. Both are refused, and neither
    remedy is automatic: choose a new path or remove the old report deliberately.
    """

    path = value.expanduser()
    if path.is_symlink():
        raise ProbeError(
            f"--report is a symlink: {path}. A probe report is written where it is "
            "named, not where a link points."
        )
    if path.exists():
        raise ProbeError(
            f"--report already exists: {path}. Each run writes a new report; give "
            "this run a path of its own rather than replacing the earlier record."
        )
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one report, creating only the directory it names."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


async def run_probe(
    args: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
    rusage: Callable[[int], Any] = resource.getrusage,
    client_scope: Callable[[Any], AbstractAsyncContextManager[Any]] = _client_scope,
    service_for: Callable[[Any, Any], ResearchService] = _service_for,
    origin: float | None = None,
) -> dict[str, Any]:
    """Measure one condition in this process and return the report it wrote.

    ``origin`` defaults to this module's ``_START``, so the two startup figures
    include the interpreter and the imports below it. The other seams exist so a
    test can measure the same protocol against a scripted clock and a scripted
    rusage record without a gateway or a model; a caller that passes them is
    measuring the harness, and the report says which figures were read from them.
    """

    started_clock = _START if origin is None else float(origin)
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        raise ProbeError(f"Project root is not a directory: {project}")
    if not 1 <= int(args.top_k) <= 50:
        raise ProbeError("--top-k must be between 1 and 50")
    if int(args.warmup_count) < 0:
        raise ProbeError("--warmup-count must not be negative")
    report_path = _report_destination(Path(args.report))

    payload, digest = load_query_set(args.queries)
    entries = _query_rows(payload)
    ordered, order = _order_for(entries, int(args.seed))
    by_id = {str(item["query_id"]): item for item in entries}

    try:
        config = resolve_config(
            project,
            source_directory=configured_source_directory(project),
            offline=True,
        )
    except ConfigurationError as exc:
        raise ProbeError(f"The project cannot be configured: {exc}") from exc

    before_environment = environment_snapshot()
    before_state = data_state_digests(config.portable_root, config.current_path, None)
    started_at = _utc_now()
    self_before = _cpu_seconds(rusage(resource.RUSAGE_SELF))

    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    generation_root: Path | None = None
    generation_id = ""
    service_policy_fingerprint: str | None = None
    service_ready_elapsed = 0.0
    cold_started_clock = 0.0
    cold_elapsed = 0.0
    startup_to_first_result = 0.0
    warmup_seconds = 0.0
    warm_seconds = 0.0
    try:
        async with client_scope(config) as client:
            service = service_for(config, client)
            status = dict(await service.status())
            generation_root, generation_id = _require_servable(status, payload)
            service_policy_fingerprint = getattr(
                service, "retrieval_policy_fingerprint", None
            )
            # Taken here, after the read-only status call and before the first
            # search, so every comparison below covers the searches themselves.
            before_state = data_state_digests(
                config.portable_root, config.current_path, generation_root
            )
            service_ready_elapsed = _seconds(clock() - started_clock)

            # The cold row. Its result is the first warm-up: the embedder and the
            # cross-encoder are loaded by it, which is why nothing else is timed
            # before the warm phase begins.
            cold_entry = ordered[0]
            cold_cpu_before = _cpu_seconds(rusage(resource.RUSAGE_SELF))
            cold_started_clock = clock()
            cold_payload = await _search(service, cold_entry, int(args.top_k))
            cold_elapsed = _seconds(clock() - cold_started_clock)
            startup_to_first_result = _seconds(clock() - started_clock)
            _require_rerank_applied(cold_entry, cold_payload, phase="cold", position=0)
            rows.append(
                _row(
                    cold_payload,
                    cold_entry,
                    phase="cold",
                    position=0,
                    top_k=int(args.top_k),
                    elapsed=cold_elapsed,
                    self_cpu=_cpu_seconds(rusage(resource.RUSAGE_SELF))
                    - cold_cpu_before,
                )
            )

            warmup_started = clock()
            for _ in range(int(args.warmup_count)):
                _require_rerank_applied(
                    cold_entry,
                    await _search(service, cold_entry, int(args.top_k)),
                    phase="warmup",
                    position=-1,
                )
            warmup_seconds = _seconds(clock() - warmup_started)

            warm_started = clock()
            for position, query_id in enumerate(order, start=1):
                entry = by_id[query_id]
                cpu_before = _cpu_seconds(rusage(resource.RUSAGE_SELF))
                started = clock()
                result = await _search(service, entry, int(args.top_k))
                elapsed = _seconds(clock() - started)
                _require_rerank_applied(entry, result, phase="warm", position=position)
                rows.append(
                    _row(
                        result,
                        entry,
                        phase="warm",
                        position=position,
                        top_k=int(args.top_k),
                        elapsed=elapsed,
                        self_cpu=_cpu_seconds(rusage(resource.RUSAGE_SELF))
                        - cpu_before,
                    )
                )
            warm_seconds = _seconds(clock() - warm_started)
    except ResearchError as exc:
        raise ProbeError(f"The project refused the probe: {exc}") from exc
    finally:
        self_after = _cpu_seconds(rusage(resource.RUSAGE_SELF))

    # Read after the gateway has been closed and reaped, so the children's total
    # is that child's rather than an estimate of a running one.
    children_usage = rusage(resource.RUSAGE_CHILDREN)
    children_cpu_seconds = _cpu_seconds(children_usage)
    after_environment = environment_snapshot()
    after_state = data_state_digests(
        config.portable_root, config.current_path, generation_root
    )
    changed = _changed_files(before_state, after_state)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "research-rag retrieval latency probe",
        "notice": PROBE_NOTICE,
        "cache_disclosure": CACHE_DISCLOSURE,
        "dates": {"started_at": started_at, "finished_at": _utc_now()},
        "harness": harness_provenance(),
        "probe": {
            "condition_id": str(payload["condition_id"]),
            "block": int(payload["block"]),
            "order_index": payload.get("metadata", {}).get("order_index"),
            "seed": int(args.seed),
            "warmup_count": int(args.warmup_count),
            "requested_top_k": int(args.top_k),
            "retrieval_method": PROBE_METHOD,
            "rerank": PROBE_RERANK,
            "include_staleness": False,
            "evaluation_trace": False,
            "offline": True,
            "query_order": order,
            "query_order_sha256": _sha256_text("\n".join(order)),
            "cold_query_id": str(ordered[0]["query_id"]),
            "warm_query_ids": order,
            "process_model": (
                "one worker per condition and block; the caller starts a fresh "
                "interpreter for each, so every cold row is a first query"
            ),
        },
        "queries": {
            "path": str(Path(args.queries)),
            "sha256": digest,
            "input_file_sha256": payload["input_file_sha256"],
            "digest_scope": "sorted query identifiers, text, and query metadata; trial-level condition/block metadata excluded",
            "digest_algorithm": "sha256",
            "schema_version": payload.get("schema_version"),
            "query_count": len(entries),
            "generation_id": str(payload["generation_id"]),
            "condition_id": str(payload["condition_id"]),
            "block": int(payload["block"]),
            "metadata": payload.get("metadata", {}),
        },
        "project": {
            "project_root": str(project),
            "project_id": status.get("project_id"),
            "project_name": status.get("project_name"),
            "generation_id": generation_id,
            "generation_root": str(generation_root or ""),
            "chunk_count": status.get("chunk_count"),
            "hybrid_ready": status.get("hybrid_ready"),
            "hybrid_upgrade_required": status.get("hybrid_upgrade_required"),
            "model_cache_root": str(config.model_cache_root),
            "offline": True,
        },
        "generation": {
            **_generation_versions(generation_root),
            "retrieval": status.get("retrieval"),
            "retrieval_policy_fingerprint": service_policy_fingerprint,
            "resolved_budget_settings": {
                "minimum_candidates": config.settings.minimum_candidates,
                "maximum_candidates": config.settings.maximum_candidates,
                "rerank_max_candidates": config.settings.rerank_max_candidates,
                "rerank_window_multiple": config.settings.rerank_window_multiple,
                "rerank_window_floor": config.settings.rerank_window_floor,
            },
            "reranker_model": config.reranker_model,
            "settings_provenance": dict(config.settings_provenance),
        },
        "environment": {
            "before": before_environment,
            "after": after_environment,
        },
        "resource": {
            "self_cpu_seconds_total": _seconds(self_after - self_before),
            "children_cpu_seconds_total_reaped": _seconds(children_cpu_seconds),
            "cpu_accounting": CPU_ACCOUNTING_NOTE,
            "scope": (
                "This worker's own rusage and the rusage of the children it started "
                "and reaped. No process tree was inspected and no other process on "
                "the machine was sampled."
            ),
            "peak_rss_mib": {
                "self": _peak_rss_mib(rusage(resource.RUSAGE_SELF)),
                "children": _peak_rss_mib(children_usage),
                "additive": False,
                "note": (
                    "Each figure is a maximum over its own window. They are not "
                    "added: a sum would describe no moment that existed on this "
                    "machine."
                ),
            },
        },
        "timing": {
            "service_ready_elapsed": service_ready_elapsed,
            "first_query_cold": {
                "query_id": str(ordered[0]["query_id"]),
                "started_at_seconds": _seconds(cold_started_clock - started_clock),
                "elapsed_seconds": cold_elapsed,
            },
            "startup_to_first_result_elapsed": startup_to_first_result,
            "warmup": {
                "search_count": int(args.warmup_count),
                "total_seconds": warmup_seconds,
                "measured": False,
                "note": (
                    "Outside every measured window by construction. These searches "
                    "load the embedder and the cross-encoder; the cold row above "
                    "loaded them first and its result is the initial warm-up."
                ),
            },
            "warm_phase_total_seconds": warm_seconds,
            "self_cpu_seconds_warm_phase": _seconds(
                sum(
                    float(row["self_cpu_seconds"])
                    for row in rows
                    if row["phase"] == "warm"
                )
            ),
        },
        "rows": rows,
        "warm_latency": latency_stats(
            [row["elapsed_seconds"] for row in rows if row["phase"] == "warm"]
        ),
        "cold_latency": latency_stats(
            [row["elapsed_seconds"] for row in rows if row["phase"] == "cold"]
        ),
        "observed_budget_points": _observed_points(rows),
        "rows_without_budget_point": sum(
            1 for row in rows if _budget_point(row) is None
        ),
        "row_semantics": {
            "cold": (
                "the first query of a fresh interpreter: it pays the index and "
                "model load, and its elapsed time is not comparable with a warm "
                "row's"
            ),
            "warm": (
                "a query timed after the cold row and the warm-up searches, so the "
                "embedder and the cross-encoder were already loaded"
            ),
            "self_cpu_seconds": (
                "this worker's own user plus system time across that one search; "
                "the gateway's CPU is in the resource totals and in no row"
            ),
            "returned_chunk_ids": (
                "rank-ordered identifiers, recorded so a repeated worker can be "
                "compared for determinism; nothing is said about whether they "
                "answer the query"
            ),
        },
        "data_state": {
            "unchanged": not changed,
            "changed_files": changed,
            "digests": after_state,
            "note": (
                "Digests of the project's portable state, its selected generation, "
                "and the current pointer, taken after the read-only status call and "
                "again after the last search. The runtime directory is not hashed: "
                "the gateway writes logs and handoff files there, so hashing it "
                "would report this harness's own traffic as a change."
            ),
        },
    }
    _write_json(report_path, report)
    return report


def _generation_versions(generation_root: Path | None) -> dict[str, Any]:
    """The schema and policy versions this generation was built under.

    A latency figure describes one generation under one policy set. Naming the
    versions is what lets a later reader tell whether two reports measured the
    same engine at all; without them two numbers from incompatible generations
    look like two measurements of one thing.
    """

    if generation_root is None:
        return {}
    try:
        manifest = json.loads(
            (generation_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, dict):
        return {}
    return {
        "generation_id": manifest.get("generation_id"),
        "schema_version": manifest.get("schema_version"),
        "extraction_policy_version": manifest.get("extraction_policy_version"),
        "cleaning_policy_version": manifest.get("cleaning_policy_version"),
        "artifact_policy_version": manifest.get("artifact_policy_version"),
        "ingestion_identity_policy_version": manifest.get(
            "ingestion_identity_policy_version"
        ),
        "created_at": manifest.get("created_at"),
        "document_count": manifest.get("document_count"),
        "chunk_count": manifest.get("chunk_count"),
        "metadata_revision": manifest.get("metadata_revision"),
        "chunking": manifest.get("chunking"),
    }


def _budget_point(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """One row's actual ``(candidate_depth, rerank_window)``, or ``None``.

    A cap the settings requested is not a window the engine ran, so the point is
    read back from the row rather than from the settings. A row that carries
    neither figure contributes no point.
    """

    depth = row.get("candidate_depth")
    window = row.get("rerank_window")
    if isinstance(depth, bool) or isinstance(window, bool):
        return None
    if not isinstance(depth, int) or not isinstance(window, int):
        return None
    return depth, window


def _observed_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The distinct ``(candidate depth, rerank window)`` points these rows used."""

    seen = {point for point in (_budget_point(row) for row in rows) if point}
    return [
        {"candidate_depth": depth, "rerank_window": window}
        for depth, window in sorted(seen)
    ]


def _report_rows(
    report: Mapping[str, Any], phase: str | None = None
) -> list[dict[str, Any]]:
    """One report's rows, optionally only those of one phase."""

    rows = report.get("rows")
    if not isinstance(rows, list):
        return []
    selected = [row for row in rows if isinstance(row, Mapping)]
    if phase is None:
        return [dict(row) for row in selected]
    return [dict(row) for row in selected if str(row.get("phase")) == phase]


# --------------------------------------------------------------------------
# The summarizer. It reads worker reports, runs no search, starts no process,
# and computes no test statistic and no winner.
# --------------------------------------------------------------------------


def _section(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """One report section, or an empty mapping when it is absent."""

    value = report.get(name)
    return value if isinstance(value, Mapping) else {}


def _report_identity(report: Mapping[str, Any]) -> dict[str, str]:
    """The three identities two reports must share before they may be compared."""

    return {
        "query_set_sha256": str(_section(report, "queries").get("sha256") or ""),
        "generation_id": str(_section(report, "generation").get("generation_id") or ""),
        "scorer_script_sha256": str(
            _section(report, "harness").get("script_sha256") or ""
        ),
    }


def _condition_of(report: Mapping[str, Any]) -> str:
    condition = str(_section(report, "probe").get("condition_id") or "").strip()
    if not condition:
        raise ProbeError("A report names no condition_id")
    return condition


def _block_of(report: Mapping[str, Any]) -> int:
    block = _section(report, "probe").get("block")
    if isinstance(block, bool) or not isinstance(block, int):
        raise ProbeError("A report names no integer block")
    return block


def _require_comparable(reports: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Refuse reports that cannot sit in one comparison, and say which identity.

    A summary over two query sets would be comparing different work; one over two
    generations would attribute a difference to whichever condition ran second;
    one over two scorers would attribute it to the harness. Each is refused by
    name rather than averaged through.
    """

    if not reports:
        raise ProbeError("No reports to summarize")
    identities = [_report_identity(report) for report in reports]
    for report in reports:
        if not isinstance(report, Mapping):
            raise ProbeError("A report in the summary is not an object")
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ProbeError(
                f"Report schema {report.get('schema_version')!r} is not "
                f"{REPORT_SCHEMA_VERSION}; this summarizer reads its own version "
                "and cannot compare across versions."
            )
    differing = sorted(
        name
        for name in ("query_set_sha256", "generation_id", "scorer_script_sha256")
        if len({identity[name] for identity in identities}) > 1
    )
    if differing:
        detail = "; ".join(
            f"{name}: {sorted({identity[name] for identity in identities})}"
            for name in differing
        )
        raise ProbeError(
            f"These reports cannot be summarized together because their "
            f"{' and their '.join(differing)} differ ({detail}). Compare runs that "
            "share one frozen query set, one generation, and one scorer."
        )
    return identities[0]


def _warm_index(
    reports: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Every warm row of these reports, keyed by the block and query that made it.

    The key is what makes a pairing honest: a block that did not run, or that ran
    a different query list, simply has no partner, and the summary reports the
    missing partner rather than pairing the nearest thing it has. A repeated key
    is refused rather than resolved, because two rows claiming the same block and
    query would be two measurements of one position and the summary would not
    know which to pair.
    """

    index: dict[tuple[int, str], dict[str, Any]] = {}
    for report in reports:
        block = _block_of(report)
        for row in _report_rows(report, "warm"):
            key = (block, str(row.get("query_id") or ""))
            if key in index:
                raise ProbeError(
                    f"Two warm rows in block {block} both claim query {key[1]!r}; a "
                    "paired comparison cannot choose between them."
                )
            index[key] = row
    return index


def _balance(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Which blocks each condition ran, and whether that is a balanced design.

    A condition measured in fewer blocks than another has fewer observations, and
    its percentiles are computed over a smaller sample. The summary says so
    rather than letting a p95 over two rows stand beside a p95 over six.
    """

    blocks = {
        condition: sorted({_block_of(report) for report in reports})
        for condition, reports in grouped.items()
    }
    every = sorted({block for values in blocks.values() for block in values})
    missing = {
        condition: sorted(set(every) - set(values))
        for condition, values in sorted(blocks.items())
        if set(values) != set(every)
    }
    return {
        "blocks": every,
        "blocks_by_condition": {
            condition: values for condition, values in sorted(blocks.items())
        },
        "every_condition_ran_every_block": not missing,
        "conditions_missing_blocks": missing,
        "note": (
            "A balanced design runs every condition in every block. When this is "
            "false the conditions have different sample sizes and their percentiles "
            "are not comparable at face value."
        ),
    }


def _per_query_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each query's warm distribution over the blocks that ran it."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("query_id") or ""), []).append(row)
    return {
        query_id: {
            "query_id": query_id,
            "repeat_count": len(items),
            "elapsed": latency_stats([item.get("elapsed_seconds") for item in items]),
            "self_cpu": latency_stats([item.get("self_cpu_seconds") for item in items]),
        }
        for query_id, items in sorted(grouped.items())
    }


def _repeat_spread(
    per_query: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """How far each query moved between its repeated blocks.

    This is the sampling uncertainty a reader can act on: the spread of the
    repeats themselves, per query, rather than a p-value computed from a model of
    what the noise should look like. Three repeats give a range, not a confidence
    interval, and the summary says that too.
    """

    spread: dict[str, dict[str, float | int | None]] = {}
    for query_id, entry in per_query.items():
        elapsed = entry.get("elapsed") or {}
        lowest = elapsed.get("min_seconds")
        highest = elapsed.get("max_seconds")
        spread[query_id] = {
            "repeat_count": entry.get("repeat_count"),
            "min_seconds": lowest,
            "p50_seconds": elapsed.get("p50_seconds"),
            "max_seconds": highest,
            "spread_seconds": (
                _seconds(float(highest) - float(lowest))
                if isinstance(highest, (int, float))
                and isinstance(lowest, (int, float))
                else None
            ),
        }
    return spread


def _determinism(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the same query returned the same thing in every block.

    Ranking and the two budget figures are compared per query across every
    observation of it. A mismatch is reported as a mismatch and every row is kept:
    dropping the odd row out would hide the one block that behaved differently,
    which is the only reason to look.
    """

    observations: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        block = _block_of(report)
        for row in _report_rows(report):
            observations.setdefault(str(row.get("query_id") or ""), []).append(
                {"block": block, "phase": row.get("phase"), "row": row}
            )
    mismatched: list[dict[str, Any]] = []
    compared = 0
    for query_id, entries in sorted(observations.items()):
        if len(entries) < 2:
            continue
        compared += 1
        first = entries[0]
        differences: list[dict[str, Any]] = []
        for entry in entries[1:]:
            for field in ("returned_chunk_ids", "candidate_depth", "rerank_window"):
                if entry["row"].get(field) != first["row"].get(field):
                    differences.append(
                        {
                            "block": entry["block"],
                            "phase": entry["phase"],
                            "field": field,
                            "first_block": first["block"],
                            "first_phase": first["phase"],
                        }
                    )
        if differences:
            mismatched.append(
                {
                    "query_id": query_id,
                    "observation_count": len(entries),
                    "differences": differences,
                }
            )
    return {
        "queries_compared": compared,
        "queries_consistent": compared - len(mismatched),
        "mismatched_queries": mismatched,
        "fields_compared": (
            "returned_chunk_ids, candidate_depth, rerank_window; every row is kept "
            "either way"
        ),
    }


def _resource_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The CPU and peak memory of every worker of one condition.

    Totals are summed over workers because each worker is a separate process and
    its cost is its own. Peaks are not: the self peak and the reaped children's
    peak are maxima over different windows, so the summary reports the larger of
    each kind across workers and never their sum.
    """

    workers: list[dict[str, Any]] = []
    for report in reports:
        resource_section = _section(report, "resource")
        peaks = resource_section.get("peak_rss_mib")
        peaks = peaks if isinstance(peaks, Mapping) else {}
        workers.append(
            {
                "block": _block_of(report),
                "self_cpu_seconds": resource_section.get("self_cpu_seconds_total"),
                "children_cpu_seconds_total_reaped": resource_section.get(
                    "children_cpu_seconds_total_reaped"
                ),
                "peak_rss_mib_self": peaks.get("self"),
                "peak_rss_mib_children": peaks.get("children"),
            }
        )
    self_peaks = [
        value
        for value in (worker["peak_rss_mib_self"] for worker in workers)
        if isinstance(value, (int, float))
    ]
    child_peaks = [
        value
        for value in (worker["peak_rss_mib_children"] for worker in workers)
        if isinstance(value, (int, float))
    ]
    return {
        "family": "cpu_seconds",
        "statistic": "sum over workers for totals, largest single worker for peaks",
        "self_cpu_seconds_total": _seconds(
            sum(float(_finite(worker["self_cpu_seconds"]) or 0.0) for worker in workers)
        ),
        "children_cpu_seconds_total_reaped": _seconds(
            sum(
                float(_finite(worker["children_cpu_seconds_total_reaped"]) or 0.0)
                for worker in workers
            )
        ),
        "peak_rss_mib": {
            "self_max": max(self_peaks) if self_peaks else None,
            "children_max": max(child_peaks) if child_peaks else None,
            "additive": False,
            "note": (
                "The largest single-worker peak of each kind. The two are not "
                "added: they are maxima over different processes and different "
                "windows, and their sum describes no moment on this machine."
            ),
        },
        "workers": sorted(workers, key=lambda worker: worker["block"]),
        "accounting": CPU_ACCOUNTING_NOTE,
    }


def _condition_summary(
    condition: str, reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One condition's cold figures, warm figures, repeats, and resource totals."""

    cold_rows = [row for report in reports for row in _report_rows(report, "cold")]
    warm_rows = [row for report in reports for row in _report_rows(report, "warm")]
    per_query = _per_query_stats(warm_rows)
    repeats = sorted({entry["repeat_count"] for entry in per_query.values()})
    return {
        "condition_id": condition,
        "worker_count": len(reports),
        "workers": [
            {
                "block": _block_of(report),
                "order_index": _section(report, "probe").get("order_index"),
                "started_at": _section(report, "dates").get("started_at"),
                "service_ready_elapsed": _section(report, "timing").get(
                    "service_ready_elapsed"
                ),
                "first_query_cold": _section(report, "timing").get("first_query_cold"),
                "startup_to_first_result_elapsed": _section(report, "timing").get(
                    "startup_to_first_result_elapsed"
                ),
                "warm_row_count": len(_report_rows(report, "warm")),
                "query_order": _section(report, "probe").get("query_order"),
            }
            for report in sorted(reports, key=_block_of)
        ],
        "cold": {
            "family": "latency_seconds",
            "statistic": "nearest rank over one figure per worker",
            "worker_count": len(cold_rows),
            "service_ready_elapsed": latency_stats(
                [
                    _section(report, "timing").get("service_ready_elapsed")
                    for report in reports
                ]
            ),
            "first_query_elapsed": latency_stats(
                [row.get("elapsed_seconds") for row in cold_rows]
            ),
            "startup_to_first_result": latency_stats(
                [
                    _section(report, "timing").get("startup_to_first_result_elapsed")
                    for report in reports
                ]
            ),
            "note": (
                "One cold row per worker, because every worker is a fresh "
                "interpreter. These figures include the process's own imports, the "
                "gateway handshake, and reading cached files that the operating "
                "system may already have in memory."
            ),
        },
        "warm": {
            "family": "latency_seconds",
            "statistic": "nearest rank over one figure per measured row",
            "row_count": len(warm_rows),
            "distinct_query_count": len(
                {str(row.get("query_id")) for row in warm_rows}
            ),
            "repeat_counts": repeats,
            "overall": latency_stats([row.get("elapsed_seconds") for row in warm_rows]),
            "self_cpu": latency_stats(
                [row.get("self_cpu_seconds") for row in warm_rows]
            ),
            "per_query": per_query,
            "note": (
                "overall pools the rows of every query, so a row is one "
                "measurement and a query is measured once per block. The rows of "
                "one query are repeats of one question and are not independent "
                "queries; per_query and repeat_spread are where a repeated "
                "measurement is read."
            ),
        },
        "repeat_spread": {
            "family": "latency_seconds",
            "statistic": "range across the blocks that ran the query",
            "per_query": _repeat_spread(per_query),
        },
        "determinism": _determinism(reports),
        "resource": _resource_summary(reports),
        "observed_budget_points": _observed_points(warm_rows + cold_rows),
    }


def _paired_ratios(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_condition: str | None,
) -> dict[str, Any]:
    """Each condition's per-query wall and CPU ratios against the baseline.

    A ratio is only meaningful between two measurements of the same query in the
    same block, so the pairing is exactly that: the blocks both conditions ran,
    with the same query list. A block whose query list differs, or whose baseline
    row is missing, is named as unpaired instead of being paired with whatever was
    nearest.

    A ratio whose baseline is zero is dropped and counted, because dividing by a
    zero would be a figure about nothing.
    """

    if baseline_condition is None or baseline_condition not in grouped:
        return {
            "baseline_condition_id": baseline_condition,
            "paired": False,
            "note": (
                "No baseline condition was named, or the named one is not in this "
                "set, so no ratio is reported. A ratio against a condition chosen "
                "after seeing the numbers would be a comparison fitted to its "
                "outcome."
            ),
        }
    baseline_index = _warm_index(grouped[baseline_condition])
    baseline_blocks = sorted({block for block, _ in baseline_index})
    conditions: dict[str, Any] = {}
    for condition, reports in sorted(grouped.items()):
        if condition == baseline_condition:
            continue
        index = _warm_index(reports)
        blocks = sorted({block for block, _ in index})
        common = sorted(set(blocks) & set(baseline_blocks))
        unpaired_blocks = sorted(set(blocks) ^ set(baseline_blocks))
        mismatched_blocks: list[dict[str, Any]] = []
        wall_by_query: dict[str, list[float]] = {}
        cpu_by_query: dict[str, list[float]] = {}
        zero_baselines = 0
        for block in common:
            here = {query for candidate, query in index if candidate == block}
            there = {query for candidate, query in baseline_index if candidate == block}
            if here != there:
                mismatched_blocks.append(
                    {
                        "block": block,
                        "only_here": sorted(here - there),
                        "only_in_baseline": sorted(there - here),
                    }
                )
                continue
            for query in sorted(here):
                row = index[(block, query)]
                base = baseline_index[(block, query)]
                base_wall = _finite(base.get("elapsed_seconds"))
                base_cpu = _finite(base.get("self_cpu_seconds"))
                wall = _finite(row.get("elapsed_seconds"))
                cpu = _finite(row.get("self_cpu_seconds"))
                if base_wall and wall is not None:
                    wall_by_query.setdefault(query, []).append(wall / base_wall)
                elif wall is not None:
                    zero_baselines += 1
                if base_cpu and cpu is not None:
                    cpu_by_query.setdefault(query, []).append(cpu / base_cpu)
                elif cpu is not None:
                    zero_baselines += 1
        mismatched = {entry["block"] for entry in mismatched_blocks}
        conditions[condition] = {
            "baseline_condition_id": baseline_condition,
            "paired_blocks": [block for block in common if block not in mismatched],
            "unpaired_blocks": unpaired_blocks,
            "blocks_with_different_query_lists": mismatched_blocks,
            "zero_baseline_observations_dropped": zero_baselines,
            "per_query": {
                query_id: {
                    "query_id": query_id,
                    "pair_count": len(wall_by_query.get(query_id, [])),
                    "wall_ratio": _ratio_stats(wall_by_query.get(query_id, [])),
                    "self_cpu_ratio": _ratio_stats(cpu_by_query.get(query_id, [])),
                }
                for query_id in sorted(set(wall_by_query) | set(cpu_by_query))
            },
        }
    return {
        "baseline_condition_id": baseline_condition,
        "paired": True,
        "family": "ratio",
        "statistic": "nearest rank over paired per-query observations",
        "conditions": conditions,
        "note": (
            "Ratios are paired by block and query, never by rank in a list of "
            "durations. No test statistic, no p-value, and no winner is computed: "
            "a significance claim from a handful of blocks over one corpus would "
            "outrun the measurement, and the parent plan records that latency "
            "decisions need a repeat protocol rather than a test."
        ),
    }


def aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    baseline_condition: str | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Group worker reports by condition into one comparison.

    The app owns the arithmetic and nothing else here reaches beyond the reports
    it is given: no project is opened, no search runs, no relevance is judged, and
    no condition is declared better than another. Cold figures and warm figures
    are reported apart, repeats are never counted as extra queries, a determinism
    mismatch is named rather than dropped, and a pairing is made only where a
    block and its query list exist on both sides.
    """

    identity = _require_comparable(reports)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(_condition_of(report), []).append(report)
    balance = _balance(grouped)
    conditions = {
        condition: _condition_summary(condition, items)
        for condition, items in sorted(grouped.items())
    }
    observed = sorted(
        {
            (point["candidate_depth"], point["rerank_window"])
            for summary in conditions.values()
            for point in summary["observed_budget_points"]
        }
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "research-rag retrieval latency summary",
        "generated_at": _utc_now(),
        "notice": (
            "Latency and resource totals over worker reports. No relevance, grade, "
            "or label is read or derived, and no test statistic or winner is "
            "computed. A p95 over a handful of blocks is a percentile of those "
            "blocks, not a stable bound, which is why every section carries its "
            "counts."
        ),
        "harness": harness_provenance(),
        "inputs": {
            "report_count": len(reports),
            "identity": identity,
            "reports": [dict(source) for source in (sources or [])],
            "condition_count": len(conditions),
        },
        "config": {
            "fixed_points": [
                {"candidate_depth": depth, "rerank_window": window}
                for depth, window in PROTOCOL_POINTS
            ],
            "observed_points": [
                {"candidate_depth": depth, "rerank_window": window}
                for depth, window in observed
            ],
            "observed_points_outside_protocol": [
                {"candidate_depth": depth, "rerank_window": window}
                for depth, window in observed
                if (depth, window) not in PROTOCOL_POINTS
            ],
            "note": (
                "The three fixed points are the conditions this experiment "
                "compares. The observed points are read back from the rows, because "
                "a cap the settings asked for is not a window the engine ran. An "
                "observed point outside the three is reported rather than averaged "
                "in with them."
            ),
        },
        "balance": balance,
        "conditions": conditions,
        "paired_ratios": _paired_ratios(grouped, baseline_condition),
        "uncertainty": {
            "method": "per-query spread over repeated blocks",
            "per_condition": {
                condition: summary["repeat_spread"]["per_query"]
                for condition, summary in conditions.items()
            },
            "note": (
                "Each figure is the range a query moved across the blocks that ran "
                "it. Three repeats bound a range and do not establish a confidence "
                "interval, so no interval is computed and no threshold is applied."
            ),
        },
    }


# --------------------------------------------------------------------------
# The summarize mode: read a manifest of worker reports, verify it, aggregate.
# --------------------------------------------------------------------------


def load_report_manifest(
    path: Path,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], str | None]:
    """Read a report manifest, verifying every digest it declares.

    A manifest entry may carry the digest the caller recorded when the report was
    written. When it does, the file's own bytes are hashed and compared, and a
    mismatch is refused: a report edited after the fact would otherwise be
    summarized as though it were the run it claims to be. An entry with no digest
    is accepted and recorded as one, because an unverified report is weaker
    evidence rather than a wrong one.
    """

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Cannot read the report manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ProbeError(f"Unsupported report manifest: {path}")
    entries = manifest.get("reports")
    if not isinstance(entries, list) or not entries:
        raise ProbeError(f"The report manifest lists no reports: {path}")
    reports: list[Mapping[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not str(entry.get("path") or "").strip():
            raise ProbeError(f"Report manifest entry names no path: {entry!r}")
        report_path = Path(str(entry["path"])).expanduser()
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProbeError(f"Cannot read a manifest report: {report_path}") from exc
        actual = _sha256_file(report_path) or ""
        expected = str(entry.get("sha256") or "")
        if expected and expected != actual:
            raise ProbeError(
                f"Report digest mismatch for {report_path}: the manifest records "
                f"{expected} and the file is {actual}. The report was replaced "
                "after the manifest was written; regenerate it and the manifest "
                "together."
            )
        reports.append(payload)
        sources.append(
            {
                "path": str(report_path),
                "sha256": actual,
                "sha256_verified": bool(expected),
                "condition_id": _condition_of(payload),
                "block": _block_of(payload),
            }
        )
    baseline = manifest.get("baseline_condition_id")
    return (
        reports,
        sources,
        str(baseline) if isinstance(baseline, str) and baseline.strip() else None,
    )


# --------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-retrieval",
        description=(
            "Measure retrieval latency for one condition on a disposable project, "
            "or aggregate the reports several workers wrote."
        ),
    )
    parser.add_argument(
        "--project",
        type=Path,
        help=(
            "Disposable project root to measure, holding the .research-rag state. "
            "Not used with --summarize."
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        help="Frozen query set naming this condition, block, and generation.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Where to write this run's new JSON report. An existing path is refused.",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=DEFAULT_WARMUP_COUNT,
        help=(
            "Searches run after the cold query and outside every measured window, "
            f"to load the embedder and the cross-encoder (default: {DEFAULT_WARMUP_COUNT})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Seed for the query order, so a repeated worker runs the same order "
            f"(default: {DEFAULT_SEED})."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Requested depth, 1 to 50 (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        help=(
            "Manifest of worker reports to aggregate. Reads only; --project and "
            "--queries are not used with it."
        ),
    )
    parser.add_argument(
        "--baseline-condition",
        help=(
            "Condition the paired per-query ratios compare against. Overrides the "
            "manifest's baseline_condition_id."
        ),
    )
    return parser


def _validate_mode(args: Any) -> None:
    """Refuse a command line that names neither mode, or both at once."""

    if args.summarize is not None:
        if args.project is not None or args.queries is not None:
            raise ProbeError(
                "--summarize aggregates reports and measures nothing, so --project "
                "and --queries do not belong with it."
            )
        return
    if args.project is None or args.queries is None:
        raise ProbeError(
            "A probe run needs --project and --queries, or --summarize for a "
            "manifest of worker reports."
        )


def summarize_reports(args: Any) -> dict[str, Any]:
    """Aggregate the reports a manifest names, and write the summary."""

    report_path = _report_destination(Path(args.report))
    reports, sources, manifest_baseline = load_report_manifest(Path(args.summarize))
    baseline = args.baseline_condition or manifest_baseline
    summary = aggregate_reports(reports, baseline_condition=baseline, sources=sources)
    _write_json(report_path, summary)
    return summary


def print_probe(report: Mapping[str, Any], path: Path) -> None:
    """The console view: what was measured, and under what cache state."""

    timing = _section(report, "timing")
    cold = timing.get("first_query_cold")
    cold = cold if isinstance(cold, Mapping) else {}
    warm = _section(report, "warm_latency")
    resource_section = _section(report, "resource")
    data_state = _section(report, "data_state")
    print(
        f"{_section(report, 'probe').get('condition_id')} "
        f"block {_section(report, 'probe').get('block')}: "
        f"{_section(report, 'queries').get('query_count')} queries, "
        f"{PROBE_METHOD}{'+rerank' if PROBE_RERANK else ''} at "
        f"top_k={_section(report, 'probe').get('requested_top_k')}"
    )
    print(
        f"  service ready {timing.get('service_ready_elapsed')} s, "
        f"cold query {cold.get('elapsed_seconds')} s, "
        f"startup to first result "
        f"{timing.get('startup_to_first_result_elapsed')} s"
    )
    print(
        f"  warm rows {warm.get('count')}: p50 {warm.get('p50_seconds')} s, "
        f"p95 {warm.get('p95_seconds')} s, max {warm.get('max_seconds')} s, "
        f"mean {warm.get('mean_seconds')} s"
    )
    print(
        f"  self CPU {resource_section.get('self_cpu_seconds_total')} s, reaped "
        f"children CPU "
        f"{resource_section.get('children_cpu_seconds_total_reaped')} s; peaks are "
        "reported separately and never added"
    )
    print(
        "  warm-up searches "
        f"{_section(timing, 'warmup').get('search_count')} outside every measured "
        f"window; project files unchanged: {data_state.get('unchanged')}"
    )
    print(f"report written to {path}")


def print_summary(summary: Mapping[str, Any], path: Path) -> None:
    """The console view of a comparison: cold, warm, repeats, and balance."""

    conditions = summary.get("conditions")
    conditions = conditions if isinstance(conditions, Mapping) else {}
    balance = _section(summary, "balance")
    print(
        f"{len(conditions)} condition(s) over blocks "
        f"{balance.get('blocks')}; balanced: "
        f"{balance.get('every_condition_ran_every_block')}"
    )
    header = (
        f"{'condition':<16}{'workers':>8}{'ready p50':>11}{'cold p50':>10}"
        f"{'warm p50':>10}{'warm p95':>10}{'warm max':>10}{'selfCPU':>10}"
        f"{'childCPU':>10}{'mismatch':>10}"
    )
    print(header)
    print("-" * len(header))
    for condition, summary_for_condition in conditions.items():
        entry = (
            summary_for_condition if isinstance(summary_for_condition, Mapping) else {}
        )
        cold = _section(entry, "cold")
        warm = _section(entry, "warm")
        determinism = _section(entry, "determinism")
        totals = _section(_section(entry, "resource"), "peak_rss_mib")
        print(
            f"{condition:<16}{entry.get('worker_count', 0):>8}"
            f"{_shown(_section(cold, 'service_ready_elapsed').get('p50_seconds')):>11}"
            f"{_shown(_section(cold, 'first_query_elapsed').get('p50_seconds')):>10}"
            f"{_shown(_section(warm, 'overall').get('p50_seconds')):>10}"
            f"{_shown(_section(warm, 'overall').get('p95_seconds')):>10}"
            f"{_shown(_section(warm, 'overall').get('max_seconds')):>10}"
            f"{_shown(_section(entry, 'resource').get('self_cpu_seconds_total')):>10}"
            f"{_shown(_section(entry, 'resource').get('children_cpu_seconds_total_reaped')):>10}"
            f"{determinism.get('queries_compared', 0) - determinism.get('queries_consistent', 0):>10}"
        )
        peaks = (
            f"  peak RSS self {totals.get('self_max')} MiB, "
            f"children {totals.get('children_max')} MiB (not summed)"
        )
        print(peaks)
    paired = _section(summary, "paired_ratios")
    print(
        f"paired ratios against {paired.get('baseline_condition_id')}: "
        f"{'yes' if paired.get('paired') else 'no'}; no p-value and no winner is "
        "computed"
    )
    print(f"summary written to {path}")


def _shown(value: Any) -> str:
    """A figure for one column, or a blank where none was measured."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.3f}"


def main() -> None:
    args = _parser().parse_args()
    try:
        _validate_mode(args)
        if args.summarize is not None:
            summary = summarize_reports(args)
            print_summary(summary, Path(args.report))
        else:
            report = asyncio.run(run_probe(args))
            print_probe(report, Path(args.report))
    except ProbeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
