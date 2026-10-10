"""Tests for the retrieval latency probe.

Three things are checked here. The protocol: which search is the cold one, that
the warm-up searches fall outside every measured window, that each warm row times
only its own search, and that the startup figures run from the process's own
origin. The payload contract: the probe reads the fields the app's search answer
really carries, checked against a real ``ResearchService`` over the repository's
deterministic fakes, so a row cannot be pinned to a shape the engine cannot
produce. And the arithmetic: nearest-rank percentiles that are finite, reported
units, repeats that are never counted as extra queries, a determinism mismatch
that is named rather than dropped, and a pairing made only where a block and its
query list exist on both sides.

No test here measures latency. Every clock and every rusage record is scripted,
so the figures under test are exact and a change in the harness shows up as a
change in a number rather than as noise.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Loaded by path rather than imported: scripts/ is not a package.
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_retrieval",
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark_retrieval.py",
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["benchmark_retrieval"] = probe
_SPEC.loader.exec_module(probe)
ProbeError = probe.ProbeError


# --------------------------------------------------------------------------
# Scripted time and CPU. A fake search advances both by a fixed amount, so every
# window in the report is exact and the assertions can be exact too.
# --------------------------------------------------------------------------


class _Clock:
    """A monotonic clock a fake gateway or search advances."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(frozen=True)
class _Usage:
    """One rusage record, with the fields the probe reads."""

    ru_utime: float
    ru_stime: float
    ru_maxrss: int


class _Rusage:
    """rusage records a fake search charges, with fixed peaks."""

    def __init__(
        self,
        *,
        children_seconds: float = 4.5,
        self_peak_kib: int = 262_144,
        children_peak_kib: int = 524_288,
    ) -> None:
        self.self_seconds = 0.0
        self.children_seconds = children_seconds
        self.self_peak_kib = self_peak_kib
        self.children_peak_kib = children_peak_kib

    def charge(self, seconds: float) -> None:
        self.self_seconds += seconds

    def __call__(self, who: int) -> _Usage:
        if who == resource.RUSAGE_SELF:
            return _Usage(
                self.self_seconds / 2, self.self_seconds / 2, self.self_peak_kib
            )
        return _Usage(
            self.children_seconds / 2,
            self.children_seconds / 2,
            self.children_peak_kib,
        )


class _FakeClient:
    """The gateway seam: a startup cost on entry, a shutdown cost on exit."""

    def __init__(self, clock: _Clock, *, startup: float, shutdown: float) -> None:
        self._clock = clock
        self._startup = startup
        self._shutdown = shutdown
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> object:
        self.entered = True
        self._clock.advance(self._startup)
        return object()

    async def __aexit__(self, *exc_info: object) -> bool:
        self.exited = True
        self._clock.advance(self._shutdown)
        return False


class _ScriptedService:
    """A service whose searches cost scripted wall time and scripted CPU.

    The steps are consumed in order and the last one repeats, so a run with more
    searches than steps still answers every one of them.
    """

    retrieval_policy_fingerprint = "policy-fingerprint-1"

    def __init__(
        self,
        *,
        status: dict[str, Any],
        clock: _Clock,
        rusage: _Rusage,
        steps: list[tuple[float, float, dict[str, Any]]],
    ) -> None:
        self._status = status
        self._clock = clock
        self._rusage = rusage
        self._steps = list(steps)
        self.search_calls = 0
        self.calls: list[dict[str, Any]] = []

    async def status(self) -> dict[str, Any]:
        return dict(self._status)

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        wall, cpu, payload = self._steps[min(self.search_calls, len(self._steps) - 1)]
        self.search_calls += 1
        self._clock.advance(wall)
        self._rusage.charge(cpu)
        return json.loads(json.dumps(payload))


def _payload(
    chunk_ids: tuple[str, ...] = ("c1", "c2", "c3"),
    *,
    candidate_depth: int = 40,
    candidate_count: int = 40,
    rerank_window: int = 20,
    reranked: bool = True,
    requested: bool = True,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A search answer in the shape `SearchWorkflow.search` returns."""

    return {
        "retrieval_method": "hybrid",
        "reranked": reranked,
        "rerank_requested": requested,
        "rerank_fallback": fallback,
        "rerank_window": rerank_window,
        "candidate_depth": candidate_depth,
        "candidate_count": candidate_count,
        "result_count": len(chunk_ids),
        "distinct_reference_count": len(chunk_ids),
        "hits": [
            {
                "chunk_id": chunk_id,
                "document_id": f"doc-{chunk_id}",
                "text": f"Passage {chunk_id} about cobalt and labour.",
            }
            for chunk_id in chunk_ids
        ],
    }


def _searches(
    count: int, *, wall: float, cpu: float
) -> list[tuple[float, float, dict[str, Any]]]:
    """Scripted steps that all answer with the same shape of payload."""

    return [
        (wall, cpu, _payload()) for _ in range(count)
    ]  # --------------------------------------------------------------------------


# The disposable project a synthetic run measures. Nothing is ingested: the
# service is a stub, so only the files the probe hashes are written.
# --------------------------------------------------------------------------


GENERATION_ID = "20261001T000000Z-probe"
QUERY_SHA = "queries-sha-256"
SCRIPT_SHA = "scorer-sha-256"


def _project(root: Path) -> Path:
    project = root / "project"
    (project / "sources").mkdir(parents=True)
    return project


def _generation(project: Path, generation_id: str = GENERATION_ID) -> Path:
    """A generation with the files the probe digests, and nothing else."""

    generation = project / ".research-rag" / "runtime" / "generations" / generation_id
    (generation / "chunks").mkdir(parents=True)
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "generation_id": generation_id,
                "schema_version": 3,
                "extraction_policy_version": 2,
                "cleaning_policy_version": 4,
                "artifact_policy_version": 5,
                "ingestion_identity_policy_version": 1,
                "created_at": "2026-10-01T00:00:00.000000Z",
                "document_count": 2,
                "chunk_count": 12,
                "metadata_revision": "metadata-revision",
                "chunking": {
                    "backend": "UltraRAG token chunker",
                    "chunk_size": 384,
                    "chunk_overlap": 64,
                },
                "retrieval": {
                    "default_method": "hybrid",
                    "available_methods": ["bm25", "dense", "hybrid"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (generation / "chunks" / "chunks.jsonl").write_text(
        '{"chunk_id": "c1"}\n', encoding="utf-8"
    )
    (project / ".research-rag" / "runtime" / "current.json").write_text(
        json.dumps({"generation_id": generation_id}), encoding="utf-8"
    )
    return generation


def _query_set(
    path: Path,
    *,
    condition_id: str = "c40r20",
    block: int = 0,
    generation_id: str = GENERATION_ID,
    queries: tuple[tuple[Any, ...], ...] = (
        ("q1", "cobalt labour"),
        ("q2", "quartz seams"),
    ),
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
) -> Path:
    # A third element on a query is that query's own declared metadata.
    records = [
        {"query_id": entry[0], "query": entry[1]}
        | ({"metadata": entry[2]} if len(entry) > 2 else {})
        for entry in queries
    ]
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "queries": records,
        "block": block,
        "condition_id": condition_id,
        "generation_id": generation_id,
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _status(
    generation_root: Path,
    *,
    generation_id: str = GENERATION_ID,
    hybrid_ready: bool = True,
) -> dict[str, Any]:
    return {
        "project_id": "project-id",
        "project_name": "probe-project",
        "generation_id": generation_id,
        "generation_root": str(generation_root),
        "chunk_count": 12,
        "hybrid_ready": hybrid_ready,
        "hybrid_upgrade_required": not hybrid_ready,
        "retrieval": {
            "default_method": "hybrid",
            "available_methods": ["bm25", "dense", "hybrid"],
        },
    }


def _args(
    project: Path,
    queries: Path,
    report: Path,
    *,
    warmup_count: int = 3,
    seed: int = 0,
    top_k: int = 10,
) -> argparse.Namespace:
    return argparse.Namespace(
        project=project,
        queries=queries,
        report=report,
        warmup_count=warmup_count,
        seed=seed,
        top_k=top_k,
        summarize=None,
        baseline_condition=None,
    )


def _run(
    tmp_path: Path,
    *,
    steps: list[tuple[float, float, dict[str, Any]]],
    queries: tuple[tuple[Any, ...], ...] = (
        ("q1", "cobalt labour"),
        ("q2", "quartz seams"),
    ),
    warmup_count: int = 3,
    seed: int = 0,
    top_k: int = 10,
    condition_id: str = "c40r20",
    block: int = 0,
    generation_id: str = GENERATION_ID,
    startup: float = 0.4,
    shutdown: float = 0.1,
    status_overrides: dict[str, Any] | None = None,
    query_set_generation_id: str | None = None,
) -> tuple[dict[str, Any], _ScriptedService, _Clock]:
    """Measure one synthetic condition and return the report, service, and clock."""

    project = _project(tmp_path)
    generation_root = _generation(project, generation_id)
    queries_path = _query_set(
        tmp_path / "queries.json",
        condition_id=condition_id,
        block=block,
        generation_id=query_set_generation_id or generation_id,
        queries=queries,
    )
    clock = _Clock()
    rusage = _Rusage()
    status = _status(generation_root, generation_id=generation_id)
    status.update(status_overrides or {})
    service = _ScriptedService(
        status=status,
        clock=clock,
        rusage=rusage,
        steps=steps,
    )

    def client_scope(config: Any) -> _FakeClient:
        return _FakeClient(clock, startup=startup, shutdown=shutdown)

    report = asyncio.run(
        probe.run_probe(
            _args(
                project,
                queries_path,
                tmp_path / "report.json",
                warmup_count=warmup_count,
                seed=seed,
                top_k=top_k,
            ),
            clock=clock,
            rusage=rusage,
            client_scope=client_scope,
            service_for=lambda config, client: service,
            origin=0.0,
        )
    )
    return (
        report,
        service,
        clock,
    )  # --------------------------------------------------------------------------


# The protocol: which search is cold, what the warm-up is for, and what a row
# measures.
# --------------------------------------------------------------------------


def test_the_first_query_is_the_cold_one_and_its_result_is_the_initial_warmup(
    tmp_path: Path,
) -> None:
    """One cold row per worker, and it is the search that loads the models."""

    cold_wall, warm_wall = 2.5, 0.4
    report, service, _ = _run(
        tmp_path,
        steps=_searches(1, wall=cold_wall, cpu=1.0)
        + _searches(8, wall=warm_wall, cpu=0.2),
        warmup_count=3,
    )
    timing = report["timing"]
    cold = timing["first_query_cold"]
    rows = report["rows"]

    assert cold["query_id"] == report["probe"]["cold_query_id"]
    assert cold["elapsed_seconds"] == cold_wall
    assert [row["phase"] for row in rows] == ["cold"] + ["warm"] * 2
    # One cold search, the warm-up count, then the complete set timed warm.
    assert service.search_calls == 1 + 3 + 2
    assert cold["query_id"] == rows[0]["query_id"]
    assert sorted(row["query_id"] for row in rows[1:]) == ["q1", "q2"]
    assert report["cold_latency"]["count"] == 1
    assert report["warm_latency"]["count"] == 2


def test_warmup_searches_fall_outside_every_measured_window(tmp_path: Path) -> None:
    """The warm-up loads the embedder and the reranker, and is measured nowhere."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(12, wall=0.5, cpu=0.25),
        warmup_count=3,
    )
    warmup = report["timing"]["warmup"]

    assert warmup["search_count"] == 3
    assert warmup["measured"] is False
    assert warmup["total_seconds"] == pytest.approx(1.5)
    # Three warm-up searches cost 1.5 s and appear in no row, in neither the cold
    # window nor the warm phase total.
    assert report["timing"]["warm_phase_total_seconds"] == pytest.approx(1.0)
    assert report["timing"]["first_query_cold"]["elapsed_seconds"] == pytest.approx(0.5)
    assert sum(row["elapsed_seconds"] for row in report["rows"]) == pytest.approx(1.5)


def test_each_row_times_only_its_own_search(tmp_path: Path) -> None:
    """A row's CPU is the difference between two rusage records around it."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(12, wall=0.5, cpu=0.25),
        warmup_count=2,
    )
    rows = report["rows"]

    assert [row["self_cpu_seconds"] for row in rows] == [0.25, 0.25, 0.25]
    # The worker total is the self cost of the whole run; the gateway's CPU is a
    # separate figure read after the gateway was reaped.
    # Five searches ran: one cold, two warm-up, two warm.
    assert report["resource"]["self_cpu_seconds_total"] == pytest.approx(1.25)
    assert report["resource"]["children_cpu_seconds_total_reaped"] == pytest.approx(4.5)


def test_startup_figures_run_from_the_process_origin(tmp_path: Path) -> None:
    """`service_ready` and `startup_to_first_result` include the gateway startup."""

    report, _, clock = _run(
        tmp_path,
        steps=_searches(8, wall=0.5, cpu=0.25),
        warmup_count=1,
        startup=1.25,
        shutdown=0.75,
    )
    timing = report["timing"]

    assert timing["service_ready_elapsed"] == pytest.approx(1.25)
    assert timing["first_query_cold"]["started_at_seconds"] == pytest.approx(1.25)
    assert timing["startup_to_first_result_elapsed"] == pytest.approx(1.75)
    # The shutdown cost is past the last measured window, so it is in no figure
    # above; the clock has it.
    assert clock() == pytest.approx(1.25 + 0.5 * 4 + 0.75)


def test_the_cold_window_is_longer_than_a_warm_one_and_says_why(
    tmp_path: Path,
) -> None:
    """The two phases are labelled apart, never pooled into one distribution."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )

    assert report["cold_latency"]["p50_seconds"] == pytest.approx(0.1)
    assert report["warm_latency"]["count"] == 2
    assert "cold" in report["row_semantics"]
    assert "fresh interpreter" in report["row_semantics"]["cold"]
    assert "cache" in report["cache_disclosure"].lower()
    assert "warm" in report["cache_disclosure"]


def _keys(value: Any, found: set[str] | None = None) -> set[str]:
    """Every mapping key a JSON report carries, at any depth."""

    collected = found if found is not None else set()
    if isinstance(value, dict):
        for key, item in value.items():
            collected.add(str(key))
            _keys(item, collected)
    elif isinstance(value, list):
        for item in value:
            _keys(item, collected)
    return collected


def test_a_report_carries_no_label_or_relevance_measure(tmp_path: Path) -> None:
    """This harness measures cost. A judgeable field would invite a quality claim."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    keys = {key.casefold() for key in _keys(report)}
    # Token-wise, so a legitimate name that merely contains a word is not caught:
    # `hybrid_upgrade_required` is the service's own field and is not a grade.
    tokens = {token for key in keys for token in key.split("_")}

    for forbidden in (
        "relevance",
        "grade",
        "graded",
        "label",
        "labeled",
        "judgment",
        "judged",
        "judgement",
        "ndcg",
        "mrr",
        "target",
        "reciprocal_rank",
        "success_at",
        "target_id",
    ):
        assert forbidden not in tokens, forbidden
        assert not any(forbidden in key for key in keys if "_" in forbidden), forbidden
    # The row's own fields are cost and identifiers, and nothing else.
    assert set(report["rows"][0]) == {
        "query_id",
        "phase",
        "position",
        "requested_top_k",
        "elapsed_seconds",
        "self_cpu_seconds",
        "returned_chunk_ids",
        "result_count",
        "retrieval_method",
        "candidate_depth",
        "candidate_count",
        "rerank_window",
        "rerank_requested",
        "reranked",
        "rerank_fallback",
        "metadata",
    }
    # The notice states the exclusion, so a reader is not left inferring it.
    assert "No relevance, grade, label, or judgment is computed" in report["notice"]


def test_the_query_digest_and_the_executed_order_identify_the_run(
    tmp_path: Path,
) -> None:
    """The frozen set and the order it ran in are both hashed into the report."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
        queries=(
            ("q1", "cobalt labour"),
            ("q2", "quartz seams"),
            ("q3", "tin tailings"),
        ),
    )
    source = json.loads((tmp_path / "queries.json").read_text())
    cohort = sorted(
        [
            {
                "query_id": q["query_id"],
                "query": q["query"],
                "metadata": q.get("metadata", {}),
            }
            for q in source["queries"]
        ],
        key=lambda q: q["query_id"],
    )
    expected = hashlib.sha256(
        json.dumps(
            cohort, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    order = report["probe"]["query_order"]

    assert report["queries"]["sha256"] == expected
    assert (
        report["queries"]["input_file_sha256"]
        == hashlib.sha256((tmp_path / "queries.json").read_bytes()).hexdigest()
    )
    assert report["queries"]["digest_algorithm"] == "sha256"
    assert sorted(order) == ["q1", "q2", "q3"]
    assert (
        report["probe"]["query_order_sha256"]
        == hashlib.sha256("\n".join(order).encode("utf-8")).hexdigest()
    )
    assert report["probe"]["seed"] == 0


def test_the_same_seed_runs_the_same_order(tmp_path: Path) -> None:
    """A repeated worker differs in timing, not in the order it ran."""

    queries = tuple((f"q{index}", f"text {index}") for index in range(6))
    first, _, _ = _run(
        tmp_path / "a",
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
        queries=queries,
    )
    second, _, _ = _run(
        tmp_path / "b",
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
        queries=queries,
    )

    assert first["probe"]["query_order"] == second["probe"]["query_order"]
    assert (
        first["probe"]["query_order_sha256"] == second["probe"]["query_order_sha256"]
    )  # --------------------------------------------------------------------------


# Refusals. Each one is a project or a frozen set the probe cannot measure, and
# each is refused before a misleading number is recorded.
# --------------------------------------------------------------------------


def test_query_identity_excludes_trial_metadata_but_keeps_file_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query-cohort.json"
    data = {
        "schema_version": 1,
        "condition_id": "baseline",
        "block": 0,
        "generation_id": "generation",
        "queries": [{"query_id": "q1", "query": "First question"}],
    }
    path.write_text(json.dumps(data))
    first, first_sha = probe.load_query_set(path)
    data.update(condition_id="wider", block=2, metadata={"settings": "different"})
    path.write_text(json.dumps(data))
    second, second_sha = probe.load_query_set(path)
    assert first_sha == second_sha
    assert first["input_file_sha256"] != second["input_file_sha256"]
    data["queries"][0]["query"] = "A different question"
    path.write_text(json.dumps(data))
    assert probe.load_query_set(path)[1] != first_sha


def test_an_empty_query_set_is_refused(tmp_path: Path) -> None:
    """A latency distribution over no query is not a latency measurement."""

    project = _project(tmp_path)
    queries = _query_set(tmp_path / "queries.json", queries=())

    with pytest.raises(ProbeError, match="no queries"):
        asyncio.run(
            probe.run_probe(
                _args(project, queries, tmp_path / "report.json"),
                client_scope=_unused_client_scope,
                service_for=_unused_service,
                origin=0.0,
            )
        )
    assert not (tmp_path / "report.json").exists()


def test_a_repeated_query_id_is_refused(tmp_path: Path) -> None:
    """Two rows under one id would make a repeat count ambiguous."""

    queries = _query_set(
        tmp_path / "queries.json",
        queries=(("q1", "cobalt labour"), ("q1", "quartz seams")),
    )

    with pytest.raises(ProbeError, match="repeats in the set"):
        probe.load_query_set(queries)


def test_a_query_set_naming_another_generation_is_refused(tmp_path: Path) -> None:
    """A row measured against one generation cannot describe another."""

    with pytest.raises(ProbeError, match="names generation"):
        _run(
            tmp_path,
            steps=_searches(4, wall=0.1, cpu=0.05),
            warmup_count=0,
            query_set_generation_id="another-generation",
        )


def test_a_project_that_cannot_serve_hybrid_is_refused(tmp_path: Path) -> None:
    """Without a dense half there is nothing for a hybrid+rerank row to time."""

    with pytest.raises(ProbeError, match="not hybrid-ready"):
        _run(
            tmp_path,
            steps=_searches(4, wall=0.1, cpu=0.05),
            warmup_count=0,
            status_overrides={"hybrid_ready": False},
        )


def test_an_existing_report_path_is_refused(tmp_path: Path) -> None:
    """Replacing the last run's record would destroy the evidence of the two."""

    report_path = tmp_path / "report.json"
    report_path.write_text('{"earlier": true}\n', encoding="utf-8")

    project = _project(tmp_path)
    queries = _query_set(tmp_path / "queries.json")

    with pytest.raises(ProbeError, match="already exists"):
        asyncio.run(
            probe.run_probe(
                _args(project, queries, report_path),
                client_scope=_unused_client_scope,
                service_for=_unused_service,
                origin=0.0,
            )
        )
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"earlier": True}


def test_a_symlinked_report_path_is_refused(tmp_path: Path) -> None:
    """A report written through a link lands where nobody will look for it."""

    target = tmp_path / "elsewhere.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    project = _project(tmp_path)
    queries = _query_set(tmp_path / "queries.json")

    with pytest.raises(ProbeError, match="is a symlink"):
        asyncio.run(
            probe.run_probe(
                _args(project, queries, link),
                client_scope=_unused_client_scope,
                service_for=_unused_service,
                origin=0.0,
            )
        )
    assert json.loads(target.read_text(encoding="utf-8")) == {}


def _unused_client_scope(config: Any) -> Any:
    raise AssertionError("the run should have been refused before the gateway")


def _unused_service(config: Any, client: Any) -> Any:
    raise AssertionError(
        "the run should have been refused before the service"
    )  # --------------------------------------------------------------------------


# The payload contract, against a stub and against the real service.
# --------------------------------------------------------------------------


def test_a_row_reads_the_fields_the_search_answer_carries(tmp_path: Path) -> None:
    """The row is a reading of the payload, not a second guess at it."""

    answer = _payload(
        ("c9", "c4", "c7"),
        candidate_depth=80,
        candidate_count=64,
        rerank_window=50,
    )
    report, _, _ = _run(
        tmp_path,
        steps=[(0.5, 0.25, answer)],
        warmup_count=0,
    )
    row = report["rows"][0]

    assert row["returned_chunk_ids"] == ["c9", "c4", "c7"]
    assert row["candidate_depth"] == 80
    assert row["candidate_count"] == 64
    assert row["rerank_window"] == 50
    assert row["rerank_requested"] is True
    assert row["reranked"] is True
    assert row["rerank_fallback"] is None
    assert row["result_count"] == 3
    assert report["observed_budget_points"] == [
        {"candidate_depth": 80, "rerank_window": 50}
    ]
    assert report["rows_without_budget_point"] == 0


def test_a_requested_rerank_that_fell_back_fails_the_run(tmp_path: Path) -> None:
    """A row timed on an unranked order would describe a path not under test."""

    fell_back = _payload(
        reranked=False,
        fallback={
            "reason": "reranker_model_unavailable",
            "effect": "unranked_candidate_order_returned",
        },
    )
    with pytest.raises(ProbeError, match="did not rerank"):
        _run(tmp_path, steps=[(0.5, 0.25, fell_back)], warmup_count=0)


def test_a_fallback_in_a_warm_row_fails_the_run_too(tmp_path: Path) -> None:
    """The warm phase is where a model can quietly stop being applied."""

    good = _payload(("c1", "c2"))
    fell_back = _payload(
        reranked=False, fallback={"reason": "reranker_model_unavailable"}
    )
    # One cold search, then one warm search that still reranks, then one that
    # does not: the third answer is where the run has to stop.
    steps = [(0.2, 0.1, good), (0.2, 0.1, good), (0.2, 0.1, fell_back)]

    with pytest.raises(ProbeError, match="warm search 2 for q2"):
        _run(tmp_path, steps=steps, warmup_count=0)


def test_a_declared_out_of_scope_empty_result_is_not_a_failure(
    tmp_path: Path,
) -> None:
    """A query the corpus cannot hold has no window for a cross-encoder to rank."""

    empty_fallback = _payload(
        (),
        reranked=False,
        fallback={
            "reason": "reranker_model_unavailable",
            "effect": "unranked_candidate_order_returned",
        },
    )
    report, _, _ = _run(
        tmp_path,
        steps=[(0.2, 0.1, empty_fallback)],
        warmup_count=2,
        queries=(("q1", "cobalt labour", {"physically_unsupported_scope": True}),),
    )
    rows = report["rows"]

    assert len(rows) == 2
    assert rows[0]["returned_chunk_ids"] == []
    assert rows[0]["rerank_requested"] is True
    assert rows[0]["reranked"] is False
    assert rows[0]["metadata"]["physically_unsupported_scope"] is True
    assert rows[0]["rerank_fallback"]["reason"] == "reranker_model_unavailable"


def test_a_declared_out_of_scope_query_that_returned_passages_is_refused(
    tmp_path: Path,
) -> None:
    """The declaration excuses an empty result only, never a skipped rerank."""

    fell_back = _payload(
        ("c1", "c2"),
        reranked=False,
        fallback={"reason": "reranker_model_unavailable"},
    )
    with pytest.raises(ProbeError, match="did not rerank"):
        _run(
            tmp_path,
            steps=[(0.2, 0.1, fell_back)],
            warmup_count=0,
            queries=(("q1", "cobalt labour", {"physically_unsupported_scope": True}),),
        )


@pytest.fixture
def probe_real_service(project: Path) -> Any:
    """A real service over the repository's deterministic fakes.

    No model binary and no gateway process: the dense half is a fake backend and
    the reranker is its deterministic stand-in, so a search returns the real
    payload shape without anything downloaded.
    """

    from research_rag.core.service import ResearchService
    from research_rag.project.config import resolve_config
    from tests.conftest import write_pdf
    from tests.core.test_service import (
        FakeDenseBackend,
        FakeUltraRAG,
        UnavailableRerankerDenseBackend,
    )

    async def build(*, reranker_available: bool = True) -> ResearchService:
        write_pdf(
            project / "sources" / "article.pdf",
            [
                (
                    "Cobalt evidence about labour and artificial intelligence in "
                    "the artisanal mines of the Congo."
                ),
                "Quartz material unrelated to the primary question.",
            ],
            title="Research Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=(
                FakeDenseBackend()
                if reranker_available
                else UnavailableRerankerDenseBackend()
            ),
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        return service

    return build


def test_the_probe_reads_a_real_search_payload(
    project: Path, probe_real_service: Any
) -> None:
    """Against the real service, a row agrees with the answer it came from."""

    async def exercise() -> dict[str, Any]:
        service = await probe_real_service()
        status = await service.status()
        generation_id = str(status["generation_id"])
        direct = await service.search(
            "cobalt labour",
            top_k=10,
            retrieval_method="hybrid",
            rerank=True,
            include_staleness=False,
        )
        queries = _query_set(
            project.parent / "queries.json",
            generation_id=generation_id,
            queries=(("q1", "cobalt labour"), ("q2", "quartz material")),
        )
        clock = _Clock()
        rusage = _Rusage()

        async def measure() -> dict[str, Any]:
            return await probe.run_probe(
                _args(project, queries, project.parent / "report.json"),
                clock=clock,
                rusage=rusage,
                client_scope=lambda config: _FakeClient(
                    clock, startup=0.2, shutdown=0.1
                ),
                service_for=lambda config, client: service,
                origin=0.0,
            )

        report = await measure()
        report["_direct"] = direct
        return report

    report = asyncio.run(exercise())
    direct = report["_direct"]
    by_query = {row["query_id"]: row for row in report["rows"]}
    cold = by_query["q1"]
    warm = by_query["q1"]

    assert report["project"]["hybrid_ready"] is True
    assert cold["candidate_depth"] == direct["candidate_depth"]
    assert cold["candidate_count"] == direct["candidate_count"]
    assert cold["rerank_window"] == direct["rerank_window"]
    assert cold["reranked"] == direct["reranked"] is True
    assert warm["returned_chunk_ids"] == [hit["chunk_id"] for hit in direct["hits"]]
    assert report["generation"]["schema_version"] is not None
    assert report["project"]["generation_root"].endswith(str(direct["generation_id"]))


def test_the_probe_refuses_a_real_rerank_fallback(
    project: Path, probe_real_service: Any
) -> None:
    """A cross-encoder that cannot load is the engine's real fallback path."""

    async def exercise() -> None:
        service = await probe_real_service(reranker_available=False)
        generation_id = str((await service.status())["generation_id"])
        queries = _query_set(
            project.parent / "queries.json",
            generation_id=generation_id,
            queries=(("q1", "cobalt labour"),),
        )
        clock = _Clock()
        rusage = _Rusage()
        await probe.run_probe(
            _args(project, queries, project.parent / "report.json"),
            clock=clock,
            rusage=rusage,
            client_scope=lambda config: _FakeClient(clock, startup=0.2, shutdown=0.1),
            service_for=lambda config, client: service,
            origin=0.0,
        )

    with pytest.raises(ProbeError, match="did not rerank"):
        asyncio.run(exercise())


# --------------------------------------------------------------------------
# The arithmetic: percentiles, units, memory peaks, and the project's own files.
# --------------------------------------------------------------------------


def test_percentiles_are_nearest_rank_finite_and_typed() -> None:
    """Every figure is an observation, and p95 is not the slowest."""

    measured = [float(value) for value in range(1, 21)]
    stats = probe.latency_stats(measured)

    assert stats["count"] == 20
    assert stats["min_seconds"] == 1.0
    assert stats["p50_seconds"] == 10.0
    assert stats["p95_seconds"] == 19.0
    assert stats["max_seconds"] == 20.0
    assert stats["p95_seconds"] != stats["max_seconds"]
    assert stats["p95_seconds"] in measured
    assert stats["mean_seconds"] == 10.5
    for key, value in stats.items():
        if key == "count":
            assert isinstance(value, int)
        else:
            assert isinstance(value, float)
            assert math.isfinite(value)


def test_a_percentile_over_nothing_is_absent_rather_than_zero() -> None:
    """A latency that was not measured is not a latency of zero."""

    stats = probe.latency_stats([])

    assert stats["count"] == 0
    for key in (
        "min_seconds",
        "p50_seconds",
        "p95_seconds",
        "max_seconds",
        "mean_seconds",
    ):
        assert stats[key] is None


def test_a_measurement_that_is_not_a_number_is_dropped() -> None:
    """A boolean or a NaN in a column is not a duration."""

    stats = probe.latency_stats([1.0, 2.0, True, float("nan"), None, "3"])

    assert stats["count"] == 2
    assert stats["max_seconds"] == 2.0


def test_peak_memory_is_reported_per_kind_and_never_summed(tmp_path: Path) -> None:
    """Two peaks over different processes do not add to a moment that happened."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    peaks = report["resource"]["peak_rss_mib"]

    assert peaks["self"] == pytest.approx(256.0)
    assert peaks["children"] == pytest.approx(512.0)
    assert peaks["additive"] is False
    assert peaks["self"] + peaks["children"] not in {
        value for value in peaks.values() if isinstance(value, float)
    }
    assert "not" in peaks["note"]
    # No key anywhere in the report offers a summed peak.
    keys = {key.casefold() for key in _keys(report)}
    assert not any("total" in key and "rss" in key for key in keys)
    assert not any("peak" in key and ("sum" in key or "total" in key) for key in keys)


def test_the_report_separates_the_workers_own_cpu_from_the_gateways(
    tmp_path: Path,
) -> None:
    """A row carries self CPU; the gateway's is a total read after it was reaped."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    resource_section = report["resource"]

    assert all("children" not in row for row in report["rows"])
    assert resource_section["children_cpu_seconds_total_reaped"] == pytest.approx(4.5)
    assert "No process tree was inspected" in resource_section["scope"]
    assert "reaped" in resource_section["cpu_accounting"]
    assert "per-query row carries self CPU only" in (resource_section["cpu_accounting"])


def test_the_environment_is_recorded_before_and_after_the_run(
    tmp_path: Path,
) -> None:
    """The load average is recorded, not judged, and the machine is named."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    environment = report["environment"]

    assert environment["before"]["captured_at"] <= environment["after"]["captured_at"]
    assert environment["before"]["cpu_count"] == environment["after"]["cpu_count"]
    assert environment["before"]["platform"]
    assert report["project"]["offline"] is True
    assert report["project"]["model_cache_root"]
    assert "recorded, not judged" in environment["before"]["loadavg_note"]
    assert report["dates"]["started_at"] <= report["dates"]["finished_at"]
    assert report["dates"]["started_at"].endswith("Z")


def test_a_run_that_changed_nothing_says_so(tmp_path: Path) -> None:
    """The digests are the evidence that the probe only read."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    data_state = report["data_state"]

    assert data_state["unchanged"] is True
    assert data_state["changed_files"] == []
    assert data_state["digests"]["generation/manifest.json"] is not None
    assert data_state["digests"]["generation/chunks/chunks.jsonl"] is not None
    assert data_state["digests"]["runtime/current.json"] is not None


def test_the_runtime_directory_is_never_hashed(tmp_path: Path) -> None:
    """The gateway writes logs and handoff files there, so it is not hashed."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.1, cpu=0.05),
        warmup_count=1,
    )
    digests = report["data_state"]["digests"]

    assert not any(
        "runtime/" in name and "current.json" not in name for name in digests
    )
    assert "runtime directory is not hashed" in report["data_state"]["note"]


def test_a_changed_project_file_is_named(tmp_path: Path) -> None:
    """A mutation is reported by name rather than absorbed into the numbers."""

    project = _project(tmp_path)
    generation_root = _generation(project)
    queries = _query_set(tmp_path / "queries.json")
    clock = _Clock()
    rusage = _Rusage()

    class _Rewriting:
        """A service whose one search also edits the project's review state."""

        retrieval_policy_fingerprint = "policy-fingerprint-1"

        def __init__(self) -> None:
            self.calls = 0

        async def status(self) -> dict[str, Any]:
            return _status(generation_root)

        async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            clock.advance(0.1)
            rusage.charge(0.05)
            if self.calls == 1:
                (project / ".research-rag" / "source-exclusions.json").write_text(
                    "[]\n", encoding="utf-8"
                )
            return _payload()

    report = asyncio.run(
        probe.run_probe(
            _args(project, queries, tmp_path / "report.json", warmup_count=0),
            clock=clock,
            rusage=rusage,
            client_scope=lambda config: _FakeClient(clock, startup=0.1, shutdown=0.1),
            service_for=lambda config, client: _Rewriting(),
            origin=0.0,
        )
    )

    assert report["data_state"]["unchanged"] is False
    assert "source-exclusions.json" in report["data_state"]["changed_files"]


# --------------------------------------------------------------------------
# The summarizer, over hand-built worker reports. A report is written once per
# worker and read many times, so these build the reports the harness would have
# written rather than re-measuring one six times.
# --------------------------------------------------------------------------


def _warm_row(
    query_id: str,
    *,
    elapsed: float = 1.0,
    cpu: float = 0.5,
    chunk_ids: tuple[str, ...] = ("c1", "c2"),
    candidate_depth: int = 40,
    rerank_window: int = 20,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "phase": "warm",
        "position": 1,
        "requested_top_k": 10,
        "elapsed_seconds": elapsed,
        "self_cpu_seconds": cpu,
        "returned_chunk_ids": list(chunk_ids),
        "result_count": len(chunk_ids),
        "retrieval_method": "hybrid",
        "candidate_depth": candidate_depth,
        "candidate_count": candidate_depth,
        "rerank_window": rerank_window,
        "rerank_requested": True,
        "reranked": True,
        "rerank_fallback": None,
        "metadata": {},
    }


def _cold_row(
    query_id: str,
    *,
    elapsed: float = 3.0,
    chunk_ids: tuple[str, ...] = ("c1", "c2"),
    candidate_depth: int = 40,
    rerank_window: int = 20,
) -> dict[str, Any]:
    row = _warm_row(
        query_id,
        elapsed=elapsed,
        cpu=1.5,
        chunk_ids=chunk_ids,
        candidate_depth=candidate_depth,
        rerank_window=rerank_window,
    )
    row.update({"phase": "cold", "position": 0})
    return row


def _worker(
    *,
    condition_id: str = "c40r20",
    block: int = 0,
    rows: list[dict[str, Any]] | None = None,
    cold_query_id: str = "q1",
    ready: float = 1.0,
    startup: float = 4.0,
    cold_elapsed: float = 3.0,
    self_cpu: float = 2.0,
    children_cpu: float = 8.0,
    generation_id: str = GENERATION_ID,
    query_sha: str = QUERY_SHA,
    script_sha: str = SCRIPT_SHA,
    candidate_depth: int = 40,
    rerank_window: int = 20,
) -> dict[str, Any]:
    warm = (
        [
            _warm_row(
                "q1", candidate_depth=candidate_depth, rerank_window=rerank_window
            ),
            _warm_row(
                "q2", candidate_depth=candidate_depth, rerank_window=rerank_window
            ),
        ]
        if rows is None
        else rows
    )
    return {
        "schema_version": probe.REPORT_SCHEMA_VERSION,
        "kind": "research-rag retrieval latency probe",
        "dates": {
            "started_at": f"2026-10-0{block + 1}T00:00:00.000000Z",
            "finished_at": f"2026-10-0{block + 1}T00:05:00.000000Z",
        },
        "harness": {
            "script": "benchmark_retrieval.py",
            "script_sha256": script_sha,
            "git_revision": "revision",
            "python": "3.12.3",
        },
        "probe": {
            "condition_id": condition_id,
            "block": block,
            "order_index": block,
            "query_order": sorted({row["query_id"] for row in warm}),
            "cold_query_id": cold_query_id,
        },
        "queries": {"sha256": query_sha, "query_count": len(warm)},
        "generation": {
            "generation_id": generation_id,
            "schema_version": 3,
        },
        "timing": {
            "service_ready_elapsed": ready,
            "startup_to_first_result_elapsed": startup,
            "first_query_cold": {
                "query_id": cold_query_id,
                "started_at_seconds": ready,
                "elapsed_seconds": cold_elapsed,
            },
        },
        "resource": {
            "self_cpu_seconds_total": self_cpu,
            "children_cpu_seconds_total_reaped": children_cpu,
            "peak_rss_mib": {
                "self": 100.0 + block,
                "children": 200.0 + block,
                "additive": False,
            },
        },
        "rows": [
            _cold_row(
                cold_query_id,
                candidate_depth=candidate_depth,
                rerank_window=rerank_window,
            )
        ]
        + list(warm),
    }


def _three_blocks(
    condition_id: str = "c40r20",
    *,
    elapsed_by_block: tuple[float, float, float] = (1.0, 1.1, 0.9),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        _worker(
            condition_id=condition_id,
            block=block,
            rows=[
                _warm_row("q1", elapsed=elapsed_by_block[block]),
                _warm_row("q2", elapsed=2.0 + block),
            ],
            **kwargs,
        )
        for block in range(3)
    ]


def test_a_balanced_design_is_reported_as_balanced() -> None:
    """Every condition ran every block, so their sample sizes match."""

    reports = _three_blocks("c40r20") + _three_blocks("c80r20", candidate_depth=80)
    summary = probe.aggregate_reports(reports)

    assert summary["balance"]["every_condition_ran_every_block"] is True
    assert summary["balance"]["blocks"] == [0, 1, 2]
    assert summary["balance"]["blocks_by_condition"] == {
        "c40r20": [0, 1, 2],
        "c80r20": [0, 1, 2],
    }
    # The block a worker ran, its order, and the order it executed are kept, so
    # a reader can see the design rather than trust it.
    for condition in ("c40r20", "c80r20"):
        worker = summary["conditions"][condition]["workers"][1]
        assert worker["block"] == 1
        assert worker["order_index"] == 1
        assert worker["query_order"] == ["q1", "q2"]
    for condition in ("c40r20", "c80r20"):
        entry = summary["conditions"][condition]
        assert entry["worker_count"] == 3
        assert [worker["block"] for worker in entry["workers"]] == [0, 1, 2]


def test_an_unbalanced_design_names_the_missing_blocks() -> None:
    """A condition measured in fewer blocks has fewer observations."""

    reports = _three_blocks("c40r20") + _three_blocks("c80r20")[:2]
    summary = probe.aggregate_reports(reports)

    assert summary["balance"]["every_condition_ran_every_block"] is False
    assert summary["balance"]["conditions_missing_blocks"] == {"c80r20": [2]}


def test_repeats_are_reported_as_repeats_and_not_as_more_queries() -> None:
    """Six rows over two questions is two questions measured three times."""

    summary = probe.aggregate_reports(_three_blocks("c40r20"))
    warm = summary["conditions"]["c40r20"]["warm"]

    assert warm["row_count"] == 6
    assert warm["distinct_query_count"] == 2
    assert warm["repeat_counts"] == [3]
    assert set(warm["per_query"]) == {"q1", "q2"}
    assert warm["per_query"]["q1"]["repeat_count"] == 3
    assert warm["per_query"]["q1"]["elapsed"]["count"] == 3
    assert "not independent queries" in warm["note"]


def test_identical_rankings_across_blocks_are_reported_as_deterministic() -> None:
    """The same question returning the same order is worth saying."""

    summary = probe.aggregate_reports(_three_blocks("c40r20"))
    determinism = summary["conditions"]["c40r20"]["determinism"]

    assert determinism["queries_compared"] == 2
    assert determinism["queries_consistent"] == 2
    assert determinism["mismatched_queries"] == []


def test_a_ranking_mismatch_is_named_and_every_row_is_kept() -> None:
    """The odd block is the reason to look, so it is reported and not dropped."""

    reports = _three_blocks("c40r20")
    reports[1]["rows"][1]["returned_chunk_ids"] = ["c9", "c8"]
    summary = probe.aggregate_reports(reports)
    condition = summary["conditions"]["c40r20"]
    determinism = condition["determinism"]

    assert determinism["queries_consistent"] == 1
    assert len(determinism["mismatched_queries"]) == 1
    mismatch = determinism["mismatched_queries"][0]
    assert mismatch["query_id"] == "q1"
    assert mismatch["differences"][0]["field"] == "returned_chunk_ids"
    assert mismatch["differences"][0]["block"] == 1
    # Every row survives: the mismatch is a note beside the numbers.
    assert condition["warm"]["row_count"] == 6
    assert condition["warm"]["per_query"]["q1"]["elapsed"]["count"] == 3


def test_a_different_actual_window_is_a_mismatch_too() -> None:
    """Ranking alone is not the whole determinism claim; the budgets are named."""

    reports = _three_blocks("c40r20")
    reports[2]["rows"][1]["rerank_window"] = 50
    summary = probe.aggregate_reports(reports)
    mismatched = summary["conditions"]["c40r20"]["determinism"]["mismatched_queries"]

    assert [entry["query_id"] for entry in mismatched] == ["q1"]
    assert {difference["field"] for difference in mismatched[0]["differences"]} == {
        "rerank_window"
    }


def test_paired_ratios_pair_the_same_query_in_the_same_block() -> None:
    """A ratio means something only between the same question in the same block."""

    reports = _three_blocks("c40r20", elapsed_by_block=(1.0, 1.1, 0.9)) + [
        _worker(
            condition_id="c80r20",
            block=block,
            rows=[
                _warm_row("q1", elapsed=(2.0, 2.2, 1.8)[block], cpu=0.75),
                _warm_row("q2", elapsed=(4.0, 4.4, 3.6)[block], cpu=0.75),
            ],
            candidate_depth=80,
        )
        for block in range(3)
    ]
    summary = probe.aggregate_reports(reports, baseline_condition="c40r20")
    paired = summary["paired_ratios"]

    assert paired["paired"] is True
    condition = paired["conditions"]["c80r20"]
    assert condition["baseline_condition_id"] == "c40r20"
    assert condition["paired_blocks"] == [0, 1, 2]
    assert condition["blocks_with_different_query_lists"] == []
    assert condition["per_query"]["q1"]["pair_count"] == 3
    assert condition["per_query"]["q1"]["wall_ratio"]["p50"] == pytest.approx(2.0)
    assert condition["per_query"]["q1"]["self_cpu_ratio"]["p50"] == pytest.approx(1.5)


def test_a_block_with_a_different_query_list_is_not_paired() -> None:
    """Missing a partner is reported, never substituted with the nearest row."""

    baseline = _three_blocks("c40r20")
    other = [
        _worker(
            condition_id="c80r20",
            block=block,
            rows=(
                [_warm_row("q1"), _warm_row("q3")]
                if block == 1
                else [_warm_row("q1"), _warm_row("q2")]
            ),
            candidate_depth=80,
        )
        for block in range(3)
    ]
    summary = probe.aggregate_reports(baseline + other, baseline_condition="c40r20")
    condition = summary["paired_ratios"]["conditions"]["c80r20"]

    assert condition["paired_blocks"] == [0, 2]
    assert condition["blocks_with_different_query_lists"] == [
        {"block": 1, "only_here": ["q3"], "only_in_baseline": ["q2"]}
    ]
    # Block 1 is skipped whole, because one side ran a different question there.
    # q3 had no partner at all and is absent; the two shared questions paired in
    # blocks 0 and 2, so each carries two pairs and not three.
    assert set(condition["per_query"]) == {"q1", "q2"}
    assert condition["per_query"]["q2"]["pair_count"] == 2
    assert condition["per_query"]["q1"]["pair_count"] == 2


def test_without_a_named_baseline_no_ratio_is_reported() -> None:
    """Choosing a baseline after seeing the numbers would fit it to the outcome."""

    summary = probe.aggregate_reports(_three_blocks("c40r20"))

    assert summary["paired_ratios"]["paired"] is False
    assert summary["paired_ratios"]["baseline_condition_id"] is None
    assert "fitted to its outcome" in summary["paired_ratios"]["note"]


def test_reports_over_different_generations_are_refused() -> None:
    """A difference between two generations is not a difference between conditions."""

    reports = _three_blocks("c40r20") + _three_blocks("c80r20", generation_id="other")

    with pytest.raises(ProbeError, match="generation_id differ"):
        probe.aggregate_reports(reports)


def test_reports_over_different_query_sets_are_refused() -> None:
    """Two frozen sets are two different questions, whatever their latency."""

    reports = _three_blocks("c40r20") + _three_blocks("c80r20", query_sha="other-sha")

    with pytest.raises(ProbeError, match="query_set_sha256 differ"):
        probe.aggregate_reports(reports)


def test_reports_from_two_scorers_are_refused() -> None:
    """A harness change is not a condition change."""

    reports = _three_blocks("c40r20") + _three_blocks("c80r20", script_sha="other-sha")

    with pytest.raises(ProbeError, match="scorer_script_sha256 differ"):
        probe.aggregate_reports(reports)


def test_the_summary_names_only_the_fixed_protocol_points() -> None:
    """The comparison is the fixed ladder; an observed point is read back."""

    reports = _three_blocks("c40r20") + _three_blocks(
        "c40r50", candidate_depth=40, rerank_window=50
    )
    summary = probe.aggregate_reports(reports)

    assert summary["config"]["fixed_points"] == [
        {"candidate_depth": 40, "rerank_window": 20},
        {"candidate_depth": 40, "rerank_window": 50},
        {"candidate_depth": 80, "rerank_window": 20},
    ]
    assert summary["config"]["observed_points"] == [
        {"candidate_depth": 40, "rerank_window": 20},
        {"candidate_depth": 40, "rerank_window": 50},
    ]
    assert summary["config"]["observed_points_outside_protocol"] == []


def test_an_observed_point_outside_the_ladder_is_reported_not_averaged_in() -> None:
    """A window the protocol did not name must not join its average."""

    reports = _three_blocks("c40r20") + _three_blocks(
        "c60r30", candidate_depth=60, rerank_window=30
    )
    summary = probe.aggregate_reports(reports)

    assert summary["config"]["observed_points_outside_protocol"] == [
        {"candidate_depth": 60, "rerank_window": 30}
    ]


def test_the_summary_reports_cold_and_warm_apart() -> None:
    """One cold row per worker; the warm rows are the timed set."""

    summary = probe.aggregate_reports(_three_blocks("c40r20"))
    condition = summary["conditions"]["c40r20"]

    assert condition["cold"]["worker_count"] == 3
    assert condition["cold"]["service_ready_elapsed"]["p50_seconds"] == 1.0
    assert condition["cold"]["first_query_elapsed"]["max_seconds"] == 3.0
    assert condition["cold"]["startup_to_first_result"]["p50_seconds"] == 4.0
    assert condition["warm"]["overall"]["count"] == 6
    assert condition["warm"]["self_cpu"]["p50_seconds"] == 0.5


def test_the_summary_reports_resource_totals_and_separate_peaks() -> None:
    """CPU sums over workers; peaks stay apart."""

    summary = probe.aggregate_reports(_three_blocks("c40r20"))
    resource_section = summary["conditions"]["c40r20"]["resource"]

    assert resource_section["self_cpu_seconds_total"] == pytest.approx(6.0)
    assert resource_section["children_cpu_seconds_total_reaped"] == pytest.approx(24.0)
    peaks = resource_section["peak_rss_mib"]
    assert peaks["self_max"] == pytest.approx(102.0)
    assert peaks["children_max"] == pytest.approx(202.0)
    assert peaks["additive"] is False
    assert len(resource_section["workers"]) == 3


def test_the_summary_reports_repeat_spread_and_no_confidence_interval() -> None:
    """Three repeats bound a range. They do not establish an interval."""

    summary = probe.aggregate_reports(
        _three_blocks("c40r20", elapsed_by_block=(1.0, 1.5, 1.25))
        + _three_blocks("c80r20", elapsed_by_block=(2.0, 2.5, 2.25)),
        baseline_condition="c40r20",
    )
    spread = summary["conditions"]["c40r20"]["repeat_spread"]["per_query"]
    uncertainty = summary["uncertainty"]

    assert spread["q1"]["repeat_count"] == 3
    assert spread["q1"]["min_seconds"] == pytest.approx(1.0)
    assert spread["q1"]["max_seconds"] == pytest.approx(1.5)
    assert spread["q1"]["spread_seconds"] == pytest.approx(0.5)
    assert uncertainty["method"] == "per-query spread over repeated blocks"
    assert "do not establish a confidence interval" in uncertainty["note"]
    # Key-wise, because the notes say what is deliberately absent.
    keys = {key.casefold() for key in _keys(summary)}
    for forbidden in ("p_value", "confidence_interval", "winner", "ranking"):
        assert forbidden not in keys, forbidden
    assert "no test statistic or winner is computed" in summary["notice"]
    assert (
        "No test statistic, no p-value, and no winner is computed"
        in (summary["paired_ratios"]["note"])
    )


def test_two_warm_rows_claiming_one_block_and_query_are_refused() -> None:
    """A pairing cannot choose between two rows that claim the same position."""

    reports = _three_blocks("c40r20")
    reports[0]["rows"].append(_warm_row("q1"))
    reports.append(_worker(condition_id="c80r20", block=3))

    with pytest.raises(ProbeError, match="both claim query"):
        probe.aggregate_reports(reports, baseline_condition="c40r20")


def test_the_manifest_digest_is_verified_and_a_changed_report_is_refused(
    tmp_path: Path,
) -> None:
    """A report replaced after the manifest was written is not summarized."""

    reports = [
        tmp_path / "block-0.json",
        tmp_path / "block-1.json",
    ]
    manifest_entries = []
    for index, path in enumerate(reports):
        path.write_text(json.dumps(_worker(block=index), indent=2), encoding="utf-8")
        manifest_entries.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"reports": manifest_entries}), encoding="utf-8")

    loaded, sources, baseline = probe.load_report_manifest(manifest)
    assert len(loaded) == 2
    assert [source["block"] for source in sources] == [0, 1]
    assert all(source["sha256_verified"] for source in sources)
    assert baseline is None

    reports[0].write_text(json.dumps(_worker(block=0, self_cpu=99.0)), encoding="utf-8")

    with pytest.raises(ProbeError, match="digest mismatch"):
        probe.load_report_manifest(manifest)


def test_a_manifest_entry_without_a_digest_is_recorded_as_unverified(
    tmp_path: Path,
) -> None:
    """No declared digest is weaker evidence, not a wrong report."""

    path = tmp_path / "block-0.json"
    path.write_text(json.dumps(_worker()), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"reports": [str(path)]}), encoding="utf-8")

    _, sources, _ = probe.load_report_manifest(manifest)

    assert sources[0]["sha256_verified"] is False
    assert sources[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sources[0]["condition_id"] == "c40r20"


def test_the_manifest_baseline_can_be_named_and_overridden(tmp_path: Path) -> None:
    """The caller says which condition the ratios are against, and may change it."""

    reports = [tmp_path / "one.json", tmp_path / "two.json"]
    reports[0].write_text(json.dumps(_worker(block=0)), encoding="utf-8")
    reports[1].write_text(
        json.dumps(_worker(condition_id="c80r20", block=0, candidate_depth=80)),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "reports": [str(path) for path in reports],
                "baseline_condition_id": "c40r20",
            }
        ),
        encoding="utf-8",
    )

    _, _, baseline = probe.load_report_manifest(manifest)
    assert baseline == "c40r20"


def test_summarize_writes_the_summary_and_leaves_the_reports_alone(
    tmp_path: Path,
) -> None:
    """The summarize mode reads worker reports and writes one new file."""

    reports = [tmp_path / f"block-{block}.json" for block in range(2)]
    for block, path in enumerate(reports):
        path.write_text(
            json.dumps(_worker(condition_id="c40r20", block=block)), encoding="utf-8"
        )
    before = {path: path.read_bytes() for path in reports}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"reports": [str(path) for path in reports]}), encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    args = argparse.Namespace(
        project=None,
        queries=None,
        report=output,
        summarize=manifest,
        baseline_condition=None,
    )

    summary = probe.summarize_reports(args)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        probe.SUMMARY_SCHEMA_VERSION
    )
    assert summary["inputs"]["report_count"] == 2
    assert summary["inputs"]["identity"]["query_set_sha256"] == QUERY_SHA
    assert {path: path.read_bytes() for path in reports} == before


def test_a_command_line_naming_both_modes_is_refused() -> None:
    """--summarize measures nothing, so a project beside it is a mistake."""

    args = argparse.Namespace(
        project=Path("/tmp/project"),
        queries=Path("/tmp/queries.json"),
        report=Path("/tmp/report.json"),
        summarize=Path("/tmp/manifest.json"),
        baseline_condition=None,
    )

    with pytest.raises(ProbeError, match="do not belong with it"):
        probe._validate_mode(args)


def test_a_command_line_naming_no_mode_is_refused() -> None:
    """Neither mode named is a mistake worth a clear message, not a stack trace."""

    args = argparse.Namespace(
        project=None,
        queries=None,
        report=Path("/tmp/report.json"),
        summarize=None,
        baseline_condition=None,
    )

    with pytest.raises(ProbeError, match="needs --project and --queries"):
        probe._validate_mode(args)


def test_the_console_view_prints_the_figures_and_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is a reader's first view, so it must not crash or announce a winner."""

    report, _, _ = _run(
        tmp_path,
        steps=_searches(8, wall=0.25, cpu=0.1),
        warmup_count=2,
        condition_id="c40r20",
        block=1,
    )
    probe.print_probe(report, tmp_path / "report.json")
    probe.print_summary(
        probe.aggregate_reports([report], baseline_condition="c40r20"),
        tmp_path / "summary.json",
    )
    printed = capsys.readouterr().out

    assert "c40r20 block 1" in printed
    assert "hybrid+rerank" in printed
    assert "never added" in printed
    assert "project files unchanged: True" in printed
    assert "no p-value and no winner" in printed
    assert "report written to" in printed


def test_a_missing_figure_prints_as_a_blank_rather_than_a_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A latency that was not measured must not print as 0.000."""

    summary = probe.aggregate_reports([_worker(block=0)])
    summary["conditions"]["c40r20"]["warm"]["overall"]["p50_seconds"] = None
    probe.print_summary(summary, Path("/tmp/summary.json"))
    printed = capsys.readouterr().out

    assert "-" in printed
    assert "0.000" not in printed
