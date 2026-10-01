"""Measure retrieval quality against a judged query set.

Harness behind the retrieval findings in ``MEASUREMENTS.md``. It runs a judged
query set (``evaluation/ai-and-fetishism-queries.json`` by default) through the
service method behind the public ``search`` tool and reports success@k, MRR,
nDCG@10, document success, and the mean number of distinct sources a result
spans. Every
mode passes ``rerank`` explicitly, so no number depends on the tool default; the
``hybrid+rerank`` row is what a default search now does.

Judgments are known-item: one relevant chunk per query, named by a verbatim
snippet so the target re-resolves after re-ingestion. That measures the
findability of a designated passage, not exhaustive recall. ``chunks.jsonl`` is
read read-only; nothing inside the project is written.

    uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project

Add --validate-only to check the judged set without searching. A throwaway
warm-up search runs first, because the first search pays a cold index load; its
cost is reported separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from fastmcp import Client

from research_rag.config import (
    configured_source_directory,
    resolve_config,
)
from research_rag.rerankers import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_CHOICES,
)
from research_rag.service import ResearchService
from research_rag.ultrarag import (
    VanillaUltraRAG,
    create_vanilla_transport,
)

DEFAULT_JUDGMENTS = Path("evaluation/ai-and-fetishism-queries.json")
# The reranked mode is one row per selected reranker: the model is an engine
# setting, so a second model is a second row over the same queries, not a second
# mode. The default model keeps the plain ``hybrid+rerank`` label.
RERANK_MODE = "hybrid+rerank"
MODES = ("bm25", "dense", "hybrid", RERANK_MODE)
QUERY_CLASSES = ("quote", "paraphrase", "entity")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
WHITESPACE = re.compile(r"\s+")
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "also",
        "among",
        "and",
        "any",
        "are",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "itself",
        "more",
        "most",
        "not",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "your",
    }
)


class EvaluationError(RuntimeError):
    """Raised when the judged set or the project cannot be evaluated."""


def normalize(value: str) -> str:
    """Collapse whitespace so snippets match across wrapping differences."""

    return WHITESPACE.sub(" ", value).strip()


def content_tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(value.casefold())
        if len(token) >= 3 and token not in STOPWORDS
    }


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of the query's content words that also occur in the passage."""

    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & content_tokens(text)) / len(query_tokens)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """First relevant item's reciprocal rank, 0 when none is ranked."""

    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def success_at(ranked: list[str], relevant: set[str], k: int) -> bool:
    """Whether any relevant item appears within the first ``k`` results."""

    return any(item in relevant for item in ranked[:k])


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG at ``k`` for a judged set with one relevant chunk."""

    discounted = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], 1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return discounted / ideal if ideal else 0.0


def load_judgments(path: Path) -> dict[str, Any]:
    """Read and structurally validate a judged query set."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read the judged set: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise EvaluationError(f"Unsupported judged-set schema in {path}")
    targets = payload.get("targets")
    queries = payload.get("queries")
    if not isinstance(targets, list) or not targets:
        raise EvaluationError(f"The judged set has no targets: {path}")
    if not isinstance(queries, list) or not queries:
        raise EvaluationError(f"The judged set has no queries: {path}")
    known = {str(target.get("target_id")) for target in targets}
    for target in targets:
        for field in ("target_id", "source_path", "snippet"):
            if not str(target.get(field) or "").strip():
                raise EvaluationError(f"Target without {field}: {target!r}")
    for query in queries:
        if str(query.get("target_id")) not in known:
            raise EvaluationError(f"Query names an unknown target: {query!r}")
        if str(query.get("class")) not in QUERY_CLASSES:
            raise EvaluationError(f"Query has an unknown class: {query!r}")
        if not str(query.get("query") or "").strip():
            raise EvaluationError(f"Query has no text: {query!r}")
    return payload


def load_generation(
    generation_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read a generation's canonical chunks and its manifest document records.

    Chunk records carry ``document_id`` and ``source_id`` but no path, so the
    manifest is the only place a target's path resolves.
    """

    manifest_path = generation_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"Cannot read the generation manifest: {manifest_path}"
        ) from exc
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise EvaluationError(
            f"The generation manifest lists no documents: {manifest_path}"
        )

    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise EvaluationError(f"Generation chunks are missing: {chunks_path}")
    records: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise EvaluationError(f"Generation has no chunks: {chunks_path}")
    return records, {
        str(document["document_id"]): document
        for document in documents
        if isinstance(document, dict) and document.get("document_id")
    }


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Return a chunk's canonical stored text, tolerating legacy field names."""

    return str(chunk.get("contents") or chunk.get("text") or "")


def resolve_targets(
    targets: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    *,
    skip: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Resolve every judged target to exactly one chunk of the measured generation.

    A target names a source path and a verbatim snippet, so it survives
    re-ingestion, chunk-ID changes, and a rename. Ambiguity is an error: a snippet
    matching two chunks would make the judgment meaningless.

    A target in ``skip`` stays unresolved on purpose. A judged source the corpus
    no longer holds is a benchmark decision, not a measurement, so it is named on
    the command line and in the report rather than relaxing resolution for all
    targets.
    """

    by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_document.setdefault(str(chunk.get("document_id")), []).append(chunk)

    resolved: dict[str, dict[str, Any]] = {}
    for target in targets:
        target_id = str(target["target_id"])
        if target_id in skip:
            continue
        snippet = normalize(str(target["snippet"])).casefold()
        wanted = {
            str(target.get("source_path") or "").strip(),
            str(target.get("source_relative_path") or "").strip(),
        } - {""}
        matched = [
            document
            for document in documents.values()
            if str(document.get("source_path") or "") in wanted
            or str(document.get("source_relative_path") or "") in wanted
        ]
        if not matched:
            named = str(target.get("document_id") or "")
            if named not in documents:
                raise EvaluationError(
                    f"{target_id}: no document in this generation matches "
                    f"{sorted(wanted) or named!r}"
                )
            matched = [documents[named]]
        if len(matched) != 1:
            raise EvaluationError(
                f"{target_id}: {len(matched)} documents match {sorted(wanted)}"
            )
        document = matched[0]
        document_id = str(document["document_id"])
        candidates = by_document.get(document_id, [])
        if not candidates:
            raise EvaluationError(
                f"{target_id}: the generation has no chunks for document {document_id}"
            )
        hits = [
            chunk
            for chunk in candidates
            if snippet in normalize(_chunk_text(chunk)).casefold()
        ]
        if len(hits) != 1:
            raise EvaluationError(
                f"{target_id}: snippet resolves to {len(hits)} chunks in "
                f"{document.get('source_relative_path') or document_id}, expected exactly one"
            )
        hit = hits[0]
        resolved[target_id] = {
            "target_id": target_id,
            "source_path": document.get("source_path") or target.get("source_path"),
            "locator": hit.get("locator"),
            "chunk_id": str(hit.get("chunk_id")),
            "document_id": document_id,
            "chunk_id_at_measurement": target.get("chunk_id_at_measurement"),
            "chunk_text": _chunk_text(hit),
        }
    return resolved


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(runs: list[dict[str, Any]], k: int) -> dict[str, Any]:
    return {
        "query_count": len(runs),
        "success_at_1": _mean([1.0 if run["success_at_1"] else 0.0 for run in runs]),
        "success_at_3": _mean([1.0 if run["success_at_3"] else 0.0 for run in runs]),
        "success_at_k": _mean([1.0 if run["success_at_k"] else 0.0 for run in runs]),
        "mrr": _mean([run["reciprocal_rank"] for run in runs]),
        "ndcg_at_k": _mean([run["ndcg_at_k"] for run in runs]),
        "document_success_at_k": _mean(
            [1.0 if run["document_success_at_k"] else 0.0 for run in runs]
        ),
        "mean_lexical_overlap": _mean([run["lexical_overlap"] for run in runs]),
        "mean_result_count": _mean([float(run["result_count"]) for run in runs]),
        "mean_distinct_sources": _mean(
            [float(run["distinct_source_count"]) for run in runs]
        ),
        "mean_withheld": _mean([float(run["withheld_total"]) for run in runs]),
        "top_k": k,
    }


def summarize(runs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate per-query records per mode and per query class."""

    summary: dict[str, Any] = {}
    for mode in dict.fromkeys(run["mode"] for run in runs):
        mode_runs = [run for run in runs if run["mode"] == mode]
        if not mode_runs:
            continue
        by_class = {
            query_class: _aggregate(
                [run for run in mode_runs if run["class"] == query_class],
                top_k,
            )
            for query_class in QUERY_CLASSES
        }
        summary[mode] = {
            "overall": _aggregate(mode_runs, top_k),
            "per_class": {
                name: values
                for name, values in by_class.items()
                if values["query_count"]
            },
        }
    return summary


def _percent(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def print_summary(section: str, summary: dict[str, Any]) -> None:
    print()
    print(f"== {section} ==")
    width = max(15, *(len(mode) for mode in summary)) if summary else 15
    header = (
        f"{'mode':<{width}}{'n':>4}{'succ@1':>8}{'succ@3':>8}{'succ@k':>8}"
        f"{'MRR':>7}{'nDCG':>7}{'doc@k':>7}{'overlap':>9}{'ret':>5}{'srcs':>6}"
    )
    print(header)
    print("-" * len(header))
    for mode, payload in summary.items():
        row = payload["overall"]
        print(
            f"{mode:<{width}}{row['query_count']:>4}"
            f"{_percent(row['success_at_1']):>8}"
            f"{_percent(row['success_at_3']):>8}{_percent(row['success_at_k']):>8}"
            f"{row['mrr']:>7.3f}{row['ndcg_at_k']:>7.3f}"
            f"{_percent(row['document_success_at_k']):>7}"
            f"{row['mean_lexical_overlap']:>9.3f}{row['mean_result_count']:>5.1f}"
            f"{row['mean_distinct_sources']:>6.1f}"
        )
    for mode, payload in summary.items():
        for query_class, row in payload["per_class"].items():
            print(
                f"  {mode} / {query_class:<12}{row['query_count']:>3}"
                f"{_percent(row['success_at_1']):>8}{_percent(row['success_at_3']):>8}"
                f"{_percent(row['success_at_k']):>8}{row['mrr']:>7.3f}"
                f"{row['ndcg_at_k']:>7.3f}{_percent(row['document_success_at_k']):>7}"
                f"{row['mean_lexical_overlap']:>9.3f}{row['mean_result_count']:>5.1f}"
                f"{row['mean_distinct_sources']:>6.1f}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-retrieval",
        description=(
            "Measure BM25, dense, hybrid, and reranked retrieval against a "
            "judged query set by calling the public MCP search tool."
        ),
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Research project root containing .research-rag.",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=DEFAULT_JUDGMENTS,
        help=f"Judged query set (default: {DEFAULT_JUDGMENTS}).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Primary depth (1-50).")
    parser.add_argument(
        "--deep-top-k",
        type=int,
        default=50,
        help="Second depth for the deep modes; set 0 to skip it.",
    )
    parser.add_argument(
        "--deep-modes",
        default="hybrid",
        help=f"Comma-separated subset of {','.join(MODES)} for the deep pass.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=f"Comma-separated subset of {','.join(MODES)}.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(QUERY_CLASSES),
        help=f"Comma-separated subset of {','.join(QUERY_CLASSES)}.",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N queries.")
    parser.add_argument(
        "--report",
        type=Path,
        help="Where to write the JSON report (default: beside the judged set).",
    )
    parser.add_argument(
        "--skip-targets",
        default="",
        help=(
            "Comma-separated judged target IDs to leave out, for a target whose "
            "source the corpus no longer holds. The report and the console name "
            "every skipped target."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve every judged target and stop without searching.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--model-cache-root",
        default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
    )
    parser.add_argument("--embedding-threads", type=int)
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=os.environ.get("RESEARCH_ULTRARAG_DENSE_BACKEND", "auto"),
    )
    parser.add_argument(
        "--reranker-model",
        action="append",
        metavar="NAME",
        help=(
            "Reranker model to measure for the hybrid+rerank row. Repeat it to "
            "compare models over the same queries in one run. Default: the "
            "server's configured model. Supported: "
            + ", ".join(RERANKER_MODEL_CHOICES)
            + "."
        ),
    )
    return parser


def _selection(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    chosen = [item.strip() for item in value.split(",") if item.strip()]
    if not chosen:
        raise EvaluationError(f"No {label} selected")
    unknown = [item for item in chosen if item not in allowed]
    if unknown:
        raise EvaluationError(f"Unknown {label}: {unknown}; allowed: {list(allowed)}")
    return chosen


def _mode_settings(mode: str) -> dict[str, Any]:
    if mode == RERANK_MODE:
        return {"retrieval_method": "hybrid", "rerank": True}
    return {"retrieval_method": mode, "rerank": False}


def _mode_variants(
    modes: list[str],
    reranker_models: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    for mode in modes:
        if mode != RERANK_MODE:
            variants.append((mode, _mode_settings(mode)))
            continue
        for model in reranker_models:
            label = mode if model == DEFAULT_RERANKER_MODEL else f"{mode}[{model}]"
            variants.append(
                (
                    label,
                    {
                        "retrieval_method": "hybrid",
                        "rerank": True,
                        "rerank_model": model,
                    },
                )
            )
    return variants


async def _run_one(
    service: ResearchService,
    *,
    query_id: str,
    query_class: str,
    query: str,
    mode: str,
    settings: dict[str, Any],
    top_k: int,
    target: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await service.search(
        query,
        top_k=top_k,
        include_staleness=False,
        **settings,
    )
    elapsed = time.perf_counter() - started
    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        raise EvaluationError(f"search returned an unexpected payload for {query_id}")

    hits = payload["hits"]
    ranked_chunk_ids = [str(hit.get("chunk_id")) for hit in hits]
    ranked_document_ids = [str(hit.get("document_id")) for hit in hits]
    relevant = {str(target["chunk_id"])}
    withheld = payload.get("withheld_candidates") or {}
    withheld_ids = [
        str(example)
        for entry in (withheld.get("reasons") or {}).values()
        for example in (entry.get("example_chunk_ids") or [])
    ]
    rank = next(
        (
            position
            for position, item in enumerate(ranked_chunk_ids, 1)
            if item in relevant
        ),
        None,
    )
    return {
        "query_id": query_id,
        "class": query_class,
        "query": query,
        "mode": mode,
        "reranker_model": payload.get("reranker_model"),
        "top_k": top_k,
        "target_chunk_id": str(target["chunk_id"]),
        "target_document_id": str(target["document_id"]),
        "rank": rank,
        "success_at_1": success_at(ranked_chunk_ids, relevant, 1),
        "success_at_3": success_at(ranked_chunk_ids, relevant, 3),
        "success_at_k": success_at(ranked_chunk_ids, relevant, top_k),
        "reciprocal_rank": reciprocal_rank(ranked_chunk_ids, relevant),
        "ndcg_at_k": ndcg_at(ranked_chunk_ids, relevant, top_k),
        "document_success_at_k": str(target["document_id"])
        in set(ranked_document_ids[:top_k]),
        "lexical_overlap": lexical_overlap(query, str(target["chunk_text"])),
        "result_count": int(payload.get("result_count") or 0),
        "distinct_source_count": int(payload.get("distinct_reference_count") or 0),
        "candidate_count": int(payload.get("candidate_count") or 0),
        "rerank_window": int(payload.get("rerank_window") or 0),
        "withheld_total": int(withheld.get("total") or 0),
        "target_listed_as_withheld": str(target["chunk_id"]) in set(withheld_ids),
        "returned_chunk_ids": ranked_chunk_ids,
        "elapsed_seconds": elapsed,
    }


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project.expanduser().resolve()
    judgments_path = args.judgments.expanduser().resolve()
    if not project.is_dir():
        raise EvaluationError(f"Project root is not a directory: {project}")
    if not 1 <= args.top_k <= 50:
        raise EvaluationError("--top-k must be between 1 and 50")
    if args.deep_top_k and not 1 <= args.deep_top_k <= 50:
        raise EvaluationError("--deep-top-k must be between 1 and 50, or 0")

    modes = _selection(args.modes, MODES, "modes")
    deep_modes = (
        _selection(args.deep_modes, MODES, "deep modes") if args.deep_top_k else []
    )
    classes = _selection(args.classes, QUERY_CLASSES, "classes")
    skip_targets = frozenset(
        item.strip() for item in args.skip_targets.split(",") if item.strip()
    )

    payload = load_judgments(judgments_path)
    known_targets = {str(target["target_id"]) for target in payload["targets"]}
    if unknown := sorted(skip_targets - known_targets):
        raise EvaluationError(f"Unknown target IDs to skip: {unknown}")
    queries = [
        item
        for item in payload["queries"]
        if item["class"] in classes and item["target_id"] not in skip_targets
    ]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise EvaluationError("No queries selected")
    if skip_targets:
        dropped = sum(
            1
            for item in payload["queries"]
            if item["class"] in classes and item["target_id"] in skip_targets
        )
        print(
            f"Skipping {len(skip_targets)} judged target(s) on request: "
            f"{', '.join(sorted(skip_targets))}; {dropped} queries will not be run."
        )

    config = resolve_config(
        project,
        source_directory=configured_source_directory(project),
        model_cache_root=args.model_cache_root,
        offline=args.offline,
        dense_backend=args.dense_backend,
        runtime_root=args.runtime_root,
        embedding_threads=args.embedding_threads,
    )
    # Without --reranker-model a run measures what this server would serve; with
    # it, exactly the models named, each over the same queries under the same
    # conditions.
    reranker_models = (
        _selection(
            ",".join(args.reranker_model),
            RERANKER_MODEL_CHOICES,
            "reranker models",
        )
        if args.reranker_model
        else [config.reranker_model]
    )
    variants = _mode_variants(modes, reranker_models)
    deep_variants = _mode_variants(deep_modes, reranker_models)

    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else judgments_path.with_name(f"{judgments_path.stem}-report.json")
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "judgments": {
            "path": str(judgments_path),
            "schema_version": payload.get("schema_version"),
            "protocol": payload.get("protocol"),
            "query_count": len(payload["queries"]),
            "evaluated_query_count": len(queries),
            "target_count": len(payload["targets"]),
        },
        "corpus": payload.get("corpus"),
        "settings": {
            "selected_modes": modes,
            "selected_deep_modes": deep_modes,
            "selected_classes": classes,
            "reranker_models": reranker_models,
            "skipped_targets": sorted(skip_targets),
            "top_k": args.top_k,
            "deep_top_k": args.deep_top_k,
            "include_staleness": False,
            # Applied at query time over already-ranked candidates, so the
            # generation's recorded policy cannot carry it and a report that
            # measures it must name it here.
            "selection_policy": {
                "method": "greedy_source_diversity",
                "source_diversity_penalty": config.settings.source_diversity_penalty,
            },
        },
        "notice": (
            "Judgment resolution reads the generation's canonical chunks.jsonl "
            "read-only, and this harness never writes inside the project. "
            "Retrieval runs through ResearchService.search in process, which is "
            "the engine the workspace serves, because every search there is "
            "hybrid-only and this harness also measures BM25 and dense."
        ),
    }

    vanilla_transport = create_vanilla_transport(config)
    async with Client(
        vanilla_transport,
        timeout=1800,
        init_timeout=1800,
    ) as vanilla_client:
        service = ResearchService(config, VanillaUltraRAG(vanilla_client, config))
        status = await service.status()
        generation_root = status.get("generation_root")
        if not generation_root:
            raise EvaluationError("The project has no selected generation to evaluate")
        chunks, documents = load_generation(Path(str(generation_root)))
        resolved = resolve_targets(
            payload["targets"],
            chunks,
            documents,
            skip=skip_targets,
        )

        report["project"] = {
            "project_root": str(project),
            "project_id": status.get("project_id"),
            "project_name": status.get("project_name"),
            "generation_id": status.get("generation_id"),
            "generation_root": str(generation_root),
            "chunk_count": status.get("chunk_count"),
            "indexed_source_count": status.get("indexed_source_count"),
            "searchable_source_count": status.get("searchable_source_count"),
            "excluded_source_count": status.get("excluded_source_count"),
            "generation_upgrade_required": status.get("generation_upgrade_required"),
        }
        report["retrieval"] = status.get("retrieval")
        report["targets"] = [
            {key: value for key, value in target.items() if key != "chunk_text"}
            | {"text_chars": len(target["chunk_text"])}
            for target in resolved.values()
        ]

        print(
            f"Project {status.get('project_name')} generation {status.get('generation_id')} "
            f"with {len(chunks)} chunks; {len(queries)} queries over {len(resolved)} targets; "
            f"source diversity penalty "
            f"{config.settings.source_diversity_penalty:g}."
        )
        if args.validate_only:
            if skip_targets:
                print(
                    "Every judged target except the requested skips resolved "
                    "uniquely; no search was run."
                )
            else:
                print("Every judged target resolved uniquely; no search was run.")
            return report

        started = time.perf_counter()
        warm_query = queries[0]
        warm_target = resolved[warm_query["target_id"]]
        warm_started = time.perf_counter()
        await _run_one(
            service,
            query_id="warmup",
            query_class=warm_query["class"],
            query=warm_query["query"],
            mode="bm25",
            settings=_mode_settings("bm25"),
            top_k=1,
            target=warm_target,
        )
        report["settings"]["warmup_seconds"] = time.perf_counter() - warm_started
        print(
            f"Warm-up search: {report['settings']['warmup_seconds']:.2f} s (discarded)"
        )

        runs: list[dict[str, Any]] = []
        for position, query in enumerate(queries, 1):
            target = resolved[query["target_id"]]
            ranks: list[str] = []
            for label, settings in variants:
                run = await _run_one(
                    service,
                    query_id=query["query_id"],
                    query_class=query["class"],
                    query=query["query"],
                    mode=label,
                    settings=settings,
                    top_k=args.top_k,
                    target=target,
                )
                runs.append(run)
                ranks.append(f"{label}={run['rank'] if run['rank'] else 'miss'}")
            print(
                f"[{position:>3}/{len(queries)}] {query['query_id']} {' '.join(ranks)}"
            )

        deep_runs: list[dict[str, Any]] = []
        for label, settings in deep_variants:
            for query in queries:
                deep_runs.append(
                    await _run_one(
                        service,
                        query_id=query["query_id"],
                        query_class=query["class"],
                        query=query["query"],
                        mode=label,
                        settings=settings,
                        top_k=args.deep_top_k,
                        target=resolved[query["target_id"]],
                    )
                )

    report["runs"] = runs
    report["deep_runs"] = deep_runs
    report["summary"] = summarize(runs, args.top_k)
    report["deep_summary"] = summarize(deep_runs, args.deep_top_k) if deep_runs else {}
    report["timing"] = {
        "total_seconds": time.perf_counter() - started,
        "mean_search_seconds": (
            sum(run["elapsed_seconds"] for run in runs + deep_runs)
            / len(runs + deep_runs)
        ),
        "search_count": len(runs) + len(deep_runs),
    }

    print_summary(f"top_k={args.top_k}", report["summary"])
    if report["deep_summary"]:
        print_summary(f"deep pass, top_k={args.deep_top_k}", report["deep_summary"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print()
    print(
        f"{report['timing']['search_count']} searches in {report['timing']['total_seconds']:.1f} s; "
        f"report written to {report_path}"
    )
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        asyncio.run(evaluate(args))
    except EvaluationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
