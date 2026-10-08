"""Measure cosine on constructed contrasts and an optional fresh known-item pool.

This diagnostic changes no retrieval setting or project artifact. Constructed
answer-selection expectations are not corpus relevance judgments. The admission
projection is explicitly a score-only simulation, not a production search. Pair
suppression calls the engine itself, using freshly computed passage vectors.

An optional project supplies canonical texts for the existing judged targets.
Those texts are re-embedded in memory with the current pinned model. Stored
generation vectors are never mixed with new query vectors. Other targets are
unjudged competitors, not assumed negatives. No full-corpus recall is measured.

Run offline; a missing cached snapshot is a setup failure, not a download request.
Reports contain no corpus text and stay outside the measured project. The approved
/tmp/kilo scratch area is allowed even when /tmp was itself initialized as a
project; nested research projects within that scratch area remain protected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from platformdirs import user_cache_path

from research_rag.project.config import (
    GATEWAY_EXECUTABLE,
    ResearchConfig,
    _contains_files,
    apply_process_priority,
    configured_source_directory,
    recorded_runtime_root,
)
from research_rag.project.settings import (
    SETTINGS,
    USER_CONFIG_DIRECTORY,
    EffectiveSettings,
    sources_for,
)
from research_rag.project.settings_layers import resolve_settings
from research_rag.project.state_files import PORTABLE_DIRECTORY, RUNTIME_DIRECTORY
from research_rag.project.support import _passage_equality_key
from research_rag.retrieval.model_runtime import ModelRuntime
from research_rag.retrieval.rerankers import resolve_reranker_model
from research_rag.retrieval.search import _collapse_repetitions
from research_rag.storage.records import (
    iter_jsonl,
    load_chunk_exclusions,
    load_current_generation,
    load_source_exclusions,
)

REPOSITORY = Path(__file__).resolve().parents[1]
REPORT_SCRATCH = Path("/tmp/kilo").resolve()
REPORT_VERSION = 2


def constructed_probes() -> list[dict[str, Any]]:
    """Expectations precede model scores; none are sampled prevalence estimates."""
    background = (
        "The report describes the organization of the workplace and the operation "
        "of its production equipment. Researchers visited the factory, interviewed "
        "employees and supervisors, and reviewed the records maintained by the "
        "administrative department. The study covers the same twelve-month period "
        "for each department. Its methods section describes recruitment, interview "
        "procedures, the handling of missing records, and the limits of the sample. "
        "The researchers distinguish their observations from earlier literature. "
        "The tables present the findings by department and employment category. "
        "The discussion considers the implications for workplace policy and notes "
        "that further research would require a larger sample of factories. "
    )
    cases = [
        (
            "paraphrase",
            "Who owns and controls the factory robots?",
            "Workers collectively own and control the factory robots.",
            "The factory's robotic equipment belongs to its workforce and is operated under their collective control.",
            "equivalent_wording",
            ["a", "b"],
        ),
        (
            "roles",
            "Who monitors the workers?",
            "The managers monitor the workers.",
            "The workers monitor the managers.",
            "different_roles",
            ["a"],
        ),
        (
            "negation",
            "Which finding says that the treatment failed to reduce mortality?",
            "The treatment did not reduce mortality in the clinical trial.",
            "The treatment reduced mortality in the clinical trial.",
            "contradiction",
            ["a"],
        ),
        (
            "direction",
            "Which intervention increased unemployment rather than reducing it?",
            "The intervention increased unemployment in the region.",
            "The intervention decreased unemployment in the region.",
            "contradiction",
            ["a"],
        ),
        (
            "quantity",
            "Which trial measured a ten-percent reduction in mortality?",
            "The trial measured a 10 percent reduction in mortality.",
            "The trial measured a 90 percent reduction in mortality.",
            "different_quantity",
            ["a"],
        ),
        (
            "sign",
            "Which reported change indicates falling wages?",
            "Wages changed by -5 percent.",
            "Wages changed by +5 percent.",
            "contradiction",
            ["a"],
        ),
        (
            "condition",
            "Which policy raises wages only when workers are unionized?",
            "The policy raises wages only when workers are unionized.",
            "The policy raises wages whether or not workers are unionized.",
            "different_condition",
            ["a"],
        ),
        (
            "attribution",
            "Which argument does Marx reject?",
            "Marx rejects the claim that machines independently create value.",
            "Marx endorses the claim that machines independently create value.",
            "different_attribution",
            ["a"],
        ),
        (
            "independent_authors",
            "What do the authors report about automation and profits?",
            "Author A independently reports that profits increased after automation.",
            "Author B independently reports that profits increased after automation.",
            "independent_evidence",
            ["a", "b"],
        ),
        (
            "long_negation",
            "Which report says automation did not increase wages?",
            background + "The study found that automation did not increase wages.",
            background + "The study found that automation increased wages.",
            "contradiction",
            ["a"],
        ),
        (
            "long_quantity",
            "Which report records a ten-percent increase in wages?",
            background + "The study measured a 10 percent increase in wages.",
            background + "The study measured a 90 percent increase in wages.",
            "different_quantity",
            ["a"],
        ),
        (
            "unsupported_near_topic",
            "What voltage powers the factory robots?",
            "Workers collectively own and control the factory robots.",
            "The factory robots assemble car doors on the production line.",
            "different_claims",
            [],
        ),
        (
            "unsupported_different_topic",
            "What is the boiling point of liquid nitrogen at standard pressure?",
            "The managers monitor the workers in the factory.",
            "The policy increased unemployment in the region.",
            "different_claims",
            [],
        ),
        (
            "exact_reprint",
            "Who monitors the workers?",
            "The managers monitor the workers.",
            "The managers monitor the workers.",
            "exact_reprint",
            ["a", "b"],
        ),
    ]
    return [
        {
            "id": name,
            "query": query,
            "pair_relation": relation,
            "expected_answer_ids": expected,
            "passages": [{"id": "a", "text": left}, {"id": "b", "text": right}],
        }
        for name, query, left, right, relation, expected in cases
    ]


def cosine_matrix(left: Any, right: Any) -> np.ndarray:
    """Exact angular scores; a bad vector invalidates a measurement."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Cosine inputs must be two matrices with matching dimensions")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cosine inputs must be finite")
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(an == 0) or np.any(bn == 0):
        raise ValueError("Cosine inputs must have nonzero norms")
    return np.clip((a / an[:, None]) @ (b / bn[:, None]).T, -1.0, 1.0)


def score_distribution(scores: Any) -> dict[str, Any]:
    """Describe score bands, not probabilities or independent sample counts."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("A score distribution needs finite, nonempty observations")
    return {
        "comparison_count": int(values.size),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "fraction_between_0_7_and_0_9": float(
            np.mean((values >= 0.7) & (values <= 0.9))
        ),
    }


def admission_projection(
    scores: Sequence[float], floor: float, margin: float
) -> dict[str, Any]:
    """Project the score rule only; eligibility and production search are untested."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Admission scores must be a finite sequence")
    best = float(values.max()) if values.size else None
    threshold = (
        min(floor, best - margin)
        if best is not None and best >= floor and margin > 0
        else floor
    )
    admitted = [bool(value >= threshold) for value in values]
    return {
        "kind": "score_only_policy_simulation",
        "production_gate_executed": False,
        "floor": floor,
        "relative_margin": margin,
        "best": best,
        "effective_floor": threshold,
        "admitted": admitted,
    }


def analyse_probe(
    probe: Mapping[str, Any],
    query_vector: Any,
    passage_vectors: Any,
    *,
    floor: float,
    margin: float,
    duplicate_threshold: float,
    vector_origin: str = "caller_supplied",
) -> dict[str, Any]:
    passages = probe["passages"]
    if len(passages) != 2 or len({item["id"] for item in passages}) != 2:
        raise ValueError("A contrast probe requires exactly two distinct passage IDs")
    vectors = np.asarray(passage_vectors, dtype=np.float32)
    if len(vectors) != len(passages):
        raise ValueError("Every passage must have exactly one vector")
    scores = cosine_matrix(np.asarray(query_vector).reshape(1, -1), vectors)[0]
    order = np.argsort(-scores, kind="stable").tolist()
    ids = [item["id"] for item in passages]
    expected = set(probe["expected_answer_ids"])
    if not expected.issubset(ids):
        raise ValueError("An expected answer is absent from the probe")
    direct = [i for i, item in enumerate(ids) if item in expected]
    contrasts = [i for i, item in enumerate(ids) if item not in expected]
    chunks = {
        item["id"]: {
            "chunk_id": item["id"],
            "document_id": item["id"],
            "contents": item["text"],
        }
        for item in passages
    }
    documents = {
        item: {
            "document_id": item,
            "source_id": "src_" + item,
            "source_relative_path": item + ".synthetic",
        }
        for item in ids
    }
    kept, collapsed = _collapse_repetitions(
        ids,
        chunks_by_id=chunks,
        documents_by_id=documents,
        vectors={item["text"]: vectors[i] for i, item in enumerate(passages)},
        threshold=duplicate_threshold,
    )
    projection = admission_projection(scores.tolist(), floor, margin)
    return {
        **dict(probe),
        "expectation_kind": "constructed_answer_selection_not_corpus_relevance",
        "query_cosines": {item: float(scores[i]) for i, item in enumerate(ids)},
        "ranked_ids": [ids[i] for i in order],
        "expected_answer_ranked_first": ids[order[0]] in expected if expected else None,
        "answer_over_contrast_margin": float(
            max(scores[i] for i in direct) - max(scores[i] for i in contrasts)
        )
        if direct and contrasts
        else None,
        "admission": projection,
        "expected_answer_rejected": [
            ids[i] for i in direct if not projection["admitted"][i]
        ],
        "unsupported_candidate_admitted": any(projection["admitted"])
        if not expected
        else None,
        "passage_pair_cosine": float(cosine_matrix(vectors[:1], vectors[1:2])[0, 0]),
        "collapse": {
            "engine_executed": True,
            "vector_origin": vector_origin,
            "input_order": ids,
            "threshold": duplicate_threshold,
            "kept_ids": kept,
            "pairs": collapsed,
            "distinct_evidence_suppressed": bool(collapsed)
            and probe["pair_relation"] not in {"equivalent_wording", "exact_reprint"},
        },
    }


def query_vectors(runtime: ModelRuntime, questions: Sequence[str]) -> np.ndarray:
    """Use the production query route, not passage_embed for both sides."""
    rows = list(runtime._embedder().query_embed(list(questions), batch_size=1))
    if len(rows) != len(questions):
        raise ValueError("The model did not return one vector per question")
    return np.asarray(rows, dtype=np.float32)


def run_constructed(
    runtime: ModelRuntime, settings: Any, compare_reranker: bool
) -> dict[str, Any]:
    probes = constructed_probes()
    texts = list(
        dict.fromkeys(item["text"] for probe in probes for item in probe["passages"])
    )
    vectors = runtime.embed_texts(texts)
    tokens = runtime.embedding_token_counts(texts)
    vector_by_text = dict(zip(texts, vectors, strict=True))
    token_by_text = dict(zip(texts, tokens, strict=True))
    queries = query_vectors(runtime, [probe["query"] for probe in probes])
    results = []
    for i, probe in enumerate(probes):
        texts_for_probe = [item["text"] for item in probe["passages"]]
        result = analyse_probe(
            probe,
            queries[i],
            [vector_by_text[text] for text in texts_for_probe],
            floor=settings.dense_minimum_cosine_similarity,
            margin=settings.dense_relative_similarity_margin,
            duplicate_threshold=settings.duplicate_cosine,
            vector_origin="fresh_pinned_model",
        )
        result["input_token_counts"] = [token_by_text[text] for text in texts_for_probe]
        result["input_truncated"] = any(
            token_by_text[text] > runtime.embedding_facts.maximum_tokens
            for text in texts_for_probe
        )
        if compare_reranker:
            result["reranker_scores"] = runtime.rerank(probe["query"], texts_for_probe)
            result["reranker_is_ground_truth"] = False
        results.append(result)
    return {
        "scope": "constructed_contrasts_real_model_not_incidence_rates",
        "cases": results,
        "summary": {
            "cases": len(results),
            "answer_selection_cases": sum(
                item["answer_over_contrast_margin"] is not None for item in results
            ),
            "answer_selection_failures_or_ties": [
                item["id"]
                for item in results
                if item["answer_over_contrast_margin"] is not None
                and item["answer_over_contrast_margin"] <= 0
            ],
            "answer_rejection_cases": [
                item["id"] for item in results if item["expected_answer_rejected"]
            ],
            "unsupported_admission_cases": [
                item["id"] for item in results if item["unsupported_candidate_admitted"]
            ],
            "distinct_suppression_cases": [
                item["id"]
                for item in results
                if item["collapse"]["distinct_evidence_suppressed"]
            ],
        },
    }


def evaluation_harness() -> Any:
    """Reuse the existing target-resolution protocol without editing its judgments."""
    spec = importlib.util.spec_from_file_location(
        "cosine_known_item_harness", REPOSITORY / "scripts/evaluate_retrieval.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the existing evaluation harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def engine_source_identity() -> dict[str, Any]:
    """Cover the executed collapse and its helpers, including module constants."""
    components = {}
    for function in (_collapse_repetitions, _passage_equality_key):
        module = inspect.getmodule(function)
        if module is None:
            raise ValueError("Cannot identify an executed engine module")
        components[module.__name__] = hashlib.sha256(
            inspect.getsource(module).encode()
        ).hexdigest()
    identity = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": identity, "module_sha256": components}


def stream_target_chunks(
    generation_root: Path,
    documents: Mapping[str, dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    harness: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain two matches per target and one source witness, not whole documents.

    The existing harness still decides target validity. A source witness preserves
    its distinction between missing document chunks and a missing snippet. Exact
    match counts are recorded even when an ambiguous snippet exceeds the retention
    bound; every such target has at least two retained matches and is refused.
    """
    needles: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for target in targets:
        wanted = {
            str(target.get("source_path") or "").strip(),
            str(target.get("source_relative_path") or "").strip(),
        } - {""}
        matched = [
            document
            for document in documents.values()
            if document.get("source_path") in wanted
            or document.get("source_relative_path") in wanted
        ]
        if not matched and str(target.get("document_id") or "") in documents:
            matched = [documents[str(target["document_id"])]]
        if len(matched) != 1:
            # The authoritative resolver will report missing or ambiguous metadata.
            continue
        document_id = str(matched[0]["document_id"])
        counts[str(target["target_id"])] = 0
        needles.setdefault(document_id, {})[str(target["target_id"])] = (
            harness.normalize(str(target["snippet"])).casefold()
        )

    retained: dict[int, dict[str, Any]] = {}
    witnessed: set[str] = set()
    scanned = 0
    for position, chunk in enumerate(
        iter_jsonl(generation_root / "chunks/chunks.jsonl")
    ):
        scanned += 1
        document_id = str(chunk.get("document_id") or "")
        if document_id not in needles:
            continue
        if document_id not in witnessed:
            retained[position] = chunk
            witnessed.add(document_id)
        text = harness.normalize(harness._chunk_text(chunk)).casefold()
        for target_id, snippet in needles[document_id].items():
            if snippet in text:
                counts[target_id] += 1
                if counts[target_id] <= 2:
                    retained[position] = chunk
    return list(retained.values()), {
        "method": "streamed_snippet_matches_with_source_witnesses",
        "scanned_chunk_count": scanned,
        "retained_record_count": len(retained),
        "exact_match_counts": counts,
    }


def corpus_pool(
    config: ResearchConfig, judgments_path: Path, skip: frozenset[str]
) -> dict[str, Any]:
    """Read a bounded judged pool, refusing builds, exclusions, and ambiguous targets."""
    if config.staging_root.exists() and any(config.staging_root.iterdir()):
        raise ValueError(
            "An ingestion staging build exists; refuse to measure alongside it"
        )
    generation_root, manifest = load_current_generation(config.state_root)
    harness = evaluation_harness()
    judged = harness.load_judgments(judgments_path)
    if not skip.issubset({item["target_id"] for item in judged["targets"]}):
        raise ValueError("A requested skip does not name an existing judged target")
    records = manifest.get("documents")
    if not isinstance(records, list) or not records:
        raise harness.EvaluationError("The generation manifest lists no documents")
    documents = {
        str(document["document_id"]): document
        for document in records
        if isinstance(document, dict) and document.get("document_id")
    }
    selected = [
        target for target in judged["targets"] if target["target_id"] not in skip
    ]
    chunks, resolution = stream_target_chunks(
        generation_root, documents, selected, harness
    )
    try:
        resolved = harness.resolve_targets(
            judged["targets"], chunks, documents, skip=skip
        )
    except harness.EvaluationError as exc:
        raise harness.EvaluationError(
            f"Validation over bounded retained records failed: {exc}\n"
            "Exact streamed snippet match counts: "
            + json.dumps(resolution["exact_match_counts"], sort_keys=True)
        ) from exc
    source_exclusions = load_source_exclusions(config.source_exclusions_path)
    chunk_exclusions = load_chunk_exclusions(config.chunk_exclusions_path)
    for target in resolved.values():
        document = documents[target["document_id"]]
        if (
            document.get("source_relative_path") in source_exclusions
            or target["chunk_id"] in chunk_exclusions
        ):
            raise ValueError(
                "A resolved target is reviewed as excluded; require an explicit judged-target skip"
            )
    return {
        "targets": resolved,
        "queries": [
            item for item in judged["queries"] if item["target_id"] not in skip
        ],
        "manifest": manifest,
        "target_resolution": resolution,
    }


def run_corpus(
    pool: Mapping[str, Any],
    runtime: ModelRuntime,
    settings: Any,
    compare_reranker: bool,
) -> dict[str, Any]:
    targets, questions = pool["targets"], pool["queries"]
    ids = list(targets)
    texts = [targets[item]["chunk_text"] for item in ids]
    passages = runtime.embed_texts(texts)
    counts = runtime.embedding_token_counts(texts)
    queries = query_vectors(runtime, [item["query"] for item in questions])
    similarities = cosine_matrix(queries, passages)
    rows = []
    for i, question in enumerate(questions):
        scores = similarities[i]
        order = np.argsort(-scores, kind="stable").tolist()
        target_position = ids.index(question["target_id"])
        projection = admission_projection(
            scores.tolist(),
            settings.dense_minimum_cosine_similarity,
            settings.dense_relative_similarity_margin,
        )
        row = {
            "query_id": question["query_id"],
            "class": question["class"],
            "target_id": question["target_id"],
            "target_rank": order.index(target_position) + 1,
            "target_cosine": float(scores[target_position]),
            "target_admitted_in_pool_simulation": projection["admitted"][
                target_position
            ],
            "cosine_range": [float(scores.min()), float(scores.max())],
            "cosine_scores": dict(zip(ids, map(float, scores), strict=True)),
            "ranked_target_ids": [ids[index] for index in order],
            "admission": projection,
        }
        if compare_reranker:
            rerank = runtime.rerank(question["query"], texts)
            ranked = np.argsort(-np.asarray(rerank), kind="stable").tolist()
            row["reranker_target_rank"] = ranked.index(target_position) + 1
        rows.append(row)
    dense = pool["manifest"].get("retrieval", {}).get("dense", {})
    return {
        "scope": "fresh_bounded_known_item_pool_not_full_corpus_retrieval",
        "generation_id": pool["manifest"]["generation_id"],
        "generation_dense_identity": dense,
        "generation_vectors_used": False,
        "reembedded_in_memory": True,
        "pool_target_count": len(ids),
        "query_count": len(rows),
        "target_resolution": pool["target_resolution"],
        "competitor_relevance": "unjudged_not_assumed_irrelevant",
        "targets": [
            {
                "target_id": item,
                "chunk_id": targets[item]["chunk_id"],
                "document_id": targets[item]["document_id"],
                "locator": targets[item]["locator"],
                "text_sha256": hashlib.sha256(texts[i].encode()).hexdigest(),
                "token_count": counts[i],
                "truncated": counts[i] > runtime.embedding_facts.maximum_tokens,
            }
            for i, item in enumerate(ids)
        ],
        "queries": rows,
        "score_distributions": {
            "all_pool_comparisons": score_distribution(similarities),
            "designated_target_comparisons": score_distribution(
                [item["target_cosine"] for item in rows]
            ),
            "pool_best_comparisons": score_distribution(similarities.max(axis=1)),
            "independent_samples": False,
        },
        "summary": {
            "designated_target_first": sum(item["target_rank"] == 1 for item in rows),
            "designated_target_top_three": sum(
                item["target_rank"] <= 3 for item in rows
            ),
            "designated_target_rejected_in_pool_simulation": sum(
                not item["target_admitted_in_pool_simulation"] for item in rows
            ),
            "reranker_designated_target_first": sum(
                item["reranker_target_rank"] == 1 for item in rows
            )
            if compare_reranker
            else None,
        },
    }


def read_only_config(project: Path | None) -> ResearchConfig:
    """Resolve existing layers without resolve_config's initialization or migration."""
    root = (project or REPOSITORY).expanduser().resolve()
    layers = sources_for(root)
    if project is None:
        layers = replace(layers, project_config=None)
    values, provenance = resolve_settings(
        SETTINGS, layers, overrides=["runtime.offline=true"]
    )
    settings = EffectiveSettings.from_values(values)
    portable = root / PORTABLE_DIRECTORY
    state = (
        (recorded_runtime_root(portable) or portable / RUNTIME_DIRECTORY)
        if project
        else portable / RUNTIME_DIRECTORY
    )
    source = root / configured_source_directory(root) if project else root
    if not root.is_dir() or not source.resolve().is_relative_to(root):
        raise ValueError("The project or source root is invalid")
    cache = (
        settings.model_cache_root
        or user_cache_path(USER_CONFIG_DIRECTORY, appauthor=False) / "models"
    )
    config = ResearchConfig(
        project_root=root,
        source_root=source,
        state_root=state,
        portable_root=portable,
        project_id="",
        project_name="constructed-only audit",
        vanilla_executable=Path(sys.executable).parent / GATEWAY_EXECUTABLE,
        runtime_cache_root=None,
        model_cache_root=cache.resolve(),
        settings=settings,
        settings_provenance=dict(provenance),
    )
    if (
        settings.offline
        and settings.model_cache_root is None
        and not _contains_files(config.model_cache_root)
        and _contains_files(config.legacy_models_root)
    ):
        config = replace(config, model_cache_root=config.legacy_models_root)
    if project:
        descriptor = json.loads(config.project_config_path.read_text(encoding="utf-8"))
        if (
            not isinstance(descriptor, dict)
            or not descriptor.get("project_id")
            or not descriptor.get("name")
        ):
            raise ValueError("The existing project descriptor has no identity")
        config = replace(
            config, project_id=descriptor["project_id"], project_name=descriptor["name"]
        )
    return config


def protected_project_hashes(config: ResearchConfig) -> dict[str, str | None]:
    """Verify canonical inputs stayed unchanged without recording their contents."""
    root, _ = load_current_generation(config.state_root)
    paths = {
        "current_pointer": config.current_path,
        "project_descriptor": config.project_config_path,
        "source_exclusions": config.source_exclusions_path,
        "chunk_exclusions": config.chunk_exclusions_path,
        "source_metadata": config.metadata_path,
        "generation_manifest": root / "manifest.json",
        "generation_chunks": root / "chunks/chunks.jsonl",
    }
    fingerprints = {}
    for name, path in paths.items():
        if path.is_file():
            with path.open("rb") as handle:
                fingerprints[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            fingerprints[name] = None
    return fingerprints


def validate_report_path(path: Path, project: Path | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(REPOSITORY):
        raise ValueError("Write the audit report outside the repository")
    if project is not None and resolved.is_relative_to(project.resolve()):
        raise ValueError("Write the audit report outside the measured project")
    parents = resolved.parents
    if resolved.is_relative_to(REPORT_SCRATCH):
        parents = [
            parent for parent in parents if parent.is_relative_to(REPORT_SCRATCH)
        ]
    if any((parent / ".research-rag").is_dir() for parent in parents):
        raise ValueError("Write the audit report outside any research project")
    if resolved.exists():
        raise ValueError("Use a new report filename; do not overwrite an earlier run")
    if not resolved.parent.is_dir():
        raise ValueError("The report's parent directory must already exist")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        help="read-only project supplying existing judged passages",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=REPOSITORY / "evaluation/ai-and-fetishism-queries.json",
    )
    parser.add_argument(
        "--skip-targets",
        default="",
        help="explicit reviewer-approved skips, comma-separated",
    )
    parser.add_argument(
        "--compare-reranker",
        action="store_true",
        help="score the same pool; never treat it as ground truth",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="diagnostic inference threads; no setting is persisted",
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = validate_report_path(args.report, args.project)
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        config = read_only_config(args.project)
        skip = frozenset(
            item.strip() for item in args.skip_targets.split(",") if item.strip()
        )
        protected_before = protected_project_hashes(config) if args.project else None
        pool = corpus_pool(config, args.judgments, skip) if args.project else None
        pointer_before = config.current_path.read_bytes() if pool else None
        apply_process_priority(config.nice)
        runtime = ModelRuntime(
            config.models_root,
            offline=True,
            embedding_threads=args.threads,
            embedding_model=config.settings.embedding_model,
            reranker_model=config.reranker_model,
        )
        facts = runtime.embedding_facts
        engine_identity = engine_source_identity()
        report = {
            "report_version": REPORT_VERSION,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "engine_source_sha256": engine_identity["sha256"],
            "engine_source_modules": engine_identity["module_sha256"],
            "model": {
                "name": facts.name,
                "repository": facts.repository,
                "revision": facts.revision,
                "dimension": facts.dimension,
                "maximum_tokens": facts.maximum_tokens,
                "threads": args.threads,
            },
            "reranker_model": config.reranker_model if args.compare_reranker else None,
            "reranker_identity": dict(
                zip(
                    ("repository", "revision"),
                    resolve_reranker_model(config.reranker_model),
                    strict=True,
                )
            )
            if args.compare_reranker
            else None,
            "retrieval_thresholds": {
                "dense_floor": config.settings.dense_minimum_cosine_similarity,
                "dense_relative_margin": config.settings.dense_relative_similarity_margin,
                "duplicate_cosine": config.settings.duplicate_cosine,
            },
            "skipped_targets": sorted(skip),
            "scope": {
                "network_used": False,
                "settings_mutated": False,
                "generation_mutated": False,
                "sources_mutated": False,
                "production_search_executed": False,
                "search_history_written": False,
            },
            "unmeasured": [
                "Full-corpus candidate recall and graded relevance",
                "Production gate eligibility and abstention calibration",
                "Corpus-wide false duplicate incidence",
                "Independent held-out validation of constructed cases",
                "Alternative embedding models or distance functions",
            ],
            "constructed": run_constructed(
                runtime, config.settings, args.compare_reranker
            ),
        }
        if pool:
            report["corpus"] = run_corpus(
                pool, runtime, config.settings, args.compare_reranker
            )
            if config.current_path.read_bytes() != pointer_before:
                raise ValueError(
                    "The selected generation changed during the audit; discard the run"
                )
            if protected_project_hashes(config) != protected_before:
                raise ValueError(
                    "Canonical project inputs changed during the audit; discard the run"
                )
            report["protected_project_files"] = {
                "sha256_before_and_after": protected_before,
                "unchanged": True,
            }
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "report": str(destination),
                    "constructed": report["constructed"]["summary"],
                    "corpus": report.get("corpus", {}).get("summary"),
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"cosine audit setup or execution failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
