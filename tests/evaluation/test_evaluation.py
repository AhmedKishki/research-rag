"""The harness is a measurement script, not a shipped module, so it is loaded by path."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate_retrieval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_retrieval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluation():
    return load_module()


def test_success_at_and_reciprocal_rank(evaluation) -> None:
    ranked = ["c1", "c2", "c3"]
    relevant = {"c2"}
    assert evaluation.success_at(ranked, relevant, 1) is False
    assert evaluation.success_at(ranked, relevant, 2) is True
    assert evaluation.reciprocal_rank(ranked, relevant) == pytest.approx(0.5)
    assert evaluation.reciprocal_rank(ranked, {"c1"}) == 1.0
    assert evaluation.reciprocal_rank(ranked, {"absent"}) == 0.0


def test_ndcg_at_rewards_higher_ranks(evaluation) -> None:
    assert evaluation.ndcg_at(["c1"], {"c1"}, 10) == pytest.approx(1.0)
    # Rank 2 is discounted by log2(3) against the ideal rank-1 placement.
    assert evaluation.ndcg_at(["c2", "c1"], {"c1"}, 10) == pytest.approx(
        1 / 1.584962500721156
    )
    assert evaluation.ndcg_at(["c2"], {"c1"}, 10) == 0.0
    assert evaluation.ndcg_at(["c2", "c3"], {"c1"}, 1) == 0.0


def test_lexical_overlap_ignores_stopwords_and_short_tokens(evaluation) -> None:
    passage = "The cobalt supply chain depends on artisanal mining in the Congo."
    assert evaluation.lexical_overlap("cobalt mining Congo", passage) == pytest.approx(
        1.0
    )
    assert evaluation.lexical_overlap("cobalt lithium Congo", passage) == pytest.approx(
        2 / 3
    )
    # Nothing to match, so no signal.
    assert evaluation.lexical_overlap("what is the of it", passage) == 0.0
    assert evaluation.content_tokens("the and it of") == set()


def test_normalize_collapses_wrapping(evaluation) -> None:
    assert evaluation.normalize("a\n  b\tc ") == "a b c"


def test_load_judgments_rejects_unknown_class_and_missing_target(
    evaluation, tmp_path: Path
) -> None:
    good = {
        "schema_version": 1,
        "targets": [
            {"target_id": "t1", "source_path": "sources/a.pdf", "snippet": "x"}
        ],
        "queries": [
            {"query_id": "q1", "class": "quote", "target_id": "t1", "query": "x"}
        ],
    }
    path = tmp_path / "set.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert evaluation.load_judgments(path)["queries"][0]["query_id"] == "q1"

    bad_class = json.loads(json.dumps(good))
    bad_class["queries"][0]["class"] = "vibes"
    path.write_text(json.dumps(bad_class), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_judgments(path)

    bad_target = json.loads(json.dumps(good))
    bad_target["queries"][0]["target_id"] = "missing"
    path.write_text(json.dumps(bad_target), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_judgments(path)

    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_judgments(path)


def _chunk(chunk_id: str, contents: str, document_id: str = "doc_1") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "contents": contents,
        "locator": {"type": "pdf_page", "page": 1},
    }


DOCUMENTS = {
    "doc_1": {
        "document_id": "doc_1",
        "source_path": "sources/a.pdf",
        "source_relative_path": "a.pdf",
    }
}


def test_resolve_targets_uses_the_snippet_as_an_identity_key(evaluation) -> None:
    chunks = [
        _chunk("c1", "the cobalt supply chain depends on artisanal mining"),
        _chunk("c2", "artisanal mining in the Congo exposes workers to hazards"),
    ]
    targets = [
        {
            "target_id": "t1",
            "source_path": "sources/a.pdf",
            "snippet": "artisanal\n  mining in the Congo",
            "chunk_id_at_measurement": "c2",
        }
    ]

    resolved = evaluation.resolve_targets(targets, chunks, DOCUMENTS)

    assert resolved["t1"]["chunk_id"] == "c2"
    assert resolved["t1"]["chunk_id_at_measurement"] == "c2"
    assert resolved["t1"]["document_id"] == "doc_1"
    assert resolved["t1"]["source_path"] == "sources/a.pdf"
    assert "artisanal mining in the Congo" in resolved["t1"]["chunk_text"]


def test_resolve_targets_matches_a_renamed_source_by_document_id(evaluation) -> None:
    """A target survives a rename because the manifest still names its document."""

    chunks = [_chunk("c1", "shared wording here")]
    targets = [
        {
            "target_id": "t1",
            "source_path": "sources/renamed.pdf",
            "source_relative_path": "renamed.pdf",
            "document_id": "doc_1",
            "snippet": "shared wording here",
        }
    ]

    resolved = evaluation.resolve_targets(targets, chunks, DOCUMENTS)

    assert resolved["t1"]["chunk_id"] == "c1"
    assert resolved["t1"]["source_path"] == "sources/a.pdf"


def test_resolve_targets_skips_requested_targets(evaluation) -> None:
    """A target whose source the corpus no longer holds is skipped on request."""

    chunks = [_chunk("c1", "shared wording here")]
    targets = [
        {
            "target_id": "t1",
            "source_path": "sources/a.pdf",
            "snippet": "shared wording here",
        },
        {
            "target_id": "t2",
            "source_path": "sources/excluded.pdf",
            "snippet": "no longer in the corpus",
        },
    ]

    resolved = evaluation.resolve_targets(
        targets,
        chunks,
        DOCUMENTS,
        skip=frozenset({"t2"}),
    )

    # The skip is explicit, so the absent target is left out rather than guessed
    # at, and every other target still resolves strictly.
    assert set(resolved) == {"t1"}


def test_resolve_targets_refuses_ambiguity_and_absent_sources(evaluation) -> None:
    chunks = [
        _chunk("c1", "shared wording here"),
        _chunk("c2", "shared wording here too"),
    ]
    ambiguous = [
        {
            "target_id": "t1",
            "source_path": "sources/a.pdf",
            "snippet": "shared wording here",
        }
    ]
    with pytest.raises(evaluation.EvaluationError):
        evaluation.resolve_targets(ambiguous, chunks, DOCUMENTS)

    absent = [{"target_id": "t1", "source_path": "sources/b.pdf", "snippet": "x"}]
    with pytest.raises(evaluation.EvaluationError):
        evaluation.resolve_targets(absent, chunks, DOCUMENTS)

    missing_text = [
        {"target_id": "t1", "source_path": "sources/a.pdf", "snippet": "nowhere"}
    ]
    with pytest.raises(evaluation.EvaluationError):
        evaluation.resolve_targets(missing_text, chunks, DOCUMENTS)


def _run(mode: str, query_class: str, *, hit: bool, rank: float) -> dict:
    return {
        "mode": mode,
        "class": query_class,
        "success_at_1": hit and rank == 1.0,
        "success_at_3": hit,
        "success_at_k": hit,
        "reciprocal_rank": rank,
        "ndcg_at_k": rank,
        "document_success_at_k": True,
        "lexical_overlap": 0.5,
        "result_count": 10,
        "distinct_source_count": 4,
        "withheld_total": 0,
    }


def test_summarize_reports_modes_and_classes_separately(evaluation) -> None:
    runs = [
        _run("bm25", "quote", hit=True, rank=1.0),
        _run("bm25", "paraphrase", hit=False, rank=0.0),
        _run("dense", "paraphrase", hit=True, rank=1.0),
    ]

    summary = evaluation.summarize(runs, 10)

    assert summary["bm25"]["overall"]["query_count"] == 2
    assert summary["bm25"]["overall"]["success_at_1"] == pytest.approx(0.5)
    assert summary["bm25"]["overall"]["mrr"] == pytest.approx(0.5)
    # A reordering change that holds success flat matters only if it moves source
    # spread, so it is reported beside the quality columns.
    assert summary["bm25"]["overall"]["mean_distinct_sources"] == pytest.approx(4.0)
    assert summary["bm25"]["per_class"]["quote"]["query_count"] == 1
    assert summary["bm25"]["per_class"]["quote"]["success_at_1"] == pytest.approx(1.0)
    assert summary["dense"]["per_class"]["paraphrase"]["success_at_1"] == pytest.approx(
        1.0
    )
    assert "quote" not in summary["dense"]["per_class"]


def test_shipped_judged_set_is_structurally_valid(evaluation) -> None:
    path = Path(__file__).parents[2] / evaluation.DEFAULT_JUDGMENTS

    payload = evaluation.load_judgments(path)

    classes = {item["class"] for item in payload["queries"]}
    assert classes == set(evaluation.QUERY_CLASSES)
    assert len(payload["queries"]) >= 30
    assert len(payload["targets"]) >= 15
    assert all(target["snippet"] for target in payload["targets"])
    assert all(target["chunk_id_at_measurement"] for target in payload["targets"])


def test_mode_variants_expand_the_reranked_mode_per_model(evaluation) -> None:
    variants = evaluation._mode_variants(
        list(evaluation.MODES),
        ["Xenova/ms-marco-MiniLM-L-6-v2", "jinaai/jina-reranker-v1-turbo-en"],
    )

    labels = [label for label, _ in variants]
    # The default model keeps the plain label its published numbers use, and a second
    # model is a second row over the same queries, not a new mode.
    assert labels == [
        "bm25",
        "dense",
        "hybrid",
        "hybrid+rerank",
        "hybrid+rerank[jinaai/jina-reranker-v1-turbo-en]",
    ]
    assert variants[3][1] == {
        "retrieval_method": "hybrid",
        "rerank": True,
        "rerank_model": evaluation.DEFAULT_RERANKER_MODEL,
    }
    assert variants[4][1]["rerank_model"] == "jinaai/jina-reranker-v1-turbo-en"
    assert all("rerank_model" not in settings for _, settings in variants[:3])
    assert all(settings["rerank"] is False for _, settings in variants[:3])


def _index(*chunks: tuple[str, str, str]) -> dict[str, dict]:
    """An index keyed by chunk id, holding canonical text and a source id."""

    return {
        chunk_id: {"chunk_id": chunk_id, "contents": contents, "source_id": source_id}
        for chunk_id, contents, source_id in chunks
    }


def test_duplicate_slots_count_exact_equality_beyond_the_first(evaluation) -> None:
    index = _index(
        ("c1", "artisanal mining in the Congo", "s1"),
        ("c2", "artisanal mining in the Congo", "s2"),
        ("c3", "the cobalt supply chain", "s3"),
    )
    exact, near = evaluation.duplicate_slots(["c1", "c2", "c3"], index)
    assert exact == 1
    assert near == 0


def test_duplicate_slots_ignore_wrapping_and_case(evaluation) -> None:
    index = _index(
        ("c1", "Artisanal mining\n  in the Congo", "s1"),
        ("c2", "artisanal  MINING in the congo.", "s2"),
    )
    exact, _ = evaluation.duplicate_slots(["c1", "c2"], index)
    assert exact == 1, "wrapping and case are not a different passage of evidence"


def test_duplicate_slots_report_a_near_duplicate_beside_the_exact_count(
    evaluation,
) -> None:
    index = _index(
        ("c1", "artisanal mining in the Congo exposes workers to hazards", "s1"),
        ("c2", "artisanal mining in the Congo", "s2"),
        ("c3", "the cobalt supply chain depends on refining", "s3"),
    )
    exact, near = evaluation.duplicate_slots(["c1", "c2", "c3"], index)
    # c2's words are wholly inside c1's, so c2 is the near duplicate; c3 is not.
    assert exact == 0
    assert near == 1


def test_duplicate_slots_keep_distinct_evidence_from_one_source(evaluation) -> None:
    index = _index(
        ("c1", "artisanal mining in the Congo", "s1"),
        ("c2", "the cobalt supply chain depends on refining", "s1"),
    )
    exact, near = evaluation.duplicate_slots(["c1", "c2"], index)
    # Two passages from one file are not one passage twice.
    assert (exact, near) == (0, 0)


def test_duplicate_slots_tolerate_a_chunk_the_generation_does_not_hold(
    evaluation,
) -> None:
    index = _index(("c1", "artisanal mining in the Congo", "s1"))
    exact, near = evaluation.duplicate_slots(["c1", "missing"], index)
    assert (exact, near) == (0, 0)


def test_same_source_pairs_counts_slots_beyond_the_first_of_a_source(
    evaluation,
) -> None:
    index = _index(
        ("c1", "first passage", "s1"),
        ("c2", "second passage", "s1"),
        ("c3", "third passage", "s1"),
        ("c4", "a passage from elsewhere", "s2"),
    )
    assert evaluation.same_source_pairs(["c1", "c2", "c3", "c4"], index) == 2


def test_same_source_pairs_is_zero_when_every_result_is_from_another_source(
    evaluation,
) -> None:
    index = _index(("c1", "first", "s1"), ("c2", "second", "s2"))
    assert evaluation.same_source_pairs(["c1", "c2"], index) == 0


def test_repeated_slot_rate_finds_a_passage_more_than_one_query_returned(
    evaluation,
) -> None:
    runs = [
        {"returned_chunk_ids": ["c1", "c2"]},
        {"returned_chunk_ids": ["c1", "c3"]},
        {"returned_chunk_ids": ["c4", "c5"]},
    ]
    # c1 occupies two of six slots, so a third of the results were a generic leader.
    assert evaluation.repeated_slot_rate(runs) == pytest.approx(2 / 6)


def test_repeated_slot_rate_is_zero_when_no_passage_repeats(evaluation) -> None:
    runs = [{"returned_chunk_ids": ["c1", "c2"]}, {"returned_chunk_ids": ["c3"]}]
    assert evaluation.repeated_slot_rate(runs) == 0.0


def test_repeated_slot_rate_of_a_run_with_no_results_is_zero(evaluation) -> None:
    assert evaluation.repeated_slot_rate([{"returned_chunk_ids": []}]) == 0.0


def test_latency_percentiles_report_measured_values_at_the_nearest_rank(
    evaluation,
) -> None:
    runs = [{"elapsed_seconds": float(value)} for value in range(1, 21)]
    percentiles = evaluation.latency_percentiles(runs)
    assert percentiles["p50_seconds"] == 10.0
    assert percentiles["p95_seconds"] == 19.0


def test_latency_percentiles_of_a_run_with_no_timing_is_zero(evaluation) -> None:
    assert evaluation.latency_percentiles([{"query": "x"}]) == {
        "p50_seconds": 0.0,
        "p95_seconds": 0.0,
    }


def test_result_list_measures_record_what_a_list_contained(evaluation) -> None:
    index = _index(
        ("c1", "artisanal mining in the Congo", "s1"),
        ("c2", "artisanal mining in the Congo", "s2"),
        ("c3", "the cobalt supply chain", "s1"),
    )
    measures = evaluation._result_list_measures(["c1", "c2", "c3"], index)
    assert measures == {
        "distinct_evidence_spans": 2,
        "exact_duplicate_slots": 1,
        "near_duplicate_slots": 0,
        "same_source_pairs": 1,
    }


def test_summarize_reports_redundancy_and_repetition_beside_quality(
    evaluation,
) -> None:
    runs = [
        {**_run("bm25", "quote", hit=True, rank=1.0), **_measures(3, 1, 2)},
        {**_run("bm25", "paraphrase", hit=True, rank=1.0), **_measures(3, 1, 2)},
    ]
    summary = evaluation.summarize(runs, 10)
    overall = summary["bm25"]["overall"]
    assert overall["mean_distinct_evidence_spans"] == pytest.approx(3.0)
    assert overall["mean_exact_duplicate_slots"] == pytest.approx(1.0)
    assert overall["mean_near_duplicate_slots"] == pytest.approx(0.0)
    assert overall["mean_same_source_pairs"] == pytest.approx(2.0)
    assert overall["p50_seconds"] == pytest.approx(0.0)
    assert summary["bm25"]["repeated_slot_rate"] == pytest.approx(1.0)


def test_summarize_reports_an_absent_measure_as_none_not_zero(evaluation) -> None:
    # A run record written before report version 2 carries no redundancy counts,
    # and a zero would read as an absence of duplication rather than of measurement.
    runs = [_run("bm25", "quote", hit=True, rank=1.0)]
    overall = evaluation.summarize(runs, 10)["bm25"]["overall"]
    assert overall["mean_distinct_evidence_spans"] is None
    assert overall["mean_same_source_pairs"] is None
    assert overall["mean_exact_duplicate_slots"] is None


def _measures(spans: int, exact: int, same_source: int) -> dict:
    return {
        "distinct_evidence_spans": spans,
        "exact_duplicate_slots": exact,
        "near_duplicate_slots": 0,
        "same_source_pairs": same_source,
        "returned_chunk_ids": ["c1", "c2", "c3"],
    }


def test_dense_gate_measures_read_the_engines_own_counts(evaluation) -> None:
    gate = {
        "minimum_cosine_similarity": 0.72,
        "relative_margin": 0.1,
        "best_cosine_similarity": 0.7312,
        "admitted_below_floor": 4,
        "rejected_below_floor": 118,
    }
    assert evaluation._dense_gate_measures(gate) == {
        "dense_rejected_below_floor": 118,
        "dense_admitted_below_floor": 4,
        "dense_best_cosine_similarity": 0.7312,
    }


def test_a_search_with_no_gate_block_reports_none_rather_than_zero(evaluation) -> None:
    # BM25 never opens the dense gate, so the block is absent. Zero rejections
    # there would be indistinguishable from a gate that rejected nothing.
    assert evaluation._dense_gate_measures({}) == {
        "dense_rejected_below_floor": None,
        "dense_admitted_below_floor": None,
        "dense_best_cosine_similarity": None,
    }


def test_withheld_candidates_says_nothing_about_the_gate(evaluation) -> None:
    # The two counts are about different things, and conflating them is how a run
    # record comes to report a gate as inert.
    measures = evaluation._dense_gate_measures({"rejected_below_floor": 118})
    assert measures["dense_rejected_below_floor"] == 118
    assert measures["dense_admitted_below_floor"] == 0


def test_summarize_counts_the_queries_where_the_gate_acted(evaluation) -> None:
    runs = [
        {**_run("hybrid", "quote", hit=True, rank=1.0), **_gate(0)},
        {**_run("hybrid", "paraphrase", hit=True, rank=1.0), **_gate(37)},
        {**_run("hybrid", "entity", hit=True, rank=1.0), **_gate(0)},
    ]
    overall = evaluation.summarize(runs, 10)["hybrid"]["overall"]
    assert overall["mean_dense_rejected_below_floor"] == pytest.approx(37 / 3)
    assert overall["mean_dense_admitted_below_floor"] == pytest.approx(0.0)
    assert overall["queries_with_dense_rejections"] == 1


def test_summarize_reports_the_gate_as_unmeasured_for_a_mode_without_it(
    evaluation,
) -> None:
    runs = [_run("bm25", "quote", hit=True, rank=1.0)]
    overall = evaluation.summarize(runs, 10)["bm25"]["overall"]
    assert overall["mean_dense_rejected_below_floor"] is None
    assert overall["queries_with_dense_rejections"] is None


def _gate(rejected: int) -> dict:
    return {
        "dense_rejected_below_floor": rejected,
        "dense_admitted_below_floor": 0,
        "dense_best_cosine_similarity": 0.8,
    }
