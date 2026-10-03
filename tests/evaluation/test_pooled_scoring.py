"""Tests for the pooled scoring harness.

Every label in this file is synthetic. The harness scores a pool somebody
judged, and no such pool exists in this repository yet, so nothing here reads
an annotation, a project, or a generation: the fixtures build a handoff in
memory and assert what the scorer reads out of it.

What is checked, and what each check is for:

- The arithmetic, against values worked out by hand. Graded nDCG with the
  gains the report publishes, and the two precision denominators kept apart so
  a short list cannot read as a full one.
- An absence read as an absence. A pool holding nothing above grade 0 is null
  rather than zero, and an empty return is null rather than a score.
- Refusals, one per cause: an unjudged candidate, a duplicate identity, a
  number written as a bool, a malformed provenance hash, an annotation status
  that is not complete, and a condition that asked for reranking and fell back.
- Pair telemetry counted only where an annotator wrote. An absent annotation is
  an unjudged pair, never an unrelated one, and nothing is inferred from equal
  text or from a contradiction.
- Aggregation weighting. Families weigh equally however many questions they
  hold, in the summary and in the comparison alike, and the dependence that
  creates is stated.
- The comparison's own honesty. One family has no interval, a seed reproduces
  an interval, and the report carries no winner, no equivalence proof and no
  policy acceptance.
- The write. A report is created exclusively, is owner-only, refuses an
  existing path and a symlink, refuses the handoff it scored, and refuses any
  path inside a root the handoff declares protected.

The harness is loaded by path, because ``scripts/`` is not a package.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_pooled",
    Path(__file__).resolve().parents[2] / "scripts" / "evaluate_pooled.py",
)
assert _SPEC and _SPEC.loader
pooled = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_pooled"] = pooled
_SPEC.loader.exec_module(pooled)
Refused = pooled.PoolProtocolError


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rubric_digest(rubric: Any) -> str:
    """The protocol's own canonical digest, spelled out here rather than borrowed.

    The scorer computes this itself, so the test holds it to the same four
    decisions: sorted keys, no whitespace, no ASCII escaping, UTF-8 bytes.
    """

    canonical = json.dumps(
        rubric, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


RUBRIC = {
    "rubric_id": "rubric-synthetic",
    "relevance": ["irrelevant", "contextual", "direct"],
    "usability": ["usable", "needs_context", "unusable"],
    "scale_notes": "Grades are ordered; direct is the strongest.",
}


def _provenance(**overrides: Any) -> dict[str, Any]:
    # Copied, so a test that edits a rubric cannot reach the next fixture's.
    rubric = copy.deepcopy(overrides.pop("rubric", RUBRIC))
    provenance = {
        "pool_id": "pool-synthetic",
        "pool_sha256": _digest("pool"),
        "annotation_sha256": _digest("annotation"),
        "rubric_sha256": _rubric_digest(rubric),
        "rubric": rubric,
        "generation_id": "generation-synthetic",
        "generation_chunks_sha256": _digest("generation chunks"),
        # A throwaway root per fixture run, so nothing here can name a real
        # corpus or project. main() refuses any report inside it.
        "protected_roots": [
            str(Path(tempfile.gettempdir()) / "research-rag-protected-synthetic")
        ],
    }
    provenance.update(overrides)
    return provenance


def _judgment(
    query_id: str,
    passage_id: str,
    relevance: Any = "direct",
    usability: Any = "usable",
    **overrides: Any,
) -> dict[str, Any]:
    judgment = {
        "query_id": query_id,
        "passage_id": passage_id,
        "relevance": relevance,
        "usability": usability,
        "source_verified": True,
        "notes": "",
        "source_relative_path": f"sources/{query_id}.pdf",
        "locator": {"page": 1},
        "content_sha256": _digest(f"{query_id}:{passage_id}"),
        "chunk_id": f"chunk-{query_id}-{passage_id}",
    }
    judgment.update(overrides)
    return judgment


def _ranking(query_id: str, passage_ids: list[str], **overrides: Any) -> dict[str, Any]:
    ranking = {
        "query_id": query_id,
        "requested_k": 3,
        "passage_ids": passage_ids,
        "candidate_depth": 40,
        "rerank_window": 0,
        "rerank_requested": False,
        "reranked": False,
        "rerank_fallback": None,
    }
    ranking.update(overrides)
    return ranking


def _condition(
    condition_id: str, rankings: list[dict[str, Any]], **overrides: Any
) -> dict[str, Any]:
    condition = {
        "condition_id": condition_id,
        "metadata": {"arm": condition_id},
        "rankings": rankings,
    }
    condition.update(overrides)
    return condition


def _handoff(
    questions: list[dict[str, str]],
    judgments: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    relations: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    handoff = {
        "schema_version": 1,
        "protocol": "author_pool_v1",
        "role": "exploratory",
        "provenance": _provenance(),
        "questions": questions,
        "judgments": judgments,
        "relations": relations if relations is not None else [],
        "conditions": conditions,
        "annotation_consistency": {"checked_items": 0, "quality_conflicts": []},
        "label_validation": {
            "status": "complete",
            "rubric_acknowledged": True,
            "annotator": "synthetic",
        },
    }
    handoff.update(overrides)
    return handoff


def _simple(**overrides: Any) -> dict[str, Any]:
    """One question, one direct passage, one contextual passage, one irrelevant.

    Returned worst first, so the graded arithmetic has three grades in play at
    once rather than a single hit.
    """

    return _handoff(
        questions=[{"query_id": "q1", "family_id": "f1"}],
        judgments=[
            _judgment("q1", "p1", "direct", "usable"),
            _judgment("q1", "p2", "contextual", "needs_context"),
            _judgment("q1", "p3", "irrelevant", "unusable"),
        ],
        conditions=[
            _condition("baseline", [_ranking("q1", ["p3", "p2", "p1"])]),
            _condition("treatment", [_ranking("q1", ["p1", "p2", "p3"])]),
        ],
        **overrides,
    )


def _query(
    payload: dict[str, Any], condition_id: str, query_id: str = "q1"
) -> dict[str, Any]:
    scored = next(
        item for item in payload["conditions"] if item["condition_id"] == condition_id
    )
    return next(query for query in scored["queries"] if query["query_id"] == query_id)


def _condition_of(payload: dict[str, Any], condition_id: str) -> dict[str, Any]:
    return next(
        item for item in payload["conditions"] if item["condition_id"] == condition_id
    )


# --------------------------------------------------------------------------
# The arithmetic, against gains worked out by hand. direct is grade 2 so its
# gain is 2**2 - 1 = 3; contextual is grade 1 for a gain of 1; irrelevant is
# grade 0 for a gain of 0.
# --------------------------------------------------------------------------


def test_ndcg_matches_the_gains_the_report_publishes() -> None:
    payload = pooled.evaluate(_simple())
    query = _query(payload, "baseline")
    expected = (0 / 1 + 1 / math.log2(3) + 3 / math.log2(4)) / (
        3 / 1 + 1 / math.log2(3) + 0 / math.log2(4)
    )
    assert payload["relevance_grades"] == {
        "irrelevant": 0,
        "contextual": 1,
        "direct": 2,
    }
    assert payload["gain"] == "2 ** grade - 1"
    assert query["ndcg_at_requested_k"] == pytest.approx(expected, rel=1e-9)
    assert query["ndcg_at_requested_k"] == pytest.approx(0.5869, abs=1e-4)


def test_ndcg_is_one_when_the_pool_order_is_returned() -> None:
    payload = pooled.evaluate(_simple())
    assert _query(payload, "treatment")["ndcg_at_requested_k"] == pytest.approx(1.0)


def test_ndcg_at_a_shortcut_is_the_gain_over_the_pool_ideal() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p1"])]
    payload = pooled.evaluate(handoff)
    ideal = 3 / 1 + 1 / math.log2(3) + 0 / math.log2(4)
    assert _query(payload, "baseline")["ndcg_at_requested_k"] == pytest.approx(
        3 / ideal
    )


def test_shorter_returns_keep_two_denominators_apart() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p1"], requested_k=5)]
    query = _query(pooled.evaluate(handoff), "baseline")
    assert query["direct_precision_at_requested_k"] == pytest.approx(1 / 5)
    assert query["direct_precision_returned"] == pytest.approx(1.0)


def test_an_empty_return_is_zero_at_requested_depth_and_null_returned() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", [])]
    query = _query(pooled.evaluate(handoff), "baseline")
    assert query["direct_precision_at_requested_k"] == 0.0
    assert query["direct_precision_returned"] is None
    assert query["direct_usable_precision_returned"] is None
    assert query["ndcg_at_requested_k"] == 0.0


def test_needs_context_and_unusable_are_counted_apart_from_usable() -> None:
    handoff = _simple()
    handoff["judgments"].append(_judgment("q1", "p4", "direct", "needs_context"))
    handoff["judgments"].append(_judgment("q1", "p5", "direct", "unusable"))
    handoff["conditions"][0]["rankings"] = [
        _ranking("q1", ["p1", "p4", "p5"], requested_k=3)
    ]
    query = _query(pooled.evaluate(handoff), "baseline")
    assert query["direct_at_k_grade_usability"] == {
        "usable": 1,
        "needs_context": 1,
        "unusable": 1,
    }
    assert query["direct_usable_precision_at_requested_k"] == pytest.approx(1 / 3)
    assert query["direct_precision_at_requested_k"] == pytest.approx(1.0)


def test_coverage_is_pool_relative_and_not_corpus_recall() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p2"])]
    query = _query(pooled.evaluate(handoff), "baseline")
    assert query["direct_pool_size"] == 1
    assert query["direct_coverage"] == 0.0
    definitions = pooled.METRIC_DEFINITIONS["direct_coverage"]
    assert "pool" in definitions and "not corpus recall" in definitions


def test_checked_originals_are_a_count_and_not_a_quotation_readiness() -> None:
    handoff = _simple()
    handoff["judgments"] = [
        _judgment("q1", "p1", "direct", "usable", source_verified=True),
        _judgment("q1", "p2", "direct", "usable", source_verified=False),
        _judgment("q1", "p3", "contextual", "usable"),
    ]
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p1", "p2", "p3"])]
    payload = pooled.evaluate(handoff)
    assert _query(payload, "baseline")["checked_original_direct_usable"] == 1
    assert (
        _condition_of(payload, "baseline")["counts"]["checked_original_direct_usable"]
        == 1
    )
    assert "checked_original_direct_usable" not in pooled.METRICS
    assert any(
        "not a claim that a quotation is ready" in limit for limit in payload["limits"]
    )


# --------------------------------------------------------------------------
# An absence read as an absence
# --------------------------------------------------------------------------


def test_an_all_irrelevant_pool_is_null_rather_than_zero() -> None:
    handoff = _simple()
    handoff["judgments"] = [
        _judgment("q1", "p1", "irrelevant", "unusable"),
        _judgment("q1", "p2", "irrelevant", "unusable"),
    ]
    for condition in handoff["conditions"]:
        condition["rankings"] = [_ranking("q1", ["p1", "p2"])]
    payload = pooled.evaluate(handoff)
    query = _query(payload, "baseline")
    assert query["ndcg_at_requested_k"] is None
    assert query["direct_coverage"] is None
    assert query["direct_pool_size"] == 0
    assert query["direct_precision_at_requested_k"] == 0.0
    aggregates = _condition_of(payload, "baseline")["aggregates"]["ndcg_at_requested_k"]
    assert aggregates["query_mean"] == {
        "value": None,
        "evaluated": 0,
        "total": 1,
        "unevaluable_query_ids": ["q1"],
    }


def test_an_unscorable_query_is_not_read_as_a_no_answer_finding() -> None:
    payload = pooled.evaluate(_simple())
    assert payload["no_answer"]["measured"] is False
    assert "unmeasured" in payload["no_answer"]["statement"]
    assert any("not a NoAnswer finding" in limit for limit in payload["limits"])


# --------------------------------------------------------------------------
# Refusals. Each names one cause, and a refusal is a refusal of the whole
# document rather than a figure computed over part of it.
# --------------------------------------------------------------------------


def test_an_unjudged_returned_candidate_is_refused_not_counted_irrelevant() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [
        _ranking("q1", ["p3", "p2", "p1", "p-absent"])
    ]
    with pytest.raises(Refused, match="no judgment for 'q1' covers"):
        pooled.evaluate(handoff)


def test_a_passage_judged_under_another_question_does_not_satisfy_a_ranking() -> None:
    handoff = _simple()
    handoff["questions"].append({"query_id": "q2", "family_id": "f2"})
    handoff["judgments"].append(_judgment("q2", "p9", "direct", "usable"))
    for condition in handoff["conditions"]:
        condition["rankings"].append(_ranking("q2", ["p9"]))
    handoff["conditions"][0]["rankings"][0]["passage_ids"] = ["p-absent"]
    with pytest.raises(Refused, match="no judgment for 'q1' covers"):
        pooled.evaluate(handoff)


@pytest.mark.parametrize(
    "field, value",
    [
        ("relevance", None),
        ("relevance", "uncertain"),
        ("relevance", "maybe"),
        ("usability", None),
        ("usability", "usable-ish"),
        ("source_verified", "yes"),
        ("notes", None),
        ("content_sha256", "not-a-digest"),
        ("locator", "page 1"),
    ],
)
def test_a_label_value_that_is_null_uncertain_or_the_wrong_type_is_refused(
    field: str, value: Any
) -> None:
    handoff = _simple()
    handoff["judgments"][0][field] = value
    with pytest.raises(Refused):
        pooled.evaluate(handoff)


def test_a_missing_label_key_is_refused() -> None:
    handoff = _simple()
    del handoff["judgments"][0]["usability"]
    with pytest.raises(Refused, match="judgments\\[0\\] is missing"):
        pooled.evaluate(handoff)


def test_a_duplicate_question_is_refused() -> None:
    handoff = _simple()
    handoff["questions"].append({"query_id": "q1", "family_id": "f2"})
    with pytest.raises(Refused, match="declared twice"):
        pooled.evaluate(handoff)


def test_a_duplicate_judgment_pair_is_refused() -> None:
    handoff = _simple()
    handoff["judgments"].append(_judgment("q1", "p1", "irrelevant", "unusable"))
    with pytest.raises(Refused, match="judges 'q1'/'p1' twice"):
        pooled.evaluate(handoff)


def test_a_duplicate_ranking_is_refused() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"].append(_ranking("q1", ["p1"]))
    with pytest.raises(Refused, match="ranks 'q1' twice"):
        pooled.evaluate(handoff)


def test_the_same_passage_at_two_ranks_is_refused() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p1", "p1", "p3"])]
    with pytest.raises(Refused, match="at two ranks"):
        pooled.evaluate(handoff)


def test_an_unknown_relation_name_is_refused() -> None:
    handoff = _simple(
        relations=[
            {
                "query_id": "q1",
                "left_passage_id": "p1",
                "right_passage_id": "p2",
                "relation": "same_thing",
                "notes": "",
            }
        ]
    )
    with pytest.raises(Refused, match="relations\\[0\\].relation must be one of"):
        pooled.evaluate(handoff)


def test_a_relation_to_itself_is_refused() -> None:
    handoff = _simple(
        relations=[
            {
                "query_id": "q1",
                "left_passage_id": "p1",
                "right_passage_id": "p1",
                "relation": "copy",
                "notes": "",
            }
        ]
    )
    with pytest.raises(Refused, match="not its own pair"):
        pooled.evaluate(handoff)


def test_a_duplicate_relation_is_refused_in_either_order() -> None:
    relation = {
        "query_id": "q1",
        "left_passage_id": "p1",
        "right_passage_id": "p2",
        "relation": "overlap",
        "notes": "",
    }
    mirrored = dict(relation, left_passage_id="p2", right_passage_id="p1")
    with pytest.raises(Refused, match="repeats the relation"):
        pooled.evaluate(_simple(relations=[relation, mirrored]))


def test_a_relation_across_two_questions_is_refused() -> None:
    handoff = _simple()
    handoff["questions"].append({"query_id": "q2", "family_id": "f2"})
    handoff["judgments"].append(_judgment("q2", "p9", "direct", "usable"))
    for condition in handoff["conditions"]:
        condition["rankings"].append(_ranking("q2", ["p9"]))
    handoff["relations"] = [
        {
            "query_id": "q2",
            "left_passage_id": "p1",
            "right_passage_id": "p9",
            "relation": "related_distinct",
            "notes": "",
        }
    ]
    with pytest.raises(Refused, match="judged under 'q1'"):
        pooled.evaluate(handoff)


def test_a_relation_naming_an_unjudged_passage_is_refused() -> None:
    handoff = _simple(
        relations=[
            {
                "query_id": "q1",
                "left_passage_id": "p1",
                "right_passage_id": "p-absent",
                "relation": "overlap",
                "notes": "",
            }
        ]
    )
    with pytest.raises(Refused, match="no judgment for 'q1' covers"):
        pooled.evaluate(handoff)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("protocol", "known_item"),
        ("role", "confirmatory"),
        ("pool_sha256", "abc"),
        ("generation_chunks_sha256", "0" * 63),
        ("pool_sha256", "A" * 64),
        ("checked_items", True),
    ],
)
def test_a_bad_version_string_or_hash_is_refused(field: str, value: Any) -> None:
    handoff = _simple()
    if field in {"schema_version", "protocol", "role"}:
        handoff[field] = value
    elif field in handoff["provenance"]:
        handoff["provenance"][field] = value
    else:
        handoff["annotation_consistency"][field] = value
    with pytest.raises(Refused):
        pooled.evaluate(handoff)


def test_an_annotation_status_that_is_not_complete_is_refused() -> None:
    handoff = _simple()
    handoff["label_validation"]["status"] = "partial"
    with pytest.raises(Refused, match="label_validation.status must be one of"):
        pooled.evaluate(handoff)


def test_an_unacknowledged_rubric_is_not_a_label() -> None:
    handoff = _simple()
    handoff["label_validation"]["rubric_acknowledged"] = False
    with pytest.raises(Refused, match="rubric_acknowledged must be true"):
        pooled.evaluate(handoff)


def test_provenance_takes_unknown_keys_and_metric_blocks_do_not() -> None:
    handoff = _simple()
    handoff["provenance"]["toolkit_extra"] = {"ticket": "synthetic"}
    assert pooled.evaluate(handoff)["report_meta"]["provenance"]["toolkit_extra"] == {
        "ticket": "synthetic"
    }
    handoff["judgments"][0]["annotator_confidence"] = "high"
    with pytest.raises(Refused, match="carries \\['annotator_confidence'\\]"):
        pooled.evaluate(handoff)


def test_condition_metadata_may_not_state_a_verdict_this_scorer_never_reached() -> None:
    handoff = _simple()
    handoff["conditions"][0]["metadata"]["verdict"] = "accepted"
    with pytest.raises(Refused, match="a verdict or completeness claim"):
        pooled.evaluate(handoff)


@pytest.mark.parametrize(
    "overrides, cause",
    [
        ({"requested_k": True}, "must be an integer"),
        ({"requested_k": 0}, "must be at least 1"),
        ({"candidate_depth": False}, "must be an integer"),
        ({"candidate_depth": 0}, "must be at least 1"),
        ({"candidate_depth": 80}, None),
        (
            {"rerank_requested": True, "reranked": False},
            "asked for reranking and did not get it",
        ),
        (
            {
                "rerank_requested": True,
                "reranked": False,
                "rerank_fallback": "model absent",
            },
            "records rerank fallback",
        ),
        ({"rerank_fallback": "model absent"}, "records rerank fallback"),
        ({"reranked": True}, "reranked without asking for it"),
        (
            {"rerank_requested": True, "reranked": True, "rerank_window": 0},
            "rerank_window must be positive",
        ),
    ],
)
def test_a_reranked_row_that_did_not_rerank_and_a_wrong_budget_are_refused(
    overrides: dict[str, Any], cause: str | None
) -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [
        _ranking("q1", ["p1", "p2", "p3"], **overrides)
    ]
    if cause is None:
        payload = pooled.evaluate(handoff)
        assert _query(payload, "baseline")["candidate_depth"] == 80
        return
    with pytest.raises(Refused, match=cause):
        pooled.evaluate(handoff)


def test_the_spent_budget_is_reported_per_question_and_equality_is_stated() -> None:
    budget = pooled.evaluate(_simple())["budget_equality"]
    assert budget["keys"] == ["requested_k", "candidate_depth", "rerank_window"]
    assert budget["equal_for_every_query"] is True
    assert budget["unequal_query_ids"] == []
    assert budget["per_query"]["q1"]["baseline"]["candidate_depth"] == 40
    assert "not a difference of budget" in budget["statement"]


def test_a_different_result_size_is_not_a_different_work_budget() -> None:
    handoff = _simple()
    # Same requested depth, same candidate depth, same window; one arm returned
    # fewer passages. That is an outcome, so the budgets still count as equal.
    handoff["conditions"][1]["rankings"] = [_ranking("q1", ["p1"], requested_k=3)]
    budget = pooled.evaluate(handoff)["budget_equality"]
    assert budget["equal_for_every_query"] is True
    assert budget["unequal_query_ids"] == []
    assert budget["per_query"]["q1"]["treatment"]["requested_k"] == 3
    assert budget["returned_counts"]["q1"] == {"baseline": 3, "treatment": 1}
    assert budget["returned_count_is"] == "outcome telemetry, not a work budget"
    assert "not read as a budget" in budget["statement"]
    assert "returned_count" not in json.dumps(budget["per_query"])


def test_a_real_work_difference_is_still_flagged() -> None:
    handoff = _simple()
    handoff["conditions"][1]["rankings"] = [
        _ranking("q1", ["p1", "p2", "p3"], candidate_depth=80)
    ]
    budget = pooled.evaluate(handoff)["budget_equality"]
    assert budget["equal_for_every_query"] is False
    assert budget["unequal_query_ids"] == ["q1"]
    assert budget["per_query"]["q1"]["treatment"]["candidate_depth"] == 80
    assert budget["returned_counts"]["q1"] == {"baseline": 3, "treatment": 3}
    assert "not attributable to policy alone" in budget["statement"]


def test_a_different_window_alone_is_a_work_difference() -> None:
    handoff = _simple()
    for condition in handoff["conditions"]:
        requested = condition["rankings"][0]["rerank_requested"]
        condition["rankings"] = [
            _ranking(
                "q1",
                ["p1", "p2", "p3"],
                rerank_requested=requested,
                reranked=requested,
                rerank_window=20 if requested else 0,
            )
        ]
    handoff["conditions"][1]["rankings"][0]["rerank_window"] = 40
    budget = pooled.evaluate(handoff)["budget_equality"]
    assert budget["equal_for_every_query"] is False
    assert budget["per_query"]["q1"]["treatment"]["rerank_window"] == 40


def test_a_condition_may_not_rank_a_question_the_inventory_does_not_declare() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q-absent", ["p1"])]
    with pytest.raises(Refused, match="is not a declared question"):
        pooled.evaluate(handoff)


def test_a_condition_missing_a_ranking_is_refused() -> None:
    handoff = _simple()
    handoff["questions"].append({"query_id": "q2", "family_id": "f2"})
    handoff["judgments"].append(_judgment("q2", "p9", "direct", "usable"))
    with pytest.raises(Refused, match="has no ranking for \\['q2'\\]"):
        pooled.evaluate(handoff)


def test_evaluate_refuses_a_document_it_built_itself() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"][0]["passage_ids"] = ["p-absent"]
    with pytest.raises(Refused):
        pooled.evaluate(handoff)


def test_the_report_carries_the_protocol_and_says_it_is_not_known_item() -> None:
    meta = pooled.evaluate(_simple())["report_meta"]
    assert meta["protocol"] == "author_pool_v1"
    assert meta["role"] == "exploratory"
    assert meta["measurement_kind"] == "pooled_author_labels"
    assert any("known-item" in entry for entry in meta["not_comparable_with"])
    assert meta["provenance"]["generation_id"] == "generation-synthetic"


# --------------------------------------------------------------------------
# Pair telemetry. Only annotated pairs count, and nothing is inferred.
# --------------------------------------------------------------------------


def test_a_returned_pair_with_no_annotation_is_unjudged_never_unrelated() -> None:
    handoff = _simple(
        relations=[
            {
                "query_id": "q1",
                "left_passage_id": "p3",
                "right_passage_id": "p2",
                "relation": "overlap",
                "notes": "",
            }
        ]
    )
    pairs = _query(pooled.evaluate(handoff), "baseline")["pairs"]
    assert pairs["returned_pairs_total"] == 3
    assert pairs["known_returned_pairs"] == 1
    assert pairs["unjudged_pair_count"] == 2
    assert pairs["pair_coverage"] == pytest.approx(1 / 3)
    assert pairs["relations"]["unrelated"] == 0
    assert pairs["relations"]["overlap"] == 1
    assert pairs["annotated_pairs_total"] == 1


def test_an_annotation_whose_ends_were_not_returned_is_not_counted() -> None:
    handoff = _simple()
    handoff["judgments"].append(_judgment("q1", "p4", "irrelevant", "unusable"))
    handoff["relations"] = [
        {
            "query_id": "q1",
            "left_passage_id": "p1",
            "right_passage_id": "p4",
            "relation": "copy",
            "notes": "",
        }
    ]
    pairs = _query(pooled.evaluate(handoff), "baseline")["pairs"]
    assert pairs["annotated_pairs_total"] == 1
    assert pairs["known_returned_pairs"] == 0
    assert pairs["unjudged_pair_count"] == 3
    assert pairs["pair_coverage"] == 0.0
    assert pairs["relations"]["copy"] == 0


def test_a_list_too_short_for_a_pair_has_no_pair_coverage() -> None:
    handoff = _simple()
    handoff["conditions"][0]["rankings"] = [_ranking("q1", ["p1"])]
    assert (
        _query(pooled.evaluate(handoff), "baseline")["pairs"]["pair_coverage"] is None
    )


def test_a_contradiction_does_not_lower_a_relevance_grade() -> None:
    handoff = _simple(
        relations=[
            {
                "query_id": "q1",
                "left_passage_id": "p1",
                "right_passage_id": "p2",
                "relation": "contradiction",
                "notes": "",
            }
        ]
    )
    # p2 becomes a second direct, so a system that graded a contradiction as an
    # irrelevant passage would read 2/3 here. Both grades stay at 2.
    handoff["judgments"][1]["relevance"] = "direct"
    payload = pooled.evaluate(handoff)
    query = _query(payload, "baseline")
    expected = (0 / 1 + 3 / math.log2(3) + 3 / math.log2(4)) / (
        3 / 1 + 3 / math.log2(3) + 0 / math.log2(4)
    )
    assert query["pairs"]["contradiction_returned_pairs"] == 1
    assert query["pairs"]["relations"]["contradiction"] == 1
    assert query["ndcg_at_requested_k"] == pytest.approx(expected)
    assert query["direct_precision_at_requested_k"] == pytest.approx(2 / 3)
    assert query["direct_pool_size"] == 2
    assert any("never lowers a relevance grade" in limit for limit in payload["limits"])


def test_equal_text_under_different_provenance_stays_two_passages() -> None:
    shared = _digest("the same words")
    handoff = _handoff(
        questions=[{"query_id": "q1", "family_id": "f1"}],
        judgments=[
            _judgment(
                "q1",
                "p1",
                "direct",
                "usable",
                content_sha256=shared,
                source_relative_path="sources/one.pdf",
                locator={"page": 1},
                chunk_id="chunk-one",
            ),
            _judgment(
                "q1",
                "p2",
                "direct",
                "usable",
                content_sha256=shared,
                source_relative_path="sources/two.pdf",
                locator={"page": 9},
                chunk_id="chunk-two",
            ),
        ],
        conditions=[
            _condition("baseline", [_ranking("q1", ["p1", "p2"])]),
            _condition("treatment", [_ranking("q1", ["p1", "p2"])]),
        ],
    )
    query = _query(pooled.evaluate(handoff), "baseline")
    assert query["direct_precision_returned"] == 1.0
    assert query["direct_precision_at_requested_k"] == pytest.approx(2 / 3)
    assert query["pairs"]["identical_content_distinct_passages_returned_pairs"] == 1
    assert query["pairs"]["known_returned_pairs"] == 0
    assert query["pairs"]["unjudged_pair_count"] == 1
    assert query["pairs"]["relations"]["copy"] == 0
    assert query["pairs"]["automatic_relation_inference"] is False
    assert query["pairs"]["transitive_grouping"] is False


def test_no_unique_evidence_or_suppression_figure_is_reported() -> None:
    payload = pooled.evaluate(_simple())
    scored = _query(payload, "baseline")
    for forbidden in ("unique_evidence", "suppression", "evidence_spans"):
        assert forbidden not in json.dumps(scored)
        assert all(forbidden not in metric for metric in pooled.METRICS)
    assert any("No unique evidence figure" in limit for limit in payload["limits"])


# --------------------------------------------------------------------------
# Aggregation. Families weigh equally however many questions they hold.
# --------------------------------------------------------------------------


def _unequal_families() -> dict[str, Any]:
    questions = [
        {"query_id": "q1", "family_id": "f1"},
        {"query_id": "q2", "family_id": "f1"},
        {"query_id": "q3", "family_id": "f1"},
        {"query_id": "q4", "family_id": "f2"},
    ]
    judgments = [
        judgment
        for query_id in ("q1", "q2", "q3", "q4")
        for judgment in (
            _judgment(query_id, "p1", "direct", "usable"),
            _judgment(query_id, "p2", "irrelevant", "unusable"),
        )
    ]
    return _handoff(
        questions=questions,
        judgments=judgments,
        conditions=[
            _condition(
                "baseline",
                [
                    _ranking("q1", ["p2"]),
                    _ranking("q2", ["p2"]),
                    _ranking("q3", ["p2"]),
                    _ranking("q4", ["p1"]),
                ],
            ),
            _condition(
                "treatment",
                [
                    _ranking("q1", ["p1"]),
                    _ranking("q2", ["p1"]),
                    _ranking("q3", ["p1"]),
                    _ranking("q4", ["p1"]),
                ],
            ),
        ],
    )


def test_a_family_with_more_questions_does_not_outvote_a_smaller_one() -> None:
    aggregates = pooled.evaluate(_unequal_families())["conditions"][0]["aggregates"]
    metric = aggregates["direct_precision_returned"]
    assert metric["query_mean"]["value"] == pytest.approx(0.25)
    assert metric["per_family"]["f1"]["value"] == pytest.approx(0.0)
    assert metric["per_family"]["f1"]["queries_total"] == 3
    assert metric["per_family"]["f2"]["value"] == pytest.approx(1.0)
    assert metric["family_mean"]["value"] == pytest.approx(0.5)
    assert metric["family_mean"]["families_evaluated"] == 2


def test_the_dependence_inside_a_family_is_stated() -> None:
    comparison = pooled.evaluate(_unequal_families())["comparison"]
    assert "not independent observations" in comparison["dependence"]
    assert any("resamples question families" in limit for limit in pooled.LIMITS)


# --------------------------------------------------------------------------
# The comparison. Paired per family, cluster bootstrap, no verdict.
# --------------------------------------------------------------------------


def test_the_paired_difference_is_family_weighted_too() -> None:
    comparison = pooled.evaluate(_unequal_families())["comparison"]
    metric = comparison["per_metric"]["direct_precision_returned"]
    trial = metric["conditions"]["treatment"]
    assert trial["per_family"]["f1"]["difference"] == pytest.approx(1.0)
    assert trial["per_family"]["f2"]["difference"] == pytest.approx(0.0)
    assert trial["family_weighted_difference"] == pytest.approx(0.5)
    assert trial["families_evaluated"] == 2
    query_weighted = 1.0 - 0.25
    assert trial["family_weighted_difference"] != pytest.approx(query_weighted)


def _opposed_trials() -> dict[str, Any]:
    """Two trials whose effects cancel if they are ever pooled.

    One family gains and the other holds, the other trial loses on the second
    family and holds on the first, so the mean of the two is zero while each is
    a real effect in its own right.
    """

    questions = [
        {"query_id": "q1", "family_id": "f1"},
        {"query_id": "q2", "family_id": "f1"},
        {"query_id": "q3", "family_id": "f2"},
    ]
    judgments = [
        judgment
        for query_id in ("q1", "q2", "q3")
        for judgment in (
            _judgment(query_id, "p1", "direct", "usable"),
            _judgment(query_id, "p2", "irrelevant", "unusable"),
        )
    ]
    return _handoff(
        questions=questions,
        judgments=judgments,
        conditions=[
            _condition(
                "baseline",
                [
                    _ranking("q1", ["p2"]),
                    _ranking("q2", ["p2"]),
                    _ranking("q3", ["p1"]),
                ],
            ),
            _condition(
                "trial_pos",
                [
                    _ranking("q1", ["p1"]),
                    _ranking("q2", ["p1"]),
                    _ranking("q3", ["p1"]),
                ],
            ),
            _condition(
                "trial_neg",
                [
                    _ranking("q1", ["p2"]),
                    _ranking("q2", ["p2"]),
                    _ranking("q3", ["p2"]),
                ],
            ),
        ],
    )


def test_two_trials_whose_effects_cancel_are_never_averaged_into_one_null() -> None:
    metric = pooled.evaluate(_opposed_trials())["comparison"]["per_metric"][
        "direct_precision_returned"
    ]
    assert set(metric) == {
        "baseline_condition",
        "conditions",
        "pooled_across_conditions",
    }
    assert metric["pooled_across_conditions"] is False
    assert set(metric["conditions"]) == {"trial_pos", "trial_neg"}
    positive = metric["conditions"]["trial_pos"]
    negative = metric["conditions"]["trial_neg"]
    assert positive["family_weighted_difference"] == pytest.approx(0.5)
    assert negative["family_weighted_difference"] == pytest.approx(-0.5)
    pooled_mean = (
        positive["family_weighted_difference"] + negative["family_weighted_difference"]
    ) / 2
    assert pooled_mean == pytest.approx(0.0)
    assert positive["family_weighted_difference"] != pooled_mean


def test_a_pooled_figure_does_not_inflate_the_family_count_of_either_trial() -> None:
    metric = pooled.evaluate(_opposed_trials())["comparison"]["per_metric"][
        "direct_precision_returned"
    ]
    for trial in metric["conditions"].values():
        assert trial["families_evaluated"] == 2
        assert trial["families_total"] == 2
        assert trial["queries_paired"] == 3
        assert trial["queries_total"] == 3


def test_every_trial_against_one_family_leaves_both_intervals_unusable() -> None:
    handoff = _simple()
    for question in handoff["questions"]:
        question["family_id"] = "one-family"
    handoff["conditions"].append(
        _condition("trial_pos", [_ranking("q1", ["p1", "p2", "p3"])])
    )
    conditions = pooled.evaluate(handoff)["comparison"]["per_metric"][
        "ndcg_at_requested_k"
    ]["conditions"]
    assert set(conditions) == {"treatment", "trial_pos"}
    for trial in conditions.values():
        assert trial["families_evaluated"] == 1
        assert trial["interval_95"]["usable"] is False
        assert trial["interval_95"]["low"] is None
        assert "fewer than two evaluable families" in trial["interval_95"]["reason"]


def _unequal_pairing() -> dict[str, Any]:
    """A baseline that could not score q1 at all, against a trial that could.

    The baseline's family mean for f1 averages one query while the trial's
    averages two. Subtracting those two means is not a difference of anything,
    so the paired set is the one query both sides scored.
    """

    questions = [
        {"query_id": "q1", "family_id": "f1"},
        {"query_id": "q2", "family_id": "f1"},
    ]
    judgments = [
        judgment
        for query_id in ("q1", "q2")
        for judgment in (
            _judgment(query_id, "p1", "direct", "usable"),
            _judgment(query_id, "p2", "irrelevant", "unusable"),
        )
    ]
    return _handoff(
        questions=questions,
        judgments=judgments,
        conditions=[
            _condition("baseline", [_ranking("q1", []), _ranking("q2", ["p1"])]),
            _condition("trial", [_ranking("q1", ["p2"]), _ranking("q2", ["p1"])]),
        ],
    )


def test_a_difference_is_taken_over_the_queries_both_sides_scored() -> None:
    trial = pooled.evaluate(_unequal_pairing())["comparison"]["per_metric"][
        "direct_precision_returned"
    ]["conditions"]["trial"]
    family = trial["per_family"]["f1"]
    assert family["difference"] == pytest.approx(0.0)
    assert family["difference"] != pytest.approx(-0.5)
    assert family["paired_query_ids"] == ["q2"]
    assert family["skipped_query_ids"] == ["q1"]
    assert family["paired_queries"] == 1
    assert family["skipped_queries"] == 1
    assert trial["queries_paired"] == 1
    assert trial["queries_skipped"] == 1
    assert trial["queries_total"] == 2
    assert trial["family_weighted_difference"] == pytest.approx(0.0)


def test_a_family_the_trial_could_not_score_is_reported_rather_than_dropped() -> None:
    handoff = _unequal_pairing()
    handoff["questions"].append({"query_id": "q3", "family_id": "f2"})
    handoff["judgments"].append(_judgment("q3", "p1", "direct", "usable"))
    for condition in handoff["conditions"]:
        condition["rankings"].append(
            _ranking("q3", [] if condition["condition_id"] == "trial" else ["p1"])
        )
    trial = pooled.evaluate(handoff)["comparison"]["per_metric"][
        "direct_precision_returned"
    ]["conditions"]["trial"]
    assert trial["per_family"]["f2"]["difference"] is None
    assert trial["per_family"]["f2"]["paired_queries"] == 0
    assert trial["per_family"]["f2"]["skipped_query_ids"] == ["q3"]
    assert trial["families_evaluated"] == 1
    assert trial["families_total"] == 2
    assert trial["skipped_query_ids"] == ["q1", "q3"]


def test_every_metric_carries_its_own_paired_coverage() -> None:
    comparison = pooled.evaluate(_unequal_pairing())["comparison"]
    for metric in pooled.METRICS:
        entry = comparison["per_metric"][metric]
        assert set(entry["conditions"]) == {"trial"}
        trial = entry["conditions"]["trial"]
        assert (
            trial["queries_paired"] + trial["queries_skipped"] == trial["queries_total"]
        )
        assert trial["families_evaluated"] <= trial["families_total"]
        for family in trial["per_family"].values():
            assert family["paired_queries"] + family["skipped_queries"] >= 1


def test_the_comparison_publishes_no_cross_arm_average() -> None:
    comparison = pooled.evaluate(_opposed_trials())["comparison"]
    assert "never pooled across trials" in comparison["independence"]
    for metric in pooled.METRICS:
        assert comparison["per_metric"][metric]["pooled_across_conditions"] is False
        assert "family_weighted_difference" not in comparison["per_metric"][metric]
    assert any(
        "No figure is averaged across trials" in line
        for line in comparison["interpretation"]
    )


def test_the_bootstrap_leaves_the_interval_null_below_its_draw_floor() -> None:
    interval = pooled._cluster_bootstrap([0.1, 0.9], 1, 0)
    assert interval["usable"] is False
    assert interval["low"] is None
    assert interval["high"] is None
    assert interval["draws"] == 0
    assert interval["draws_requested"] == 1
    assert "below the 200-draw floor" in interval["reason"]
    assert "not a 95% interval" in interval["reason"]
    assert "neither stability nor power" in interval["reason"]


def test_the_draw_floor_is_exactly_where_it_is_stated() -> None:
    below = pooled._cluster_bootstrap([0.1, 0.9], pooled.BOOTSTRAP_MIN_SAMPLES - 1, 0)
    at_floor = pooled._cluster_bootstrap([0.1, 0.9], pooled.BOOTSTRAP_MIN_SAMPLES, 3)
    assert below["usable"] is False
    assert at_floor["usable"] is True
    assert at_floor["draws"] == pooled.BOOTSTRAP_MIN_SAMPLES
    assert pooled.BOOTSTRAP_MIN_SAMPLES == 200
    assert pooled.DEFAULT_BOOTSTRAP_SAMPLES == 2000


def test_one_draw_through_the_api_leaves_the_point_estimate_and_no_interval() -> None:
    trial = pooled.evaluate(_unequal_families(), bootstrap_samples=1)["comparison"][
        "per_metric"
    ]["direct_precision_returned"]["conditions"]["treatment"]
    assert trial["family_weighted_difference"] == pytest.approx(0.5)
    assert trial["interval_95"]["usable"] is False
    assert trial["interval_95"]["low"] is None
    assert trial["interval_95"]["high"] is None
    assert "below the 200-draw floor" in trial["interval_95"]["reason"]


def test_one_evaluable_family_is_refused_before_the_draw_floor_is_reached() -> None:
    interval = pooled._cluster_bootstrap([0.5], 2000, 0)
    assert interval["usable"] is False
    assert interval["draws"] == 0
    assert interval["draws_requested"] == 2000
    assert "fewer than two evaluable families" in interval["reason"]


def test_the_baseline_defaults_to_the_first_declared_condition_and_is_selectable() -> (
    None
):
    first = pooled.evaluate(_unequal_families())["comparison"]
    selected = pooled.evaluate(_unequal_families(), baseline="treatment")["comparison"]
    assert first["baseline_condition"] == "baseline"
    assert first["conditions_compared"] == ["treatment"]
    assert selected["baseline_condition"] == "treatment"
    assert selected["conditions_compared"] == ["baseline"]


def test_an_undeclared_baseline_is_refused() -> None:
    with pytest.raises(Refused, match="is not a declared condition"):
        pooled.evaluate(_unequal_families(), baseline="arm-that-was-never-run")


def test_one_evaluable_family_has_no_interval() -> None:
    handoff = _simple()
    for question in handoff["questions"]:
        question["family_id"] = "one-family"
    interval = pooled.evaluate(handoff)["comparison"]["per_metric"][
        "ndcg_at_requested_k"
    ]["conditions"]["treatment"]["interval_95"]
    assert interval["usable"] is False
    assert interval["low"] is None and interval["high"] is None
    assert interval["draws"] == 0
    assert "fewer than two evaluable families" in interval["reason"]


def test_the_interval_reproduces_from_its_seed() -> None:
    handoff = _unequal_families()
    first = pooled.evaluate(handoff, bootstrap_samples=200, seed=7)["comparison"]
    again = pooled.evaluate(handoff, bootstrap_samples=200, seed=7)["comparison"]
    assert first["per_metric"] == again["per_metric"]
    meta = pooled.evaluate(handoff, bootstrap_samples=200, seed=7)["report_meta"][
        "bootstrap"
    ]
    assert meta["method"] == "cluster_bootstrap_over_families"
    assert meta["samples"] == 200
    assert meta["seed"] == 7
    assert meta["conditional_on"] == "observed families"
    assert first["interval"].endswith("seed 7, conditional on the observed families")


def test_the_interval_is_drawn_by_a_seeded_generator_not_a_global_one() -> None:
    # Two values leave the interval at the values themselves whatever the seed,
    # so the draw count is what is pinned there. Seed dependence needs a spread
    # of families wide enough for the percentile to land between them.
    degenerate = pooled._cluster_bootstrap([0.1, 0.9], 250, 3)
    assert degenerate["usable"] is True
    assert (degenerate["low"], degenerate["high"]) == (0.1, 0.9)
    assert degenerate == pooled._cluster_bootstrap([0.1, 0.9], 250, 3)

    families = [0.1, 0.2, 0.4, 0.6, 0.8]
    first = pooled._cluster_bootstrap(families, 250, 3)
    assert first["usable"] is True
    assert first == pooled._cluster_bootstrap(families, 250, 3)
    assert first != pooled._cluster_bootstrap(families, 250, 4)


def test_a_single_condition_is_reported_as_having_nothing_to_compare() -> None:
    handoff = _simple()
    handoff["conditions"] = [handoff["conditions"][0]]
    comparison = pooled.evaluate(handoff)["comparison"]
    assert comparison["status"] == "not_available"
    assert "fewer than two conditions" in comparison["reason"]


def test_the_comparison_states_it_is_exploratory_and_not_adjusted() -> None:
    comparison = pooled.evaluate(_unequal_families())["comparison"]
    assert comparison["status"] == "exploratory"
    assert comparison["multiple_comparisons"] == "not adjusted"
    assert "per query over the questions both sides scored" in comparison["pairing"]
    assert "never pooled across trials" in comparison["independence"]
    assert "per trial condition independently" in comparison["interval"]
    assert any(
        "not a population interval" in line for line in comparison["interpretation"]
    )


def test_the_report_names_no_winner_no_p_value_and_no_policy_acceptance() -> None:
    payload = pooled.evaluate(_unequal_families())
    decision = payload["decision"]
    assert decision["p_value"] is None
    assert decision["winner"] is None
    assert decision["recommendation"] is None
    assert decision["policy_acceptance"] is None
    assert decision["equivalence_proven"] is False
    assert decision["policy_ready"] is False
    statements = decision["statements"]
    assert any("No hypothesis test was run" in line for line in statements)
    assert any("no acceptance decision was performed" in line for line in statements)
    assert any("not a separation" in line for line in statements)
    assert any("not an equivalence" in line for line in statements)
    assert any(
        "conditional on the families observed here" in line for line in statements
    )
    assert not any("so nothing here separates" in line for line in statements)
    assert any("not an equivalence proof" in limit for limit in payload["limits"])


# --------------------------------------------------------------------------
# Reading and writing
# --------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_handoff_is_read_without_being_changed(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    before = source.read_bytes()
    document = pooled.load_handoff(source)
    pooled.evaluate(document)
    assert source.read_bytes() == before


def test_a_report_is_written_exclusively_and_readable(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    report = tmp_path / "report.json"
    assert pooled.main(["--input", str(source), "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["report_meta"]["protocol"] == "author_pool_v1"
    assert [item["condition_id"] for item in payload["conditions"]] == [
        "baseline",
        "treatment",
    ]


def test_a_report_is_owner_only(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    report = tmp_path / "report.json"
    pooled.main(["--input", str(source), "--report", str(report)])
    assert report.stat().st_mode & 0o777 == 0o600


def test_an_existing_report_path_is_refused_and_left_alone(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    report = tmp_path / "report.json"
    report.write_text("earlier findings", encoding="utf-8")
    assert pooled.main(["--input", str(source), "--report", str(report)]) == 2
    assert report.read_text(encoding="utf-8") == "earlier findings"


def test_a_symlinked_report_path_is_refused(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    target = tmp_path / "elsewhere.json"
    target.write_text("untouched", encoding="utf-8")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    assert pooled.main(["--input", str(source), "--report", str(link)]) == 2
    assert target.read_text(encoding="utf-8") == "untouched"
    assert link.is_symlink()


def test_a_report_may_not_be_its_own_input(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    assert pooled.main(["--input", str(source), "--report", str(source)]) == 2
    assert (
        json.loads(source.read_text(encoding="utf-8"))["protocol"] == "author_pool_v1"
    )


def test_a_missing_input_or_report_directory_is_refused(tmp_path: Path) -> None:
    assert (
        pooled.main(
            [
                "--input",
                str(tmp_path / "absent.json"),
                "--report",
                str(tmp_path / "r.json"),
            ]
        )
        == 2
    )
    source = _write(tmp_path, "handoff.json", _simple())
    assert (
        pooled.main(
            ["--input", str(source), "--report", str(tmp_path / "absent" / "r.json")]
        )
        == 2
    )


def test_a_missing_bootstrap_draw_count_is_refused(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    report = tmp_path / "report.json"
    assert (
        pooled.main(
            [
                "--input",
                str(source),
                "--report",
                str(report),
                "--bootstrap-samples",
                "0",
            ]
        )
        == 2
    )
    assert not report.exists()


def test_a_malformed_handoff_file_is_refused_by_cause(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    report = tmp_path / "report.json"
    assert pooled.main(["--input", str(broken), "--report", str(report)]) == 2
    assert not report.exists()


def test_help_documents_the_protocol_and_its_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        pooled.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    assert "author_pool_v1" in text
    assert "not corpus recall" in text
    assert "Abstention is unmeasured" in text
    assert "no winner, no recommendation, no policy acceptance" in text
    assert "--baseline" in text
    assert "--seed" in text


# --------------------------------------------------------------------------
# Provenance the scorer enforces: the rubric a reader must see, the roots a
# report must stay out of, and the fact that a digest is only checked for form.
# --------------------------------------------------------------------------


def test_the_rubric_digest_is_verified_and_published_as_verified() -> None:
    rubric = {
        "rubric_id": "rubric-2",
        "relevance": ["irrelevant", "contextual", "direct"],
        "notes": "A boundary call is a needs_context direct, never a context.",
    }
    payload = pooled.evaluate(_simple(provenance=_provenance(rubric=rubric)))
    meta = payload["report_meta"]
    assert meta["provenance"]["rubric"] == rubric
    assert meta["rubric_verification"] == {
        "rubric_sha256": _rubric_digest(rubric),
        "verified": True,
        "canonical_json": (
            "sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False, "
            "separators=(',',':'), allow_nan=False).encode('utf-8'))"
        ),
    }
    assert not any("form only" in limit for limit in payload["limits"])


def test_the_canonical_digest_sorts_keys_and_drops_whitespace() -> None:
    assert _rubric_digest({"b": 1, "a": 2}) == _digest('{"a":2,"b":1}')
    assert _rubric_digest({"a": {"d": 1, "c": 2}}) == _digest('{"a":{"c":2,"d":1}}')
    assert _rubric_digest([1, 2]) == _digest("[1,2]")


def test_the_canonical_digest_does_not_escape_non_ascii() -> None:
    assert _rubric_digest({"note": "grade é"}) == _digest('{"note":"grade é"}')
    assert _rubric_digest({"note": "grade é"}) != _digest('{"note":"grade \\u00e9"}')


def test_a_rubric_edited_after_it_was_hashed_is_refused() -> None:
    handoff = _simple()
    before = handoff["provenance"]["rubric_sha256"]
    handoff["provenance"]["rubric"]["relevance"] = [
        "irrelevant",
        "contextual",
        "direct",
        "off",
    ]
    with pytest.raises(Refused, match="does not match the digest it arrived with"):
        pooled.evaluate(handoff)
    assert handoff["provenance"]["rubric_sha256"] == before
    assert _rubric_digest(handoff["provenance"]["rubric"]) != before


def test_a_rubric_swapped_for_another_is_refused() -> None:
    handoff = _simple()
    handoff["provenance"]["rubric"] = dict(RUBRIC, rubric_id="rubric-other")
    with pytest.raises(Refused, match="does not match the digest"):
        pooled.evaluate(handoff)


def test_a_tampered_rubric_is_refused_on_the_way_through_a_file(tmp_path: Path) -> None:
    handoff = _simple()
    source = _write(tmp_path, "handoff.json", handoff)
    tampered = json.loads(source.read_text(encoding="utf-8"))
    tampered["provenance"]["rubric"]["scale_notes"] = "direct means almost relevant"
    tampered["provenance"]["rubric"]["usability"] = ["usable"]
    _write(tmp_path, "tampered.json", tampered)
    report = tmp_path / "report.json"
    assert (
        pooled.main(
            ["--input", str(tmp_path / "tampered.json"), "--report", str(report)]
        )
        == 2
    )
    assert not report.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_rubric_value_that_is_not_finite_is_refused(value: float) -> None:
    handoff = _simple()
    handoff["provenance"]["rubric"]["calibration_offset"] = value
    with pytest.raises(Refused, match="none of those is a NaN or an infinity"):
        pooled.evaluate(handoff)


def test_a_non_finite_token_in_a_handoff_file_is_refused(tmp_path: Path) -> None:
    source = _write(tmp_path, "handoff.json", _simple())
    text = source.read_text(encoding="utf-8")
    source.write_text(
        text.replace('"scale_notes"', '"calibration_offset": NaN, "scale_notes"', 1),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    assert pooled.main(["--input", str(source), "--report", str(report)]) == 2
    assert not report.exists()


def test_a_nested_non_finite_value_names_its_own_place() -> None:
    handoff = _simple()
    handoff["provenance"]["rubric"]["calibration"] = {"offset": [1.0, float("nan")]}
    with pytest.raises(Refused, match=r"provenance\.rubric\.calibration\.offset\[1\]"):
        pooled.evaluate(handoff)


def test_a_rubric_value_json_cannot_carry_is_refused() -> None:
    handoff = _simple()
    handoff["provenance"]["rubric"]["anchors"] = {"not", "a", "json", "value"}
    with pytest.raises(Refused, match="has no canonical JSON form"):
        pooled.evaluate(handoff)


def test_the_other_three_digests_stay_declared_receipts() -> None:
    payload = pooled.evaluate(_simple())
    provenance = payload["report_meta"]["provenance"]
    assert len(provenance["rubric_sha256"]) == 64
    assert any(
        "receipts this harness cannot check" in limit for limit in payload["limits"]
    )
    assert any(
        "pool_sha256, annotation_sha256 and generation_chunks_sha256" in limit
        for limit in payload["limits"]
    )


def test_an_empty_or_missing_rubric_is_refused() -> None:
    with pytest.raises(Refused, match="provenance.rubric is empty"):
        pooled.evaluate(_simple(provenance=_provenance(rubric={})))
    handoff = _simple()
    del handoff["provenance"]["rubric"]
    with pytest.raises(Refused, match="provenance is missing"):
        pooled.evaluate(handoff)


@pytest.mark.parametrize("roots", [[], [""], "not-a-list", [None]])
def test_protected_roots_must_be_declared_as_a_list_of_paths(roots: Any) -> None:
    with pytest.raises(Refused, match="provenance.protected_roots"):
        pooled.evaluate(_simple(provenance=_provenance(protected_roots=roots)))


def test_a_report_inside_a_protected_root_is_refused(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    source = _write(
        tmp_path,
        "handoff.json",
        _simple(provenance=_provenance(protected_roots=[str(protected)])),
    )
    report = protected / "report.json"
    assert pooled.main(["--input", str(source), "--report", str(report)]) == 2
    assert not report.exists()
    assert list(protected.iterdir()) == []


def test_a_protected_root_covers_itself_and_every_path_below_it(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    nested = protected / "deeper" / "still"
    nested.mkdir(parents=True)
    source = _write(
        tmp_path,
        "handoff.json",
        _simple(provenance=_provenance(protected_roots=[str(protected)])),
    )
    assert (
        pooled.main(["--input", str(source), "--report", str(protected / "r.json")])
        == 2
    )
    assert (
        pooled.main(["--input", str(source), "--report", str(nested / "r.json")]) == 2
    )
    allowed = tmp_path / "elsewhere" / "r.json"
    allowed.parent.mkdir()
    assert pooled.main(["--input", str(source), "--report", str(allowed)]) == 0
    assert allowed.exists()


def test_a_sibling_directory_sharing_a_name_prefix_is_not_protected(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "project"
    sibling = tmp_path / "project-reports"
    protected.mkdir()
    sibling.mkdir()
    source = _write(
        tmp_path,
        "handoff.json",
        _simple(provenance=_provenance(protected_roots=[str(protected)])),
    )
    assert (
        pooled.main(["--input", str(source), "--report", str(sibling / "r.json")]) == 0
    )


def test_the_refusal_names_the_root_it_compared_against(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(Refused, match="is inside the protected root"):
        pooled._refuse_report_target(str(protected / "r.json"), None, [str(protected)])


def test_the_report_publishes_the_roots_it_was_kept_out_of(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    payload = pooled.evaluate(
        _simple(provenance=_provenance(protected_roots=[str(protected)]))
    )
    assert payload["report_meta"]["protected_roots_checked"] == [
        str(protected.resolve())
    ]


def test_a_handoff_inside_a_protected_root_is_read_and_never_written_over(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    source = _write(
        protected,
        "handoff.json",
        _simple(provenance=_provenance(protected_roots=[str(protected)])),
    )
    before = source.read_bytes()
    assert (
        pooled.main(["--input", str(source), "--report", str(protected / "r.json")])
        == 2
    )
    assert source.read_bytes() == before
    assert [entry.name for entry in protected.iterdir()] == ["handoff.json"]


def test_a_condition_id_may_contain_a_slash_and_no_path_is_formed_from_one(
    tmp_path: Path,
) -> None:
    handoff = _simple(
        provenance=_provenance(protected_roots=[str(tmp_path / "protected")])
    )
    for condition in handoff["conditions"]:
        condition["condition_id"] = f"sweep/{condition['condition_id']}"
    payload = pooled.evaluate(handoff, baseline="sweep/baseline")
    assert [item["condition_id"] for item in payload["conditions"]] == [
        "sweep/baseline",
        "sweep/treatment",
    ]
    assert payload["comparison"]["baseline_condition"] == "sweep/baseline"
    assert payload["comparison"]["conditions_compared"] == ["sweep/treatment"]
    source = _write(tmp_path, "handoff.json", handoff)
    report = tmp_path / "report.json"
    assert pooled.main(["--input", str(source), "--report", str(report)]) == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["conditions"][1]["condition_id"] == "sweep/treatment"
    assert not (tmp_path / "sweep").exists()
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "handoff.json",
        "report.json",
    ]
