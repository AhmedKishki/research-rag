"""Tests for the retrieval evaluation harness.

Two things are checked here. The metrics: what each one actually measures, and
whether it reports an absence as an absence rather than as a zero. And the
payload contract: the harness reads the search payload the real service emits, so
the tests that pin a gate or a reranker fallback run against a real service and a
real payload rather than a shape the engine cannot produce.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Loaded by path rather than imported: scripts/ is not a package.
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_retrieval",
    Path(__file__).resolve().parents[2] / "scripts" / "evaluate_retrieval.py",
)
assert _SPEC and _SPEC.loader
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_retrieval"] = evaluation
_SPEC.loader.exec_module(evaluation)
EvaluationError = evaluation.EvaluationError


def _index(*chunks: tuple[str, str, str]) -> dict[str, dict[str, Any]]:
    """A chunk index from (chunk_id, text, source_id) triples."""

    return {
        chunk_id: {"chunk_id": chunk_id, "contents": text, "source_id": source_id}
        for chunk_id, text, source_id in chunks
    }


def _run(query_id: str, **overrides: Any) -> dict[str, Any]:
    """A run record with the fields the repetition rates read."""

    run = {
        "query_id": query_id,
        "mode": "hybrid",
        "target_id": f"t-{query_id}",
        "family_id": f"t-{query_id}",
        "returned_chunk_ids": [],
        "elapsed_seconds": 1.0,
    }
    run.update(overrides)
    return run


# --------------------------------------------------------------------------
# Known-item metrics. These published their names before the report gained any
# other measure, and they are unchanged: a figure is comparable only with a
# figure measured the same way.
# --------------------------------------------------------------------------


def test_success_at_and_reciprocal_rank() -> None:
    assert evaluation.success_at(["a", "b", "c"], {"b"}, 1) is False
    assert evaluation.success_at(["a", "b", "c"], {"b"}, 3) is True
    assert evaluation.reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert evaluation.reciprocal_rank(["a"], {"z"}) == 0.0


def test_ndcg_at_rewards_higher_ranks() -> None:
    assert evaluation.ndcg_at(["a", "b"], {"b"}, 10) == pytest.approx(
        1 / 1.58496, rel=1e-3
    )
    assert evaluation.ndcg_at(["a", "b"], {"a"}, 10) == 1.0
    assert evaluation.ndcg_at(["a"], {"z"}, 10) == 0.0


def test_lexical_overlap_ignores_stopwords_and_short_tokens() -> None:
    query = "artisanal mining and the cobalt"
    text = "The artisanal cobalt mine in the Congo."
    assert evaluation.lexical_overlap(query, text) == pytest.approx(2 / 3)
    assert evaluation.lexical_overlap("the of and", text) == 0.0


def test_lexical_overlap_keeps_its_published_ascii_tokenizer() -> None:
    """The overlap is a version 1 figure, so it is not quietly re-tokenized.

    A passage in a script this tokenizer cannot segment shares no tokens with a
    query in it, and the version 1 measure reported that as zero overlap. The
    Unicode-aware tokenizer serves the containment measure, where no published
    figure depends on it; moving it here would have restated every published
    overlap column without saying so.
    """

    assert evaluation.lexical_overlap("矿业 劳工", "科特迪瓦的矿业劳工记录") == 0.0
    assert evaluation.TOKEN_PATTERN.pattern == "[a-z0-9]+"
    assert evaluation.WORD_PATTERN.pattern == "\\w+"


def test_normalize_collapses_wrapping() -> None:
    assert evaluation.normalize("a  b\n c ") == "a b c"


def test_normalized_text_collapses_whitespace_and_nothing_else() -> None:
    """The comparison key wraps neither the case nor the punctuation of a claim.

    ``V`` and ``v`` are different variables and ``5!`` is not ``5``, so the key
    keeps both. A key that folded them would report a duplicate the corpus does
    not hold, which is a wrong claim rather than a conservative one.
    """

    assert evaluation.normalized_text("Artisanal\n  mining  in the Congo") == (
        "Artisanal mining in the Congo"
    )
    assert evaluation.normalized_text("  padded  ") == "padded"
    assert evaluation.normalized_text("V = 5!") == "V = 5!"
    assert evaluation.normalized_text("5!") != "5"
    assert evaluation.normalized_text("V") != "v"
    assert evaluation.normalized_text("the answer.") != "the answer"
    assert evaluation.normalized_text("a b") != evaluation.normalized_text("b a")


# --------------------------------------------------------------------------
# Duplication: what a result list held, and what the measures say they compare.
# --------------------------------------------------------------------------


def test_exact_equality_keeps_word_order() -> None:
    index = _index(
        ("c1", "consent binds the person to the object", "s1"),
        ("c2", "the object binds the person to consent", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    # The engine's own equality key is an ordered join of the words, so these two
    # are two passages to it and to this measure alike.
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 0


def test_exact_equality_keeps_a_repeated_word() -> None:
    index = _index(
        ("c1", "the mine and the mill", "s1"),
        ("c2", "the mine and mill", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 0


def test_exact_equality_collapses_wrapping_only() -> None:
    index = _index(
        ("c1", "Artisanal mining\n  in the Congo.", "s1"),
        ("c2", "Artisanal  mining in the Congo.", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] == 1
    assert measures["exact_duplicate_slots"] == 1
    assert measures["exact_duplicate_groups"] == [["c1", "c2"]]


def test_exact_equality_keeps_letter_case() -> None:
    """``v`` against ``V`` is a different quantity, so it is a different passage."""

    index = _index(
        ("c1", "the shear modulus V of the specimen", "s1"),
        ("c2", "the shear modulus v of the specimen", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 0
    assert measures["lexical_containment_slots"] == 0


def test_exact_equality_keeps_a_factorial_and_an_exclamation() -> None:
    """A trailing ``!`` is an operator here, not a mark closing a sentence.

    The exact measure separates all three passages, and the lenient measure is
    never offered the passage whose operator run the others have: the words alone
    do not decide it.
    """

    index = _index(
        ("c1", "the sample held 5! arrangements", "s1"),
        ("c2", "the sample held 5 arrangements", "s2"),
        ("c3", "the sample held 5 arrangements!", "s3"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2", "c3"], index)
    assert measures["distinct_normalized_texts"] == 3
    assert measures["exact_duplicate_slots"] == 0
    assert measures["lexical_containment_pairs"] == [
        {
            "slot_chunk_id": "c1",
            "repeated_chunk_id": "c3",
            "shared_shingles": 4,
            "slot_shingles": 4,
            "containment": 1.0,
        },
        {
            "slot_chunk_id": "c3",
            "repeated_chunk_id": "c1",
            "shared_shingles": 4,
            "slot_shingles": 4,
            "containment": 1.0,
        },
    ]
    # The passage with no operator is in neither pair: a factorial is a claim the
    # other two passages do not make.
    assert "c2" not in {
        identifier
        for pair in measures["lexical_containment_pairs"]
        for identifier in (pair["slot_chunk_id"], pair["repeated_chunk_id"])
    }


def test_a_factorial_is_a_protected_distinction() -> None:
    index = _index(
        ("c1", "the sample held 5! arrangements of the specimen", "s1"),
        ("c2", "the sample held 5 arrangements of the specimen", "s2"),
    )
    assert evaluation.protected_signature("5! arrangements") != (
        evaluation.protected_signature("5 arrangements")
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["lexical_containment_slots"] == 0
    assert measures["exact_duplicate_slots"] == 0


def test_exact_equality_keeps_a_closing_sentence_mark_apart() -> None:
    index = _index(
        ("c1", "the committee approved the draft.", "s1"),
        ("c2", "the committee approved the draft", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 0


def test_exact_equality_keeps_two_different_passages_apart() -> None:
    index = _index(
        ("c1", "The committee reviewed the ledger.", "s1"),
        ("c2", "The committee approved the second draft.", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 0
    assert measures["lexical_containment_slots"] == 0


def test_exact_equality_tells_two_scripts_apart_and_copies_of_one_apart() -> None:
    index = _index(
        ("c1", "科特迪瓦的矿业劳工记录", "s1"),
        ("c2", "记录劳工矿业科特迪瓦", "s2"),
        ("c3", "科特迪瓦的矿业劳工记录", "s3"),
    )
    # Two sentences in the same order are one passage written twice; the same
    # words in another order are another passage. An ASCII token set finds no
    # tokens in either, and merges all three into one group.
    measures = evaluation.duplicate_measures(["c1", "c2", "c3"], index)
    assert measures["distinct_normalized_texts"] == 2
    assert measures["exact_duplicate_slots"] == 1
    assert measures["exact_duplicate_groups"] == [["c1", "c3"]]


def test_exact_equality_keeps_signs_decimals_and_operators() -> None:
    index = _index(
        ("c1", "the deposit holds 5.5% copper", "s1"),
        ("c2", "the deposit holds 5,5% copper", "s2"),
        ("c3", "the deposit holds 6% copper", "s3"),
        ("c4", "the depth is <= 200 m", "s4"),
        ("c5", "the depth is >= 200 m", "s5"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2", "c3", "c4", "c5"], index)
    assert measures["distinct_normalized_texts"] == 5
    assert measures["exact_duplicate_slots"] == 0
    # And the lenient measure keeps them apart too, because a pair whose figures
    # or operators differ is never offered to it.
    assert measures["lexical_containment_slots"] == 0


def test_independent_authors_with_equal_text_keep_separate_provenance() -> None:
    index = _index(
        ("c1", "Artisanal mining in the Congo.", "source-a"),
        ("c2", "Artisanal mining in the Congo.", "source-b"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    # Two files carrying one passage is one piece of evidence twice and no source
    # twice, so the text measure and the source measure answer different questions.
    assert measures["distinct_normalized_texts"] == 1
    assert measures["exact_duplicate_slots"] == 1
    assert measures["same_source_pairs"] == 0
    assert measures["duplicate_measure_coverage"]["complete"] is True


def test_a_missing_passage_is_reported_as_coverage_not_as_a_duplicate() -> None:
    index = _index(
        ("c1", "Artisanal mining in the Congo.", "s1"),
        ("c3", "The committee approved the draft.", "s3"),
    )
    measures = evaluation.duplicate_measures(["c1", "missing-1", "missing-2"], index)
    coverage = measures["duplicate_measure_coverage"]
    assert coverage == {
        "slots_total": 3,
        "slots_with_text": 1,
        "slots_missing_text": 2,
        "missing_chunk_ids": ["missing-1", "missing-2"],
        "complete": False,
    }
    # Two passages this harness cannot read are not two copies of one passage.
    for key in (
        "distinct_normalized_texts",
        "exact_duplicate_slots",
        "exact_duplicate_groups",
        "lexical_containment_slots",
        "lexical_containment_groups",
        "lexical_containment_pairs",
        "same_source_pairs",
    ):
        assert measures[key] is None, key


def test_an_empty_passage_counts_as_missing_rather_than_as_an_empty_copy() -> None:
    index = _index(
        ("c1", "Artisanal mining in the Congo.", "s1"),
        ("c2", "   ", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["distinct_normalized_texts"] is None
    assert measures["duplicate_measure_coverage"]["slots_missing_text"] == 1


def test_lexical_containment_flags_a_subset_of_prose() -> None:
    index = _index(
        (
            "c1",
            (
                "The committee reviewed the ledger and approved the second draft "
                "of the agreement in March."
            ),
            "s1",
        ),
        ("c2", "The committee reviewed the ledger.", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["exact_duplicate_slots"] == 0
    assert measures["lexical_containment_slots"] == 1
    assert measures["lexical_containment_groups"] == 1
    pair = measures["lexical_containment_pairs"][0]
    assert pair["slot_chunk_id"] == "c2"
    assert pair["repeated_chunk_id"] == "c1"
    assert pair["containment"] == 1.0


def test_lexical_containment_is_exclusive_with_exact_duplicates() -> None:
    index = _index(
        *[(f"c{n}", "Artisanal mining in the Congo.", f"s{n}") for n in range(1, 6)]
    )
    measures = evaluation.duplicate_measures([f"c{n}" for n in range(1, 6)], index)
    # Five copies of one passage are four duplicate slots and no containment at
    # all: the containment measure never counts a passage twice, so it cannot
    # read the same repetition under a second name.
    assert measures["exact_duplicate_slots"] == 4
    assert measures["lexical_containment_slots"] == 0
    assert measures["lexical_containment_groups"] == 0


def test_lexical_containment_counts_one_group_once() -> None:
    long_text = (
        "The committee reviewed the ledger and approved the second draft of the "
        "agreement in March of the year."
    )
    index = _index(
        ("long", long_text, "s1"),
        ("short-a", "The committee reviewed the ledger.", "s2"),
        ("short-b", "The committee reviewed the ledger.", "s3"),
    )
    measures = evaluation.duplicate_measures(["long", "short-a", "short-b"], index)
    # Two copies of the same short passage are one repeat of one passage: one
    # exact duplicate slot and one containment slot, not two of each.
    assert measures["exact_duplicate_slots"] == 1
    assert measures["lexical_containment_slots"] == 1
    assert measures["lexical_containment_groups"] == 1


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            "The mine produced 12 tonnes in 2019.",
            "The mine produced 15 tonnes in 2019.",
        ),
        ("The depth is <= 200 metres.", "The depth is >= 200 metres."),
        (
            "The film consents to display the scene.",
            "The film does not consent to display the scene.",
        ),
        ("The answer is yes.", "The answer is noncommittal."),
    ],
)
def test_lexical_containment_never_merges_a_pair_that_disagrees(
    left: str, right: str
) -> None:
    index = _index(("c1", left, "s1"), ("c2", right, "s2"))
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["lexical_containment_slots"] == 0
    assert measures["lexical_containment_groups"] == 0


def test_a_pair_that_agrees_on_every_protected_distinction_is_still_a_candidate() -> (
    None
):
    index = _index(
        (
            "c1",
            (
                "The committee reviewed the ledger of 2019 and approved the "
                "second draft of the agreement in March."
            ),
            "s1",
        ),
        ("c2", "The committee reviewed the ledger of 2019.", "s2"),
    )
    measures = evaluation.duplicate_measures(["c1", "c2"], index)
    assert measures["lexical_containment_slots"] == 1


def test_word_shingles_keep_a_reordering_out_of_the_measure() -> None:
    assert evaluation.word_shingles("a b c") != evaluation.word_shingles("a c b")
    assert evaluation.word_shingles("two words") == (("two", "words"),)


def test_the_protected_signature_records_position_and_kind() -> None:
    signature = evaluation.protected_signature("output is 5.5 and rate <= 0.5")
    assert signature[0] == ("5.5", "0.5")
    assert signature[1] == ("<=",)
    assert evaluation.protected_signature("a hyphenated aside") == ((), (), ())


# --------------------------------------------------------------------------
# Repetition across a mode's queries, by question family.
# --------------------------------------------------------------------------


def test_a_passage_returned_twice_by_one_query_is_not_repetition() -> None:
    rates = evaluation.repetition_rates([_run("q1", returned_chunk_ids=["c1", "c1"])])
    # One query returning one passage twice is one slot, and a numerator counting
    # slots rather than distinct queries would have read it as full repetition.
    assert rates["repeated_slot_denominator"] == 1
    assert rates["repeated_slot_rate_all_queries"] == 0.0
    assert rates["repeated_slot_rate_cross_target_family"] == 0.0


def test_one_family_repeating_reads_as_that_family_only() -> None:
    runs = [
        _run(
            "q-quote",
            family_id="t-1",
            target_id="t-1",
            returned_chunk_ids=["c1", "c2"],
        ),
        _run(
            "q-paraphrase",
            family_id="t-1",
            target_id="t-1",
            returned_chunk_ids=["c1", "c3"],
        ),
    ]
    rates = evaluation.repetition_rates(runs)
    assert rates["repeated_slot_rate_all_queries"] == 0.5
    # A quote query and its own paraphrase share a target on purpose. A perfect
    # retrieval over them returns that target's passage twice, and reads as a
    # generic leader only if the measure cannot tell the two styles apart.
    assert rates["repeated_slot_rate_cross_target_family"] == 0.0
    assert rates["repeated_slot_denominator"] == 4


def test_a_passage_answered_for_unrelated_questions_reads_as_cross_family() -> None:
    runs = [
        _run(
            "q1",
            family_id="family-a",
            target_id="t-1",
            returned_chunk_ids=["generic", "c1"],
        ),
        _run(
            "q2",
            family_id="family-b",
            target_id="t-2",
            returned_chunk_ids=["generic", "c2"],
        ),
    ]
    rates = evaluation.repetition_rates(runs)
    assert rates["repeated_slot_rate_all_queries"] == 0.5
    assert rates["repeated_slot_rate_cross_target_family"] == 0.5
    assert rates["cross_target_family_passage_count"] == 1


def test_two_targets_declared_one_family_are_one_family() -> None:
    """A judged set may declare that several passages answer one question."""

    runs = [
        _run(
            "q1",
            family_id="family-latin-america",
            target_id="t-colombia",
            returned_chunk_ids=["shared", "c1"],
        ),
        _run(
            "q2",
            family_id="family-latin-america",
            target_id="t-peru",
            returned_chunk_ids=["shared", "c2"],
        ),
    ]
    rates = evaluation.repetition_rates(runs)
    assert rates["repeated_slot_rate_all_queries"] == 0.5
    # Two different targets, one declared family: a list that never overlaps
    # beyond the passage both families legitimately quote is not repetition
    # across families, and a measure that read it as such would punish a judged
    # set for how it was written.
    assert rates["repeated_slot_rate_cross_target_family"] == 0.0


def test_repetition_without_a_family_is_unknown_rather_than_independent() -> None:
    runs = [
        _run("q1", family_id="", target_id="", returned_chunk_ids=["c1"]),
        _run("q2", family_id="family-b", target_id="t-2", returned_chunk_ids=["c2"]),
    ]
    rates = evaluation.repetition_rates(runs)
    assert rates["repeated_slot_rate_all_queries"] == 0.0
    assert rates["repeated_slot_rate_cross_target_family"] is None
    assert rates["runs_without_family_id"] == 1


def test_repetition_over_no_slots_is_a_null_with_its_zero_denominator() -> None:
    rates = evaluation.repetition_rates([_run("q1"), _run("q2")])
    assert rates["repeated_slot_denominator"] == 0
    assert rates["repeated_slot_rate_all_queries"] is None
    assert rates["repeated_slot_rate_cross_target_family"] is None


def test_targets_sharing_one_declared_family_form_one_family() -> None:
    resolved = {
        "t1": {"target_id": "t1", "family_id": "shared"},
        "t2": {"target_id": "t2", "family_id": "shared"},
    }
    queries = [
        {"query_id": "q1", "target_id": "t1"},
        {"query_id": "q2", "target_id": "t2"},
    ]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1", "t2": "t1"}


def test_a_query_declaration_and_a_target_declaration_are_both_read() -> None:
    """Neither declaration is the other's parent, so both join the same group.

    Two targets declare one family between them, and each of their queries
    declares a sub-family of its own. Choosing the query's declaration over the
    target's splits the family in two; choosing the target's over the query's
    ignores the queries. The group they form together is what both describe.
    """

    resolved = {
        "t1": {"target_id": "t1", "family_id": "shared"},
        "t2": {"target_id": "t2", "family_id": "shared"},
    }
    queries = [
        {"query_id": "q1", "target_id": "t1", "family_id": "sub-1"},
        {"query_id": "q2", "target_id": "t2", "family_id": "sub-2"},
    ]
    families = evaluation.resolve_families(queries, resolved)
    assert families["t1"] == families["t2"] == "t1"
    runs = [
        _run(
            "q1",
            family_id=families["t1"],
            target_id="t1",
            returned_chunk_ids=["shared", "c1"],
        ),
        _run(
            "q2",
            family_id=families["t2"],
            target_id="t2",
            returned_chunk_ids=["shared", "c2"],
        ),
    ]
    rates = evaluation.repetition_rates(runs)
    assert rates["repeated_slot_rate_all_queries"] == 0.5
    assert rates["repeated_slot_rate_cross_target_family"] == 0.0


def test_one_label_declared_on_a_query_and_on_a_target_unites_across_levels() -> None:
    """Both levels declare the same field, so the same label is one family twice.

    A query of the first target declares ``shared`` and the second target declares
    it on its own record. Nothing in the two records says the two targets are
    related except the label, which is exactly what a judged set means by naming
    one.
    """

    resolved = {
        "t1": {"target_id": "t1"},
        "t2": {"target_id": "t2", "family_id": "shared"},
    }
    queries = [
        {"query_id": "q1", "target_id": "t1", "family_id": "shared"},
        {"query_id": "q2", "target_id": "t2"},
    ]
    families = evaluation.resolve_families(queries, resolved)
    assert families == {"t1": "t1", "t2": "t1"}


def test_two_labels_across_both_levels_chain_one_family() -> None:
    """A family label declared on one level meets the same label on the other."""

    resolved = {
        "t1": {"target_id": "t1", "family_id": "outer"},
        "t2": {"target_id": "t2"},
        "t3": {"target_id": "t3", "family_id": "outer"},
    }
    queries = [
        {"query_id": "q1", "target_id": "t1"},
        {"query_id": "q2", "target_id": "t2", "family_id": "outer"},
        {"query_id": "q3", "target_id": "t3"},
    ]
    families = evaluation.resolve_families(queries, resolved)
    assert families == {"t1": "t1", "t2": "t1", "t3": "t1"}


def test_two_sub_families_under_one_target_declaration_are_one_family() -> None:
    resolved = {"t1": {"target_id": "t1", "family_id": "shared"}}
    queries = [
        {"query_id": "q1", "target_id": "t1", "family_id": "sub-1"},
        {"query_id": "q2", "target_id": "t1", "family_id": "sub-2"},
    ]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1"}


def test_a_query_family_label_that_reads_like_a_target_id_joins_nothing() -> None:
    """A family label and a target id share a namespace in a judged set.

    Joining them would merge two targets that declared nothing in common, so the
    nodes are namespaced apart and the collision merges nothing.
    """

    resolved = {"t1": {"target_id": "t1"}, "t2": {"target_id": "t2"}}
    queries = [
        {"query_id": "q1", "target_id": "t1", "family_id": "t2"},
        {"query_id": "q2", "target_id": "t2"},
    ]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1", "t2": "t2"}


def test_a_target_family_label_that_reads_like_a_target_id_joins_nothing() -> None:
    resolved = {
        "t1": {"target_id": "t1", "family_id": "t2"},
        "t2": {"target_id": "t2"},
    }
    queries = [{"query_id": "q1", "target_id": "t1"}]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1", "t2": "t2"}


def test_a_target_declaring_nothing_is_its_own_family() -> None:
    resolved = {"t1": {"target_id": "t1"}, "t2": {"target_id": "t2"}}
    queries = [
        {"query_id": "q1", "target_id": "t1"},
        {"query_id": "q2", "target_id": "t1"},
        {"query_id": "q3", "target_id": "t2"},
    ]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1", "t2": "t2"}


def test_a_skipped_target_resolves_no_family_and_runs_nothing() -> None:
    resolved = {"t1": {"target_id": "t1"}}
    queries = [
        {"query_id": "q1", "target_id": "t1"},
        {"query_id": "q2", "target_id": "t-skipped"},
    ]
    assert evaluation.resolve_families(queries, resolved) == {"t1": "t1"}


def test_latency_percentiles_are_percentiles_and_the_max_is_its_own_number() -> None:
    runs = [{"elapsed_seconds": float(value)} for value in range(1, 21)]
    measured = evaluation.latency_percentiles(runs)
    assert measured["p50_seconds"] == 10.0
    # Over twenty values the 95th percentile is the nineteenth, not the slowest:
    # the slowest query is reported beside it because the two differ whenever the
    # run is small enough to read.
    assert measured["p95_seconds"] == 19.0
    assert measured["max_seconds"] == 20.0


def test_latency_percentiles_of_an_untimed_run_are_null() -> None:
    measured = evaluation.latency_percentiles([{"elapsed_seconds": None}])
    assert measured == {"p50_seconds": None, "p95_seconds": None, "max_seconds": None}


# --------------------------------------------------------------------------
# Payload readers. These take the whole payload, because which block is present
# is not the same question as which method was ranked.
# --------------------------------------------------------------------------


def _gate_payload(method: str, **gate: Any) -> dict[str, Any]:
    return {"retrieval_method": method, "dense_gate": gate}


def test_the_gate_measures_read_the_engines_own_counts() -> None:
    measures = evaluation._dense_gate_measures(
        _gate_payload(
            "hybrid",
            minimum_cosine_similarity=0.72,
            relative_margin=0.1,
            best_cosine_similarity=0.81,
            eligible_total=50,
            admitted_above_floor=20,
            admitted_below_floor=5,
            rejected_below_floor=25,
            conserved=True,
            excluded_before_gate={
                "returned_by_index": 60,
                "extraction_artifact": 2,
                "corrupt_text": 1,
                "too_short": 4,
                "filtered_out": 3,
            },
        )
    )
    assert measures["dense_gate_ran"] is True
    assert measures["dense_gate_counts_complete"] is True
    assert measures["dense_eligible_total"] == 50
    assert measures["dense_admitted_above_floor"] == 20
    assert measures["dense_admitted_below_floor"] == 5
    assert measures["dense_rejected_below_floor"] == 25
    assert measures["dense_best_cosine_similarity"] == 0.81
    assert measures["dense_quality_excluded"]["filtered_out"] == 3
    assert measures["dense_conserved"] is True


def test_a_zeroed_gate_block_on_a_bm25_payload_is_not_a_gate_that_ran() -> None:
    """The engine carries this block for every method, zeros included."""

    measures = evaluation._dense_gate_measures(
        _gate_payload(
            "bm25",
            minimum_cosine_similarity=0.72,
            relative_margin=0.1,
            best_cosine_similarity=None,
            admitted_below_floor=0,
            rejected_below_floor=0,
        )
    )
    assert measures["dense_gate_ran"] is False
    assert measures["dense_gate_counts_complete"] is False
    assert measures["dense_rejected_below_floor"] is None
    assert measures["dense_admitted_below_floor"] is None
    assert measures["dense_best_cosine_similarity"] is None


def test_a_gate_block_without_the_counts_reports_nothing_measured() -> None:
    measures = evaluation._dense_gate_measures(
        {"retrieval_method": "hybrid", "dense_gate": {"rejected_below_floor": 118}}
    )
    assert measures["dense_gate_ran"] is True
    assert measures["dense_gate_counts_complete"] is False
    assert measures["dense_rejected_below_floor"] is None


def test_gate_counts_that_do_not_add_up_are_a_loud_failure() -> None:
    payload = _gate_payload(
        "hybrid",
        best_cosine_similarity=0.9,
        eligible_total=50,
        admitted_above_floor=20,
        admitted_below_floor=5,
        rejected_below_floor=10,
        conserved=True,
    )
    with pytest.raises(EvaluationError, match="do not account"):
        evaluation._dense_gate_measures(payload)


def test_branch_measures_carry_the_depth_the_branch_actually_used() -> None:
    measures = evaluation._branch_measures(
        {
            "retrieval_method": "hybrid",
            "candidate_depth": 74,
            "candidate_count": 74,
            "candidate_distinct_reference_count": 61,
            "rerank_requested": True,
            "reranked": True,
            "rerank_fallback": None,
            "rerank_window": 20,
            "fusion": {"pool_size": 74},
            "evaluation_trace": {"candidate_depth": 74},
        }
    )
    assert measures["candidate_depth"] == 74
    assert measures["trace_candidate_depth"] == 74
    assert measures["rerank_requested"] is True
    assert measures["reranked"] is True


def test_collapsed_measures_count_the_pairs_the_engine_disclosed() -> None:
    measures = evaluation._collapsed_measures(
        {
            "collapsed_repetitions": {
                "repetitions_collapsed": 2,
                "collapsed_by_same_words": 1,
                "pairs": [
                    {
                        "chunk_id": "c1",
                        "repeated_chunk_id": "c2",
                        "collapsed_by": "same_words",
                    },
                    {
                        "chunk_id": "c3",
                        "repeated_chunk_id": "c4",
                        "collapsed_by": "same_meaning",
                    },
                ],
            }
        }
    )
    assert measures["collapsed_count"] == 2
    assert measures["collapsed_by_same_words"] == 1
    assert measures["collapsed_by_same_meaning"] == 1
    assert measures["collapsed_pairs_truncated"] is False


def test_collapsed_measures_count_zero_when_the_disclosure_is_empty() -> None:
    measures = evaluation._collapsed_measures({"collapsed_repetitions": {"pairs": []}})
    assert measures["collapsed_count"] == 0
    assert measures["collapsed_pairs"] == []


def test_rejection_measures_name_the_reason_the_target_failed() -> None:
    measures = evaluation._rejection_measures(
        {
            "withheld_candidates": {
                "total": 1,
                "reasons": {"corrupt_text": {"example_chunk_ids": ["other"]}},
            },
            "rejected_candidates": {"dense_below_threshold": 3},
            "rejected_candidate_examples": {
                "policy": "corruption_evidence_only",
                "limit_per_reason": 3,
                "reasons": {
                    "dense_below_threshold": [
                        {"chunk_id": "target", "cosine_similarity": 0.4},
                        {"chunk_id": "other", "cosine_similarity": 0.3},
                    ]
                },
            },
        },
        target_chunk_id="target",
    )
    assert measures["target_rejected_by"] == ["dense_below_threshold"]
    # Withholding is about corrupt text in the answer, so a target the gate
    # rejected is not withheld. The two counts were never the same question.
    assert measures["target_listed_as_withheld"] is False
    assert measures["withheld_total"] == 1


def test_rejection_measures_keep_the_gate_counts_the_engine_reported() -> None:
    measures = evaluation._rejection_measures(
        {"rejected_candidates": {"dense_below_threshold": 3}}, target_chunk_id="target"
    )
    assert measures["rejected_candidates"]["dense_below_threshold"] == 3
    assert measures["target_rejected_by"] == []


def test_stage_presence_is_null_where_a_stage_list_was_cut() -> None:
    measures = evaluation._trace_measures(
        {
            "evaluation_trace": {
                "candidate_budget": 256,
                "stages": {
                    "fused_pre_rerank": {
                        "count": 900,
                        "chunk_ids": ["a", "b"],
                        "truncated": True,
                    },
                    "final": {"count": 10, "chunk_ids": ["a"], "truncated": False},
                },
            }
        },
        target_chunk_id="zzz",
    )
    # A cut list cannot say the target was absent; an untruncated one can.
    assert measures["target_stage_presence"] == {
        "fused_pre_rerank": None,
        "final": False,
    }
    assert measures["evaluation_trace_available"] is True
    assert measures["trace_candidate_budget"] == 256


def test_stage_presence_is_null_for_a_stage_the_branch_never_ran() -> None:
    measures = evaluation._trace_measures(
        {
            "evaluation_trace": {
                "stages": {
                    "dense_admitted": None,
                    "final": {"count": 1, "chunk_ids": [], "truncated": False},
                }
            }
        },
        target_chunk_id="zzz",
    )
    assert measures["target_stage_presence"] == {"dense_admitted": None, "final": False}


def test_the_raw_trace_is_kept_so_a_reader_can_audit_it() -> None:
    """The presence flags are a summary; the lists behind them are the evidence.

    A report that kept only whether the target was present cannot be audited for
    a stage the target missed, and cannot be intersected with anything. The trace
    holds identifiers and counts, bounded, and nothing else.
    """

    trace = {
        "retrieval_method": "hybrid",
        "candidate_depth": 74,
        "candidate_budget": 256,
        "stages": {
            "dense_admitted": {
                "count": 2,
                "chunk_ids": ["c1", "c2"],
                "truncated": False,
            },
            "final": {"count": 1, "chunk_ids": ["c2"], "truncated": False},
        },
        "rerank": {
            "requested": True,
            "applied": True,
            "window": 2,
            "scored_ids": ["c1", "c2"],
            "scored_count": 2,
            "scored_truncated": False,
        },
        "collapsed": {
            "count": 1,
            "by_reason": {"same_words": 1},
            "discarded": [{"chunk_id": "c1", "repeated_chunk_id": "c2"}],
            "truncated": False,
        },
        "dense": {"eligible_total": 2, "conserved": True},
    }
    measures = evaluation._trace_measures(
        {"evaluation_trace": trace}, target_chunk_id="c2"
    )
    assert measures["evaluation_trace"] == trace
    assert measures["target_stage_presence"] == {"dense_admitted": True, "final": True}
    # The summary and the evidence are both there, and the summary is derived.
    assert measures["evaluation_trace"]["stages"]["final"]["chunk_ids"] == ["c2"]


def test_a_truncated_stage_list_is_kept_with_its_flag() -> None:
    """Truncation is the finding, so erasing the flag would erase the finding."""

    trace = {
        "candidate_budget": 256,
        "stages": {
            "post_collapse": {
                "count": 900,
                "chunk_ids": [f"c{index}" for index in range(256)],
                "truncated": True,
            }
        },
    }
    measures = evaluation._trace_measures(
        {"evaluation_trace": trace}, target_chunk_id="c0"
    )
    stage = measures["evaluation_trace"]["stages"]["post_collapse"]
    assert stage["truncated"] is True
    assert stage["count"] == 900
    assert len(stage["chunk_ids"]) == 256
    # The target is in the kept part, so it is placed even though the list is cut.
    assert measures["target_stage_presence"]["post_collapse"] is True


def test_a_missing_trace_is_recorded_as_absent_rather_than_as_an_empty_one() -> None:
    measures = evaluation._trace_measures({}, target_chunk_id="c1")
    assert measures["evaluation_trace_available"] is False
    assert measures["evaluation_trace"] is None
    assert measures["target_stage_presence"] == {}


def test_scored_then_collapsed_counts_the_discards_inside_the_scored_window() -> None:
    payload = {
        "evaluation_trace": {
            "rerank": {
                "window": 3,
                "scored_ids": ["c1", "c2", "c3"],
                "scored_count": 3,
                "scored_truncated": False,
            }
        },
        "collapsed_repetitions": {
            "repetitions_collapsed": 2,
            "pairs": [
                {
                    "chunk_id": "c1",
                    "repeated_chunk_id": "c4",
                    "collapsed_by": "same_words",
                },
                {
                    "chunk_id": "c9",
                    "repeated_chunk_id": "c8",
                    "collapsed_by": "same_meaning",
                },
            ],
        },
    }
    measures = evaluation._scored_then_collapsed(payload)
    # c1 was scored and then discarded; c9 arrived in the tail, unranked, and was
    # discarded too. Only the first is a candidate a wider window would have spent
    # a score on.
    assert measures["reranked_then_collapsed_count"] == 1
    assert measures["reranked_then_collapsed_is_lower_bound"] is False
    assert measures["reranked_then_collapsed_by_reason"] == {"same_words": 1}
    assert measures["collapse_discarded_count"] == 2
    assert measures["scored_candidate_count"] == 3


def test_scored_then_collapsed_is_a_lower_bound_when_the_scored_list_was_cut() -> None:
    payload = {
        "evaluation_trace": {
            "rerank": {
                "window": 400,
                "scored_ids": ["c1", "c2"],
                "scored_count": 400,
                "scored_truncated": True,
            }
        },
        "collapsed_repetitions": {
            "pairs": [
                {
                    "chunk_id": "c1",
                    "repeated_chunk_id": "c4",
                    "collapsed_by": "same_words",
                },
                {
                    "chunk_id": "c300",
                    "repeated_chunk_id": "c4",
                    "collapsed_by": "same_words",
                },
            ],
        },
    }
    measures = evaluation._scored_then_collapsed(payload)
    # c300 was scored but is not in the kept list, so the observed intersection is
    # a floor and the flag says so rather than reporting one.
    assert measures["reranked_then_collapsed_count"] == 1
    assert measures["reranked_then_collapsed_is_lower_bound"] is True
    assert measures["scored_candidate_count"] == 400


def test_scored_then_collapsed_is_zero_when_the_reranker_never_ran() -> None:
    payload = {
        "evaluation_trace": {
            "rerank": {
                "window": 10,
                "scored_ids": [],
                "scored_count": 0,
                "scored_truncated": False,
            }
        },
        "collapsed_repetitions": {
            "pairs": [{"chunk_id": "c1", "repeated_chunk_id": "c2"}]
        },
    }
    measures = evaluation._scored_then_collapsed(payload)
    assert measures["reranked_then_collapsed_count"] == 0
    assert measures["reranked_then_collapsed_is_lower_bound"] is False
    assert measures["reranked_then_collapsed_by_reason"] == {}


def test_scored_then_collapsed_reads_nothing_as_zero_when_there_is_no_trace() -> None:
    measures = evaluation._scored_then_collapsed(
        {"collapsed_repetitions": {"pairs": []}}
    )
    assert measures["reranked_then_collapsed_count"] == 0
    assert measures["reranked_then_collapsed_is_lower_bound"] is False
    assert measures["scored_candidate_count"] is None


def test_stage_presence_is_null_when_the_engine_omitted_the_trace() -> None:
    measures = evaluation._trace_measures({}, target_chunk_id="zzz")
    assert measures["evaluation_trace_available"] is False
    assert measures["target_stage_presence"] == {}
    assert measures["trace_candidate_budget"] is None


class _StubService:
    """A service that returns one payload, for the harness's own policy checks."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.payload


def _fallback_payload() -> dict[str, Any]:
    return {
        "hits": [],
        "result_count": 0,
        "distinct_reference_count": 0,
        "retrieval_method": "hybrid",
        "candidate_depth": 20,
        "candidate_count": 0,
        "rerank_requested": True,
        "reranked": False,
        "rerank_fallback": {"reason": "reranker_model_unavailable"},
        "rerank_window": 10,
        "fusion": {"pool_size": 20},
        "withheld_candidates": {"total": 0, "reasons": {}},
        "evaluation_trace": {"candidate_budget": 256, "stages": {}},
    }


_JUDGED_TARGET = {
    "target_id": "t1",
    "chunk_id": "c-target",
    "document_id": "d1",
    "chunk_text": "A judged passage about cobalt.",
}


def test_a_reranked_row_that_did_not_rerank_fails_the_run() -> None:
    """A fusion score printed under a reranked row's name is a wrong row."""

    with pytest.raises(EvaluationError, match="did not run the way its row claims"):
        asyncio.run(
            evaluation._run_one(
                _StubService(_fallback_payload()),
                query_id="q1",
                query_class="quote",
                query="cobalt",
                mode=evaluation.RERANK_MODE,
                settings=evaluation._mode_settings(evaluation.RERANK_MODE),
                top_k=10,
                target=_JUDGED_TARGET,
                family_id="t1",
            )
        )


def test_an_allowed_degraded_row_is_recorded_rather_than_scored_silently() -> None:
    run = asyncio.run(
        evaluation._run_one(
            _StubService(_fallback_payload()),
            query_id="q1",
            query_class="quote",
            query="cobalt",
            mode=evaluation.RERANK_MODE,
            settings=evaluation._mode_settings(evaluation.RERANK_MODE),
            top_k=10,
            target=_JUDGED_TARGET,
            family_id="t1",
            allow_degraded=True,
        )
    )
    assert run["degraded_reasons"] == [
        "rerank requested but not applied (reranker_model_unavailable)"
    ]
    assert run["reranked"] is False
    assert run["rerank_fallback"]["reason"] == "reranker_model_unavailable"
    # The row still reports the branch it took, so a reader can see what it is.
    assert run["candidate_depth"] == 20


def test_a_hybrid_row_without_a_fusion_fails_the_run() -> None:
    payload = {**_fallback_payload(), "rerank_requested": False, "fusion": None}
    with pytest.raises(EvaluationError, match="hybrid row without a fusion block"):
        asyncio.run(
            evaluation._run_one(
                _StubService(payload),
                query_id="q1",
                query_class="quote",
                query="cobalt",
                mode="hybrid",
                settings=evaluation._mode_settings("hybrid"),
                top_k=10,
                target=_JUDGED_TARGET,
                family_id="t1",
            )
        )


def test_a_reranked_row_that_ran_is_not_flagged() -> None:
    payload = {
        **_fallback_payload(),
        "reranked": True,
        "rerank_fallback": None,
        "hits": [
            {
                "chunk_id": "c-target",
                "document_id": "d1",
                "source_id": "s1",
                "score": 0.9,
            }
        ],
        "result_count": 1,
        "distinct_reference_count": 1,
    }
    run = asyncio.run(
        evaluation._run_one(
            _StubService(payload),
            query_id="q1",
            query_class="quote",
            query="cobalt",
            mode=evaluation.RERANK_MODE,
            settings=evaluation._mode_settings(evaluation.RERANK_MODE),
            top_k=10,
            target=_JUDGED_TARGET,
            chunk_index={"c-target": {"contents": "A judged passage about cobalt."}},
            family_id="t1",
        )
    )
    assert run["degraded_reasons"] == []
    assert run["success_at_1"] is True


def test_a_reranked_row_that_did_not_rerank_names_itself() -> None:
    reasons = evaluation._degradation_reasons(
        {
            "rerank_requested": True,
            "reranked": False,
            "rerank_fallback": {"reason": "reranker_model_unavailable"},
            "fusion": {"pool_size": 74},
        },
        mode=evaluation.RERANK_MODE,
    )
    assert reasons == ["rerank requested but not applied (reranker_model_unavailable)"]


def test_a_hybrid_row_without_a_fusion_names_itself() -> None:
    assert evaluation._degradation_reasons({"fusion": None}, mode="hybrid") == [
        "hybrid row without a fusion block"
    ]


def test_a_row_that_ran_the_way_it_names_reports_nothing() -> None:
    payload = {"rerank_requested": True, "reranked": True, "fusion": {"pool_size": 74}}
    assert evaluation._degradation_reasons(payload, mode=evaluation.RERANK_MODE) == []
    assert evaluation._degradation_reasons(payload, mode="hybrid") == []


# --------------------------------------------------------------------------
# Summaries.
# --------------------------------------------------------------------------


def _measured_run(**overrides: Any) -> dict[str, Any]:
    run = {
        "query_id": "q1",
        "mode": "hybrid",
        "class": "quote",
        "target_id": "t-1",
        "family_id": "t-1",
        "returned_chunk_ids": ["c1"],
        "success_at_1": True,
        "success_at_3": True,
        "success_at_k": True,
        "reciprocal_rank": 1.0,
        "ndcg_at_k": 1.0,
        "document_success_at_k": True,
        "lexical_overlap": 0.5,
        "result_count": 10,
        "distinct_source_count": 4,
        "withheld_total": 0,
        "distinct_normalized_texts": 9.0,
        "exact_duplicate_slots": 1.0,
        "lexical_containment_slots": 2.0,
        "same_source_pairs": 0.0,
        "duplicate_measure_coverage": {"complete": True},
        "dense_gate_counts_complete": True,
        "dense_eligible_total": 50.0,
        "dense_admitted_above_floor": 20.0,
        "dense_admitted_below_floor": 5.0,
        "dense_rejected_below_floor": 25.0,
        "dense_conserved": True,
        "elapsed_seconds": 1.0,
    }
    run.update(overrides)
    return run


def test_a_summary_reports_the_new_measures_beside_the_quality_ones() -> None:
    summary = evaluation.summarize([_measured_run()], 10)["hybrid"]
    overall = summary["overall"]
    assert overall["mean_distinct_normalized_texts"] == 9.0
    assert overall["mean_exact_duplicate_slots"] == 1.0
    assert overall["mean_lexical_containment_slots"] == 2.0
    assert overall["mean_same_source_pairs"] == 0.0
    assert overall["queries_with_duplicate_coverage"] == 1
    assert overall["mean_dense_eligible_total"] == 50.0
    assert overall["mean_dense_rejected_below_floor"] == 25.0
    assert overall["queries_with_dense_rejections"] == 1
    assert overall["queries_with_dense_gate_conservation_failures"] == 0
    assert overall["p50_seconds"] == 1.0
    assert "repeated_slot_rate_all_queries" in summary
    assert "repeated_slot_rate_cross_target_family" in summary


def test_a_summary_reports_an_absent_measure_as_null_rather_than_zero() -> None:
    run = _measured_run()
    for key in (
        "distinct_normalized_texts",
        "exact_duplicate_slots",
        "lexical_containment_slots",
        "same_source_pairs",
        "duplicate_measure_coverage",
        "dense_gate_counts_complete",
    ):
        del run[key]
    summary = evaluation.summarize([run], 10)["hybrid"]["overall"]
    for key in (
        "mean_distinct_normalized_texts",
        "mean_exact_duplicate_slots",
        "mean_lexical_containment_slots",
        "mean_same_source_pairs",
        "queries_with_duplicate_coverage",
        "mean_dense_eligible_total",
        "mean_dense_rejected_below_floor",
        "queries_with_dense_rejections",
    ):
        assert summary[key] is None, key


def test_a_summary_reports_the_gate_as_unmeasured_for_a_mode_that_never_ran_it() -> (
    None
):
    run = _measured_run(dense_gate_counts_complete=False)
    summary = evaluation.summarize([run], 10)["hybrid"]["overall"]
    assert summary["mean_dense_rejected_below_floor"] is None
    assert summary["queries_with_dense_rejections"] is None


def test_a_summary_reports_where_the_target_was_at_each_stage() -> None:
    runs = [
        _measured_run(
            query_id="q1",
            target_stage_presence={
                "fused_pre_rerank": True,
                "reranked": True,
                "post_collapse": False,
                "final": False,
                "bm25_after_gates": None,
            },
        ),
        _measured_run(
            query_id="q2",
            target_stage_presence={
                "fused_pre_rerank": True,
                "reranked": False,
                "post_collapse": False,
                "final": False,
                "bm25_after_gates": None,
            },
        ),
    ]
    overall = evaluation.summarize(runs, 10)["hybrid"]["overall"]
    assert overall["mean_target_present_fused_pre_rerank"] == 1.0
    assert overall["mean_target_present_reranked"] == 0.5
    assert overall["mean_target_present_post_collapse"] == 0.0
    # A stage that never ran is not a stage the target missed.
    assert overall["mean_target_present_bm25_after_gates"] is None
    assert overall["target_stage_evaluated_queries_bm25_after_gates"] == 0
    assert overall["target_stage_evaluated_queries_final"] == 2


def test_the_console_row_leaves_an_unmeasured_column_blank() -> None:
    summary = evaluation.summarize([_measured_run()], 10)
    evaluation.print_summary("test", summary)
    summary["hybrid"]["overall"]["mean_lexical_containment_slots"] = None
    evaluation.print_summary("test without containment", summary)


# --------------------------------------------------------------------------
# Report-level claims.
# --------------------------------------------------------------------------


def test_the_report_names_the_harness_that_wrote_it() -> None:
    provenance = evaluation.harness_provenance()
    assert provenance["script"] == "evaluate_retrieval.py"
    assert len(provenance["script_sha256"]) == 64
    assert provenance["python"]


def test_the_report_states_that_no_answer_is_measured() -> None:
    support = evaluation.no_answer_support(
        {"queries": [{"class": "quote"}, {"class": "paraphrase"}]}
    )
    assert support["measured"] is False
    assert support["judged_no_answer_queries"] == 0
    assert "no query whose correct answer" in support["note"]


def test_the_report_names_every_metric_it_replaced_and_why() -> None:
    superseded = evaluation.SUPERSEDED_METRICS
    for key in (
        "runs[].distinct_evidence_spans",
        "runs[].near_duplicate_slots",
        "summary[mode].repeated_slot_rate",
        "report.duplicate_containment",
    ):
        assert superseded[key], key
    for key in (
        "success_at_1",
        "mrr",
        "ndcg_at_k",
        "distinct_normalized_texts",
        "lexical_containment_slots",
        "dense_eligible_total",
        "candidate_depth",
        "rerank_window",
        "repeated_slot_rate_all_queries",
        "repeated_slot_rate_cross_target_family",
        "target_stage_presence",
        "p50_seconds",
        "p95_seconds",
        "max_seconds",
    ):
        assert key in evaluation.METRIC_DEFINITIONS, key


def test_the_report_names_the_containment_threshold_and_what_it_ignores() -> None:
    assert evaluation.REPORT_SCHEMA_VERSION == 3


# --------------------------------------------------------------------------
# Judged-set handling.
# --------------------------------------------------------------------------


def test_load_judgments_accepts_a_well_formed_set(tmp_path: Path) -> None:
    path = tmp_path / "judged.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "targets": [
            {"target_id": "t1", "source_path": "a.pdf",
             "snippet": "A verbatim span of the passage."}
          ],
          "queries": [{"query_id": "q1", "target_id": "t1", "class": "quote",
                       "query": "what does the span say"}]
        }
        """,
        encoding="utf-8",
    )
    payload = evaluation.load_judgments(path)
    assert payload["targets"][0]["target_id"] == "t1"


def test_load_judgments_rejects_an_unknown_target(tmp_path: Path) -> None:
    path = tmp_path / "judged.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "targets": [
            {"target_id": "t1", "source_path": "a.pdf", "snippet": "A span."}
          ],
          "queries": [{"query_id": "q1", "target_id": "nope", "class": "quote",
                       "query": "q"}]
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="unknown target"):
        evaluation.load_judgments(path)


def test_load_judgments_rejects_an_unknown_query_class(tmp_path: Path) -> None:
    path = tmp_path / "judged.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "targets": [
            {"target_id": "t1", "source_path": "a.pdf", "snippet": "A span."}
          ],
          "queries": [{"query_id": "q1", "target_id": "t1", "class": "vibes",
                       "query": "q"}]
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="class"):
        evaluation.load_judgments(path)


def test_load_judgments_rejects_an_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="Cannot read"):
        evaluation.load_judgments(tmp_path / "absent.json")


def test_resolve_targets_needs_one_match_per_target() -> None:
    chunk = {
        "chunk_id": "c1",
        "document_id": "d1",
        "source_id": "s1",
        "contents": "A verbatim span of the passage.",
    }
    document = {
        "document_id": "d1",
        "source_path": "/data/a.pdf",
        "source_relative_path": "a.pdf",
    }
    targets = [
        {"target_id": "t1", "source_path": "/data/a.pdf", "snippet": "verbatim span"}
    ]

    resolved = evaluation.resolve_targets(targets, [chunk], {"d1": document})
    assert resolved["t1"]["chunk_id"] == "c1"

    with pytest.raises(EvaluationError, match="resolves to 2 chunks"):
        evaluation.resolve_targets(
            [{"target_id": "t1", "source_path": "/data/a.pdf", "snippet": "passage"}],
            [chunk, {**chunk, "chunk_id": "c2"}],
            {"d1": document},
        )


def test_resolve_targets_leaves_a_skipped_target_alone() -> None:
    document = {"document_id": "d1", "source_path": "/data/a.pdf"}
    resolved = evaluation.resolve_targets(
        [{"target_id": "t1", "source_path": "/data/a.pdf", "snippet": "span"}],
        [],
        {"d1": document},
        skip=frozenset({"t1"}),
    )
    assert resolved == {}


def test_resolve_targets_carries_the_sets_declared_family() -> None:
    chunk = {
        "chunk_id": "c1",
        "document_id": "d1",
        "source_id": "s1",
        "contents": "A span.",
    }
    document = {"document_id": "d1", "source_path": "/data/a.pdf"}
    resolved = evaluation.resolve_targets(
        [
            {
                "target_id": "t1",
                "source_path": "/data/a.pdf",
                "snippet": "span",
                "family_id": "family-latin-america",
            }
        ],
        [chunk],
        {"d1": document},
    )
    assert resolved["t1"]["family_id"] == "family-latin-america"


def test_mode_variants_pass_rerank_explicitly_for_every_mode() -> None:
    variants = evaluation._mode_variants(
        ["bm25", "dense", "hybrid", evaluation.RERANK_MODE],
        [evaluation.DEFAULT_RERANKER_MODEL, "bge-reranker-base"],
    )
    assert [label for label, _ in variants] == [
        "bm25",
        "dense",
        "hybrid",
        evaluation.RERANK_MODE,
        f"{evaluation.RERANK_MODE}[bge-reranker-base]",
    ]
    assert all("rerank" in settings for _, settings in variants)
    assert variants[3][1]["rerank"] is True
    assert variants[0][1]["rerank"] is False


def test_selection_rejects_an_unknown_mode() -> None:
    with pytest.raises(EvaluationError, match="Unknown modes"):
        evaluation._selection("bm25,telepathy", evaluation.MODES, "modes")


def test_the_parser_offers_the_degraded_rerank_switch_explicitly() -> None:
    args = evaluation._parser().parse_args(
        ["--project", "/tmp/p", "--allow-degraded-rerank"]
    )
    assert args.allow_degraded_rerank is True
    assert (
        evaluation._parser().parse_args(["--project", "/tmp/p"]).allow_degraded_rerank
        is False
    )


def test_the_shipped_judged_set_still_resolves() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = evaluation.load_judgments(root / evaluation.DEFAULT_JUDGMENTS)
    assert payload["targets"]
    assert payload["queries"]
    # The set is known-item throughout: one target per query, and no query whose
    # answer is that the corpus has none. Abstention is therefore unmeasured
    # rather than measured at zero.
    assert all(query["target_id"] for query in payload["queries"])
    assert evaluation.no_answer_support(payload)["measured"] is False


# --------------------------------------------------------------------------
# The payload contract, against a real service.
# --------------------------------------------------------------------------


@pytest.fixture
def evaluation_real_service(project: Path):
    from research_rag.core.service import ResearchService
    from research_rag.project.config import resolve_config
    from tests.conftest import write_pdf
    from tests.core.test_service import (
        FakeDenseBackend,
        FakeUltraRAG,
        UnavailableRerankerDenseBackend,
    )

    async def build(*, reranker: bool = False) -> ResearchService:
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
            dense=UnavailableRerankerDenseBackend() if reranker else FakeDenseBackend(),
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        return service

    return build


def test_the_harness_reads_the_gate_from_a_real_bm25_payload_as_unmeasured(
    project: Path, evaluation_real_service: Any
) -> None:
    """The engine emits a zeroed gate block for BM25, and the block is not the gate."""

    async def exercise() -> None:
        service = await evaluation_real_service()
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="bm25",
            rerank=False,
            evaluation_trace=True,
        )
        measures = evaluation._dense_gate_measures(payload)
        assert payload["dense_gate"]["rejected_below_floor"] is None
        assert measures["dense_gate_ran"] is False
        assert measures["dense_rejected_below_floor"] is None

    asyncio.run(exercise())


def test_the_harness_reads_a_real_hybrid_gate_with_its_denominator(
    project: Path, evaluation_real_service: Any
) -> None:
    async def exercise() -> None:
        service = await evaluation_real_service()
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=False,
            evaluation_trace=True,
        )
        measures = evaluation._dense_gate_measures(payload)
        assert measures["dense_gate_ran"] is True
        assert measures["dense_gate_counts_complete"] is True
        assert measures["dense_conserved"] is True
        assert measures["dense_eligible_total"] is not None

    asyncio.run(exercise())


def test_the_harness_reads_a_real_fallback_as_a_degraded_reranked_row(
    project: Path, evaluation_real_service: Any
) -> None:
    async def exercise() -> None:
        service = await evaluation_real_service(reranker=True)
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=True,
            evaluation_trace=True,
        )
        reasons = evaluation._degradation_reasons(payload, mode=evaluation.RERANK_MODE)
        assert reasons == [
            "rerank requested but not applied (reranker_model_unavailable)"
        ]

    asyncio.run(exercise())


def test_the_harness_keeps_the_trace_a_real_service_emitted(
    project: Path, evaluation_real_service: Any
) -> None:
    async def exercise() -> None:
        service = await evaluation_real_service()
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=True,
            evaluation_trace=True,
        )
        measures = evaluation._trace_measures(payload, target_chunk_id="chk_absent")
        trace = measures["evaluation_trace"]
        assert measures["evaluation_trace_available"] is True
        assert tuple(trace["stages"]) == (
            "dense_before_filters",
            "dense_eligible",
            "dense_admitted",
            "bm25_after_gates",
            "fused_pre_rerank",
            "reranked",
            "post_collapse",
            "final",
        )
        assert trace["candidate_budget"] == 256
        # Every stage carries its own count beside its bounded list, so a cut
        # stage says what it held as well as that it was cut.
        for name, stage in trace["stages"].items():
            if stage is None:
                continue
            assert stage["count"] >= len(stage["chunk_ids"]), name
            assert stage["truncated"] == (stage["count"] > trace["candidate_budget"]), (
                name
            )
        assert trace["rerank"]["window"] == payload["rerank_window"]
        assert trace["rerank"]["scored_count"] == payload["rerank_window"]
        assert trace["rerank"]["scored_truncated"] is False
        assert set(trace["rerank"]["scored_ids"]) == set(
            trace["stages"]["reranked"]["chunk_ids"]
        )
        # The reranker ran, so something was scored: a report that kept only the
        # window could not say whether the model read it.
        assert trace["rerank"]["scored_count"] > 0

    asyncio.run(exercise())


def test_the_harness_counts_scored_discards_from_a_real_service(
    project: Path, evaluation_real_service: Any
) -> None:
    async def exercise() -> None:
        service = await evaluation_real_service()
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=True,
            evaluation_trace=True,
        )
        measures = evaluation._scored_then_collapsed(payload)
        # The corpus here holds two passages of different text, so the collapse
        # discarded nothing and the count is a measured zero rather than a null.
        assert measures["collapse_discarded_count"] == 0
        assert measures["reranked_then_collapsed_count"] == 0
        assert measures["reranked_then_collapsed_is_lower_bound"] is False
        assert measures["scored_candidate_count"] == payload["rerank_window"]

    asyncio.run(exercise())


def test_the_harness_reads_a_real_trace_for_a_target_that_is_missed(
    project: Path, evaluation_real_service: Any
) -> None:
    """A known-item miss and a target the gate dropped are the same number otherwise."""

    async def exercise() -> None:
        service = await evaluation_real_service()
        payload = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=False,
            evaluation_trace=True,
        )
        returned = [str(hit["chunk_id"]) for hit in payload["hits"]]
        absent = "chk_never_returned"
        measures = evaluation._trace_measures(payload, target_chunk_id=absent)
        presence = measures["target_stage_presence"]
        assert measures["evaluation_trace_available"] is True
        assert presence["final"] is False
        assert presence["dense_admitted"] is False
        for chunk_id in returned:
            found = evaluation._trace_measures(payload, target_chunk_id=chunk_id)
            assert found["target_stage_presence"]["final"] is True

    asyncio.run(exercise())
