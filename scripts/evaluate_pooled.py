"""Score pooled author labels against ranked candidate lists.

This harness reads one verified annotation handoff (``author_pool_v1``) and
reports graded retrieval measures for each condition it declares. It is the
answer to the known-item harness's own limit: ``scripts/evaluate_retrieval.py``
names one target passage per query and therefore cannot see a relevant
alternative, so it measures the findability of a designated passage and calls
nothing recall. This harness scores against a pool where every candidate a
condition returned has been judged for relevance and usability, which is what
pool-relative recall needs.

It is a different measurement, not a newer one. The protocol string
``author_pool_v1``, ``measurement_kind: pooled_author_labels`` and
``role: exploratory`` travel with every report so a pooled figure is never read
against a known-item figure. The known-item harness, its metric names and its
definitions are untouched.

    uv run python scripts/evaluate_pooled.py \\
        --input handoff.json --report pooled-report.json \\
        --baseline baseline-arm --bootstrap-samples 2000 --seed 0

What the protocol is, and what it is not
-----------------------------------------

A handoff declares ``schema_version`` 1, the protocol and role strings above,
a ``provenance`` block, the question inventory with a ``family_id`` per
question, the ``judgments``, the pairwise ``relations``, and one entry per
``condition`` holding a ranking per question. Unknown keys are refused in every
block that carries a metric, because a metric field this harness does not read
is a field nobody is measuring. ``provenance`` is the exception: it carries
harness-side detail this script has no opinion about, so it takes unknown keys
and still requires the eight keys it checks. Two of them carry weight here.
``rubric`` travels whole, so a reader can see the grades behind every figure,
and ``rubric_sha256`` is verified against it: the digest is the SHA-256 of the
rubric's canonical JSON, which this scorer can produce on its own, so a rubric
edited after it was hashed is refused rather than read as the acknowledged one.
The other three digests are receipts: this harness holds no pool, no annotation
file and no generation artifact, so it states them and does not vouch for them.
``protected_roots`` is the list of paths a report may never be written into, and
an empty list is refused rather than read as permission everywhere.

A ``condition_id`` is an identifier, not a file name, and it may contain a
slash. Nothing here forms a path from one: the report is the only file this
harness writes, and it is named on the command line.

Labels are current-generation scoped. A judgment keeps its source-relative
path, its locator and its content hash, and a ranking joins on ``chunk_id``;
the chunk id does not claim across rechunking. This harness therefore joins
nothing to a live project, reads no generation artifact, imports no app module,
and scores a handoff as bytes.

A refusal is the normal answer to a malformed handoff. A null, an uncertain,
or a missing label value is refused rather than scored, an unjudged candidate is
refused rather than counted as irrelevant, and no unknown passage is treated as
a negative. A condition that fell back from reranking is refused rather than
scored under the row name it did not earn; a degraded protocol is not
implemented, so nothing is scored that way.

What the numbers mean
---------------------

Grades are ``irrelevant`` 0, ``contextual`` 1, ``direct`` 2, and a gain is
``2 ** grade - 1``. nDCG is graded at the requested depth with its ideal gain
taken from the graded pool of the same question. A pool holding no passage above
grade 0 has an ideal gain of zero, so nDCG is null for that question: it is
unscorable, and it is neither a zero nor a finding about abstention.

``direct_coverage`` is relative to the graded pool. It is not corpus recall and
is not named as one. ``checked_original_direct_usable`` counts returned passages
that are direct, usable, and source-verified; it is telemetry about what was
checked, never a claim that a quotation is ready.

Pair telemetry counts only annotated pairs whose two ends were both returned. A
returned pair carrying no annotation is counted as unjudged and never as
unrelated, because an absent annotation is an absent measurement. Nothing is
grouped transitively, nothing is deduplicated, and no relation is inferred: two
judgments that share a content hash stay distinct while their provenance
differs, so a copy is reported only where an annotator wrote one. Contradiction
is a relation label and never lowers a relevance grade.

Aggregation and comparison
--------------------------

Query means are macro means with an evaluated count beside every figure. Family
means weight each family equally by averaging its queries first, so a family
with more questions cannot outvote one with fewer; that also makes the
questions inside a family non-independent, which the report states.

Each trial is compared to the baseline on its own. Effects are never pooled
across trials: trials sharing one baseline are correlated, and a pooled figure
would both average away two opposed effects and count each family once per
trial. Pairing happens per question before it happens per family, over the
queries both sides scored, so a family is never averaged over one question set
on one side and a different one on the other. The paired and skipped question
ids are published, because a difference over fewer pairs is a difference over
less.

The interval is a cluster bootstrap over families with a fixed seed, conditional
on the observed families. One evaluable family leaves no interval, and fewer
draws than ``BOOTSTRAP_MIN_SAMPLES`` leaves no interval either. The floor is a
floor: at or above it the interval still promises neither stability nor power.

The report carries no p-value, no winner, no recommendation and no policy
acceptance. It reports the budgets each condition actually spent per question,
and where those budgets differ it says a difference is not attributable to
policy alone. It is exploratory evidence, not an equivalence proof.

Writing
-------

The input is opened read-only. The report is created exclusively, with owner
only permissions, and an existing path, a symlink, a path that overlaps the
input, and a path inside a protected root are all refused rather than written.
No project is served, no model is loaded, no control service is contacted, and
nothing outside the report file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1
PROTOCOL = "author_pool_v1"
ROLE = "exploratory"
MEASUREMENT_KIND = "pooled_author_labels"

#: The rubric's three grades as the numbers the gain is built from. Published in
#: every report, because a graded figure is unreadable without its mapping.
RELEVANCE_GRADES = {"irrelevant": 0, "contextual": 1, "direct": 2}
USABILITY_GRADES = ("usable", "needs_context", "unusable")
RELATIONS = (
    "copy",
    "overlap",
    "independent_corroboration",
    "contradiction",
    "related_distinct",
    "unrelated",
)

#: Every measure carried through aggregation and through the paired comparison.
#: A count is not here: telemetry is totalled, never averaged, so a count cannot
#: be read as a rate.
METRICS = (
    "ndcg_at_requested_k",
    "direct_precision_at_requested_k",
    "direct_precision_returned",
    "direct_usable_precision_at_requested_k",
    "direct_usable_precision_returned",
    "direct_coverage",
    "pair_coverage",
)

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_SEED = 0
#: Below this many resamples a percentile is read off too few draws to bear the
#: name 95% interval, so the interval is left null. It is a floor, not a
#: guarantee: at or above it the interval still promises neither stability nor
#: power, which is why the default is an order of magnitude higher.
BOOTSTRAP_MIN_SAMPLES = 200
INTERVAL_LOWER = 0.025
INTERVAL_UPPER = 0.975

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "role",
        "provenance",
        "questions",
        "judgments",
        "relations",
        "conditions",
        "annotation_consistency",
        "label_validation",
    }
)
PROVENANCE_REQUIRED = (
    "pool_id",
    "pool_sha256",
    "annotation_sha256",
    "rubric_sha256",
    "rubric",
    "generation_id",
    "generation_chunks_sha256",
    "protected_roots",
)
QUESTION_KEYS = frozenset({"query_id", "family_id"})
JUDGMENT_KEYS = frozenset(
    {
        "query_id",
        "passage_id",
        "relevance",
        "usability",
        "source_verified",
        "notes",
        "source_relative_path",
        "locator",
        "content_sha256",
        "chunk_id",
    }
)
RELATION_KEYS = frozenset(
    {"query_id", "left_passage_id", "right_passage_id", "relation", "notes"}
)
CONDITION_KEYS = frozenset({"condition_id", "metadata", "rankings"})
RANKING_KEYS = frozenset(
    {
        "query_id",
        "requested_k",
        "passage_ids",
        "candidate_depth",
        "rerank_window",
        "rerank_requested",
        "reranked",
        "rerank_fallback",
    }
)
CONSISTENCY_KEYS = frozenset({"checked_items", "quality_conflicts"})
VALIDATION_KEYS = frozenset({"status", "rubric_acknowledged", "annotator"})

#: Condition metadata this harness reports but never decides. A handoff that
#: carries one of these would state a verdict or a completeness claim the
#: scoring never derived, so it is refused rather than passed through.
FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "accepted",
        "complete",
        "policy_acceptance",
        "p_value",
        "recommendation",
        "verdict",
        "winner",
    }
)

METRIC_DEFINITIONS = {
    "ndcg_at_requested_k": (
        "Graded nDCG at requested_k. Gain is 2 ** grade - 1 with grade "
        "irrelevant 0, contextual 1, direct 2. The ideal gain is taken from the "
        "graded pool of the same question. A pool whose ideal gain is zero is "
        "null, not zero, and is not a finding about abstention."
    ),
    "direct_precision_at_requested_k": (
        "Direct passages in the returned positions up to requested_k, over "
        "requested_k. A list shorter than requested_k keeps requested_k as the "
        "denominator, so a short list is charged for the depth it did not fill."
    ),
    "direct_precision_returned": (
        "Direct passages in the returned positions, over the number of passages "
        "returned. An empty list is null, because the denominator would be zero."
    ),
    "direct_usable_precision_at_requested_k": (
        "Returned passages up to requested_k whose relevance is direct and whose "
        "usability is usable, over requested_k. Needs-context and unusable "
        "directs are counted separately and are never folded into this figure."
    ),
    "direct_usable_precision_returned": (
        "The same count over the number of passages returned. An empty list is null."
    ),
    "direct_coverage": (
        "Direct passages in the returned list over the direct passages in the "
        "graded pool of the same question. A pool holding no direct passage is "
        "null. This is pool relative and is not corpus recall."
    ),
    "pair_coverage": (
        "Annotated pairs whose two ends were both returned, over every unordered "
        "pair in the returned list. A returned pair carrying no annotation is "
        "counted as unjudged, never as unrelated. A list of one or zero pairs is "
        "null."
    ),
}

LIMITS = (
    "nDCG is null, not zero, when the graded pool holds no passage above grade 0. An unscorable question is not a NoAnswer finding.",
    "direct_coverage is relative to the graded pool, not to the corpus. A passage the pool does not cover is not a miss.",
    "checked_original_direct_usable counts checked originals that are direct and usable. It is telemetry, not a claim that a quotation is ready.",
    "Pair telemetry counts only annotated pairs whose two ends were both returned. A pair with no annotation is an absent measurement.",
    "No relation is inferred. Two judgments sharing a content hash stay distinct while their provenance differs, so a copy is reported only where an annotator wrote one.",
    "Contradiction is a relation label. It never lowers a relevance grade and is never scored as an irrelevant passage.",
    "No unique evidence figure is computed and no false suppression claim is made. The returned list is scored as it stands.",
    "Annotated pairs are not grouped transitively, so a pair inside a larger annotated cluster is still counted on its own.",
    "Abstention is unmeasured. The protocol carries no no-answer label, so nothing here estimates whether the corpus fails to answer.",
    "Coverage is the coverage of the pool. Exhaustive corpus coverage is not claimed.",
    "Repeated observations of one question are not independent, so the interval resamples question families rather than observations.",
    "Each trial is compared to the baseline on its own. Effects are never pooled across trials, because trials sharing one baseline are correlated.",
    "A family difference is taken over the questions both sides scored. A family averaged over different question sets on each side is not a difference.",
    "Fewer than two evaluable families leaves no interval, and fewer draws than the stated floor leaves no interval either. The floor promises neither stability nor power.",
    "The interval is a cluster bootstrap over the observed families. It is conditional on them, it is not a population interval, and it is not an equivalence proof.",
    "provenance.rubric_sha256 is verified against provenance.rubric as the SHA-256 of sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False).encode('utf-8')), so a rubric edited after it was hashed is refused rather than read as the acknowledged one.",
    "pool_sha256, annotation_sha256 and generation_chunks_sha256 are receipts this harness cannot check: it holds no pool, no annotation file and no generation artifact to compare them against.",
    "Rubric keys are not contracted here. The rubric travels whole and its digest is verified, so a reader sees the grades behind every figure without this harness holding an opinion on what those grades mean.",
    "Contrasts sharing one baseline are not adjusted for multiplicity.",
    "The report is exploratory. It carries no p-value, no winner, no recommendation and no policy acceptance.",
)

DECISION_STATEMENTS = (
    "No hypothesis test was run, and no acceptance decision was performed.",
    "An interval excluding zero is not a separation: it bounds where the observed families put the difference, and nothing is declared significant.",
    "An interval spanning zero is not an equivalence: no equivalence test was run and none is implied.",
    "The intervals are conditional on the families observed here and are not population statements.",
    "No winner is named. A difference is a difference in the observed families only.",
    "No recommendation is issued and no policy is accepted on this evidence.",
    "Equivalence is neither proven nor disproven.",
    "This report is not policy ready.",
)


class PoolProtocolError(ValueError):
    """The handoff cannot be scored as written, so it is refused.

    Every refusal names one cause. A handoff is refused rather than scored
    partially, because a figure computed over a pool this script could not read
    is a figure whose denominator nobody can state.
    """


# --------------------------------------------------------------------------
# Typed readers. A bool is never read as a number: `True` is a label about
# verification, not a depth, and a harness that reads it as 1 has silently
# invented a budget.
# --------------------------------------------------------------------------


def _refuse(message: str) -> None:
    raise PoolProtocolError(message)


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _refuse(f"{where} must be a JSON object, not {type(value).__name__}")
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _refuse(f"{where} must be a JSON array, not {type(value).__name__}")
    return value


def _require_str(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _refuse(f"{where} must be a string, not {type(value).__name__}")
    if not value and not allow_empty:
        _refuse(f"{where} must not be empty")
    return value


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _refuse(f"{where} must be an integer, not {type(value).__name__}")
    if minimum is not None and value < minimum:
        _refuse(f"{where} must be at least {minimum}, not {value}")
    return value


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _refuse(f"{where} must be true or false, not {type(value).__name__}")
    return value


def _require_enum(value: Any, choices: Any, where: str) -> str:
    text = _require_str(value, where)
    if text not in choices:
        _refuse(f"{where} must be one of {list(choices)}, not {text!r}")
    return text


def _require_sha256(value: Any, where: str) -> str:
    text = _require_str(value, where)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _refuse(f"{where} must be 64 lowercase hexadecimal characters, not {text!r}")
    return text


def _require_keys(
    mapping: dict[str, Any], contract: frozenset[str], where: str
) -> None:
    missing = sorted(contract - set(mapping))
    if missing:
        _refuse(f"{where} is missing {missing}")
    unknown = sorted(set(mapping) - contract)
    if unknown:
        _refuse(f"{where} carries {unknown}, which this contract does not define")


def _require_present_keys(
    mapping: dict[str, Any], required: tuple[str, ...], where: str
) -> None:
    missing = [name for name in required if name not in mapping]
    if missing:
        _refuse(f"{where} is missing {missing}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# --------------------------------------------------------------------------
# Document validation
# --------------------------------------------------------------------------


def validate_document(document: Any) -> None:
    """Refuse anything this harness cannot score as written.

    The structural blocks carry an explicit contract, so an unknown key in one
    of them is refused: a metric field the scorer does not read is a field
    nobody measured. ``provenance`` takes unknown keys, because it is the
    annotator's and the toolkit's block rather than this script's.
    """

    _require_mapping(document, "the handoff")
    _require_keys(document, TOP_LEVEL_KEYS, "the handoff")
    _check_header(document)
    _check_provenance(document["provenance"])
    questions = _check_questions(document["questions"])
    judgments = _check_judgments(
        document["judgments"], {q["query_id"] for q in questions}
    )
    _check_relations(document["relations"], judgments)
    _check_annotation_quality(document)
    _check_conditions(document["conditions"], questions, judgments)


def _check_header(document: dict[str, Any]) -> None:
    version = _require_int(document["schema_version"], "schema_version")
    if version != REPORT_SCHEMA_VERSION:
        _refuse(f"schema_version must be {REPORT_SCHEMA_VERSION}, not {version}")
    protocol = _require_enum(document["protocol"], (PROTOCOL,), "protocol")
    if protocol != PROTOCOL:
        _refuse(f"protocol must be {PROTOCOL!r}, not {protocol!r}")
    role = _require_enum(document["role"], (ROLE,), "role")
    if role != ROLE:
        _refuse(f"role must be {ROLE!r}, not {role!r}")


def _check_provenance(value: Any) -> None:
    provenance = _require_mapping(value, "provenance")
    _require_present_keys(provenance, PROVENANCE_REQUIRED, "provenance")
    _require_str(provenance["pool_id"], "provenance.pool_id")
    _require_str(provenance["generation_id"], "provenance.generation_id")
    for name in (
        "pool_sha256",
        "annotation_sha256",
        "rubric_sha256",
        "generation_chunks_sha256",
    ):
        _require_sha256(provenance[name], f"provenance.{name}")
    _check_rubric_digest(provenance)
    _check_protected_roots(provenance["protected_roots"])


def _check_rubric_digest(provenance: dict[str, Any]) -> None:
    """The rubric travels whole, so its digest is checked rather than trusted.

    The digest is the protocol's canonical JSON of the rubric object. That makes
    it verifiable here without the toolkit: this scorer can serialize what it
    was handed, so a rubric edited after it was hashed is refused rather than
    read as the rubric the annotator acknowledged.
    """

    rubric = _require_mapping(provenance["rubric"], "provenance.rubric")
    if not rubric:
        _refuse(
            "provenance.rubric is empty; a named rubric must travel with its digest"
        )
    _reject_non_finite(rubric, "provenance.rubric")
    digest = _canonical_digest(rubric, "provenance.rubric")
    declared = provenance["rubric_sha256"]
    if digest != declared:
        _refuse(
            f"provenance.rubric_sha256 is {declared}, but provenance.rubric serializes to "
            f"{digest}; the rubric does not match the digest it arrived with"
        )


def _reject_non_finite(value: Any, where: str) -> None:
    """A rubric cannot carry a NaN or an infinity.

    ``json.loads`` accepts both tokens, so a hand-edited handoff can smuggle one
    past a parse and into a figure's provenance. The canonical serializer refuses
    them too; this pass exists so the refusal names the offending value instead
    of a JSON compliance rule.
    """

    if isinstance(value, float) and not math.isfinite(value):
        _refuse(
            f"{where} holds {value}; a rubric value is a grade, a name or a bound, and none of "
            f"those is a NaN or an infinity"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{where}[{index}]")


def _canonical_digest(value: Any, where: str) -> str:
    """The protocol's canonical JSON of an object, hashed as UTF-8.

    Keys sorted, no insignificant whitespace, and no ASCII escaping, so the same
    rubric always produces the same digest on any machine and any locale. A
    non-finite float or a value JSON cannot carry is refused here as well, so a
    caller that skipped the explicit pass still cannot get a digest out.
    """

    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _refuse(f"{where} has no canonical JSON form: {error}")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_protected_roots(value: Any) -> None:
    """The roots a report may not be written into.

    An empty list is refused rather than read as permission everywhere: a gate
    that silently protects nothing is worse than one that refuses a run whose
    roots were lost.
    """

    roots = _require_list(value, "provenance.protected_roots")
    if not roots:
        _refuse(
            "provenance.protected_roots is empty; a report must not be writable into a corpus or "
            "a project, so the roots to keep it out of have to be declared"
        )
    for index, entry in enumerate(roots):
        _require_str(entry, f"provenance.protected_roots[{index}]")


def _check_questions(value: Any) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(_require_list(value, "questions")):
        where = f"questions[{index}]"
        item = _require_mapping(entry, where)
        _require_keys(item, QUESTION_KEYS, where)
        query_id = _require_str(item["query_id"], f"{where}.query_id")
        family_id = _require_str(item["family_id"], f"{where}.family_id")
        if query_id in seen:
            _refuse(
                f"{where}.query_id {query_id!r} is declared twice; a question has one family"
            )
        seen.add(query_id)
        questions.append({"query_id": query_id, "family_id": family_id})
    if not questions:
        _refuse("questions is empty; there is nothing to score")
    return questions


def _check_judgments(
    value: Any, declared: set[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(_require_list(value, "judgments")):
        where = f"judgments[{index}]"
        item = _require_mapping(entry, where)
        _require_keys(item, JUDGMENT_KEYS, where)
        query_id = _require_str(item["query_id"], f"{where}.query_id")
        if query_id not in declared:
            _refuse(f"{where}.query_id {query_id!r} is not a declared question")
        passage_id = _require_str(item["passage_id"], f"{where}.passage_id")
        key = (query_id, passage_id)
        if key in judgments:
            _refuse(
                f"{where} judges {query_id!r}/{passage_id!r} twice; one pair has one label"
            )
        _require_enum(item["relevance"], RELEVANCE_GRADES, f"{where}.relevance")
        _require_enum(item["usability"], USABILITY_GRADES, f"{where}.usability")
        _require_bool(item["source_verified"], f"{where}.source_verified")
        _require_str(item["notes"], f"{where}.notes", allow_empty=True)
        _require_str(item["source_relative_path"], f"{where}.source_relative_path")
        _require_mapping(item["locator"], f"{where}.locator")
        _require_sha256(item["content_sha256"], f"{where}.content_sha256")
        _require_str(item["chunk_id"], f"{where}.chunk_id")
        judgments[key] = item
    return judgments


def _check_relations(
    value: Any, judgments: dict[tuple[str, str], dict[str, Any]]
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(_require_list(value, "relations")):
        where = f"relations[{index}]"
        item = _require_mapping(entry, where)
        _require_keys(item, RELATION_KEYS, where)
        query_id = _require_str(item["query_id"], f"{where}.query_id")
        left = _require_str(item["left_passage_id"], f"{where}.left_passage_id")
        right = _require_str(item["right_passage_id"], f"{where}.right_passage_id")
        _require_enum(item["relation"], RELATIONS, f"{where}.relation")
        _require_str(item["notes"], f"{where}.notes", allow_empty=True)
        if left == right:
            _refuse(
                f"{where} relates {left!r} to itself; a passage is not its own pair"
            )
        for passage_id in (left, right):
            _check_relation_endpoint(query_id, passage_id, judgments, where)
        low, high = sorted((left, right))
        key = (query_id, low, high)
        if key in seen:
            _refuse(
                f"{where} repeats the relation between {low!r} and {high!r} for {query_id!r}"
            )
        seen.add(key)


def _check_relation_endpoint(
    query_id: str,
    passage_id: str,
    judgments: dict[tuple[str, str], dict[str, Any]],
    where: str,
) -> None:
    if (query_id, passage_id) in judgments:
        return
    elsewhere = sorted(
        other
        for other, judged in judgments.items()
        if judged["passage_id"] == passage_id
    )
    if elsewhere:
        other_query = elsewhere[0][0]
        _refuse(
            f"{where} relates {passage_id!r} within {query_id!r}, but that passage is "
            f"judged under {other_query!r}; a relation cannot relate passages across "
            f"questions"
        )
    _refuse(f"{where} names {passage_id!r}, which no judgment for {query_id!r} covers")


def _check_annotation_quality(document: dict[str, Any]) -> None:
    consistency = _require_mapping(
        document["annotation_consistency"], "annotation_consistency"
    )
    _require_keys(consistency, CONSISTENCY_KEYS, "annotation_consistency")
    _require_int(
        consistency["checked_items"], "annotation_consistency.checked_items", minimum=0
    )
    _require_list(
        consistency["quality_conflicts"], "annotation_consistency.quality_conflicts"
    )

    validation = _require_mapping(document["label_validation"], "label_validation")
    _require_keys(validation, VALIDATION_KEYS, "label_validation")
    status = _require_enum(
        validation["status"], ("complete",), "label_validation.status"
    )
    if status != "complete":
        _refuse(f"label_validation.status must be 'complete', not {status!r}")
    acknowledged = _require_bool(
        validation["rubric_acknowledged"], "label_validation.rubric_acknowledged"
    )
    if not acknowledged:
        _refuse(
            "label_validation.rubric_acknowledged must be true; an unacknowledged rubric is not a label"
        )
    _require_str(validation["annotator"], "label_validation.annotator")


def _check_conditions(
    value: Any,
    questions: list[dict[str, str]],
    judgments: dict[tuple[str, str], dict[str, Any]],
) -> None:
    inventory = {question["query_id"] for question in questions}
    declared: set[str] = set()
    for index, entry in enumerate(_require_list(value, "conditions")):
        where = f"conditions[{index}]"
        item = _require_mapping(entry, where)
        _require_keys(item, CONDITION_KEYS, where)
        condition_id = _require_str(item["condition_id"], f"{where}.condition_id")
        if condition_id in declared:
            _refuse(f"{where}.condition_id {condition_id!r} is declared twice")
        declared.add(condition_id)
        _check_metadata(item["metadata"], f"{where}.metadata")
        _check_rankings(
            item["rankings"], f"{where}.rankings", condition_id, inventory, judgments
        )
    if not declared:
        _refuse("conditions is empty; there is nothing to score")


def _check_metadata(value: Any, where: str) -> None:
    metadata = _require_mapping(value, where)
    intruders = sorted(set(metadata) & FORBIDDEN_METADATA_KEYS)
    if intruders:
        _refuse(
            f"{where} carries {intruders}; a verdict or completeness claim is the scorer's to state"
        )


def _check_rankings(
    value: Any,
    where: str,
    condition_id: str,
    inventory: set[str],
    judgments: dict[tuple[str, str], dict[str, Any]],
) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(_require_list(value, where)):
        row_where = f"{where}[{index}]"
        item = _require_mapping(entry, row_where)
        _require_keys(item, RANKING_KEYS, row_where)
        query_id = _require_str(item["query_id"], f"{row_where}.query_id")
        if query_id not in inventory:
            _refuse(f"{row_where}.query_id {query_id!r} is not a declared question")
        if query_id in seen:
            _refuse(
                f"{condition_id!r} ranks {query_id!r} twice; a condition has one ranking per question"
            )
        seen.add(query_id)
        _check_ranking_row(item, row_where, condition_id, query_id, judgments)
    absent = sorted(inventory - seen)
    if absent:
        _refuse(
            f"{condition_id!r} has no ranking for {absent}; conditions share one question inventory"
        )


def _check_ranking_row(
    row: dict[str, Any],
    where: str,
    condition_id: str,
    query_id: str,
    judgments: dict[tuple[str, str], dict[str, Any]],
) -> None:
    _require_int(row["requested_k"], f"{where}.requested_k", minimum=1)
    _require_int(row["candidate_depth"], f"{where}.candidate_depth", minimum=1)
    rerank_window = _require_int(
        row["rerank_window"], f"{where}.rerank_window", minimum=0
    )
    requested = _require_bool(row["rerank_requested"], f"{where}.rerank_requested")
    reranked = _require_bool(row["reranked"], f"{where}.reranked")
    fallback = row["rerank_fallback"]
    if fallback is not None and not isinstance(fallback, str):
        _refuse(
            f"{where}.rerank_fallback must be null or a string, not {type(fallback).__name__}"
        )
    if fallback is not None:
        _refuse(
            f"{where} records rerank fallback {fallback!r} for {query_id!r}; a fallback row is "
            f"not the row its name claims and no degraded protocol is implemented"
        )
    if requested and not reranked:
        _refuse(
            f"{where} asked for reranking and did not get it for {query_id!r}; an unranked "
            f"candidate order is not scored under a reranked row"
        )
    if not requested and reranked:
        _refuse(
            f"{where} reranked without asking for it; the row does not describe what ran"
        )
    if requested and rerank_window < 1:
        _refuse(f"{where}.rerank_window must be positive when reranking was requested")
    _check_passage_ids(row["passage_ids"], f"{where}.passage_ids", query_id, judgments)


def _check_passage_ids(
    value: Any,
    where: str,
    query_id: str,
    judgments: dict[tuple[str, str], dict[str, Any]],
) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(_require_list(value, where)):
        passage_id = _require_str(entry, f"{where}[{index}]")
        if passage_id in seen:
            _refuse(f"{where} returns {passage_id!r} at two ranks for {query_id!r}")
        if (query_id, passage_id) not in judgments:
            _refuse(
                f"{where} returns {passage_id!r}, which no judgment for {query_id!r} covers; an "
                f"unjudged candidate is not scored as irrelevant"
            )
        seen.add(passage_id)


def load_handoff(source: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one handoff read-only and validate it before it is scored.

    ``source`` is a path, or ``-`` for standard input. The file is opened for
    reading and never written, renamed or moved. JSON that does not parse, and
    a document that does not satisfy the protocol, are both refused by cause.
    """

    if str(source) == "-":
        raw = sys.stdin.read()
        origin = "<stdin>"
    else:
        path = Path(source)
        if path.is_dir():
            _refuse(f"{path} is a directory, not a handoff file")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as error:
            _refuse(f"{path} cannot be read: {error.strerror or error}")
        origin = str(path)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        _refuse(f"{origin} is not JSON: {error}")
    validate_document(document)
    return document


# --------------------------------------------------------------------------
# Normalization. Validation has already refused every malformed value, so what
# follows reads typed values without re-checking them.
# --------------------------------------------------------------------------


def _normalize(handoff: dict[str, Any]) -> dict[str, Any]:
    questions = [
        {
            "query_id": item["query_id"],
            "family_id": item["family_id"],
        }
        for item in handoff["questions"]
    ]
    pools = _build_pools(handoff["judgments"])
    relations = _build_relations(handoff["relations"])
    conditions = []
    for condition in handoff["conditions"]:
        rankings = {row["query_id"]: dict(row) for row in condition["rankings"]}
        conditions.append(
            {
                "condition_id": condition["condition_id"],
                "metadata": dict(condition["metadata"]),
                "rankings": rankings,
            }
        )
    return {
        "questions": questions,
        "pools": pools,
        "relations": relations,
        "conditions": conditions,
    }


def _build_pools(
    judgments: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    pools: dict[str, dict[str, dict[str, Any]]] = {}
    for item in judgments:
        pool = pools.setdefault(
            item["query_id"],
            {"judgments": {}, "grades": {}, "usability": {}, "verified": {}},
        )
        pool["judgments"][item["passage_id"]] = item
        pool["grades"][item["passage_id"]] = RELEVANCE_GRADES[item["relevance"]]
        pool["usability"][item["passage_id"]] = item["usability"]
        pool["verified"][item["passage_id"]] = item["source_verified"]
    return pools


def _build_relations(
    relations: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], str]]:
    built: dict[str, dict[tuple[str, str], str]] = {}
    for item in relations:
        low, high = sorted((item["left_passage_id"], item["right_passage_id"]))
        built.setdefault(item["query_id"], {})[(low, high)] = item["relation"]
    return built


# --------------------------------------------------------------------------
# Per-question scoring
# --------------------------------------------------------------------------


def _gain(grade: int) -> float:
    return 2.0**grade - 1.0


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _graded_ndcg(
    returned_grades: list[int], pool_grades: list[int], cutoff: int
) -> float | None:
    """Graded nDCG at ``cutoff`` against the graded pool of the same question."""

    ideal = sorted((_gain(grade) for grade in pool_grades), reverse=True)[:cutoff]
    best = _dcg(ideal)
    if best == 0:
        return None
    return _dcg([_gain(grade) for grade in returned_grades[:cutoff]]) / best


def _usability_split(
    passages: list[str], pool: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts = {grade: 0 for grade in USABILITY_GRADES}
    for passage_id in passages:
        if pool["grades"].get(passage_id) == RELEVANCE_GRADES["direct"]:
            counts[pool["usability"][passage_id]] += 1
    return counts


def _is_direct(pool: dict[str, dict[str, Any]], passage_id: str) -> bool:
    return pool["grades"].get(passage_id) == RELEVANCE_GRADES["direct"]


def _is_direct_usable(pool: dict[str, dict[str, Any]], passage_id: str) -> bool:
    return (
        _is_direct(pool, passage_id) and pool["usability"].get(passage_id) == "usable"
    )


def _pair_telemetry(
    returned: list[str],
    annotated: dict[tuple[str, str], str],
    pool: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Count only annotated pairs both of whose ends were returned."""

    total = len(returned) * (len(returned) - 1) // 2
    counts = {relation: 0 for relation in RELATIONS}
    known = 0
    identical_content = 0
    for index, left in enumerate(returned):
        for right in returned[index + 1 :]:
            relation = annotated.get((left, right)) or annotated.get((right, left))
            if relation is not None:
                known += 1
                counts[relation] += 1
            if _shares_text_while_provenance_differs(pool, left, right):
                identical_content += 1
    return {
        "returned_pairs_total": total,
        "annotated_pairs_total": len(annotated),
        "known_returned_pairs": known,
        "unjudged_pair_count": total - known,
        "pair_coverage": _ratio(known, total),
        "relations": counts,
        "contradiction_returned_pairs": counts["contradiction"],
        "identical_content_distinct_passages_returned_pairs": identical_content,
        "automatic_relation_inference": False,
        "transitive_grouping": False,
    }


def _shares_text_while_provenance_differs(
    pool: dict[str, dict[str, Any]],
    left: str,
    right: str,
) -> bool:
    left_judgment = pool["judgments"].get(left)
    right_judgment = pool["judgments"].get(right)
    if left_judgment is None or right_judgment is None:
        return False
    if left_judgment["content_sha256"] != right_judgment["content_sha256"]:
        return False
    left_provenance = (
        left_judgment["source_relative_path"],
        left_judgment["locator"],
        left_judgment["chunk_id"],
    )
    right_provenance = (
        right_judgment["source_relative_path"],
        right_judgment["locator"],
        right_judgment["chunk_id"],
    )
    return left_provenance != right_provenance


def _evaluate_question(
    question: dict[str, str],
    ranking: dict[str, Any],
    condition_id: str,
    pool: dict[str, dict[str, Any]],
    annotated: dict[tuple[str, str], str],
) -> dict[str, Any]:
    returned = list(ranking["passage_ids"])
    requested_k = ranking["requested_k"]
    window = returned[:requested_k]
    direct_pool = [
        passage_id
        for passage_id, grade in pool["grades"].items()
        if grade == RELEVANCE_GRADES["direct"]
    ]
    pairs = _pair_telemetry(returned, annotated, pool)
    return {
        "query_id": question["query_id"],
        "family_id": question["family_id"],
        "condition_id": condition_id,
        "requested_k": requested_k,
        "candidate_depth": ranking["candidate_depth"],
        "rerank_window": ranking["rerank_window"],
        "rerank_requested": ranking["rerank_requested"],
        "reranked": ranking["reranked"],
        "returned_count": len(returned),
        "graded_pool_size": len(pool["grades"]),
        "direct_pool_size": len(direct_pool),
        "direct_usable_pool_size": sum(
            1 for passage_id in direct_pool if _is_direct_usable(pool, passage_id)
        ),
        "ndcg_at_requested_k": _graded_ndcg(
            [pool["grades"][passage_id] for passage_id in returned],
            list(pool["grades"].values()),
            requested_k,
        ),
        "direct_precision_at_requested_k": sum(
            1 for passage_id in window if _is_direct(pool, passage_id)
        )
        / requested_k,
        "direct_precision_returned": _ratio(
            sum(1 for passage_id in returned if _is_direct(pool, passage_id)),
            len(returned),
        ),
        "direct_usable_precision_at_requested_k": sum(
            1 for passage_id in window if _is_direct_usable(pool, passage_id)
        )
        / requested_k,
        "direct_usable_precision_returned": _ratio(
            sum(1 for passage_id in returned if _is_direct_usable(pool, passage_id)),
            len(returned),
        ),
        "direct_coverage": _ratio(
            sum(1 for passage_id in returned if _is_direct(pool, passage_id)),
            len(direct_pool),
        ),
        "checked_original_direct_usable": sum(
            1
            for passage_id in returned
            if _is_direct_usable(pool, passage_id) and pool["verified"].get(passage_id)
        ),
        "direct_at_k_grade_usability": _usability_split(window, pool),
        "direct_returned_grade_usability": _usability_split(returned, pool),
        "pool_direct_grade_usability": _usability_split(direct_pool, pool),
        # Hoisted beside the other measures so one list of metric names drives
        # the query means, the family means and the paired comparison alike.
        "pair_coverage": pairs["pair_coverage"],
        "pairs": pairs,
    }


# --------------------------------------------------------------------------
# Aggregation. Every figure carries the count it was computed over, and a
# measure with nothing to average is null rather than zero.
# --------------------------------------------------------------------------


def _query_means(queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = [query[metric] for query in queries if query[metric] is not None]
        unevaluable = [query["query_id"] for query in queries if query[metric] is None]
        aggregates[metric] = {
            "query_mean": {
                "value": _mean(values),
                "evaluated": len(values),
                "total": len(queries),
                "unevaluable_query_ids": unevaluable,
            }
        }
    return aggregates


def _family_means(queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        families.setdefault(query["family_id"], []).append(query)
    return families


def _aggregate_condition(queries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates = _query_means(queries)
    families = _family_means(queries)
    for metric in METRICS:
        per_family: dict[str, Any] = {}
        for family_id, members in sorted(families.items()):
            values = [query[metric] for query in members if query[metric] is not None]
            per_family[family_id] = {
                "value": _mean(values),
                "queries_evaluated": len(values),
                "queries_total": len(members),
            }
        evaluated = [
            entry["value"]
            for entry in per_family.values()
            if entry["value"] is not None
        ]
        aggregates[metric]["family_mean"] = {
            "value": _mean(evaluated),
            "families_evaluated": len(evaluated),
            "families_total": len(per_family),
            "families_unevaluable": sorted(
                family_id
                for family_id, entry in per_family.items()
                if entry["value"] is None
            ),
        }
        aggregates[metric]["per_family"] = per_family
    return aggregates


def _condition_counts(queries: list[dict[str, Any]]) -> dict[str, Any]:
    pair_counts = [query["pairs"] for query in queries]
    return {
        "queries": len(queries),
        "families": len(_family_means(queries)),
        "returned_passages": sum(query["returned_count"] for query in queries),
        "queries_without_pooled_labels": sum(
            1 for query in queries if query["graded_pool_size"] == 0
        ),
        "checked_original_direct_usable": sum(
            query["checked_original_direct_usable"] for query in queries
        ),
        "returned_pairs_total": sum(
            counts["returned_pairs_total"] for counts in pair_counts
        ),
        "known_returned_pairs": sum(
            counts["known_returned_pairs"] for counts in pair_counts
        ),
        "unjudged_pair_count": sum(
            counts["unjudged_pair_count"] for counts in pair_counts
        ),
        "contradiction_returned_pairs": sum(
            counts["contradiction_returned_pairs"] for counts in pair_counts
        ),
    }


# --------------------------------------------------------------------------
# Budgets. What a condition actually spent, compared question by question.
# --------------------------------------------------------------------------


def _budget_equality(normalized: dict[str, Any]) -> dict[str, Any]:
    """Compare the work each condition spent, question by question.

    ``returned_count`` is an outcome, not a budget: a condition that returned
    fewer passages did not search less, so it is reported beside the comparison
    and kept out of it. Reading it as a work difference would let a result size
    masquerade as a retrieval limit.
    """

    conditions = normalized["conditions"]
    per_query: dict[str, dict[str, Any]] = {}
    returned_counts: dict[str, dict[str, int]] = {}
    unequal: list[str] = []
    for question in normalized["questions"]:
        query_id = question["query_id"]
        spent = {
            condition["condition_id"]: {
                "requested_k": condition["rankings"][query_id]["requested_k"],
                "candidate_depth": condition["rankings"][query_id]["candidate_depth"],
                "rerank_window": condition["rankings"][query_id]["rerank_window"],
            }
            for condition in conditions
        }
        per_query[query_id] = spent
        returned_counts[query_id] = {
            condition["condition_id"]: len(
                condition["rankings"][query_id]["passage_ids"]
            )
            for condition in conditions
        }
        if len({tuple(sorted(entry.items())) for entry in spent.values()}) > 1:
            unequal.append(query_id)
    statement = (
        "Every condition spent the same requested depth, candidate depth and rerank window on "
        "every question, so a difference between conditions is not a difference of budget. "
        "returned_count is an outcome and is reported separately; it is not read as a budget."
        if not unequal
        else "Conditions spent different requested depth, candidate depth or rerank window on at "
        "least one question, so a difference between them is not attributable to policy alone. "
        "returned_count is an outcome and is reported separately; it is not read as a budget."
    )
    return {
        "keys": ["requested_k", "candidate_depth", "rerank_window"],
        "equal_for_every_query": not unequal,
        "unequal_query_ids": unequal,
        "per_query": per_query,
        "returned_counts": returned_counts,
        "returned_count_is": "outcome telemetry, not a work budget",
        "statement": statement,
    }


# --------------------------------------------------------------------------
# Paired comparison against a baseline
# --------------------------------------------------------------------------


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    span = position - lower
    return values[lower] + (values[upper] - values[lower]) * span


def _cluster_bootstrap(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    """Percentile interval from resampling whole families with replacement.

    Families are the clusters: questions inside one family are not independent
    observations, so a bootstrap over questions would treat a family with more
    questions as more evidence than it is.

    Two things leave the interval null rather than narrow. Fewer than two
    evaluable families gives one cluster, which has no spread to resample. And
    fewer than ``BOOTSTRAP_MIN_SAMPLES`` draws gives percentiles read off too few
    resamples to bear the name: one draw is a point, not a 95% interval. The
    floor is a floor and no more, so a draw count at or above it still promises
    neither stability nor power.
    """

    if len(values) < 2:
        return _unusable_interval(
            samples,
            0,
            "fewer than two evaluable families; one cluster has no spread to resample",
        )
    if samples < BOOTSTRAP_MIN_SAMPLES:
        return _unusable_interval(
            samples,
            0,
            f"{samples} draws is below the {BOOTSTRAP_MIN_SAMPLES}-draw floor; a percentile read "
            f"off this few resamples is not a 95% interval, and the floor promises neither "
            f"stability nor power",
        )
    generator = random.Random(seed)
    size = len(values)
    draws = sorted(
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return {
        "low": _percentile(draws, INTERVAL_LOWER),
        "high": _percentile(draws, INTERVAL_UPPER),
        "draws": samples,
        "draws_requested": samples,
        "usable": True,
        "reason": None,
    }


def _unusable_interval(samples: int, draws: int, reason: str) -> dict[str, Any]:
    return {
        "low": None,
        "high": None,
        "draws": draws,
        "draws_requested": samples,
        "usable": False,
        "reason": reason,
    }


def _paired_families(
    baseline: dict[str, Any],
    trial: dict[str, Any],
    metric: str,
) -> dict[str, dict[str, Any]]:
    """Per-family paired difference over the questions *both* sides scored.

    Pairing happens per query before it happens per family. A family mean taken
    over one query on the baseline and three on the trial is two different
    populations subtracted, so the intersection of the evaluable queries is taken
    first and both sides are averaged over that same set.
    """

    trial_values = {query["query_id"]: query[metric] for query in trial["queries"]}
    paired: dict[str, list[tuple[str, float]]] = {}
    skipped: dict[str, list[str]] = {}
    for query in baseline["queries"]:
        query_id = query["query_id"]
        family_id = query["family_id"]
        baseline_value = query[metric]
        trial_value = trial_values.get(query_id)
        if baseline_value is None or trial_value is None:
            skipped.setdefault(family_id, []).append(query_id)
            continue
        paired.setdefault(family_id, []).append(
            (query_id, trial_value - baseline_value)
        )
    families: dict[str, dict[str, Any]] = {}
    for family_id in sorted(set(paired) | set(skipped)):
        entries = paired.get(family_id, [])
        deltas = [delta for _, delta in entries]
        families[family_id] = {
            "difference": _mean(deltas),
            "paired_query_ids": [query_id for query_id, _ in entries],
            "paired_queries": len(entries),
            "skipped_query_ids": sorted(skipped.get(family_id, [])),
            "skipped_queries": len(skipped.get(family_id, [])),
        }
    return families


def _paired_overview(
    families: dict[str, dict[str, Any]], questions_total: int
) -> dict[str, Any]:
    """The trial-level summary, naming the paired and the skipped questions."""

    paired_ids = sorted(
        query_id
        for entry in families.values()
        for query_id in entry["paired_query_ids"]
    )
    skipped_ids = sorted(
        query_id
        for entry in families.values()
        for query_id in entry["skipped_query_ids"]
    )
    differences = [
        entry["difference"]
        for entry in families.values()
        if entry["difference"] is not None
    ]
    return {
        "family_weighted_difference": _mean(differences),
        "families_evaluated": len(differences),
        "families_total": len(families),
        "queries_paired": len(paired_ids),
        "queries_total": questions_total,
        "queries_skipped": len(skipped_ids),
        "paired_query_ids": paired_ids,
        "skipped_query_ids": skipped_ids,
    }


def _compare(
    conditions: list[dict[str, Any]],
    baseline_name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Every trial is compared to the baseline on its own.

    Effects are never pooled across trials. A difference of plus one against a
    baseline and minus one against that same baseline average to nothing, and a
    pooled figure would report that as a null finding while also counting each
    family once per trial as though the trials were independent evidence.
    """

    names = [condition["condition_id"] for condition in conditions]
    if baseline_name is None:
        baseline_name = names[0]
    if baseline_name not in names:
        _refuse(
            f"--baseline {baseline_name!r} is not a declared condition; declared: {names}"
        )
    if len(conditions) < 2:
        return {
            "status": "not_available",
            "reason": "fewer than two conditions; a comparison needs a baseline and a trial",
            "baseline_condition": baseline_name,
            "conditions_compared": [],
            "per_metric": {},
        }
    baseline = next(
        item for item in conditions if item["condition_id"] == baseline_name
    )
    trials = [item for item in conditions if item["condition_id"] != baseline_name]
    questions_total = len(baseline["queries"])
    per_metric: dict[str, Any] = {}
    for metric in METRICS:
        per_condition: dict[str, Any] = {}
        for trial in trials:
            families = _paired_families(baseline, trial, metric)
            differences = sorted(
                entry["difference"]
                for entry in families.values()
                if entry["difference"] is not None
            )
            per_condition[trial["condition_id"]] = {
                **_paired_overview(families, questions_total),
                "interval_95": _cluster_bootstrap(differences, bootstrap_samples, seed),
                "per_family": families,
            }
        per_metric[metric] = {
            "baseline_condition": baseline_name,
            "conditions": per_condition,
            "pooled_across_conditions": False,
        }
    return {
        "status": "exploratory",
        "baseline_condition": baseline_name,
        "conditions_compared": [trial["condition_id"] for trial in trials],
        "pairing": (
            "per query over the questions both sides scored, then per family over that same "
            "intersected set; a family is never averaged over one question set on one side and "
            "a different one on the other"
        ),
        "independence": (
            "Each trial is compared to the baseline on its own. Effects are never pooled across "
            "trials, because trials sharing one baseline are correlated and a pooled figure "
            "would count each family once per trial."
        ),
        "dependence": (
            "Questions inside a family share a source and a target family, so they are not "
            "independent observations. Each family contributes one value and the families are "
            "resampled as clusters."
        ),
        "multiple_comparisons": "not adjusted",
        "interval": (
            f"per trial condition independently: cluster bootstrap over question families, "
            f"{bootstrap_samples} draws, seed {seed}, conditional on the observed families"
        ),
        "per_metric": per_metric,
        "interpretation": [
            "Read one trial at a time against the baseline. No figure is averaged across trials.",
            "The interval is conditional on the families observed here. It is not a population interval.",
            "An interval spanning zero is not an equivalence test; no equivalence test was run.",
            "An interval excluding zero is not a separation test; no hypothesis test was run.",
            "Contrasts sharing one baseline are exploratory and not adjusted for multiplicity.",
            "queries_skipped names the questions one side could not score, so a difference over a smaller paired set is visible rather than silent.",
            "Where budgets differ per question, read the difference against budget_equality before reading it at all.",
        ],
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(
    handoff: dict[str, Any],
    baseline: str | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score a validated handoff and return the report payload.

    The document is validated again here, so a caller holding a dict it built
    itself gets the same refusal a file would. Nothing is read outside the
    handoff: no project, no generation artifact, no model, no service.
    """

    validate_document(handoff)
    if bootstrap_samples < 1:
        _refuse(f"bootstrap_samples must be at least 1, not {bootstrap_samples}")
    normalized = _normalize(handoff)
    conditions = []
    for condition in normalized["conditions"]:
        queries = [
            _evaluate_question(
                question,
                condition["rankings"][question["query_id"]],
                condition["condition_id"],
                normalized["pools"].get(question["query_id"], _empty_pool()),
                normalized["relations"].get(question["query_id"], {}),
            )
            for question in normalized["questions"]
        ]
        scored = {
            "condition_id": condition["condition_id"],
            "metadata": condition["metadata"],
            "queries": queries,
            "counts": _condition_counts(queries),
        }
        scored["aggregates"] = _aggregate_condition(queries)
        conditions.append(scored)
    return {
        "report_meta": _report_meta(handoff, bootstrap_samples, seed),
        "relevance_grades": dict(RELEVANCE_GRADES),
        "gain": "2 ** grade - 1",
        "judgments_are": "current-generation scoped; chunk ids join rankings and claim nothing across rechunking",
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "conditions": conditions,
        "budget_equality": _budget_equality(normalized),
        "comparison": _compare(conditions, baseline, bootstrap_samples, seed),
        "no_answer": {
            "measured": False,
            "statement": (
                "The protocol carries no abstention label. Whether the corpus fails to answer a "
                "question is unmeasured, and no figure here estimates it."
            ),
        },
        "decision": {
            "p_value": None,
            "winner": None,
            "recommendation": None,
            "policy_acceptance": None,
            "equivalence_proven": False,
            "policy_ready": False,
            "statements": list(DECISION_STATEMENTS),
        },
        "limits": list(LIMITS),
    }


def _empty_pool() -> dict[str, dict[str, Any]]:
    return {"judgments": {}, "grades": {}, "usability": {}, "verified": {}}


def _report_meta(
    handoff: dict[str, Any], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    provenance = handoff["provenance"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "harness": "scripts/evaluate_pooled.py",
        "measurement_kind": MEASUREMENT_KIND,
        "protocol": PROTOCOL,
        "role": ROLE,
        "not_comparable_with": [
            "known-item measures",
            "success@k, MRR, and known-item nDCG from scripts/evaluate_retrieval.py",
        ],
        "provenance": dict(provenance),
        "rubric_verification": {
            "rubric_sha256": provenance["rubric_sha256"],
            "verified": True,
            "canonical_json": (
                "sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False, "
                "separators=(',',':'), allow_nan=False).encode('utf-8'))"
            ),
        },
        "protected_roots_checked": [
            str(root)
            for root in _resolved_protected_roots(provenance["protected_roots"])
        ],
        "annotation_consistency": dict(handoff["annotation_consistency"]),
        "label_validation": dict(handoff["label_validation"]),
        "bootstrap": {
            "method": "cluster_bootstrap_over_families",
            "samples": bootstrap_samples,
            "seed": seed,
            "interval": "percentile_95",
            "conditional_on": "observed families",
        },
    }


# --------------------------------------------------------------------------
# Command line and the exclusive report write
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="evaluate_pooled.py",
        description=(
            "Score pooled author labels against ranked candidate lists. Reads one verified "
            f"{PROTOCOL} handoff and writes one exploratory report."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        metavar="HANDOFF.json",
        help="The handoff to score, or - for standard input. Opened read-only.",
    )
    parser.add_argument(
        "--report",
        required=True,
        metavar="NEWOUTPUT.json",
        help=(
            "Where to write the report. Created exclusively with owner-only permissions. "
            "An existing path, a symlink, a path overlapping the input, or a path inside a "
            "protected_roots entry is refused."
        ),
    )
    parser.add_argument(
        "--baseline",
        metavar="CONDITION",
        help="The condition every other condition is compared against. Defaults to the first declared.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        metavar="N",
        help=(
            f"Bootstrap draws per trial condition over question families "
            f"(default {DEFAULT_BOOTSTRAP_SAMPLES}). Below {BOOTSTRAP_MIN_SAMPLES} the interval "
            f"is left null rather than read off too few resamples."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=f"Seed for the cluster bootstrap, so the interval reproduces (default {DEFAULT_SEED}).",
    )


_EPILOG = f"""\
protocol {PROTOCOL}, schema_version {REPORT_SCHEMA_VERSION}, role {ROLE}

The handoff declares one provenance block, the question inventory with a
family per question, the judgments, the pairwise relations, and one condition
holding a ranking per question. Structural blocks carry an explicit key
contract, so an unknown key is refused there. provenance takes unknown keys and
still requires pool_id, pool_sha256, annotation_sha256, rubric_sha256, rubric,
generation_id, generation_chunks_sha256 and protected_roots. The rubric travels
whole and its digest is verified against it, as the SHA-256 of the rubric's
canonical JSON: sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False,
separators=(',',':')).encode('utf-8')). A rubric carrying a NaN or an infinity
is refused, and a rubric edited after it was hashed no longer matches. The other
three digests are receipts this harness cannot check. A condition_id is an
identifier and may contain a slash: nothing here forms a path from one.

Labels are current-generation scoped. A judgment keeps its source-relative path,
its locator and its content hash; a ranking joins on chunk_id, and a chunk id
claims nothing across rechunking. Nothing outside the handoff is read.

Refused rather than scored: a null, uncertain or missing label value; an
unjudged returned candidate; a duplicate question, judgment, ranking or
relation; a relation naming a passage judged under another question; a
malformed hash; and a ranking that asked for reranking and did not get it.

Grades and gains
  irrelevant 0, contextual 1, direct 2; gain is 2 ** grade - 1.

Measures, all pool relative
  ndcg_at_requested_k            graded, ideal gain from the graded pool; a
                                 pool with no passage above grade 0 is null
  direct_precision_at_requested_k   direct / requested_k, so a short list is
                                 charged for the depth it did not fill
  direct_precision_returned      direct / returned; an empty list is null
  direct_usable_precision_*      direct and usable on the same denominators
  direct_coverage                direct returned / direct in the pool; this is
                                 not corpus recall
  pair_coverage                  annotated pairs both of whose ends were
                                 returned, over every returned pair

Reported alongside
  checked_original_direct_usable  a count of checked originals, telemetry and
                                 never a claim that a quotation is ready
  pairs                           relation counts, unjudged pair count, and a
                                 count of returned pairs sharing a content hash
                                 while their provenance differs

Limits
  Abstention is unmeasured: the protocol carries no no-answer label.
  Coverage is the pool's, not the corpus's.
  A contradiction is a relation label and never lowers a grade.
  Nothing is deduplicated, grouped transitively, or inferred; no relation is
  derived from equal text, and no unique evidence or suppression figure exists.
  Queries inside a family are not independent, so the interval resamples
  families. It is conditional on them and is not an equivalence proof.
  Each trial is compared to the baseline on its own; no figure is averaged
  across trials, and trials sharing a baseline are correlated.
  Pairing is per query over the questions both sides scored, then per family
  over that same set, and the paired and skipped question ids are published.
  One evaluable family gives no interval, and fewer draws than the floor gives
  no interval either. The floor promises neither stability nor power.
  Contrasts sharing one baseline are not adjusted for multiplicity.
  Budgets actually spent are reported per question; unequal budgets mean a
  difference is not attributable to policy alone.
  A report is never written into a protected_roots entry, so a corpus and a
  project stay byte-identical to a run that scored them.
  No p-value, no winner, no recommendation, no policy acceptance. The report
  is exploratory and not policy ready.

Examples
  python scripts/evaluate_pooled.py --input handoff.json --report report.json
  python scripts/evaluate_pooled.py --input handoff.json --report report.json \\
      --baseline baseline-arm --bootstrap-samples 2000 --seed 0
"""


def _resolved_protected_roots(roots: list[str]) -> list[Path]:
    """The roots as absolute paths, so what is enforced is what is reported."""

    return [Path(root).expanduser().resolve() for root in roots]


def _protecting_root(path: Path, roots: list[Path]) -> Path | None:
    """The root a report would land inside, or None when it lands outside all."""

    resolved = path.resolve()
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            return root
    return None


def _refuse_report_target(
    report: str, input_path: str | None, roots: list[str]
) -> None:
    path = Path(report)
    if path.is_symlink():
        _refuse(
            f"{report} is a symlink; a report is written to a new file, not to a link"
        )
    if path.exists():
        _refuse(
            f"{report} already exists; a report is written exclusively and never overwrites"
        )
    if not path.parent.is_dir():
        _refuse(
            f"{path.parent} does not exist; create the directory before writing the report"
        )
    if input_path is not None and path.resolve() == Path(input_path).resolve():
        _refuse(
            f"{report} is the input file; a report never overwrites the handoff it scored"
        )
    resolved_roots = _resolved_protected_roots(roots)
    guarding = _protecting_root(path, resolved_roots)
    if guarding is not None:
        _refuse(
            f"{report} is inside the protected root {guarding}; a report is never written into a "
            f"corpus or a project"
        )


def _write_report(report: str, payload: dict[str, Any]) -> Path:
    path = Path(report)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _summarise(payload: dict[str, Any], written: Path) -> None:
    comparison = payload["comparison"]
    print(
        f"protocol {payload['report_meta']['protocol']}, role {payload['report_meta']['role']}"
    )
    print(f"{payload['relevance_grades']} with gain {payload['gain']}")
    for condition in payload["conditions"]:
        means = condition["aggregates"]
        direct = means["direct_precision_at_requested_k"]["query_mean"]
        coverage = means["direct_coverage"]["query_mean"]
        print(
            f"  {condition['condition_id']}: "
            f"direct@requested_k {direct['value']} over {direct['evaluated']}/{direct['total']} queries, "
            f"direct_coverage {coverage['value']}"
        )
        print(
            f"    checked_original_direct_usable "
            f"{condition['counts']['checked_original_direct_usable']} "
            f"(telemetry), unjudged pairs {condition['counts']['unjudged_pair_count']}"
        )
    budget = payload["budget_equality"]
    print(f"  budgets equal for every question: {budget['equal_for_every_query']}")
    if comparison["status"] == "exploratory":
        print(
            f"  baseline {comparison['baseline_condition']} against {comparison['conditions_compared']}"
        )
        for trial in comparison["conditions_compared"]:
            print(f"    {trial} vs {comparison['baseline_condition']}")
            for metric in METRICS:
                entry = comparison["per_metric"][metric]["conditions"][trial]
                interval = entry["interval_95"]
                span = (
                    f"[{interval['low']}, {interval['high']}]"
                    if interval["usable"]
                    else f"null ({interval['reason']})"
                )
                print(
                    f"      {metric}: {entry['family_weighted_difference']} {span}"
                    f"  families {entry['families_evaluated']}/{entry['families_total']},"
                    f" paired queries {entry['queries_paired']}/{entry['queries_total']}"
                    f", skipped {entry['queries_skipped']}"
                )
    else:
        print(f"  comparison unavailable: {comparison['reason']}")
    print("  no_answer unmeasured; no winner, recommendation, or policy acceptance")
    print(f"report written to {written}")


def main(argv: list[str] | None = None) -> int:
    """Score a handoff named on the command line and write its report."""

    parser = _parser()
    _add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.bootstrap_samples < 1:
            _refuse(
                f"--bootstrap-samples must be at least 1, not {args.bootstrap_samples}"
            )
        input_path = None if str(args.input) == "-" else str(args.input)
        if input_path is not None and not Path(input_path).exists():
            _refuse(f"--input {input_path} does not exist")
        # Read first, check the target second: the roots the report must stay
        # out of are declared by the handoff, and reading writes nothing.
        document = load_handoff(args.input)
        _refuse_report_target(
            args.report, input_path, document["provenance"]["protected_roots"]
        )
        payload = evaluate(
            document,
            baseline=args.baseline,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        written = _write_report(args.report, payload)
    except PoolProtocolError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    _summarise(payload, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
