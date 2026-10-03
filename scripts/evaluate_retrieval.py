"""Measure retrieval quality against a judged query set.

Harness behind the retrieval findings in ``MEASUREMENTS.md``. It runs a judged
query set (``evaluation/ai-and-fetishism-queries.json`` by default) through the
service method behind the public ``search`` tool and reports success@k, MRR,
nDCG@10, document success, and the mean number of distinct sources a result
spans. Every
mode passes ``rerank`` explicitly, so no number depends on the tool default; the
``hybrid+rerank`` row is what a default search now does, and a run where the
cross-encoder did not run fails rather than scoring an unreordered fusion under
a reranked row's name.

Judgments are known-item: one relevant chunk per query, named by a verbatim
snippet so the target re-resolves after re-ingestion. That measures the
findability of a designated passage, not exhaustive recall. ``chunks.jsonl`` is
read read-only; nothing inside the project is written.

Report version 3. The known-item metrics version 1 published keep their names and
their definitions. What a result list *contained* is reported by measures that
say what they compare: two passages are exact duplicates when their normalized
text is equal as an ordered string, and normalization collapses whitespace and
nothing else, so word order, a repeated word, letter case, a sign, a decimal
separator, an operator, a closing mark, and a non-Latin script are all
differences the measure keeps. ``lexical_containment_slots`` is a lexical measure and is named
as one: it counts slots whose ordered word shingles are mostly present in another
returned passage, refuses any pair whose numbers, operators, or negations
differ, and is not evidence that two passages say the same thing. A returned
passage the generation no longer holds is reported as missing coverage, because
two passages it cannot read are not two copies of one passage.

Every run also records the pipeline's own account of itself: the candidate depth
the branch actually used, what the cosine gate admitted above its floor and
what it rejected, whether reranking ran, what the repetition collapse discarded,
and whether the judged passage was present at each ranking stage before and
after each gate. A known-item score cannot see a target lost to a gate, because
the target is the only relevant passage in the set and its loss is a miss rather
than a redundancy.

    uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project

Add --validate-only to check the judged set without searching. A throwaway
warm-up search runs first, because the first search pays a cold index load; its
cost is reported separately.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from fastmcp import Client

from research_rag.core.service import ResearchService
from research_rag.project.config import (
    configured_source_directory,
    resolve_config,
)
from research_rag.project.support import retrieval_policy_fingerprint
from research_rag.retrieval.rerankers import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_CHOICES,
)
from research_rag.retrieval.ultrarag import (
    VanillaUltraRAG,
    create_vanilla_transport,
)

DEFAULT_JUDGMENTS = Path("evaluation/ai-and-fetishism-queries.json")
# The reranked mode is one row per selected reranker: the model is an engine
# setting, so a second model is a second row over the same queries, not a second
# mode. The default model keeps the plain ``hybrid+rerank`` label.
RERANK_MODE = "hybrid+rerank"
MODES = ("bm25", "dense", "hybrid", RERANK_MODE)
#: The methods whose ranking opens the cosine gate. A payload from any other
#: method carries a zeroed gate block, and a block that reads zero is not
#: evidence that a gate ran.
DENSE_METHODS = frozenset({"dense", "hybrid"})
QUERY_CLASSES = ("quote", "paraphrase", "entity")
WHITESPACE = re.compile(r"\s+")
#: The tokenizer the query-to-target overlap was published with: ASCII letters
#: and digits, under three characters dropped. It stays as it was, because a
#: figure is comparable only with a figure measured the same way.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
#: The tokenizer the containment measure compares with: Unicode word character
#: runs, so a script this harness cannot segment into Latin words is compared as
#: the run it is rather than as nothing at all. It is not the tokenizer above,
#: and a measure that used it for the overlap would restate a published figure.
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
#: A number as it is written, sign and decimal separator included. Two passages
#: that disagree about a figure disagree about the world, so a pair whose numbers
#: differ is not a repetition however alike their words are.
NUMBER_PATTERN = re.compile(r"([-+]?\d+(?:[.,]\d+)*)\s*(%?)")
#: Operator characters, compared as the runs they form. A run of hyphens alone is
#: dropped: a dash in prose changes no claim, and treating it as a distinction
#: would take most prose pairs out of consideration for no gain. An exclamation
#: mark is in the class because ``5!`` is a factorial and not a closing sentence
#: mark, so two passages differing in one are not offered to the containment
#: measure as a repetition.
OPERATOR_RUN_PATTERN = re.compile(r"[-+*/^%=<>~≈≤≥≠±×÷!]+")
#: Words that deny rather than assert. A passage and its negation overlap almost
#: completely, and the words are the same words.
NEGATION_TOKENS = frozenset(
    {
        "cannot",
        "false",
        "neither",
        "no",
        "non",
        "nor",
        "not",
        "nothing",
        "nobody",
        "nowhere",
        "un",
        "without",
    }
)
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

#: The most identifiers one report field lists before it is cut, so a wide
#: result list cannot turn a diagnostic into the document.
REPORT_LIST_LIMIT = 10
#: The share of one passage's ordered word shingles that must also occur in
#: another's for the harness to call the shorter slot a lexical repetition of the
#: longer. It is a lexical overlap, deliberately generous where the two passages
#: say the same thing in different words, and it is not a semantic judgement.
LEXICAL_CONTAINMENT_THRESHOLD = 0.9
#: Consecutive words compared at once, so a reordered passage and a passage with
#: a different word in the middle are not contained in one another.
LEXICAL_SHINGLE_SIZE = 2
#: The protocol this harness speaks. Report version 3 replaces the two version 2
#: result-list measures and the version 2 repeated-slot rate, because they
#: described something other than what their names said. Every metric report
#: version 1 carried keeps its name and its definition, so a published quality
#: figure is still comparable.
REPORT_SCHEMA_VERSION = 3


class EvaluationError(RuntimeError):
    """Raised when the judged set or the project cannot be evaluated."""


def harness_provenance() -> dict[str, Any]:
    """Which harness produced a report: its digest, its revision, its interpreter.

    Two reports are comparable only when the code that wrote them is the code that
    was meant to write them, and a file's digest says that where a version number
    cannot: an edited script and its tag share a version. The revision is read from
    git when the checkout can answer, and is ``None`` when it cannot, because a
    missing revision is not a clean tree.
    """

    digest = None
    with contextlib.suppress(OSError):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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


#: What each reported name means, so a reader does not have to hold the harness in
#: mind to read its output. Every name the report carries that a reader could
#: mistake for another measure is defined here.
METRIC_DEFINITIONS: dict[str, str] = {
    "success_at_1": "Whether the judged chunk is the first result.",
    "success_at_3": "Whether the judged chunk is among the first three results.",
    "success_at_k": "Whether the judged chunk is among the first top_k results.",
    "reciprocal_rank": (
        "1 divided by the judged chunk's rank, and 0 when it is not returned."
    ),
    "mrr": "The mean of reciprocal_rank over a mode's queries.",
    "ndcg_at_k": (
        "Binary-gain nDCG at top_k against an ideal ranking of the one judged "
        "chunk, with log2 discounting."
    ),
    "document_success_at_k": (
        "Whether the judged chunk's document is among the first top_k results, "
        "beside chunk-level success."
    ),
    "lexical_overlap": (
        "Share of the query's content words, ASCII tokens of three or more "
        "characters with the corpus language's function words removed, that occur "
        "in the judged chunk. Measured between the query and its judged chunk, so "
        "it describes the judged set rather than any mode's retrieval."
    ),
    "distinct_normalized_texts": (
        "How many different passages a result list held, by equality of normalized "
        "text as an ordered string, where normalization collapses whitespace runs "
        "to one space and trims the ends and changes nothing else: letter case, "
        "punctuation, operators, signs, and word order all count as differences."
    ),
    "exact_duplicate_slots": (
        "Slots beyond the first in a normalized-text equality group, and the "
        "identifier list of each such group. Two passages differing only in "
        "wrapping are one passage; two differing only in case or in a trailing "
        "mark are not, so the count is a lower bound and the containment measure "
        "below is where a reprint that punctuates differently is found."
    ),
    "lexical_containment_slots": (
        "Slots whose ordered word shingles are at least the threshold present in "
        "another returned passage's, counted once per slot. Lexical overlap, not "
        "semantic redundancy: it says the words recur, never that the passages say "
        "the same thing."
    ),
    "lexical_containment_groups": (
        "Distinct passages another returned passage repeats at that threshold, so "
        "several copies of one passage are one repetition rather than several."
    ),
    "same_source_pairs": (
        "Slots sharing a source file with another slot, a source-spread measure "
        "rather than a duplication one."
    ),
    "dense_eligible_total": (
        "Dense candidates that reached the cosine gate, after the quality filters "
        "and the length floor: the denominator a share of rejections needs."
    ),
    "dense_admitted_above_floor": "Eligible candidates admitted at or above the cosine floor.",
    "dense_admitted_below_floor": (
        "Eligible candidates admitted below the floor by the relative margin."
    ),
    "dense_rejected_below_floor": "Eligible candidates rejected below the admission floor.",
    "dense_gate_ran": (
        "Whether the method ranked the dense half at all. A payload carries a gate "
        "block for every method, so a block of zeros is not a gate that ran."
    ),
    "candidate_depth": (
        "The depth the branch actually asked for: min(active chunks, maximum "
        "candidates, max(minimum candidates, top_k * 4))."
    ),
    "rerank_window": (
        "The reranked window: min(candidates ranked, rerank_max_candidates, "
        "max(top_k * rerank_window_multiple, rerank_window_floor))."
    ),
    "collapsed_count": (
        "Candidates the repetition collapse removed after reranking, with the pair "
        "naming each discarded passage's source and the passage it repeated."
    ),
    "target_stage_presence": (
        "Whether the judged chunk was present at each pipeline stage, before and "
        "after the gate, the fusion, the reranked window, and the collapse. null "
        "where the stage did not run or its identifier list was truncated."
    ),
    "evaluation_trace": (
        "The pipeline's own account of the search, kept whole: every stage's "
        "identifier list with its count and truncation flag, the cosine gate's "
        "counters, the reranked window and the identifiers it scored, and what the "
        "repetition collapse discarded with each discard's source and "
        "representative. Identifiers and counts only, every list bounded by the "
        "engine. A run whose engine emitted no trace carries null."
    ),
    "reranked_then_collapsed_count": (
        "Candidates the cross-encoder returned scores for that the repetition "
        "collapse then discarded, read from the intersection of the scored "
        "window's identifiers with the discarded passages of "
        "collapsed_repetitions.pairs. The discarded side is complete, so the "
        "count is the whole figure unless the scored list was cut."
    ),
    "reranked_then_collapsed_is_lower_bound": (
        "Whether the count above is a lower bound rather than the whole figure, "
        "which it is when the engine's trace budget cut the scored identifier "
        "list. A count of zero beside a true flag means nothing was measured."
    ),
    "reranked_then_collapsed_by_reason": (
        "The engine's own collapsed_by labels for those discards. The engine "
        "decided them; this reports them and asserts no duplication of its own."
    ),
    "repeated_slot_rate_all_queries": (
        "Share of a mode's query-passage pairs whose passage more than one distinct "
        "query returned. Each query contributes one slot per distinct passage."
    ),
    "repeated_slot_rate_cross_target_family": (
        "The same share restricted to passages that answered queries from more "
        "than one question family. A family is the group the judged set's "
        "declarations form: a target declares one family_id of its own, its "
        "queries may declare others, and a target is grouped with every family "
        "either declares, so a target-level and a query-level declaration are "
        "both honoured rather than one chosen over the other. A target declaring "
        "nothing is its own family. A quote query and its own paraphrase sharing "
        "a target is one family and is not counted as a generic leader."
    ),
    "p50_seconds": "Nearest-rank median of the measured query times.",
    "p95_seconds": (
        "Nearest-rank 95th percentile of the measured query times, which over a "
        "small run is not the slowest query."
    ),
    "max_seconds": "The slowest measured query.",
}

#: Report version 2 names this harness did not compute. Each entry says why, so a
#: comparison refuses them by cause rather than by a missing key.
SUPERSEDED_METRICS: dict[str, str] = {
    "runs[].distinct_evidence_spans": (
        "Report version 3's distinct_normalized_texts. The version 2 value grouped "
        "passages by an unordered set of ASCII tokens, so it dropped word order, "
        "repeated words, signs, decimals, and operators, merged every passage "
        "outside Latin script into one empty group, and was arithmetically the "
        "result count minus the exact-duplicate count rather than a second "
        "observation."
    ),
    "runs[].near_duplicate_slots": (
        "Report version 3's lexical_containment_slots. The version 2 value took "
        "containment over unordered word sets, which read a negation as the same "
        "evidence as its assertion and double-counted containment chains, and its "
        "name claimed redundancy rather than word overlap."
    ),
    "summary[mode].repeated_slot_rate": (
        "Report version 3's repeated_slot_rate_all_queries and "
        "repeated_slot_rate_cross_target_family. The version 2 value counted slots "
        "rather than distinct queries, so one list returning a passage twice counted "
        "as repetition across queries, and it counted a quote query and its own "
        "paraphrase sharing a target as repetition across targets."
    ),
    "report.duplicate_containment": (
        "Report version 3's lexical_containment, which names the threshold, the "
        "shingle size, the denominator, and the distinctions a pair must keep."
    ),
}


def no_answer_support(judged_set: dict[str, Any]) -> dict[str, Any]:
    """Whether the judged set says anything about a query the corpus cannot answer.

    Abstention is measurable only with judged queries whose answer is that nothing
    answers them. This set holds none, so nothing here is measured and nothing is
    invented: a harness that labelled its own misses as unanswerable would be
    scoring its own abstentions against a label it wrote.
    """

    classes = {str(item.get("class") or "") for item in judged_set.get("queries") or []}
    return {
        "judged_no_answer_queries": 0,
        "classes_present": sorted(classes),
        "measured": False,
        "note": (
            "The judged set holds no query whose correct answer is that the corpus "
            "does not answer it, so abstention is not measured by this run and no "
            "abstention figure is reported."
        ),
    }


def normalize(value: str) -> str:
    """Collapse whitespace so snippets match across wrapping differences."""

    return WHITESPACE.sub(" ", value).strip()


def normalized_text(value: str) -> str:
    """The passage's text as two passages are compared for exact equality.

    Wrapping changes no evidence, so runs of whitespace collapse to one space and
    the ends are trimmed. Nothing else is, because every remaining character can
    be part of a claim: letter case (``V`` against ``v``, ``k`` against ``K``),
    word order, a repeated word, a sign, a decimal separator, an operator, a
    closing mark (``5!`` is not ``5``, and a factorial is not a sentence), and a
    script with no spaces in it. Collapsing any of those turns two passages into
    one and reports an exact duplicate the corpus does not hold, which is a
    mislabelled claim rather than a conservative reading. A measure that under-
    reports is repairable; one that invents a duplicate is not. The canonical text
    is never rewritten; this is a comparison key and nothing else.
    """

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


def word_shingles(
    value: str, size: int = LEXICAL_SHINGLE_SIZE
) -> tuple[tuple[str, ...], ...]:
    """The passage's overlapping word shingles, in the order they occur.

    Consecutive words are compared together, so a passage with a word in a
    different place and one with a word missing are not contained in one another
    on the strength of the words they share. A passage shorter than the shingle
    size is one shingle, so a two-word caption is comparable rather than empty.
    """

    tokens = WORD_PATTERN.findall(value)
    if not tokens:
        return ()
    if len(tokens) < size:
        return (tuple(tokens),)
    return tuple(
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def protected_signature(value: str) -> tuple[tuple[str, ...], ...]:
    """The parts of a passage a repetition may not quietly change.

    Two passages whose figures disagree, whose arithmetic differs, or where one
    denies what the other asserts are not the same evidence however alike their
    remaining words are, so a pair whose signature differs is not offered to the
    containment measure at all. The signature is a list rather than a set because
    a figure's position in a passage is part of what it says.
    """

    numbers = tuple(
        f"{figure}{unit}"
        for figure, unit in NUMBER_PATTERN.findall(
            WHITESPACE.sub(" ", value).casefold()
        )
    )
    operators = tuple(
        run for run in OPERATOR_RUN_PATTERN.findall(value) if set(run) - {"-"}
    )
    negations = tuple(
        sorted(
            {
                token
                for token in WORD_PATTERN.findall(value.casefold())
                if token in NEGATION_TOKENS or token.startswith(("un", "non"))
            }
        )
    )
    return (numbers, operators, negations)


def duplicate_measures(
    chunk_ids: list[str], index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """What one result list contains, as distinct from how well it ranked.

    Two passages are exact duplicates when their normalized text is equal as an
    ordered string, where normalization collapses whitespace and nothing else, so
    a reordered passage, a repeated word, a changed letter case, a changed sign, a
    dropped factorial, and a script without spaces in it are six differences
    rather than none.
    ``lexical_containment_slots`` is the lenient companion: a slot whose ordered
    word shingles are mostly present in another returned passage, where the two
    agree about every figure, operator, and negation. It is a statement about
    words, and it is not a claim that two passages make the same point — that is
    what the reranker and the reader are for. The exact count is reported beside
    it because exact equality on a corpus that reprints material reads a
    confident low number, and a measure that can only read low is not evidence of
    an absence. It now also reads low on a passage whose case or closing mark
    varies from its reprint's, which is the direction to err in: a missed
    duplicate is a weaker claim than an invented one.

    A returned passage the generation does not hold, or one whose stored text is
    empty, is reported as missing coverage and the four counts become ``None``.
    Two passages this harness cannot read are not two copies of one passage, and
    the alternative — treating every unreadable passage as the same empty one —
    turns a broken index into a perfect duplication score.
    """

    keys: list[str | None] = []
    missing: list[str] = []
    for chunk_id in chunk_ids:
        chunk = index.get(chunk_id)
        key = normalized_text(_chunk_text(chunk)) if isinstance(chunk, dict) else ""
        if key:
            keys.append(key)
        else:
            keys.append(None)
            missing.append(str(chunk_id))

    coverage = {
        "slots_total": len(chunk_ids),
        "slots_with_text": len(chunk_ids) - len(missing),
        "slots_missing_text": len(missing),
        "missing_chunk_ids": missing[:REPORT_LIST_LIMIT],
        "complete": not missing,
    }
    if missing:
        return {
            "duplicate_measure_coverage": coverage,
            "distinct_normalized_texts": None,
            "exact_duplicate_slots": None,
            "exact_duplicate_groups": None,
            "lexical_containment_slots": None,
            "lexical_containment_groups": None,
            "lexical_containment_pairs": None,
            "same_source_pairs": None,
        }

    representatives: list[int] = []
    seen: dict[str, int] = {}
    exact_groups: dict[int, list[str]] = {}
    exact_duplicate_slots = 0
    for position, key in enumerate(keys):
        assert key is not None
        representative = seen.get(key)
        if representative is None:
            seen[key] = position
            representatives.append(position)
            exact_groups[position] = [str(chunk_ids[position])]
            continue
        exact_duplicate_slots += 1
        exact_groups[representative].append(str(chunk_ids[position]))

    # Only the first passage of each exact group is compared for containment, so
    # a reprinted passage is counted once as a reprint and never again as a
    # repetition of itself.
    shingle_sets = {
        position: frozenset(word_shingles(key or ""))
        for position, key in enumerate(keys)
    }
    signatures = {
        position: protected_signature(key or "") for position, key in enumerate(keys)
    }
    containment_slots = 0
    containment_groups: set[int] = set()
    containment_pairs: list[dict[str, Any]] = []
    for position in representatives:
        own = shingle_sets[position]
        if not own:
            continue
        for other in representatives:
            if other == position:
                continue
            if signatures[other] != signatures[position]:
                continue
            shared = own & shingle_sets[other]
            if len(shared) / len(own) < LEXICAL_CONTAINMENT_THRESHOLD:
                continue
            containment_slots += 1
            containment_groups.add(other)
            containment_pairs.append(
                {
                    "slot_chunk_id": str(chunk_ids[position]),
                    "repeated_chunk_id": str(chunk_ids[other]),
                    "shared_shingles": len(shared),
                    "slot_shingles": len(own),
                    "containment": round(len(shared) / len(own), 4),
                }
            )
            break

    return {
        "duplicate_measure_coverage": coverage,
        "distinct_normalized_texts": len(seen),
        "exact_duplicate_slots": exact_duplicate_slots,
        "exact_duplicate_groups": [
            group for group in exact_groups.values() if len(group) > 1
        ][:REPORT_LIST_LIMIT],
        "lexical_containment_slots": containment_slots,
        "lexical_containment_groups": len(containment_groups),
        "lexical_containment_pairs": containment_pairs[:REPORT_LIST_LIMIT],
        "same_source_pairs": same_source_pairs(chunk_ids, index),
    }


def same_source_pairs(chunk_ids: list[str], index: dict[str, dict[str, Any]]) -> int:
    """Slots sharing a source file with another slot in the same result list.

    Distinct evidence comes from one file often enough that this is reported beside
    the text measures rather than instead of them: a source-only diversity penalty
    moves this number and leaves the text measures alone. Two passages carrying
    equal text from two sources are one piece of evidence twice and no source
    twice, so the text measure and this one answer different questions.
    """

    counts: dict[str, int] = {}
    for chunk_id in chunk_ids:
        source_id = str(index.get(chunk_id, {}).get("source_id") or "")
        if source_id:
            counts[source_id] = counts.get(source_id, 0) + 1
    return sum(size - 1 for size in counts.values() if size > 1)


def repetition_rates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The share of a mode's slots held by a passage more than one query returned.

    A passage that answers every question is a generic leader rather than an
    answer, and no per-query metric sees it: each query's own list looks
    reasonable. It is measured across the run for that reason.

    Two rates are reported because one number conflates two different things. The
    all-queries rate counts every passage more than one query returned, including
    the passages a quote query and its own paraphrase share, which is what a
    judged set built from paired query styles produces at full marks and by
    itself. The cross-target-family rate counts only passages that answered
    queries judged about more than one question family, so that floor does not
    register as a generic leader.

    A question family is the judged set's own declaration where it makes one: a
    query's or its target's ``family_id``, and otherwise the target. Two targets
    the judged set declares as one family are one family, however differently
    their snippets read, and a rate that treated them as independent would score
    a perfectly disjoint result list as repetition. A run naming neither is
    counted, and the cross-family rate is ``None`` rather than a figure resting
    on an assumed independence.

    Each query contributes one slot per distinct passage, so a passage returned
    twice by one query does not inflate the numerator, and the denominator is
    reported beside both rates.
    """

    slots = 0
    queries_by_chunk: dict[str, set[str]] = {}
    families_by_chunk: dict[str, set[str]] = {}
    missing_family = 0
    for run in runs:
        distinct = {str(item) for item in (run.get("returned_chunk_ids") or [])}
        query_id = str(run.get("query_id") or "")
        family_id = str(run.get("family_id") or "")
        if not family_id:
            missing_family += 1
        for chunk_id in distinct:
            queries_by_chunk.setdefault(chunk_id, set()).add(query_id)
            if family_id:
                families_by_chunk.setdefault(chunk_id, set()).add(family_id)
        slots += len(distinct)

    repeated = [
        chunk_id for chunk_id, queries in queries_by_chunk.items() if len(queries) > 1
    ]
    cross_family = [
        chunk_id
        for chunk_id in repeated
        if len(families_by_chunk.get(chunk_id, set())) > 1
    ]

    def rate(chunk_ids: list[str]) -> float | None:
        if not slots:
            return None
        return sum(len(queries_by_chunk[chunk_id]) for chunk_id in chunk_ids) / slots

    return {
        "repeated_slot_rate_all_queries": rate(repeated),
        "repeated_slot_rate_cross_target_family": (
            None if missing_family else rate(cross_family)
        ),
        "repeated_slot_denominator": slots,
        "repeated_query_count": len(queries_by_chunk),
        "distinct_query_count": len({str(run.get("query_id") or "") for run in runs}),
        "repeated_passage_count": len(repeated),
        "cross_target_family_passage_count": len(cross_family),
        "runs_without_family_id": missing_family,
    }


def latency_percentiles(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    """The median, the 95th percentile, and the slowest measured query.

    Nearest-rank rather than interpolated, so every number is a query this harness
    actually ran. The slowest query is reported separately from the percentile: a
    p95 over thirty queries is the second-slowest of them, and calling that the
    slowest query overstates it by a factor the sample size decides. A run with no
    timing reports ``None``, because a latency was not measured rather than
    measured as zero.
    """

    values = sorted(
        float(run["elapsed_seconds"])
        for run in runs
        if isinstance(run.get("elapsed_seconds"), (int, float))
    )
    if not values:
        return {"p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    return {
        "p50_seconds": values[max(math.ceil(0.50 * len(values)) - 1, 0)],
        "p95_seconds": values[max(math.ceil(0.95 * len(values)) - 1, 0)],
        "max_seconds": values[-1],
    }


def _resolve_target(
    target: dict[str, Any],
    by_document: dict[str, list[dict[str, Any]]],
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one judged target, or raise with why it cannot be resolved.

    The reasons are kept distinct because the remedies are: a source the
    generation does not hold, two sources answering one path, a snippet the
    extraction no longer produces, and a snippet two passages both contain.
    """

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
                f"no document in this generation matches {sorted(wanted) or named!r}"
            )
        matched = [documents[named]]
    if len(matched) != 1:
        raise EvaluationError(f"{len(matched)} documents match {sorted(wanted)}")
    document = matched[0]
    document_id = str(document["document_id"])
    candidates = by_document.get(document_id, [])
    if not candidates:
        raise EvaluationError(
            f"the generation holds no chunks for document {document_id}"
        )
    hits = [
        chunk
        for chunk in candidates
        if snippet in normalize(_chunk_text(chunk)).casefold()
    ]
    if len(hits) != 1:
        raise EvaluationError(
            f"snippet resolves to {len(hits)} chunks in "
            f"{document.get('source_relative_path') or document_id}, "
            "expected exactly one"
        )
    hit = hits[0]
    return {
        "target_id": str(target["target_id"]),
        # A judged set may declare that several targets answer one question
        # family. The declaration is the set's own, so it is carried through
        # rather than inferred from the snippets.
        "family_id": target.get("family_id"),
        "source_path": document.get("source_path") or target.get("source_path"),
        "locator": hit.get("locator"),
        "chunk_id": str(hit.get("chunk_id")),
        "document_id": document_id,
        "chunk_id_at_measurement": target.get("chunk_id_at_measurement"),
        "chunk_text": _chunk_text(hit),
    }


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

    Every target is attempted before anything is reported, because a run that
    stops at the first failure records one reason per attempt and leaves a
    reviewer to rerun it to find the next. The failures are collected and raised
    together, each with its own reason, so one run names every target that needs
    a decision. Nothing is resolved for the searches: a run that searched the
    subset that happened to resolve would report a protocol over fewer judgments
    than the set declares, which is the number a reader would take away.

    A target in ``skip`` is not attempted at all. A judged source the corpus no
    longer holds is a benchmark decision, not a measurement, so it is named on
    the command line and in the report rather than relaxing resolution for all
    targets, and a skip never excuses a failure in a target that was not skipped.
    """

    by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_document.setdefault(str(chunk.get("document_id")), []).append(chunk)

    resolved: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []
    for target in targets:
        target_id = str(target["target_id"])
        if target_id in skip:
            continue
        try:
            resolved[target_id] = _resolve_target(target, by_document, documents)
        except EvaluationError as exc:
            failures.append((target_id, str(exc)))
    if failures:
        raise EvaluationError(
            f"{len(failures)} of "
            f"{len(resolved) + len(failures)} judged targets did not resolve "
            "uniquely:\n"
            + "\n".join(f"  {target_id}: {reason}" for target_id, reason in failures)
        )
    return resolved


def resolve_families(
    queries: list[dict[str, Any]], resolved: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Map each target to the question family the judged set declares for it.

    A set may declare a family twice: once on a target, covering every query of
    it, and once on a query, covering that query. Both declarations name the same
    field and the same family, so neither is the other's parent and choosing
    between them is wrong in both directions. A target whose query declares
    ``sub-1`` while the target itself declares ``shared`` with another target
    belongs with both, and the family answering for it is the group they form
    together. A label declared on a query of one target and on the record of
    another is the same declaration twice, and unites across the two levels.

    The declarations are therefore read as a graph. Two namespaces, and only two:
    ``target:<id>`` for a judged passage and ``family:<label>`` for a declared
    family, the same node whether it was declared on a target or on a query. A
    target is joined to every family its own record and its queries declare, and a
    target's family is then the connected component it belongs to, named by the
    first target id in that component. The namespaces are separate, so a family
    label that happens to read like a target id joins a family to nothing and
    cannot merge two targets that share no declaration.

    A target declaring nothing is its own family, because two queries about the
    same passage are one family by construction. A target absent from
    ``resolved`` was skipped, so no query of it ran.
    """

    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != root:
            parent[name], name = root, parent[name]
        return root

    def join(left: str, right: str) -> None:
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)

    # Every resolved target is a node, so a target whose queries the run did not
    # select still has a family and still belongs to a group.
    targets = sorted(resolved)
    for target_id in targets:
        find(f"target:{target_id}")
    for query in queries:
        target_id = str(query.get("target_id") or "")
        target = resolved.get(target_id)
        if target is None:
            continue
        for label in (target.get("family_id"), query.get("family_id")):
            name = str(label or "").strip()
            if name:
                join(f"target:{target_id}", f"family:{name}")

    groups: dict[str, list[str]] = {}
    for target_id in targets:
        groups.setdefault(find(f"target:{target_id}"), []).append(target_id)
    return {
        target_id: members[0] for members in groups.values() for target_id in members
    }


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
        **latency_percentiles(runs),
        **_redundancy_means(runs),
        **_gate_means(runs),
        **_stage_means(runs),
    }


def _measure_mean(runs: list[dict[str, Any]], key: str) -> float | None:
    """The mean of a per-run measure, or ``None`` when no run carries it.

    A run record that does not carry the measure has not measured it, and a mean
    over no such runs would read as a measured zero rather than as an absent
    measure.
    """

    values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
    return _mean(values) if values else None


def _redundancy_means(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The result-list and source-spread means beside the quality columns."""

    return {
        "mean_distinct_normalized_texts": _measure_mean(
            runs, "distinct_normalized_texts"
        ),
        "mean_exact_duplicate_slots": _measure_mean(runs, "exact_duplicate_slots"),
        "mean_lexical_containment_slots": _measure_mean(
            runs, "lexical_containment_slots"
        ),
        "mean_same_source_pairs": _measure_mean(runs, "same_source_pairs"),
        "queries_with_duplicate_coverage": (
            sum(
                1
                for run in runs
                if run.get("duplicate_measure_coverage", {}).get("complete")
            )
            if any("duplicate_measure_coverage" in run for run in runs)
            else None
        ),
    }


def _gate_means(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The cosine gate's own counts, and how often it acted.

    These are the engine's numbers, read rather than recomputed: the gate decides
    before fusion, so a harness that inferred its rejections from the fused list
    would be measuring the wrong thing. ``withheld_candidates`` is a different
    count entirely — passages withheld from the *answer*, after ranking, for
    corrupt text alone — and a search rejecting a hundred dense candidates
    reports zero withheld.
    """

    counted = [run for run in runs if run.get("dense_gate_counts_complete") is True]
    if not counted:
        return {
            "mean_dense_eligible_total": None,
            "mean_dense_admitted_above_floor": None,
            "mean_dense_admitted_below_floor": None,
            "mean_dense_rejected_below_floor": None,
            "queries_with_dense_rejections": None,
            "queries_with_dense_gate_conservation_failures": None,
        }
    return {
        "mean_dense_eligible_total": _mean(
            [float(run["dense_eligible_total"]) for run in counted]
        ),
        "mean_dense_admitted_above_floor": _mean(
            [float(run["dense_admitted_above_floor"]) for run in counted]
        ),
        "mean_dense_admitted_below_floor": _mean(
            [float(run["dense_admitted_below_floor"]) for run in counted]
        ),
        "mean_dense_rejected_below_floor": _mean(
            [float(run["dense_rejected_below_floor"]) for run in counted]
        ),
        "queries_with_dense_rejections": sum(
            1 for run in counted if int(run["dense_rejected_below_floor"]) > 0
        ),
        "queries_with_dense_gate_conservation_failures": sum(
            1 for run in counted if run.get("dense_conserved") is False
        ),
    }


def _stage_means(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the judged passage was present at each ranking stage.

    This is the measure a known-item score cannot substitute for: a target the
    cosine gate dropped is a miss, but so is a target the fusion never saw, and
    the two need different remedies. A stage that never ran, or whose identifier
    list the engine cut before the target's place in it, contributes nothing to
    its mean and is counted as unevaluated beside it.
    """

    stages: list[str] = []
    for run in runs:
        for stage in run.get("target_stage_presence") or {}:
            if stage not in stages:
                stages.append(stage)
    means: dict[str, Any] = {}
    for stage in stages:
        values = [
            run["target_stage_presence"][stage]
            for run in runs
            if stage in (run.get("target_stage_presence") or {})
        ]
        evaluated = [value for value in values if value is not None]
        means[f"mean_target_present_{stage}"] = (
            _mean([1.0 if value else 0.0 for value in evaluated]) if evaluated else None
        )
        means[f"target_stage_evaluated_queries_{stage}"] = len(evaluated)
    return means


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
            **repetition_rates(mode_runs),
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
        f"{'texts':>7}{'dup':>5}{'lex':>5}{'1src':>5}{'repXF':>7}{'repAll':>7}"
        f"{'p50s':>6}{'p95s':>6}{'rej':>5}"
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
            f"{row['mean_distinct_sources']:>6.1f}" + _result_list_columns(row, payload)
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


def _result_list_columns(row: dict[str, Any], payload: dict[str, Any]) -> str:
    """The result-list columns, or blanks when the run carries no such measure.

    A blank is where a measure was not taken. Printing a zero there would read as
    an absence of duplication rather than as an absence of measurement.
    """

    def count(key: str) -> str:
        value = row.get(key)
        return f"{value:>5.1f}" if isinstance(value, (int, float)) else "     "

    def rate(key: str) -> str:
        value = payload.get(key)
        return (
            f"{100.0 * value:>6.1f}%" if isinstance(value, (int, float)) else "      "
        )

    def seconds(key: str) -> str:
        value = row.get(key)
        return f"{value:>6.2f}" if isinstance(value, (int, float)) else "      "

    return (
        count("mean_distinct_normalized_texts")
        + count("mean_exact_duplicate_slots")
        + count("mean_lexical_containment_slots")
        + count("mean_same_source_pairs")
        + rate("repeated_slot_rate_cross_target_family")
        + rate("repeated_slot_rate_all_queries")
        + seconds("p50_seconds")
        + seconds("p95_seconds")
        + count("mean_dense_rejected_below_floor")
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
    parser.add_argument(
        "--allow-degraded-rerank",
        action="store_true",
        help=(
            "Record a reranked row that did not rerank instead of failing. The "
            "row is then an unreordered fusion, and the report records every "
            "affected run under degraded."
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
    chunk_index: dict[str, dict[str, Any]] | None = None,
    allow_degraded: bool = False,
    family_id: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await service.search(
        query,
        top_k=top_k,
        include_staleness=False,
        evaluation_trace=True,
        **settings,
    )
    elapsed = time.perf_counter() - started
    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        raise EvaluationError(f"search returned an unexpected payload for {query_id}")

    hits = payload["hits"]
    ranked_chunk_ids = [str(hit.get("chunk_id")) for hit in hits]
    ranked_document_ids = [str(hit.get("document_id")) for hit in hits]
    relevant = {str(target["chunk_id"])}
    target_chunk_id = str(target["chunk_id"])
    rank = next(
        (
            position
            for position, item in enumerate(ranked_chunk_ids, 1)
            if item in relevant
        ),
        None,
    )
    degraded = _degradation_reasons(payload, mode=mode)
    if degraded and not allow_degraded:
        raise EvaluationError(
            f"{mode} did not run the way its row claims for {query_id}: "
            f"{'; '.join(degraded)}. Pass --allow-degraded-rerank to record the "
            "row as an unreordered fusion instead."
        )
    return {
        "query_id": query_id,
        "target_id": str(target["target_id"]),
        "family_id": str(family_id),
        "class": query_class,
        "query": query,
        "mode": mode,
        "reranker_model": payload.get("reranker_model"),
        "top_k": top_k,
        "target_chunk_id": target_chunk_id,
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
        "relevance_limited": payload.get("relevance_limited"),
        "degraded_reasons": degraded,
        **_branch_measures(payload),
        **_collapsed_measures(payload),
        **_scored_then_collapsed(payload),
        **_rejection_measures(payload, target_chunk_id=target_chunk_id),
        **_dense_gate_measures(payload),
        **_trace_measures(payload, target_chunk_id=target_chunk_id),
        "returned_chunk_ids": ranked_chunk_ids,
        "elapsed_seconds": elapsed,
        **duplicate_measures(ranked_chunk_ids, chunk_index or {}),
    }


def _degradation_reasons(payload: dict[str, Any], *, mode: str) -> list[str]:
    """Why a row would be reporting something other than what its label says.

    A reranked row whose cross-encoder did not run is an unreordered fusion, and a
    hybrid row with no fusion block was ranked by one branch alone. Either scores
    under the wrong name, so it is named rather than averaged in.
    """

    reasons: list[str] = []
    if bool(payload.get("rerank_requested")) and not bool(payload.get("reranked")):
        fallback = payload.get("rerank_fallback") or {}
        reasons.append(
            "rerank requested but not applied"
            + (f" ({fallback.get('reason')})" if fallback.get("reason") else "")
        )
    if mode == "hybrid" and not payload.get("fusion"):
        reasons.append("hybrid row without a fusion block")
    return reasons


def _branch_measures(payload: dict[str, Any]) -> dict[str, Any]:
    """The branch the search actually took, read from the payload it returned."""

    trace = payload.get("evaluation_trace")
    trace = trace if isinstance(trace, dict) else {}
    return {
        "retrieval_method": payload.get("retrieval_method"),
        "candidate_depth": payload.get("candidate_depth"),
        "candidate_count": payload.get("candidate_count"),
        "candidate_distinct_reference_count": payload.get(
            "candidate_distinct_reference_count"
        ),
        "rerank_requested": bool(payload.get("rerank_requested")),
        "reranked": payload.get("reranked"),
        "rerank_fallback": payload.get("rerank_fallback"),
        "rerank_window": payload.get("rerank_window"),
        "fusion": payload.get("fusion"),
        "trace_candidate_depth": trace.get("candidate_depth"),
    }


def _collapsed_measures(payload: dict[str, Any]) -> dict[str, Any]:
    """What the repetition collapse removed, and which passage each copy lost to.

    The pairs are the engine's own disclosure, kept whole: a collapse is a
    decision about two passages from two files, and the pair is the only part of
    it a reader can act on. A search that collapsed nothing reports zero pairs,
    which is a measurement, because the pairs list is present either way.
    """

    collapse = payload.get("collapsed_repetitions") or {}
    pairs = [
        entry for entry in (collapse.get("pairs") or []) if isinstance(entry, dict)
    ]
    return {
        "collapsed_count": len(pairs),
        "collapsed_by_same_words": sum(
            1 for pair in pairs if pair.get("collapsed_by") == "same_words"
        ),
        "collapsed_by_same_meaning": sum(
            1 for pair in pairs if pair.get("collapsed_by") == "same_meaning"
        ),
        "collapsed_pairs": pairs[:REPORT_LIST_LIMIT],
        "collapsed_pairs_truncated": len(pairs) > REPORT_LIST_LIMIT,
    }


def _scored_then_collapsed(payload: dict[str, Any]) -> dict[str, Any]:
    """How many candidates the cross-encoder scored were then collapsed away.

    The reranked window is a budget: only so many candidates reach the model, and
    the rest arrive in fused order. The repetition collapse runs afterwards over
    both, so a discarded candidate may have been scored or may have arrived in
    the tail. Which of the two it was decides what a larger window would buy, and
    no other field in the report can say it.

    The two sides are the engine's own identifiers: the window the reranker
    returned scores for, and the passages ``collapsed_repetitions.pairs`` names as
    discarded. That list holds one entry per collapse and is not bounded, so it
    is the complete set. The scored side is bounded by the trace budget, so where
    that budget cut it the intersection is a lower bound and says so rather than
    reading as the whole.

    The reasons are the engine's own ``collapsed_by`` labels, repeated here rather
    than renamed: this counts the collapses the engine decided on, and says
    nothing about whether any two passages say the same thing.
    """

    trace = payload.get("evaluation_trace")
    trace = trace if isinstance(trace, dict) else {}
    rerank = trace.get("rerank")
    rerank = rerank if isinstance(rerank, dict) else {}
    scored = {str(item) for item in (rerank.get("scored_ids") or [])}
    collapse = payload.get("collapsed_repetitions")
    collapse = collapse if isinstance(collapse, dict) else {}
    pairs = [
        entry for entry in (collapse.get("pairs") or []) if isinstance(entry, dict)
    ]
    discarded = {str(pair["chunk_id"]) for pair in pairs if pair.get("chunk_id")}
    both = scored & discarded
    by_reason: dict[str, int] = {}
    for pair in pairs:
        if str(pair.get("chunk_id") or "") in both:
            reason = str(pair.get("collapsed_by") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
    complete = not bool(rerank.get("scored_truncated"))
    return {
        "reranked_then_collapsed_count": len(both),
        "reranked_then_collapsed_is_lower_bound": not complete,
        "reranked_then_collapsed_by_reason": dict(sorted(by_reason.items())),
        "collapse_discarded_count": len(discarded),
        "scored_candidate_count": rerank.get("scored_count"),
    }


def _rejection_measures(
    payload: dict[str, Any], *, target_chunk_id: str
) -> dict[str, Any]:
    """What the gates did to this query's candidates, and to the judged passage.

    ``withheld_candidates`` counts corrupt passages withheld from the answer and
    nothing else. The gates' own rejections live in ``rejected_candidates`` and
    in the bounded examples beside it, which is where a target dropped for scoring
    below the cosine floor or for being too short to cite is named.
    """

    withheld = payload.get("withheld_candidates") or {}
    withheld_ids = {
        str(example)
        for entry in (withheld.get("reasons") or {}).values()
        for example in (entry.get("example_chunk_ids") or [])
    }
    examples = payload.get("rejected_candidate_examples") or {}
    reasons = examples.get("reasons") or {}
    return {
        "withheld_total": int(withheld.get("total") or 0),
        "target_listed_as_withheld": target_chunk_id in withheld_ids,
        "rejected_candidates": payload.get("rejected_candidates"),
        "rejected_candidate_examples": {
            "policy": examples.get("policy"),
            "limit_per_reason": examples.get("limit_per_reason"),
            "reasons": reasons,
        },
        # Bounded by the engine, so an empty list is not proof the target passed
        # every gate. The stage measures are what say where it went.
        "target_rejected_by": sorted(
            reason
            for reason, entries in reasons.items()
            if target_chunk_id
            in {str(entry.get("chunk_id")) for entry in entries or []}
        ),
    }


def _dense_gate_measures(payload: dict[str, Any]) -> dict[str, Any]:
    """What the pre-fusion cosine gate did to this query's dense candidates.

    These are the engine's own counts, read rather than recomputed: the gate
    decides before fusion, so a harness that inferred its rejections from the
    fused list would be measuring the wrong thing. ``withheld_candidates`` is a
    different count entirely — passages withheld from the *answer*, after ranking,
    for corrupt text alone — and a search that rejects a hundred dense candidates
    can report zero withheld, which is why a run record carrying only that total
    says nothing about whether the gate is doing work.

    The block's presence is not evidence that the gate ran: the payload carries
    it for every method, so a BM25 search reports a gate block of zeros. The
    method says which half was ranked, and a method that never runs the dense half
    reports ``None`` here rather than a zero nobody measured. ``eligible_total`` is
    the denominator a share rejected needs, and the counts are checked against
    each other here: a gate whose counts do not account for its own eligible
    candidates has not measured anything a reader could use.
    """

    gate = payload.get("dense_gate")
    gate = gate if isinstance(gate, dict) else {}
    ran = str(payload.get("retrieval_method") or "") in DENSE_METHODS
    counts = {
        "eligible_total": gate.get("eligible_total"),
        "admitted_above_floor": gate.get("admitted_above_floor"),
        "admitted_below_floor": gate.get("admitted_below_floor"),
        "rejected_below_floor": gate.get("rejected_below_floor"),
    }
    complete = (
        ran
        and "best_cosine_similarity" in gate
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in counts.values()
        )
    )
    unmeasured = {
        "dense_gate_ran": ran,
        "dense_gate_counts_complete": False,
        "dense_eligible_total": None,
        "dense_admitted_above_floor": None,
        "dense_admitted_below_floor": None,
        "dense_rejected_below_floor": None,
        "dense_best_cosine_similarity": None,
        "dense_quality_excluded": None,
        "dense_conserved": None,
    }
    if not complete:
        return unmeasured
    if counts["eligible_total"] != (
        counts["admitted_above_floor"]
        + counts["admitted_below_floor"]
        + counts["rejected_below_floor"]
    ):
        raise EvaluationError(
            "the cosine gate's counts do not account for its eligible candidates: "
            f"{counts['eligible_total']} eligible, "
            f"{counts['admitted_above_floor']} admitted above the floor, "
            f"{counts['admitted_below_floor']} admitted below it, "
            f"{counts['rejected_below_floor']} rejected"
        )
    return {
        "dense_gate_ran": True,
        "dense_gate_counts_complete": True,
        "dense_eligible_total": counts["eligible_total"],
        "dense_admitted_above_floor": counts["admitted_above_floor"],
        "dense_admitted_below_floor": counts["admitted_below_floor"],
        "dense_rejected_below_floor": counts["rejected_below_floor"],
        "dense_best_cosine_similarity": gate.get("best_cosine_similarity"),
        "dense_quality_excluded": gate.get("excluded_before_gate"),
        "dense_conserved": gate.get("conserved"),
    }


def _trace_measures(payload: dict[str, Any], *, target_chunk_id: str) -> dict[str, Any]:
    """Where the judged passage was, at each stage, before and after each gate.

    A target ranked fifth and a target the cosine gate dropped are the same number
    in every quality column and are not the same event, so the pipeline's own
    per-stage identifier lists are what tell them apart, and a run can be compared
    with the same query's unranked fusion over the same passages. Where a stage did
    not run, or its list was cut before the target's place in it, the answer is
    ``None``: an identifier absent from a truncated list was not necessarily absent
    from the stage, and a measure that cannot tell has not measured it.
    """

    trace = payload.get("evaluation_trace")
    trace = trace if isinstance(trace, dict) else {}
    stages = trace.get("stages")
    if not isinstance(stages, dict):
        return {
            "evaluation_trace_available": False,
            "trace_candidate_budget": None,
            "target_stage_presence": {},
            "evaluation_trace": trace or None,
        }
    presence: dict[str, bool | None] = {}
    for stage, entry in stages.items():
        if not isinstance(entry, dict):
            presence[stage] = None
            continue
        listed = {str(item) for item in (entry.get("chunk_ids") or [])}
        if target_chunk_id in listed:
            presence[stage] = True
        else:
            presence[stage] = None if entry.get("truncated") else False
    return {
        "evaluation_trace_available": True,
        "trace_candidate_budget": trace.get("candidate_budget"),
        "target_stage_presence": presence,
        # The pipeline's own account, kept whole so a reader can audit the stage
        # memberships and their truncation without re-running the search. It
        # holds identifiers and counts only: no passage text, and every list in
        # it is bounded by the engine with its truncation flag beside it.
        "evaluation_trace": trace,
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
        "schema_version": REPORT_SCHEMA_VERSION,
        "harness": harness_provenance(),
        "lexical_containment": {
            "threshold": LEXICAL_CONTAINMENT_THRESHOLD,
            "shingle_size": LEXICAL_SHINGLE_SIZE,
            "denominator": (
                "the number of ordered word shingles in the shorter slot, so the "
                "count describes the slot being contained rather than the passage "
                "containing it"
            ),
            "measure": (
                "ordered word shingles shared between two returned passages; a "
                "lexical overlap and not a claim that two passages make the same "
                "point"
            ),
            "protected_distinctions": [
                "numbers, with sign and decimal separator as written",
                "operator runs other than a lone hyphen",
                "negation words and un- and non- prefixes",
            ],
            "excluded_from_candidates": (
                "a pair whose protected signature differs is never offered to the "
                "measure, and only the first passage of an exact-equality group is "
                "compared at all"
            ),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "superseded_metrics": SUPERSEDED_METRICS,
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
            "allow_degraded_rerank": bool(args.allow_degraded_rerank),
            # Applied at query time over already-ranked candidates, so the
            # generation's recorded policy cannot carry it and a report that
            # measures it must name it here.
            "selection_policy": {
                "method": "greedy_source_diversity",
                "source_diversity_penalty": config.settings.source_diversity_penalty,
            },
            # What the resolved settings decide, as the engine hashes it, beside
            # the values the run cannot be reproduced without: a report that
            # names no policy is a report no other run can be compared with.
            "retrieval_policy_fingerprint": retrieval_policy_fingerprint(
                config.settings
            ),
            "resolved_retrieval_settings": {
                "duplicate_cosine": config.settings.duplicate_cosine,
                "rrf_k": config.settings.rrf_k,
                "bm25_weight": config.settings.bm25_weight,
                "dense_weight": config.settings.dense_weight,
                "minimum_candidates": config.settings.minimum_candidates,
                "maximum_candidates": config.settings.maximum_candidates,
                "dense_minimum_cosine_similarity": (
                    config.settings.dense_minimum_cosine_similarity
                ),
                "dense_relative_similarity_margin": (
                    config.settings.dense_relative_similarity_margin
                ),
                "rerank_max_candidates": config.settings.rerank_max_candidates,
                "rerank_window_multiple": config.settings.rerank_window_multiple,
                "rerank_window_floor": config.settings.rerank_window_floor,
                "prf": config.settings.prf,
                "prf_documents": config.settings.prf_documents,
                "prf_terms": config.settings.prf_terms,
                "minimum_passage_words": config.settings.minimum_passage_words,
                "minimum_passage_token_fraction": (
                    config.settings.minimum_passage_token_fraction
                ),
                "maximum_withheld_examples": config.settings.maximum_withheld_examples,
            },
        },
        "no_answer_support": no_answer_support(payload),
        "notice": (
            "Judgment resolution reads the generation's canonical chunks.jsonl "
            "read-only, and this harness never writes inside the project. "
            "Retrieval runs through ResearchService.search in process, which is "
            "the engine the workspace serves, because every search there is "
            "hybrid-only and this harness also measures BM25 and dense. Every "
            "search asks for the pipeline's evaluation trace, which changes no "
            "ranking decision."
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
        # The result-list measures compare returned passages against each other,
        # so they need the canonical text and the source each chunk came from. The
        # generation was already read for target resolution; this is that same read
        # indexed, not a second one.
        chunk_index = {
            str(chunk.get("chunk_id")): chunk
            for chunk in chunks
            if chunk.get("chunk_id")
        }
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

        # One family per target, resolved across both declarations at once, so
        # every mode's row compares the same queries against the same families.
        families = resolve_families(queries, resolved)
        report["settings"]["family_count"] = len(set(families.values()))

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
            allow_degraded=True,
            family_id=families.get(str(warm_target["target_id"]), ""),
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
                    chunk_index=chunk_index,
                    allow_degraded=args.allow_degraded_rerank,
                    family_id=families.get(str(target["target_id"]), ""),
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
                        chunk_index=chunk_index,
                        allow_degraded=args.allow_degraded_rerank,
                        family_id=families.get(str(query["target_id"]), ""),
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
        **latency_percentiles(runs + deep_runs),
    }

    degraded = [
        {
            "query_id": run["query_id"],
            "mode": run["mode"],
            "reasons": run["degraded_reasons"],
        }
        for run in runs + deep_runs
        if run.get("degraded_reasons")
    ]
    if degraded:
        # Only reachable with --allow-degraded-rerank, since otherwise the first
        # such run raises. The affected rows are named rather than averaged in.
        report["degraded"] = {
            "allowed_by": "--allow-degraded-rerank",
            "run_count": len(degraded),
            "runs": degraded[:REPORT_LIST_LIMIT],
            "runs_truncated": len(degraded) > REPORT_LIST_LIMIT,
            "note": (
                "A row recorded here is not the configuration its label names. A "
                "reranked row that fell back is an unranked candidate order, so "
                "its ranking columns measure fusion rather than the cross-encoder."
            ),
        }
    uncovered = [
        run
        for run in runs + deep_runs
        if not run["duplicate_measure_coverage"]["complete"]
    ]
    if uncovered:
        missing = sum(
            run["duplicate_measure_coverage"]["slots_missing_text"] for run in uncovered
        )
        print(
            f"WARNING: {len(uncovered)} of {len(runs) + len(deep_runs)} result lists "
            f"held {missing} passages this generation does not carry. Duplication "
            "measures are null for those queries, not zero."
        )

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
