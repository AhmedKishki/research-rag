"""Audit the repetition collapse against controlled contracts, and report what it did.

A pure diagnostic. It reads two engine functions, one gate, and one eligibility
helper, and runs them over contracts written as data in this file. Every contract
states, in the words of a person, what two synthetic passages mean and whether the
engine kept both. The report says what the engine did. The audit names no preferred
policy and changes none: the shipped thresholds, the length floors, and the source
metadata are read as they ship and are never set here.

Scope is a synthetic controlled probe. Nothing is embedded, ranked, or indexed: the
vectors below are arbitrary coordinates chosen by hand, so a pair that the cosine
path merges is a property of that choice, not a measurement of any model, any
corpus, or any collection a reader owns. A vector deliberately made equal to another
carries the input tag ``embedding_confusable`` and states in its own record that the
collision was constructed. No count here is an incidence rate.

Exit codes:

* 0 -- every contract was met.
* 1 -- at least one contract was violated. The record is still written.
* 2 -- the probe could not be set up, so nothing was measured.
* 3 -- no contract was violated, but at least one was undecidable.

    uv run python scripts/audit_collapse_safety.py --report /tmp/collapse.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from research_rag.project.support import _passage_equality_key
from research_rag.retrieval.search import SearchWorkflow, _collapse_repetitions

#: Report shape version. Version 1 is the first report this script has written.
REPORT_VERSION = 1

#: How many identifiers one probe stage may list. The bound keeps a stage from
#: growing a record without limit, and ``truncated`` says whether it was cut.
PROBE_TRACE_BUDGET = 64

#: Keys every case record carries. A record missing one of them cannot be read as an
#: observation, so a missing key is itself a setup failure rather than a contract.
CASE_RECORD_KEYS = (
    "name",
    "expected_preserve",
    "expected_collapse",
    "observed_representative_ids",
    "collapsed_pairs_provenance",
    "pass",
    "category",
    "reason",
    "evidence_kind",
)

#: The interface this probe was written against. A difference is reported as
#: undecidable rather than as a contract the engine failed.
COLLAPSE_PARAMETERS = (
    "ordered_ids",
    "chunks_by_id",
    "documents_by_id",
    "vectors",
    "threshold",
)
FILTER_PARAMETERS = (
    "categories_any",
    "keywords",
    "projects_any",
    "languages_any",
    "authors_any",
    "titles_any",
    "document_filter",
    "excluded_document_ids",
    "excluded_chunk_ids",
)

#: What the tag on a record means, stated once so a reader does not have to infer it
#: from a case name. The tags describe the probe's own inputs.
INPUT_TAG_GLOSSARY = {
    "embedding_confusable": (
        "the probe gave two passages the same, or a near, synthetic vector. That "
        "makes the cosine path able to merge them whatever the words say. It is a "
        "property of this input, not evidence that a retrieval model embeds these "
        "sentences alike, and not a statement about any corpus."
    ),
    "words_key_operator_collision": (
        "the two passages reach the same equality key because the key keeps "
        "alphanumeric runs and drops every operator between them. The collision is "
        "deterministic: it happens with no vector at all."
    ),
    "words_key_casefold_collision": (
        "the two passages reach the same equality key because the key casefolds. "
        "Case carries meaning in a symbol; the key cannot see it."
    ),
    "exact_text_reprint": (
        "the two passages are byte-identical after the engine reads their text, so "
        "one is a reprint of the other and both copies stay in the corpus."
    ),
    "distinct_synthetic_vectors": (
        "the probe gave the two passages orthogonal synthetic vectors, so the cosine "
        "path cannot merge them and only the equality key could."
    ),
    "exclusion_gate": (
        "one copy is withheld by a reviewed exclusion before the collapse runs, "
        "which is the order the pipeline applies: a gate drops a candidate, and the "
        "collapse settles only what survived."
    ),
    "shipped_defaults_unchanged": (
        "the length floors were read as they ship and were not set here, so this "
        "case reports the engine's own verdict rather than a floor's effect."
    ),
}

#: What this run does not measure, listed rather than left to the reader.
UNMEASURED = (
    (
        "No embedding model was loaded and no vector was computed: every vector here "
        "is a hand-written coordinate."
    ),
    (
        "No index, generation, or project was opened, and no corpus was read, so no "
        "count is an incidence rate for any collection."
    ),
    (
        "No reranker ran and no ranking was scored: the contracts are about which "
        "candidates the collapse merges, not about order quality."
    ),
    (
        "The cosine threshold was read from the shipped settings and not varied, so "
        "the report says where the shipped number sits and nothing about other "
        "values."
    ),
    (
        "The length floor behaviour under a configured floor is unmeasured: the "
        "shipped default configures none, and this run does not add one."
    ),
)


class ProbeSetupError(RuntimeError):
    """The probe could not be set up, so no contract was measured."""


@dataclass(frozen=True)
class SyntheticPassage:
    """One synthetic passage: fixed identifiers, no corpus behind them."""

    chunk_id: str
    document_id: str
    source_id: str
    source_relative_path: str
    text: str
    authors: str = ""
    #: Arbitrary coordinates in a small space, or ``None`` for a passage the probe
    #: gives no vector, which is how a vector store without this row behaves.
    vector: tuple[float, ...] | None = None

    def chunk_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "contents": self.text,
        }

    def document_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_relative_path": self.source_relative_path,
            "authors": [self.authors] if self.authors else [],
        }


@dataclass(frozen=True)
class CaseGates:
    """The exclusions and the filter a case applies before the collapse."""

    excluded_chunk_ids: frozenset[str] = frozenset()
    excluded_document_ids: frozenset[str] = frozenset()
    document_filter: frozenset[str] = frozenset()
    authors_any: frozenset[str] = frozenset()
    apply: bool = True

    def declares_exclusions(self) -> bool:
        """Whether this case withholds a candidate before the collapse runs."""

        return bool(
            self.excluded_chunk_ids
            or self.excluded_document_ids
            or self.document_filter
            or self.authors_any
        )


@dataclass(frozen=True)
class ContractCase:
    """One contract: two or more synthetic passages and what they mean."""

    name: str
    category: str
    relation: str
    expectation: str
    expected_representative_ids: tuple[str, ...]
    passages: tuple[SyntheticPassage, ...]
    input_tags: tuple[str, ...] = ()
    gates: CaseGates = field(default_factory=CaseGates)
    apply_length_floor: bool = False
    require_provenance: bool = False
    note: str = ""
    unmeasured: tuple[str, ...] = ()


def _passage(
    suffix: str,
    text: str,
    *,
    source_suffix: str = "",
    authors: str = "",
    vector: tuple[float, ...] | None = None,
) -> SyntheticPassage:
    """A synthetic passage whose identifiers are fixed and readable."""

    source = source_suffix or suffix
    return SyntheticPassage(
        chunk_id=f"audit-chunk-{suffix}",
        document_id=f"audit-document-{source}",
        source_id=f"audit-source-{source}",
        source_relative_path=f"sources/audit-{source}.pdf",
        text=text,
        authors=authors,
        vector=vector,
    )


#: Three coordinates. Distinct vectors here are orthogonal, so their cosine is 0.0
#: and no shipped threshold reaches them.
_A, _B, _C = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
#: The same coordinate twice, and a coordinate a hair off it. Both reach a cosine of
#: about 1.0, which is above the shipped threshold. They say nothing about a model.
_IDENTICAL_VECTOR = _A
_NEAR_VECTOR = (1.0, 0.02, 0.0)

_SHARED_LEAD = "The survey enrolled 512 respondents aged 18 to 24 across three cities. "

CONTRACT_CASES: tuple[ContractCase, ...] = (
    ContractCase(
        name="exact_reprint_keeps_both_copies_in_the_record",
        category="allowed_collapse_with_provenance",
        relation=(
            "The two passages are the same sentence printed in two files. They are "
            "one statement, and the second copy is a reprint of the first."
        ),
        expectation="collapse",
        expected_representative_ids=("audit-chunk-copy-alpha",),
        passages=(
            _passage(
                "copy-alpha",
                "Recovery reached 5% in the cohort after the second wave.",
                authors="R. Vance",
                vector=_A,
            ),
            _passage(
                "copy-beta",
                "Recovery reached 5% in the cohort after the second wave.",
                authors="R. Vance",
                vector=_A,
            ),
        ),
        input_tags=("exact_text_reprint",),
        require_provenance=True,
        note=(
            "A collapse here is the rule working: one representative is shown and "
            "both copies stay in the corpus, so the record has to name both."
        ),
    ),
    ContractCase(
        name="same_text_independent_authors_retain_the_other_source",
        category="allowed_collapse_with_provenance",
        relation=(
            "Two teams printed the same sentence in their own files under their own "
            "names. The words are one statement; the provenance is two sources."
        ),
        expectation="collapse",
        expected_representative_ids=("audit-chunk-indep-alpha",),
        passages=(
            _passage(
                "indep-alpha",
                "Wage growth accelerated in the northern sector after 2021.",
                authors="A. Mensah",
                vector=_A,
            ),
            _passage(
                "indep-beta",
                "Wage growth accelerated in the northern sector after 2021.",
                authors="B. Oyelaran",
                vector=_A,
            ),
        ),
        input_tags=("exact_text_reprint",),
        require_provenance=True,
        note=(
            "The collapsed pair names the second file as well as the first, so the "
            "copy that is not shown is still nameable by the record."
        ),
    ),
    ContractCase(
        name="sign_flips_the_direction_of_a_number",
        category="equality_key_operator_collision",
        relation=(
            "One passage reports a rise of 5 units and the other a fall of 5 units. "
            "The claims are opposites."
        ),
        expectation="preserve",
        expected_representative_ids=("audit-chunk-sign-plus", "audit-chunk-sign-minus"),
        passages=(
            _passage(
                "sign-plus",
                "The fitted slope is +5 units per decade.",
                vector=_A,
            ),
            _passage(
                "sign-minus",
                "The fitted slope is -5 units per decade.",
                vector=_B,
            ),
        ),
        input_tags=("words_key_operator_collision",),
        note=(
            "The equality key keeps digits and drops the sign, so both passages "
            "reach one key. The collision needs no vector and no model: it is a "
            "property of the key function."
        ),
    ),
    ContractCase(
        name="percent_sign_and_bare_number_share_one_key",
        category="equality_key_operator_collision",
        relation=(
            "One passage reports a proportion and the other a count of the same "
            "digits. A share and a count are different claims."
        ),
        expectation="preserve",
        expected_representative_ids=(
            "audit-chunk-rate-percent",
            "audit-chunk-rate-bare",
        ),
        passages=(
            _passage(
                "rate-percent",
                "Recovery reached 5% in the cohort after the second wave.",
                vector=_A,
            ),
            _passage(
                "rate-bare",
                "Recovery reached 5 in the cohort after the second wave.",
                vector=_B,
            ),
        ),
        input_tags=("words_key_operator_collision",),
        note=(
            "The percent sign is not an alphanumeric run, so the key cannot tell a "
            "share from the same digits without one."
        ),
    ),
    ContractCase(
        name="factorial_and_product_share_one_key",
        category="equality_key_operator_collision",
        relation=(
            "One passage names a factorial term and the other the plain number in the "
            "expansion. The values differ by every factor above it."
        ),
        expectation="preserve",
        expected_representative_ids=("audit-chunk-factorial", "audit-chunk-product"),
        passages=(
            _passage(
                "factorial",
                "The expansion contains the term 5! for this order.",
                vector=_A,
            ),
            _passage(
                "product",
                "The expansion contains the term 5 for this order.",
                vector=_B,
            ),
        ),
        input_tags=("words_key_operator_collision",),
        note=("An operator between digits is dropped, so the two reach one key."),
    ),
    ContractCase(
        name="case_sensitive_symbol_shares_one_key",
        category="equality_key_casefold_collision",
        relation=(
            "One passage reports a probability and the other a pressure under the "
            "same letter. The symbol is case sensitive, so the claims differ."
        ),
        expectation="preserve",
        expected_representative_ids=(
            "audit-chunk-symbol-lower",
            "audit-chunk-symbol-upper",
        ),
        passages=(
            _passage(
                "symbol-lower",
                "The value of p is reported for every site.",
                vector=_A,
            ),
            _passage(
                "symbol-upper",
                "The value of P is reported for every site.",
                vector=_B,
            ),
        ),
        input_tags=("words_key_casefold_collision",),
        note=(
            "Casefolding is deliberate for prose and is invisible to a symbol, so "
            "the two reach one key."
        ),
    ),
    ContractCase(
        name="negated_and_asserted_claims_with_equal_vectors",
        category="conditional_cosine_merge",
        relation=(
            "One passage asserts that the intervention reduced read time and the "
            "other denies it. The two cannot both be true."
        ),
        expectation="preserve",
        expected_representative_ids=("audit-chunk-negated", "audit-chunk-asserted"),
        passages=(
            _passage(
                "negated",
                "The intervention reduced median read time in every site.",
                vector=_IDENTICAL_VECTOR,
            ),
            _passage(
                "asserted",
                "The intervention did not reduce median read time in any site.",
                vector=_IDENTICAL_VECTOR,
            ),
        ),
        input_tags=("embedding_confusable",),
        note=(
            "The words differ, so the equality key keeps both; the probe then gave "
            "both the same synthetic vector, which is what lets the cosine path "
            "merge them. The vector is a constructed input."
        ),
        unmeasured=(
            (
                "how often two opposite statements would receive embeddings this "
                "close in any model is not measured here"
            ),
        ),
    ),
    ContractCase(
        name="independent_claims_from_different_authors_with_equal_vectors",
        category="conditional_cosine_merge",
        relation=(
            "Two independent studies, by different authors, report different "
            "findings about the same sector. Merging them states one of them twice."
        ),
        expectation="preserve",
        expected_representative_ids=(
            "audit-chunk-claim-alpha",
            "audit-chunk-claim-beta",
        ),
        passages=(
            _passage(
                "claim-alpha",
                "Order volumes rose in the northern corridor last year.",
                authors="A. Mensah",
                vector=_IDENTICAL_VECTOR,
            ),
            _passage(
                "claim-beta",
                "Order volumes fell in the southern corridor last year.",
                authors="B. Oyelaran",
                vector=_IDENTICAL_VECTOR,
            ),
        ),
        input_tags=("embedding_confusable", "exact_text_reprint"),
        note=(
            "Different findings from different sources reach one representative "
            "when the probe's vectors are equal; the texts and the authors are not "
            "what the cosine path reads."
        ),
        unmeasured=(
            (
                "whether a retrieval model would place these two findings this close "
                "is not measured here"
            ),
        ),
    ),
    ContractCase(
        name="independent_claims_with_distinct_vectors_survive",
        category="meaning_distinct",
        relation=(
            "The same two independent findings, with vectors the probe kept apart, "
            "so the words key and the cosine path both leave them alone."
        ),
        expectation="preserve",
        expected_representative_ids=(
            "audit-chunk-claim-alpha",
            "audit-chunk-claim-beta",
        ),
        passages=(
            _passage(
                "claim-alpha",
                "Order volumes rose in the northern corridor last year.",
                authors="A. Mensah",
                vector=_A,
            ),
            _passage(
                "claim-beta",
                "Order volumes fell in the southern corridor last year.",
                authors="B. Oyelaran",
                vector=_B,
            ),
        ),
        input_tags=("distinct_synthetic_vectors",),
        note=(
            "This record is the control for the record above it: same passages, "
            "same expectation, vectors the probe did not make equal."
        ),
    ),
    ContractCase(
        name="excluded_copy_does_not_suppress_the_allowed_copy",
        category="gate_before_collapse",
        relation=(
            "The higher-ranked of two identical copies is withheld by a reviewed "
            "passage exclusion. The allowed copy still has to be answerable."
        ),
        expectation="preserve",
        expected_representative_ids=("audit-chunk-copy-allowed",),
        passages=(
            _passage(
                "copy-excluded",
                "Recovery reached 5% in the cohort after the second wave.",
                vector=_A,
            ),
            _passage(
                "copy-allowed",
                "Recovery reached 5% in the cohort after the second wave.",
                vector=_A,
            ),
        ),
        input_tags=("exclusion_gate",),
        gates=CaseGates(excluded_chunk_ids=frozenset({"audit-chunk-copy-excluded"})),
        note=(
            "The gate runs before the collapse, so the withheld copy is not a "
            "survivor for the allowed copy to be merged into."
        ),
    ),
    ContractCase(
        name="excluded_source_copy_does_not_suppress_the_allowed_copy",
        category="gate_before_collapse",
        relation=(
            "The higher-ranked copy sits in a source under a reviewed exclusion. "
            "The copy in the allowed source still has to be answerable."
        ),
        expectation="preserve",
        expected_representative_ids=("audit-chunk-src-allowed",),
        passages=(
            _passage(
                "src-excluded",
                "Wage growth accelerated in the northern sector after 2021.",
                source_suffix="excluded",
                vector=_A,
            ),
            _passage(
                "src-allowed",
                "Wage growth accelerated in the northern sector after 2021.",
                source_suffix="allowed",
                vector=_A,
            ),
        ),
        input_tags=("exclusion_gate",),
        gates=CaseGates(excluded_document_ids=frozenset({"audit-document-excluded"})),
        note=(
            "A source exclusion drops the whole document before the collapse sees "
            "it, which is the same ordering as the passage exclusion above."
        ),
    ),
)


def _repo_root() -> Path | None:
    """The checkout holding this script, found from the package, never assumed."""

    here = Path(str(resources.files("research_rag"))).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _relative(path: Path) -> str:
    """A path relative to the checkout, so a record carries no absolute path."""

    root = _repo_root()
    if root is not None:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_revision(root: Path | None) -> dict[str, Any]:
    """The commit this checkout is at, read only, and nothing when git is absent."""

    if root is None:
        return {"available": False, "reason": "no_git_checkout_found"}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "reason": "git_unavailable"}
    revision = head.stdout.strip()
    if head.returncode != 0 or not revision:
        return {"available": False, "reason": "git_rev_parse_failed"}
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/research_rag"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    modified = [line[3:] for line in changed.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": revision,
        "engine_files_modified_in_working_tree": modified,
    }


def engine_identity() -> dict[str, Any]:
    """What ran: the files, their content hashes, and the signatures read here.

    Two records from two checkouts can be compared only if each says which code
    produced it, so every function is named by its module, its file's content
    hash, and the signature this run read.
    """

    files: dict[str, Any] = {}
    for label, target in (
        ("research_rag.retrieval.search._collapse_repetitions", _collapse_repetitions),
        ("research_rag.project.support._passage_equality_key", _passage_equality_key),
        (
            "research_rag.retrieval.search.SearchWorkflow._matches_filters",
            SearchWorkflow._matches_filters,
        ),
        (
            "research_rag.retrieval.search.SearchWorkflow._passage_too_short",
            SearchWorkflow._passage_too_short,
        ),
    ):
        source_file = Path(inspect.getfile(target))
        files[label] = {
            "path": _relative(source_file),
            "file_sha256": _sha256(source_file.read_bytes()),
            "definition_sha256": _sha256(inspect.getsource(target).encode("utf-8")),
            "signature": str(inspect.signature(target)),
        }
    return {
        "functions": files,
        "revision": _git_revision(_repo_root()),
        "note": (
            "The content hashes identify the code these observations came from. Two "
            "reports are comparable only when the hashes agree; a record is not "
            "evidence about a different checkout."
        ),
    }


def engine_interface_problems() -> list[str]:
    """Where this probe's interface and the engine's differ, as readable names."""

    problems: list[str] = []
    collapse = inspect.signature(_collapse_repetitions).parameters
    for name in COLLAPSE_PARAMETERS:
        if name not in collapse:
            problems.append(f"_collapse_repetitions has no parameter {name!r}")
    try:
        gate = inspect.signature(SearchWorkflow._matches_filters).parameters
    except (TypeError, ValueError) as exc:
        problems.append(f"_matches_filters has no readable signature: {exc}")
    else:
        for name in FILTER_PARAMETERS:
            if name not in gate:
                problems.append(f"_matches_filters has no parameter {name!r}")
    try:
        equality_parameters = inspect.signature(_passage_equality_key).parameters
    except (TypeError, ValueError) as exc:
        problems.append(f"_passage_equality_key has no readable signature: {exc}")
    else:
        if "text" not in equality_parameters:
            problems.append("_passage_equality_key has no parameter 'text'")
    return problems


def shipped_settings() -> dict[str, Any]:
    """The retrieval settings as they ship, read from the shipped file."""

    path = Path(str(resources.files("research_rag.project") / "default.toml"))
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProbeSetupError(f"shipped settings could not be read: {exc}") from exc
    retrieval = parsed.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ProbeSetupError("shipped settings record no [retrieval] block")
    missing = [
        name
        for name in ("duplicate_cosine", "minimum_passage_words")
        if name not in retrieval
    ]
    if missing:
        raise ProbeSetupError(f"shipped settings omit retrieval.{missing}")
    return {
        "source": _relative(path),
        "content_sha256": _sha256(path.read_bytes()),
        "retrieval": {
            "duplicate_cosine": retrieval["duplicate_cosine"],
            "minimum_passage_words": retrieval["minimum_passage_words"],
            "minimum_passage_token_fraction": retrieval.get(
                "minimum_passage_token_fraction"
            ),
        },
        "note": (
            "Read as they ship and used unchanged. No threshold and no floor was "
            "set for this run, so every record describes the shipped numbers."
        ),
    }


def _vectors_for(
    passages: Sequence[SyntheticPassage],
) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float32]]], int | None]:
    """The probe's vectors, keyed the way the engine looks them up: by text."""

    dimension: int | None = None
    vectors: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for item in passages:
        if item.vector is None:
            continue
        if dimension is None:
            dimension = len(item.vector)
        elif len(item.vector) != dimension:
            raise ProbeSetupError(
                f"case vector dimensions disagree: {item.chunk_id} has "
                f"{len(item.vector)}, the case established {dimension}"
            )
        array = np.asarray(item.vector, dtype=np.float32)
        if not bool(np.all(np.isfinite(array))):
            raise ProbeSetupError(f"{item.chunk_id} carries a non-finite coordinate")
        if float(np.linalg.norm(array)) == 0.0:
            raise ProbeSetupError(f"{item.chunk_id} carries a zero vector")
        vectors[item.text] = array
    return vectors, dimension


def _settings_stub(settings: Mapping[str, Any]) -> SimpleNamespace:
    """The two settings the eligibility helpers read, from the shipped values.

    The helpers are methods on the workflow because a search owns them. Only
    ``config.settings`` is reached, and only these two fields, so the probe
    supplies those and nothing else.
    """

    retrieval = settings["retrieval"]
    return SimpleNamespace(
        config=SimpleNamespace(
            settings=SimpleNamespace(
                minimum_passage_words=retrieval["minimum_passage_words"],
                minimum_passage_token_fraction=retrieval.get(
                    "minimum_passage_token_fraction"
                ),
            )
        )
    )


def _stage(identifiers: Sequence[str]) -> dict[str, Any]:
    """One probe stage: its own size, and a bounded copy of what it held."""

    listed = list(identifiers[:PROBE_TRACE_BUDGET])
    return {
        "count": len(identifiers),
        "ids": listed,
        "truncated": len(listed) < len(identifiers),
    }


def _pair_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One collapsed pair, with both copies named and their sources attached."""

    record = {
        "collapsed_by": str(entry.get("collapsed_by") or ""),
        "chunk_id": str(entry.get("chunk_id") or ""),
        "source_id": str(entry.get("source_id") or ""),
        "source_relative_path": str(entry.get("source_relative_path") or ""),
        "repeated_chunk_id": str(entry.get("repeated_chunk_id") or ""),
        "repeated_source_id": str(entry.get("repeated_source_id") or ""),
        "repeated_source_relative_path": str(
            entry.get("repeated_source_relative_path") or ""
        ),
    }
    if "similarity" in entry:
        record["similarity"] = entry["similarity"]
    return record


def _provenance_complete(pairs: Sequence[Mapping[str, Any]]) -> bool:
    """Every collapsed pair names both copies, each with a source of its own."""

    if not pairs:
        return False
    for pair in pairs:
        named = (
            pair["chunk_id"],
            pair["source_id"],
            pair["source_relative_path"],
            pair["repeated_chunk_id"],
            pair["repeated_source_id"],
            pair["repeated_source_relative_path"],
        )
        if not all(named):
            return False
        if (
            not pair["repeated_chunk_id"]
            or pair["repeated_chunk_id"] == pair["chunk_id"]
        ):
            return False
    return True


def run_case(
    case: ContractCase,
    *,
    settings: Mapping[str, Any],
    undecided_reason: str | None = None,
) -> dict[str, Any]:
    """Run one contract through the engine's own functions and record what came out."""

    engine_path = ["research_rag.retrieval.search._collapse_repetitions"]
    if undecided_reason is not None:
        return _undecided_record(case, undecided_reason, engine_path)

    passages = case.passages
    chunks_by_id = {item.chunk_id: item.chunk_record() for item in passages}
    documents_by_id = {item.document_id: item.document_record() for item in passages}
    ordered_ids = [item.chunk_id for item in passages]
    try:
        vectors, dimension = _vectors_for(passages)
    except ProbeSetupError as exc:
        raise ProbeSetupError(f"case {case.name}: {exc}") from exc

    gates = case.gates
    eligible_ids = list(ordered_ids)
    gate_excluded_ids: list[str] = []
    if gates.apply:
        engine_path.insert(
            0, "research_rag.retrieval.search.SearchWorkflow._matches_filters"
        )
        eligible_ids = [
            item.chunk_id
            for item in passages
            if SearchWorkflow._matches_filters(
                item.chunk_record(),
                documents_by_id,
                categories_any=set(),
                keywords=set(),
                projects_any=set(),
                languages_any=set(),
                authors_any=set(gates.authors_any),
                titles_any=set(),
                document_filter=set(gates.document_filter),
                excluded_document_ids=set(gates.excluded_document_ids),
                excluded_chunk_ids=set(gates.excluded_chunk_ids),
            )
        ]
        gate_excluded_ids = [
            item for item in ordered_ids if item not in set(eligible_ids)
        ]

    length_verdicts: dict[str, bool] = {}
    length_policy: dict[str, Any] = {}
    if case.apply_length_floor:
        engine_path += [
            "research_rag.retrieval.search.SearchWorkflow._passage_token_policy",
            "research_rag.retrieval.search.SearchWorkflow._passage_too_short",
        ]
        stub = _settings_stub(settings)
        # A synthetic manifest: the policy reads the chunk size and tokenizer the
        # generation recorded, and no generation exists in this probe.
        length_policy = SearchWorkflow._passage_token_policy(  # type: ignore[arg-type]
            stub,
            {"chunking": {"chunk_size": 512, "tokenizer": "gpt2"}},
        )
        length_verdicts = {
            item.chunk_id: bool(
                SearchWorkflow._passage_too_short(  # type: ignore[arg-type]
                    stub,
                    item.chunk_record(),
                    token_policy=length_policy,
                )
            )
            for item in passages
            if item.chunk_id in set(eligible_ids)
        }
        withheld = [name for name, too_short in length_verdicts.items() if too_short]
        if withheld:
            eligible_ids = [name for name in eligible_ids if name not in set(withheld)]

    threshold = float(settings["retrieval"]["duplicate_cosine"])
    kept, collapsed = _collapse_repetitions(
        eligible_ids,
        chunks_by_id=chunks_by_id,
        documents_by_id=documents_by_id,
        vectors=vectors,
        threshold=threshold,
    )
    pairs = [_pair_record(entry) for entry in collapsed]

    expected_ids = tuple(case.expected_representative_ids)
    observed_ids = tuple(kept)
    provenance_ok = not case.require_provenance or _provenance_complete(pairs)
    matched = observed_ids == expected_ids and provenance_ok
    if case.expectation == "preserve":
        expectation_met = matched
    elif case.expectation == "collapse":
        expectation_met = matched and bool(pairs)
    else:
        return _undecided_record(
            case,
            f"the fixture states no decidable expectation: {case.expectation!r}",
            engine_path,
        )

    keys = {item.chunk_id: _passage_equality_key(item.text) for item in passages}
    merged_by = sorted({pair["collapsed_by"] for pair in pairs})
    status = "met" if expectation_met else "violated"
    record: dict[str, Any] = {
        "name": case.name,
        "category": case.category,
        "relation": case.relation,
        "expected_preserve": case.expectation == "preserve",
        "expected_collapse": case.expectation == "collapse",
        "expected_representative_ids": list(expected_ids),
        "observed_representative_ids": list(observed_ids),
        "collapsed_pairs_provenance": pairs,
        "pass": expectation_met,
        "status": status,
        "evidence_kind": (
            "engine_functions_shipped_defaults"
            if case.apply_length_floor
            else "engine_functions_pipeline_order"
            if case.gates.declares_exclusions()
            else "engine_functions_direct"
        ),
        "reason": _reason(
            case,
            status=status,
            observed_ids=observed_ids,
            merged_by=merged_by,
            keys=keys,
            provenance_ok=provenance_ok,
        ),
        "engine_call_path": engine_path,
        "input_tags": list(case.input_tags),
        "input_tag_notes": [
            INPUT_TAG_GLOSSARY[tag]
            for tag in case.input_tags
            if tag in INPUT_TAG_GLOSSARY
        ],
        "note": case.note,
        "unmeasured": list(case.unmeasured),
        "probe_budget": PROBE_TRACE_BUDGET,
        "probe_trace": {
            "candidates_in": _stage(ordered_ids),
            "gate_withheld": _stage(gate_excluded_ids),
            "eligible_in": _stage(eligible_ids),
            "collapsed": _stage([pair["chunk_id"] for pair in pairs]),
            "representatives_out": _stage(observed_ids),
        },
        "observed": {
            "duplicate_cosine_threshold": threshold,
            "vector_dimension": dimension,
            "equality_keys": keys,
            "collapsed_by": merged_by,
            "distinct_surviving_sources": len(
                {chunks_by_id[name]["document_id"] for name in observed_ids}
            ),
            "gate_applied": gates.apply,
            "length_floor": {
                "policy": length_policy,
                "too_short": length_verdicts,
                "behaviour_under_a_configured_floor": (
                    "unmeasured" if case.apply_length_floor else "not_exercised"
                ),
            },
        },
    }
    missing = [key for key in CASE_RECORD_KEYS if key not in record]
    if missing:
        raise ProbeSetupError(f"case {case.name} produced no {missing}")
    return record


def _undecided_record(
    case: ContractCase,
    reason: str,
    engine_path: Sequence[str],
) -> dict[str, Any]:
    """A record that says the probe did not measure this case.

    An undecided case is not a pass. It is reported as ``unknown`` so a reader
    sees a gap where an observation would be, and it is counted apart from the
    contracts the engine failed.
    """

    return {
        "name": case.name,
        "category": case.category,
        "relation": case.relation,
        "expected_preserve": case.expectation == "preserve",
        "expected_collapse": case.expectation == "collapse",
        "expected_representative_ids": list(case.expected_representative_ids),
        "observed_representative_ids": [],
        "collapsed_pairs_provenance": [],
        "pass": False,
        "status": "unknown",
        "evidence_kind": "none",
        "reason": reason,
        "engine_call_path": list(engine_path),
        "input_tags": list(case.input_tags),
        "input_tag_notes": [
            INPUT_TAG_GLOSSARY[tag]
            for tag in case.input_tags
            if tag in INPUT_TAG_GLOSSARY
        ],
        "note": case.note,
        "unmeasured": list(case.unmeasured),
        "probe_budget": PROBE_TRACE_BUDGET,
        "probe_trace": {},
        "observed": {},
    }


def _reason(
    case: ContractCase,
    *,
    status: str,
    observed_ids: Sequence[str],
    merged_by: Sequence[str],
    keys: Mapping[str, str],
    provenance_ok: bool,
) -> str:
    """One sentence per record naming the mechanism, with no policy in it."""

    expected = ", ".join(case.expected_representative_ids)
    observed = ", ".join(observed_ids) or "no representative"
    dropped = ", ".join(merged_by) if merged_by else "no merge"
    if status == "met":
        verdict = (
            f"expected {expected}, observed {observed}"
            if case.expectation == "preserve"
            else (
                f"expected one representative {expected}, observed {observed} and a "
                f"merge by {dropped} with both copies named"
            )
        )
    else:
        verdict = (
            f"expected both copies to survive ({expected}), observed {observed}"
            f"{f' after a merge by {dropped}' if merged_by else ''}"
        )
    detail = ""
    if case.category in (
        "equality_key_operator_collision",
        "equality_key_casefold_collision",
    ):
        shared = len({key for key in keys.values()}) == 1
        detail = (
            "Both passages reach one equality key"
            f"{' (confirmed by this run)' if shared else ''}, so the collision is a "
            "property of the key function and needs no vector."
        )
    if case.gates.declares_exclusions():
        detail += (
            " The gate was the engine's own and it ran before the collapse, which is "
            "the order the pipeline applies; the retrieval window and the index were "
            "not exercised."
        )
    if not provenance_ok:
        detail += (
            " A pair is missing one copy or one source, so the copy that was not "
            "shown is not nameable from this record."
        )
    return f"{verdict}. {detail.strip()}".strip()


def build_report(cases: Sequence[ContractCase] | None = None) -> dict[str, Any]:
    """Run every contract and return the report, without writing anything."""

    selected = tuple(cases) if cases is not None else CONTRACT_CASES
    settings = shipped_settings()
    problems = engine_interface_problems()
    records = [
        run_case(
            case,
            settings=settings,
            undecided_reason=(
                "the engine interface differs from the one this probe was written "
                f"against: {'; '.join(problems)}"
                if problems
                else None
            ),
        )
        for case in selected
    ]
    violated = [item["name"] for item in records if item["status"] == "violated"]
    unknown = [item["name"] for item in records if item["status"] == "unknown"]
    suppressed = [
        item["name"]
        for item in records
        if item["expected_preserve"] and item["collapsed_pairs_provenance"]
    ]
    return {
        "report_version": REPORT_VERSION,
        "audit": "repetition_collapse_safety",
        "engine_identity": engine_identity(),
        "engine_interface_problems": problems,
        "scope": {
            "corpus": "synthetic_controlled_probe",
            "actual_corpus_measured": False,
            "embeddings": "synthetic_arbitrary_vectors",
            "models_loaded": False,
            "reranker_run": False,
            "index_opened": False,
            "network_used": False,
            "generation_artifacts_written": False,
            "settings_mutated": False,
            "note": (
                "Every passage here is written in this file and means what its "
                "record says it means. Nothing was measured against a real corpus, "
                "so no count in this report is an incidence rate."
            ),
        },
        "settings_read": settings,
        "input_tag_glossary": INPUT_TAG_GLOSSARY,
        "unmeasured": list(UNMEASURED),
        "contract_outcome": {
            "cases": len(records),
            "met": sum(1 for item in records if item["status"] == "met"),
            "violated": len(violated),
            "unknown": len(unknown),
            "violated_cases": violated,
            "unknown_cases": unknown,
            "false_suppression_pairs_synthetic_stage": len(suppressed),
            "false_suppression_scope": (
                "Counts the pairs this synthetic probe constructed and the engine "
                "merged, nothing else. It is not a rate, and it says nothing about "
                "any corpus."
            ),
            "false_suppression_cases": suppressed,
        },
        "code_unit_result": {
            "measured_by": "pytest",
            "in_this_report": False,
            "note": (
                "This report says what the engine did with these contracts. Whether "
                "the code unit tests pass is a separate result, reported by the test "
                "run and never merged into these counts."
            ),
        },
        "policy_recommendation": None,
        "policy_note": (
            "This run reports observations and names no preferred change. The "
            "shipped behaviour is the shipped behaviour here, including on every "
            "contract it fails."
        ),
        "cases": records,
    }


def _assert_report_path(path: Path) -> None:
    """Refuse a report path that would write into a project.

    The probe writes one file and it is the record. A path inside a project
    directory would put a diagnostic into a collection's own artifacts, so the
    path is refused before anything is written.
    """

    resolved = path if path.is_absolute() else Path.cwd() / path
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".research-rag").is_dir():
            raise ProbeSetupError(
                f"report path {path} lies inside a project directory ({parent}); "
                "the probe writes its record outside a project"
            )


def write_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Write the record, and refuse any path that is not outside a project."""

    _assert_report_path(path)
    resolved = path if path.is_absolute() else Path.cwd() / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    """The command line: one required record path, and nothing that mutates."""

    parser = argparse.ArgumentParser(
        prog="audit_collapse_safety",
        description=(
            "Run the repetition-collapse contracts and write the expected and "
            "observed result for each."
        ),
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="path for the JSON record, outside any project directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit. Returns the process exit code."""

    args = build_parser().parse_args(argv)
    try:
        report = build_report()
        destination = write_report(args.report, report)
    except ProbeSetupError as exc:
        print(f"probe setup failed: {exc}", file=sys.stderr)
        return 2

    outcome = report["contract_outcome"]
    print(
        f"cases={outcome['cases']} met={outcome['met']} "
        f"violated={outcome['violated']} unknown={outcome['unknown']}"
    )
    for item in report["cases"]:
        if item["status"] != "met":
            print(f"{item['status']}: {item['name']} — {item['reason']}")
    print(f"record: {_relative(destination)}")
    if outcome["violated"]:
        return 1
    if outcome["unknown"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
