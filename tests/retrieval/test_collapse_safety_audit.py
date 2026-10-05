"""The collapse-safety audit reports what the engine did, and reports it as it is.

The audit is a diagnostic, so the tests here read its record rather than assert a
policy. A contract the engine fails has to come out of it as a violation with the
mechanism named, and the controlled input the audit constructs on purpose has to
come out as a violation too: a run that turned those into passes would be a
diagnostic that hides what it found. The contracts that pass are asserted as well,
so a record cannot be green because nothing was checked.

Nothing here asserts that the shipped collapse behaviour is right. Each assertion
is about the audit: that it called the engine's own functions, that it kept
expectation and observation apart, that a stage it cannot decide is reported as
unknown rather than as a pass, and that a probe which cannot run says so instead of
emitting a record.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.support import _passage_equality_key
from research_rag.retrieval.search import _collapse_repetitions
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

#: The script is read by path, the way a person runs it, so these tests hold the
#: command's own behaviour rather than a package import of a module.
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_collapse_safety.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("audit_collapse_safety", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: a dataclass declared in it resolves its own
    # module through ``sys.modules``, and the script declares several.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()

#: Every key a record has to carry before a reader can treat it as an observation.
REQUIRED_KEYS = audit.CASE_RECORD_KEYS


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """One audit record, shared by the reads below. Building it runs no I/O."""

    return audit.build_report()


def _case(record: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in record["cases"] if item["name"] == name]
    assert len(matches) == 1, name
    return matches[0]


def test_the_script_is_importable_and_exposes_its_parser_and_report(
    report: dict[str, Any],
) -> None:
    """A person runs this as a command, so the two entry points have to import."""

    assert callable(audit.main)
    parser = audit.build_parser()
    assert parser.parse_args(["--report", "record.json"]).report == Path("record.json")
    # A record path is the one argument: the probe writes its record and nothing
    # else, so an argument that could point it at a corpus would be a way to run
    # it against something the probe does not measure.
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert sorted(action.dest for action in parser._actions) == ["help", "report"]
    assert report["report_version"] == audit.REPORT_VERSION


def test_every_record_carries_the_keys_a_reader_needs(report: dict[str, Any]) -> None:
    assert report["cases"]
    for item in report["cases"]:
        missing = [key for key in REQUIRED_KEYS if key not in item]
        assert not missing, (item.get("name"), missing)
        # Exactly one expectation is stated, so a record cannot claim both that
        # the copies were kept and that they were merged.
        assert [item["expected_preserve"], item["expected_collapse"]].count(
            True
        ) == 1, item["name"]
        assert item["pass"] is (item["status"] == "met"), item["name"]


def test_the_probe_calls_the_engine_functions_and_copies_neither(
    report: dict[str, Any],
) -> None:
    """The collapse and the key are the engine's, so a reader can go read them."""

    assert audit._collapse_repetitions is _collapse_repetitions
    assert audit._passage_equality_key is _passage_equality_key
    for item in report["cases"]:
        assert (
            "research_rag.retrieval.search._collapse_repetitions"
            in item["engine_call_path"]
        ), item["name"]
        assert item["relation"], item["name"]
    # The collapse is imported rather than written here, so the only collapse in
    # this file's chain is the engine's.
    assert inspect.getsource(audit).count("def _collapse_repetitions") == 0


def test_the_record_says_which_code_produced_it(report: dict[str, Any]) -> None:
    """Two records are comparable only when each names the code behind it."""

    identity = report["engine_identity"]
    for label, entry in identity["functions"].items():
        assert not Path(entry["path"]).is_absolute(), label
        assert len(entry["file_sha256"]) == 64, label
        assert len(entry["definition_sha256"]) == 64, label
        assert entry["signature"].startswith("("), label
    collapse = identity["functions"][
        "research_rag.retrieval.search._collapse_repetitions"
    ]
    # The hash is over the code this run read, so a reader can check the record
    # against the checkout rather than trust it.
    assert (
        collapse["definition_sha256"]
        == hashlib.sha256(
            inspect.getsource(_collapse_repetitions).encode("utf-8")
        ).hexdigest()
    )
    assert (
        collapse["file_sha256"]
        == hashlib.sha256(
            Path(inspect.getfile(_collapse_repetitions)).read_bytes()
        ).hexdigest()
    )
    assert "revision" in identity


def test_the_record_holds_no_absolute_path_and_no_wall_clock(
    report: dict[str, Any],
) -> None:
    rendered = json.dumps(report)
    assert str(Path.home()) not in rendered
    assert '"/' not in rendered
    # A diagnostic is a reading of one state of the code, so two runs over the same
    # code are the same document. A timestamp would make them differ for no reason
    # a reader could act on.
    assert audit.build_report() == report


def test_the_scope_says_synthetic_and_names_what_was_not_measured(
    report: dict[str, Any],
) -> None:
    scope = report["scope"]
    assert scope["corpus"] == "synthetic_controlled_probe"
    assert scope["actual_corpus_measured"] is False
    assert scope["models_loaded"] is False
    assert scope["embeddings"] == "synthetic_arbitrary_vectors"
    assert scope["reranker_run"] is False
    assert scope["index_opened"] is False
    assert scope["network_used"] is False
    assert scope["settings_mutated"] is False
    assert report["unmeasured"]
    # The settings are read as they ship, and the record carries the file's hash,
    # so a reader can tell which thresholds the observations belong to.
    settings = report["settings_read"]
    assert len(settings["content_sha256"]) == 64
    assert settings["retrieval"]["duplicate_cosine"] == pytest.approx(0.99)


def test_the_contracts_that_hold_are_reported_as_holding(
    report: dict[str, Any],
) -> None:
    """A record that only lists failures cannot be told from one that checks."""

    met = [item for item in report["cases"] if item["status"] == "met"]
    names = {item["name"] for item in met}
    assert "exact_reprint_keeps_both_copies_in_the_record" in names
    assert "independent_claims_with_distinct_vectors_survive" in names
    assert "excluded_copy_does_not_suppress_the_allowed_copy" in names
    assert "excluded_source_copy_does_not_suppress_the_allowed_copy" in names
    for item in met:
        assert (
            item["observed_representative_ids"] == (item["expected_representative_ids"])
        ), item["name"]


def test_an_operator_collision_is_reported_as_a_property_of_the_key(
    report: dict[str, Any],
) -> None:
    """The sign, the percent sign, and the operator are dropped by the key itself.

    No vector is involved, so the finding is deterministic: the two passages reach
    one key, and the run confirms it rather than inferring it.
    """

    for name in (
        "sign_flips_the_direction_of_a_number",
        "percent_sign_and_bare_number_share_one_key",
        "factorial_and_product_share_one_key",
    ):
        item = _case(report, name)
        assert item["category"] == "equality_key_operator_collision", name
        assert item["status"] == "violated", name
        assert (
            item["observed_representative_ids"] != (item["expected_representative_ids"])
        ), name
        assert item["collapsed_pairs_provenance"][0]["collapsed_by"] == "same_words"
        # One key for both passages is the evidence, and it is read off the engine's
        # own key function rather than off the fixture.
        keys = set(item["observed"]["equality_keys"].values())
        assert len(keys) == 1, name
        assert "equality key" in item["reason"], name
        assert "needs no vector" in item["reason"], name


def test_the_confusable_vector_cases_say_the_vector_was_constructed(
    report: dict[str, Any],
) -> None:
    """A merge the probe caused is named as the probe's doing.

    The two records with equal vectors violate the contract, and each says the
    cosine path merged a pair whose synthetic vectors were made equal. Neither
    claims a model did it, and neither is about a corpus.
    """

    for name in (
        "negated_and_asserted_claims_with_equal_vectors",
        "independent_claims_from_different_authors_with_equal_vectors",
    ):
        item = _case(report, name)
        assert item["status"] == "violated", name
        assert "embedding_confusable" in item["input_tags"], name
        assert item["collapsed_pairs_provenance"][0]["collapsed_by"] == (
            "same_meaning"
        ), name
        notes = " ".join(item["input_tag_notes"]) + " " + item["note"]
        assert "constructed" in notes or "probe" in notes, name
        assert item["unmeasured"], name
    # The control record holds the same passages apart with vectors the probe kept
    # distinct, so the two records differ only in the input this audit built.
    control = _case(report, "independent_claims_with_distinct_vectors_survive")
    confusable = _case(
        report, "independent_claims_from_different_authors_with_equal_vectors"
    )
    assert control["status"] == "met"
    # Same passages, same expected representatives, and the only difference is the
    # vectors this audit built: the control holds, the confusable pair merges.
    assert (
        control["expected_representative_ids"]
        == (confusable["expected_representative_ids"])
    )
    assert sorted(control["observed"]["equality_keys"].values()) == sorted(
        confusable["observed"]["equality_keys"].values()
    )
    assert control["collapsed_pairs_provenance"] == []
    assert "distinct_synthetic_vectors" in control["input_tags"]


def test_an_allowed_collapse_keeps_both_copies_nameable(
    report: dict[str, Any],
) -> None:
    """One representative, both copies, and the source of each."""

    for name in (
        "exact_reprint_keeps_both_copies_in_the_record",
        "same_text_independent_authors_retain_the_other_source",
    ):
        item = _case(report, name)
        assert item["status"] == "met", name
        assert len(item["observed_representative_ids"]) == 1, name
        pairs = item["collapsed_pairs_provenance"]
        assert len(pairs) == 1, name
        pair = pairs[0]
        named = {
            pair["chunk_id"],
            pair["repeated_chunk_id"],
            pair["source_id"],
            pair["repeated_source_id"],
            pair["source_relative_path"],
            pair["repeated_source_relative_path"],
        }
        assert all(named), name
        assert pair["repeated_chunk_id"] != pair["chunk_id"], name
        assert pair["repeated_source_id"] != pair["source_id"], name
        assert item["observed"]["distinct_surviving_sources"] == 1, name


def test_an_excluded_copy_is_withheld_before_it_can_suppress_the_allowed_one(
    report: dict[str, Any],
) -> None:
    """The gate runs first, so the withheld copy is not there to be merged into."""

    for name, withheld in (
        (
            "excluded_copy_does_not_suppress_the_allowed_copy",
            "audit-chunk-copy-excluded",
        ),
        (
            "excluded_source_copy_does_not_suppress_the_allowed_copy",
            "audit-chunk-src-excluded",
        ),
    ):
        item = _case(report, name)
        assert item["status"] == "met", name
        assert (
            item["observed_representative_ids"] == (item["expected_representative_ids"])
        ), name
        # Nothing was merged, because one candidate was never admitted.
        assert item["collapsed_pairs_provenance"] == [], name
        assert item["probe_trace"]["gate_withheld"]["ids"] == [withheld], name
        assert withheld not in item["observed_representative_ids"], name
        assert item["evidence_kind"] == "engine_functions_pipeline_order", name
        assert "exclusion_gate" in item["input_tags"], name
        assert (
            "research_rag.retrieval.search.SearchWorkflow._matches_filters"
            in item["engine_call_path"]
        ), name


def test_the_probe_trace_is_bounded_and_carries_no_model_output(
    report: dict[str, Any],
) -> None:
    for item in report["cases"]:
        assert item["probe_budget"] == audit.PROBE_TRACE_BUDGET, item["name"]
        for stage in item["probe_trace"].values():
            assert stage["count"] == len(stage["ids"]) or stage["truncated"], item[
                "name"
            ]
            assert len(stage["ids"]) <= audit.PROBE_TRACE_BUDGET, item["name"]
        # The trace holds identifiers and counts. A number a model produced would
        # be the one thing in it that a reader could not reproduce from this file.
        rendered = json.dumps(item["probe_trace"])
        for absent in ("embedding", "rerank_score", "cosine_similarity"):
            assert absent not in rendered, (item["name"], absent)


def test_the_summary_counts_violations_and_unit_results_apart(
    report: dict[str, Any],
) -> None:
    outcome = report["contract_outcome"]
    assert outcome["cases"] == len(report["cases"])
    assert (
        outcome["met"] + outcome["violated"] + outcome["unknown"] == (outcome["cases"])
    )
    assert outcome["violated_cases"] == [
        item["name"] for item in report["cases"] if item["status"] == "violated"
    ]
    assert "not a rate" in outcome["false_suppression_scope"]
    # Whether the code unit tests pass is another measurement, and this record
    # does not fold it into its own counts.
    assert report["code_unit_result"]["in_this_report"] is False


def test_the_record_names_no_policy_and_no_probability(report: dict[str, Any]) -> None:
    assert report["policy_recommendation"] is None
    assert "names no preferred change" in report["policy_note"]
    forbidden = ("recommend", "should ", "winner", "prefer ", "likely", "probab")
    for item in report["cases"]:
        for token in forbidden:
            assert token not in item["reason"].casefold(), (item["name"], token)


def test_the_collapsed_count_is_the_number_of_merged_pairs(
    report: dict[str, Any],
) -> None:
    """The synthetic-stage count is what the probe built, counted exactly."""

    suppressed = [
        item
        for item in report["cases"]
        if item["expected_preserve"] and item["collapsed_pairs_provenance"]
    ]
    assert report["contract_outcome"]["false_suppression_pairs_synthetic_stage"] == (
        len(suppressed)
    )
    assert all(len(item["collapsed_pairs_provenance"]) == 1 for item in suppressed)
    # Every counted pair is one this file's own fixture built, so the count is a
    # property of the probe and not a count over any corpus.
    assert all(
        set(item["input_tags"])
        & {
            "embedding_confusable",
            "words_key_operator_collision",
            "words_key_casefold_collision",
        }
        for item in suppressed
    )


def _incompatible_collapse(
    ordered_ids: Any,
    *,
    chunks_by_id: Any,
    documents_by_id: Any,
    vectors: Any,
) -> Any:
    """A collapse whose interface is not the one this probe was written against."""

    return list(ordered_ids), []


def test_an_engine_the_probe_cannot_call_is_reported_as_undecided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A signature this probe does not know is a gap in the record, not a pass."""

    monkeypatch.setattr(audit, "_collapse_repetitions", _incompatible_collapse)
    destination = tmp_path / "record.json"
    assert audit.main(["--report", str(destination)]) == 3
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["contract_outcome"]["violated"] == 0
    assert record["contract_outcome"]["unknown"] == record["contract_outcome"]["cases"]
    assert record["engine_interface_problems"]
    for item in record["cases"]:
        assert item["status"] == "unknown", item["name"]
        assert item["pass"] is False, item["name"]
        assert item["evidence_kind"] == "none", item["name"]
        assert item["observed_representative_ids"] == [], item["name"]
        assert "differs from the one this probe" in item["reason"], item["name"]


def test_a_case_it_cannot_decide_is_undecided_rather_than_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    undecided = audit.ContractCase(
        name="fixture_without_a_decidable_expectation",
        category="meaning_distinct",
        relation="Two passages, and a fixture that states no verdict to check.",
        expectation="unspecified",
        expected_representative_ids=("audit-chunk-a",),
        passages=(
            audit._passage("a", "First synthetic statement.", vector=(1.0, 0.0, 0.0)),
            audit._passage("b", "Second synthetic statement.", vector=(0.0, 1.0, 0.0)),
        ),
    )
    monkeypatch.setattr(audit, "CONTRACT_CASES", (undecided,))
    item = audit.build_report()["cases"][0]
    assert item["status"] == "unknown"
    assert item["pass"] is False
    assert "no decidable expectation" in item["reason"]


def test_malformed_vectors_stop_the_probe_instead_of_being_measured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A vector of the wrong shape is a broken fixture, not an observation."""

    ragged = audit.ContractCase(
        name="fixture_with_vectors_of_two_shapes",
        category="meaning_distinct",
        relation="Two passages whose probe vectors do not share a dimension.",
        expectation="preserve",
        expected_representative_ids=("audit-chunk-wide", "audit-chunk-narrow"),
        passages=(
            audit._passage("wide", "First statement.", vector=(1.0, 0.0, 0.0)),
            audit._passage("narrow", "Second statement.", vector=(1.0, 0.0)),
        ),
    )
    monkeypatch.setattr(audit, "CONTRACT_CASES", (ragged,))
    destination = tmp_path / "record.json"
    assert audit.main(["--report", str(destination)]) == 2
    # Nothing was measured, so nothing is written: a record from a probe that could
    # not run would read as a result.
    assert not destination.exists()
    with pytest.raises(audit.ProbeSetupError):
        audit.build_report()


def test_a_non_finite_coordinate_stops_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = audit.ContractCase(
        name="fixture_with_a_non_finite_coordinate",
        category="meaning_distinct",
        relation="A passage whose probe vector holds infinity.",
        expectation="preserve",
        expected_representative_ids=("audit-chunk-inf",),
        passages=(
            audit._passage("inf", "First statement.", vector=(float("inf"), 0.0, 0.0)),
        ),
    )
    monkeypatch.setattr(audit, "CONTRACT_CASES", (broken,))
    with pytest.raises(audit.ProbeSetupError):
        audit.build_report()


def test_the_record_is_written_only_outside_a_project(
    tmp_path: Path,
) -> None:
    """One file is written, and never one inside a collection's own directory."""

    project = tmp_path / "a-project"
    (project / ".research-rag").mkdir(parents=True)
    inside = project / ".research-rag" / "collapse-audit.json"
    with pytest.raises(audit.ProbeSetupError):
        audit.write_report(inside, {"cases": []})
    assert not inside.exists()

    outside = tmp_path / "records" / "collapse-audit.json"
    written = audit.write_report(outside, {"cases": []})
    assert written == outside
    assert json.loads(outside.read_text(encoding="utf-8")) == {"cases": []}

    # A record path inside a project is refused by the command too, so the guard
    # is not something a caller can step around by shelling out.
    assert audit.main(["--report", str(inside)]) == 2


def test_the_command_writes_the_violations_it_found_and_exits_one(
    report: dict[str, Any], tmp_path: Path
) -> None:
    destination = tmp_path / "record.json"
    assert report["contract_outcome"]["violated"] > 0
    assert audit.main(["--report", str(destination)]) == 1
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["contract_outcome"]["violated_cases"]
    for item in written["cases"]:
        if item["status"] == "violated":
            assert (
                item["observed_representative_ids"]
                or (item["collapsed_pairs_provenance"])
            ), item["name"]


def test_a_run_with_nothing_to_report_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exit code follows the contracts, so it is not pinned to failing.

    One contract the engine holds is enough: a run that found nothing to report
    has to say so through its own code, or a reader cannot tell a clean run from a
    run that never checked.
    """

    holding = audit.ContractCase(
        name="fixture_the_engine_holds",
        category="meaning_distinct",
        relation="Two statements with different words, and vectors kept apart.",
        expectation="preserve",
        expected_representative_ids=("audit-chunk-first", "audit-chunk-second"),
        passages=(
            audit._passage("first", "First statement.", vector=(1.0, 0.0, 0.0)),
            audit._passage("second", "Second statement.", vector=(0.0, 1.0, 0.0)),
        ),
    )
    monkeypatch.setattr(audit, "CONTRACT_CASES", (holding,))
    destination = tmp_path / "record.json"
    assert audit.main(["--report", str(destination)]) == 0
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["contract_outcome"] == {
        **written["contract_outcome"],
        "cases": 1,
        "met": 1,
        "violated": 0,
        "unknown": 0,
        "false_suppression_pairs_synthetic_stage": 0,
    }


def _identical_sources(project: Path) -> ResearchService:
    """Two sources holding one identical sentence, ingested with the test fakes."""

    for name in ("alpha", "beta"):
        write_pdf(
            project / "sources" / f"{name}.pdf",
            ["Recovery reached 5% in the cohort after the second wave."],
            title=name.capitalize(),
        )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))
    return service


def test_the_real_pipeline_collapses_an_exact_reprint_and_names_both_sources(
    project: Path,
) -> None:
    """The probe's expectation for an exact reprint, against the real pipeline.

    Both halves of the retrieval run on the test fakes, so this is the engine's
    own order: rank, gate, collapse. Nothing is embedded and no model is loaded;
    the two copies collapse on their words before any vector is read.
    """

    service = _identical_sources(project)
    answer = asyncio.run(service.search("recovery reached", top_k=5, rerank=False))
    assert len(answer["hits"]) == 1
    disclosure = answer["collapsed_repetitions"]
    assert disclosure["repetitions_collapsed"] == 1
    assert disclosure["collapsed_by_same_words"] == 1
    pairs = disclosure["pairs"]
    assert len(pairs) == 1
    named = {
        pairs[0]["source_relative_path"],
        pairs[0]["repeated_source_relative_path"],
    }
    assert named == {"alpha.pdf", "beta.pdf"}
    assert pairs[0]["source_id"] != pairs[0]["repeated_source_id"]


def test_the_real_pipeline_keeps_the_allowed_copy_when_the_other_is_excluded(
    project: Path,
) -> None:
    """A withheld copy does not stand in for the copy that is still answerable."""

    service = _identical_sources(project)
    first = asyncio.run(service.search("recovery reached", top_k=5, rerank=False))
    withheld = first["hits"][0]

    asyncio.run(
        service.set_chunk_inclusion(
            withheld["chunk_id"], included=False, reason="Audit fixture."
        )
    )
    second = asyncio.run(service.search("recovery reached", top_k=5, rerank=False))
    assert second["excluded_chunk_count"] == 1
    assert len(second["hits"]) == 1
    # The survivor is the copy that was never withheld, and it is not standing in
    # for the withheld one: nothing was merged, because nothing was merged into.
    assert second["hits"][0]["chunk_id"] != withheld["chunk_id"]
    assert second["hits"][0]["source_path"] != withheld["source_path"]
    assert second["collapsed_repetitions"].get("pairs", []) == []
