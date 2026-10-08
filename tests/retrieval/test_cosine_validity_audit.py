"""The cosine-validity audit is a diagnostic, and these tests hold it to that.

Every assertion is about the audit, not about the corpus: that it normalises
before it scores, that it keeps the score rule and the vectors it was handed
apart, and that a fixture it cannot run is refused rather than measured. The
equal vectors over distinct text are fabricated in this file and named
synthetic, so no case here claims to establish behaviour over a real corpus, and
no model or index is loaded anywhere. The pool checks stand a controlled
generation and judged set in the shipped resolver and real review-exclusion
files, so they exercise the audit's streamed retention and its refusal path
without private corpus or model work.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from research_rag.project.config import resolve_config
from research_rag.storage.records import (
    write_chunk_exclusions,
    write_source_exclusions,
)

#: The script is read by path, the way a person runs it, so these tests hold the
#: command's own behaviour rather than a package import of a module.
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_cosine_validity.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("audit_cosine_validity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def _point(cosine: float) -> list[float]:
    """A unit vector whose cosine with ``(1, 0)`` is exactly ``cosine``."""

    return [cosine, float(np.sqrt(1.0 - cosine * cosine))]


def _probe(
    probe_id: str,
    relation: str,
    expected: list[str],
    passages: list[tuple[str, str]],
) -> dict[str, Any]:
    """A constructed probe whose expected identifiers are stated up front."""

    return {
        "id": probe_id,
        "query": "A constructed question.",
        "pair_relation": relation,
        "expected_answer_ids": expected,
        "passages": [{"id": pid, "text": text} for pid, text in passages],
    }


# --- cosine_matrix ---------------------------------------------------------


def test_cosine_matrix_normalises_each_row_before_scoring() -> None:
    result = audit.cosine_matrix([[3.0, 4.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(0.6)
    assert result[0, 1] == pytest.approx(0.8)
    assert result[1, 0] == pytest.approx(1.0)
    assert result[1, 1] == pytest.approx(0.0)


def test_cosine_matrix_is_invariant_under_positive_row_scaling() -> None:
    query = [[1.0, 0.0, 0.0]]
    base = audit.cosine_matrix([[1.0, 2.0, 2.0]], query)
    scaled = audit.cosine_matrix([[5.0, 10.0, 10.0]], query)
    # Direction decides the score, so a longer vector in the same direction is
    # the same measurement rather than a stronger one.
    assert scaled[0, 0] == pytest.approx(base[0, 0])


def test_cosine_matrix_is_symmetric_up_to_transpose() -> None:
    left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    right = [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
    assert np.allclose(
        audit.cosine_matrix(left, right), audit.cosine_matrix(right, left).T
    )


@pytest.mark.parametrize(
    ("left", "right", "match"),
    [
        ([1.0, 2.0], [[1.0, 2.0]], "matching dimensions"),
        ([[1.0, 2.0]], [[1.0, 2.0, 3.0]], "matching dimensions"),
        ([[float("nan"), 0.0]], [[1.0, 0.0]], "finite"),
        ([[float("inf"), 0.0]], [[1.0, 0.0]], "finite"),
        ([[0.0, 0.0]], [[1.0, 0.0]], "nonzero norms"),
    ],
)
def test_cosine_matrix_refuses_a_vector_it_cannot_measure(
    left: Any, right: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        audit.cosine_matrix(left, right)


def test_cosine_matrix_keeps_the_matrix_shape() -> None:
    result = audit.cosine_matrix(np.ones((3, 4)), np.ones((2, 4)))
    assert result.shape == (3, 2)


def test_score_distribution_counts_comparisons_not_independent_queries() -> None:
    result = audit.score_distribution([[0.6, 0.7], [0.8, 0.95]])
    assert result["comparison_count"] == 4
    assert result["median"] == pytest.approx(0.75)
    assert result["fraction_between_0_7_and_0_9"] == 0.5


def test_read_only_config_does_not_initialize_a_constructed_only_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_rag.project.settings import sources_for

    monkeypatch.setattr(audit, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        audit,
        "sources_for",
        lambda root: replace(sources_for(root), user_config=None, project_config=None),
    )
    monkeypatch.setattr(
        audit, "user_cache_path", lambda *args, **kwargs: tmp_path / "cache"
    )
    config = audit.read_only_config(None)
    assert config.settings.offline is True
    assert not config.portable_root.exists()
    assert not config.model_cache_root.exists()


def test_analyse_probe_preserves_caller_vector_provenance() -> None:
    probe = _probe(
        "provenance",
        "different_claims",
        ["a"],
        [("a", "First claim."), ("b", "Second claim.")],
    )
    result = audit.analyse_probe(
        probe,
        [1, 0],
        [[1, 0], [0, 1]],
        floor=0.72,
        margin=0.1,
        duplicate_threshold=0.99,
        vector_origin="synthetic_unit_test",
    )
    assert result["collapse"]["vector_origin"] == "synthetic_unit_test"


def test_read_only_config_refuses_to_initialize_an_unregistered_project(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError):
        audit.read_only_config(tmp_path)
    assert not (tmp_path / ".research-rag").exists()


def _constructed_only_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the audit at a root with no project or user settings layer."""

    from research_rag.project.settings import sources_for

    monkeypatch.setattr(audit, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        audit,
        "sources_for",
        lambda root: replace(sources_for(root), user_config=None, project_config=None),
    )
    monkeypatch.setattr(
        audit, "user_cache_path", lambda *args, **kwargs: tmp_path / "cache"
    )


def _configured_model_cache(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Force runtime.model_cache_root as a resolved settings layer would."""

    real = audit.resolve_settings

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        values, provenance = real(*args, **kwargs)
        if value is not None:
            values = {**values, "model_cache_root": value}
        return values, provenance

    monkeypatch.setattr(audit, "resolve_settings", wrapper)


def _legacy_models(tmp_path: Path) -> Path:
    legacy = tmp_path / ".research-rag" / "runtime" / "models"
    legacy.mkdir(parents=True)
    (legacy / "model.bin").write_bytes(b"binary")
    return legacy


def test_read_only_config_falls_back_to_legacy_models_when_shared_cache_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _constructed_only_root(tmp_path, monkeypatch)
    legacy = _legacy_models(tmp_path)
    config = audit.read_only_config(None)
    assert config.settings.model_cache_root is None
    assert config.legacy_models_root == legacy.resolve()
    assert config.model_cache_root == legacy.resolve()


def test_read_only_config_prefers_a_populated_shared_cache_over_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _constructed_only_root(tmp_path, monkeypatch)
    shared = tmp_path / "cache" / "models"
    shared.mkdir(parents=True)
    (shared / "model.bin").write_bytes(b"binary")
    _legacy_models(tmp_path)
    config = audit.read_only_config(None)
    assert config.model_cache_root == shared.resolve()
    assert config.model_cache_root != config.legacy_models_root


def test_read_only_config_does_not_fall_back_when_a_cache_root_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _constructed_only_root(tmp_path, monkeypatch)
    _legacy_models(tmp_path)
    explicit = tmp_path / "explicit" / "models"
    _configured_model_cache(monkeypatch, str(explicit))
    config = audit.read_only_config(None)
    assert config.settings.model_cache_root == explicit
    # An explicitly configured root is a decision, so an empty one beside a
    # populated legacy tree is still the root the audit uses.
    assert config.model_cache_root == explicit.resolve()
    assert config.model_cache_root != config.legacy_models_root


def test_read_only_config_does_not_fall_back_to_an_empty_legacy_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _constructed_only_root(tmp_path, monkeypatch)
    (tmp_path / ".research-rag" / "runtime" / "models").mkdir(parents=True)
    config = audit.read_only_config(None)
    assert config.model_cache_root == (tmp_path / "cache" / "models").resolve()
    assert config.model_cache_root != config.legacy_models_root


def test_read_only_config_falls_back_under_a_relocated_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "research-project"
    (project / "sources").mkdir(parents=True)
    portable = project / ".research-rag"
    portable.mkdir()
    (portable / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": "p1",
                "name": "Relocated",
                "source_directory": "sources",
            }
        ),
        encoding="utf-8",
    )
    relocated = tmp_path / "relocated-state"
    (portable / "runtime-root").write_text(f"{relocated}\n", encoding="utf-8")
    legacy = relocated / "models"
    legacy.mkdir(parents=True)
    (legacy / "model.bin").write_bytes(b"binary")
    monkeypatch.setattr(
        audit, "user_cache_path", lambda *args, **kwargs: tmp_path / "cache"
    )
    config = audit.read_only_config(project)
    assert config.state_root == relocated
    assert config.legacy_models_root == legacy
    assert config.model_cache_root == legacy


def test_engine_source_identity_covers_both_executed_modules_and_repeats() -> None:
    identity = audit.engine_source_identity()
    assert set(identity) == {"sha256", "module_sha256"}
    assert len(identity["sha256"]) == 64
    expected = {
        inspect.getmodule(audit._collapse_repetitions).__name__,
        inspect.getmodule(audit._passage_equality_key).__name__,
    }
    assert set(identity["module_sha256"]) == expected
    assert all(len(digest) == 64 for digest in identity["module_sha256"].values())
    # Identity is a digest of the code, so two calls over an unchanged tree agree.
    assert audit.engine_source_identity() == identity


@pytest.mark.parametrize(
    "attribute", ["_collapse_repetitions", "_passage_equality_key"]
)
def test_engine_source_identity_changes_with_a_covered_module(
    monkeypatch: pytest.MonkeyPatch, attribute: str
) -> None:
    identity = audit.engine_source_identity()
    target = inspect.getmodule(getattr(audit, attribute))
    assert target is not None
    real_getsource = inspect.getsource

    def fake_getsource(obj: Any) -> str:
        source = real_getsource(obj)
        if obj is target:
            return source + "\n# a changed source region\n"
        return source

    monkeypatch.setattr(audit.inspect, "getsource", fake_getsource)
    changed = audit.engine_source_identity()
    assert changed["sha256"] != identity["sha256"]
    assert (
        changed["module_sha256"][target.__name__]
        != identity["module_sha256"][target.__name__]
    )
    others = set(identity["module_sha256"]) - {target.__name__}
    for name in others:
        assert changed["module_sha256"][name] == identity["module_sha256"][name]


def test_analyse_probe_requires_a_pair() -> None:
    probe = _probe("one", "different_claims", ["a"], [("a", "First claim.")])
    with pytest.raises(ValueError, match="exactly two distinct"):
        audit.analyse_probe(
            probe, [1, 0], [[1, 0]], floor=0.72, margin=0.1, duplicate_threshold=0.99
        )


# --- admission_projection --------------------------------------------------


def test_admission_projection_of_an_empty_sequence_has_no_best() -> None:
    result = audit.admission_projection([], floor=0.72, margin=0.10)
    assert result["best"] is None
    assert result["effective_floor"] == 0.72
    assert result["admitted"] == []


def test_admission_projection_clears_scores_below_the_floor() -> None:
    result = audit.admission_projection([0.80, 0.60], floor=0.72, margin=0.0)
    # A non-positive margin disables the relative rule, so the floor stands.
    assert result["effective_floor"] == 0.72
    assert result["admitted"] == [True, False]


def test_admission_projection_rescues_a_near_miss_by_the_margin() -> None:
    # Best clears the floor, and best - margin falls below it, so the effective
    # floor is the relative one and a score beneath the absolute floor passes.
    result = audit.admission_projection([0.74, 0.66], floor=0.72, margin=0.10)
    assert result["best"] == pytest.approx(0.74)
    assert result["effective_floor"] == pytest.approx(0.64)
    assert result["admitted"] == [True, True]


def test_admission_projection_keeps_the_floor_when_nothing_clears_it() -> None:
    result = audit.admission_projection([0.40, 0.30], floor=0.72, margin=0.10)
    assert result["effective_floor"] == 0.72
    assert result["admitted"] == [False, False]


def test_admission_projection_discloses_itself_as_a_simulation() -> None:
    result = audit.admission_projection([0.90], floor=0.72, margin=0.10)
    assert result["kind"] == "score_only_policy_simulation"
    assert result["production_gate_executed"] is False
    assert result["floor"] == 0.72
    assert result["relative_margin"] == 0.10


def test_admission_projection_refuses_a_non_finite_or_shaped_sequence() -> None:
    with pytest.raises(ValueError, match="finite sequence"):
        audit.admission_projection([[0.9]], floor=0.72, margin=0.10)
    with pytest.raises(ValueError, match="finite sequence"):
        audit.admission_projection([float("nan")], floor=0.72, margin=0.10)


# --- analyse_probe ---------------------------------------------------------


def test_analyse_probe_ranks_by_descending_cosine() -> None:
    probe = _probe(
        "order",
        "contradiction",
        ["a"],
        [
            ("a", "A low-scoring passage."),
            ("b", "A high-scoring passage."),
        ],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [_point(0.2), _point(0.9)],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["id"] == "order"
    assert result["ranked_ids"] == ["b", "a"]
    assert result["expected_answer_ranked_first"] is False
    assert result["query_cosines"] == pytest.approx({"a": 0.2, "b": 0.9})


def test_analyse_probe_breaks_a_tie_by_the_input_order() -> None:
    probe = _probe(
        "ties",
        "equivalent_wording",
        ["a", "b"],
        [("a", "First paraphrase."), ("b", "Second paraphrase.")],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[1.0, 0.0], [1.0, 0.0]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["ranked_ids"] == ["a", "b"]
    assert result["expected_answer_ranked_first"] is True


def test_analyse_probe_calls_the_engine_and_merges_fabricated_equal_vectors() -> None:
    """Synthetic: the equal vectors are this file's, not a model's output.

    Distinct text with equal vectors is the constructed contrast that tests the
    collapse path. It says what the engine does with an input this probe built;
    it does not say any embedder would produce such vectors for this text.
    """

    probe = _probe(
        "collapse",
        "contradiction",
        ["a"],
        [
            ("a", "Workers own and control the robots."),
            ("b", "Managers own and control the robots."),
        ],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[1.0, 0.0], [1.0, 0.0]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    collapse = result["collapse"]
    assert collapse["engine_executed"] is True
    assert collapse["threshold"] == 0.99
    assert collapse["input_order"] == ["a", "b"]
    assert collapse["kept_ids"] == ["a"]
    assert len(collapse["pairs"]) == 1
    assert collapse["pairs"][0]["collapsed_by"] == "same_meaning"
    assert collapse["distinct_evidence_suppressed"] is True
    assert result["passage_pair_cosine"] == pytest.approx(1.0)


def test_analyse_probe_marks_an_exact_reprint_without_flagging_distinct_suppression() -> (
    None
):
    same = "The managers monitor the workers."
    probe = _probe("reprint", "exact_reprint", ["a", "b"], [("a", same), ("b", same)])
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[1.0, 0.0], [1.0, 0.0]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    collapse = result["collapse"]
    assert collapse["kept_ids"] == ["a"]
    assert collapse["pairs"][0]["collapsed_by"] == "same_words"
    # An exact reprint is the one merge the relation allows, so it is not
    # reported as distinct evidence being suppressed.
    assert collapse["distinct_evidence_suppressed"] is False


def test_analyse_probe_reports_an_expected_answer_the_gate_rejects() -> None:
    probe = _probe(
        "reject",
        "different_roles",
        ["a"],
        [
            ("a", "The managers monitor the workers."),
            ("b", "The workers monitor the managers."),
        ],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[0.0, 1.0], [1.0, 0.0]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["expected_answer_rejected"] == ["a"]
    assert result["answer_over_contrast_margin"] == pytest.approx(-1.0)
    # With an expected answer named, the unsupported-admission figure does not
    # apply and is reported as unset rather than guessed.
    assert result["unsupported_candidate_admitted"] is None


def test_analyse_probe_reports_an_unsupported_candidate_being_admitted() -> None:
    probe = _probe(
        "unsupported",
        "different_claims",
        [],
        [("a", "Workers own the robots."), ("b", "Robots assemble car doors.")],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[1.0, 0.0], [0.9, float(np.sqrt(0.19))]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["unsupported_candidate_admitted"] is True
    assert result["expected_answer_ranked_first"] is None
    assert result["expected_answer_rejected"] == []
    assert result["answer_over_contrast_margin"] is None


def test_analyse_probe_reports_no_admission_when_nothing_clears_the_floor() -> None:
    probe = _probe(
        "unsupported-low",
        "different_claims",
        [],
        [("a", "Workers own the robots."), ("b", "Robots assemble car doors.")],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [[0.5, float(np.sqrt(0.75))], [0.4, float(np.sqrt(0.84))]],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["unsupported_candidate_admitted"] is False


def test_analyse_probe_names_a_constructed_expectation_and_a_simulation() -> None:
    probe = _probe(
        "scope",
        "equivalent_wording",
        ["a", "b"],
        [("a", "One wording."), ("b", "Another wording.")],
    )
    result = audit.analyse_probe(
        probe,
        [1.0, 0.0],
        [_point(0.9), _point(0.8)],
        floor=0.72,
        margin=0.10,
        duplicate_threshold=0.99,
    )
    assert result["expectation_kind"] == (
        "constructed_answer_selection_not_corpus_relevance"
    )
    assert result["admission"]["kind"] == "score_only_policy_simulation"
    assert result["admission"]["production_gate_executed"] is False


def test_analyse_probe_refuses_an_expected_answer_absent_from_the_probe() -> None:
    probe = _probe(
        "bad-expected",
        "different_roles",
        ["z"],
        [
            ("a", "The managers monitor the workers."),
            ("b", "The workers monitor the managers."),
        ],
    )
    with pytest.raises(ValueError, match="absent from the probe"):
        audit.analyse_probe(
            probe,
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            floor=0.72,
            margin=0.10,
            duplicate_threshold=0.99,
        )


def test_analyse_probe_refuses_a_passage_without_its_own_vector() -> None:
    probe = _probe(
        "bad-vectors",
        "different_roles",
        ["a"],
        [
            ("a", "The managers monitor the workers."),
            ("b", "The workers monitor the managers."),
        ],
    )
    with pytest.raises(ValueError, match="exactly one vector"):
        audit.analyse_probe(
            probe,
            [1.0, 0.0],
            [[1.0, 0.0]],
            floor=0.72,
            margin=0.10,
            duplicate_threshold=0.99,
        )


def test_constructed_probes_state_ids_and_distinct_relations() -> None:
    probes = audit.constructed_probes()
    ids = [probe["id"] for probe in probes]
    assert len(ids) == len(set(ids))
    relations = {probe["pair_relation"] for probe in probes}
    assert {"equivalent_wording", "contradiction", "different_roles"} <= relations
    assert {"exact_reprint", "different_claims", "different_quantity"} <= relations
    for probe in probes:
        passage_ids = [item["id"] for item in probe["passages"]]
        assert len(passage_ids) == len(set(passage_ids)) == 2
        assert probe["query"].strip()
        # Expectations are predetermined per case, never sampled prevalence.
        assert set(probe["expected_answer_ids"]) <= set(passage_ids)


# --- validate_report_path --------------------------------------------------


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An approved scratch root whose ancestors the guard ignores.

    ``/tmp`` and this checkout's home are both initialized projects. The guard
    ignores markers above the approved scratch root and still protects markers at
    or below it, so the tests point the root at their own temporary tree.
    """

    root = (tmp_path / "scratch").resolve()
    root.mkdir()
    monkeypatch.setattr(audit, "REPORT_SCRATCH", root)
    return root


def test_validate_report_path_accepts_a_new_path_within_the_scratch_root(
    scratch: Path,
) -> None:
    (scratch / "records").mkdir()
    destination = scratch / "records" / "cosine-audit.json"
    assert audit.validate_report_path(destination) == destination.resolve()


def test_validate_report_path_ignores_a_marker_above_the_scratch_root(
    scratch: Path,
) -> None:
    # A marker above the approved root, such as the one at /tmp, is ignored.
    (scratch.parent / ".research-rag").mkdir()
    (scratch / "records").mkdir()
    destination = scratch / "records" / "cosine-audit.json"
    assert audit.validate_report_path(destination) == destination.resolve()


def test_validate_report_path_protects_a_marker_below_the_scratch_root(
    scratch: Path,
) -> None:
    project = scratch / "project"
    (project / ".research-rag").mkdir(parents=True)
    (project / "records").mkdir()
    with pytest.raises(ValueError, match="research project"):
        audit.validate_report_path(project / "records" / "cosine-audit.json")


def test_validate_report_path_protects_a_marker_at_the_scratch_root(
    scratch: Path,
) -> None:
    (scratch / ".research-rag").mkdir()
    with pytest.raises(ValueError, match="research project"):
        audit.validate_report_path(scratch / "cosine-audit.json")


def test_validate_report_path_refuses_a_path_inside_the_repository() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        audit.validate_report_path(audit.REPOSITORY / "cosine-audit.json")


def test_validate_report_path_refuses_the_measured_project_before_scratch(
    scratch: Path,
) -> None:
    (scratch / "records").mkdir()
    # The measured project is refused even when it is the approved scratch root.
    with pytest.raises(ValueError, match="outside the measured project"):
        audit.validate_report_path(scratch / "records" / "cosine-audit.json", scratch)


def test_validate_report_path_refuses_an_existing_filename(scratch: Path) -> None:
    (scratch / "records").mkdir()
    existing = scratch / "records" / "cosine-audit.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="new report filename"):
        audit.validate_report_path(existing)


def test_validate_report_path_refuses_a_missing_parent(scratch: Path) -> None:
    with pytest.raises(ValueError, match="parent directory"):
        audit.validate_report_path(scratch / "missing" / "cosine-audit.json")


# --- stream_target_chunks --------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


_REAL_HARNESS: Any = None


def _real_harness() -> Any:
    """The shipped evaluation harness, loaded once per test run."""

    global _REAL_HARNESS
    if _REAL_HARNESS is None:
        _REAL_HARNESS = audit.evaluation_harness()
    return _REAL_HARNESS


def _stream_harness() -> SimpleNamespace:
    """The two harness calls the streaming helper makes, without a model."""

    return SimpleNamespace(
        normalize=lambda value: _WHITESPACE.sub(" ", value).strip(),
        _chunk_text=lambda chunk: str(chunk.get("contents") or chunk.get("text") or ""),
    )


def _write_chunks(generation_root: Path, chunks: list[dict[str, Any]]) -> None:
    """Write canonical chunk rows, one JSON object per line, in order."""

    directory = generation_root / "chunks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


def _stream(
    root: Path,
    documents: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_root = root / "generation"
    _write_chunks(generation_root, chunks)
    by_id = {str(document["document_id"]): document for document in documents}
    return audit.stream_target_chunks(
        generation_root, by_id, targets, _stream_harness()
    )


def test_stream_target_chunks_retains_a_witness_and_bounds_matches(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}]
    chunks = [
        {"chunk_id": "c0", "document_id": "d1", "contents": "an introduction"},
        *[
            {"chunk_id": f"u{i}", "document_id": "d9", "contents": "unrelated"}
            for i in range(40)
        ],
        {"chunk_id": "c1", "document_id": "d1", "contents": "a needle appears"},
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["method"] == "streamed_snippet_matches_with_source_witnesses"
    assert stats["scanned_chunk_count"] == len(chunks)
    assert stats["exact_match_counts"] == {"t1": 1}
    assert stats["retained_record_count"] == 2
    assert [entry["chunk_id"] for entry in retained] == ["c0", "c1"]


def test_stream_target_chunks_unrelated_chunks_do_not_add_retention(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}]
    base = [
        {"chunk_id": "c0", "document_id": "d1", "contents": "an introduction"},
        {"chunk_id": "c1", "document_id": "d1", "contents": "a needle appears"},
    ]
    noise = [
        {"chunk_id": f"u{i}", "document_id": "d9", "contents": "unrelated"}
        for i in range(500)
    ]
    small, small_stats = _stream(tmp_path / "small", documents, targets, base)
    large, large_stats = _stream(
        tmp_path / "large", documents, targets, [*base, *noise]
    )
    assert small_stats["retained_record_count"] == 2
    assert large_stats["retained_record_count"] == 2
    assert [entry["chunk_id"] for entry in small] == [
        entry["chunk_id"] for entry in large
    ]


def test_stream_target_chunks_matches_across_case_and_whitespace(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [
        {"target_id": "t1", "source_path": "alpha.pdf", "snippet": "  Needle   PHRASE "}
    ]
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "contents": "a needle\nphrase follows",
        },
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["exact_match_counts"] == {"t1": 1}
    assert [entry["chunk_id"] for entry in retained] == ["c1"]


def test_stream_target_chunks_keeps_one_witness_and_a_match_per_target(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [
        {"target_id": "t1", "source_path": "alpha.pdf", "snippet": "alpha claim"},
        {"target_id": "t2", "source_path": "alpha.pdf", "snippet": "beta claim"},
    ]
    chunks = [
        {"chunk_id": "c0", "document_id": "d1", "contents": "header"},
        {"chunk_id": "c1", "document_id": "d1", "contents": "the alpha claim stands"},
        {"chunk_id": "c2", "document_id": "d1", "contents": "the beta claim stands"},
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["exact_match_counts"] == {"t1": 1, "t2": 1}
    # The document's first chunk is kept once as a witness, so the two matches
    # are added beside it rather than each pulling in the whole document.
    assert [entry["chunk_id"] for entry in retained] == ["c0", "c1", "c2"]


def test_stream_target_chunks_preserves_duplicate_rows_by_position(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}]
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "contents": "the needle"},
        {"chunk_id": "c1", "document_id": "d1", "contents": "the needle"},
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    # Two rows sharing one chunk_id are two records at two line positions, so
    # the ambiguity is visible to the resolver rather than hidden by the ID.
    assert stats["exact_match_counts"] == {"t1": 2}
    assert [entry["chunk_id"] for entry in retained] == ["c1", "c1"]


def test_stream_target_chunks_retention_stays_bounded_with_thousands_of_matches(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}]
    chunks = [
        {"chunk_id": f"c{i}", "document_id": "d1", "contents": "the needle"}
        for i in range(2000)
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["exact_match_counts"] == {"t1": 2000}
    assert stats["scanned_chunk_count"] == 2000
    assert stats["retained_record_count"] == 2
    assert len(retained) == 2


def test_stream_target_chunks_skips_a_target_with_no_matching_document(
    tmp_path: Path,
) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "source_path": "missing.pdf", "snippet": "needle"}]
    chunks = [{"chunk_id": "c1", "document_id": "d1", "contents": "the needle"}]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["exact_match_counts"] == {}
    assert retained == []


def test_stream_target_chunks_falls_back_to_the_document_id(tmp_path: Path) -> None:
    documents = [{"document_id": "d1", "source_relative_path": "alpha.pdf"}]
    targets = [{"target_id": "t1", "document_id": "d1", "snippet": "needle"}]
    chunks = [{"chunk_id": "c1", "document_id": "d1", "contents": "the needle"}]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    assert stats["exact_match_counts"] == {"t1": 1}
    assert [entry["chunk_id"] for entry in retained] == ["c1"]


def test_stream_target_chunks_skips_an_ambiguous_source_metadata(
    tmp_path: Path,
) -> None:
    documents = [
        {"document_id": "d1", "source_relative_path": "alpha.pdf"},
        {"document_id": "d2", "source_relative_path": "alpha.pdf"},
    ]
    targets = [{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}]
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "contents": "a needle"},
        {"chunk_id": "c2", "document_id": "d2", "contents": "a needle"},
    ]
    retained, stats = _stream(tmp_path, documents, targets, chunks)
    # Selection is left to the authoritative resolver rather than guessed, so an
    # unidentified target is not counted here.
    assert stats["exact_match_counts"] == {}
    assert retained == []


def test_streamed_resolution_matches_the_full_generation(tmp_path: Path) -> None:
    real = _real_harness()
    generation_root = tmp_path / "generation"
    documents = [
        {
            "document_id": "d1",
            "source_path": "alpha.pdf",
            "source_relative_path": "alpha.pdf",
        },
        {
            "document_id": "d2",
            "source_path": "beta.pdf",
            "source_relative_path": "beta.pdf",
        },
    ]
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "contents": "first passage"},
        {"chunk_id": "c2", "document_id": "d1", "contents": "second passage"},
        {"chunk_id": "c3", "document_id": "d2", "contents": "third passage"},
    ]
    targets = [
        {"target_id": "t1", "source_path": "alpha.pdf", "snippet": "first passage"},
        {"target_id": "t2", "source_path": "beta.pdf", "snippet": "third passage"},
    ]
    _write_chunks(generation_root, chunks)
    (generation_root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "generation_id": "g1", "documents": documents}
        ),
        encoding="utf-8",
    )
    records, by_id = real.load_generation(generation_root)
    full = real.resolve_targets(targets, records, by_id)
    bounded, stats = audit.stream_target_chunks(generation_root, by_id, targets, real)
    streamed = real.resolve_targets(targets, bounded, by_id)
    assert streamed == full
    assert stats["exact_match_counts"] == {"t1": 1, "t2": 1}


# --- corpus_pool -----------------------------------------------------------


def _build_pool(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    generation_id: str = "g1",
) -> SimpleNamespace:
    """A controlled generation and judged set, resolved by the shipped harness.

    Only the generation pointer and the judged set are supplied here. Normalizing
    a snippet and resolving a target use the shipped harness, so a streamed pool
    is held to the same rule a real run uses.
    """

    config = resolve_config(project, vanilla_executable=sys.executable)
    generation_root = tmp_path / "generation"
    _write_chunks(generation_root, chunks)
    manifest = {"generation_id": generation_id, "documents": documents}
    judged = {"schema_version": 1, "targets": targets, "queries": queries}
    real = _real_harness()
    harness = SimpleNamespace(
        load_judgments=lambda _path: judged,
        normalize=real.normalize,
        _chunk_text=real._chunk_text,
        resolve_targets=real.resolve_targets,
        EvaluationError=real.EvaluationError,
    )
    monkeypatch.setattr(audit, "evaluation_harness", lambda: harness)
    monkeypatch.setattr(
        audit,
        "load_current_generation",
        lambda _state_root: (generation_root, manifest),
    )
    return SimpleNamespace(
        config=config,
        judgments_path=tmp_path / "judgments.json",
        generation_root=generation_root,
        harness=harness,
        judged=judged,
        documents=documents,
    )


@pytest.fixture
def pool_setup(project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A controlled generation and judged set, resolved by the shipped harness."""

    return _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[
            {
                "document_id": "d1",
                "source_path": "alpha.pdf",
                "source_relative_path": "alpha.pdf",
            },
            {
                "document_id": "d2",
                "source_path": "beta.pdf",
                "source_relative_path": "beta.pdf",
            },
        ],
        chunks=[
            {"chunk_id": "c1", "document_id": "d1", "contents": "first passage"},
            {"chunk_id": "c2", "document_id": "d2", "contents": "second passage"},
        ],
        targets=[
            {"target_id": "t1", "source_path": "alpha.pdf", "snippet": "first passage"},
            {"target_id": "t2", "source_path": "beta.pdf", "snippet": "second passage"},
        ],
        queries=[
            {"target_id": "t1", "query": "First question?", "class": "quote"},
            {"target_id": "t2", "query": "Second question?", "class": "quote"},
        ],
    )


def test_corpus_pool_returns_resolved_targets_and_filtered_queries(
    pool_setup: Any,
) -> None:
    pool = audit.corpus_pool(pool_setup.config, pool_setup.judgments_path, frozenset())
    assert set(pool["targets"]) == {"t1", "t2"}
    assert pool["manifest"]["generation_id"] == "g1"
    assert [query["target_id"] for query in pool["queries"]] == ["t1", "t2"]


def test_corpus_pool_drops_a_skipped_target_and_its_query(pool_setup: Any) -> None:
    pool = audit.corpus_pool(
        pool_setup.config, pool_setup.judgments_path, frozenset({"t2"})
    )
    assert set(pool["targets"]) == {"t1"}
    assert [query["target_id"] for query in pool["queries"]] == ["t1"]


def test_corpus_pool_refuses_a_skip_that_names_no_judged_target(
    pool_setup: Any,
) -> None:
    with pytest.raises(ValueError, match="does not name an existing judged target"):
        audit.corpus_pool(
            pool_setup.config, pool_setup.judgments_path, frozenset({"absent"})
        )


def test_corpus_pool_refuses_a_target_the_reviewer_excluded_as_a_source(
    pool_setup: Any,
) -> None:
    write_source_exclusions(
        pool_setup.config.source_exclusions_path,
        {"alpha.pdf": {"reason": "Duplicate copy.", "excluded_at": "2026-01-01"}},
    )
    with pytest.raises(ValueError, match="reviewed as excluded"):
        audit.corpus_pool(pool_setup.config, pool_setup.judgments_path, frozenset())


def test_corpus_pool_refuses_a_target_the_reviewer_excluded_as_a_passage(
    pool_setup: Any,
) -> None:
    write_chunk_exclusions(
        pool_setup.config.chunk_exclusions_path,
        {
            "c2": {
                "reason": "Withheld passage.",
                "excluded_at": "2026-01-01",
                "generation_id": "g1",
            }
        },
    )
    with pytest.raises(ValueError, match="reviewed as excluded"):
        audit.corpus_pool(pool_setup.config, pool_setup.judgments_path, frozenset())


def test_corpus_pool_does_not_stream_a_skipped_target(pool_setup: Any) -> None:
    pool = audit.corpus_pool(
        pool_setup.config, pool_setup.judgments_path, frozenset({"t2"})
    )
    # The skip is applied before streaming, so a skipped target is neither
    # resolved nor counted among the streamed matches.
    assert pool["target_resolution"]["exact_match_counts"] == {"t1": 1}


def test_corpus_pool_resolves_a_target_through_the_document_id_fallback(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[
            {"document_id": "d1", "source_relative_path": "alpha.pdf"},
        ],
        chunks=[{"chunk_id": "c1", "document_id": "d1", "contents": "the needle"}],
        targets=[{"target_id": "t1", "document_id": "d1", "snippet": "needle"}],
        queries=[{"target_id": "t1", "query": "Which?", "class": "quote"}],
    )
    pool = audit.corpus_pool(setup.config, setup.judgments_path, frozenset())
    assert pool["targets"]["t1"]["chunk_id"] == "c1"
    assert pool["target_resolution"]["exact_match_counts"] == {"t1": 1}


def test_corpus_pool_reports_a_missing_source_with_streamed_counts(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[{"document_id": "d1", "source_relative_path": "alpha.pdf"}],
        chunks=[{"chunk_id": "c1", "document_id": "d1", "contents": "the needle"}],
        targets=[
            {"target_id": "t1", "source_path": "missing.pdf", "snippet": "needle"}
        ],
        queries=[{"target_id": "t1", "query": "Which?", "class": "quote"}],
    )
    with pytest.raises(setup.harness.EvaluationError) as info:
        audit.corpus_pool(setup.config, setup.judgments_path, frozenset())
    message = str(info.value)
    assert "bounded retained records" in message
    assert "Exact streamed snippet match counts" in message
    assert "no document in this generation matches" in message
    assert "Exact streamed snippet match counts: {}" in message


def test_corpus_pool_reports_a_snippet_that_matches_no_chunk(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[{"document_id": "d1", "source_relative_path": "alpha.pdf"}],
        chunks=[{"chunk_id": "c1", "document_id": "d1", "contents": "the needle"}],
        targets=[{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "absent"}],
        queries=[{"target_id": "t1", "query": "Which?", "class": "quote"}],
    )
    with pytest.raises(setup.harness.EvaluationError) as info:
        audit.corpus_pool(setup.config, setup.judgments_path, frozenset())
    message = str(info.value)
    assert "snippet resolves to 0 chunks" in message
    assert '"t1": 0' in message


def test_corpus_pool_refuses_duplicate_matching_rows_even_with_one_chunk_id(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[{"document_id": "d1", "source_relative_path": "alpha.pdf"}],
        chunks=[
            {"chunk_id": "c1", "document_id": "d1", "contents": "the needle"},
            {"chunk_id": "c1", "document_id": "d1", "contents": "the needle"},
        ],
        targets=[{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}],
        queries=[{"target_id": "t1", "query": "Which?", "class": "quote"}],
    )
    with pytest.raises(setup.harness.EvaluationError) as info:
        audit.corpus_pool(setup.config, setup.judgments_path, frozenset())
    message = str(info.value)
    # Both rows are retained as separate records, so the resolver sees the
    # ambiguity and the streamed count states how many matches it refused.
    assert "bounded retained records" in message
    assert '"t1": 2' in message


def test_corpus_pool_reports_ambiguous_source_metadata(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _build_pool(
        project,
        monkeypatch,
        tmp_path,
        documents=[
            {"document_id": "d1", "source_relative_path": "alpha.pdf"},
            {"document_id": "d2", "source_relative_path": "alpha.pdf"},
        ],
        chunks=[
            {"chunk_id": "c1", "document_id": "d1", "contents": "a needle"},
            {"chunk_id": "c2", "document_id": "d2", "contents": "a needle"},
        ],
        targets=[{"target_id": "t1", "source_path": "alpha.pdf", "snippet": "needle"}],
        queries=[{"target_id": "t1", "query": "Which?", "class": "quote"}],
    )
    with pytest.raises(setup.harness.EvaluationError) as info:
        audit.corpus_pool(setup.config, setup.judgments_path, frozenset())
    message = str(info.value)
    assert "2 documents match" in message
    assert "Exact streamed snippet match counts: {}" in message
