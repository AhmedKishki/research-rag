"""The PDF backend registry and its bounded worker, without loading Docling."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research_rag.corpus import docling_worker, pdf_backend
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.pdf_backend import (
    CUSTOM_BACKEND_FINGERPRINT,
    DoclingUnavailableError,
    effective_maximum_unclean_percent,
    legacy_custom_backend_record,
    normalize_pdf_backend,
    pdf_backend_fingerprint,
    recorded_pdf_backend_fingerprint,
    run_docling_batch,
)
from research_rag.project.policy import DEFAULT_PDF_BACKEND_FINGERPRINT

_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={
        "docling": "2.135.0",
        "docling-core": "2.0.0",
        "rapidocr": "3.9.1",
        "pymupdf": "1.26.0",
    },
)


def test_custom_is_a_read_alias_that_keeps_the_legacy_record() -> None:
    # The retired token resolves to the operational reader, so an existing config
    # keeps loading and a new build targets Docling.
    assert normalize_pdf_backend("custom") == "docling"
    assert pdf_backend_fingerprint("custom") == pdf_backend_fingerprint("docling")

    # The historical record is a preserved constant, never relabelled: a manifest
    # that predates the field and one that recorded 'custom' keep their identity.
    assert CUSTOM_BACKEND_FINGERPRINT == DEFAULT_PDF_BACKEND_FINGERPRINT
    assert (
        legacy_custom_backend_record()["fingerprint"] == DEFAULT_PDF_BACKEND_FINGERPRINT
    )
    assert recorded_pdf_backend_fingerprint({}) == DEFAULT_PDF_BACKEND_FINGERPRINT
    assert (
        recorded_pdf_backend_fingerprint({"extraction_backend": {"backend": "custom"}})
        == DEFAULT_PDF_BACKEND_FINGERPRINT
    )


def _record_with_env(
    monkeypatch: pytest.MonkeyPatch, env: DoclingEnv
) -> dict[str, Any]:
    monkeypatch.setattr(pdf_backend, "resolve_docling_env", lambda _config: env)
    return pdf_backend.pdf_backend_record("docling", SimpleNamespace())


def test_docling_fingerprint_tracks_managed_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = _record_with_env(monkeypatch, _FAKE_ENV)
    assert pinned["version"] == "2.135.0"
    assert pinned["components"]["docling-core"] == "2.0.0"

    upgraded = _record_with_env(
        monkeypatch,
        DoclingEnv(
            root=Path("/nonexistent/docling-env"),
            spec_fingerprint="spec-test",
            versions={**_FAKE_ENV.versions, "docling": "2.140.0"},
        ),
    )
    assert pinned["fingerprint"] != upgraded["fingerprint"]


def test_docling_record_exposes_adapter_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record_with_env(monkeypatch, _FAKE_ENV)

    assert (
        record["adapter_policy_version"] == pdf_backend.DOCLING_ADAPTER_POLICY_VERSION
    )
    assert record["components"]["rapidocr"] == "3.9.1"


def test_docling_fingerprint_without_env_uses_the_stable_spec() -> None:
    first = pdf_backend.pdf_backend_record("docling")
    second = pdf_backend.pdf_backend_record("docling")

    assert first["fingerprint"] == second["fingerprint"]
    assert first["components"] == {}
    assert first["version"] is None


def test_the_legacy_record_is_preserved_and_the_operational_record_is_docling() -> None:
    legacy = legacy_custom_backend_record()

    assert legacy["fingerprint"] == CUSTOM_BACKEND_FINGERPRINT
    assert legacy["backend"] == "custom"
    assert legacy["components"] == {}
    assert legacy["adapter_policy_version"] is None

    # The active record path never emits the retired token, even when handed it.
    record = pdf_backend.pdf_backend_record("custom")
    assert record["backend"] == "docling"
    assert (
        record["adapter_policy_version"] == pdf_backend.DOCLING_ADAPTER_POLICY_VERSION
    )


def test_the_shared_budget_is_the_projects_configured_one() -> None:
    # The configured budget is shared unchanged: 2% stays 2%, the default stays 1%.
    assert effective_maximum_unclean_percent("custom", 2.0) == 2.0
    assert effective_maximum_unclean_percent("docling", 2.0) == 2.0
    assert effective_maximum_unclean_percent("docling", 1.0) == 1.0


def _cgroup(tmp_path: Path, *, memory: str, swap: str, cpu: str) -> Path:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(memory, encoding="utf-8")
    (root / "memory.swap.max").write_text(swap, encoding="utf-8")
    (root / "cpu.max").write_text(cpu, encoding="utf-8")
    return root


def test_worker_verifies_exact_boundary_before_import(tmp_path: Path) -> None:
    root = _cgroup(
        tmp_path,
        memory=str(2 * 1024**3),
        swap="0",
        cpu="200000 100000",
    )

    readback = docling_worker.verify_resource_boundary(root)

    assert readback == {
        "memory.max": str(2 * 1024**3),
        "memory.swap.max": "0",
        "cpu.max": "200000 100000",
    }


@pytest.mark.parametrize(
    ("memory", "swap", "cpu"),
    [
        (str(1024**3), "0", "200000 100000"),
        (str(2 * 1024**3), "max", "200000 100000"),
        (str(2 * 1024**3), "0", "100000 100000"),
        (str(2 * 1024**3), "0", "max 100000"),
    ],
)
def test_worker_fails_closed_on_a_wrong_boundary(
    tmp_path: Path, memory: str, swap: str, cpu: str
) -> None:
    root = _cgroup(tmp_path, memory=memory, swap=swap, cpu=cpu)

    with pytest.raises(docling_worker.ResourceBoundaryError):
        docling_worker.verify_resource_boundary(root)


def test_worker_fails_closed_when_a_boundary_file_is_missing(tmp_path: Path) -> None:
    root = _cgroup(tmp_path, memory=str(2 * 1024**3), swap="0", cpu="200000 100000")
    (root / "cpu.max").unlink()

    with pytest.raises(docling_worker.ResourceBoundaryError):
        docling_worker.verify_resource_boundary(root)


@pytest.mark.parametrize(
    ("message", "missing"),
    [
        ("We couldn't connect to huggingface.co (offline mode)", True),
        ("LocalEntryNotFoundError: not found in cache", True),
        ("Cannot find model weights", True),
        ("Page 3 has an unsupported content stream", False),
    ],
)
def test_offline_model_errors_are_recognised(message: str, missing: bool) -> None:
    assert docling_worker._looks_like_a_missing_model(message) is missing


def test_missing_systemd_runner_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_backend.shutil, "which", lambda name: None)

    with pytest.raises(DoclingUnavailableError) as error:
        run_docling_batch(
            source_path=Path("/tmp/does-not-matter.pdf"),
            start=1,
            end=1,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
        )

    assert "systemd-run" in str(error.value)


def test_extraction_environment_follows_offline() -> None:
    offline = pdf_backend._worker_environment("extract", True, None)
    assert offline["HF_HUB_OFFLINE"] == "1"
    assert offline["TRANSFORMERS_OFFLINE"] == "1"

    # An online extraction may fetch lazily, so it carries no no-download marker.
    online = pdf_backend._worker_environment("extract", False, None)
    assert "HF_HUB_OFFLINE" not in online
    assert "TRANSFORMERS_OFFLINE" not in online

    provisioning = pdf_backend._worker_environment("prefetch", False, None)
    assert "HF_HUB_OFFLINE" not in provisioning
    assert "TRANSFORMERS_OFFLINE" not in provisioning


def test_remedy_points_at_provisioning_not_a_manual_install() -> None:
    remedy = pdf_backend.DOCLING_PACKAGE_REMEDY
    assert "doctor --prefetch-models" in remedy
    assert "rapidocr" in remedy
    assert "research-rag[docling]" not in remedy
    assert "pip install docling==" not in remedy


def test_worker_environment_mode_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def verify(mode: str, offline: bool, environment: dict[str, str]) -> None:
        monkeypatch.setattr(os, "environ", environment)
        docling_worker._verify_environment(mode, offline=offline)

    # The worker accepts exactly the environment its parent builds for each mode.
    for mode, offline in (("extract", True), ("extract", False), ("prefetch", False)):
        verify(mode, offline, pdf_backend._worker_environment(mode, offline, tmp_path))

    # Offline extraction needs the no-download markers, an online extraction and
    # provisioning must not carry them, and a worker the app did not start refuses.
    offline = pdf_backend._worker_environment("extract", True, tmp_path)
    online = pdf_backend._worker_environment("extract", False, tmp_path)
    unmanaged = {k: v for k, v in offline.items() if k != "RESEARCH_RAG_MANAGED_CHILD"}
    for mode, flag, environment in (
        ("extract", True, online),
        ("extract", False, offline),
        ("prefetch", False, offline),
        ("extract", True, unmanaged),
    ):
        with pytest.raises(docling_worker.ResourceBoundaryError):
            verify(mode, flag, environment)


def test_cache_readiness_is_read_only(tmp_path: Path) -> None:
    assert pdf_backend.docling_cache_has_models(tmp_path) is False
    (tmp_path / "docling" / "hub").mkdir(parents=True)
    assert pdf_backend.docling_cache_has_models(tmp_path) is False
    (tmp_path / "docling" / "hub" / "model.bin").write_bytes(b"x")
    assert pdf_backend.docling_cache_has_models(tmp_path) is True


def _status(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SUCCESS", None),
        ("PARTIAL_SUCCESS", "status=PARTIAL_SUCCESS"),
        ("FAILURE", "status=FAILURE"),
    ],
)
def test_only_success_is_accepted(name: str, expected: str | None) -> None:
    result = SimpleNamespace(status=_status(name), errors=[])
    problem = docling_worker.conversion_result_problem(result)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and problem[0].startswith(expected)


def test_partial_success_with_errors_is_reported_and_classified() -> None:
    result = SimpleNamespace(
        status=_status("PARTIAL_SUCCESS"),
        errors=[
            SimpleNamespace(error_message="page 3 could not be converted"),
            SimpleNamespace(error_message="model weights not found in cache"),
        ],
    )
    problem = docling_worker.conversion_result_problem(result)

    assert problem is not None
    summary, missing = problem
    assert "PARTIAL_SUCCESS" in summary
    assert "page 3 could not be converted" in summary
    assert missing is True


def test_partial_success_is_rejected_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported: list[bool] = []

    class FakeDocument:
        def export_to_dict(self) -> dict[str, Any]:
            exported.append(True)
            return {}

    class FakeConverter:
        def convert(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                status=_status("PARTIAL_SUCCESS"),
                errors=[SimpleNamespace(error_message="page 2 failed")],
                document=FakeDocument(),
            )

    monkeypatch.setattr(
        docling_worker, "verify_resource_boundary", lambda: {"memory.max": "x"}
    )
    monkeypatch.setattr(
        docling_worker, "_verify_environment", lambda mode, offline: None
    )
    monkeypatch.setattr(docling_worker, "apply_and_verify_nice", lambda: 10)
    monkeypatch.setattr(docling_worker, "_docling_version", lambda: "2.135.0")
    monkeypatch.setattr(docling_worker, "_build_converter", lambda: FakeConverter())

    with pytest.raises(docling_worker.WorkerError):
        docling_worker.run(
            {"mode": "extract", "source_path": "/tmp/x.pdf", "start": 1, "end": 2}
        )

    assert exported == []


def _armed_session(monkeypatch: pytest.MonkeyPatch, converter: Any) -> list[str]:
    """Stub the boundary checks and record each converter build."""

    built: list[str] = []

    def build() -> Any:
        built.append("converter")
        return converter

    monkeypatch.setattr(
        docling_worker, "verify_resource_boundary", lambda: {"memory.max": "x"}
    )
    monkeypatch.setattr(
        docling_worker, "_verify_environment", lambda mode, offline: None
    )
    monkeypatch.setattr(docling_worker, "apply_and_verify_nice", lambda: 10)
    monkeypatch.setattr(docling_worker, "_docling_version", lambda: "2.135.0")
    monkeypatch.setattr(docling_worker, "_build_converter", build)
    return built


def _serve(lines: list[str]) -> tuple[int, list[dict[str, Any]]]:
    replies = io.StringIO()
    status = docling_worker.serve(io.StringIO("".join(lines)), replies)
    return status, [json.loads(line) for line in replies.getvalue().splitlines()]


def test_serve_verifies_once_and_answers_each_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _armed_session(monkeypatch, object())
    monkeypatch.setattr(
        docling_worker,
        "_extract",
        lambda job, *_args: {"start": job["start"]},
    )
    job = {
        "protocol": pdf_backend.WORKER_PROTOCOL,
        "mode": "extract",
        "source_path": "/tmp/x.pdf",
        "offline": True,
    }

    status, replies = _serve(
        [json.dumps({**job, "start": start}) + "\n" for start in (1, 9)]
    )

    assert status == 0
    assert built == ["converter"]
    assert replies == [
        {"ok": True, "result": {"start": 1}},
        {"ok": True, "result": {"start": 9}},
    ]


def test_serve_stops_after_the_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _armed_session(monkeypatch, object())
    monkeypatch.setattr(docling_worker, "_extract", lambda job, *_args: {})
    job = {
        "protocol": pdf_backend.WORKER_PROTOCOL,
        "mode": "extract",
        "source_path": "/tmp/x.pdf",
        "start": 1,
    }

    # A job that differs from the verified mode needs a new worker.
    status, replies = _serve(
        [
            json.dumps({**job, "offline": True}) + "\n",
            json.dumps({**job, "offline": False}) + "\n",
            json.dumps({**job, "offline": True}) + "\n",
        ]
    )
    assert status == 5
    assert [reply["ok"] for reply in replies] == [True, False]
    assert replies[1]["code"] == 5

    status, replies = _serve(["not json\n", json.dumps(job) + "\n"])
    assert status == 2
    assert len(replies) == 1


def test_a_job_from_an_older_app_is_refused_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An app started before the worker file changed sends a job with no protocol.
    # Exit status 2 stops its build; 3 would omit the source as unreadable.
    built = _armed_session(monkeypatch, object())
    legacy = {"mode": "extract", "source_path": "/tmp/x.pdf", "start": 1, "end": 8}

    status, replies = _serve([json.dumps(legacy)])

    assert status == 2
    assert "Restart the app" in replies[0]["error"]
    assert built == []
    assert docling_worker.PROTOCOL == pdf_backend.WORKER_PROTOCOL


def test_image_only_probe_splits_blank_raster_and_text_pages(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "media.pdf"
    document = pymupdf.open()
    document.new_page()  # true blank: no image, no text
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
    pixmap.set_rect(pixmap.irect, (200, 120, 40))
    raster = document.new_page()
    raster.insert_image(raster.rect, pixmap=pixmap)  # raster, no text layer
    with_text = document.new_page()
    with_text.insert_image(with_text.rect, pixmap=pixmap)
    with_text.insert_text((72, 72), "OCR text layer")  # raster plus text
    document.save(str(path))
    document.close()

    _metadata, _labels, image_only = docling_worker._pdf_metadata_and_labels(path, 1, 3)

    # Only the raster page with no native text is image-only; the true blank is
    # not an image, and the raster with a text layer is not counted.
    assert image_only == 1
