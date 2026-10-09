"""The PDF backend registry and its bounded worker, without loading Docling."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research_rag.corpus import docling_worker, pdf_backend
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.pdf_backend import (
    CUSTOM_BACKEND_FINGERPRINT,
    DoclingConversionError,
    DoclingResourceError,
    DoclingUnavailableError,
    PdfBackendError,
    effective_maximum_unclean_percent,
    pdf_backend_fingerprint,
    prefetch_docling_models,
    recorded_pdf_backend_fingerprint,
    run_docling_batch,
)
from research_rag.project.support import DEFAULT_PDF_BACKEND_FINGERPRINT

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


def test_custom_is_the_documented_legacy_default() -> None:
    assert CUSTOM_BACKEND_FINGERPRINT == DEFAULT_PDF_BACKEND_FINGERPRINT
    assert pdf_backend_fingerprint("custom") == DEFAULT_PDF_BACKEND_FINGERPRINT
    # A manifest that predates the feature carries the same fingerprint.
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


def test_custom_record_is_unchanged_and_records_no_components() -> None:
    record = pdf_backend.pdf_backend_record("custom")

    assert record["fingerprint"] == CUSTOM_BACKEND_FINGERPRINT
    assert record["components"] == {}
    assert record["adapter_policy_version"] is None


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


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"{}",
        stderr: bytes = b"",
        timeout: bool = False,
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._timeout = timeout
        self.pid = 4242

    def communicate(self, *_args: object, **_kwargs: object):
        if self._timeout:
            raise subprocess.TimeoutExpired(cmd="systemd-run", timeout=1)
        return self._stdout, self._stderr

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def _arm_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_backend.shutil, "which", lambda name: f"/usr/bin/{name}")
    # The launcher retires the scope in a finally; tests that fake Popen must not
    # let the real systemctl path run.
    monkeypatch.setattr(pdf_backend, "_terminate_scope", lambda unit, process: None)


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (5, DoclingResourceError),
        (6, DoclingUnavailableError),
        (3, DoclingConversionError),
        (1, PdfBackendError),
    ],
)
def test_worker_exit_codes_map_to_distinct_failures(
    monkeypatch: pytest.MonkeyPatch, returncode: int, expected: type[Exception]
) -> None:
    _arm_runner(monkeypatch)
    monkeypatch.setattr(
        pdf_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(returncode=returncode, stderr=b"boom"),
    )

    with pytest.raises(expected):
        run_docling_batch(
            source_path=Path("/tmp/source.pdf"),
            start=1,
            end=1,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
        )


def test_timeout_kills_the_scope_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _arm_runner(monkeypatch)
    terminated: list[str] = []
    monkeypatch.setattr(
        pdf_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(timeout=True),
    )
    monkeypatch.setattr(
        pdf_backend, "_terminate_scope", lambda unit, process: terminated.append(unit)
    )

    with pytest.raises(DoclingResourceError):
        run_docling_batch(
            source_path=Path("/tmp/source.pdf"),
            start=1,
            end=8,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
        )

    assert len(terminated) == 1


def test_launch_requests_the_exact_resource_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _arm_runner(monkeypatch)
    commands: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        commands.append(list(command))
        return _FakeProcess(returncode=0, stdout=b"{}")

    monkeypatch.setattr(pdf_backend.subprocess, "Popen", fake_popen)

    run_docling_batch(
        source_path=Path("/tmp/source.pdf"),
        start=1,
        end=8,
        offline=False,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
    )

    command = commands[0]
    assert "--scope" in command
    assert f"--property=MemoryMax={2 * 1024**3}" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=CPUQuota=200%" in command
    # The managed interpreter runs isolated and executes the shipped worker file.
    assert str(_FAKE_ENV.python) in command
    assert "-I" in command
    assert command[-1].endswith("docling_worker.py")
    assert "site-packages" not in command[-1]


def test_worker_result_is_returned_when_the_worker_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _arm_runner(monkeypatch)
    monkeypatch.setattr(
        pdf_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(
            returncode=0, stdout=b'{"version": "2.135.0", "pages": {}}'
        ),
    )

    result = run_docling_batch(
        source_path=Path("/tmp/source.pdf"),
        start=1,
        end=1,
        offline=False,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
    )

    assert result["version"] == "2.135.0"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_RAG_MANAGED_CHILD", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    # Offline extraction needs the no-download markers; an online extraction and
    # provisioning must not carry them.
    with pytest.raises(docling_worker.ResourceBoundaryError):
        docling_worker._verify_environment("extract", offline=True)
    docling_worker._verify_environment("extract", offline=False)
    docling_worker._verify_environment("prefetch", offline=False)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    docling_worker._verify_environment("extract", offline=True)
    with pytest.raises(docling_worker.ResourceBoundaryError):
        docling_worker._verify_environment("extract", offline=False)
    with pytest.raises(docling_worker.ResourceBoundaryError):
        docling_worker._verify_environment("prefetch", offline=False)


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


def test_prefetch_launches_online_and_passes_the_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _arm_runner(monkeypatch)
    captured: list[bytes] = []

    class _CapturingProcess(_FakeProcess):
        def __init__(self) -> None:
            super().__init__(returncode=0, stdout=b'{"ready": true}')

        def communicate(self, payload: bytes, *_args: object, **_kwargs: object):
            captured.append(payload)
            return self._stdout, self._stderr

    def fake_popen(_command, **kwargs):
        environment = kwargs.get("env") or {}
        assert "HF_HUB_OFFLINE" not in environment
        return _CapturingProcess()

    monkeypatch.setattr(pdf_backend.subprocess, "Popen", fake_popen)
    installation_options = []

    def ensure(config, **options):
        installation_options.append(options)
        return _FAKE_ENV

    monkeypatch.setattr(pdf_backend, "ensure_docling_env", ensure)

    result = prefetch_docling_models(
        SimpleNamespace(model_cache_root=None, offline=True)
    )

    assert result["ready"] is True
    assert installation_options == [{"allow_install": True}]
    assert b'"mode": "prefetch"' in captured[0]


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
