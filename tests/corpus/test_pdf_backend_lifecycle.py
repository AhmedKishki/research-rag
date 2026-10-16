"""The warm worker's scope, reuse, failures, and retirement, without Docling.

A real subprocess speaks the worker's line protocol in place of the managed
interpreter, and the scope prefix is stripped, so pipes, waits, and exits are
real while no systemd scope is created. A job's file name picks its behaviour.
"""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from research_rag.corpus import docling_worker, pdf_backend, scoped_child
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.pdf_backend import (
    DoclingConversionError,
    DoclingResourceError,
    DoclingUnavailableError,
    DoclingWorker,
    PdfBackendError,
    prefetch_docling_models,
    run_docling_batch,
)

_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={"docling": "2.135.0"},
)

# `ok` answers, `flaky` fails once this process has answered a job, `codeN`
# replies failure N and exits N, `die` exits 9 without a reply, and `hang` never
# answers.
_FAKE_WORKER = """
import json, os, sys, time
served = 0
for line in iter(sys.stdin.readline, ""):
    job = json.loads(line)
    name = os.path.basename(job.get("source_path", "")).removesuffix(".pdf")
    if name == "die":
        print("worker diagnostics", file=sys.stderr)
        sys.exit(9)
    if name == "hang":
        time.sleep(60)
    code = int(name[4:]) if name.startswith("code") else 0
    if name == "flaky" and served:
        code = 3
    if code:
        print(json.dumps({"ok": False, "code": code, "error": name}), flush=True)
        sys.exit(code)
    served += 1
    print(json.dumps({"ok": True, "result": {"pid": os.getpid(), "job": job}}), flush=True)
"""


@pytest.fixture
def scopes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Launch the fake worker for every scope, and record launches and kills."""

    launched: list[dict[str, Any]] = []
    retired: list[str] = []
    popen = subprocess.Popen

    def launch(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        launched.append({"argv": list(argv), "env": kwargs.get("env") or {}})
        return popen([sys.executable, "-c", _FAKE_WORKER], **kwargs)

    def terminate(unit: str, process: subprocess.Popen[bytes]) -> None:
        retired.append(unit)
        process.kill()
        process.wait()

    monkeypatch.setattr(pdf_backend.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(scoped_child.subprocess, "Popen", launch)
    monkeypatch.setattr(scoped_child, "terminate_scope", terminate)
    return SimpleNamespace(launched=launched, retired=retired)


def _batch(worker: DoclingWorker | None, name: str, **options: Any) -> dict[str, Any]:
    return run_docling_batch(
        source_path=Path(f"/tmp/{name}.pdf"),
        start=1,
        end=8,
        offline=True,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
        worker=worker,
        **options,
    )


def _worker() -> DoclingWorker:
    return DoclingWorker(docling_env=_FAKE_ENV, offline=True, model_cache_root=None)


def test_launch_requests_the_exact_resource_boundary(scopes) -> None:
    _batch(None, "ok")

    command = scopes.launched[0]["argv"]
    assert "--scope" in command
    assert f"--property=MemoryMax={2 * 1024**3}" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=CPUQuota=200%" in command
    assert f"--property=RuntimeMaxSec={pdf_backend.WORKER_LIFETIME_SECONDS}" in command
    # The managed interpreter runs isolated and executes the shipped worker file.
    assert str(_FAKE_ENV.python) in command
    assert "-I" in command
    assert command[-1].endswith("docling_worker.py")
    assert "site-packages" not in command[-1]


def test_one_worker_serves_every_batch_until_it_is_closed(scopes) -> None:
    with _worker() as worker:
        assert scopes.launched == []
        first = _batch(worker, "ok")
        second = _batch(worker, "ok")
        assert first["pid"] == second["pid"]
        assert len(scopes.launched) == 1
        assert scopes.retired == []
    assert len(scopes.retired) == 1

    # A batch without a worker gets its own, retired when it returns.
    _batch(None, "ok")
    assert len(scopes.launched) == len(scopes.retired) == 2


def test_a_batch_that_could_outlive_the_scope_starts_a_fresh_worker(
    scopes, monkeypatch
) -> None:
    monkeypatch.setattr(pdf_backend, "WORKER_LIFETIME_SECONDS", 0)
    with _worker() as worker:
        first = _batch(worker, "ok", timeout=5)
        second = _batch(worker, "ok", timeout=5)

    assert first["pid"] != second["pid"]
    assert "--property=RuntimeMaxSec=5" in scopes.launched[0]["argv"]
    assert len(scopes.retired) == 2


@pytest.mark.parametrize(
    ("name", "expected", "message"),
    [
        ("code5", DoclingResourceError, "code5"),
        ("code6", DoclingUnavailableError, "code6"),
        ("code3", DoclingConversionError, "code3"),
        ("code1", PdfBackendError, "exit 1"),
        ("die", PdfBackendError, "worker diagnostics"),
    ],
)
def test_worker_failures_map_to_distinct_errors(
    scopes, name: str, expected: type[Exception], message: str
) -> None:
    with pytest.raises(expected, match=message):
        _batch(None, name)

    # A fresh worker's failure is reported, not retried, and its scope is retired.
    assert len(scopes.launched) == len(scopes.retired) == 1


def test_a_warm_worker_failure_is_decided_by_a_fresh_worker(scopes) -> None:
    with _worker() as worker:
        first = _batch(worker, "ok")
        retried = _batch(worker, "flaky")
        assert retried["pid"] != first["pid"]
        assert len(scopes.retired) == 1

        # A failure the fresh worker repeats is the batch's own.
        with pytest.raises(DoclingConversionError):
            _batch(worker, "code3")
    assert len(scopes.launched) == len(scopes.retired) == 3


def test_timeout_kills_the_scope_and_is_not_retried(scopes) -> None:
    with _worker() as worker:
        _batch(worker, "ok")
        with pytest.raises(DoclingResourceError, match="0.5s wall limit"):
            _batch(worker, "hang", timeout=0.5)
        assert len(scopes.launched) == len(scopes.retired) == 1
        # The next batch starts a fresh worker.
        _batch(worker, "ok")
    assert len(scopes.launched) == len(scopes.retired) == 2


def test_an_interrupted_batch_retires_the_worker(scopes, monkeypatch) -> None:
    def interrupt(_line: bytes) -> dict[str, Any]:
        raise KeyboardInterrupt

    parse = pdf_backend._parse_reply
    monkeypatch.setattr(pdf_backend, "_parse_reply", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _batch(_worker(), "ok")

    assert len(scopes.retired) == 1
    # The process-wide worker slot is free again.
    monkeypatch.setattr(pdf_backend, "_parse_reply", parse)
    assert _batch(None, "ok")["job"]["start"] == 1


def test_prefetch_launches_online_and_passes_the_mode(scopes, monkeypatch) -> None:
    installation_options = []

    def ensure(config, **options):
        installation_options.append(options)
        return _FAKE_ENV

    monkeypatch.setattr(pdf_backend, "ensure_docling_env", ensure)

    result = prefetch_docling_models(
        SimpleNamespace(model_cache_root=None, offline=True)
    )

    assert result["job"]["mode"] == "prefetch"
    assert "HF_HUB_OFFLINE" not in scopes.launched[0]["env"]
    assert installation_options == [{"allow_install": True}]
    assert len(scopes.retired) == 1


def test_worker_requires_scope_cleanup_executable(monkeypatch):
    monkeypatch.setattr(
        pdf_backend.shutil,
        "which",
        lambda name: "/bin/systemd-run" if name == "systemd-run" else None,
    )
    with pytest.raises(pdf_backend.DoclingUnavailableError, match="systemctl"):
        pdf_backend._require_systemd_run()


def test_converter_explicitly_disables_enrichments_and_gpu(monkeypatch):
    captured = {}

    def converter(**kwargs):
        captured.update(kwargs)
        return captured

    modules = {
        "docling.datamodel.base_models": {"InputFormat": SimpleNamespace(PDF="pdf")},
        "docling.datamodel.pipeline_options": {
            "PdfPipelineOptions": lambda: SimpleNamespace(
                accelerator_options=SimpleNamespace()
            )
        },
        "docling.document_converter": {
            "DocumentConverter": converter,
            "PdfFormatOption": lambda **kwargs: SimpleNamespace(**kwargs),
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    docling_worker._build_converter()
    options = captured["format_options"]["pdf"].pipeline_options
    for flag in (
        "do_ocr",
        "do_table_structure",
        "do_code_enrichment",
        "do_formula_enrichment",
        "do_picture_classification",
        "do_picture_description",
        "do_chart_extraction",
        "generate_page_images",
        "generate_picture_images",
        "generate_table_images",
        "enable_remote_services",
    ):
        assert getattr(options, flag) is False
    assert options.accelerator_options.device == "cpu"
    assert options.accelerator_options.num_threads == 2
