"""The managed Docling environment: install once, reuse, fail closed offline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import research_rag.corpus.docling_env as env_module
from research_rag.corpus.docling_env import (
    CONTROLLER_SOURCE,
    DoclingEnv,
    DoclingEnvironmentError,
    backend_root,
    ensure_docling_env,
    env_root,
    resolve_docling_env,
    spec_fingerprint,
    version_satisfies,
    worker_command,
    worker_path,
)
from research_rag.project.config import resolve_config

_VERSIONS = {
    "docling": "2.135.0",
    "docling-core": "2.0.0",
    "docling-parse": "4.0.0",
    "docling-ibm-models": "3.0.0",
    "rapidocr": "3.9.1",
    "pymupdf": "1.26.0",
}


def _config(project: Path, *, offline: bool = False):
    overrides = ["runtime.offline=true"] if offline else []
    return resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_cache_root=project / "runtime-cache",
        settings_overrides=overrides,
    )


def _materialize_env(staging: Path, versions: dict[str, str] | None = None) -> None:
    (staging / "bin").mkdir(parents=True, exist_ok=True)
    (staging / "bin" / "python").write_text("", encoding="utf-8")
    site = (
        staging
        / "lib"
        / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        / "site-packages"
    )
    site.mkdir(parents=True, exist_ok=True)
    for name, version in (versions or _VERSIONS).items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Name: {name}\nVersion: {version}\n", encoding="utf-8"
        )


def _arm(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(env_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(
        command: list[str], *, timeout: float, env: dict[str, str], description: str
    ) -> bytes:
        calls.append(description)
        if command[1] == "venv":
            _materialize_env(Path(command[-1]))
        return b""

    monkeypatch.setattr(env_module, "_run_bounded", fake_run)
    monkeypatch.setattr(
        env_module, "_read_env_versions", lambda python, *, timeout: dict(_VERSIONS)
    )


class _FakeProcess:
    def __init__(
        self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
    ):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.pid = 4242

    def communicate(self, *_args: object, **_kwargs: object):
        return self._stdout, self._stderr

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def test_first_use_installs_once_then_reuses(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project)

    first = ensure_docling_env(config)

    assert first.root == env_root(config)
    assert first.python.is_file()
    assert json.loads((first.root / "ready.json").read_text())["ready"] is True
    assert calls == [
        "Docling environment creation",
        "Docling dependency install",
    ]

    second = ensure_docling_env(config)

    assert second.root == first.root
    assert calls == [
        "Docling environment creation",
        "Docling dependency install",
    ]


def test_offline_with_no_environment_is_forbidden(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project, offline=True)

    with pytest.raises(DoclingEnvironmentError):
        ensure_docling_env(config)

    assert calls == []
    assert not backend_root(config).exists()


def test_dependency_install_selects_cpu_pytorch_wheels(project, monkeypatch) -> None:
    _arm(monkeypatch, [])
    fake_run = env_module._run_bounded
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return fake_run(command, **kwargs)

    monkeypatch.setattr(env_module, "_run_bounded", capture)

    ensure_docling_env(_config(project))

    install = next(
        command for command in commands if command[1:3] == ["pip", "install"]
    )
    assert install[install.index("--torch-backend") + 1] == "cpu"


def test_a_ready_environment_is_reusable_offline(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _arm(monkeypatch, calls)
    installed = ensure_docling_env(_config(project))

    reused = ensure_docling_env(_config(project, offline=True))

    assert reused.root == installed.root
    assert calls == [
        "Docling environment creation",
        "Docling dependency install",
    ]


def test_failed_install_cleans_staging_and_writes_no_ready(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project)

    def failing_read(python: Path, *, timeout: float) -> dict[str, Any]:
        raise DoclingEnvironmentError("staged environment is missing packages")

    monkeypatch.setattr(env_module, "_read_env_versions", failing_read)

    with pytest.raises(DoclingEnvironmentError):
        ensure_docling_env(config)

    assert not env_root(config).exists()
    assert not list(backend_root(config).glob(".staging-*"))


def test_a_mutated_dependency_invalidates_readiness(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project)
    installed = ensure_docling_env(config)

    # Someone mutates the managed environment: the recorded docling version no
    # longer matches the dist-info on disk.
    site = next((installed.root / "lib").glob("python*/site-packages"))
    for info in site.glob("docling-*.dist-info"):
        if info.name.startswith("docling-2"):
            (info / "METADATA").write_text(
                "Name: docling\nVersion: 2.999.0\n", encoding="utf-8"
            )

    assert resolve_docling_env(config) is None


def test_an_unready_environment_is_not_resolved(project: Path) -> None:
    config = _config(project)
    final = env_root(config)
    _materialize_env(final)
    # No ready.json yet.
    assert resolve_docling_env(config) is None


def test_spec_fingerprint_is_stable() -> None:
    first = spec_fingerprint()
    assert first == spec_fingerprint()
    assert len(first) == 16


def test_worker_command_runs_the_worker_file_isolated() -> None:
    env = DoclingEnv(
        root=Path("/managed/env"),
        spec_fingerprint="spec",
        versions=dict(_VERSIONS),
    )

    command = worker_command(env)

    assert command == ["/managed/env/bin/python", "-I", str(worker_path())]
    assert command[-1].endswith("docling_worker.py")
    assert "site-packages" not in command[-1]


def test_worker_file_executes_without_the_app_package_or_a_parent_dependency() -> None:
    program = worker_path()
    literal = repr(str(program))
    child = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('numpy', 'research_rag'):\n"
        "            raise ImportError(name + ' must not be imported')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "sys.argv = ['worker']\n"
        f"namespace = {{'__name__': '__main__', '__file__': {literal}, "
        "'__package__': None}\n"
        f"exec(compile(open({literal}).read(), {literal}, 'exec'), namespace)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child],
        input=b'{"mode": "extract", "source_path": "/tmp/x.pdf", "start": 1, "end": 1, "offline": true}',
        capture_output=True,
        timeout=60,
        check=False,
    )

    # The boundary fails closed before Docling loads, and neither the app package
    # nor an unrelated parent dependency was imported.
    assert completed.returncode == 5, completed.stderr
    assert b"must not be imported" not in completed.stderr


def test_controller_source_verifies_boundary_and_niceness() -> None:
    assert "memory.max" in CONTROLLER_SOURCE
    assert "memory.swap.max" in CONTROLLER_SOURCE
    assert "cpu.max" in CONTROLLER_SOURCE
    assert "os.nice" in CONTROLLER_SOURCE
    assert "os.execv" in CONTROLLER_SOURCE


def test_version_specs_are_exact_and_bounded() -> None:
    assert version_satisfies("2.135.0", "==2.135.0")
    assert not version_satisfies("2.136.0", "==2.135.0")
    assert version_satisfies("1.26.5", ">=1.26,<2")
    assert not version_satisfies("2.0.0", ">=1.26,<2")


def test_install_lock_does_not_deadlock_the_run_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        env_module.subprocess, "Popen", lambda *a, **k: _FakeProcess(returncode=0)
    )
    monkeypatch.setattr(env_module, "_terminate_scope", lambda unit, process: None)
    config = _config(project)
    backend_root(config).mkdir(parents=True, exist_ok=True)

    # Holding the install lock, a real _run_bounded step must still complete: the
    # run lock is separate, so this would hang under a shared lock.
    with env_module._install_lock(config):
        result = env_module._run_bounded(
            ["uv", "venv", "/tmp/staging"],
            timeout=5,
            env={},
            description="probe",
        )

    assert result == b""


def test_custom_never_builds_the_environment(project: Path) -> None:
    from research_rag.corpus import pdf_backend

    config = _config(project)

    assert pdf_backend.pdf_backend_available("custom", config) is True
    assert pdf_backend.pdf_backend_fingerprint("custom", config)
    assert not backend_root(config).exists()


def test_identity_is_stable_across_first_use_and_reuse(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_rag.corpus import pdf_backend
    from research_rag.project.support import _checkpoint_identity
    from research_rag.retrieval.embeddings import resolve_embedding_model

    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project)
    ensure_docling_env(config)
    fingerprint = pdf_backend.pdf_backend_fingerprint("docling", config)

    ensure_docling_env(config)
    assert pdf_backend.pdf_backend_fingerprint("docling", config) == fingerprint

    model = resolve_embedding_model(config.settings.embedding_model)
    common = {
        "project_id": config.project_id,
        "inventory": [],
        "exclusion_revision": "none",
        "baseline_generation_id": None,
        "chunk_size": 100,
        "chunk_overlap": 10,
        "chunk_headers": False,
        "force_recompute": False,
        "embedding": model,
        "maximum_unclean_percent": 1.0,
    }
    first = _checkpoint_identity(**common, pdf_backend_fingerprint=fingerprint)
    second = _checkpoint_identity(**common, pdf_backend_fingerprint=fingerprint)
    assert first == second


def test_install_lock_is_bounded(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from filelock import FileLock

    calls: list[str] = []
    _arm(monkeypatch, calls)
    config = _config(project)
    root = backend_root(config)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(env_module, "INSTALL_LOCK_TIMEOUT_SECONDS", 0.2)

    held = FileLock(root / ".install.lock", timeout=1)
    held.acquire()
    try:
        with pytest.raises(DoclingEnvironmentError):
            ensure_docling_env(config)
    finally:
        held.release()

    assert calls == []
