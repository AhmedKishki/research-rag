"""The doctor command: one line per dependency, and no repair without a flag.

The default run is the behaviour that matters most: it must read the installation
without changing it, name the condition it found, and name the command that fixes
it. The two operations that reach the network are tested against stand-ins, so
these tests never download anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from vanilla_ultra_rag_mcp import runtime as vanilla_runtime

import research_rag.doctor as doctor_module
from research_rag.config import ResearchConfig, resolve_config
from research_rag.doctor import run_doctor

READY_STATUS: dict[str, Any] = {
    "ready": True,
    "stale": False,
    "generation_id": "20260101T000000Z-abcdef",
    "upgrade_reasons": [],
    "retained_generation_bytes": 1024,
    "message": "Ready.",
}


@pytest.fixture
def config(project: Path, tmp_path: Path) -> ResearchConfig:
    return resolve_config(
        project_root=project,
        model_cache_root=tmp_path / "models",
        runtime_cache_root=tmp_path / "runtime-cache",
    )


@pytest.fixture
def healthy(config: ResearchConfig, monkeypatch: pytest.MonkeyPatch) -> ResearchConfig:
    """A project whose dependencies are all in place."""

    from research_rag.embeddings import resolve_embedding_model
    from research_rag.rerankers import resolve_reranker_model

    for name, revision in (
        (
            "qdrant/bge-small-en-v1.5-onnx-q",
            resolve_embedding_model(config.settings.embedding_model).revision,
        ),
        resolve_reranker_model(config.reranker_model),
    ):
        snapshot = (
            config.model_cache_root
            / f"models--{name.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True, exist_ok=True)
    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-test"
    root.mkdir(parents=True, exist_ok=True)
    (root / vanilla_runtime.MARKER_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", lambda p: p)
    return config


def _run(config: ResearchConfig, **kwargs: Any) -> Any:
    return run_doctor(
        config,
        dict(READY_STATUS),
        running=kwargs.pop("running", []),
        **kwargs,
    )


def test_the_default_run_reports_every_check_and_changes_nothing(
    config: ResearchConfig,
) -> None:
    before = sorted(path.name for path in config.state_root.rglob("*"))

    result = _run(config)

    assert sorted(path.name for path in config.state_root.rglob("*")) == before
    for name in ("project_identity", "runtime_root", "vanilla_runtime", "lock"):
        assert any(name in line for line in result.lines)
    # Nothing is installed and nothing blocks: an online project downloads what
    # it needs on first use, which is a warning, not a broken installation.
    assert result.exit_code == 0
    assert any(line.startswith("warn") for line in result.lines)


def test_every_check_is_one_line_naming_its_state_and_its_remedy(
    config: ResearchConfig,
) -> None:
    offline = resolve_config(
        project_root=config.project_root,
        model_cache_root=config.model_cache_root,
        runtime_cache_root=config.runtime_cache_root,
        offline=True,
    )

    result = _run(offline)

    checks = [
        line for line in result.lines if line[:8].strip() in {"ok", "warn", "blocked"}
    ]
    assert len(checks) == 9
    for line in checks:
        state, name, _rest = line.split(maxsplit=2)
        assert state in {"ok", "warn", "blocked", "unknown"}
        assert name
    # Anything a caller must act on names the command that acts on it.
    for line in checks:
        if line.startswith("blocked"):
            assert "  ->  " in line


def test_a_healthy_installation_exits_zero(healthy: ResearchConfig) -> None:
    result = _run(healthy)

    assert result.exit_code == 0
    assert not [line for line in result.lines if line.startswith(("warn", "blocked"))]


def test_offline_without_an_installed_runtime_is_blocked(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline is the mode where a missing dependency stops the server."""

    offline = resolve_config(
        project_root=config.project_root,
        model_cache_root=config.model_cache_root,
        runtime_cache_root=config.runtime_cache_root,
        offline=True,
    )

    result = _run(offline)

    assert result.exit_code == 1
    assert any(
        line.startswith("blocked") and "vanilla_runtime" in line
        for line in result.lines
    )


def test_processes_are_reported_with_the_stop_command(config: ResearchConfig) -> None:
    running = [
        (4321, f"research-rag --project-root {config.project_root} serve"),
    ]

    result = _run(config, running=running)

    text = result.text()
    assert "4321" in text
    assert "stop --servers" in text
    # The doctor is not one of them, so it does not report itself.
    doctor_pid = run_doctor(
        config,
        dict(READY_STATUS),
        running=[(9999, f"research-rag --project-root {config.project_root} doctor")],
    )
    assert "9999" not in doctor_pid.text()


def test_nothing_running_says_so(config: ResearchConfig) -> None:
    assert "Nothing of this app is running" in _run(config).text()


def test_a_relocated_root_in_use_is_named(
    config: ResearchConfig, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    running = [
        (
            4321,
            (
                "research-rag --project-root "
                f"{config.project_root} --runtime-root {elsewhere}"
            ),
        ),
    ]

    text = _run(config, running=running).text()

    assert str(elsewhere) in text
    assert "--runtime-root" in text


def test_repair_moves_a_mismatched_tree_aside_and_installs(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tree that failed validation is evidence, so it is kept, never deleted."""

    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-deadbeef"
    (root / "servers").mkdir(parents=True)
    (root / "servers" / "stray.pyc").write_bytes(b"stale")
    installed: list[bool] = []

    def install(cache_root: Any) -> Path:
        installed.append(True)
        root.mkdir(parents=True, exist_ok=True)
        (root / "servers").mkdir(parents=True, exist_ok=True)
        (root / "servers" / "stray.pyc").unlink(missing_ok=True)
        return root

    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(
        vanilla_runtime,
        "validate_managed_runtime",
        lambda path: _raise_unless_fresh(path, root),
    )
    monkeypatch.setattr(vanilla_runtime, "install_managed_runtime", install)

    lines = doctor_module.repair_runtime(config)

    assert installed == [True]
    preserved = next(line for line in lines if "preserved" in line)
    assert "quarantine" in preserved
    assert "deadbeef" in preserved
    # The evidence stays on disk, and the new tree is beside it, not in its place.
    assert Path(preserved.split()[-1].rstrip(".")).is_dir()
    assert (root / "servers" / "stray.pyc").exists() is False


def _raise_unless_fresh(path: Path, root: Path) -> Path:
    if (root / "servers" / "stray.pyc").exists():
        raise vanilla_runtime.RuntimeValidationError(
            "Managed runtime content hash mismatch: got 0123456789abcdef, "
            "expected fedcba9876543210. The tree differs at "
            "servers/stray.pyc: unexpected in the installed tree, mode -rw-r--r--."
        )
    return path


def test_repair_leaves_a_valid_tree_alone(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-good"
    root.mkdir(parents=True)
    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", lambda p: p)
    monkeypatch.setattr(
        vanilla_runtime,
        "install_managed_runtime",
        lambda _c: pytest.fail("a valid tree must not be reinstalled"),
    )

    lines = doctor_module.repair_runtime(config)

    assert lines == (
        f"The managed runtime at {root} already matches the pinned snapshot.",
    )


def test_prefetch_reports_what_it_cached(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    import research_rag.dense as dense_module

    monkeypatch.setattr(
        dense_module,
        "_load_embedder",
        lambda _root, offline, model: loaded.append(model),
    )
    monkeypatch.setattr(
        dense_module,
        "_load_cross_encoder",
        lambda _root, offline, model: loaded.append(model),
    )

    lines = doctor_module.prefetch_models(config)

    assert len(loaded) == 2
    assert any("bge-small-en-v1.5" in line for line in lines)
    assert any("MiniLM" in line for line in lines)
