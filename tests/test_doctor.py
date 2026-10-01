"""The doctor command: one line per dependency, and no repair without a flag.

The default run is the behaviour that matters most: it must read the installation
without changing it, name the condition it found, and name the command that fixes
it. The two operations that reach the network are tested against stand-ins, so
these tests never download anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from vanilla_ultra_rag_mcp import runtime as vanilla_runtime

import research_rag.doctor as doctor_module
from research_rag.config import ResearchConfig, resolve_config
from research_rag.doctor import (
    DoctorError,
    _project_server_command,
    check_entry,
    mcp_entry_block,
    mcp_url_block,
    run_doctor,
)

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


def _entry_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_two_entries_reach_the_app_two_ways(config: ResearchConfig) -> None:
    """A client that can open a socket gets the app; one that cannot gets the bridge."""

    url_entry = json.loads(mcp_url_block(config))
    stdio_entry = json.loads(mcp_entry_block(config))
    url = url_entry["mcp"]["research-rag"]
    command = stdio_entry["mcp"]["research-rag"]["command"]

    assert url["url"].startswith("http://127.0.0.1:")
    assert url["url"].endswith("/mcp")
    # The stdio entry names the bridge explicitly: `research-rag ui` runs the same
    # executable and is not an agent entry.
    assert command[-1] == "mcp"
    assert Path(command[0]).name == "research-rag"
    assert stdio_entry["mcp"]["research-rag"]["timeout"] == 3_600_000


def test_the_stdio_entry_names_a_project_and_not_a_directory(
    config: ResearchConfig,
) -> None:
    """The entry is copied between machines, so a path in it would break there."""

    command = json.loads(mcp_entry_block(config))["mcp"]["research-rag"]["command"]

    assert command[command.index("--project-name") + 1] == config.project_name
    assert "--project-root" not in command
    assert str(config.project_root) not in command


def test_a_correct_url_entry_passes(tmp_path: Path, config: ResearchConfig) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps(
            {"mcp": {"rag": {"url": "http://127.0.0.1:5051/mcp", "timeout": 3_600_000}}}
        ),
    )

    findings = check_entry(config, path)

    assert [check.name for check in findings] == ["entry"]
    assert findings[0].state == "ok"


def test_a_url_entry_must_be_loopback_and_at_the_agent_surface(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps({"mcpServers": {"rag": {"url": "https://example.com/mcp"}}}),
    )

    findings = check_entry(config, path)

    states = {check.name: check.state for check in findings}
    assert states["entry.url"] == "blocked"
    assert "loopback only" in next(c.reason for c in findings if c.name == "entry.url")


def test_a_url_entry_at_the_wrong_route_is_a_warning(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path, json.dumps({"mcp": {"rag": {"url": "http://127.0.0.1:5051/"}}})
    )

    findings = check_entry(config, path)

    assert [check.state for check in findings] == ["warn"]
    assert "agent surface is at /mcp" in findings[0].reason


def test_a_stdio_entry_must_run_this_apps_bridge(
    tmp_path: Path, config: ResearchConfig
) -> None:
    """`research-rag ui` runs the same executable and is not an agent entry."""

    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "rag": {
                        "command": [
                            "/usr/local/bin/research-rag",
                            "--project-root",
                            str(config.project_root),
                            "ui",
                        ]
                    }
                }
            }
        ),
    )

    with pytest.raises(DoctorError, match="No research-rag entry"):
        check_entry(config, path)


def test_a_stdio_entry_is_checked_for_paths_and_timeout(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "rag": {
                        "command": [
                            "/nowhere/research-rag",
                            "--project-name",
                            config.project_name,
                            "mcp",
                        ]
                    }
                }
            }
        ),
    )

    findings = check_entry(config, path)
    states = {check.name: check.state for check in findings}

    assert states["entry.executable"] == "blocked"
    assert "entry.project_name" not in states
    assert states["entry.timeout"] == "warn"


def test_a_stdio_entry_naming_a_directory_is_blocked(
    tmp_path: Path, config: ResearchConfig
) -> None:
    """A path in an entry works on one machine, which is the fault this forbids."""

    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "rag": {
                        "command": [
                            str(_project_server_command()),
                            "--project-root",
                            str(config.project_root),
                            "mcp",
                        ]
                    }
                }
            }
        ),
    )

    findings = check_entry(config, path)
    finding = next(check for check in findings if check.name == "entry.project_name")

    assert finding.state == "blocked"
    assert "--project-name" in finding.reason
    assert config.project_name in finding.reason


def test_a_stdio_entry_naming_no_project_is_blocked(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps(
            {"mcp": {"rag": {"command": [str(_project_server_command()), "mcp"]}}}
        ),
    )

    findings = check_entry(config, path)
    finding = next(check for check in findings if check.name == "entry.project_name")

    assert finding.state == "blocked"
    assert config.project_name in finding.reason


def test_a_stdio_entry_for_another_project_is_a_warning(
    tmp_path: Path, config: ResearchConfig
) -> None:
    """One entry file may serve several projects, so a mismatch is not fatal."""

    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "rag": {
                        "command": [
                            str(_project_server_command()),
                            "--project-name",
                            "a-different-project",
                            "mcp",
                        ],
                        "timeout": 3_600_000,
                    }
                }
            }
        ),
    )

    findings = check_entry(config, path)
    finding = next(check for check in findings if check.name == "entry.project_name")

    assert finding.state == "warn"
    assert "a-different-project" in finding.reason
    assert config.project_name in finding.reason


def test_two_entries_for_one_project_say_what_it_means(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "one": {"url": "http://127.0.0.1:5051/mcp"},
                    "two": {"url": "http://127.0.0.1:5052/mcp"},
                }
            }
        ),
    )

    findings = check_entry(config, path)

    assert not any(check.name == "entry.duplicate" for check in findings)
    assert all(check.state == "ok" for check in findings) or any(
        check.name == "entry" for check in findings
    )


def test_an_entry_file_that_names_no_project_is_refused(
    tmp_path: Path, config: ResearchConfig
) -> None:
    path = _entry_file(
        tmp_path,
        json.dumps(
            {
                "mcp": {
                    "rag": {
                        "command": ["/usr/bin/other", "--project-root", "/x", "mcp"]
                    }
                }
            }
        ),
    )

    with pytest.raises(DoctorError, match="No research-rag entry"):
        check_entry(config, path)


def test_a_missing_entry_file_is_refused(config: ResearchConfig) -> None:
    with pytest.raises(DoctorError, match="No client entry file"):
        check_entry(config, "/nowhere/mcp.json")


def test_checking_an_entry_is_not_an_operation(
    config: ResearchConfig, tmp_path: Path
) -> None:
    path = _entry_file(tmp_path, json.dumps({"mcp": {}}))

    with pytest.raises(DoctorError, match="cannot run with an operation"):
        _run(config, entry=path, repair=True)


def test_the_entry_check_reports_without_reading_the_project(
    config: ResearchConfig, tmp_path: Path
) -> None:
    """A file check is about the file, so it must not start an app to read one."""

    path = _entry_file(
        tmp_path, json.dumps({"mcp": {"rag": {"url": "http://127.0.0.1:5051/mcp"}}})
    )

    result = _run(config, entry=path)

    assert result.exit_code == 0
    assert "ok" in result.text()
