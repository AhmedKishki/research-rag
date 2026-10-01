"""Updating: the check that writes nothing, and the update that will not break a build.

Nothing here reaches a network. Every external command goes through an injected
runner that answers from a recorded table, so a test can put the remote anywhere
it likes, including out of reach, and the decision it checks is the one the
command makes in production.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import research_rag.surfaces.cli as cli_module
from research_rag import registry
from research_rag import update as update_module
from research_rag.launcher import launcher_path
from research_rag.support import ResearchError
from research_rag.surfaces.cli import _parser

CHECKOUT = update_module.CHECKOUT
DISTRIBUTION = update_module.DISTRIBUTION
COMMIT = "a" * 40
REMOTE_HEAD = "b" * 40


class Recorder:
    """A runner that answers from a table and remembers every call.

    A command the table does not name fails the way a missing tool does, so a
    test that forgets one finds out rather than reaching a real binary.
    """

    def __init__(self, answers: dict[Any, Any] | None = None) -> None:
        self.answers = {self._key(key): value for key, value in (answers or {}).items()}
        self.calls: list[tuple[str, ...]] = []

    @staticmethod
    def _key(key: Any) -> tuple[str, ...]:
        return tuple(key.split(" ")) if isinstance(key, str) else tuple(key)

    def __call__(
        self, argv: Any, *, cwd: Path | None = None
    ) -> update_module.CommandResult:
        command = tuple(str(part) for part in argv)
        self.calls.append(command)
        answer = self.answers.get(command)
        if answer is None:
            return update_module.CommandResult(127, "", f"{command[0]} is not here")
        if isinstance(answer, update_module.CommandResult):
            return answer
        return update_module.CommandResult(0, str(answer), "")

    def ran(self, *command: str) -> bool:
        return tuple(command) in self.calls


def _checkout(root: Path, *, behind: str = "0", changed: str = "README.md") -> Recorder:
    """The answers one git checkout gives, as `git -C <root> ...`."""

    return Recorder(
        {
            f"git -C {root} rev-parse --abbrev-ref HEAD": "main",
            f"git -C {root} rev-parse --abbrev-ref --symbolic-full-name @{{u}}": (
                "origin/main"
            ),
            f"git -C {root} rev-parse HEAD": COMMIT,
            f"git -C {root} rev-parse origin/main": REMOTE_HEAD,
            f"git -C {root} fetch --quiet --prune --tags": "",
            f"git -C {root} rev-list --count HEAD..origin/main": behind,
            f"git -C {root} diff --name-only HEAD..origin/main": changed,
        }
    )


def _local(checkout: Path | None, **overrides: Any) -> update_module.LocalState:
    values: dict[str, Any] = {
        "shape": CHECKOUT if checkout is not None else DISTRIBUTION,
        "version": "0.1.0",
        "checkout": checkout,
        "branch": "main" if checkout is not None else None,
        "upstream": "origin/main" if checkout is not None else None,
        "commit": COMMIT if checkout is not None else None,
        "tool": None if checkout is not None else "uv",
        "tool_specifier": None if checkout is not None else "git+https://example",
    }
    values.update(overrides)
    return update_module.LocalState(**values)


def _run(*arguments: str) -> Any:
    return asyncio.run(cli_module._run(_parser().parse_args(list(arguments))))


@pytest.fixture
def as_an_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """This installation has no git metadata at or above it."""

    monkeypatch.setattr(update_module, "nearest_checkout", lambda: None)


def _a_dead_pid() -> int:
    """A process id nothing is running under any more."""

    finished = subprocess.Popen(["true"])
    finished.wait()
    return finished.pid


def test_a_checkout_behind_by_three_plans_a_pull_a_lock_and_a_sync(
    tmp_path: Path,
) -> None:
    """The plan names what it would run, in order, and why the lock step is there."""

    local = _local(tmp_path)
    remote = update_module.RemoteState(
        reachable=True, head=REMOTE_HEAD, behind=3, pins_moving=True
    )

    plan = update_module.plan_update(local, remote)

    assert plan.available is True
    assert plan.behind == 3
    assert plan.local_revision == COMMIT[:12]
    assert plan.remote_revision == REMOTE_HEAD[:12]
    assert plan.labels() == ("git pull --ff-only", "uv lock", "uv sync")
    assert "3 commit(s) ahead" in plan.notes[0]


def test_a_commit_that_moves_no_pin_needs_no_lock(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True, head=REMOTE_HEAD, behind=1, pins_moving=False
        ),
    )

    assert plan.labels() == ("git pull --ff-only", "uv sync")


def test_a_checkout_level_with_its_remote_has_nothing_to_do(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path), update_module.RemoteState(reachable=True, behind=0)
    )

    assert plan.available is False
    assert plan.commands == ()
    assert "level with origin/main" in plan.notes[0]


def test_an_unreachable_remote_is_an_answer_and_not_a_failure(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=False, detail="`git fetch` failed: could not resolve host"
        ),
    )

    assert plan.available is False
    assert plan.blocked is None
    assert plan.commands == ()
    assert "could not resolve host" in " ".join(plan.notes)
    assert "Nothing was changed" in " ".join(plan.notes)


def test_offline_asks_nothing_and_says_the_remote_was_not_asked(
    tmp_path: Path,
) -> None:
    run = Recorder()

    remote = update_module.probe_remote(_local(tmp_path), run, offline=True)

    assert remote.reachable is False
    assert "--offline" in remote.detail
    assert run.calls == []


def test_the_probe_reads_the_checkout_it_is_given_and_nothing_more(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    run = _checkout(root, behind="2", changed="pyproject.toml\nsrc/research_rag/app.py")

    local = update_module.LocalState(
        shape=CHECKOUT,
        version="0.1.0",
        checkout=root,
        branch="main",
        upstream="origin/main",
        commit=COMMIT,
    )
    remote = update_module.probe_remote(local, run)
    plan = update_module.plan_update(local, remote)

    assert remote.behind == 2
    assert remote.pins_moving is True
    assert plan.available is True
    # A check asks the remote and stops there: no pull, no lock, no sync.
    assert not run.ran("git", "pull", "--ff-only")
    assert not run.ran("uv", "lock")
    assert not run.ran("uv", "sync")


def test_applying_runs_the_plan_and_reports_each_step(tmp_path: Path) -> None:
    run = Recorder(
        {
            ("git", "pull", "--ff-only"): "Fast-forward\n",
            ("uv", "lock"): "Resolved\n",
            ("uv", "sync"): "Installed\n",
        }
    )
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True, head=REMOTE_HEAD, behind=2, pins_moving=True
        ),
    )

    performed = update_module.apply_plan(plan, run=run, cwd=tmp_path)

    assert [step["command"] for step in performed] == list(plan.labels())
    assert all(step["returncode"] == 0 for step in performed)


def test_a_failing_step_stops_the_update_and_names_the_command(
    tmp_path: Path,
) -> None:
    run = Recorder(
        {
            ("git", "pull", "--ff-only"): update_module.CommandResult(
                1, "", "fatal: Not possible to fast-forward"
            )
        }
    )
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True, head=REMOTE_HEAD, behind=2, pins_moving=True
        ),
    )

    with pytest.raises(ResearchError) as refused:
        update_module.apply_plan(plan, run=run, cwd=tmp_path)

    assert "git pull --ff-only" in str(refused.value)
    assert "Not possible to fast-forward" in str(refused.value)
    assert not run.ran("uv", "sync")


def test_a_checkout_with_no_upstream_says_so_instead_of_guessing(
    tmp_path: Path,
) -> None:
    local = _local(tmp_path, upstream=None)
    run = Recorder(
        {
            f"git -C {tmp_path} rev-parse --abbrev-ref HEAD": "main",
            f"git -C {tmp_path} rev-parse HEAD": COMMIT,
        }
    )

    remote = update_module.probe_remote(local, run)

    assert remote.reachable is False
    assert "no upstream branch" in remote.detail
    assert not run.ran(
        "git", "-C", str(tmp_path), "fetch", "--quiet", "--prune", "--tags"
    )


def test_a_distribution_is_updated_by_the_tool_that_installed_it(
    as_an_installed_distribution: None,
) -> None:
    """The tool is asked which one it is, and it is the one that does the work."""

    run = Recorder(
        {
            "uv tool list --show-version-specifiers": (
                "research-rag v0.1.0 (from git+https://example/research-rag.git)\n"
                "other-tool v2.0.0\n"
            ),
            "uv tool list --outdated": (
                "research-rag v0.1.0\n`-- upgrade available to: v0.2.0\n"
            ),
        }
    )

    local = update_module.probe_local(run=run)
    plan = update_module.plan_update(local, update_module.probe_remote(local, run))

    assert local.shape == DISTRIBUTION
    assert local.tool == "uv"
    assert local.tool_specifier == "git+https://example/research-rag.git"
    assert plan.available is True
    assert plan.labels() == ("uv tool upgrade research-rag",)


def test_pipx_is_used_when_it_is_the_tool_that_claims_the_install(
    as_an_installed_distribution: None,
) -> None:
    run = Recorder(
        {
            "uv tool list --show-version-specifiers": "no tools installed\n",
            "pipx list --json": json.dumps(
                {
                    "venvs": {
                        "research-rag": {
                            "metadata": {
                                "main_package": {
                                    "package": "research-rag",
                                    "package_version": "0.1.0",
                                }
                            }
                        }
                    }
                }
            ),
            (
                "pipx",
                "runpip",
                "research-rag",
                "index",
                "versions",
                "research-rag",
            ): "research-rag (0.1.0)\nAvailable versions: 0.3.0, 0.2.0, 0.1.0",
        }
    )

    local = update_module.probe_local(run=run)
    plan = update_module.plan_update(local, update_module.probe_remote(local, run))

    assert local.tool == "pipx"
    assert local.tool_specifier == ""
    assert plan.available is True
    assert plan.labels() == ("pipx upgrade research-rag",)


def test_an_install_no_tool_claims_is_refused_with_both_commands(
    as_an_installed_distribution: None,
) -> None:
    run = Recorder()
    local = update_module.probe_local(run=run)

    plan = update_module.plan_update(local, update_module.probe_remote(local, run))

    assert local.tool is None
    assert plan.available is False
    assert plan.commands == ()
    assert plan.blocked is not None
    assert "uv tool install" in plan.blocked
    assert "pipx install" in plan.blocked


def test_an_install_at_the_version_the_tool_offers_has_nothing_to_do() -> None:
    plan = update_module.plan_update(
        update_module.LocalState(shape=DISTRIBUTION, version="0.1.0", tool="uv"),
        update_module.RemoteState(reachable=True, available_version="0.1.0"),
    )

    assert plan.available is False
    assert plan.commands == ()


def test_a_tool_that_cannot_be_asked_is_offline_rather_than_a_failure() -> None:
    run = Recorder(
        {
            "uv tool list --outdated": update_module.CommandResult(
                2, "", "error: network disabled"
            )
        }
    )
    local = update_module.LocalState(shape=DISTRIBUTION, version="0.1.0", tool="uv")

    remote = update_module.probe_remote(local, run)
    plan = update_module.plan_update(local, remote)

    assert remote.reachable is False
    assert "network disabled" in remote.detail
    assert plan.blocked is None
    assert plan.available is False


def _project_with_a_running_build(root: Path, phase: str = "embedding") -> Any:
    project = update_module.ProjectState(root / "thesis", "My Thesis")
    staging = project.state_root / "staging" / "build-1"
    staging.mkdir(parents=True)
    (staging / "checkpoint.json").write_text(
        json.dumps({"build_id": "build-1", "phase": phase}), encoding="utf-8"
    )
    (project.state_root / "project.lock").write_text("1", encoding="utf-8")
    return project


def test_a_project_lock_held_by_a_build_refuses_the_whole_update(
    tmp_path: Path,
) -> None:
    """The refusal names the project, its phase, and the command that reports it."""

    project = _project_with_a_running_build(tmp_path)

    held = update_module.held_projects([project])

    assert len(held) == 1
    refusal = held[0].refusal()
    assert "My Thesis" in refusal
    assert str(project.project_root) in refusal
    assert "embedding" in refusal
    assert "research-rag --project-root" in refusal
    assert "status" in refusal


def test_a_lock_nobody_holds_is_not_a_refusal(tmp_path: Path) -> None:
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "project.lock").write_text(
        str(_a_dead_pid()), encoding="utf-8"
    )

    assert update_module.held_projects([project]) == ()


def test_stopping_an_app_uses_the_projects_own_launcher(tmp_path: Path) -> None:
    """The launcher owns the pid, the port, and the process group."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    script = project.portable_root / "bin" / "open-research-rag-ui.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    run = Recorder({f"{script} --stop": "Stopped the app for this project.\n"})

    stopped = update_module.stop_app(project, run)

    assert run.calls == [(str(script), "--stop")]
    assert stopped.stopped is True
    assert "Stopped the app" in stopped.detail


def test_a_project_with_no_app_is_left_alone(tmp_path: Path) -> None:
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    run = Recorder()

    stopped = update_module.stop_app(project, run)

    assert stopped.stopped is False
    assert run.calls == []


def test_the_portable_state_digest_ignores_the_runtime_root(tmp_path: Path) -> None:
    """Stopping an app writes its pid under `runtime/`; nothing else is touched."""

    project = tmp_path / "thesis"
    (project / ".research-rag" / "runtime" / "logs").mkdir(parents=True)
    (project / ".research-rag" / "runtime" / "logs" / "app.log").write_text(
        "log\n", encoding="utf-8"
    )
    descriptor = project / ".research-rag" / "project.json"
    descriptor.write_text("{}", encoding="utf-8")

    before = update_module.portable_state_digest(project)
    (project / ".research-rag" / "runtime" / "research-rag-ui.pid").write_text(
        "1", encoding="utf-8"
    )
    assert update_module.portable_state_digest(project) == before

    descriptor.write_text('{"name": "x"}', encoding="utf-8")
    after = update_module.portable_state_digest(project)
    assert update_module.state_changes(
        {str(project): before}, {str(project): after}
    ) == {str(project): ["project.json"]}


def test_the_command_refuses_an_update_while_a_build_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole update is refused before anything is pulled or stopped."""

    project = _project_with_a_running_build(tmp_path, phase="dense_indexing")
    monkeypatch.setattr(
        registry,
        "load",
        lambda: [registry.RegisteredProject("id", "thesis", project.project_root, "")],
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(tmp_path)
    )
    monkeypatch.setattr(
        update_module,
        "probe_remote",
        lambda *_args, **_kwargs: update_module.RemoteState(
            reachable=True, behind=2, pins_moving=True
        ),
    )

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply")

    assert "dense_indexing" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)


def test_applying_stops_every_app_and_prints_the_command_that_starts_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graceful means the launcher stops it, and the reader starts it."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    script = project.portable_root / "bin" / "open-research-rag-ui.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    monkeypatch.setattr(
        registry,
        "load",
        lambda: [registry.RegisteredProject("id", "thesis", project.project_root, "")],
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(tmp_path)
    )
    monkeypatch.setattr(
        update_module,
        "probe_remote",
        lambda *_args, **_kwargs: update_module.RemoteState(
            reachable=True, behind=1, pins_moving=True
        ),
    )
    stopped = Recorder({f"{script} --stop": "Stopped the app for this project.\n"})
    applied = Recorder({"uv tool upgrade research-rag": "Updated research-rag.\n"})
    monkeypatch.setattr(update_module, "subprocess_runner", stopped)
    monkeypatch.setattr(
        update_module,
        "apply_plan",
        lambda plan, **_kwargs: [
            {"command": "uv tool upgrade research-rag", "returncode": 0, "output": ""}
        ],
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(tmp_path)
    )

    payload = _run("update", "--apply").payload

    assert payload is not None
    assert stopped.calls == [(str(script), "--stop")]
    assert payload["applied"] is True
    assert payload["stopped"][0]["stopped"] is True
    assert payload["start_again"] == [
        f"research-rag --project-root {project.project_root} ui"
    ]
    assert payload["project_state_untouched"] is True
    assert applied.calls == []


def test_applying_with_nothing_to_apply_stops_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An update that would change nothing must not stop a reader's app."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    monkeypatch.setattr(
        registry,
        "load",
        lambda: [registry.RegisteredProject("id", "thesis", project.project_root, "")],
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(tmp_path)
    )
    monkeypatch.setattr(
        update_module,
        "probe_remote",
        lambda *_args, **_kwargs: update_module.RemoteState(reachable=True, behind=0),
    )
    run = Recorder()
    monkeypatch.setattr(update_module, "subprocess_runner", run)

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply")

    assert "Nothing to apply" in str(refused.value)
    assert run.calls == []


def test_a_relocated_runtime_root_is_read_from_the_launcher(tmp_path: Path) -> None:
    """The lock and the pid live where the launcher says they do."""

    elsewhere = tmp_path / "fast-disk"
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    script = launcher_path(project.portable_root)
    script.parent.mkdir(parents=True)
    script.write_text(
        f'#!/bin/sh\nPROJECT_ROOT="{project.project_root}"\n'
        f'RUNTIME_ROOT="{elsewhere}"\n',
        encoding="utf-8",
    )

    assert project.state_root == elsewhere
    assert project.default_state_root == project.portable_root / "runtime"


def test_offline_with_apply_is_refused_rather_than_reported_as_empty() -> None:
    with pytest.raises(ResearchError) as refused:
        _run("--offline", "update", "--apply")

    assert "--offline cannot update" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)
