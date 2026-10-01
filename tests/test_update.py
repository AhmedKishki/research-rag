"""The decision under test is the release one: whether this checkout is at, behind, or
ahead of the latest published release.

The branch head is reported beside that answer and never decides it.

Nothing here reaches a network: every external command goes through an injected runner
that answers from a recorded table, so a test can put the remote anywhere it likes.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

import research_rag.surfaces.cli as cli_module
from research_rag import config as config_module
from research_rag import registry
from research_rag import update as update_module
from research_rag.release import FOUND, NONE, Release, ReleaseSet
from research_rag.support import ResearchError
from research_rag.surfaces.cli import _parser
from research_rag.version import version_block

CHECKOUT = update_module.CHECKOUT
DISTRIBUTION = update_module.DISTRIBUTION
COMMIT = "a" * 40
REMOTE_HEAD = "b" * 40
RELEASE_COMMIT = "c" * 40


class Recorder:
    """A command the table does not name fails the way a missing tool does, so a test that
    forgets one finds out rather than reaching a real binary.
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


def _tag_listing(*tags: tuple[str, str]) -> str:
    return "".join(f"{commit}\trefs/tags/{tag}\n" for tag, commit in tags)


def _published(*versions: str) -> ReleaseSet:
    if not versions:
        return ReleaseSet(
            NONE,
            detail=(
                "the remote publishes no tag naming a release version, so it has "
                "published no release"
            ),
        )
    newest = versions[-1]
    return ReleaseSet(
        FOUND,
        Release(newest, f"v{newest}", RELEASE_COMMIT),
        detail=f"release {newest} is the highest version the remote's tags name",
    )


def _checkout(
    root: Path,
    *,
    behind: str = "0",
    changed: str = "README.md",
    versions: tuple[str, ...] = ("0.1.0",),
) -> Recorder:
    return Recorder(
        {
            f"git -C {root} rev-parse --abbrev-ref HEAD": "main",
            f"git -C {root} rev-parse --abbrev-ref --symbolic-full-name @{{u}}": (
                "origin/main"
            ),
            f"git -C {root} rev-parse HEAD": COMMIT,
            f"git -C {root} rev-parse origin/main": REMOTE_HEAD,
            f"git -C {root} fetch --quiet --prune --tags": "",
            f"git -C {root} ls-remote --tags origin": _tag_listing(
                *((f"v{version}", RELEASE_COMMIT) for version in versions)
            ),
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
        "declared": "0.1.0" if checkout is not None else None,
    }
    values.update(overrides)
    return update_module.LocalState(**values)


def _run(*arguments: str) -> Any:
    return asyncio.run(cli_module._run(_parser().parse_args(list(arguments))))


@pytest.fixture
def as_an_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_module, "nearest_checkout", lambda: None)


def _a_dead_pid() -> int:
    finished = subprocess.Popen(["true"])
    finished.wait()
    return finished.pid


def test_a_checkout_behind_a_release_plans_to_detach_onto_the_release_tag(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            head=REMOTE_HEAD,
            behind=3,
            pins_moving=True,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    assert plan.available is True
    assert plan.release_position == update_module.BEHIND_RELEASE
    assert plan.release_version == "0.2.0"
    assert plan.labels() == ("git checkout --detach v0.2.0", "uv lock", "uv sync")
    assert "behind release 0.2.0, tagged v0.2.0" in " ".join(plan.notes)


def test_the_answer_prints_the_command_that_returns_to_the_branch(
    tmp_path: Path,
) -> None:
    """Installing a release leaves the checkout detached, so the way back is named."""

    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True, releases=_published("0.1.0", "0.2.0")
        ),
    )

    assert f"git -C {tmp_path} checkout main" in " ".join(plan.notes)


def test_a_commit_that_moves_no_pin_needs_no_lock(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            behind=1,
            pins_moving=False,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    assert plan.labels() == ("git checkout --detach v0.2.0", "uv sync")


def test_a_checkout_at_the_release_has_nothing_to_apply(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            behind=0,
            releases=_published("0.1.0"),
        ),
    )

    assert plan.available is False
    assert plan.commands == ()
    assert plan.release_position == update_module.AT_RELEASE
    assert plan.release_version == "0.1.0"
    assert "at release 0.1.0, tagged v0.1.0" in " ".join(plan.notes)


def test_the_release_version_decides_and_not_the_branch_head(tmp_path: Path) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            behind=0,
            releases=_published("0.1.0"),
        ),
    )

    assert plan.available is False
    assert plan.release_position == update_module.AT_RELEASE


def test_a_checkout_ahead_of_the_release_reports_unreleased_work(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path, declared="0.3.0"),
        update_module.RemoteState(
            reachable=True,
            behind=0,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    assert plan.available is False
    assert plan.commands == ()
    assert plan.release_position == update_module.AHEAD_OF_RELEASE
    assert plan.release_version == "0.2.0"
    assert "carries unreleased work" in " ".join(plan.notes)


def test_a_repository_publishing_no_release_says_so_and_compares_nothing(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(reachable=True, behind=4, releases=_published()),
    )

    assert plan.available is False
    assert plan.commands == ()
    assert plan.release_position == update_module.NO_RELEASE
    said = " ".join(plan.notes)
    assert "published no release" in said
    assert "nothing can be compared with it" in said
    assert "git tag v<version>" in said
    assert "maintainer action" in said


def test_two_tags_claiming_one_version_are_refused_rather_than_guessed(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            releases=ReleaseSet(
                "ambiguous", detail="two tags claim version 0.1.0 (v0.1.0, 0.1.0)"
            ),
        ),
    )

    assert plan.available is False
    assert plan.commands == ()
    assert plan.release_position == update_module.UNREADABLE_RELEASE
    assert "Nothing was changed" in " ".join(plan.notes)


def test_the_branch_head_position_is_reported_beside_the_release_answer(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            head=REMOTE_HEAD,
            behind=7,
            releases=_published("0.1.0"),
        ),
    )

    assert "7 commit(s) ahead" in " ".join(plan.notes)
    assert plan.as_dict()["commits_behind_branch"] == 7


def test_a_dirty_working_tree_refuses_the_update_and_names_the_files_in_the_way(
    tmp_path: Path,
) -> None:
    plan = update_module.plan_update(
        _local(tmp_path, dirty=("src/research_rag/update.py", "README.md")),
        update_module.RemoteState(
            reachable=True, releases=_published("0.1.0", "0.2.0")
        ),
    )

    assert plan.commands == ()
    assert plan.blocked is not None
    refused = plan.blocked
    assert "src/research_rag/update.py" in refused
    assert "README.md" in refused
    assert f"git -C {tmp_path} status" in refused
    assert "Nothing was changed" in refused


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


def test_the_probe_reads_the_release_tags_and_the_branch_head_and_nothing_more(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    run = _checkout(root, behind="2", changed="pyproject.toml")

    local = update_module.LocalState(
        shape=CHECKOUT,
        version="0.1.0",
        checkout=root,
        branch="main",
        upstream="origin/main",
        commit=COMMIT,
        declared="0.1.0",
    )
    remote = update_module.probe_remote(local, run)
    plan = update_module.plan_update(local, remote)

    assert remote.behind == 2
    assert remote.releases.version == "0.1.0"
    assert plan.available is False
    assert not run.ran("git", "checkout", "--detach", "v0.1.0")
    assert not run.ran("uv", "lock")
    assert not run.ran("uv", "sync")


def test_the_probe_reports_the_untracked_files_nothing_else(tmp_path: Path) -> None:
    run = Recorder(
        {
            f"git -C {tmp_path} status --porcelain --untracked-files=no": (
                " M src/research_rag/update.py\nA  tests/test_release.py\n"
            )
        }
    )

    assert update_module.dirty_files(tmp_path, run) == (
        "src/research_rag/update.py",
        "tests/test_release.py",
    )


def test_applying_runs_the_plan_and_reports_each_step(tmp_path: Path) -> None:
    run = Recorder(
        {
            ("git", "checkout", "--detach", "v0.2.0"): "HEAD is now at abc\n",
            ("uv", "lock"): "Resolved\n",
            ("uv", "sync"): "Installed\n",
        }
    )
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            behind=2,
            pins_moving=True,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    performed = update_module.apply_plan(plan, run=run, cwd=tmp_path)

    assert [step["command"] for step in performed] == list(plan.labels())
    assert all(step["returncode"] == 0 for step in performed)


def test_a_failing_step_stops_the_update_and_names_the_command(tmp_path: Path) -> None:
    run = Recorder(
        {
            ("git", "checkout", "--detach", "v0.2.0"): update_module.CommandResult(
                1, "", "error: Your local changes to the following files would be lost"
            )
        }
    )
    plan = update_module.plan_update(
        _local(tmp_path),
        update_module.RemoteState(
            reachable=True,
            behind=2,
            pins_moving=True,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    with pytest.raises(ResearchError) as refused:
        update_module.apply_plan(plan, run=run, cwd=tmp_path)

    assert "git checkout --detach v0.2.0" in str(refused.value)
    assert "would be lost" in str(refused.value)
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


def test_stopping_an_app_signals_the_pid_the_app_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app is asked to stop the way its own terminal would ask it."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        update_module.os, "kill", lambda pid, sig: sent.append((pid, sig))
    )
    run = Recorder()

    stopped = update_module.stop_app(project, run)

    assert run.calls == []
    # Signal 0 is the liveness probe that found the pid; only the TERM is the request.
    assert [entry for entry in sent if entry[1] != 0] == [(os.getpid(), signal.SIGTERM)]
    assert stopped.stopped is True
    assert str(os.getpid()) in stopped.detail


def test_a_pid_that_cannot_be_asked_to_stop_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )

    def _refuse(_pid: int, _sig: int) -> None:
        raise PermissionError(13, "Operation not permitted")

    monkeypatch.setattr(update_module.os, "kill", _refuse)

    stopped = update_module.stop_app(project, Recorder())

    assert stopped.stopped is False
    assert "not permitted" in stopped.detail


def test_a_project_with_no_app_is_left_alone(tmp_path: Path) -> None:
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    run = Recorder()

    stopped = update_module.stop_app(project, run)

    assert stopped.stopped is False
    assert run.calls == []


def test_the_portable_state_digest_ignores_the_runtime_root(tmp_path: Path) -> None:
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
            reachable=True,
            behind=2,
            pins_moving=True,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply")

    assert "dense_indexing" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)


def test_applying_stops_every_app_and_prints_the_command_that_starts_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        update_module.os, "kill", lambda pid, sig: sent.append((pid, sig))
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
            reachable=True,
            behind=1,
            pins_moving=True,
            releases=_published("0.1.0", "0.2.0"),
        ),
    )
    stopped = Recorder()
    applied = Recorder({"git checkout --detach v0.2.0": "HEAD is now at abc\n"})
    monkeypatch.setattr(update_module, "subprocess_runner", stopped)
    monkeypatch.setattr(
        update_module,
        "apply_plan",
        lambda plan, **_kwargs: [
            {"command": "git checkout --detach v0.2.0", "returncode": 0, "output": ""}
        ],
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(tmp_path)
    )

    payload = _run("update", "--apply").payload

    assert payload is not None
    assert stopped.calls == []
    assert [entry for entry in sent if entry[1] != 0] == [(os.getpid(), signal.SIGTERM)]
    assert payload["applied"] is True
    assert payload["stopped"][0]["stopped"] is True
    assert payload["start_again"] == [
        f"research-rag --project-root {project.project_root} start"
    ]
    assert payload["project_state_untouched"] is True
    assert applied.calls == []


def test_applying_with_nothing_to_apply_stops_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        lambda *_args, **_kwargs: update_module.RemoteState(
            reachable=True, behind=0, releases=_published("0.1.0")
        ),
    )
    run = Recorder()
    monkeypatch.setattr(update_module, "subprocess_runner", run)

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply")

    assert "Nothing to apply" in str(refused.value)
    assert run.calls == []


def test_a_relocated_runtime_root_is_read_from_the_projects_own_record(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "fast-disk"
    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.portable_root.mkdir(parents=True)
    config_module.record_runtime_root(project.portable_root, elsewhere)

    assert project.state_root == elsewhere
    assert project.default_state_root == project.portable_root / "runtime"

    config_module.record_runtime_root(project.portable_root, None)
    assert project.state_root == project.default_state_root


def test_the_version_flag_and_update_report_the_same_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_module.main(["--version"])
    printed = capsys.readouterr().out

    payload = _run("--offline", "update").payload

    assert payload is not None
    assert payload["version"] == version_block()
    assert printed.splitlines() == [
        f"research-rag {version_block()['app']}",
        f"installed {version_block()['installed']}",
        (
            f"UI {version_block()['ui']}"
            if version_block()["ui"]
            else "UI not installed"
        ),
        f"restart_required {str(version_block()['restart_required']).lower()}",
    ]


def test_the_check_reports_a_declared_version_that_names_no_release_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_module, "nearest_checkout", lambda: tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "9.9.9"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        update_module,
        "declared_version",
        lambda _checkout: "9.9.9",
    )

    payload = _run("--offline", "update").payload

    assert payload is not None
    assert payload["declared_version_problems"] == []
    assert payload["install"]["declared_version"] == "9.9.9"


def test_offline_with_apply_is_refused_rather_than_reported_as_empty() -> None:
    with pytest.raises(ResearchError) as refused:
        _run("--offline", "update", "--apply")

    assert "--offline cannot update" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)
