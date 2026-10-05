"""The decision under test is the release one: whether this checkout is at, behind, or
ahead of the latest published release.

The branch head is reported beside that answer and never decides it, and nothing is
installed without an approval that names the version it approves.

Nothing here reaches a network: every external command goes through an injected runner
that answers from a recorded table, and the one published-release request answers from a
recorded document, so a test can put the remote and the release anywhere it likes.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import research_rag.project.config as config_module
import research_rag.runtime.ownership as ownership_module
import research_rag.runtime.process as process_module
import research_rag.runtime.update as update_module
import research_rag.surfaces.cli as cli_module
from research_rag.project import registry
from research_rag.project.policy import ResearchError
from research_rag.runtime.release import (
    FOUND,
    NONE,
    GitHubRelease,
    PublishedRelease,
    Release,
    ReleaseSet,
)
from research_rag.runtime.version import version_block
from research_rag.surfaces.cli import _parser

CHECKOUT = update_module.CHECKOUT
DISTRIBUTION = update_module.DISTRIBUTION
COMMIT = "a" * 40
REMOTE_HEAD = "b" * 40
RELEASE_COMMIT = "c" * 40
# The published release the stubbed fetch returns, in the shape the API answers in.
PUBLISHED = {
    "tag_name": "v0.2.0",
    "html_url": "https://github.com/AhmedKishki/research-rag/releases/tag/v0.2.0",
    "published_at": "2026-10-01T12:00:00Z",
    "body": "## 0.2.0\n\n- The changelog a reader approves before installing.",
    "name": "research-rag 0.2.0",
    "draft": False,
    "prerelease": False,
}
ANSWERED = "Install this release? [y/N]:"


def _fetch(document: dict[str, Any]) -> Any:
    """The one request this command makes, answered from a recorded document."""

    def request(url: str, timeout: float) -> str:
        return json.dumps(document)

    return request


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file reaches GitHub, whatever it asks for."""

    monkeypatch.setattr(cli_module, "_fetch_published_release", _fetch(PUBLISHED))


class Question:
    """A reader in a terminal, and the questions this command put to them.

    The question is recorded rather than answered twice, so a test can read what
    was shown before it decides what the answer was.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
        self.asked: list[tuple[str, tuple[str, ...]]] = []
        monkeypatch.setattr(cli_module, "_interactive_terminal", lambda: True)
        monkeypatch.setattr(cli_module, "_ask_to_install", self._ask(answer))

    def _ask(self, answer: str) -> Any:
        def ask(
            published: PublishedRelease,
            notes: Sequence[str],
            *,
            target: str,
            stdin: Any = None,
            stderr: Any = None,
        ) -> bool:
            self.asked.append((target, tuple(notes)))
            return answer.strip().lower() in {"y", "yes"}

        return ask


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
    ) -> process_module.CommandResult:
        command = tuple(str(part) for part in argv)
        self.calls.append(command)
        answer = self.answers.get(command)
        if answer is None:
            return process_module.CommandResult(127, "", f"{command[0]} is not here")
        if isinstance(answer, process_module.CommandResult):
            return answer
        return process_module.CommandResult(0, str(answer), "")

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
            ("git", "checkout", "--detach", "v0.2.0"): process_module.CommandResult(
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
            "uv tool list --outdated": process_module.CommandResult(
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


def test_stopping_an_app_asks_the_process_behind_the_pid_the_app_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app is asked to stop the way its own terminal would ask it.

    The request now goes through the ownership proof, so this watches that seam:
    the pid is still the one the app recorded, and the proof is what is asked.
    """

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    asked: list[tuple[int, Path]] = []
    monkeypatch.setattr(
        update_module.ownership,
        "ask_to_stop",
        lambda pid, root, *args, **kwargs: (
            asked.append((pid, root))
            or ownership_module.Outcome(True, f"pid {pid} was asked to stop")
        ),
    )
    run = Recorder()

    stopped = update_module.stop_app(project, run)

    assert run.calls == []
    assert asked == [(os.getpid(), project.project_root)]
    assert stopped.stopped is True
    assert str(os.getpid()) in stopped.detail


def test_a_pid_the_proof_refuses_is_reported_and_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is reported with its reason, because a number is not a process."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    # This process is alive and therefore passes the record's liveness probe, and
    # it is not this project's app. Nothing is signalled: the proof refuses first.
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    asked: list[int] = []
    real = ownership_module.ask_to_stop

    def _watch(
        pid: int, root: Path, *args: Any, **kwargs: Any
    ) -> ownership_module.Outcome:
        asked.append(pid)
        return real(pid, root, *args, **kwargs)

    monkeypatch.setattr(update_module.ownership, "ask_to_stop", _watch)

    stopped = update_module.stop_app(project, Recorder())

    assert asked == [os.getpid()]
    assert stopped.stopped is False
    assert "this process" in stopped.detail


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
    """`--apply --yes` is the unattended run: the approval was given already."""

    project = update_module.ProjectState(tmp_path / "thesis", "thesis")
    project.state_root.mkdir(parents=True)
    (project.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    asked: list[int] = []
    monkeypatch.setattr(
        update_module.ownership,
        "ask_to_stop",
        lambda pid, root, *args, **kwargs: (
            asked.append(pid)
            or ownership_module.Outcome(True, f"pid {pid} was asked to stop")
        ),
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
    monkeypatch.setattr(process_module, "subprocess_runner", stopped)
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

    payload = _run("update", "--apply", "--yes").payload

    assert payload is not None
    # The app was asked through the pid it recorded, and no command that changes
    # anything was run: the update itself is stubbed below.
    assert not stopped.ran("git", "checkout", "--detach", "v0.2.0")
    assert not stopped.ran("uv", "sync")
    assert asked == [os.getpid()]
    assert payload["applied"] is True
    # What was approved is what was installed, named in the report beside it.
    assert payload["approval"] == {"method": "yes_flag", "granted": True}
    assert payload["previewed_version"] == "0.2.0"
    assert payload["plan"]["release_tag"] == "v0.2.0"
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
    monkeypatch.setattr(process_module, "subprocess_runner", run)

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


def _behind_a_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Recorder:
    """A checkout one release behind, with every probe answered and nothing fetched."""

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
    monkeypatch.setattr(registry, "load", list)
    run = Recorder()
    monkeypatch.setattr(process_module, "subprocess_runner", run)
    return run


def test_the_check_previews_the_published_changelog_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script reads the report; it is never asked and never installs anything."""

    run = _behind_a_release(monkeypatch, tmp_path)

    payload = _run("update").payload

    assert payload is not None
    assert payload["applied"] is False
    assert payload["available"] is True
    assert payload["decision"] == update_module.BEHIND_RELEASE
    assert payload["approval"] == {"method": "not_requested", "granted": False}
    published = payload["published"]
    assert published["state"] == FOUND
    assert published["version"] == "0.2.0"
    assert published["tag"] == "v0.2.0"
    assert published["published"] == "2026-10-01"
    assert published["url"].endswith("/releases/tag/v0.2.0")
    assert "changelog a reader approves" in published["body"]
    assert payload["notes"][-1].startswith("Changelog:")
    assert not run.ran("git", "checkout", "--detach", "v0.2.0")
    assert not run.ran("uv", "sync")
    assert not run.ran("uv", "lock")


def test_apply_where_no_reader_can_be_asked_is_refused_until_yes_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _behind_a_release(monkeypatch, tmp_path)

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply")

    said = str(refused.value)
    assert "cannot be asked" in said
    assert "--yes" in said
    assert "Nothing was changed" in said
    assert not run.ran("git", "checkout", "--detach", "v0.2.0")
    assert not run.ran("uv", "sync")


def test_a_terminal_is_asked_once_and_a_yes_installs_the_release_it_previewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole feature in one call: detect, show, ask, install only on yes."""

    _behind_a_release(monkeypatch, tmp_path)
    question = Question(monkeypatch, "y")
    performed: list[str] = []
    monkeypatch.setattr(
        update_module,
        "apply_plan",
        lambda plan, **_kwargs: [
            performed.extend(plan.labels()) or {"command": "done", "returncode": 0}
        ],
    )

    payload = _run("update").payload

    assert payload is not None
    assert len(question.asked) == 1, "the reader is asked once"
    target, notes = question.asked[0]
    assert "release 0.2.0" in target and "v0.2.0" in target
    assert any("Applying detaches this checkout onto v0.2.0" in note for note in notes)
    assert payload["applied"] is True
    assert payload["approval"] == {"method": "prompt", "granted": True}
    # The version approved is the version installed, named in both places.
    assert payload["previewed_version"] == "0.2.0"
    assert performed == ["git checkout --detach v0.2.0", "uv lock", "uv sync"]


@pytest.mark.parametrize("answer", ["", "\n", "n\n", "no\n", "later\n", "1\n"])
def test_anything_other_than_yes_declines_and_stops_nothing(
    answer: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behind_a_release(monkeypatch, tmp_path)
    Question(monkeypatch, answer)
    stopped: list[Any] = []
    monkeypatch.setattr(
        update_module, "stop_app", lambda *args, **kwargs: stopped.append(args)
    )
    monkeypatch.setattr(
        update_module,
        "apply_plan",
        lambda plan, **_kwargs: pytest.fail("a declined update installs nothing"),
    )

    payload = _run("update").payload

    assert payload is not None
    assert payload["applied"] is False
    assert payload["approval"] == {"method": "declined", "granted": False}
    assert payload["notes"][-1] == "Declined, so nothing was changed."
    assert stopped == []
    assert "steps" not in payload


def test_nothing_is_asked_when_there_is_no_release_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(registry, "load", list)
    monkeypatch.setattr(process_module, "subprocess_runner", Recorder())
    question = Question(monkeypatch, "y")

    payload = _run("update").payload

    assert payload is not None
    assert question.asked == []
    assert payload["decision"] == update_module.AT_RELEASE
    assert payload["applied"] is False


def test_apply_asks_the_same_question_the_bare_command_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behind_a_release(monkeypatch, tmp_path)
    question = Question(monkeypatch, "yes")
    monkeypatch.setattr(update_module, "apply_plan", lambda plan, **_kwargs: [])

    payload = _run("update", "--apply").payload

    assert payload is not None
    assert len(question.asked) == 1
    assert payload["applied"] is True


def test_a_distribution_that_moves_after_the_preview_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool decides the version at install time, so it is asked once more."""

    offered = iter(["0.2.0", "0.3.0"])
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(None, tool="uv")
    )
    monkeypatch.setattr(
        update_module,
        "probe_remote",
        lambda *_args, **_kwargs: update_module.RemoteState(
            reachable=True, available_version=next(offered)
        ),
    )
    monkeypatch.setattr(registry, "load", list)
    monkeypatch.setattr(process_module, "subprocess_runner", Recorder())

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply", "--yes")

    said = str(refused.value)
    assert "previewed as 0.2.0" in said
    assert "0.3.0" in said
    assert "Nothing was changed" in said


def test_a_distribution_whose_version_cannot_be_reread_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(
        [
            update_module.RemoteState(reachable=True, available_version="0.2.0"),
            update_module.RemoteState(reachable=False, detail="network is unreachable"),
        ]
    )
    monkeypatch.setattr(
        update_module, "probe_local", lambda **_kwargs: _local(None, tool="pipx")
    )
    monkeypatch.setattr(
        update_module,
        "probe_remote",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(registry, "load", list)
    monkeypatch.setattr(process_module, "subprocess_runner", Recorder())

    with pytest.raises(ResearchError) as refused:
        _run("update", "--apply", "--yes")

    said = str(refused.value)
    assert "could not be read again" in said
    assert "network is unreachable" in said
    assert "Nothing was changed" in said


def _published_release(body: str = "## 1.1.0\n\n- approved change") -> PublishedRelease:
    return PublishedRelease(
        FOUND,
        GitHubRelease(
            version="1.1.0",
            tag="v1.1.0",
            url="https://github.com/AhmedKishki/research-rag/releases/tag/v1.1.0",
            published="2026-10-04",
            body=body,
        ),
    )


def test_the_changelog_is_shown_on_stderr_and_the_answer_comes_from_stdin(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shown = io.StringIO()

    asked = cli_module._ask_to_install(
        _published_release(),
        ("What applying does.",),
        target="release 1.1.0, tagged v1.1.0",
        stdin=io.StringIO("y\n"),
        stderr=shown,
    )

    said = shown.getvalue()
    assert asked is True
    assert "release 1.1.0, tagged v1.1.0" in said
    assert "## 1.1.0" in said and "- approved change" in said
    assert "https://github.com/AhmedKishki/research-rag/releases/tag/v1.1.0" in said
    assert "What applying does." in said
    assert said.rstrip().endswith(ANSWERED)
    # stdout carries the report, so the question may not put anything there.
    assert capsys.readouterr().out == ""


def test_a_release_with_no_body_is_still_previewed_and_named() -> None:
    shown = io.StringIO()

    cli_module._ask_to_install(
        _published_release(body=""),
        (),
        target="release 1.1.0, tagged v1.1.0",
        stdin=io.StringIO(""),
        stderr=shown,
    )

    said = shown.getvalue()
    assert "release 1.1.0, tagged v1.1.0" in said
    assert "releases/tag/v1.1.0" in said


def test_a_changelog_that_was_not_read_is_said_so_rather_than_invented() -> None:
    shown = io.StringIO()

    cli_module._ask_to_install(
        PublishedRelease(
            update_module.UNREADABLE_RELEASE,
            detail="the published release could not be read: timed out",
        ),
        (),
        target="release 1.1.0, tagged v1.1.0",
        stdin=io.StringIO(""),
        stderr=shown,
    )

    assert "timed out" in shown.getvalue()
    assert "No changelog was published with it." in shown.getvalue()


@pytest.mark.parametrize("answer", ["y\n", "Y\n", "yes\n", "  yes  \n"])
def test_a_yes_installs_and_nothing_else_does(answer: str) -> None:
    asked = cli_module._ask_to_install(
        _published_release(),
        (),
        target="release 1.1.0, tagged v1.1.0",
        stdin=io.StringIO(answer),
        stderr=io.StringIO(),
    )

    assert asked is True


@pytest.mark.parametrize("answer", ["\n", "n\n", "no\n", "sure\n", "", "yes please\n"])
def test_every_other_answer_declines(answer: str) -> None:
    asked = cli_module._ask_to_install(
        _published_release(),
        (),
        target="release 1.1.0, tagged v1.1.0",
        stdin=io.StringIO(answer),
        stderr=io.StringIO(),
    )

    assert asked is False


def test_ctrl_c_at_the_question_is_a_decline_and_not_a_failure() -> None:
    class Interrupted:
        def readline(self) -> str:
            raise KeyboardInterrupt

    asked = cli_module._ask_to_install(
        _published_release(),
        (),
        target="release 1.1.0, tagged v1.1.0",
        stdin=Interrupted(),
        stderr=io.StringIO(),
    )

    assert asked is False


def test_read_preview_answers_without_fetching_stopping_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the browser asks: a read-only report, cached by the caller."""

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\n', encoding="utf-8"
    )
    run = _checkout(
        root, behind="2", changed="pyproject.toml", versions=("0.1.0", "0.2.0")
    )
    # The preview compares the release range, not the branch range, so that is the
    # answer it needs for the pin question.
    run.answers[("git", "-C", str(root), "diff", "--name-only", "HEAD..v0.2.0")] = (
        "pyproject.toml\n"
    )
    monkeypatch.setattr(update_module, "nearest_checkout", lambda: root)
    monkeypatch.setattr(update_module, "installed_version", lambda: "0.1.0")

    preview = update_module.read_preview(run=run, fetch=_fetch(PUBLISHED))

    assert preview["applied"] is False
    assert preview["available"] is True
    assert preview["decision"] == update_module.BEHIND_RELEASE
    assert preview["release"]["version"] == "0.2.0"
    assert preview["release"]["tag"] == "v0.2.0"
    assert preview["published"]["body"] == PUBLISHED["body"]
    assert preview["would_run"] == [
        "git checkout --detach v0.2.0",
        "uv lock",
        "uv sync",
    ]
    assert preview["install_command"] == "research-rag update --apply"
    assert "approval" not in preview, "a preview grants nothing"
    json.dumps(preview)
    # A preview writes nothing: no fetch of the checkout, no tag checkout, no sync.
    assert not run.ran("git", "-C", str(root), "fetch", "--quiet", "--prune", "--tags")
    assert not run.ran("git", "-C", str(root), "checkout", "--detach", "v0.2.0")
    assert not run.ran("uv", "sync")


def test_the_preview_reports_an_unreachable_release_as_a_normal_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(url: str, timeout: float) -> str:
        pytest.fail("an offline preview must not reach the network")

    monkeypatch.setattr(update_module, "nearest_checkout", lambda: None)

    preview = update_module.read_preview(
        run=Recorder(), offline=True, fetch=unreachable
    )

    assert preview["available"] is False
    assert preview["published"]["state"] == "skipped"
    assert "--offline" in preview["published"]["detail"]
    assert preview["published"].get("body") is None
