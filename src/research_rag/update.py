"""Say what a newer version means for this installation, and put it in place.

The command checks by default and writes nothing: it compares what is installed
with the latest published release, reports the branch head beside that answer,
and says whether anything has to stop first. `--apply` performs the update. The
release a remote publishes is read from its tags by `release.py`, which needs no
token and no API key.

Two install shapes exist and neither is assumed. A git checkout is updated by
detaching it onto the release tag and syncing its environment, and the answer
prints the command that returns it to the branch. An installed distribution is
updated to the release version by the tool that owns the install, and that tool
is detected by asking each of `uv` and `pipx` rather than by guessing from `PATH`
order.

Whether an update is available comes from the release comparison alone: a commit
on the branch head that nobody published is unreleased work, which is a fact to
report and not a reason to change anything. Every decision is a pure function of
the two states it is handed, and every subprocess call arrives through an
injected runner, so the behaviour is testable with no remote and no network. An
unreachable remote is a normal answer rather than an error: the report says so,
changes nothing, and names what to try.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import project_command
from .launcher import launcher_path
from .process import COMMAND_TIMEOUT_SECONDS, CommandResult, Runner, subprocess_runner
from .release import (
    NONE,
    ReleaseSet,
    compare_versions,
    declared_version,
    is_release_version,
    remote_release,
)
from .storage import read_json
from .support import ResearchError
from .version import DISTRIBUTION_NAME, installed_version, nearest_checkout

CHECKOUT = "checkout"
DISTRIBUTION = "distribution"
UV = "uv"
PIPX = "pipx"
# Where a checkout stands against the release a remote publishes. `no_release` is
# its own answer: the repository has published nothing to compare against, which
# is a different finding from being at the latest release.
AT_RELEASE = "at_release"
BEHIND_RELEASE = "behind_release"
AHEAD_OF_RELEASE = "ahead_of_release"
NO_RELEASE = "no_release"
UNREADABLE_RELEASE = "unreadable_release"
# The two files that decide what the environment installs. A commit that touches
# neither cannot move a pin, so `uv lock` has nothing to resolve and is skipped.
PIN_FILES = ("pyproject.toml", "uv.lock")
PORTABLE_DIRECTORY = ".research-rag"
RUNTIME_DIRECTORY = "runtime"
PID_FILE = "research-rag-ui.pid"
LOCK_FILE = "project.lock"
# The one line of a generated launcher that records where a project keeps its
# derived state when it is not the default.
_RUNTIME_ROOT_LINE = re.compile(r'^RUNTIME_ROOT="(.*)"$', re.MULTILINE)
# The two commands that put this app on a machine in the first place. A refusal
# names them so a reader can run one instead of being told a guess is wrong.
ALTERNATIVES = (
    "uv tool install git+https://github.com/AhmedKishki/research-rag.git",
    "pipx install git+https://github.com/AhmedKishki/research-rag.git",
)

__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "CommandResult",
    "Runner",
    "subprocess_runner",
]


@dataclass(frozen=True, slots=True)
class LocalState:
    """What this installation is, as far as reading it can tell."""

    shape: str
    version: str
    checkout: Path | None = None
    branch: str | None = None
    upstream: str | None = None
    commit: str | None = None
    tool: str | None = None
    tool_specifier: str = ""
    declared: str | None = None
    dirty: tuple[str, ...] = ()

    @property
    def is_checkout(self) -> bool:
        return self.shape == CHECKOUT

    def as_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {"shape": self.shape, "version": self.version}
        for name in ("checkout", "branch", "upstream", "commit", "tool"):
            value = getattr(self, name)
            report[name] = str(value) if isinstance(value, Path) else value
        if self.tool_specifier:
            report["tool_specifier"] = self.tool_specifier
        if self.declared:
            report["declared_version"] = self.declared
        if self.dirty:
            report["dirty_files"] = list(self.dirty)
        return report


@dataclass(frozen=True, slots=True)
class RemoteState:
    """What the other end offers, or why it could not be asked."""

    reachable: bool
    detail: str = ""
    head: str | None = None
    behind: int | None = None
    available_version: str | None = None
    pins_moving: bool = False
    releases: ReleaseSet = field(default_factory=ReleaseSet)

    def as_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {"reachable": self.reachable}
        for name in ("detail", "head", "behind", "available_version", "pins_moving"):
            report[name] = getattr(self, name)
        report["releases"] = self.releases.as_dict()
        return report


@dataclass(frozen=True, slots=True)
class UpdatePlan:
    """What an update here would do, decided from two states alone."""

    shape: str
    available: bool
    notes: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...] = ()
    behind: int | None = None
    local_revision: str | None = None
    remote_revision: str | None = None
    blocked: str | None = None
    release_version: str | None = None
    release_position: str = ""

    def labels(self) -> tuple[str, ...]:
        return tuple(" ".join(command) for command in self.commands)

    def as_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "shape": self.shape,
            "update_available": self.available,
            "local_revision": self.local_revision,
            "remote_revision": self.remote_revision,
            "would_run": list(self.labels()),
            "notes": list(self.notes),
            "blocked": self.blocked,
        }
        # Whether an update is available is the release comparison, so its two
        # keys are the answer. The commit count beside it is the branch head's
        # position, which is a fact about unreleased work and never the decision.
        if self.release_position:
            report["release_version"] = self.release_version
            report["release_position"] = self.release_position
        if self.behind is not None:
            report["commits_behind_branch"] = self.behind
        return report


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _text(argv: Sequence[str], run: Runner) -> str | None:
    """One command's trimmed output, or None when it failed or said nothing."""

    result = run(argv)
    if not result.ok:
        return None
    return result.stdout.strip() or None


def _owns_tool(tool: str, run: Runner) -> tuple[str, str] | None:
    """What one tool says it installed from, or None when it claims nothing.

    The second element is the specifier `uv` records for the install and an
    empty string for `pipx`, which records none. It is empty rather than guessed
    because a reader is told what the tool said, not what it might have meant.
    """

    if tool == UV:
        result = run(["uv", "tool", "list", "--show-version-specifiers"])
        if not result.ok:
            return None
        for line in result.stdout.splitlines():
            if not line.startswith(f"{DISTRIBUTION_NAME} "):
                continue
            specifier = re.search(r"\(from (.+)\)\s*$", line)
            return (
                DISTRIBUTION_NAME,
                specifier.group(1) if specifier else line.split(maxsplit=1)[1],
            )
        return None
    result = run(["pipx", "list", "--json"])
    if not result.ok:
        return None
    try:
        document = json.loads(result.stdout or "{}")
    except ValueError:
        return None
    venvs = document.get("venvs") if isinstance(document, dict) else None
    if not isinstance(venvs, dict):
        return None
    for venv in venvs.values():
        metadata = venv.get("metadata") if isinstance(venv, dict) else None
        main = metadata.get("main_package") if isinstance(metadata, dict) else None
        packages = main.get("package") if isinstance(main, dict) else None
        if packages and DISTRIBUTION_NAME in packages:
            return DISTRIBUTION_NAME, ""
    return None


def owning_tool(run: Runner) -> tuple[str | None, str | None]:
    """The tool that installed this distribution, found by asking each one."""

    for tool in (UV, PIPX):
        claimed = _owns_tool(tool, run)
        if claimed is not None:
            return tool, claimed[1]
    return None, None


def dirty_files(root: Path, run: Runner) -> tuple[str, ...]:
    """The tracked paths this checkout has uncommitted work in.

    An untracked file is not in the way: `git checkout` keeps it, so it is not
    named. The tracked paths are the ones a move to another commit would carry
    away, and applying refuses while any of them is dirty.
    """

    result = run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
    )
    if not result.ok:
        return ()
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) > 3:
            paths.append(line[3:].strip())
    return tuple(paths)


def probe_local(*, run: Runner) -> LocalState:
    """Read this installation's own state, touching no remote."""

    version = installed_version()
    checkout = nearest_checkout()
    if checkout is not None:
        return LocalState(
            shape=CHECKOUT,
            version=version,
            checkout=checkout,
            branch=_text(
                ["git", "-C", str(checkout), "rev-parse", "--abbrev-ref", "HEAD"], run
            ),
            upstream=_text(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "rev-parse",
                    "--abbrev-ref",
                    "--symbolic-full-name",
                    "@{u}",
                ],
                run,
            ),
            commit=_text(["git", "-C", str(checkout), "rev-parse", "HEAD"], run),
            declared=declared_version(checkout),
            dirty=dirty_files(checkout, run),
        )
    tool, specifier = owning_tool(run=run)
    return LocalState(
        shape=DISTRIBUTION, version=version, tool=tool, tool_specifier=specifier
    )


def _checkout_remote(local: LocalState, run: Runner, *, offline: bool) -> RemoteState:
    """What this checkout's remote publishes: its release tags and its branch head.

    The release tags are asked for over `git ls-remote`, which needs no token and
    no API key, and the branch head is asked for with an ordinary fetch. Both are
    reported, and neither is derived from the other: the branch head says what
    unreleased work this checkout carries, and the release tags say what a reader
    may install.
    """

    root = local.checkout
    if offline:
        return RemoteState(
            reachable=False,
            detail=(
                f"the remote was not contacted because --offline was given, so "
                f"{root} was not fetched"
            ),
        )
    if local.upstream is None:
        return RemoteState(
            reachable=False,
            detail=(
                f"{root} has no upstream branch, so the branch head cannot be "
                "read. Name one with `git branch --set-upstream-to`."
            ),
        )
    fetched = run(["git", "-C", str(root), "fetch", "--quiet", "--prune", "--tags"])
    if not fetched.ok:
        return RemoteState(
            reachable=False,
            detail=f"`git fetch` failed: {_first_line(fetched.stderr or fetched.stdout)}",
        )
    range_expression = f"HEAD..{local.upstream}"
    behind = _text(
        ["git", "-C", str(root), "rev-list", "--count", range_expression], run
    )
    releases = remote_release(root, run)
    # A pin moves when the release's own commits touch one, so the release range
    # decides it and the branch range is only the fallback.
    changed_range = f"HEAD..{releases.tag}" if releases.tag else range_expression
    changed = _text(["git", "-C", str(root), "diff", "--name-only", changed_range], run)
    return RemoteState(
        reachable=True,
        head=_text(["git", "-C", str(root), "rev-parse", local.upstream], run),
        behind=int(behind) if behind and behind.isdigit() else None,
        pins_moving=any(name in (changed or "").split() for name in PIN_FILES),
        releases=releases,
    )


def _distribution_remote(
    local: LocalState, run: Runner, *, offline: bool
) -> RemoteState:
    if offline:
        return RemoteState(
            reachable=False,
            detail=("the release index was not asked because --offline was given"),
        )
    if local.tool is None:
        return RemoteState(reachable=True, detail="no tool claims this installation")
    if local.tool == UV:
        result = run(["uv", "tool", "list", "--outdated"])
        available = _uv_available(result)
    else:
        result = run(
            [
                "pipx",
                "runpip",
                DISTRIBUTION_NAME,
                "index",
                "versions",
                DISTRIBUTION_NAME,
            ]
        )
        available = _pipx_available(result)
    if not result.ok:
        return RemoteState(
            reachable=False,
            detail=(
                f"`{local.tool}` could not be asked what is available: "
                f"{_first_line(result.stderr or result.stdout)}"
            ),
        )
    if available is None:
        return RemoteState(
            reachable=True,
            detail=f"`{local.tool}` named no available version",
        )
    return RemoteState(reachable=True, available_version=available)


def _uv_available(result: CommandResult) -> str | None:
    found = re.search(r"upgrade available to:\s*v?([0-9][^\s`]*)", result.stdout)
    return found.group(1) if found else None


def _pipx_available(result: CommandResult) -> str | None:
    found = re.search(r"Available versions:\s*([0-9][^,\s]*)", result.stdout)
    return found.group(1) if found else None


def probe_remote(
    local: LocalState, run: Runner, *, offline: bool = False
) -> RemoteState:
    """Ask the other end what it offers, or say why it could not be asked."""

    if local.is_checkout:
        return _checkout_remote(local, run, offline=offline)
    return _distribution_remote(local, run, offline=offline)


def _where_sentence(local: LocalState, revision: str) -> str:
    root = local.checkout
    branch = local.branch or "an unknown branch"
    declared = local.declared or local.version
    return f"{root} is on {branch} at {revision} and declares version {declared}."


def _branch_sentence(local: LocalState, remote: RemoteState) -> str:
    """Where the branch head stands, beside every answer about the release.

    A developer on a branch has to know what unreleased work it carries. That is
    a fact to report, and it is never the reason anything is applied.
    """

    if remote.behind is None:
        return (
            f"Where {local.upstream} stands could not be counted: "
            f"{remote.detail or 'git reported no count'}."
        )
    if remote.behind == 0:
        return f"{local.upstream} is level with this checkout."
    return (
        f"{local.upstream} is at {(remote.head or '')[:12] or 'an unknown commit'}, "
        f"{remote.behind} commit(s) ahead."
    )


def _return_command(root: Path, branch: str | None) -> str:
    """The command that puts this checkout back on the branch it left."""

    if branch:
        return f"git -C {root} checkout {branch}"
    return f"git -C {root} branch"


def _release_order(release_version: str, declared: str) -> int | None:
    """Which of two versions is higher, or None when one cannot be compared.

    A version carrying a pre-release suffix or build metadata is not one a
    release tag may carry, so no honest comparison exists and none is invented.
    """

    if not (is_release_version(release_version) and is_release_version(declared)):
        return None
    return compare_versions(release_version, declared)


def _dirty_refusal(root: Path, dirty: Sequence[str]) -> str:
    """The refusal a working tree with uncommitted work gets."""

    shown = list(dirty[:5])
    named = ", ".join(shown)
    if len(dirty) > len(shown):
        named = f"{named}, and {len(dirty) - len(shown)} more"
    return (
        f"Nothing was changed. {root} has uncommitted changes in {named}, and "
        "moving to another commit would carry them away. `git -C "
        f"{root} status` names them; commit them, or set them aside with `git -C "
        f"{root} stash`, then run this again. An untracked file is left where it "
        "is."
    )


def _release_commands(
    local: LocalState, remote: RemoteState
) -> tuple[tuple[str, ...], ...]:
    """Detach onto the release tag, then re-sync the environment it needs."""

    tag = remote.releases.tag or "HEAD"
    commands: list[tuple[str, ...]] = [("git", "checkout", "--detach", tag)]
    if remote.pins_moving:
        commands.append(("uv", "lock"))
    commands.append(("uv", "sync"))
    return tuple(commands)


def _checkout_plan(local: LocalState, remote: RemoteState) -> UpdatePlan:
    """What an update here would do, from the release comparison alone.

    Whether an update is available is decided by comparing the version this
    checkout declares with the latest release the remote publishes. The branch
    head is reported beside that answer and never decides it: a commit nobody
    published is not an update.
    """

    root = local.checkout
    revision = (local.commit or "")[:12] or "an unknown commit"
    where = _where_sentence(local, revision)
    report: dict[str, Any] = {
        "shape": CHECKOUT,
        "local_revision": revision,
        "remote_revision": (remote.head or "")[:12] or None,
        "behind": remote.behind,
    }
    if not remote.reachable:
        return UpdatePlan(
            available=False,
            notes=(
                where,
                f"The remote could not be reached: {remote.detail}.",
                (
                    "Nothing was changed. Retry with a network, or read the local "
                    "state above as the answer."
                ),
            ),
            **report,
        )
    releases = remote.releases
    branch_sentence = _branch_sentence(local, remote)
    if releases.state == NONE:
        return UpdatePlan(
            available=False,
            notes=(
                where,
                f"{releases.detail}, and nothing can be compared with it.",
                branch_sentence,
                (
                    "Publishing a release is a maintainer action: tag the commit "
                    "that is the release with `git tag v<version>`, push the tag, "
                    "and open the release on the repository. Nothing was changed."
                ),
            ),
            release_position=NO_RELEASE,
            **report,
        )
    if not releases.found:
        return UpdatePlan(
            available=False,
            notes=(where, releases.detail, branch_sentence, "Nothing was changed."),
            release_position=UNREADABLE_RELEASE,
            **report,
        )
    release = releases.release
    declared = local.declared or local.version
    order = _release_order(release.version, declared)
    tagged = f"release {release.version}, tagged {release.tag}"
    if order is None:
        return UpdatePlan(
            available=False,
            notes=(
                where,
                (
                    f"The remote publishes {tagged}, which the version this "
                    "checkout declares cannot be compared with, because a release "
                    "version is MAJOR.MINOR.PATCH with no pre-release suffix and "
                    "no build metadata."
                ),
                branch_sentence,
                "Nothing was changed.",
            ),
            release_version=release.version,
            release_position=UNREADABLE_RELEASE,
            **report,
        )
    if order == 0:
        elsewhere = (
            ""
            if not release.commit or release.commit == (local.commit or "")
            else (
                f" It carries that version on commit "
                f"{(local.commit or '')[:12]}, which is not the commit the tag "
                f"names ({release.commit[:12]})."
            )
        )
        return UpdatePlan(
            available=False,
            notes=(
                where,
                (
                    f"{releases.detail}. This checkout declares that version, so "
                    f"it is at {tagged}.{elsewhere}"
                ),
                branch_sentence,
                "Nothing was changed.",
            ),
            release_version=release.version,
            release_position=AT_RELEASE,
            **report,
        )
    if order < 0:
        return UpdatePlan(
            available=False,
            notes=(
                where,
                (
                    f"The remote publishes {tagged}, which is older than the "
                    f"{declared} this checkout declares, so this checkout is ahead "
                    "of the release and carries unreleased work. That is a fact "
                    "about this checkout, not a reason to change anything."
                ),
                branch_sentence,
                "Nothing was changed.",
            ),
            release_version=release.version,
            release_position=AHEAD_OF_RELEASE,
            **report,
        )
    notes = [
        where,
        f"{releases.detail}. This checkout declares {declared}, so it is behind "
        f"{tagged}" + (f" at {release.commit[:12]}." if release.commit else "."),
        (
            f"Applying detaches this checkout onto {release.tag} and re-syncs its "
            "environment. No untracked file is removed and no branch is rewritten."
        ),
        (
            f"Return to the branch afterwards with `{_return_command(root, local.branch)}`."
        ),
    ]
    if local.dirty:
        return UpdatePlan(
            available=True,
            notes=tuple(notes),
            commands=(),
            blocked=_dirty_refusal(root, local.dirty),
            release_version=release.version,
            release_position=BEHIND_RELEASE,
            **report,
        )
    return UpdatePlan(
        available=True,
        notes=tuple(notes),
        commands=_release_commands(local, remote),
        release_version=release.version,
        release_position=BEHIND_RELEASE,
        **report,
    )


def _distribution_plan(local: LocalState, remote: RemoteState) -> UpdatePlan:
    if local.tool is None:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                (
                    f"research-rag {local.version} is installed as a distribution "
                    "with no git metadata at or above it."
                ),
                (
                    "Neither `uv` nor `pipx` claims this installation, so no tool "
                    "owns it and this command will not guess which one to run."
                ),
            ),
            blocked=(
                "This installation belongs to no tool uv or pipx recognises. "
                f"Install it with `{ALTERNATIVES[0]}` or `{ALTERNATIVES[1]}`, or "
                "update the checkout it came from by hand."
            ),
        )
    if not remote.reachable:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                (
                    f"research-rag {local.version} was installed by "
                    f"`{local.tool}"
                    + (
                        f" from {local.tool_specifier}`."
                        if local.tool_specifier
                        else "`."
                    )
                ),
                f"What is available could not be read: {remote.detail}.",
                "Nothing was changed. Retry with a network.",
            ),
            local_revision=local.version,
        )
    if remote.available_version is None:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                f"research-rag {local.version} was installed by `{local.tool}`.",
                f"What is available could not be read: {remote.detail}.",
                "Nothing was changed.",
            ),
            local_revision=local.version,
        )
    if remote.available_version == local.version:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                (
                    f"research-rag {local.version} is the version `{local.tool}` "
                    "offers, so there is nothing to update."
                ),
            ),
            local_revision=local.version,
            remote_revision=remote.available_version,
        )
    return UpdatePlan(
        shape=DISTRIBUTION,
        available=True,
        notes=(
            (
                f"research-rag {local.version} was installed by `{local.tool}`; "
                f"{remote.available_version} is available."
            ),
        ),
        commands=((local.tool, *_upgrade_arguments(local.tool)),),
        local_revision=local.version,
        remote_revision=remote.available_version,
    )


def _upgrade_arguments(tool: str) -> tuple[str, ...]:
    return (
        ("tool", "upgrade", DISTRIBUTION_NAME)
        if tool == UV
        else ("upgrade", DISTRIBUTION_NAME)
    )


def plan_update(local: LocalState, remote: RemoteState) -> UpdatePlan:
    """What an update would do here, from the two states alone."""

    if local.is_checkout:
        return _checkout_plan(local, remote)
    return _distribution_plan(local, remote)


def apply_plan(
    plan: UpdatePlan, *, run: Runner, cwd: Path | None = None
) -> list[dict[str, Any]]:
    """Run the plan's commands in order, stopping at the first that fails."""

    performed: list[dict[str, Any]] = []
    for command in plan.commands:
        label = " ".join(command)
        result = run(command, cwd=cwd)
        performed.append(
            {
                "command": label,
                "returncode": result.returncode,
                "output": _first_line(result.stdout) or _first_line(result.stderr),
            }
        )
        if not result.ok:
            raise ResearchError(
                f"`{label}` failed: "
                f"{_first_line(result.stderr or result.stdout) or 'no output'}. "
                "Nothing after it was attempted, and the working tree is where "
                "that command left it."
            )
    return performed


@dataclass(frozen=True, slots=True)
class ProjectState:
    """One project this installation can serve."""

    project_root: Path
    project_name: str

    @property
    def portable_root(self) -> Path:
        return self.project_root / PORTABLE_DIRECTORY

    @property
    def default_state_root(self) -> Path:
        return self.portable_root / RUNTIME_DIRECTORY

    @property
    def state_root(self) -> Path:
        """Where this project's launcher keeps its pid, its lock, and its staging.

        A relocated runtime root is recorded in the launcher that acts on it, so
        the answer is read from that file rather than from a second rule about
        where a project keeps its derived state.
        """

        script = launcher_path(self.portable_root)
        try:
            text = script.read_text(encoding="utf-8")
        except OSError:
            return self.default_state_root
        found = _RUNTIME_ROOT_LINE.search(text)
        if found is None or not found.group(1).strip():
            return self.default_state_root
        return Path(found.group(1).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "project_name": self.project_name,
        }


@dataclass(frozen=True, slots=True)
class HeldProject:
    """A project whose lock another process is holding."""

    project: ProjectState
    pid: int
    phase: str | None = None
    build_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project.project_root),
            "project_name": self.project.project_name,
            "pid": self.pid,
            "phase": self.phase,
            "build_id": self.build_id,
            "report_command": project_command(self.project.project_root, "status"),
        }

    def refusal(self) -> str:
        where = (
            f" (build {self.build_id}, in {self.phase})"
            if self.phase
            else " (in a phase it does not record)"
        )
        return (
            f"{self.project.project_name} at {self.project.project_root} is being "
            f"worked on: process {self.pid} holds its project lock{where}. An "
            f"update would replace the code that work is running. Run "
            f"`{project_command(self.project.project_root, 'status')}` to see the "
            "build, and run this again once it has finished."
        )


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def resident_build(state_root: Path) -> tuple[str | None, str | None]:
    """The phase and build id a project's newest checkpoint records, if any.

    The same two fields a service names when it refuses a second build. This
    command reads them from the files themselves because it must answer before
    any project is resolved, and it will not build a service to ask.
    """

    staging = state_root / "staging"
    try:
        roots = sorted(
            staging.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True
        )
    except OSError:
        return None, None
    for root in roots:
        checkpoint = root / "checkpoint.json"
        if not checkpoint.is_file():
            continue
        try:
            document = read_json(checkpoint)
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        phase = document.get("phase")
        build_id = document.get("build_id")
        return (
            str(phase) if isinstance(phase, str) and phase else None,
            str(build_id) if isinstance(build_id, str) and build_id else None,
        )
    return None, None


def held_projects(projects: Sequence[ProjectState]) -> tuple[HeldProject, ...]:
    """Every project whose lock another live process is holding."""

    held: list[HeldProject] = []
    for project in projects:
        path = project.state_root / LOCK_FILE
        try:
            recorded = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not recorded.isdigit():
            continue
        pid = int(recorded)
        if pid == os.getpid() or not _process_alive(pid):
            continue
        phase, build_id = resident_build(project.state_root)
        held.append(HeldProject(project, pid, phase, build_id))
    return tuple(held)


@dataclass(frozen=True, slots=True)
class StoppedProject:
    """One project's app, and whether it was running."""

    project: ProjectState
    stopped: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        report = self.project.as_dict()
        report.update({"stopped": self.stopped, "detail": self.detail})
        return report


def recorded_pid(project: ProjectState) -> int | None:
    """The pid a project's launcher recorded, when that process is still alive."""

    try:
        recorded = (project.state_root / PID_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not recorded.isdigit():
        return None
    pid = int(recorded)
    return pid if _process_alive(pid) else None


def stop_app(project: ProjectState, run: Runner) -> StoppedProject:
    """Stop one project's app through its own launcher, so the group stops too.

    The launcher owns the pid, the port, and the process group, so this command
    does not signal anything itself. A project with no live app is left alone.
    """

    if recorded_pid(project) is None:
        return StoppedProject(project, False, "no app was running for this project")
    script = launcher_path(project.portable_root)
    if not script.is_file():
        return StoppedProject(
            project,
            False,
            f"no launcher at {script}, so nothing was stopped",
        )
    result = run([str(script), "--stop"])
    detail = _first_line(result.stdout) or _first_line(result.stderr)
    return StoppedProject(project, result.ok, detail or "the launcher said nothing")


def portable_state_digest(project_root: Path) -> dict[str, tuple[int, int]]:
    """Each portable state file's size and modification time, `runtime/` excluded.

    The digest answers one question an update must be able to answer: did the
    code change touch anything the project owns? `runtime/` rebuilds from the
    sources and is excluded because stopping an app writes its pid file there.
    """

    portable = project_root / PORTABLE_DIRECTORY
    digest: dict[str, tuple[int, int]] = {}
    for path in sorted(portable.rglob("*")):
        relative = path.relative_to(portable)
        if RUNTIME_DIRECTORY in relative.parts:
            continue
        try:
            observed = path.stat()
        except OSError:
            continue
        digest[relative.as_posix()] = (observed.st_size, observed.st_mtime_ns)
    return digest


def state_changes(
    before: dict[str, dict[str, tuple[int, int]]],
    after: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, list[str]]:
    """The portable state files each project gained, lost, or rewrote."""

    changes: dict[str, list[str]] = {}
    for project_root, digests in before.items():
        other = after.get(project_root, {})
        touched = sorted(
            name
            for name in set(digests) | set(other)
            if digests.get(name) != other.get(name)
        )
        if touched:
            changes[project_root] = touched
    return changes
