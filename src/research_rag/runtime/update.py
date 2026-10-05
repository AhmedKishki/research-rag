"""Say what a newer version means for this installation, and put it in place.

The command checks by default and writes nothing: it compares what is installed
with the latest published release, reports the branch head beside that answer,
previews the changelog the release was published with, and says whether anything
has to stop first. `--apply` performs the update, and only after a reader approves
it. The release a remote publishes is read from its tags by `release.py`, which
needs no token and no API key, and the changelog that describes it is read from
the same repository's published release, which needs no credential either.

Nothing is installed without an approval that names the version it approves. A
terminal is asked once, with the changelog in front of it and no as the default
answer; a terminal that cannot be asked is refused, and `--yes` is how automation
says the approval was given already. An answer that is not yes — nothing typed,
EOF, Ctrl-C, or a plain no — is a declined update and nothing else: no process is
stopped, no checkout moves, and no file is written. What the reader approved is
the version the preview showed, and an installation whose target has changed
since is refused rather than moved to a different release than the one previewed.

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
changes nothing, and names what to try. The same holds for a changelog that could
not be read, which is reported beside the release answer rather than in place of
it.

`read_preview` is the one helper another surface calls. It answers the same
question read-only: it fetches nothing, writes nothing, stops nothing, and returns
plain JSON, so a browser can ask what is published and show the changelog without
installing anything itself.

What left this module, and why. `tool_ownership.py` took the `uv` and `pipx`
vocabulary: the detection that asks each tool whether it claims this
distribution, the commands each takes to report and to perform an upgrade, the
install commands a refusal names, and that refusal's prose. `state_files.py` took
the on-disk names, `ProjectState`, and `resident_build`, so the two questions
this module asked about a project it has not resolved are answered beside the
paths they walk. What remains is the decision, the two states it is made from,
and the approval that stands between the decision and the install.

It imports no retrieval module, and must not begin to: `research-rag update` is
one of the commands that has to answer on a machine where the retrieval stack is
not importable. `ResearchError` still has no leaf home, so the failure path below
pays for the import that one exception needs and the reporting path does not.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..project.config import CLI_COMMAND, project_command
from ..project.state_files import (
    LOCK_FILE,
    PORTABLE_DIRECTORY,
    RUNTIME_DIRECTORY,
    ProjectState,
    process_alive,
    recorded_pid,
    resident_build,
)
from . import ownership
from .process import Runner, subprocess_runner
from .release import (
    NONE,
    Fetch,
    PublishedRelease,
    ReleaseSet,
    compare_versions,
    declared_version,
    is_release_version,
    published_release,
    remote_release,
    skipped_published_release,
)
from .tool_ownership import (
    available_version,
    outdated_command,
    owning_tool,
    unowned_refusal,
    upgrade_arguments,
)
from .version import (
    DISTRIBUTION_NAME,
    installed_version,
    nearest_checkout,
    version_block,
)

CHECKOUT = "checkout"
DISTRIBUTION = "distribution"
# Where a checkout stands against the release a remote publishes. `no_release` is
# its own answer: the repository has published nothing to compare against, which
# is a different finding from being at the latest release.
AT_RELEASE = "at_release"
BEHIND_RELEASE = "behind_release"
AHEAD_OF_RELEASE = "ahead_of_release"
NO_RELEASE = "no_release"
UNREADABLE_RELEASE = "unreadable_release"
# A distribution stands against its release the same way, and one installation no
# tool claims has no position at all: there is nothing to compare and nothing that
# could install it.
UNOWNED_INSTALL = "unowned_install"
# The two files that decide what the environment installs. A commit that touches
# neither cannot move a pin, so `uv lock` has nothing to resolve and is skipped.
PIN_FILES = ("pyproject.toml", "uv.lock")

# How an approval was reached. `not_requested` is a check that changes nothing;
# `offered` is a check in a terminal that will ask once a release is previewed;
# `prompt` is `--apply` in a terminal; `yes_flag` is `--apply --yes`, where the
# approval was given before the command ran; `declined` is an answer that was not
# yes; and `approval_unavailable` is `--apply` where no question can be asked.
NOT_REQUESTED = "not_requested"
OFFERED = "offered"
PROMPT = "prompt"
YES_FLAG = "yes_flag"
DECLINED = "declined"
UNAVAILABLE = "approval_unavailable"


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
    release_tag: str | None = None
    release_commit: str | None = None
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
            report["release_tag"] = self.release_tag
            report["release_position"] = self.release_position
        if self.behind is not None:
            report["commits_behind_branch"] = self.behind
        return report


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _failure(message: str) -> Exception:
    """The one error this command raises, reached without a module-level import.

    `support` imports the retrieval stack, and this command has to answer on a
    machine where that stack is absent. Until `ResearchError` has a leaf home the
    failure path pays for the import and every other path does not.
    """

    from ..project.policy import ResearchError

    return ResearchError(message)


def _text(argv: Sequence[str], run: Runner) -> str | None:
    result = run(argv)
    if not result.ok:
        return None
    return result.stdout.strip() or None


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


def origin_url(root: Path | None, run: Runner) -> str | None:
    """The URL this checkout's `origin` names, or None when it names none.

    Read for one reason: a published changelog is this app's, and a remote that
    names another repository publishes notes that are not. A checkout with no
    remote, and an installed distribution, both name none.
    """

    if root is None:
        return None
    result = run(["git", "-C", str(root), "remote", "get-url", "origin"])
    if not result.ok:
        return None
    return result.stdout.strip() or None


def _checkout_remote(
    local: LocalState, run: Runner, *, offline: bool, refresh: bool = True
) -> RemoteState:
    """What this checkout's remote publishes: its release tags and its branch head.

    The release tags are asked for over `git ls-remote`, which needs no token and
    no API key, and the branch head is asked for with an ordinary fetch. Both are
    reported, and neither is derived from the other: the branch head says what
    unreleased work this checkout carries, and the release tags say what a reader
    may install.

    `refresh=False` fetches nothing and reads the branch position from the refs
    this checkout already holds, which is what a preview that must write nothing
    needs. The release answer stays a live one: `ls-remote` asks the remote and
    stores nothing.
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
    if refresh:
        fetched = run(["git", "-C", str(root), "fetch", "--quiet", "--prune", "--tags"])
        if not fetched.ok:
            return RemoteState(
                reachable=False,
                detail=(
                    f"`git fetch` failed: "
                    f"{_first_line(fetched.stderr or fetched.stdout)}"
                ),
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
    result = run(outdated_command(local.tool))
    available = available_version(local.tool, result)
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


def probe_remote(
    local: LocalState, run: Runner, *, offline: bool = False, refresh: bool = True
) -> RemoteState:
    if local.is_checkout:
        return _checkout_remote(local, run, offline=offline, refresh=refresh)
    return _distribution_remote(local, run, offline=offline)


def published_release_for(
    local: LocalState,
    run: Runner,
    *,
    offline: bool = False,
    fetch: Fetch | None = None,
) -> PublishedRelease:
    """The changelog a reader would approve, read from where the release was published.

    `offline` is answered rather than attempted, and the answer says that nothing
    was asked of the network. The fetch is injected so a test reads a recorded
    payload and never reaches the API.
    """

    if offline:
        return skipped_published_release(
            "the published release was not asked because --offline was given, so "
            "there is no changelog to show"
        )
    return published_release(remote_url=origin_url(local.checkout, run), fetch=fetch)


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
            release_position=UNREADABLE_RELEASE,
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
            release_tag=release.tag,
            release_commit=release.commit,
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
            release_tag=release.tag,
            release_commit=release.commit,
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
            release_tag=release.tag,
            release_commit=release.commit,
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
            release_tag=release.tag,
            release_commit=release.commit,
            release_position=BEHIND_RELEASE,
            **report,
        )
    return UpdatePlan(
        available=True,
        notes=tuple(notes),
        commands=_release_commands(local, remote),
        release_version=release.version,
        release_tag=release.tag,
        release_commit=release.commit,
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
                    f"{DISTRIBUTION_NAME} {local.version} is installed as a "
                    "distribution with no git metadata at or above it."
                ),
                (
                    "Neither `uv` nor `pipx` claims this installation, so no tool "
                    "owns it and this command will not guess which one to run."
                ),
            ),
            blocked=unowned_refusal(),
            release_position=UNOWNED_INSTALL,
        )
    if not remote.reachable:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                (
                    f"{DISTRIBUTION_NAME} {local.version} was installed by "
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
            release_position=UNREADABLE_RELEASE,
        )
    if remote.available_version is None:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                f"{DISTRIBUTION_NAME} {local.version} was installed by `{local.tool}`.",
                f"What is available could not be read: {remote.detail}.",
                "Nothing was changed.",
            ),
            local_revision=local.version,
            release_position=UNREADABLE_RELEASE,
        )
    if remote.available_version == local.version:
        return UpdatePlan(
            shape=DISTRIBUTION,
            available=False,
            notes=(
                (
                    f"{DISTRIBUTION_NAME} {local.version} is the version "
                    f"`{local.tool}` offers, so there is nothing to update."
                ),
            ),
            local_revision=local.version,
            remote_revision=remote.available_version,
            release_position=AT_RELEASE,
        )
    return UpdatePlan(
        shape=DISTRIBUTION,
        available=True,
        notes=(
            (
                f"{DISTRIBUTION_NAME} {local.version} was installed by "
                f"`{local.tool}`; {remote.available_version} is available."
            ),
        ),
        commands=((local.tool, *upgrade_arguments(local.tool)),),
        local_revision=local.version,
        remote_revision=remote.available_version,
        release_position=BEHIND_RELEASE,
    )


def plan_update(local: LocalState, remote: RemoteState) -> UpdatePlan:
    if local.is_checkout:
        return _checkout_plan(local, remote)
    return _distribution_plan(local, remote)


@dataclass(frozen=True, slots=True)
class Approval:
    """How this call reached an install, and whether it may do one.

    An approval is never inferred. It is either given before the command ran
    (`--yes`), or read from a reader, or it is absent — and an absent approval
    installs nothing.
    """

    method: str
    granted: bool = False

    @property
    def asked(self) -> bool:
        """Whether a reader is waiting to be asked, so a prompt is owed."""

        return self.method in {OFFERED, PROMPT}

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "granted": self.granted}


def approval_for(
    *, apply_requested: bool, preselected: bool, interactive: bool
) -> Approval:
    """The approval this call needs, from three facts and nothing else.

    `preselected` is `--yes`, which is how automation says the approval was given
    already. `interactive` is whether a reader can be asked at all. A call that
    asked for an install and can be asked nothing is refused rather than assumed.
    """

    if not apply_requested:
        return Approval(OFFERED if interactive else NOT_REQUESTED)
    if preselected:
        return Approval(YES_FLAG, granted=True)
    return Approval(PROMPT if interactive else UNAVAILABLE)


def answered(approval: Approval, answer: bool) -> Approval:
    """The approval a reader gave: yes, or anything else, which is no.

    An answer that is not yes needs no test for which key it was: empty, EOF,
    Ctrl-C, and every other spelling are the same declined update.
    """

    if not approval.asked:
        return approval
    return Approval(PROMPT if answer else DECLINED, granted=answer)


def approval_refusal() -> str:
    """What an `--apply` in a terminal that cannot answer is told.

    Refused rather than assumed, because installing code is the one thing this
    command does that a reader cannot undo by reading the answer afterwards.
    """

    return (
        "Nothing was changed. --apply installs only what a reader approves, and "
        "this terminal cannot be asked. Pass --yes to say the approval is already "
        f"given, or run `{CLI_COMMAND} update` without --apply to read the "
        "changelog first."
    )


def target_moved(
    local: LocalState, plan: UpdatePlan, run: Runner, *, offline: bool = False
) -> str | None:
    """A refusal when installing would move somewhere other than the preview.

    A checkout is detached onto one tag, decided when the plan was built, so the
    commit the preview named is the commit it lands on. An installed distribution
    is upgraded by the tool that owns it, and that tool picks whatever is newest
    at the moment it runs, so its answer is asked once more and a changed answer
    refuses the install rather than delivering a release nobody approved.
    """

    if local.is_checkout:
        return None
    previewed = plan.remote_revision
    remote = probe_remote(local, run, offline=offline)
    if not remote.reachable:
        return (
            "Nothing was changed. The version to install was read as "
            f"{previewed}, and it could not be read again: {remote.detail}. Run "
            "this again once it can be."
        )
    if remote.available_version != previewed:
        return (
            "Nothing was changed. The version to install was previewed as "
            f"{previewed}, and `{local.tool}` now offers "
            f"{remote.available_version or 'no version'}. Read the new changelog "
            f"and approve that one: `{CLI_COMMAND} update`."
        )
    return None


def preview_notes(plan: UpdatePlan, published: PublishedRelease) -> tuple[str, ...]:
    """The report beside the preview: what was compared, and what was read."""

    return (*plan.notes, f"Changelog: {published.detail}")


def read_preview(
    *,
    run: Runner | None = None,
    offline: bool = False,
    fetch: Fetch | None = None,
) -> dict[str, Any]:
    """What is installed, what is published, and the changelog to approve.

    The one helper a surface other than this command line calls, and it answers
    without touching anything: no fetch of the checkout, no tag checkout, no
    process signal, no file written, no model downloaded. What it returns is
    plain JSON, so a caller may cache it and render the body as text.

    A browser asks for this when a reader opens it, off the event loop that serves
    the workspace, and never to install anything: an install needs a terminal the
    reader answers in, which is what the command line is. `install_command` names
    that terminal command, and it takes the reader's own approval.
    """

    runner = run if run is not None else subprocess_runner
    local = probe_local(run=runner)
    remote = probe_remote(local, runner, offline=offline, refresh=False)
    plan = plan_update(local, remote)
    published = published_release_for(local, runner, offline=offline, fetch=fetch)
    return {
        "command": "update",
        # An approval is never carried across into another surface, so a preview
        # grants nothing and reports `applied: false` whatever it finds.
        "applied": False,
        "available": plan.available,
        "decision": plan.release_position,
        "version": version_block(),
        "install": local.as_dict(),
        "release": {
            "version": plan.release_version,
            "tag": plan.release_tag,
            "commit": plan.release_commit,
            "position": plan.release_position,
            "detail": remote.releases.detail,
        },
        "published": published.as_dict(),
        "would_run": list(plan.labels()),
        "notes": list(preview_notes(plan, published)),
        "blocked": plan.blocked,
        "install_command": f"{CLI_COMMAND} update --apply",
    }


def apply_plan(
    plan: UpdatePlan, *, run: Runner, cwd: Path | None = None
) -> list[dict[str, Any]]:
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
            raise _failure(
                f"`{label}` failed: "
                f"{_first_line(result.stderr or result.stdout) or 'no output'}. "
                "Nothing after it was attempted, and the working tree is where "
                "that command left it."
            )
    return performed


@dataclass(frozen=True, slots=True)
class HeldProject:
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


def held_projects(projects: Sequence[ProjectState]) -> tuple[HeldProject, ...]:
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
        if pid == os.getpid() or not process_alive(pid):
            continue
        phase, build_id = resident_build(project.state_root)
        held.append(HeldProject(project, pid, phase, build_id))
    return tuple(held)


@dataclass(frozen=True, slots=True)
class StoppedProject:
    project: ProjectState
    stopped: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        report = self.project.as_dict()
        report.update({"stopped": self.stopped, "detail": self.detail})
        return report


def stop_app(project: ProjectState, run: Runner) -> StoppedProject:
    """Ask one project's app to stop, the way its own terminal would.

    The app records its own pid, so that number is where this looks. A number is
    not a process, so `ownership` proves the process behind it runs this app for
    this project before any signal is sent, and a record that has been reused
    since it was written leaves its process running and says why.
    """

    pid = recorded_pid(project.state_root)
    if pid is None:
        return StoppedProject(project, False, "no app was running for this project")
    outcome = ownership.ask_to_stop(pid, project.project_root)
    if not outcome.asked:
        return StoppedProject(project, False, outcome.reason)
    return StoppedProject(
        project,
        True,
        f"pid {pid} was asked to stop; the terminal that started it ends it.",
    )


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


__all__ = [
    "AHEAD_OF_RELEASE",
    "AT_RELEASE",
    "BEHIND_RELEASE",
    "CHECKOUT",
    "DECLINED",
    "DISTRIBUTION",
    "NOT_REQUESTED",
    "NO_RELEASE",
    "OFFERED",
    "PIN_FILES",
    "PROMPT",
    "UNAVAILABLE",
    "UNOWNED_INSTALL",
    "UNREADABLE_RELEASE",
    "YES_FLAG",
    "Approval",
    "HeldProject",
    "LocalState",
    "ProjectState",
    "RemoteState",
    "StoppedProject",
    "UpdatePlan",
    "answered",
    "apply_plan",
    "approval_for",
    "approval_refusal",
    "dirty_files",
    "held_projects",
    "origin_url",
    "plan_update",
    "portable_state_digest",
    "preview_notes",
    "probe_local",
    "probe_remote",
    "published_release_for",
    "read_preview",
    "state_changes",
    "stop_app",
    "target_moved",
]
