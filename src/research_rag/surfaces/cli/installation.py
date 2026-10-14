"""`install` and `update`: this installation's PATH entry and its releases."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...runtime.release import PublishedRelease

from ...project.config import (
    project_command,
)
from ...project.policy import ResearchError
from ...runtime import process as process_module


def _install(args: argparse.Namespace) -> dict[str, Any]:
    """Make this installation reachable, or take back what it made reachable.

    One command with two selectors rather than two commands: an uninstall has to
    know which of the two the reader meant, and `--desktop` beside `--uninstall`
    answers that.
    """

    from ...runtime.installation import (
        install_console_entry,
        install_desktop_entry,
        uninstall_console_entry,
        uninstall_desktop_entry,
    )

    if args.desktop:
        # The entry belongs to this installation rather than to a project, so no
        # project is resolved: resolving one writes its portable state, and adding
        # a menu entry must touch nothing a project owns.
        if args.uninstall:
            report = uninstall_desktop_entry()
        else:
            report = install_desktop_entry(force=args.force)
        return {"command": "install", "desktop": report}
    if args.uninstall:
        report = uninstall_console_entry()
    else:
        report = install_console_entry(force=args.force)
    return {"command": "install", "console": report.as_dict()}


def _interactive_terminal() -> bool:
    """Whether a reader can be asked here, rather than a pipe or a log read.

    Both ends matter. The question is written to stderr and the answer is read
    from stdin, so one terminal and one file is a prompt in a log rather than a
    reader, and that is a check that prints and installs nothing.
    """

    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _fetch_published_release(url: str, timeout: float) -> str:
    """The one network request this command makes, named so a test can replace it.

    It reads the published release of this app's public repository and carries no
    token, key, or header that could. `--offline` never reaches it, and nothing is
    installed because of what it returns: it is the changelog a reader approves.
    """

    from ...runtime.release import fetch_text

    return fetch_text(url, timeout)


def _ask_to_install(
    published: PublishedRelease,
    notes: Sequence[str],
    *,
    target: str,
    stdin: Any = None,
    stderr: Any = None,
) -> bool:
    """Show the published changelog and ask once, with no as the answer.

    Everything is written to stderr, so stdout stays the machine-readable report
    the other flags promise, and the answer is one line. Anything that is not
    `y` or `yes` declines: an empty line, EOF, and Ctrl-C are the same no.
    Declining is an answer, not a failure, so it changes nothing and returns.
    """

    from ...runtime import release as release_module

    source = sys.stdin if stdin is None else stdin
    sink = sys.stderr if stderr is None else stderr
    lines = [f"A published release is available: {target}."]
    if published.found and published.release is not None:
        lines.extend(release_module.changelog_lines(published.release))
    else:
        lines.append(f"No changelog was published with it. {published.detail}.")
    lines.extend(("", *notes))
    sink.write("\n".join(lines) + "\nInstall this release? [y/N]: ")
    sink.flush()
    try:
        answer = source.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        answer = ""
    return answer.strip().lower() in {"y", "yes"}


async def _update(args: argparse.Namespace) -> dict[str, Any]:
    """Report what a newer version means, preview its changelog, install it approved.

    The check is the default and writes nothing. The comparison is against the
    latest published release, so the version in the answer and the version
    `research-rag --version` prints are the same number, and the changelog shown
    beside that answer is the body the maintainer published with the release.

    An install is the one thing here a reader approves: a terminal is asked once
    with the changelog in front of it and no as the default, `--apply` asks the
    same question, and `--yes` says the approval was given already. A terminal
    that cannot be asked is refused rather than assumed, and an answer that is not
    yes stops nothing and writes nothing. Applying also refuses a build that holds
    a project lock, refuses a checkout with uncommitted work, stops every app
    through that project's own recorded pid, and prints the command that starts
    each one again.
    """

    from ...project.registry import load as load_registered
    from ...runtime import update as update_module
    from ...runtime.version import version_block

    offline = bool(args.offline)
    runner = process_module.subprocess_runner
    local = update_module.probe_local(run=runner)
    remote = update_module.probe_remote(local, runner, offline=offline)
    plan = update_module.plan_update(local, remote)
    payload: dict[str, Any] = {
        "command": "update",
        "applied": False,
        # The two keys a caller reads instead of deriving: whether a release is
        # waiting to be installed, and where this installation stands against
        # what the remote publishes. A branch head is never either of them.
        "available": plan.available,
        "decision": plan.release_position,
        "notes": list(plan.notes),
        # The same four numbers `research-rag --version` prints, from the same
        # functions, so an answer cannot carry one version and the flag another.
        "version": version_block(),
        "install": local.as_dict(),
        "remote": remote.as_dict(),
        "plan": plan.as_dict(),
    }
    if not args.apply and local.is_checkout and local.checkout is not None:
        # On the reporting path only: an applied update moves the checkout, and a
        # tag it read a moment before may be one commit from being irrelevant.
        from ...runtime import release as release_module

        payload["declared_version_problems"] = list(
            release_module.release_consistency(
                local.checkout, local.declared or local.version, runner
            )
        )
    if plan.blocked:
        raise ResearchError(plan.blocked)
    if args.apply and offline:
        raise ResearchError(
            "--offline cannot update: nothing on the remote was asked. Nothing "
            "was changed. Run this again without --offline."
        )
    if args.apply and not plan.available:
        # Stopping an app is not free, so a run with nothing to apply changes
        # nothing at all, and says why.
        raise ResearchError("Nothing to apply. " + " ".join(plan.notes))

    # What the reader would approve, read before anything is asked or changed.
    published = update_module.published_release_for(
        local, runner, offline=offline, fetch=_fetch_published_release
    )
    notes = list(update_module.preview_notes(plan, published))
    payload["published"] = published.as_dict()
    payload["notes"] = notes
    target = (
        f"release {plan.release_version}, tagged {plan.release_tag}"
        if local.is_checkout
        else f"version {plan.remote_revision}"
    )
    projects = [
        update_module.ProjectState(entry.project_root, entry.project_name)
        for entry in load_registered()
        if (entry.project_root / ".research-rag").is_dir()
    ]
    held = update_module.held_projects(projects)
    interactive = _interactive_terminal()
    # A terminal that shows a preview is offered the install once; `--yes` is not
    # an answer to a question nobody asked, so it never asks and never offers.
    apply_requested = bool(args.apply) or (
        interactive and plan.available and not args.yes
    )
    refusal = "Nothing was changed. " + " ".join(one.refusal() for one in held)
    if args.apply and held:
        raise ResearchError(refusal)
    if held:
        payload["projects"] = [project.as_dict() for project in projects]
        payload["held"] = [one.as_dict() for one in held]
        notes.append(
            f"{len(held)} project(s) are being worked on, so an update would have "
            "to stop them first."
        )
    approval = update_module.approval_for(
        apply_requested=apply_requested,
        preselected=bool(args.yes),
        interactive=interactive,
    )
    payload["approval"] = approval.as_dict()
    if approval.method == update_module.UNAVAILABLE:
        raise ResearchError(update_module.approval_refusal())
    if approval.asked and plan.available:
        # The question goes to stderr with the changelog above it, and stdout
        # carries the report either way.
        approval = update_module.answered(
            approval, _ask_to_install(published, notes, target=target)
        )
        payload["approval"] = approval.as_dict()
    if not apply_requested:
        return payload
    if not approval.granted:
        payload["notes"] = [*notes, "Declined, so nothing was changed."]
        return payload
    expected_version = plan.release_version or plan.remote_revision
    if not published.found or published.version != expected_version:
        raise ResearchError(
            "Nothing was changed. The published GitHub release and changelog "
            f"must match the approved version {expected_version}. "
            "Run 'research-rag update' again after release information is available."
        )
    # An approved release is the one previewed, not whatever is newest by now.
    moved = update_module.target_moved(local, plan, runner, offline=offline)
    if moved:
        raise ResearchError(moved)
    if held:
        raise ResearchError(refusal)
    payload["projects"] = [project.as_dict() for project in projects]
    payload["previewed_version"] = plan.release_version or plan.remote_revision

    def digests() -> dict[str, dict[str, tuple[int, int]]]:
        return {
            str(project.project_root): update_module.portable_state_digest(
                project.project_root
            )
            for project in projects
        }

    def state_sentence(before_run: dict[str, dict[str, tuple[int, int]]]) -> str:
        changes = update_module.state_changes(before_run, digests())
        if not changes:
            return "no project's portable state changed."
        return "the portable state that changed: " + "; ".join(
            f"{root} ({', '.join(names)})" for root, names in sorted(changes.items())
        )

    before = digests()
    stopped = [update_module.stop_app(project, runner) for project in projects]
    payload["stopped"] = [one.as_dict() for one in stopped]
    try:
        payload["steps"] = update_module.apply_plan(
            plan, run=runner, cwd=local.checkout
        )
    except ResearchError as exc:
        raise ResearchError(
            f"{exc} After the failed step, {state_sentence(before)}"
        ) from exc
    applied = update_module.probe_local(run=runner)
    payload["applied"] = True
    payload["install_after"] = applied.as_dict()
    changes = update_module.state_changes(before, digests())
    payload["project_state_changes"] = changes
    payload["project_state_untouched"] = not changes
    payload["start_again"] = [
        project_command(one.project.project_root, "start")
        for one in stopped
        if one.stopped
    ]
    return payload
