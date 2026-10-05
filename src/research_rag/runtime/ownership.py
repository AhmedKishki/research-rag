"""Proof that a process is one this project may signal, and the signalling itself.

`AGENTS.md` holds the rule: signal only proven project-owned processes, requiring
the app entrypoint and a matching `--project-root`, and never the sweep itself. A
pid record names a number, and a number alone proves nothing: the record can be
stale, the number can have been reused, and the process holding it now may be
something else entirely. `recorded_pid` answers whether a number is alive; this
module answers whether the thing behind it is ours.

Three facts have to hold together, and all three are read from `/proc` rather than
inferred:

- the process runs this app, as a console script or as ``python -m research_rag``;
- it names this project as ``--project-root``, exactly, because a relative path
  would be resolved against the other process's own directory;
- it is neither this process nor one this process descends from, because a sweep
  that signalled its own parent would take the terminal running it.

Signalling goes through a pidfd where the kernel has one. That is what makes a
reused number harmless: a pidfd names the process that held the number when it was
opened, so a signal sent through it cannot reach whoever holds the number now. A
kernel without `pidfd_open`, or a refusal to open one, leaves the target running
and names the reason. There is no fallback to signalling a bare number, because a
bare number is the case this module exists to refuse.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The program names that are this app. A console script passes its own path and
#: `python -m research_rag` passes the module, so both appear.
PROGRAM_NAMES = frozenset({"research_rag", "research-rag"})


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a pid is a process this project may signal, and why."""

    owned: bool
    reason: str
    command: str = ""


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of asking one process to stop."""

    asked: bool
    reason: str


def arguments_of(pid: int, proc_root: Path = Path("/proc")) -> list[str] | None:
    """The argument vector of one pid, or None when it cannot be read."""

    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def parent_of(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """The ppid recorded for one pid, or None when it cannot be read.

    `stat` is read after the field holding the executable name, because that name
    is parenthesised and may itself contain spaces.
    """

    try:
        raw = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    try:
        fields = raw[raw.rindex(")") + 1 :].split()
        recorded = fields[1]
    except (ValueError, IndexError):
        return None
    return int(recorded) if recorded.isdigit() else None


def lineage(pid: int, proc_root: Path = Path("/proc")) -> Iterator[int]:
    """This pid and every pid it descends from, nearest first."""

    seen: set[int] = set()
    current: int | None = pid
    while current is not None and current > 0 and current not in seen:
        seen.add(current)
        yield current
        current = parent_of(current, proc_root)


def invokes_this_app(arguments: Sequence[str], project_root: Path) -> bool:
    """Whether one argument vector runs this app as the program being run.

    A project path under `.research-rag` carries the product name in a directory
    of its own, and that is a value rather than a program, so an argument inside
    the project is not this app naming itself.
    """

    inside = project_root.resolve()
    for index, argument in enumerate(arguments):
        if argument == "-m":
            if index + 1 < len(arguments) and arguments[index + 1] in PROGRAM_NAMES:
                return True
            continue
        if argument in PROGRAM_NAMES:
            return True
        if Path(argument).name not in PROGRAM_NAMES:
            continue
        try:
            resolved = Path(argument).resolve()
        except OSError:
            continue
        if resolved == inside or inside in resolved.parents:
            continue
        return True
    return False


def project_argument(arguments: Sequence[str]) -> str | None:
    """The `--project-root` value this process was given, in either spelling."""

    for index, argument in enumerate(arguments):
        if argument == "--project-root" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--project-root="):
            return argument.split("=", 1)[1]
    return None


def judge(pid: int, project_root: Path, proc_root: Path = Path("/proc")) -> Verdict:
    """Whether this pid is an app of this project, and the reason either way."""

    here = os.getpid()
    if pid <= 0:
        return Verdict(False, f"pid {pid} names no process")
    if pid == here:
        return Verdict(False, f"pid {pid} is this process")
    ancestors = set(lineage(here, proc_root))
    if pid in ancestors:
        return Verdict(False, f"pid {pid} is a process this one descends from")
    arguments = arguments_of(pid, proc_root)
    if arguments is None or not arguments:
        return Verdict(False, f"pid {pid} has no readable command line")
    command = " ".join(arguments)
    if not invokes_this_app(arguments, project_root):
        return Verdict(False, f"pid {pid} does not run this app: {command}")
    if project_argument(arguments) != str(project_root):
        return Verdict(
            False,
            f"pid {pid} runs this app for another project: {command}",
        )
    return Verdict(True, f"pid {pid} is this project's app: {command}", command)


def _pidfd_for(pid: int) -> int | None:
    """A handle naming the process holding this pid now, or None."""

    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        return None
    try:
        return int(opener(pid, 0))
    except (OSError, ValueError):
        return None


def ask_to_stop(
    pid: int,
    project_root: Path,
    proc_root: Path = Path("/proc"),
    number: int = signal.SIGTERM,
) -> Outcome:
    """Ask one process to stop, but only after proving it is this project's app.

    The proof is taken, the handle is opened, and the proof is taken again: a pid
    reused between the two readings would change the command line, and a handle
    opened before the reuse still names the process that is gone. A kernel without
    `pidfd_send_signal` leaves the target running, because signalling a bare number
    is the case that cannot be made safe here.
    """

    first = judge(pid, project_root, proc_root)
    if not first.owned:
        return Outcome(False, first.reason)
    handle = _pidfd_for(pid)
    if handle is None:
        return Outcome(
            False,
            f"pid {pid} is this project's app, but this kernel does not offer a "
            "handle that survives a reused pid, so it was left running.",
        )
    try:
        second = judge(pid, project_root, proc_root)
        if not second.owned or second.command != first.command:
            return Outcome(
                False,
                f"pid {pid} changed while it was being asked to stop, so it was "
                "left running.",
            )
        sender = getattr(signal, "pidfd_send_signal", None)
        if sender is None:
            return Outcome(
                False,
                f"pid {pid} is this project's app, but this runtime cannot signal "
                "through a handle, so it was left running.",
            )
        try:
            sender(handle, number)
        except OSError as exc:
            return Outcome(
                False,
                f"pid {pid} could not be asked to stop: {exc.strerror or exc}",
            )
    finally:
        os.close(handle)
    return Outcome(True, f"pid {pid} was asked to stop: {first.command}")
