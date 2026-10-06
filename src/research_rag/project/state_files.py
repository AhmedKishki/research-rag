"""The on-disk names every copy of this app agrees on, and the two live-process
questions more than one caller asks of them.

A rename strands a running installation and every installed copy along with it.
The pid file a moving reader wrote is a file nothing reads, and a name two
products share lets each stop the other's process. That is why these names are
declared once here, and why this module holds nothing else: no reading of a
project's contents, no writing, and no opinion about a name beyond the path it
composes.

`process_alive` and `recorded_pid` are the answers to the two questions every
caller was writing itself: whether a pid is still running, and which pid a project
recorded. A pid file survives the terminal that wrote it, so the process behind it
is checked rather than trusted.

`ProjectState` and `resident_build` moved here from `update.py`, which asked both
questions about a project it had not resolved and must not resolve. The staging
checkpoint is ingestion's own format and is read here from the files themselves,
without importing `ingestion`, because the answer is needed before a service can
be built to ask it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.records import read_json
from .config import recorded_runtime_root

# Where every project-owned artifact lives beneath its project root.
PORTABLE_DIRECTORY = ".research-rag"
# The derived state inside it, rebuildable from the sources and excluded from the
# digest an update compares.
RUNTIME_DIRECTORY = "runtime"
# What a serving app records so a second terminal can find it. Three products
# sharing one of these names could stop each other's process.
PID_FILE = "research-rag-ui.pid"
PORT_FILE = "research-rag-ui.port"
TTY_FILE = "research-rag-ui.tty"
# The lock a project serves and builds under.
LOCK_FILE = "project.lock"
# What searches returned, counted on this machine. No query text is kept.
SEARCH_STATS_FILE = "search-stats.sqlite3"


def process_alive(pid: int) -> bool:
    """Whether a process still exists, without signalling it.

    A `PermissionError` means alive: a process this app cannot signal has not
    stopped.

    A pid that is not a positive number names a process group, and `os.kill`
    answers for the caller's own group given one, so it names no process here.
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_root(state_root: Path, portable_root: Path | None) -> Path:
    """Where this project's derived state actually lives.

    A relocated runtime root is recorded by the project itself when its
    configuration is resolved, so the answer is read from that record rather than
    from a second rule about where a project keeps its derived state.
    """

    if portable_root is None:
        return state_root
    relocated = recorded_runtime_root(portable_root)
    return relocated if relocated is not None else state_root


def recorded_pid(state_root: Path, portable_root: Path | None = None) -> int | None:
    """The pid a project's app recorded, when that process is still alive.

    `portable_root` is the project's own directory, for a caller that holds it
    rather than the resolved state root; the record there is what says where the
    state went.

    The answer is only that the number is in use. `runtime/ownership.py` owns
    whether the process behind it is this project's app, which is what a signal
    needs.
    """

    try:
        pid = int(
            (_state_root(state_root, portable_root) / PID_FILE)
            .read_text(encoding="utf-8")
            .strip()
        )
    except (OSError, ValueError):
        return None
    return pid if process_alive(pid) else None


@dataclass(frozen=True, slots=True)
class ProjectState:
    """One project as a path, before any of it is resolved or served."""

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
        """Where this project keeps its pid, its lock, and its staging."""

        return _state_root(self.default_state_root, self.portable_root)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "project_name": self.project_name,
        }


def resident_build(state_root: Path) -> tuple[str | None, str | None]:
    """The phase and build id a project's newest checkpoint records, if any.

    The same two fields a service names when it refuses a second build. This
    reads them from the files themselves because it must answer before any
    project is resolved, and it will not build a service to ask.
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


__all__ = [
    "LOCK_FILE",
    "PID_FILE",
    "PORTABLE_DIRECTORY",
    "PORT_FILE",
    "RUNTIME_DIRECTORY",
    "TTY_FILE",
    "ProjectState",
    "process_alive",
    "recorded_pid",
    "resident_build",
]
