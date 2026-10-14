"""The projects one installation knows about.

One command serves many projects. Each project is a directory holding
`.research-rag` beside whatever the user keeps there, each is served by its own
app process on its own port, and nothing about one project's corpus reaches
another. What that arrangement lacks is the machine's memory: without a record a
caller can only reach a project by repeating its absolute path, and nothing can
report every project the installation serves at once.

The record is a pointer: it holds each project's stable `project_id`, its
recorded name, and its root, and every byte of state stays in the project it
belongs to. It sits beside the account settings this app already reads, so one
user has one directory for both. It is written atomically, and every write that
reads the record first holds a lock beside it, because a replace without one drops
the entry another command registered in the same moment.

`AGENTS.md` states the rules about the recorded name and the pointer file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import BaseFileLock, FileLock
from filelock import Timeout as FileLockTimeout

from ..storage.durable_io import atomic_file
from .policy import ResearchError, _utc_now
from .settings import USER_CONFIG_DIRECTORY
from .state_files import PORTABLE_DIRECTORY

# The account-wide record, beside the settings that already live there. The name
# is this app's own: the MCP server shares the directory but has no registry to
# read.
REGISTRY_FILE = "projects.json"

# The lock a write holds. One command registers a project while another removes
# one, and a replace is atomic while a read-then-replace is not: without this, the
# later writer publishes a record that never saw the earlier writer's entry.
REGISTRY_LOCK_FILE = "projects.lock"
REGISTRY_LOCK_TIMEOUT_SECONDS = 20

SCHEMA_VERSION = 1


def registry_path() -> Path:
    from platformdirs import user_config_path

    return user_config_path(USER_CONFIG_DIRECTORY) / REGISTRY_FILE


def registry_lock_path() -> Path:
    return registry_path().with_name(REGISTRY_LOCK_FILE)


@contextmanager
def _held_lock() -> Iterator[BaseFileLock]:
    """Hold the record's lock for the length of one read-modify-write."""

    path = registry_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = FileLock(path, timeout=REGISTRY_LOCK_TIMEOUT_SECONDS)
    try:
        handle.acquire()
    except FileLockTimeout as exc:
        raise ResearchError(
            "Another research command is writing the project record. Nothing was "
            "written. Run it again once it finishes."
        ) from exc
    try:
        yield handle
    finally:
        handle.release()


@dataclass(frozen=True, slots=True)
class RegisteredProject:
    project_id: str
    project_name: str
    project_root: Path
    registered_at: str

    def as_record(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_root": str(self.project_root),
            "registered_at": self.registered_at,
        }

    @property
    def descriptor_path(self) -> Path:
        return self.project_root / PORTABLE_DIRECTORY / "project.json"

    def initialised(self) -> bool:
        """Whether the directory this record points at holds a project.

        A record is a pointer and a project can be moved or deleted while the
        pointer stays, so a caller that has to serve the project asks this
        first. The descriptor's presence is the whole test: no project state is
        read to decide it.
        """

        return self.descriptor_path.is_file()


def _read(path: Path) -> dict[str, Any]:
    """Return the stored record, or an empty one when there is nothing usable."""

    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "projects": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchError(
            f"The project record at {path} could not be read ({exc}). Delete it "
            "to rebuild it, or fix the file."
        ) from exc
    if not isinstance(document, dict):
        raise ResearchError(f"The project record at {path} is not an object.")
    return document


def _write(path: Path, document: dict[str, Any]) -> None:
    """Replace the record atomically, so a reader never sees half of one."""

    with atomic_file(path, private=True) as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load() -> list[RegisteredProject]:
    """Return every registered project, newest record last and ordered by name.

    A record whose fields are not strings is not a pointer this module can follow,
    so it is skipped rather than guessed at. One unreadable entry must not hide
    every project a user still has.
    """

    document = _read(registry_path())
    projects: list[RegisteredProject] = []
    for entry in document.get("projects") or []:
        if not isinstance(entry, dict):
            continue
        project_id = entry.get("project_id")
        name = entry.get("project_name")
        root = entry.get("project_root")
        if not all(
            isinstance(value, str) and value for value in (project_id, name, root)
        ):
            continue
        projects.append(
            RegisteredProject(
                project_id=project_id,
                project_name=name,
                project_root=Path(root),
                registered_at=str(entry.get("registered_at") or ""),
            )
        )
    projects.sort(key=lambda item: (item.project_name.lower(), item.project_root))
    return projects


def register(
    project_id: str,
    project_name: str,
    project_root: str | Path,
) -> RegisteredProject:
    """Record one project, replacing any earlier record of the same root or id.

    Re-registering a project refreshes its name and root rather than adding a
    second entry, so `init` is safe to run again and a project that moved is
    followed to where it went.
    """

    root = str(Path(project_root).expanduser().resolve())
    entry = RegisteredProject(
        project_id=project_id,
        project_name=project_name,
        project_root=Path(root),
        registered_at=_utc_now(),
    )
    with _held_lock():
        path = registry_path()
        document = _read(path)
        kept = [
            item
            for item in document.get("projects") or []
            if isinstance(item, dict)
            and item.get("project_root") != root
            and item.get("project_id") != project_id
        ]
        document["schema_version"] = SCHEMA_VERSION
        document["projects"] = [*kept, entry.as_record()]
        _write(path, document)
    return entry


def forget(project_id: str) -> bool:
    root = registry_path()
    with _held_lock():
        document = _read(root)
        projects = document.get("projects") or []
        kept = [
            item
            for item in projects
            if not (isinstance(item, dict) and item.get("project_id") == project_id)
        ]
        if len(kept) == len(projects):
            return False
        document["schema_version"] = SCHEMA_VERSION
        document["projects"] = kept
        _write(root, document)
        return True


def matches(query: str) -> list[RegisteredProject]:
    """Return the registered projects a name or id resolves to.

    A selector is matched against a project's id and name, case insensitively, and
    exactly as the caller typed it. A substring is a guess about a project's
    name, and the record has the name, so this is not the place to be generous.
    """

    term = query.strip().casefold()
    if not term:
        return []
    return [
        project
        for project in load()
        if term == project.project_id.casefold()
        or term == project.project_name.casefold()
    ]


def named(query: str) -> list[RegisteredProject]:
    """Return the registered projects a recorded *name* resolves to.

    Only the name matches, never the id: an agent's client entry carries a
    project's name, and resolving it through an id would let a name and another
    project's id stand for the same directory.
    """

    term = query.strip().casefold()
    if not term:
        return []
    return [project for project in load() if term == project.project_name.casefold()]


def resolve(query: str) -> RegisteredProject:
    found = matches(query)
    if not found:
        raise ResearchError(
            f"No registered project is called {query!r}. Run "
            f"'research-rag projects' for the list, or pass --project-root."
        )
    if len(found) > 1:
        names = ", ".join(
            f"{project.project_name} ({project.project_root})" for project in found
        )
        raise ResearchError(
            f"{query!r} names more than one registered project: {names}. "
            "Use the project id."
        )
    return found[0]


def registered_at_label(project: RegisteredProject) -> str:
    stamp = project.registered_at
    if not stamp:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


__all__ = [
    "REGISTRY_FILE",
    "REGISTRY_LOCK_FILE",
    "REGISTRY_LOCK_TIMEOUT_SECONDS",
    "SCHEMA_VERSION",
    "RegisteredProject",
    "forget",
    "load",
    "matches",
    "named",
    "register",
    "registered_at_label",
    "registry_lock_path",
    "registry_path",
    "resolve",
]
