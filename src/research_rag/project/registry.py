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
user has one directory for both. It is written atomically because two commands
may register two projects at the same moment.

`AGENTS.md` states the rules about the recorded name and the pointer file.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import USER_CONFIG_DIRECTORY
from .state_files import PORTABLE_DIRECTORY
from .support import ResearchError, _utc_now

# The account-wide record, beside the settings that already live there. The name
# is this app's own: the MCP server shares the directory but has no registry to
# read.
REGISTRY_FILE = "projects.json"
SCHEMA_VERSION = 1


def registry_path() -> Path:
    from platformdirs import user_config_path

    return user_config_path(USER_CONFIG_DIRECTORY) / REGISTRY_FILE


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

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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

    path = registry_path()
    document = _read(path)
    root = str(Path(project_root).expanduser().resolve())
    entry = RegisteredProject(
        project_id=project_id,
        project_name=project_name,
        project_root=Path(root),
        registered_at=_utc_now(),
    )
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
    path = registry_path()
    document = _read(path)
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
    _write(path, document)
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


def _attached_to(config: Any) -> str | None:
    from .state_files import TTY_FILE

    try:
        terminal = (config.state_root / TTY_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return terminal or None


def detached_from_terminal(config: Any, running: bool) -> bool | None:
    """Whether the serving process has no controlling terminal.

    ``True`` is the condition this app holds itself against, so it is stated
    rather than left to be inferred from a missing terminal file. ``None`` is the
    answer when the platform cannot be asked, which is not the same as "attached"
    and is never reported as such.
    """

    if not running:
        return False
    from ..runtime.app import has_terminal, recorded_pid

    pid = recorded_pid(config)
    if pid is None:
        return None
    terminal = has_terminal(pid)
    return None if terminal is None else not terminal


def project_app_state(project_root: Path) -> dict[str, Any]:
    """Whether an app is serving one project, and what that app reports.

    The probe reaches the project's own control API, so the answer is the one
    `research-rag clients` gives rather than a second reading of the same state,
    and a project with no app up is reported as such rather than started: a
    listing is a question about what exists.

    The imports are local because the record is a pointer file and the modules
    that answer about a running app pull in the web stack. Nothing here is cached:
    every call reads the record and asks again.
    """

    from ..core.tool_views import lean_status
    from ..runtime.app import running_url
    from ..runtime.control import Control, ControlError
    from .config import resolve_config

    config = resolve_config(project_root)
    url = running_url(config)
    state: dict[str, Any] = {
        "app": {
            "running": url is not None,
            "url": url,
            "port": _port_of(url),
            # None when the serving process was started detached. A reader
            # deciding whether Ctrl-C in this terminal would stop that app needs
            # the difference.
            "attached_to": _attached_to(config),
            "detached": detached_from_terminal(config, url is not None),
        },
        "attached_clients": 0,
    }
    if url is None:
        return state
    try:
        with Control(url, timeout=10.0) as handle:
            verdict = lean_status(handle.status())
            clients = handle.clients()
    except ControlError as exc:
        state["error"] = str(exc)
        return state
    state["attached_clients"] = len(clients)
    for key in ("ready", "stale", "requires"):
        if key in verdict:
            state[key] = verdict[key]
    return state


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    _, _, port = url.rpartition(":")
    return int(port) if port.isdigit() else None


def account_projects() -> dict[str, Any]:
    """Every registered project, and whether an app is serving it.

    This is the one answer to that question: the command line prints it and the
    workspace selects from it, so a project the terminal calls down is a project
    the browser calls down too.
    """

    from .config import ConfigurationError

    entries: list[dict[str, Any]] = []
    for project in load():
        entry: dict[str, Any] = {
            "project_name": project.project_name,
            "project_id": project.project_id,
            "project_root": str(project.project_root),
            "registered_at": registered_at_label(project),
            "root_exists": (project.project_root / PORTABLE_DIRECTORY).is_dir(),
            "app": {"running": False, "url": None, "port": None},
            "attached_clients": 0,
        }
        if entry["root_exists"]:
            try:
                entry.update(project_app_state(project.project_root))
            except (ConfigurationError, ResearchError) as exc:
                entry["error"] = str(exc)
        entries.append(entry)
    answer: dict[str, Any] = {
        "registry_path": str(registry_path()),
        "project_count": len(entries),
        "projects": entries,
    }
    if not entries:
        answer["message"] = (
            "No project is registered yet. Run 'research-rag --project-root "
            "<path> init' once per project, and this install can address each one "
            "by name."
        )
    return answer


__all__ = [
    "REGISTRY_FILE",
    "SCHEMA_VERSION",
    "RegisteredProject",
    "account_projects",
    "detached_from_terminal",
    "forget",
    "load",
    "matches",
    "named",
    "project_app_state",
    "register",
    "registered_at_label",
    "registry_path",
    "resolve",
]
