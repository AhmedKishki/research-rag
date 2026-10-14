"""Every registered project, and whether an app is serving it.

The registry names the projects. This module asks each project's own control API
whether its app is up, so a listing never starts one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.tool_views import lean_status
from ..project.config import ConfigurationError, resolve_config
from ..project.policy import ResearchError
from ..project.registry import load, registered_at_label, registry_path
from ..project.state_files import PORTABLE_DIRECTORY, TTY_FILE
from .app import has_terminal, recorded_pid, running_url
from .control import Control, ControlError


def _attached_to(config: Any) -> str | None:
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

    Nothing here is cached: every call reads the record and asks again.
    """

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
            verdict = lean_status(handle.operation("status", {}))
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
