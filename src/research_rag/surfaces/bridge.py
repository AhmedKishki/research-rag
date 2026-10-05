"""`research-rag mcp`: a stdio front end to the app that serves a named project.

An MCP client that speaks only stdio cannot open a socket, so this is the command
that connects one: it makes sure the app is up, then proxies stdio to the app's
agent endpoint on the app's own port. The proxy adds nothing — the tools, the
answer projection, the project, and the UltraRAG gateway are the app's, so a
stdio client and a browser cannot see two different states.

The bridge names a project and never a directory. A client configuration is
written once and copied between machines, a phone, and a repository, and an
absolute path in it is true on exactly one of them; a project's recorded name is
resolvable on every machine where that project was initialised. The resolution
belongs to this installation's own record, so the bridge asks the record for the
directory and refuses to accept one itself.

A project that this machine has not initialised is answered, not refused: the
connection is established, `status` reports that the name resolves to nothing
here, and the answer carries the command that creates the project. The other seven
operations are absent, because there is no corpus behind them. An agent's entry
therefore needs no editing after that command runs.

The bridge names itself and says where it is running, so a disconnect is legible
and a list of clients is a list of agents rather than a list of session ids: the
app lists clients by name, and dropping one ends its session, which ends the pipe
and therefore the client.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server import create_proxy

from ..project import registry
from ..project.config import (
    ConfigurationError,
    ResearchConfig,
    configured_source_directory,
    initialise_command,
    resolve_config,
)
from ..project.policy import ResearchError
from ..runtime.app import (
    CLIENT_IDENTITY_HEADER,
    CLIENT_NAME_ENV,
    CLIENT_NAME_HEADER,
)
from ..runtime.control import connect, is_serving, not_running_reason
from ..runtime.version import APP_VERSION
from .mcp import AGENT_BRIDGE_NAME, create_blocked_mcp

DEFAULT_CLIENT_NAME = "stdio-bridge"


def client_name() -> str:
    """The name this bridge reports itself under, or a stated default."""

    return os.environ.get(CLIENT_NAME_ENV) or DEFAULT_CLIENT_NAME


def _host_program(environ: Mapping[str, str]) -> str:
    """The program this bridge was started by, as the environment names it.

    A client that starts a bridge is an editor, a terminal, or a shell over ssh,
    and each declares itself in the environment it passes down. A shell says
    nothing beyond its terminal, so the terminal type stands in for it.
    """

    if (
        environ.get("VSCODE_PID")
        or environ.get("VSCODE_IPC_HOOK")
        or environ.get("TERM_PROGRAM", "").casefold().startswith("vscode")
    ):
        return "Visual Studio Code"
    program = environ.get("TERM_PROGRAM", "").strip()
    if program:
        return program
    term = environ.get("TERM", "").strip()
    return "" if term in {"", "dumb"} else f"a {term} terminal"


def client_identity(
    project_name: str, *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """What this bridge tells the app about the client it runs for.

    A client entry names a command and a project, so the app would otherwise
    show the same row for every agent on the machine. These are the facts the
    bridge inherited rather than chose: the program that started it, whether it
    came over ssh or inside a terminal multiplexer, the directory it was
    started in, and the project its client asked for.
    """

    environment = os.environ if environ is None else environ
    try:
        directory = os.getcwd()
    except OSError:
        directory = ""
    program = _host_program(environment)
    return {
        "agent": client_name(),
        "project": project_name,
        "pid": os.getpid(),
        "cwd": directory,
        "host": {
            "program": program,
            "term": environment.get("TERM", "").strip() if not program else "",
            "ssh": bool(
                environment.get("SSH_CONNECTION") or environment.get("SSH_CLIENT")
            ),
            "tmux": bool(environment.get("TMUX") or environment.get("TMUX_PANE")),
        },
    }


def build_proxy(url: str, *, name: str, identity: Mapping[str, Any]) -> Any:
    """Return a stdio server that forwards everything to the app at `url`.

    The tools are proxied, not re-declared: a second copy of the seven operations
    would be a second place for them to be wrong, and would give the agent a
    different answer than the workspace for the same question. The transport is
    built here rather than from a URL so the bridge can name itself and say where
    it is running, which is what makes it identifiable in the app's client list.
    """

    return create_proxy(
        StreamableHttpTransport(
            url,
            headers={
                CLIENT_NAME_HEADER: name,
                CLIENT_IDENTITY_HEADER: json.dumps(identity, separators=(",", ":")),
            },
        ),
        name=AGENT_BRIDGE_NAME,
        version=APP_VERSION,
    )


def uninitialised_reason(project_name: str) -> str | None:
    """Why this machine cannot serve that project name, or None when it can.

    Nothing is created to find out: a record this installation never wrote, and
    a record whose directory no longer holds a project, are the same condition
    from the entry's point of view, and both are fixed by initialising the
    project. A name two records share is a different fault, and no amount of
    initialising settles it, so it is raised.
    """

    found = registry.named(project_name)
    if not found:
        return (
            f"No project is initialised under the name {project_name!r} on this "
            f"machine. {initialise_command(project_name)} creates one here, and "
            "this entry serves it unchanged."
        )
    if len(found) > 1:
        roots = ", ".join(str(entry.project_root) for entry in found)
        raise ResearchError(
            f"{project_name!r} is recorded for more than one project: {roots}. A "
            "client entry names a project, and it cannot name two."
        )
    entry = found[0]
    if not entry.initialised():
        return (
            f"The project {project_name!r} is recorded at {entry.project_root}, "
            f"which holds no project. {initialise_command(project_name, entry.project_root)} "
            f"initialises one there, and this entry serves it unchanged."
        )
    return None


def resolve_project(
    project_name: str,
    *,
    settings: Mapping[str, Any] | None = None,
) -> tuple[ResearchConfig | None, str | None]:
    """The app's configuration for a named project, or the refusal that replaces it.

    A project this installation can serve resolves to its own configuration. One
    it cannot resolves to `None` and the sentence an agent is told instead, so
    the caller never has to decide whether a missing project is an error.
    """

    reason = uninitialised_reason(project_name)
    if reason is not None:
        return None, reason
    entry = registry.named(project_name)[0]
    return (
        resolve_config(
            entry.project_root,
            source_directory=configured_source_directory(entry.project_root),
            **dict(settings or {}),
        ),
        None,
    )


def serve_blocked(
    project_name: str, reason: str, *, config: ResearchConfig | None
) -> None:
    """Answer an agent whose project this machine cannot serve, and start nothing."""

    create_blocked_mcp(project_name, reason, config=config).run(
        transport="stdio", show_banner=False
    )


def run(
    project_name: str,
    *,
    settings: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> None:
    """Connect stdio to the app serving this project name, and stay up as long as it does.

    Nothing is started here. An app runs in a terminal and ends when that terminal
    closes, so a client that arrives while the project is not being served is told
    which command to run rather than quietly given a server of its own to leave
    behind.
    """

    config, reason = resolve_project(project_name, settings=settings)
    if config is None:
        serve_blocked(project_name, reason or "", config=None)
        return
    if not is_serving(config):
        serve_blocked(project_name, not_running_reason(), config=config)
        return
    with connect(config) as control:
        url = f"{control.base_url}/mcp"
    name = name or client_name()
    build_proxy(url, name=name, identity=client_identity(project_name)).run(
        transport="stdio", show_banner=False
    )


def main(
    project_name: str,
    *,
    settings: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> None:
    try:
        run(project_name, settings=settings, name=name)
    except (ConfigurationError, ResearchError) as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:  # pragma: no cover - a client closing the pipe
        pass
