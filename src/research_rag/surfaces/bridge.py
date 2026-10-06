"""`research-rag mcp`: a stdio front end to the app that serves a named project.

An MCP client that speaks only stdio cannot open a socket, so this is the command
that connects one: it makes sure the app is up, then proxies stdio to the app's
agent endpoint on the app's own port. The proxy adds nothing — the tools, the
answer projection, the project, and the UltraRAG gateway are the app's, so a
stdio client and a browser cannot see two different states.

The bridge decides nothing at launch. Whether the app is up, and whether the
project exists here, are asked on every call, so an agent that connected before
the app started, or while it restarted, finds the same tools working as soon as
the app answers. While it does not, each call returns the command that starts it.

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
therefore needs no editing after that command runs, though a client that listed
its tools before may need to list them again to see the other seven.

The bridge names itself and says where it is running, so a disconnect is legible
and a list of clients is a list of agents rather than a list of session ids: the
app lists clients by name, and dropping one ends its session, which ends the pipe
and therefore the client.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any

from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import FastMCPError, ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.server.providers.proxy import FastMCPProxy, ProxyClient
from mcp.shared.exceptions import McpError

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
from .mcp import AGENT_BRIDGE_NAME, create_blocked_mcp, create_unserved_mcp

DEFAULT_CLIENT_NAME = "stdio-bridge"
# Asking whether the app answers is a local HTTP round trip, so a few seconds is
# generous, and an app that does not answer in them is not serving.
HEALTH_TIMEOUT_SECONDS = 3.0
# How long an upstream session may take to open.
INIT_TIMEOUT_SECONDS = 15.0
# A call waits at most this much beyond the longest build the app allows one call.
CALL_TIMEOUT_MARGIN_SECONDS = 600.0
# The tools that change nothing, so repeating one after a dropped connection is
# safe. A write is never repeated for the caller: it may have been applied.
READ_ONLY_TOOLS = frozenset({"status", "search", "find_source", "get_passage"})


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


def _headers(name: str, identity: Mapping[str, Any]) -> dict[str, str]:
    return {
        CLIENT_NAME_HEADER: name,
        CLIENT_IDENTITY_HEADER: json.dumps(identity, separators=(",", ":")),
    }


class Backends:
    """Where each call goes: the app when it answers, a stand-in when it does not.

    The decision is made per call and never cached, so the bridge cannot be left
    in a state its first call chose. The tools are proxied, not re-declared: a
    second copy of the operations would be a second place for them to be wrong,
    and would give the agent a different answer than the workspace for the same
    question. The stand-in is built from the same declarations as the app's
    surface, so the tool list does not change when the app comes and goes.
    """

    def __init__(
        self,
        project_name: str,
        *,
        name: str,
        identity: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.project_name = project_name
        self.name = name
        self.identity = identity
        self.settings = settings
        self._config: ResearchConfig | None = None
        self._standin: FastMCP[Any] | None = None

    def _resolve(self) -> tuple[ResearchConfig | None, str | None]:
        """The project's configuration, read again until the project exists here.

        Resolving a configuration writes the project's portable state, so a
        project that resolved is remembered; one that did not is looked for again
        on every call, because `init` may have been run since.
        """

        if self._config is not None:
            return self._config, None
        config, reason = resolve_project(self.project_name, settings=self.settings)
        self._config = config
        return config, reason

    def _serving_url(self, config: ResearchConfig) -> str | None:
        if not is_serving(config, timeout=HEALTH_TIMEOUT_SECONDS):
            return None
        control = connect(config)
        if control is None:
            return None
        with control:
            return f"{control.base_url}/mcp"

    async def serving_url(self) -> str | None:
        """The app's agent endpoint, or None while it is not answering."""

        config, _reason = await asyncio.to_thread(self._resolve)
        if config is None:
            return None
        return await asyncio.to_thread(self._serving_url, config)

    async def client(self) -> ProxyClient:
        """The client this one call goes through."""

        config, reason = await asyncio.to_thread(self._resolve)
        if config is None:
            return ProxyClient(
                create_blocked_mcp(self.project_name, reason or "", config=None)
            )
        url = await asyncio.to_thread(self._serving_url, config)
        if url is None:
            if self._standin is None:
                self._standin = create_unserved_mcp(config, not_running_reason())
            return ProxyClient(self._standin)
        return self._upstream(url, config)

    def _upstream(self, url: str, config: ResearchConfig) -> ProxyClient:
        """A session on the app's agent endpoint, named and located for its client list."""

        return ProxyClient(
            StreamableHttpTransport(url, headers=_headers(self.name, self.identity)),
            timeout=config.settings.work_budget_seconds + CALL_TIMEOUT_MARGIN_SECONDS,
            init_timeout=INIT_TIMEOUT_SECONDS,
        )

    async def explain(self, error: Exception) -> str:
        """What a failed call means, in words that name the command that fixes it."""

        config, reason = await asyncio.to_thread(self._resolve)
        if config is None:
            return reason or "This project is not initialised on this machine."
        if await self.serving_url() is None:
            from ..core.blocked_answers import not_served_status

            standin = not_served_status(config, not_running_reason())
            remedy = standin["blocked_by"][0]["remedy"]
            return (
                f"The app stopped answering during this call. Run `{remedy}` in a "
                "terminal, then call this tool again; this entry needs no change."
            )
        return (
            "The connection to the app dropped during this call "
            f"({type(error).__name__}), and the app is still up. Call `status` to "
            "see whether the change was applied before repeating a write."
        )


def _transport_failure(error: Exception) -> Exception | None:
    """The connection failure behind `error`, or None when the backend itself refused.

    FastMCP wraps whatever a call raised in a `ToolError` that carries the
    original as its cause. The app's own refusals reach the bridge as a plain
    `ToolError` with no cause, or as a protocol error, and are the answer, not a
    fault: only a failure of the connection is explained here.
    """

    cause = error.__cause__ if isinstance(error, ToolError) else error
    if cause is None or isinstance(cause, FastMCPError | McpError):
        return None
    return cause if isinstance(cause, Exception) else None


class _Resilience(Middleware):
    """Turn a dropped connection into an answer, and repeat a read once.

    An agent told only that a connection closed has nothing to act on, so the
    failure is replaced by whether the app is still there and the command that
    starts it when it is not.
    """

    def __init__(self, backends: Backends) -> None:
        self._backends = backends

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        try:
            return await call_next(context)
        except Exception as error:
            failure = _transport_failure(error)
            if failure is None:
                raise
        if (
            context.message.name in READ_ONLY_TOOLS
            and await self._backends.serving_url() is not None
        ):
            try:
                return await call_next(context)
            except Exception as error:
                again = _transport_failure(error)
                if again is None:
                    raise
                failure = again
        raise ToolError(await self._backends.explain(failure)) from failure


def build_bridge(
    project_name: str,
    *,
    name: str,
    identity: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
) -> Any:
    """The stdio server for one project name, which never decides at launch."""

    backends = Backends(project_name, name=name, identity=identity, settings=settings)
    server = FastMCPProxy(
        client_factory=backends.client,
        name=AGENT_BRIDGE_NAME,
        version=APP_VERSION,
    )
    server.add_middleware(_Resilience(backends))
    return server


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


def run(
    project_name: str,
    *,
    settings: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> None:
    """Connect stdio to the app serving this project name, and stay up for the client.

    Nothing is started here. An app runs in a terminal and ends when that terminal
    closes, so a call that arrives while the project is not being served is told
    which command to run rather than quietly given a server of its own to leave
    behind. The bridge itself is up as long as its client is, whatever the app
    does meanwhile.
    """

    build_bridge(
        project_name,
        name=name or client_name(),
        identity=client_identity(project_name),
        settings=settings,
    ).run(transport="stdio", show_banner=False)


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
