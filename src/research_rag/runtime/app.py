"""The running app: one project, one port, one service, three front ends.

One process owns the project lock, opens the UltraRAG gateway once, and serves the
workspace and the agent surface on one loopback port, so a passage an agent
retrieves and one the workspace renders are the same object.

**MCP clients attach and detach.** A streamable-HTTP session is recorded on its
first request, and a forced detach refuses the rest of it. The bridge in
`bridge.py` names itself, so an agent is identifiable by more than a peer address.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..core.service import ResearchService
from ..project.config import ConfigurationError, ResearchConfig
from ..project.state_files import PID_FILE, PORT_FILE, TTY_FILE
from ..project.state_files import process_alive as alive
from ..project.state_files import recorded_pid as _recorded_pid
from ..retrieval.ultrarag import LazyGateway, VanillaUltraRAG

LOGGER = logging.getLogger(__name__)

UI_HOST = "127.0.0.1"
CONTROL_PREFIX = "/control"
# Matches uvicorn's own default accept backlog, so a claimed socket is in the
# state uvicorn would have created for itself.
_CLAIM_BACKLOG = 2048
# A session that has said nothing for this long is reported as attached no
# longer. It is kept rather than dropped, so a client that reconnects keeps one
# identity in the list and a reader can see that it was here.
CLIENT_IDLE_SECONDS = 90.0
# The header the stdio bridge sets so an agent is identifiable by name. A direct
# HTTP client that does not set it is identified by its peer address.
CLIENT_NAME_HEADER = "x-research-rag-client"


def _own_tty() -> str | None:
    try:
        return os.ttyname(0)
    except OSError:
        return None


def has_terminal(pid: int, proc_root: Path = Path("/proc")) -> bool | None:
    """Whether ``pid`` still has a controlling terminal, or None when unasked.

    An app is attached to the terminal that started it, so a serving process with
    no controlling terminal is serving outside the rule this app holds itself to.
    It is a fact about the live process rather than about ``TTY_FILE``, because a
    file is deleted when an app that recorded it stops, and a missing file would
    then read as a detached app that is not there.

    ``/proc/<pid>/stat`` is read rather than ``ps`` because this app runs where
    procfs is and no process tree is inspected: field 7 is the controlling
    terminal's device number, and 0 is the kernel's own answer for "none". The
    second field is in parentheses and may itself contain a space or one, so the
    fields are counted from after its final ``)`` rather than by splitting the
    whole line.
    """

    try:
        raw = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    try:
        fields = raw[raw.rindex(")") + 1 :].split()
    except ValueError:
        return None
    # fields[0] is the state, which is field 3 of the line, so the controlling
    # terminal's device number is field 7.
    if len(fields) < 5:
        return None
    try:
        return int(fields[4]) != 0
    except ValueError:
        return None


def _record_state(config: ResearchConfig, port: int) -> None:
    """Write the running state a second terminal reads to find this app.

    The app writes these for itself wherever it was started from, so `clients`,
    `stop`, and the workspace's project selector see the same thing either way.
    """

    state = config.state_root
    tty = _own_tty()
    with contextlib.suppress(OSError):
        state.mkdir(parents=True, exist_ok=True)
        (state / PORT_FILE).write_text(f"{port}\n", encoding="utf-8")
        (state / PID_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
        if tty is None:
            (state / TTY_FILE).unlink(missing_ok=True)
        else:
            (state / TTY_FILE).write_text(f"{tty}\n", encoding="utf-8")


def _forget_state(config: ResearchConfig) -> None:
    """Remove the running state this process wrote, and no other process's.

    An app that is stopping must not delete the record of an app that has since
    taken the project over, so the pid is compared before anything is removed.
    """

    state = config.state_root
    try:
        recorded = int((state / PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if recorded != os.getpid():
        return
    for name in (PORT_FILE, PID_FILE, TTY_FILE):
        with contextlib.suppress(OSError):
            (state / name).unlink(missing_ok=True)


class ClientError(Exception):
    """A control request that named something this app does not have."""


def _now() -> float:
    return time.monotonic()


@dataclass
class Client:
    """One MCP client this app has seen.

    `attached` is a property of recency and of an explicit drop, never a stored
    flag.
    """

    session_id: str
    name: str
    first_seen: float
    last_seen: float
    requests: int = 0
    streams: int = 1
    dropped: bool = False
    detached_reason: str | None = None
    # A sighting without a session id: the client opened a connection that has
    # not carried one yet. It is never counted as attached, because a connection
    # is not a client, and the MCP SDK opens the notification stream before the
    # session exists, so counting sightings would count one agent twice.
    pending: bool = False

    @property
    def attached(self) -> bool:
        if self.pending or self.dropped:
            return False
        return _now() - self.last_seen < CLIENT_IDLE_SECONDS

    @property
    def idle_seconds(self) -> float:
        return round(_now() - self.last_seen, 1)

    def report(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "attached": self.attached,
            "pending": self.pending,
            # One client opens as many connections as it has streams, and the
            # count is reported rather than hidden so a row is not read as one
            # socket.
            "streams": self.streams,
            "requests": self.requests,
            "idle_seconds": self.idle_seconds,
            "detached_reason": self.detached_reason,
        }


class ClientRegistry:
    """The clients attached to this app, and the authority to drop one.

    Identity is the MCP session id once a request carries one. The initialize request
    has no id yet, so it counts against the peer address, which keeps a client that
    never initializes visible.
    """

    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        # A client's initialize request carries its name but no session id, so it
        # is recorded against the peer address and the entry is carried over when
        # the id appears. Without that, one client reads as two.
        self._awaiting: dict[str, str] = {}

    def _touch(self, session_id: str, name: str, *, pending: bool = False) -> Client:
        existing = self._clients.get(session_id)
        if existing is None:
            existing = Client(
                session_id=session_id,
                name=name or session_id,
                first_seen=_now(),
                last_seen=_now(),
                pending=pending,
            )
            self._clients[session_id] = existing
        existing.last_seen = _now()
        existing.requests += 1
        if name:
            existing.name = name
        return existing

    def observe(self, request: Any) -> str | None:
        session_id = (request.headers.get("mcp-session-id") or "").strip()
        name = (request.headers.get(CLIENT_NAME_HEADER) or "").strip()
        peer = f"{request.client.host}:{request.client.port}" if request.client else "?"
        if not session_id:
            self._awaiting[peer] = peer
            return self._touch(peer, name, pending=True).session_id
        client = self._touch(session_id, name)
        if client.streams == 1:
            # A client opens one connection per stream, and the streams arrive
            # before the session id exists, so they are folded onto the session by
            # the name the client declared. Two clients that declare the same name
            # are one client as far as this app can tell, which is why the row
            # reports its stream count rather than pretending to be one socket.
            folded = [
                key
                for key, sighting in self._clients.items()
                if sighting.pending
                and sighting is not client
                and (sighting.name == client.name or not name)
            ]
            for key in folded:
                sighting = self._clients.pop(key)
                self._awaiting.pop(sighting.session_id, None)
                client.streams += 1
                client.requests += sighting.requests
        return client.session_id

    def get(self, session_id: str) -> Client | None:
        return self._clients.get(session_id)

    def disconnect(self, session_id: str, reason: str) -> Client:
        client = self._clients.get(session_id)
        if client is None:
            raise ClientError(f"No client is attached with session {session_id}")
        if client.pending:
            raise ClientError(
                f"{session_id} is a connection that carries no session yet, so "
                "there is nothing to disconnect; wait for the client to "
                "initialize and ask again."
            )
        if not client.attached:
            raise ClientError(f"Client {client.name} is already detached")
        client.dropped = True
        client.detached_reason = reason
        # The MCP SDK opens a notification stream before the session exists, so
        # that stream carries no session id and cannot be matched by one. It is
        # matched by the name the client gave itself, and dropping it makes the
        # client end rather than sit on an open stream with a session the app has
        # already refused. A client that named nothing cannot have its stream
        # identified, and only its session is refused.
        for other in self._clients.values():
            if other is not client and other.pending and other.name == client.name:
                other.dropped = True
                other.detached_reason = reason
        return client

    def report(self) -> list[dict[str, Any]]:
        """Every client this app has seen, attached or not.

        A client stays in the listing after it is dropped so the outcome of a
        disconnect is readable: its `attached` flag says it is gone and
        `detached_reason` says why. Recency alone decides `attached`, so a session
        whose process has ended reads as detached and is still listed.
        """

        return [client.report() for client in self._clients.values()]

    @property
    def attached(self) -> list[Client]:
        return [client for client in self._clients.values() if client.attached]


class ClientGate:
    """Refuse a dropped session before its requests reach the agent.

    The registry is the only authority, so a forced detach is enforced here rather
    than at the tools: a disconnected client sees its session fail rather than a
    silent no-op.
    """

    def __init__(self, app: Any, registry: ClientRegistry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.requests import Request

        session_id = self.registry.observe(Request(scope, receive=receive))
        client = self.registry.get(session_id)
        if client is not None and client.dropped:
            response = JSONResponse(
                {
                    "error": (
                        "This session was disconnected from the app: "
                        f"{client.detached_reason}"
                    ),
                    "session_id": session_id,
                },
                status_code=404,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class Surfaces:
    """Route one port to the agent surface or the workspace.

    A prefix dispatcher rather than three mounts: mounting the agent surface under
    its own path would prefix its routes twice, and a catch-all mount does not fall
    through on a 404.
    """

    def __init__(
        self,
        *,
        mcp: Any,
        workspace: Any,
        registry: ClientRegistry,
        mcp_prefix: str,
    ) -> None:
        self.mcp = ClientGate(mcp, registry)
        self.workspace = workspace
        self.mcp_prefix = mcp_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if path == self.mcp_prefix or path.startswith(f"{self.mcp_prefix}/"):
            await self.mcp(scope, receive, send)
            return
        await self.workspace(scope, receive, send)


def _claim_loopback_port(host: str, port: int) -> socket.socket:
    """Return the socket holding the claimed loopback port.

    A bound socket that is not listening can be bound again under `SO_REUSEADDR`,
    which Linux allows, so the claim is the listen and not a probe followed by a
    bind. The loser of a race gets an `OSError` here, before any server exists.
    `SO_REUSEADDR` is set anyway, matching uvicorn, so a port in `TIME_WAIT` after
    a stop is not mistaken for one another process holds.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_CLAIM_BACKLOG)
    except OSError:
        sock.close()
        raise
    return sock


class App:
    """The one running instance: a service, a port, and two HTTP surfaces.

    The service is built on construction; the gateway opens on the first operation
    that needs it, so a health check starts no process.
    """

    def __init__(self, config: ResearchConfig, *, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        self.config = config
        self.host = UI_HOST
        self.port = port
        self.error: str | None = None
        self.started_at: float | None = None
        self.gateway = LazyGateway(config)
        self.service = ResearchService(config, VanillaUltraRAG(self.gateway, config))
        self.clients = ClientRegistry()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def mcp_url(self) -> str:
        from ..surfaces.mcp import MCP_PATH

        return f"{self.url}{MCP_PATH}"

    @property
    def ready(self) -> bool:
        """Whether the app is serving right now.

        uvicorn sets `Server.started` once and never clears it, so the live task is
        part of the test: after `stop` or a failed claim the task is finished.
        """

        return (
            self._server is not None
            and bool(self._server.started)
            and self._task is not None
            and not self._task.done()
        )

    def state(self) -> dict[str, Any]:
        """The app's own state, which the agent surface reports in `status`."""

        return {
            "ui_url": self.url,
            "ui_ready": self.ready,
            "ui_error": self.error,
            "mcp_url": self.mcp_url,
            "mcp_clients": len(self.clients.attached),
        }

    def build(self) -> Starlette:
        """Compose the one application every front end is served by.

        Starlette does not run a mounted application's lifespan, so both are entered
        here: the agent app's for its session machinery and the workspace's to
        install the adapter it was handed.
        """

        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware

        from ..surfaces.mcp import MCP_PATH, create_mcp
        from ..surfaces.ui import create_ui_app
        from .control import control_routes

        mcp_http = create_mcp(
            self.config, connect=self._service, app_state=self.state
        ).http_app(path=MCP_PATH)
        workspace = create_ui_app(
            self.config,
            service=self.service,
            clients=self.clients,
            app_state=self.state,
        )

        @asynccontextmanager
        async def lifespan(_: Starlette) -> AsyncIterator[None]:
            async with (
                mcp_http.router.lifespan_context(mcp_http),
                workspace.router.lifespan_context(workspace),
            ):
                try:
                    yield
                finally:
                    await self.gateway.aclose()

        return Starlette(
            routes=[
                *control_routes(self),
                Mount(
                    "/",
                    app=Surfaces(
                        mcp=mcp_http,
                        workspace=workspace,
                        registry=self.clients,
                        mcp_prefix=MCP_PATH,
                    ),
                ),
            ],
            middleware=[Middleware(BaseHTTPMiddleware, dispatch=_security_headers)],
            lifespan=lifespan,
            exception_handlers={Exception: _unexpected_error},
        )

    async def _service(self) -> ResearchService:
        return self.service

    async def start(self) -> None:
        try:
            claim = _claim_loopback_port(self.host, self.port)
        except OSError as exc:
            reason = exc.strerror or str(exc)
            self.error = (
                f"Port {self.port} is already in use on {self.host}, so the app "
                f"was not started; choose another --port ({reason})."
            )
            return
        self._socket = claim
        self._server = uvicorn.Server(
            uvicorn.Config(
                self.build(),
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self.started_at = _now()
        _record_state(self.config, self.port)
        self._task = asyncio.create_task(self._server.serve(sockets=[claim]))
        self._task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        """Record an app that ended on its own, and release its claim.

        uvicorn can end the task by raising, which a probe-then-bind order hides
        behind an empty `error`.
        """

        if not task.cancelled():
            failure = task.exception()
            if failure is not None:
                self.error = (
                    f"The app on {self.host}:{self.port} stopped: "
                    f"{failure.__class__.__name__}: {failure}"
                )
        self._release_claim()
        _forget_state(self.config)

    def _release_claim(self) -> None:
        """Close the claimed socket, which uvicorn also closes on shutdown."""

        claim, self._socket = self._socket, None
        if claim is not None:
            with contextlib.suppress(OSError):
                claim.close()

    async def wait(self) -> None:
        """Block until the serving task ends, whether it served or it failed."""

        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def stop(self) -> None:
        """Ask the app to stop and wait for its task to finish.

        A failure must not take this process down: `_task_finished` records whatever
        ended the task and does not re-raise it.
        """
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._release_claim()
        await self.gateway.aclose()
        _forget_state(self.config)


@asynccontextmanager
async def running(config: ResearchConfig, port: int) -> AsyncIterator[App]:
    app = App(config, port=port)
    await app.start()
    try:
        yield app
    finally:
        await app.stop()


async def _security_headers(request: Any, call_next: Any) -> Any:
    """Apply one set of browser-facing headers to every route the app serves.

    The shared workspace sets these for its own routes, so setting them here covers
    the control API and the agent surface.
    """

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def _unexpected_error(_: Any, exc: Exception) -> JSONResponse:
    """Report an unexpected failure without its detail, and keep the reason.

    The detail goes to the app's log, which `research-rag doctor` names, because
    no reader can act on a traceback.
    """

    LOGGER.exception("research-rag request failed", exc_info=exc)
    return JSONResponse(
        {"error": "Unexpected app failure; inspect the app log."}, status_code=500
    )


def recorded_port(config: ResearchConfig) -> int | None:
    """Return the port a running app serves this project on, if there is one.

    A recorded port whose app has gone reads as no app rather than as a connection
    failure.
    """

    state = config.state_root
    try:
        port = int((state / PORT_FILE).read_text(encoding="utf-8").strip())
        pid = int((state / PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return port if alive(pid) else None


def recorded_pid(config: ResearchConfig) -> int | None:
    """Return the pid a running app recorded for this project, if it is still alive.

    The companion to `recorded_port`: a pid file survives a terminal that closed,
    so the process behind it is checked rather than trusted. The reader is
    `state_files`, and this is where a resolved config becomes a state root for it.
    """

    return _recorded_pid(config.state_root)


def running_url(config: ResearchConfig) -> str | None:
    """The running app's base URL, or None when the project has no app up."""

    port = recorded_port(config)
    return f"http://{UI_HOST}:{port}" if port is not None else None


__all__ = [
    "CONTROL_PREFIX",
    "PID_FILE",
    "PORT_FILE",
    "TTY_FILE",
    "UI_HOST",
    "App",
    "Client",
    "ClientError",
    "ClientGate",
    "ClientRegistry",
    "Surfaces",
    "alive",
    "has_terminal",
    "recorded_pid",
    "recorded_port",
    "running",
    "running_url",
]
