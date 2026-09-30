"""The running app: one project, one port, one service, three front ends.

`research-rag` is a server product that also has a browser. An agent, the browser
workspace, and the command line are three front ends to the same running
instance, not three processes that each build their own view of the project.
That is why this module exists: one process owns the project lock, opens the
UltraRAG gateway once, serves the workspace and the agent surface on one loopback
port, and knows which agents are attached to it.

The app is up or it is down, and that is the one thing to check. When it is up
there is exactly one service, so the passage an agent retrieves and the passage
the workspace renders are the same object from the same call.

**MCP clients attach and detach.** A streamable-HTTP session is recorded when its
first request arrives, and a forced detach refuses the rest of that session,
which is what ends it. The bridge in `bridge.py` names itself, so an agent is
identifiable by more than a peer address.
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
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from .config import ConfigurationError, ResearchConfig
from .service import ResearchService
from .ultrarag import LazyGateway, VanillaUltraRAG

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
PORT_FILE = "research-rag-ui.port"
PID_FILE = "research-rag-ui.pid"


class ClientError(Exception):
    """A control request that named something this app does not have."""


def _now() -> float:
    return time.monotonic()


@dataclass
class Client:
    """One MCP client this app has seen.

    `attached` is a property of recency and of an explicit drop, never a stored
    flag: a client that stopped talking and a client this app disconnected are
    different facts, and the reason is kept so the list can say which.
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

    Identity is the MCP session id once a request carries one, because that is
    the handle every later request of the same client repeats. The initialize
    request has no id yet, so it is counted against the peer address, which keeps
    a client that never initializes visible instead of invisible.
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
        """Record one request and return the session id it belongs to."""

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
        # matched by the name the client gave itself, and dropping it is what
        # makes the client end rather than sit on an open stream with a session
        # the app has already refused. A client that named nothing cannot have
        # its stream identified, and only its session is refused.
        for other in self._clients.values():
            if other is not client and other.pending and other.name == client.name:
                other.dropped = True
                other.detached_reason = reason
        return client

    def report(self) -> list[dict[str, Any]]:
        return [client.report() for client in self._clients.values()]

    @property
    def attached(self) -> list[Client]:
        return [client for client in self._clients.values() if client.attached]


class ClientGate:
    """Count and refuse MCP requests, so the registry is the only authority.

    A forced detach has to end a live session, and the only place that can be
    enforced is in front of the session: the id is refused here and never reaches
    the agent, so a disconnected client sees its session fail rather than a tool
    that silently does nothing.
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
    """Route one port to the agent surface, the control API, or the workspace.

    A prefix dispatcher rather than three mounts, because mounting the agent
    surface under its own path would prefix its own routes twice, and a
    catch-all mount does not fall through to the next mount on a 404. Dispatching
    on the path is explicit, so what answers a given URL is stated in one place.
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
    """Bind and listen on the loopback port, and return the socket that holds it.

    The claim is the bind and the listen, not a probe followed by a bind, so two
    apps that start at the same time cannot both believe they hold the port: the
    loser gets an `OSError` here, before any server exists, and reports it.
    Listening is what makes the claim exclusive, because a socket that is bound
    but not listening can still be bound again under `SO_REUSEADDR`, which Linux
    allows for a socket that is not accepting. `SO_REUSEADDR` is set anyway,
    matching uvicorn, so a port whose connections are still in `TIME_WAIT` after
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


def alive(pid: int) -> bool:
    """Whether a process still exists, without signalling it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class App:
    """The one running instance: a service, a port, and two HTTP surfaces.

    The service is built on construction and opens nothing: the gateway opens on
    the first operation that needs it, so an app that is only being asked whether
    it is healthy never starts a process below it.
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
        from .surfaces.mcp import MCP_PATH

        return f"{self.url}{MCP_PATH}"

    @property
    def ready(self) -> bool:
        """Whether the app is serving right now.

        uvicorn sets `Server.started` once and never clears it, so the live task
        is part of the test: after `stop` and after a failed claim the task is
        finished and the app is not serving.
        """

        return (
            self._server is not None
            and bool(self._server.started)
            and self._task is not None
            and not self._task.done()
        )

    def state(self) -> dict[str, Any]:
        """The app's own state, which the agent surface reports in `status`.

        It is the app's state rather than the project's: where the workspace is,
        whether it is up, and who is attached.
        """

        return {
            "ui_url": self.url,
            "ui_ready": self.ready,
            "ui_error": self.error,
            "mcp_url": self.mcp_url,
            "mcp_clients": len(self.clients.attached),
        }

    def build(self) -> Starlette:
        """Compose the one application every front end is served by.

        Each mounted application owns state its own handlers read back, and
        Starlette does not run a mounted application's lifespan, so both are
        entered here: the agent app's for its session machinery, which is what
        makes the endpoint answer a session at all, and the workspace's to
        install the adapter it was handed.
        """

        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware

        from .control import control_routes
        from .surfaces.mcp import MCP_PATH, create_mcp
        from .surfaces.ui import create_ui_app

        mcp_http = create_mcp(
            self.config, connect=self._service, app_state=self.state
        ).http_app(path=MCP_PATH)
        workspace = create_ui_app(self.config, service=self.service)

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
        """Claim the port and serve until stopped."""

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
        self._task = asyncio.create_task(self._server.serve(sockets=[claim]))
        self._task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        """Record an app that ended on its own, and release its claim.

        uvicorn can end the task by raising, which a probe-then-bind order hides
        behind an empty `error`; the reason is kept here instead of showing a dead
        app with no explanation.
        """

        if not task.cancelled():
            failure = task.exception()
            if failure is not None:
                self.error = (
                    f"The app on {self.host}:{self.port} stopped: "
                    f"{failure.__class__.__name__}: {failure}"
                )
        self._release_claim()

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

        A failure must not take this process down with it: whatever ended the
        task is recorded by `_task_finished`, so it is not re-raised.
        """

        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._release_claim()
        await self.gateway.aclose()


@asynccontextmanager
async def running(config: ResearchConfig, port: int) -> AsyncIterator[App]:
    """Serve one project for the life of this block, then stop it."""

    app = App(config, port=port)
    await app.start()
    try:
        yield app
    finally:
        await app.stop()


async def _security_headers(request: Any, call_next: Any) -> Any:
    """Apply one set of browser-facing headers to every route the app serves.

    The shared workspace sets these for its own routes; setting them once here
    means the control API and the agent surface are covered by the same rule
    rather than by nobody's attention.
    """

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def _unexpected_error(_: Any, exc: Exception) -> JSONResponse:
    """Report an unexpected failure without its detail, and keep the reason.

    The detail goes to the app's log, which the launcher records and `research-rag
    doctor` names, because a message in an answer is read by a person or an agent
    that cannot act on a traceback and will report it as the whole of what
    happened.
    """

    LOGGER.exception("research-rag request failed", exc_info=exc)
    return JSONResponse(
        {"error": "Unexpected app failure; inspect the app log."}, status_code=500
    )


def recorded_port(config: ResearchConfig) -> int | None:
    """Return the port a running app serves this project on, if there is one.

    The generated launcher records the port it chose next to the pid it started,
    so a command can find the app without being told. A recorded port whose app
    has gone reads as no app rather than as a connection failure.
    """

    state = config.state_root
    try:
        port = int((state / PORT_FILE).read_text(encoding="utf-8").strip())
        pid = int((state / PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return port if alive(pid) else None


def running_url(config: ResearchConfig) -> str | None:
    """The running app's base URL, or None when the project has no app up."""

    port = recorded_port(config)
    return f"http://{UI_HOST}:{port}" if port is not None else None


__all__ = [
    "CONTROL_PREFIX",
    "PID_FILE",
    "PORT_FILE",
    "UI_HOST",
    "App",
    "Client",
    "ClientError",
    "ClientGate",
    "ClientRegistry",
    "Surfaces",
    "alive",
    "recorded_port",
    "running",
    "running_url",
]
