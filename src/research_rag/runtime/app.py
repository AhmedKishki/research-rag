"""The running app: one project, one port, one service, three front ends.

One process owns the project lock and direct retrieval backend, and serves the
workspace and the agent surface on one port, so a passage an agent retrieves
and one the workspace renders are the same object. Explicit home-LAN mode
exposes the workspace; control and agent access remain loopback-only.

**MCP clients attach and detach.** A streamable-HTTP session is recorded on its
first request, and a forced detach refuses the rest of it. The bridge in
`bridge.py` names itself and declares the host it runs for, so an agent is
identifiable by more than a peer address, and a client that is gone is
forgotten rather than listed for ever.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..project.config import ConfigurationError, ResearchConfig
from ..project.state_files import PID_FILE, PORT_FILE, TTY_FILE
from ..project.state_files import process_alive as alive
from ..project.state_files import recorded_pid as _recorded_pid
from .network import NetworkAccess, discover_lan_addresses

if TYPE_CHECKING:
    # The engine is reached when the app serves, not when it is imported: a command
    # that only reads this module's port and pid records has to work on a machine
    # where the retrieval stack is absent.
    from ..core.service import ResearchService

LOGGER = logging.getLogger(__name__)

UI_HOST = "127.0.0.1"
CONTROL_PREFIX = "/control"
# Matches uvicorn's own default accept backlog, so a claimed socket is in the
# state uvicorn would have created for itself.
_CLAIM_BACKLOG = 2048
# A session that has said nothing for this long is reported as attached no
# longer. It is listed rather than dropped straight away, so a reader can see
# that it was here, and is forgotten once it has been silent for longer than
# CLIENT_RETENTION_SECONDS.
CLIENT_IDLE_SECONDS = 90.0
# How long a client that has gone quiet is still listed. It is longer than the
# idle window that reports it as attached, so a row is not removed while it is
# still the answer to "who is here".
CLIENT_RETENTION_SECONDS = 900.0
# The header the stdio bridge sets so an agent is identifiable by name. A direct
# HTTP client that does not set it is identified by its peer address.
CLIENT_NAME_HEADER = "x-research-rag-client"
# The variable a stdio bridge reads to learn what to call itself, which is how a
# client names its own session. It is declared here beside the header it becomes,
# because the app is what has to be told.
CLIENT_NAME_ENV = "RESEARCH_RAG_CLIENT_NAME"
# The header the stdio bridge sets with what the client it runs for inherited:
# the program that started it, whether it came over ssh, the directory it was
# started in, and the project it was asked for. A name says which agent; this
# says where that agent is running, which a name never does.
CLIENT_IDENTITY_HEADER = "x-research-rag-client-identity"
# What a client is shown when it named itself nothing. The session id is an
# opaque handle, so a row that leads with one answers nothing.
STDIO_CLIENT_LABEL = "stdio bridge, unnamed"
HTTP_CLIENT_LABEL = "http client, unnamed"
# A client declares its identity in a request header it writes itself, so the
# length is bounded and every field is taken by name: nothing a client sends
# becomes part of the app's own answer unchecked.
IDENTITY_HEADER_LIMIT = 1024
IDENTITY_TEXT_LIMIT = 200


def parse_client_identity(header: str | None) -> dict[str, Any]:
    """The facts a client declared about itself, or nothing.

    The declaration is read as a small JSON object and rebuilt field by field, so
    a header this app does not recognise adds no key and a value that is not the
    kind this app reads adds no fact. A client is on this app's loopback and is
    therefore someone on this machine; the declaration is what that client says
    about itself, and the app reports it as said rather than verifying it.
    """

    if not header or len(header) > IDENTITY_HEADER_LIMIT:
        return {}
    try:
        declared = json.loads(header)
    except ValueError:
        return {}
    if not isinstance(declared, dict):
        return {}

    def text(container: Mapping[str, Any], key: str) -> str:
        value = container.get(key)
        if not isinstance(value, str):
            return ""
        return value.strip()[:IDENTITY_TEXT_LIMIT]

    facts: dict[str, Any] = {
        key: text(declared, key)
        for key in ("agent", "project", "cwd")
        if text(declared, key)
    }
    pid = declared.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        facts["pid"] = pid
    host = declared.get("host")
    if isinstance(host, Mapping):
        # The four host facts are always present or none of them are, so a reader
        # of the report never has to tell an undeclared fact from a false one.
        facts["host"] = {
            "program": text(host, "program"),
            "term": text(host, "term"),
            "ssh": bool(host.get("ssh")),
            "tmux": bool(host.get("tmux")),
        }
    return facts


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


def _same_client(one: Client, other: Client) -> bool:
    """Whether two sightings are one client opening more than one connection.

    A bridge declares the process it runs in, so two sightings from one process
    are one client. Two that declared nothing but a name are one client as far
    as this app can tell, which is why one client's several connections are
    counted on its own row rather than left to look like several clients.
    """

    if one.identity and other.identity:
        return one.identity.get("pid") == other.identity.get("pid")
    return one.name == other.name


@dataclass
class Client:
    """One MCP client this app has seen.

    `attached` is a property of recency and of an explicit drop, never a stored
    flag. `name` is what the client called itself, or its session id when it
    called itself nothing; `label` is what a reader is shown instead, and
    `identity` is what the client said about the machine it runs in.

    One client can hold more than one session id: a stdio bridge opens a new
    upstream session per operation, so the process behind the pipe, not the
    session, is the client. `session_id` is the one a disconnect is asked for
    and `session_ids` is every one this client has opened.
    """

    session_id: str
    name: str
    first_seen: float
    last_seen: float
    requests: int = 0
    dropped: bool = False
    detached_reason: str | None = None
    # A sighting without a session id: the client opened a connection that has
    # not carried one yet. It is never counted as attached, because a connection
    # is not a client, and the MCP SDK opens the notification stream before the
    # session exists, so counting sightings would count one agent twice.
    pending: bool = False
    # The name the client actually sent, empty when it sent none. It is kept
    # beside `name` because that one falls back to the session id, and a reader
    # is told whether a row is named rather than shown a handle as if it were
    # a name.
    declared_name: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    peer: str = ""
    user_agent: str = ""
    session_ids: list[str] = field(default_factory=list)

    @property
    def attached(self) -> bool:
        if self.pending or self.dropped:
            return False
        return _now() - self.last_seen < CLIENT_IDLE_SECONDS

    @property
    def idle_seconds(self) -> float:
        return round(_now() - self.last_seen, 1)

    @property
    def transport(self) -> str:
        """Which way this client reached the app, as the app can tell.

        The stdio bridge is the only client that declares the host it runs for,
        so a declared identity is a bridge and nothing else is.
        """

        return "stdio" if self.identity else "http"

    @property
    def label(self) -> str:
        """The name a reader is shown, which is never a session id alone.

        A client that named itself is shown its name. One that named nothing is
        shown the kind of connection it opened, because an id in that place
        tells a reader which row to disconnect and nothing about who sent it.
        """

        if self.declared_name:
            return self.declared_name
        return STDIO_CLIENT_LABEL if self.identity else HTTP_CLIENT_LABEL

    def report(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_ids": list(self.session_ids),
            "name": self.name,
            "label": self.label,
            "transport": self.transport,
            "declared_name": self.declared_name,
            "attached": self.attached,
            "pending": self.pending,
            # A client opens as many sessions as its work needs, and the count
            # is reported rather than hidden: the several rows one agent used to
            # produce are now one row that says how many there were.
            "sessions": len(self.session_ids),
            "requests": self.requests,
            "idle_seconds": self.idle_seconds,
            "detached_reason": self.detached_reason,
            "peer": self.peer,
            "user_agent": self.user_agent,
            "identity": self.identity,
        }


class ClientRegistry:
    """The clients attached to this app, and the authority to drop one.

    A client is the process behind the pipe where it declared one, and the MCP
    session id where it did not. A stdio bridge opens a new upstream session per
    operation, so a session id is not a client: one agent is one row, and the
    sessions it has opened are listed beside it. A client that declared nothing
    is the session, which is the most this app can say about a hand-written
    HTTP client.
    """

    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        # Every session id this app has seen, mapped to the client that owns it.
        self._sessions: dict[str, str] = {}
        # The sightings that carry no session id, keyed by the peer address they
        # arrived from, so one client does not read as two while its first
        # session is still being created.
        self._awaiting: dict[str, str] = {}

    @staticmethod
    def _key(session_id: str, identity: Mapping[str, Any] | None) -> str:
        """The one client `session_id` belongs to.

        A declared process id is unique among the processes running on this
        machine, so every session it opens lands on the same client. A pid the
        operating system has since handed to a new bridge joins the row its
        predecessor left, which carries the facts the new bridge declares and is
        forgotten once it has been quiet for a while.
        """

        pid = (identity or {}).get("pid")
        return f"bridge:{pid}" if isinstance(pid, int) else f"session:{session_id}"

    def _touch(
        self,
        session_id: str,
        name: str,
        *,
        pending: bool = False,
        identity: Mapping[str, Any] | None = None,
        peer: str = "",
        user_agent: str = "",
    ) -> Client:
        key = self._key(session_id, identity)
        existing = self._clients.get(key)
        if existing is None:
            existing = Client(
                session_id=session_id,
                name=name or session_id,
                first_seen=_now(),
                last_seen=_now(),
                pending=pending,
            )
            self._clients[key] = existing
        existing.last_seen = _now()
        existing.requests += 1
        if name:
            existing.name = name
            existing.declared_name = name
        if identity:
            existing.identity = dict(identity)
        if peer:
            existing.peer = peer
        if user_agent:
            existing.user_agent = user_agent
        if pending:
            # A connection is not a session, so it is not an id a disconnect can
            # be asked for, and it does not put a client whose session has
            # already been seen back into the pending state.
            self._awaiting[session_id] = key
            return existing
        if session_id not in existing.session_ids:
            existing.session_ids.append(session_id)
        if existing.pending or not existing.session_id:
            # The first session this client carried replaces the peer address the
            # connection was counted against.
            existing.session_id = session_id
        existing.pending = False
        self._sessions[session_id] = key
        return existing

    def observe(self, request: Any) -> str | None:
        headers = request.headers
        session_id = (headers.get("mcp-session-id") or "").strip()
        name = (headers.get(CLIENT_NAME_HEADER) or "").strip()
        identity = parse_client_identity(headers.get(CLIENT_IDENTITY_HEADER))
        user_agent = (headers.get("user-agent") or "").strip()[:IDENTITY_TEXT_LIMIT]
        peer = f"{request.client.host}:{request.client.port}" if request.client else "?"
        if not session_id:
            return self._touch(
                peer,
                name,
                pending=True,
                identity=identity,
                peer=peer,
                user_agent=user_agent,
            ).session_id
        carried = session_id in self._sessions
        client = self._touch(
            session_id, name, identity=identity, peer=peer, user_agent=user_agent
        )
        if not carried:
            self._fold_pending(client, name)
        return client.session_id

    def _fold_pending(self, client: Client, name: str) -> None:
        """Fold the sightings that carry no session onto the session they belong to.

        A client opens its notification connection before the session exists, so
        a connection is recorded against the peer address. A client that declared
        a process is already one client; one that declared only a name is folded
        by that name, and one that declared nothing at all is folded onto the
        next session to arrive, which is the most this app can say about it.
        """

        for key, sighting in list(self._clients.items()):
            if sighting is client or not sighting.pending:
                continue
            if name and not _same_client(sighting, client):
                continue
            self._clients.pop(key)
            self._awaiting.pop(sighting.session_id, None)
            client.requests += sighting.requests

    def _forget_gone(self) -> None:
        """Drop the clients that have been silent for longer than they are kept.

        A row outlives its client so a reader can see that one was here and that
        a disconnect took effect. A machine starts a bridge per agent session,
        so keeping every client this app has ever seen fills the list with agents
        that are gone, and a row that outlives its process carries the process
        id, the directory, and the host of a program no longer running. A client
        that comes back is recorded again, carrying the facts its current process
        declares.
        """

        for key, client in list(self._clients.items()):
            if client.attached or _now() - client.last_seen < CLIENT_RETENTION_SECONDS:
                continue
            self._clients.pop(key)
            self._awaiting.pop(client.session_id, None)
            for session_id in client.session_ids:
                self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Client | None:
        """The client a session id belongs to, whichever of its sessions it is.

        A peer address is answered as well as a session id, so a client whose
        first session has not been created yet is refused as what it is rather
        than as a client this app does not have.
        """

        key = self._sessions.get(session_id) or self._awaiting.get(session_id)
        return self._clients.get(key or session_id)

    def disconnect(self, session_id: str, reason: str) -> Client:
        self._forget_gone()
        client = self.get(session_id)
        if client is None:
            raise ClientError(f"No client is attached with session {session_id}")
        if client.pending:
            raise ClientError(
                f"{session_id} is a connection that carries no session yet, so "
                "there is nothing to disconnect; wait for the client to "
                "initialize and ask again."
            )
        if not client.attached:
            raise ClientError(f"Client {client.label} is already detached")
        client.dropped = True
        client.detached_reason = reason
        # The MCP SDK opens a notification connection before the session exists,
        # so that connection carries no session id and cannot be matched by one.
        # It is matched by what identifies the client, and dropping it makes the
        # client end rather than sit on an open connection with a session the app
        # has already refused. A client that declared nothing cannot have its
        # connection identified, and only its sessions are refused.
        for other in self._clients.values():
            if other is not client and other.pending and _same_client(other, client):
                other.dropped = True
                other.detached_reason = reason
        return client

    def report(self) -> list[dict[str, Any]]:
        """Every client this app is still holding, attached or not.

        A client stays in the listing after it is dropped so the outcome of a
        disconnect is readable: its `attached` flag says it is gone and
        `detached_reason` says why. Recency alone decides `attached`, so a session
        whose process has ended reads as detached and is still listed, until it
        has been silent for longer than a row is kept.
        """

        self._forget_gone()
        return [client.report() for client in self._clients.values()]

    @property
    def attached(self) -> list[Client]:
        self._forget_gone()
        return [client for client in self._clients.values() if client.attached]


def request_caller() -> str:
    """Who the agent request being handled came from, as a fair queue names it.

    One agent is one caller however many sessions it opens, which is the grain
    the client list already keeps, so a bridge that opens a session per call
    waits in one place in the rounds.
    """

    from fastmcp.server.dependencies import get_http_headers

    headers = get_http_headers(include_all=True)
    identity = parse_client_identity(headers.get(CLIENT_IDENTITY_HEADER))
    session_id = (headers.get("mcp-session-id") or "").strip()
    return ClientRegistry._key(session_id, identity) if session_id or identity else ""


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
    """Return the socket holding the claimed serving port.

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


def _service_pair(config: ResearchConfig) -> tuple[Any, ResearchService]:
    """The gateway and the service that speaks through it, built together.

    The engine is imported here rather than at module scope so that importing this
    module — to read a port record, name a pid, or claim a port — does not need the
    retrieval stack. `AGENTS.md` holds the rule that a command which only reports
    has to answer on a machine where the runtime is absent.
    """

    from ..core.service import ResearchService
    from ..retrieval.direct import DirectRetrieval

    backend = DirectRetrieval(config)
    return backend, ResearchService(config, backend)


class App:
    """The one running instance: a service, a port, and two HTTP surfaces.

    The service is built on construction; retrieval dependencies load on the first
    operation that needs them, so a health check starts no process.
    """

    def __init__(self, config: ResearchConfig, *, port: int, lan: bool = False) -> None:
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        self.config = config
        self.host = "0.0.0.0" if lan else UI_HOST
        self.port = port
        self.network_access = NetworkAccess(
            port=port, lan=lan, addresses=discover_lan_addresses() if lan else ()
        )
        self.error: str | None = None
        self.started_at: float | None = None
        self.gateway, self.service = _service_pair(config)
        self.clients = ClientRegistry()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self._network_lock = asyncio.Lock()

    async def set_lan(self, enabled: bool) -> dict[str, Any]:
        """Replace accepting sockets, not the app, gateway, lifespan, or port.

        Existing connections stay alive but every request checks the live policy.
        A failed bind restores the old listener before returning a refusal.
        """
        from ..project.policy import ResearchError

        async with self._network_lock:
            if not self.ready or self._server is None:
                raise ResearchError("The app is not serving; run research-rag start.")
            if enabled == self.network_access.lan:
                return self.network_access.report()
            try:
                addresses = discover_lan_addresses() if enabled else ()
            except OSError as exc:
                raise ResearchError(
                    f"Cannot read LAN interfaces ({exc}); previous listener unchanged. "
                    "Retry research-rag lan enable."
                ) from exc
            old_host = self.host
            host = "0.0.0.0" if enabled else UI_HOST
            server = self._server
            for listener in server.servers:
                listener.close()
            # Python 3.12 wait_closed also waits for active transports, including
            # this control request. close releases the accepting socket now;
            # connections remain owned by uvicorn's shared server_state.
            self._release_claim()

            async def listen(bind_host: str) -> None:
                claim = _claim_loopback_port(bind_host, self.port)
                try:
                    listener = await asyncio.get_running_loop().create_server(
                        lambda: server.config.http_protocol_class(
                            config=server.config,
                            server_state=server.server_state,
                            app_state=server.lifespan.state,
                        ),
                        sock=claim,
                        backlog=_CLAIM_BACKLOG,
                    )
                except BaseException:
                    claim.close()
                    raise
                self._socket = claim
                server.servers = [listener]

            try:
                await listen(host)
            except (Exception, asyncio.CancelledError) as exc:
                try:
                    await listen(old_host)
                except Exception as rollback:
                    self.network_access.lan = False
                    self.network_access.addresses = ()
                    self.error = f"Listener rollback failed: {rollback}; stop and restart research-rag."
                    server.should_exit = True
                    raise ResearchError(self.error) from rollback
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise ResearchError(
                    f"LAN change failed ({exc}); previous listener restored. "
                    "Retry research-rag lan "
                    + ("enable" if enabled else "disable")
                    + "."
                ) from exc
            self.host = host
            self.network_access.addresses = addresses
            self.network_access.lan = enabled
            return self.network_access.report()

    @property
    def url(self) -> str:
        return f"http://{UI_HOST}:{self.port}"

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
            "lan": self.network_access.report(),
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
            self.config,
            connect=self._service,
            app_state=self.state,
            caller=request_caller,
        ).http_app(path=MCP_PATH)
        workspace = create_ui_app(
            self.config,
            service=self.service,
            clients=self.clients,
            app_state=self.state,
            network_access=self.network_access,
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
            middleware=[
                Middleware(BaseHTTPMiddleware, dispatch=self._security_headers)
            ],
            lifespan=lifespan,
            exception_handlers={Exception: _unexpected_error},
        )

    async def _security_headers(self, request: Any, call_next: Any) -> Any:
        refusal = self.network_access.request_refusal(request)
        # MCP has its own protocol; the control API retains its strict
        # loopback write guard. Browser writes use the serving policy.
        if (
            refusal is None
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and not request.url.path.startswith(("/control", "/mcp"))
        ):
            refusal = self.network_access.write_refusal(request)
        if refusal is not None:
            code, error = refusal
            return JSONResponse({"error": error}, status_code=code)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

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
                proxy_headers=False,
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
        async with self._network_lock:
            if self._server is not None:
                self._server.should_exit = True
            task, self._task = self._task, None
        if task is not None:
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

    from .write_guard import served_authority

    if served_authority(request) is None:
        return JSONResponse(
            {"error": "Requests require a loopback Host"}, status_code=403
        )
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
