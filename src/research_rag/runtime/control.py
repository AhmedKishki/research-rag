"""The control API: how the command line reaches the one running app.

A command that built its own service would be a second instance of the app, so
every command-line operation arrives here over loopback, on the port and process
that serves the workspace and the agent surface.

A corpus operation arrives at one route and is checked against the table in
`core/operations.py`, the one the workspace reads. The other routes are the app's
own: its health, its LAN mode, and its clients. Loopback only, and the service
is the only writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..core.admission import as_caller
from ..core.operations import run
from ..project.config import ResearchConfig
from ..project.policy import ResearchError
from ..surfaces.workspace import write_guard
from .app import ClientError, running_url
from .network import CONTROL_PREFIX

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Self

    from .app import App


class ControlError(Exception):
    """A control call the app refused, or could not be delivered to one."""


def _write_refusal(request: Request) -> JSONResponse | None:
    """Refuse a request this app cannot show came from its own page on this machine.

    An operation may change what the project holds: a build, an exclusion, a
    removed generation. Each takes the gates `write_guard` holds, which are the
    workspace's own.
    """

    refusal = write_guard.write_refusal(request)
    if refusal is not None:
        status, reason = refusal
        return JSONResponse({"error": reason}, status_code=status)
    if not write_guard.json_content_type(request):
        return JSONResponse(
            {"error": "Write requests require application/json"}, status_code=415
        )
    return None


async def _body(request: Request) -> dict[str, Any]:
    """Return the request's JSON object, or refuse the request as malformed.

    A body that is not a JSON object is named as such rather than failing on a
    missing key later.
    """

    try:
        parsed = await request.json()
    except ValueError as exc:
        raise ResearchError(f"Expected a JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ResearchError("Expected a JSON object")
    return parsed


async def _operation(app: App, request: Request) -> JSONResponse:
    """One corpus operation, named in the path and argued in the body."""

    name = request.path_params["name"]
    payload = await run(app.service, name, await _body(request))
    # The command line reads the app's own state beside the service's, the
    # answer the workspace's status carries.
    if name == "status":
        payload = {**payload, **app.state()}
    return JSONResponse(payload)


async def _health(app: App, request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "project_root": str(app.config.project_root),
            "source_root": str(app.config.source_root),
            "ready": app.ready,
        }
    )


async def _clients(app: App, request: Request) -> JSONResponse:
    return JSONResponse({"clients": app.clients.report()})


async def _lan(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    if set(body) != {"enabled"} or not isinstance(body.get("enabled"), bool):
        raise ResearchError("LAN control requires only a boolean enabled")
    return JSONResponse(await app.set_lan(body["enabled"]))


async def _disconnect(app: App, request: Request) -> JSONResponse:
    session_id = request.path_params["session_id"]
    reason = (await _body(request)).get("reason") or "Disconnected by request."
    client = app.clients.disconnect(session_id, str(reason))
    return JSONResponse({"disconnected": client.report()})


async def _handle(app: App, endpoint: Any, request: Request) -> JSONResponse:
    """Turn a named failure into a named error, and leave the rest to the app.

    A refusal the caller can act on is a 400 with the reason; a fault it cannot is
    the app's own 500, which names the log.
    """

    try:
        with as_caller("command line"):
            return await endpoint(app, request)
    except (ClientError, ResearchError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def control_routes(app: App) -> list[Route]:
    """The routes the command line talks to, on the app's own port.

    They are declared before the workspace mount, so a URL under `/control` is
    answered here. A route that does not accept GET carries the loopback,
    same-origin, and JSON gates before its body is read: the gate is one
    implementation here rather than a line each endpoint remembers to copy.
    """

    def route(path: str, endpoint: Any, methods: list[str]) -> Route:
        async def bound(request: Request) -> JSONResponse:
            if "GET" not in methods:
                refusal = _write_refusal(request)
                if refusal is not None:
                    return refusal
            return await _handle(app, endpoint, request)

        bound.__name__ = getattr(endpoint, "__name__", "control")
        return Route(f"{CONTROL_PREFIX}{path}", bound, methods=methods)

    return [
        route("/operations/{name}", _operation, ["POST"]),
        route("/health", _health, ["GET"]),
        route("/lan", _lan, ["POST"]),
        route("/clients", _clients, ["GET"]),
        route("/clients/{session_id}/disconnect", _disconnect, ["POST"]),
    ]


class Control:
    """A command line's handle on the running app.

    Every method answers with the app's own payload, so a terminal answer and a
    workspace answer are the same object. A refusal comes back as `ControlError`
    carrying the app's message.
    """

    def __init__(self, base_url: str, *, timeout: float = 3600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"{CONTROL_PREFIX}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise ControlError(
                f"The app at {self.base_url} did not answer ({exc}); if it has "
                "stopped, start it with 'research-rag start'."
            ) from exc
        if response.status_code >= 400:
            try:
                message = response.json().get("error") or response.text
            except ValueError:
                message = response.text
            raise ControlError(str(message))
        return response.json()

    def operation(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One corpus operation from `core/operations.py`, run by the app."""

        return self._call("POST", f"/operations/{name}", json=arguments)

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def set_lan(self, enabled: bool) -> dict[str, Any]:
        return self._call("POST", "/lan", json={"enabled": enabled})

    def clients(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/clients")["clients"])

    def disconnect(self, session_id: str, reason: str) -> dict[str, Any]:
        return self._call(
            "POST", f"/clients/{session_id}/disconnect", json={"reason": reason}
        )


def connect(config: ResearchConfig, *, timeout: float = 3600.0) -> Control | None:
    """Return a handle on the running app, or None when there is not one.

    Reading local state through a daemon that is not up would mean opening a second
    service, so a command that only reads answers in process. The handle is built
    from the port this project recorded; `is_serving` is what says the app behind
    it is still there.
    """

    url = running_url(config)
    return Control(url, timeout=timeout) if url is not None else None


def is_serving(config: ResearchConfig, *, timeout: float = 5.0) -> bool:
    """Whether the app this project records is answering right now.

    A port file survives a terminal that closed, so `connect` alone can hand back a
    handle to an app that is gone. Asking it is the difference between telling a
    reader their project is being served and telling them it is not.
    """

    handle = connect(config, timeout=timeout)
    if handle is None:
        return False
    try:
        handle.health()
    except (ControlError, httpx.HTTPError):
        return False
    finally:
        handle.close()
    return True


def not_running_reason() -> str:
    """Why a project is not being served, in one sentence.

    Nothing is started to produce it. An app belongs to the terminal that started
    it, so a client that reaches for one is told that rather than quietly given a
    server of its own to leave behind. The command is `project_command`, so the
    remedy is written once where a client entry is built.
    """

    return (
        "An app runs in a terminal and ends when that terminal closes, so this "
        "project is not being served and nothing was started to change that."
    )
