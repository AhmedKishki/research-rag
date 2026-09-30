"""The control API: how the command line reaches the one running app.

The app process owns the project lock, the gateway, and the service, so a
command that built its own service would be a second instance of the thing the
app exists to be singular. Everything the command line does to the project
therefore arrives here over loopback, on the same port and the same process the
browser workspace and the agent surface are served by.

This is a fourth surface, not a fifth argument to a third: it names the arguments
the command line sends, and the command line names what it sends, so the two
cannot drift without a test saying so. That is the same rule the workspace
adapter and the agent tools follow.

Loopback only, JSON only, and the same operations the service exposes. There is
no control operation that changes project state; the service is the only writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .app import CONTROL_PREFIX, ClientError, running_url
from .config import ResearchConfig
from .support import ResearchError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Self

    from .app import App


class ControlError(Exception):
    """A control call the app refused, or could not be delivered to one."""


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


async def _body(request: Request) -> dict[str, Any]:
    """Return the request's JSON object, or refuse the request as malformed.

    A control request that is not a JSON object is a caller bug, and saying so is
    more useful than letting the operation fail on a missing key later.
    """

    try:
        parsed = await request.json()
    except ValueError as exc:
        raise ResearchError(f"Expected a JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ResearchError("Expected a JSON object")
    return parsed


def _string_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ResearchError(f"{name} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _bounded(value: Any, *, default: int, maximum: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ResearchError(f"{name} must be between 1 and {maximum}")
    return value


async def _status(app: App, request: Request) -> JSONResponse:
    payload = await app.service.status()
    return _json({**payload, **app.state()})


async def _health(app: App, request: Request) -> JSONResponse:
    return _json(
        {
            "status": "ok",
            "project_root": str(app.config.project_root),
            "source_root": str(app.config.source_root),
            "ready": app.ready,
        }
    )


async def _clients(app: App, request: Request) -> JSONResponse:
    return _json({"clients": app.clients.report()})


async def _disconnect(app: App, request: Request) -> JSONResponse:
    session_id = request.path_params["session_id"]
    reason = (await _body(request)).get("reason") or "Disconnected by request."
    client = app.clients.disconnect(session_id, str(reason))
    return _json({"disconnected": client.report()})


async def _ingest(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    force = body.get("force_recompute", False)
    if not isinstance(force, bool):
        raise ResearchError("force_recompute must be a boolean")
    return _json(await app.service.ingest(force_recompute=force))


async def _search(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ResearchError("search requires a non-empty query")
    # A control request cannot choose the retrieval: the engine fixes hybrid,
    # reranking, and the freshness check, so no surface offers a mode.
    return _json(
        await app.service.search(
            query,
            top_k=_bounded(body.get("top_k"), default=10, maximum=50, name="top_k"),
            categories_any=_string_list(body.get("categories_any"), "categories_any"),
            projects_any=_string_list(body.get("projects_any"), "projects_any"),
            keywords=_string_list(body.get("keywords"), "keywords"),
            authors_any=_string_list(body.get("authors_any"), "authors_any"),
            titles_any=_string_list(body.get("titles_any"), "titles_any"),
            languages_any=_string_list(body.get("languages_any"), "languages_any"),
            source_ids=_string_list(body.get("source_ids"), "source_ids"),
            exclude_source_ids=_string_list(
                body.get("exclude_source_ids"), "exclude_source_ids"
            ),
            retrieval_method="hybrid",
            rerank=True,
            include_staleness=True,
        )
    )


async def _sources(app: App, request: Request) -> JSONResponse:
    return _json(await app.service.list_sources())


async def _passage(app: App, request: Request) -> JSONResponse:
    chunk_id = request.path_params["chunk_id"]
    context_chunks = _bounded(
        request.query_params.get("context_chunks"),
        default=1,
        maximum=5,
        name="context_chunks",
    )
    return _json(await app.service.get_passage(chunk_id, context_chunks=context_chunks))


async def _source_inclusion(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    included = body.get("included")
    if not isinstance(included, bool):
        raise ResearchError("an inclusion request needs a boolean included")
    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ResearchError("an exclusion reason must be a string")
    return _json(
        await app.service.set_source_inclusion(
            source_path=body.get("source_path"),
            source_id=body.get("source_id"),
            included=included,
            reason=reason,
        )
    )


async def _source_metadata(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        raise ResearchError("a metadata request needs a metadata object")
    return _json(
        await app.service.set_source_metadata(
            metadata=metadata,
            source_path=body.get("source_path"),
            source_id=body.get("source_id"),
        )
    )


async def _handle(app: App, endpoint: Any, request: Request) -> JSONResponse:
    """Turn a named failure into a named error, and leave the rest to the app.

    A refusal the caller can act on and a fault the caller cannot are different
    answers: the first is a 400 with the reason, the second is the app's own 500,
    which names the log instead of pretending to explain a traceback.
    """

    try:
        return await endpoint(app, request)
    except (ClientError, ResearchError) as exc:
        return _json({"error": str(exc)}, status_code=400)


def control_routes(app: App) -> list[Route]:
    """The routes the command line talks to, on the app's own port.

    They are declared before the workspace mount, so a URL under `/control` is
    answered here and every other URL falls through to the workspace.
    """

    def route(path: str, endpoint: Any, methods: list[str]) -> Route:
        async def bound(request: Request) -> JSONResponse:
            return await _handle(app, endpoint, request)

        bound.__name__ = getattr(endpoint, "__name__", "control")
        return Route(f"{CONTROL_PREFIX}{path}", bound, methods=methods)

    return [
        route("/status", _status, ["GET"]),
        route("/health", _health, ["GET"]),
        route("/clients", _clients, ["GET"]),
        route("/clients/{session_id}/disconnect", _disconnect, ["POST"]),
        route("/ingest", _ingest, ["POST"]),
        route("/search", _search, ["POST"]),
        route("/sources", _sources, ["GET"]),
        route("/passages/{chunk_id}", _passage, ["GET"]),
        route("/source-inclusion", _source_inclusion, ["POST"]),
        route("/source-metadata", _source_metadata, ["POST"]),
    ]


class Control:
    """A command line's handle on the running app.

    Every method answers with the app's own payload, so a terminal answer and a
    workspace answer are the same object. A refusal comes back as
    `ControlError` carrying the app's message, which is the message a reader of
    the workspace would have seen, not a transport failure.
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

    def status(self) -> dict[str, Any]:
        return self._call("GET", "/status")

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def clients(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/clients")["clients"])

    def disconnect(self, session_id: str, reason: str) -> dict[str, Any]:
        return self._call(
            "POST", f"/clients/{session_id}/disconnect", json={"reason": reason}
        )

    def ingest(self, *, force_recompute: bool = False) -> dict[str, Any]:
        return self._call("POST", "/ingest", json={"force_recompute": force_recompute})

    def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self._call("POST", "/search", json={"query": query, **arguments})

    def sources(self) -> dict[str, Any]:
        return self._call("GET", "/sources")

    def passage(self, chunk_id: str, *, context_chunks: int = 1) -> dict[str, Any]:
        return self._call(
            "GET", f"/passages/{chunk_id}", params={"context_chunks": context_chunks}
        )

    def set_source_inclusion(
        self,
        *,
        source_path: str | None = None,
        source_id: str | None = None,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "POST",
            "/source-inclusion",
            json={
                "source_path": source_path,
                "source_id": source_id,
                "included": included,
                "reason": reason,
            },
        )

    def set_source_metadata(
        self,
        *,
        source_path: str | None = None,
        source_id: str | None = None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self._call(
            "POST",
            "/source-metadata",
            json={
                "source_path": source_path,
                "source_id": source_id,
                "metadata": metadata,
            },
        )


def connect(config: ResearchConfig) -> Control | None:
    """Return a handle on the running app, or None when there is not one.

    Reading local state through a daemon that is not up would mean opening a
    second service for a project nobody is serving, so a command that only reads
    answers in process and a command that touches the corpus asks here first.
    """

    url = running_url(config)
    return Control(url) if url is not None else None


def ensure_running(config: ResearchConfig, *, timeout: float = 180.0) -> Control:
    """Return a handle on the app, starting it if the project has none.

    The launcher does the starting, because it is what already owns the free-port
    choice, the lock, the pid file, and the log. This waits for the port it
    recorded rather than probing, so the app this handle reaches is the one the
    launcher reported, not a stranger that answered the port first.
    """

    import time

    existing = connect(config)
    if existing is not None:
        return existing
    from .launcher import start_app

    started = start_app(config)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = connect(config)
        if handle is not None:
            return handle
        if not started.get("running", True):
            break
        time.sleep(0.2)
    raise ControlError(
        f"The app for {config.project_root} did not start; read "
        f"{config.state_root / 'logs' / 'research-rag-ui.log'}"
    )
