"""The control API: how the command line reaches the one running app.

A command that built its own service would be a second instance of the app, so
every command-line operation arrives here over loopback, on the port and process
that serves the workspace and the agent surface.

It names what the command line sends and the command line names what it sends, so
the two cannot drift without a test saying so. Loopback only, and the service is
the only writer. A settings write carries the browser's own gates as well, because
it changes what the project's next build records.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..project.config import ResearchConfig
from ..project.policy import ResearchError
from ..project.registry import account_projects
from . import write_guard
from .app import CONTROL_PREFIX, ClientError, running_url
from .doctor import mcp_entry_block

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Self

    from .app import App


class ControlError(Exception):
    """A control call the app refused, or could not be delivered to one."""


# The bounds a control request may not exceed. They are the engine's own, and the
# command line's `--help` states the same two ranges, so a request outside one is
# refused here rather than answered differently per surface.
MAXIMUM_TOP_K = 50
MAXIMUM_CONTEXT_CHUNKS = 5


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def _write_refusal(request: Request) -> JSONResponse | None:
    """Refuse a write this app cannot show came from its own page on this machine.

    Every control write changes what the project holds: a build, an exclusion, a
    removed generation, or a settings file, so each takes the gates
    `write_guard` holds, which are the workspace's own.
    """

    refusal = write_guard.write_refusal(request)
    if refusal is not None:
        status, reason = refusal
        return _json({"error": reason}, status_code=status)
    if not write_guard.json_content_type(request):
        return _json(
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


def _string_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ResearchError(f"{name} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _bounded(
    value: Any,
    *,
    default: int,
    maximum: int,
    name: str,
    minimum: int = 1,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ResearchError(f"{name} must be between {minimum} and {maximum}")
    return value


def _query_int(
    request: Request,
    name: str,
    *,
    default: int,
    maximum: int,
    minimum: int = 1,
) -> int:
    """Read one bounded integer out of a query string, where every value is text.

    A query parameter arrives as a string, so the integer it names is read here
    rather than refused for not being one: the command line sends `1` and means
    one neighbour, and a bound that rejects the text rejects the command.
    """

    raw = request.query_params.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ResearchError(f"{name} must be an integer") from exc
    return _bounded(value, default=default, maximum=maximum, name=name, minimum=minimum)


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
            top_k=_bounded(
                body.get("top_k"),
                default=10,
                maximum=MAXIMUM_TOP_K,
                name="top_k",
            ),
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


async def _stats(app: App, _request: Request) -> JSONResponse:
    return _json(await app.service.search_stats())


async def _passage(app: App, request: Request) -> JSONResponse:
    chunk_id = request.path_params["chunk_id"]
    # The engine reads this bound, and the command line documents the same range,
    # so the route takes the engine's rather than narrowing one of the three
    # surfaces on its own.
    context_chunks = _query_int(
        request,
        "context_chunks",
        default=1,
        maximum=MAXIMUM_CONTEXT_CHUNKS,
        minimum=0,
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


async def _chunk_inclusion(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    included = body.get("included")
    if not isinstance(included, bool):
        raise ResearchError("a chunk decision needs a boolean included")
    chunk_id = body.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ResearchError("a chunk decision needs a chunk_id string")
    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ResearchError("a chunk exclusion reason must be a string")
    return _json(
        await app.service.set_chunk_inclusion(
            chunk_id=chunk_id,
            included=included,
            reason=reason,
        )
    )


async def _generations(app: App, _request: Request) -> JSONResponse:
    """Every retained generation, as the status payload reports it."""

    status = await app.service.status()
    generations = status.get("generations")
    return _json(
        {
            "generations": generations if isinstance(generations, list) else [],
            "retained_generation_count": int(
                status.get("retained_generation_count") or 0
            ),
            "retained_generation_bytes": int(
                status.get("retained_generation_bytes") or 0
            ),
            "current_generation_id": status.get("generation_id"),
        }
    )


async def _use_generation(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    generation_id = body.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ResearchError("a generation selection needs a generation_id string")
    return _json(await app.service.use_generation(generation_id))


async def _remove_generation(app: App, request: Request) -> JSONResponse:
    # The confirmation is carried over the wire rather than inferred from the
    # caller's intent. The command line already had to repeat the id, and a
    # surface that dropped the repeat would delete a generation a person did not
    # choose.
    body = await _body(request)
    generation_id = body.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ResearchError("a generation removal needs a generation_id string")
    confirm = body.get("confirm")
    if not isinstance(confirm, str):
        raise ResearchError("a generation removal needs a confirm string")
    return _json(await app.service.remove_generation(generation_id, confirm=confirm))


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


async def _settings_read(app: App, _request: Request) -> JSONResponse:
    return _json(await app.service.settings_read())


async def _settings_write(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    # The body names settings and nothing else. A path would move the file a
    # write lands in, so one is refused rather than ignored.
    unknown = set(body) - {"values", "expected_revision", "confirm"}
    if unknown:
        raise ResearchError(
            "A settings write takes values, expected_revision, and confirm: "
            f"{', '.join(sorted(unknown))}"
        )
    values = body.get("values")
    if not isinstance(values, dict) or not values:
        raise ResearchError("a settings write needs a values object")
    expected_revision = body.get("expected_revision")
    if not isinstance(expected_revision, str) or not expected_revision.strip():
        raise ResearchError(
            "a settings write needs the expected_revision it read, so a change "
            "made since is refused instead of overwritten"
        )
    confirm = body.get("confirm", False)
    if not isinstance(confirm, bool):
        raise ResearchError("confirm must be a boolean")
    return _json(
        await app.service.settings_write(
            values,
            expected_revision=expected_revision,
            confirm=confirm,
        )
    )


async def _projects(app: App, _request: Request) -> JSONResponse:
    """Every project this installation serves, read over the control API.

    The account record and the probe each project's own app are the ones the
    workspace's selector uses, so a terminal and a browser cannot disagree about
    which projects exist or which are up. The probe is blocking loopback I/O, so
    it runs off the event loop rather than holding up the app serving it.
    """

    return _json(await asyncio.to_thread(account_projects))


async def _agent_entry(app: App, _request: Request) -> JSONResponse:
    """This project's client entry, the text `doctor --mcp-entry` prints."""

    return _json({"entry": mcp_entry_block(app.config)})


async def _handle(app: App, endpoint: Any, request: Request) -> JSONResponse:
    """Turn a named failure into a named error, and leave the rest to the app.

    A refusal the caller can act on is a 400 with the reason; a fault it cannot is
    the app's own 500, which names the log.
    """

    try:
        return await endpoint(app, request)
    except (ClientError, ResearchError) as exc:
        return _json({"error": str(exc)}, status_code=400)


def control_routes(app: App) -> list[Route]:
    """The routes the command line talks to, on the app's own port.

    They are declared before the workspace mount, so a URL under `/control` is
    answered here. A route that does not accept GET changes what the project
    holds, so it carries the loopback, same-origin, and JSON gates before its body
    is read: the gate is one implementation here rather than a line each write
    endpoint remembers to copy.
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
        route("/status", _status, ["GET"]),
        route("/health", _health, ["GET"]),
        route("/clients", _clients, ["GET"]),
        route("/clients/{session_id}/disconnect", _disconnect, ["POST"]),
        route("/ingest", _ingest, ["POST"]),
        route("/search", _search, ["POST"]),
        route("/sources", _sources, ["GET"]),
        route("/stats", _stats, ["GET"]),
        route("/passages/{chunk_id}", _passage, ["GET"]),
        route("/source-inclusion", _source_inclusion, ["POST"]),
        route("/chunk-inclusion", _chunk_inclusion, ["POST"]),
        route("/source-metadata", _source_metadata, ["POST"]),
        route("/generations", _generations, ["GET"]),
        route("/generations/use", _use_generation, ["POST"]),
        route("/generations/remove", _remove_generation, ["POST"]),
        route("/settings", _settings_read, ["GET"]),
        route("/settings", _settings_write, ["POST"]),
        route("/projects", _projects, ["GET"]),
        route("/agent-entry", _agent_entry, ["GET"]),
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

    def stats(self) -> dict[str, Any]:
        return self._call("GET", "/stats")

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

    def set_chunk_inclusion(
        self,
        *,
        chunk_id: str,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "POST",
            "/chunk-inclusion",
            json={
                "chunk_id": chunk_id,
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

    def generations(self) -> dict[str, Any]:
        return self._call("GET", "/generations")

    def use_generation(self, generation_id: str) -> dict[str, Any]:
        return self._call(
            "POST", "/generations/use", json={"generation_id": generation_id}
        )

    def remove_generation(self, generation_id: str, *, confirm: str) -> dict[str, Any]:
        return self._call(
            "POST",
            "/generations/remove",
            json={"generation_id": generation_id, "confirm": confirm},
        )

    def settings(self) -> dict[str, Any]:
        return self._call("GET", "/settings")

    def write_settings(
        self,
        values: dict[str, Any],
        *,
        expected_revision: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        return self._call(
            "POST",
            "/settings",
            json={
                "values": values,
                "expected_revision": expected_revision,
                "confirm": confirm,
            },
        )

    def projects(self) -> dict[str, Any]:
        return self._call("GET", "/projects")

    def agent_entry(self) -> dict[str, Any]:
        return self._call("GET", "/agent-entry")


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
