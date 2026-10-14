"""Secure loopback web host for this app's document workspace."""

from __future__ import annotations

import hashlib
import html
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from . import write_guard
from .contracts import SourceFile, UIAdapter, UIRequestError

LOGGER = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).with_name("static")
# The page assets and their media types; nothing else under the static
# root is served.
_ASSETS = {
    "app.css": "text/css",
    "app.js": "text/javascript",
    "remote.js": "text/javascript",
}
MAX_ERROR_LENGTH = 1200
# The panels that act on this machine. A LAN viewer is not shown them, and the
# network policy refuses their routes to it.
LOCAL_PANELS = ("clients", "projects", "agent_entry", "updates")


async def _adapter_call(
    request: Request,
    operation: str,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter: UIAdapter = request.app.state.adapter
    try:
        value = await adapter.call(operation, arguments or {})
    except UIRequestError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail[:MAX_ERROR_LENGTH],
        ) from exc
    if not isinstance(value, Mapping):
        raise HTTPException(
            status_code=502,
            detail=f"Adapter operation {operation!r} returned a non-object response",
        )
    return dict(value)


async def _json_body(request: Request) -> dict[str, Any]:
    refusal = write_guard.write_refusal(request)
    if refusal is not None:
        status, detail = refusal
        raise HTTPException(status_code=status, detail=detail)
    if not write_guard.json_content_type(request):
        raise HTTPException(
            status_code=415,
            detail="Write requests require application/json",
        )
    try:
        value = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON request") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return value


def _stamped_page(static_root: Path) -> str:
    """Return the page with each asset address carrying a digest of its bytes.

    The app is updated in place, and a browser may reuse a cached asset whatever
    the cache headers say, such as one a process without them served earlier. A
    page that names its assets by content can only load the script and the
    stylesheet it was written with, so a newer page never runs an older script
    against elements that script does not know.
    """

    page = (static_root / "index.html").read_text(encoding="utf-8")
    for filename in _ASSETS:
        digest = hashlib.sha256((static_root / filename).read_bytes()).hexdigest()
        page = page.replace(
            f'"/assets/{filename}"', f'"/assets/{filename}?v={digest[:16]}"'
        )
    return page


async def _index(request: Request) -> Response:
    page = await run_in_threadpool(_stamped_page, request.app.state.static_root)
    return HTMLResponse(page)


async def _remote(request: Request) -> Response:
    policy = request.app.state.network_access
    report = policy.report() if policy is not None else {"enabled": False, "urls": []}
    cards = []
    if policy is not None and policy.is_local(request):
        enabled = bool(report.get("enabled"))
        cards.append(
            '<button type="button" id="lan-toggle" data-enabled="'
            + str(not enabled).lower()
            + '">'
            + ("Disable" if enabled else "Enable")
            + ' LAN browser access</button><p id="lan-result" role="status"></p>'
            "<p>This change is immediate and lasts only for this app run. "
            "Control and MCP remain local-only.</p>"
        )
    if report.get("enabled"):
        import qrcode
        import qrcode.image.svg

        for base in report.get("urls", []):
            url = str(base).rstrip("/") + "/next/"
            image = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
            svg = image.to_string(encoding="unicode")
            cards.append(
                '<section class="remote-card"><h2>Open on your phone</h2>'
                f'<a class="remote-url" href="{html.escape(url, quote=True)}">'
                f'{html.escape(url)}</a><div class="remote-qr" role="img" '
                f'aria-label="QR code for {html.escape(url, quote=True)}">{svg}</div></section>'
            )
        if not report.get("urls"):
            cards.append(
                '<section class="remote-card"><h2>No home-LAN address found</h2>'
                "<p>Connect this computer to your home network, then disable and enable LAN access on this machine to refresh its addresses.</p></section>"
            )
    else:
        remedy = report.get(
            "startup_remedy",
            'Stop this app, then run: research-rag --project-root "<project>" start --lan',
        )
        cards.append(
            '<section class="remote-card"><h2>LAN access is off</h2>'
            f"<p>{html.escape(str(remedy))}</p></section>"
        )
    page = (request.app.state.static_root / "remote.html").read_text(encoding="utf-8")
    return HTMLResponse(page.replace("<!-- REMOTE_CARDS -->", "".join(cards)))


async def _asset(request: Request) -> Response:
    filename = request.path_params["filename"]
    media_type = _ASSETS.get(filename)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(request.app.state.static_root / filename, media_type=media_type)


async def _ui_profile(request: Request) -> Response:
    policy = request.app.state.network_access
    local = policy is None or policy.is_local(request)
    value: dict[str, Any] = {
        "version_label": request.app.state.version_label,
        "capabilities": dict.fromkeys(LOCAL_PANELS, local),
    }
    if not local:
        value["source_open_mode"] = "browser"
    return JSONResponse(value)


async def _health(request: Request) -> Response:
    adapter: UIAdapter = request.app.state.adapter
    try:
        adapter_health = await adapter.health()
    except UIRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if not isinstance(adapter_health, Mapping):
        raise HTTPException(
            status_code=502, detail="Adapter health returned a non-object"
        )
    return JSONResponse({"status": "ok", **dict(adapter_health)})


async def _status(request: Request) -> Response:
    return JSONResponse(await _adapter_call(request, "status"))


def _query_list(request: Request, name: str) -> list[str] | None:
    values: list[str] = []
    for raw in request.query_params.getlist(name):
        values.extend(item.strip() for item in raw.split(",") if item.strip())
    return values or None


async def _sources(request: Request) -> Response:
    names = ("categories", "categories_any", "projects", "projects_any", "keywords")
    return JSONResponse(
        await _adapter_call(
            request,
            "list_sources",
            {name: _query_list(request, name) for name in names},
        )
    )


async def _search(request: Request) -> Response:
    body = await _json_body(request)
    allowed = {
        "query",
        "top_k",
        "categories",
        "categories_any",
        "projects",
        "projects_any",
        "keywords",
        "authors_any",
        "titles_any",
        "languages_any",
        "document_ids",
        "source_ids",
        "exclude_source_ids",
    }
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported search fields: {', '.join(sorted(unknown))}",
        )
    if not isinstance(body.get("query"), str) or not body["query"].strip():
        raise HTTPException(status_code=400, detail="Search requires a non-empty query")
    return JSONResponse(await _adapter_call(request, "search", body))


async def _passage(request: Request) -> Response:
    try:
        context_chunks = int(request.query_params.get("context_chunks", "1"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="context_chunks must be an integer",
        ) from exc
    return JSONResponse(
        await _adapter_call(
            request,
            "get_passage",
            {
                "chunk_id": request.path_params["chunk_id"],
                "context_chunks": context_chunks,
            },
        )
    )


async def _ingest(request: Request) -> Response:
    body = await _json_body(request)
    unknown = set(body) - {"force_recompute"}
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ingestion fields: {', '.join(sorted(unknown))}",
        )
    if "force_recompute" in body and not isinstance(body["force_recompute"], bool):
        raise HTTPException(
            status_code=400,
            detail="force_recompute must be a boolean",
        )
    return JSONResponse(await _adapter_call(request, "ingest", body))


async def _set_metadata(request: Request) -> Response:
    body = await _json_body(request)
    if set(body) != {"source_path", "metadata"}:
        raise HTTPException(
            status_code=400,
            detail="Metadata requests require source_path and metadata",
        )
    return JSONResponse(await _adapter_call(request, "set_source_metadata", body))


async def _set_inclusion(request: Request) -> Response:
    body = await _json_body(request)
    allowed = {"source_path", "included", "reason"}
    unknown = set(body) - allowed
    if unknown or not {"source_path", "included"}.issubset(body):
        raise HTTPException(
            status_code=400,
            detail="Inclusion requests require source_path and included",
        )
    return JSONResponse(await _adapter_call(request, "set_source_inclusion", body))


async def _remove_generation(request: Request) -> Response:
    """Delete one retained generation, which is the only way a generation leaves."""

    body = await _json_body(request)
    allowed = {"generation_id", "confirm"}
    unknown = set(body) - allowed
    if unknown or not {"generation_id", "confirm"}.issubset(body):
        # The confirmation is required rather than defaulted. A host that
        # supplied its own default would be a host where a click removed a
        # generation nobody chose, which is the one thing this dialog exists to
        # prevent.
        raise HTTPException(
            status_code=400,
            detail="Generation removal requires generation_id and confirm",
        )
    if body["confirm"] != body["generation_id"]:
        raise HTTPException(
            status_code=400,
            detail="The confirmation must repeat the generation_id",
        )
    return JSONResponse(await _adapter_call(request, "remove_generation", body))


async def _use_generation(request: Request) -> Response:
    """Select one retained generation for search, which changes indexed evidence.

    The body names exactly the one generation to load. Selection is not removal:
    the generation already on disk is validated and becomes the one a search
    reads, and nothing is deleted or rebuilt.
    """

    body = await _json_body(request)
    if set(body) != {"generation_id"}:
        raise HTTPException(
            status_code=400,
            detail="Generation selection requires only generation_id",
        )
    if not isinstance(body["generation_id"], str) or not body["generation_id"].strip():
        raise HTTPException(
            status_code=400,
            detail="generation_id must be a non-empty string",
        )
    return JSONResponse(await _adapter_call(request, "use_generation", body))


async def _call_adapter(adapter: Any, method: str, *args: Any) -> Any:
    """One direct adapter call, with the adapter's own refusal kept intact.

    A host refuses a direct request for a reason a reader can act on — a session
    that is already gone, a store that is read-only, a statement it will not run
    — and translating that into a generic failure would throw the reason away.
    The translation `_adapter_call` applies is therefore applied here too.
    """

    try:
        return await getattr(adapter, method)(*args)
    except UIRequestError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail[:MAX_ERROR_LENGTH],
        ) from exc


async def _clients(request: Request) -> Response:
    """The MCP clients attached to the process serving this workspace."""

    adapter = request.app.state.adapter
    return JSONResponse({"clients": list(await _call_adapter(adapter, "list_clients"))})


async def _disconnect(request: Request) -> Response:
    """End one attached session, which is the only way a client is stopped here."""

    adapter = request.app.state.adapter
    session_id = request.path_params["session_id"]
    body = await _json_body(request)
    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise HTTPException(status_code=400, detail="A reason must be a string.")
    return JSONResponse(
        await _call_adapter(
            adapter,
            "disconnect_client",
            session_id,
            reason or "Disconnected by request.",
        )
    )


async def _source_file(request: Request) -> Response:
    raw_path = request.query_params.get("path", "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Source path is required")
    adapter: UIAdapter = request.app.state.adapter
    try:
        source: SourceFile = await adapter.source_file(raw_path)
    except UIRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if not isinstance(source, SourceFile):
        raise HTTPException(
            status_code=502, detail="Adapter returned an invalid source file"
        )
    return FileResponse(
        source.path,
        media_type=source.media_type,
        filename=source.filename,
        content_disposition_type=source.content_disposition_type,
    )


async def _settings_read(request: Request) -> Response:
    """The settings the server has, forwarded as the server reported them.

    Nothing here is named, defaulted, or dropped: which settings exist, what a
    value is, where it came from, and what changing it costs are the server's
    facts, and this repository states none of them.
    """

    return JSONResponse(await _adapter_call(request, "settings_read"))


async def _settings_write(request: Request) -> Response:
    """A settings change, forwarded verbatim so the server can refuse it.

    The body reaches the adapter exactly as the page sent it, because the keys
    are the server's setting names and the revision is its own comparison
    token; the refusal that follows is shown unchanged and never retried.
    """

    body = await _json_body(request)
    return JSONResponse(await _adapter_call(request, "settings_write", body))


def _query_number(request: Request, name: str) -> float | None:
    """A numeric query parameter, or None when absent; anything else is a 400."""

    raw = request.query_params.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc


def _present_arguments(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


async def _stats(request: Request) -> Response:
    """Search counts by rank and corpus facts, as the server reports them."""

    top = _query_number(request, "top")
    return JSONResponse(
        await _adapter_call(
            request,
            "search_stats",
            _present_arguments(
                days=_query_number(request, "days"),
                top=None if top is None else int(top),
                largest_by=request.query_params.get("largest_by") or None,
            ),
        )
    )


async def _stats_history(request: Request) -> Response:
    limit = _query_number(request, "limit")
    return JSONResponse(
        await _adapter_call(
            request,
            "search_history",
            _present_arguments(
                days=_query_number(request, "days"),
                limit=None if limit is None else int(limit),
            ),
        )
    )


async def _stats_history_clear(request: Request) -> Response:
    await _json_body(request)
    return JSONResponse(await _adapter_call(request, "clear_search_history"))


async def _source_chunks(request: Request) -> Response:
    page = _query_number(request, "page")
    size = _query_number(request, "page_size")
    return JSONResponse(
        await _adapter_call(
            request,
            "source_chunks",
            _present_arguments(
                source_id=request.query_params.get("source_id") or None,
                source_path=request.query_params.get("path") or None,
                page=None if page is None else int(page),
                page_size=None if size is None else int(size),
            ),
        )
    )


async def _open_source(request: Request) -> Response:
    """Open one source in the desktop's own viewer, forwarded to the host's decision."""

    body = await _json_body(request)
    return JSONResponse(await _adapter_call(request, "open_source", body))


async def _chunk_exclusions(request: Request) -> Response:
    """The chunks this server has excluded from retrieval."""

    return JSONResponse(await _adapter_call(request, "list_chunk_exclusions"))


async def _set_chunk_inclusion(request: Request) -> Response:
    """Exclude or restore one chunk, forwarded verbatim to the server's decision."""

    body = await _json_body(request)
    return JSONResponse(await _adapter_call(request, "set_chunk_inclusion", body))


async def _projects(request: Request) -> Response:
    """The projects this installation serves, as the adapter reported them.

    Which projects exist, which one this page serves, whether an app is up for
    one, and what address it is on are the host's facts, so the answer is
    forwarded whole: nothing is added, renamed, dropped, or defaulted here.
    """

    return JSONResponse(await _adapter_call(request, "list_projects"))


async def _agent_entry(request: Request) -> Response:
    """This project's MCP client entry, generated by the adapter that owns it.

    The text reaches the browser as the host wrote it. A second generator here
    would be a second place for a client's configuration to be wrong, so the
    entry is shown and copied unchanged.
    """

    return JSONResponse(await _adapter_call(request, "agent_entry"))


async def _updates(request: Request) -> Response:
    """Read a release preview; this route never installs or approves an update."""

    return JSONResponse(await _adapter_call(request, "check_updates"))


async def _security_headers(request: Request, call_next: Any) -> Response:
    policy = getattr(request.app.state, "network_access", None)
    if policy is not None:
        refusal = policy.request_refusal(request)
        if refusal is not None:
            return JSONResponse({"error": refusal[1]}, status_code=refusal[0])
    if write_guard.served_authority(request) is None:
        return JSONResponse(
            {"error": "Requests require a loopback Host"}, status_code=403
        )
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'; object-src 'self'; script-src 'self'; "
        "style-src 'self'; connect-src 'self'; img-src 'self' data:"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:
        # The page and its assets are revalidated on every load rather than
        # reused from a browser cache. A stale script against a newer page throws
        # before it draws, and this app is updated in place, so the two must never
        # come from different revisions. FileResponse answers the revalidation
        # with a 304 while the bytes are unchanged.
        response.headers["Cache-Control"] = "no-cache"
    return response


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Research workspace request failed", exc_info=exc)
    return JSONResponse(
        {"error": "Unexpected UI failure; inspect the UI host log."},
        status_code=500,
    )


def create_ui_app(
    *,
    adapter: UIAdapter,
    version_label: str = "",
    static_root: Path | None = None,
    network_access: Any = None,
) -> Starlette:
    """Create the local workspace app around this app's adapter."""

    routes = [
        Route("/", _index),
        Route("/next", _index),
        Route("/next/", _index),
        Route("/remote", _remote),
        Route("/remote/", _remote),
        Route("/assets/{filename:str}", _asset),
        Route("/api/ui", _ui_profile),
        Route("/api/health", _health),
        Route("/api/status", _status),
        Route("/api/sources", _sources),
        Route("/api/search", _search, methods=["POST"]),
        Route("/api/passages/{chunk_id:str}", _passage),
        Route("/api/ingest", _ingest, methods=["POST"]),
        Route("/api/source-metadata", _set_metadata, methods=["POST"]),
        Route("/api/source-inclusion", _set_inclusion, methods=["POST"]),
        Route("/api/generations/remove", _remove_generation, methods=["POST"]),
        Route("/api/generations/use", _use_generation, methods=["POST"]),
        Route("/api/clients", _clients),
        Route(
            "/api/clients/{session_id:str}/disconnect", _disconnect, methods=["POST"]
        ),
        Route("/api/source-file", _source_file),
        Route("/api/settings", _settings_read),
        Route("/api/settings", _settings_write, methods=["POST"]),
        Route("/api/chunk-exclusions", _chunk_exclusions),
        Route("/api/chunk-inclusion", _set_chunk_inclusion, methods=["POST"]),
        Route("/api/projects", _projects),
        Route("/api/agent-entry", _agent_entry),
        Route("/api/updates", _updates),
        Route("/api/stats", _stats),
        Route("/api/stats/history", _stats_history),
        Route("/api/stats/history/clear", _stats_history_clear, methods=["POST"]),
        Route("/api/source-chunks", _source_chunks),
        Route("/api/open-source", _open_source, methods=["POST"]),
    ]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(BaseHTTPMiddleware, dispatch=_security_headers)],
        exception_handlers={HTTPException: _http_error, Exception: _unexpected_error},
    )
    app.state.adapter = adapter
    app.state.version_label = version_label
    app.state.network_access = network_access
    # The compatibility guard exports also serve control routes. Only the
    # workspace app opts into this policy; control remains strictly loopback.
    app.state.workspace_network_access = network_access
    app.state.static_root = (static_root or STATIC_ROOT).resolve()
    return app
