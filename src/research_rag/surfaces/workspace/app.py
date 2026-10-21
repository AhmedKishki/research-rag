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
LOCAL_PANELS = ("clients", "agent_entry", "updates", "credentials")


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


# The routes that each forward one adapter operation. A GET carries its
# arguments in the query string and a POST in a JSON body, and a path parameter is
# an argument too. The adapter checks the arguments; this host checks only that a
# write came from this page.
_OPERATION_ROUTES = (
    ("/api/status", "GET", "status"),
    ("/api/sources", "GET", "list_sources"),
    ("/api/search", "POST", "search"),
    ("/api/answer", "POST", "answer"),
    ("/api/credentials", "GET", "credentials_read"),
    ("/api/credentials", "POST", "credentials_write"),
    ("/api/passages/{chunk_id:str}", "GET", "get_passage"),
    ("/api/ingest", "POST", "ingest"),
    ("/api/source-metadata", "POST", "set_source_metadata"),
    ("/api/source-inclusion", "POST", "set_source_inclusion"),
    ("/api/generations/remove", "POST", "remove_generation"),
    ("/api/generations/use", "POST", "use_generation"),
    ("/api/settings", "GET", "settings_read"),
    ("/api/settings", "POST", "settings_write"),
    ("/api/chunk-exclusions", "GET", "list_chunk_exclusions"),
    ("/api/chunk-inclusion", "POST", "set_chunk_inclusion"),
    ("/api/agent-entry", "GET", "agent_entry"),
    ("/api/updates", "GET", "check_updates"),
    ("/api/stats", "GET", "search_stats"),
    ("/api/stats/history", "GET", "search_history"),
    ("/api/stats/history/clear", "POST", "clear_search_history"),
    ("/api/source-chunks", "GET", "source_chunks"),
    ("/api/open-source", "POST", "open_source"),
)


def _operation_route(path: str, method: str, operation: str) -> Route:
    async def endpoint(request: Request) -> Response:
        if method == "POST":
            arguments = await _json_body(request)
        else:
            query = request.query_params
            arguments = {
                name: ",".join(query.getlist(name)) for name in query if query[name]
            }
        arguments.update(request.path_params)
        return JSONResponse(await _adapter_call(request, operation, arguments))

    return Route(path, endpoint, methods=[method])


async def _call_adapter(adapter: Any, method: str, *args: Any) -> Any:
    """One direct adapter call, with the adapter's own refusal kept intact."""

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
    source: SourceFile = await _call_adapter(
        request.app.state.adapter, "source_file", raw_path
    )
    return FileResponse(
        source.path,
        media_type=source.media_type,
        filename=source.filename,
        content_disposition_type=source.content_disposition_type,
    )


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
        *(_operation_route(*route) for route in _OPERATION_ROUTES),
        Route("/api/clients", _clients),
        Route(
            "/api/clients/{session_id:str}/disconnect", _disconnect, methods=["POST"]
        ),
        Route("/api/source-file", _source_file),
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
