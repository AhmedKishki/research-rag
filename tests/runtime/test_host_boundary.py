"""Reject rebound hosts before reads reach any application or workspace route."""

from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from research_rag.runtime.app import _security_headers as app_guard
from research_rag.surfaces.workspace.app import _security_headers as workspace_guard


@pytest.mark.parametrize("guard", [app_guard, workspace_guard])
@pytest.mark.parametrize("path", ["/", "/api/status", "/control/describe", "/mcp"])
@pytest.mark.parametrize("host", ["attacker.example:5051", "[broken", "localhost:bad"])
def test_untrusted_hosts_cannot_read_any_route(
    guard: Any, path: str, host: str
) -> None:
    calls: list[str] = []

    async def read(request: Request) -> JSONResponse:
        calls.append(request.url.path)
        return JSONResponse({"private": "not for a rebound page"})

    app = Starlette(
        routes=[Route(path, read)],
        middleware=[Middleware(BaseHTTPMiddleware, dispatch=guard)],
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(path, headers={"Host": host, "Origin": f"http://{host}"})
    assert response.status_code == 403
    assert calls == []


@pytest.mark.parametrize("guard", [app_guard, workspace_guard])
def test_loopback_read_still_reaches_the_route(guard: Any) -> None:
    async def read(_: Request) -> JSONResponse:
        return JSONResponse({"ready": True})

    app = Starlette(
        routes=[Route("/api/status", read)],
        middleware=[Middleware(BaseHTTPMiddleware, dispatch=guard)],
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json() == {"ready": True}
