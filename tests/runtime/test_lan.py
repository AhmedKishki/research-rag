"""Opt-in interface authorities, unspoofed peers, and browser write boundaries."""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

from research_rag.project.config import resolve_config
from research_rag.runtime.app import App
from research_rag.runtime.network import (
    LOCAL_ONLY_PATHS,
    NetworkAccess,
    discover_lan_addresses,
)


def request(
    *,
    peer: str = "192.168.1.8",
    host: str = "192.168.1.2:8765",
    path: str = "/api/search",
    headers: dict[str, str] | None = None,
) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "scheme": "http",
            "server": ("0.0.0.0", 8765),
            "client": (peer, 50000),
            "headers": [
                (key.encode(), value.encode())
                for key, value in {"host": host, **(headers or {})}.items()
            ],
            "query_string": b"",
        }
    )


@pytest.fixture
def access() -> NetworkAccess:
    return NetworkAccess(8765, lan=True, addresses=("192.168.1.2",))


@pytest.mark.parametrize(
    "host",
    [
        "192.168.1.3:8765",
        "10.0.0.1:8765",
        "8.8.8.8:8765",
        "attacker.example:8765",
        "192.168.1.2:1234",
        "192.168.1.2",
        "user@192.168.1.2:8765",
        "192.168.1.2:8765/path",
        "[::1",
        "0.0.0.0:8765",
    ],
)
def test_only_actual_interface_and_serving_port_are_authorities(
    access: NetworkAccess, host: str
) -> None:
    assert access.request_refusal(request(host=host))[0] == 403


@pytest.mark.parametrize(
    "peer",
    [
        "8.8.8.8",
        "169.254.1.1",
        "100.64.0.1",
        "::ffff:192.168.1.8",
        "fd00::1",
        "testclient",
    ],
)
def test_remote_peer_must_be_rfc1918_ipv4(access: NetworkAccess, peer: str) -> None:
    assert (
        access.request_refusal(
            request(peer=peer, headers={"x-forwarded-for": "127.0.0.1"})
        )[0]
        == 403
    )


def test_default_never_accepts_lan(access: NetworkAccess) -> None:
    assert NetworkAccess(8765).request_refusal(request())[0] == 403
    assert access.request_refusal(request()) is None
    assert access.request_refusal(request(peer="10.0.0.8")) is None


@pytest.mark.parametrize("prefix", LOCAL_ONLY_PATHS)
def test_local_only_paths_cannot_be_reached_by_remote_peers(
    access: NetworkAccess, prefix: str
) -> None:
    for path in (prefix, prefix + "/anything"):
        assert access.request_refusal(request(path=path))[0] == 403
        assert (
            access.request_refusal(
                request(peer="127.0.0.1", host="127.0.0.1:8765", path=path)
            )
            is None
        )


@pytest.mark.parametrize(
    "headers,code",
    [
        ({"content-type": "application/json"}, 403),
        ({"origin": "null", "content-type": "application/json"}, 403),
        (
            {"origin": "http://192.168.1.2:1234", "content-type": "application/json"},
            403,
        ),
        (
            {"origin": "https://192.168.1.2:8765", "content-type": "application/json"},
            403,
        ),
        (
            {
                "origin": "http://192.168.1.2:8765/path",
                "content-type": "application/json",
            },
            403,
        ),
        ({"origin": "http://[::1", "content-type": "application/json"}, 403),
        (
            {
                "origin": "http://192.168.1.2:8765",
                "sec-fetch-site": "cross-site",
                "content-type": "application/json",
            },
            403,
        ),
        ({"origin": "http://192.168.1.2:8765", "content-type": "text/plain"}, 415),
    ],
)
def test_remote_browser_write_refusals(
    access: NetworkAccess, headers: dict[str, str], code: int
) -> None:
    assert access.write_refusal(request(headers=headers))[0] == code


def test_exact_origin_json_and_local_terminal_writes(access: NetworkAccess) -> None:
    assert (
        access.write_refusal(
            request(
                headers={
                    "origin": "http://192.168.1.2:8765",
                    "content-type": "application/json; charset=utf-8",
                }
            )
        )
        is None
    )
    assert (
        access.write_refusal(
            request(
                peer="127.0.0.1",
                host="127.0.0.1:8765",
                headers={"content-type": "application/json"},
            )
        )
        is None
    )
    assert access.request_refusal(request(host="localhost:8765"))[0] == 403


def test_duplicate_host_and_origin_are_rejected(access: NetworkAccess) -> None:
    req = request(
        headers={
            "origin": "http://192.168.1.2:8765",
            "content-type": "application/json",
        }
    )
    req.scope["headers"].append((b"origin", b"http://192.168.1.2:8765"))
    assert access.write_refusal(req)[0] == 403
    req = request()
    req.scope["headers"].append((b"host", b"192.168.1.2:8765"))
    assert access.served_authority(req) is None


def test_interface_discovery_filters_assigned_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    monkeypatch.setattr(
        socket,
        "if_nameindex",
        lambda: [(1, "lo"), (2, "eth0"), (3, "public"), (4, "unassigned")],
    )

    def ioctl(fd: int, operation: int, value: bytes) -> bytes:
        assert operation == 0x8915
        name = value.split(b"\0", 1)[0]
        address = {
            b"lo": "127.0.0.1",
            b"eth0": "192.168.1.2",
            b"public": "8.8.8.8",
        }.get(name)
        if address is None:
            raise OSError("no address")
        return bytes(20) + socket.inet_aton(address) + bytes(232)

    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    assert discover_lan_addresses() == ("192.168.1.2",)


def test_lan_metadata_is_ephemeral_and_loopback_url_stays_local(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "research_rag.runtime.app.discover_lan_addresses", lambda: ("192.168.1.2",)
    )
    config = resolve_config(project)
    app = App(config, port=8765, lan=True)
    assert app.host == "0.0.0.0"
    assert app.url == "http://127.0.0.1:8765"
    assert app.mcp_url == app.url + "/mcp"
    assert app.state()["lan"]["urls"] == ["http://192.168.1.2:8765"]
    assert App(config, port=8765).host == "127.0.0.1"
    assert NetworkAccess(8765).report()["urls"] == []


@pytest.mark.anyio
async def test_outer_runtime_gate_blocks_control_and_mcp_before_dispatch(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "research_rag.runtime.app.discover_lan_addresses", lambda: ("192.168.1.2",)
    )
    app = App(resolve_config(project), port=8765, lan=True)
    transport = httpx.ASGITransport(app=app.build(), client=("192.168.1.8", 50000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.2:8765"
    ) as client:
        for prefix in LOCAL_ONLY_PATHS:
            response = await client.get(
                prefix, headers={"x-forwarded-for": "127.0.0.1"}
            )
            assert response.status_code == 403, prefix
        response = await client.post("/api/search", json={"query": "heron"})
        assert response.status_code == 403
        response = await client.post("/control/lan", json={"enabled": False})
        assert response.status_code == 403
        page = await client.get("/remote")
        assert 'id="lan-toggle"' not in page.text
        app.network_access.lan = False
        assert (await client.get("/remote")).status_code == 403
    assert app.clients.report() == []


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_live_lan_listener_and_rollback(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from research_rag.project.policy import ResearchError
    from research_rag.runtime import app as runtime

    monkeypatch.setattr(runtime, "discover_lan_addresses", lambda: ("192.168.1.2",))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = App(resolve_config(project), port=port)
    policy = app.network_access
    await app.start()
    try:
        for _ in range(200):
            if app.ready:
                break
            await asyncio.sleep(0.01)
        assert app.ready
        task, server, gateway = app._task, app._server, app.gateway
        async with httpx.AsyncClient(base_url=app.url) as client:
            page = await client.get("/remote")
            assert "Enable LAN browser access" in page.text
            assert '<script src="/assets/remote.js"></script>' in page.text
            assert "script-src 'self'" in page.headers["content-security-policy"]
            asset = await client.get("/assets/remote.js")
            assert asset.status_code == 200
            assert "fetch('/control/lan'" in asset.text
            response = await client.post("/control/lan", json={"enabled": True})
            assert response.status_code == 200, response.text
            assert response.json()["enabled"] is True
            assert app.network_access is policy
            assert app._socket.getsockname() == ("0.0.0.0", port)
            async with httpx.AsyncClient(base_url=app.url) as fresh:
                assert (await fresh.get("/control/health")).status_code == 200
            assert "Disable LAN browser access" in (await client.get("/remote")).text
            assert (
                await client.post("/control/lan", json={"enabled": "false"})
            ).status_code == 400
            assert (
                await client.post(
                    "/control/lan",
                    json={"enabled": False},
                    headers={"Origin": "http://evil.example"},
                )
            ).status_code == 403
            assert (
                await client.post("/control/lan", json={"enabled": False})
            ).status_code == 200
            assert app._socket.getsockname() == ("127.0.0.1", port)
            async with httpx.AsyncClient(base_url=app.url) as fresh:
                assert (await fresh.get("/control/health")).status_code == 200
            assert policy.lan is False
            claim = runtime._claim_loopback_port

            def fail_lan(host: str, port: int) -> socket.socket:
                if host == "0.0.0.0":
                    raise OSError("occupied")
                return claim(host, port)

            monkeypatch.setattr(runtime, "_claim_loopback_port", fail_lan)
            with pytest.raises(ResearchError, match="previous listener restored"):
                await app.set_lan(True)
            assert (await client.get("/control/health")).status_code == 200
            assert not policy.lan
        assert (app._task, app._server, app.gateway) == (task, server, gateway)
    finally:
        await app.stop()
