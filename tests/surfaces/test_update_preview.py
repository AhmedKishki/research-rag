"""Browser update previews never grant approval or invoke an installer."""

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from research_rag.runtime import update
from research_rag.surfaces.ui import RESEARCH_UI_PROFILE, ResearchUIAdapter
from research_rag.surfaces.workspace import UICapabilities, UIProfile, create_ui_app


def preview(version: str = "1.2.0") -> dict:
    return {
        "install": {"version": "1.1.0"},
        "published": {
            "state": "found",
            "version": version,
            "tag": f"v{version}",
            "name": f"research-rag {version}",
            "published": "2026-10-05",
            "url": f"https://github.com/AhmedKishki/research-rag/releases/tag/v{version}",
            "body": "# Changes\n- Readable interface.\n<script>unsafe()</script>",
        },
        "blocked": None,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "version,available", [("1.2.0", True), ("1.1.0", False), ("1.0.0", False)]
)
async def test_release_preview_is_bounded_and_cached(monkeypatch, version, available):
    calls = []

    def read_preview(*, offline):
        calls.append(offline)
        return preview(version)

    monkeypatch.setattr(update, "read_preview", read_preview)
    adapter = ResearchUIAdapter(SimpleNamespace(offline=False), None)
    first = await adapter.call("check_updates", {"apply": True, "confirm": True})
    second = await adapter.call("check_updates", {})
    assert calls == [False]
    assert first == second
    assert first["update_available"] is available
    assert first["release"]["body"].endswith("<script>unsafe()</script>")
    assert set(first) == {
        "installed_version",
        "update_available",
        "offline",
        "release",
        "apply_command",
        "message",
    }
    assert "would_run" not in first
    assert "approval" not in first


@pytest.mark.anyio
async def test_offline_preview_does_not_claim_up_to_date(monkeypatch):
    def read_preview(*, offline):
        assert offline is True
        return {
            "install": {"version": "1.1.0"},
            "published": {"state": "skipped", "detail": "Offline: no network check."},
        }

    monkeypatch.setattr(update, "read_preview", read_preview)
    adapter = ResearchUIAdapter(SimpleNamespace(offline=True), None)
    answer = await adapter.call("check_updates", {})
    assert answer["offline"] is True
    assert answer["release"] is None
    assert answer["update_available"] is False
    assert "Offline" in answer["message"]


def test_updates_route_is_read_only_and_capability_gated(monkeypatch):
    monkeypatch.setattr(update, "read_preview", lambda **_: preview())
    adapter = ResearchUIAdapter(SimpleNamespace(offline=False), None)
    with TestClient(
        create_ui_app(profile=RESEARCH_UI_PROFILE, adapter=adapter),
        base_url="http://127.0.0.1",
    ) as client:
        assert client.get("/api/updates").json()["update_available"] is True
        assert client.post("/api/updates", json={"apply": True}).status_code == 405
        assert (
            client.get("/api/updates", headers={"Host": "attacker.example"}).status_code
            == 403
        )

    disabled = UIProfile(
        application_name="No updater", capabilities=UICapabilities(updates=False)
    )
    with TestClient(
        create_ui_app(profile=disabled, adapter=adapter), base_url="http://127.0.0.1"
    ) as client:
        assert client.get("/api/updates").status_code == 404
