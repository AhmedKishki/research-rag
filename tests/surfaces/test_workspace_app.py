from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from research_rag.surfaces.workspace import (
    SourceFile,
    UIRequestError,
    create_ui_app,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
# The test transport reports itself as `testclient`, and its Host is
# `testserver`; a real request arrives from a loopback peer addressed to a
# loopback name, which is what the write guard requires.
LOOPBACK_TEST_CLIENT = ("127.0.0.1", 50000)


class FakeAdapter:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def health(self) -> Mapping[str, Any]:
        return {"project_root": "/project", "source_root": "/project/sources"}

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append((operation, dict(arguments)))
        responses: dict[str, dict[str, Any]] = {
            "status": {
                "ready": True,
                "stale": False,
                "project_root": "/project",
                "generation_id": "generation-1",
                "indexed_source_count": 1,
                "searchable_source_count": 1,
                "excluded_source_count": 0,
                "chunk_count": 1,
                "hybrid_ready": True,
                "available_retrieval_methods": ["bm25", "dense", "hybrid"],
                "default_retrieval_method": "hybrid",
            },
            "list_sources": {
                "ready": True,
                "sources": [{"document_id": "doc-1", "title": "Evidence"}],
                "excluded_sources": [],
            },
            "search": {
                "query": arguments.get("query"),
                "result_count": 0,
                "hits": [],
            },
            "get_passage": {
                "requested_chunk_id": arguments.get("chunk_id"),
                "context": [],
            },
            "ingest": {"generation_id": "generation-2", "chunk_count": 2},
            "set_source_metadata": {"requires_ingest": True},
            "set_source_inclusion": {"included": arguments.get("included")},
            "remove_generation": {
                "status": "removed",
                "generation_id": arguments.get("generation_id"),
                "freed_bytes": 1024,
            },
            "use_generation": {
                "status": "changed",
                "generation_id": arguments.get("generation_id"),
                "previous_generation_id": "generation-1",
                "effective_immediately": True,
                "message": "Search now reads the selected generation.",
            },
            "settings_read": {
                "revision": "rev-1",
                "default_file": "/opt/research-rag/default.toml",
                "sections": [
                    {
                        "key": "retrieval",
                        "title": "Retrieval",
                        "settings": [
                            {
                                "key": "retrieval.rrf_k",
                                "label": "Reciprocal rank fusion k",
                                "value": 60,
                                "default": 60,
                                "defaulted": False,
                                "kind": "int",
                                "layer": "retrieval",
                                "origin": "project",
                                "writable": True,
                                "cost": {
                                    "level": "none",
                                    "message": "Applies to the next search.",
                                },
                            },
                            {
                                "key": "retrieval.model",
                                "label": "Embedding model",
                                "value": "bge-small",
                                "default": "BAAI/bge-small-en-v1.5",
                                "defaulted": True,
                                "kind": "str",
                                "layer": "retrieval",
                                "origin": "environment",
                                "writable": False,
                                "cost": {
                                    "level": "model",
                                    "message": "Re-embeds the corpus.",
                                },
                            },
                        ],
                    }
                ],
                "message": "Settings read.",
            },
            "settings_write": {
                "revision": "rev-2",
                "changed": sorted(arguments.get("values", {})),
                "requires_ingest": True,
                "message": "Settings saved.",
            },
            "list_chunk_exclusions": {
                "exclusions": [
                    {
                        "chunk_id": "chunk-9",
                        "source_relative_path": "essay.pdf",
                        "locator": "Page 12",
                        "reason": "Repeats the passage beside it.",
                        "excluded_at": "2026-10-01T09:00:00Z",
                        "in_current_generation": True,
                    }
                ],
                "message": "One chunk is excluded.",
            },
            "set_chunk_inclusion": {
                "chunk_id": arguments.get("chunk_id"),
                "included": arguments.get("included"),
                "reason": arguments.get("reason"),
            },
            "list_projects": {
                "projects": [
                    {
                        "project_name": "thesis",
                        "project_root": "/project",
                        "running": True,
                        "url": "http://127.0.0.1:5051",
                        "attached_clients": 1,
                    },
                    {
                        "project_name": "archive",
                        "project_root": "/other/archive",
                        "running": False,
                        "url": None,
                        "attached_clients": 0,
                    },
                ],
                "current": "thesis",
                "message": "",
            },
            "agent_entry": {
                "entry": (
                    '{\n  "mcp": {\n    "example": {"type": "local", '
                    '"command": ["example", "mcp"]}\n  }\n}\n'
                ),
            },
        }
        return responses[operation]

    async def source_file(self, source_path: str) -> SourceFile:
        if source_path != "evidence.pdf":
            raise UIRequestError("Source was not found", status_code=404)
        return SourceFile(
            path=self.source,
            media_type="application/pdf",
            filename=self.source.name,
        )


def test_remote_page_is_local_and_preserves_workspace(tmp_path: Path) -> None:
    app = create_ui_app(adapter=FakeAdapter(tmp_path / "source.pdf"))
    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        remote = client.get("/remote")
        assert remote.status_code == 200
        assert "LAN access is off" in remote.text
        assert "start --lan" in remote.text
        assert 'href="/next/"' in remote.text
        assert "http://" not in remote.text
        legacy = client.get("/")
        assert 'href="/remote"' in legacy.text
        assert client.get("/next/").text == legacy.text
        assert client.get("/next").text == legacy.text


def test_remote_qr_encodes_workspace_url_from_policy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from urllib.parse import urlsplit

    import qrcode

    encoded = []
    original = qrcode.make

    def capture(data: str, **kwargs: Any) -> Any:
        encoded.append(data)
        return original(data, **kwargs)

    monkeypatch.setattr(qrcode, "make", capture)

    class Policy:
        def report(self) -> dict[str, Any]:
            return {"enabled": True, "urls": ["http://192.168.1.20:5051/"]}

        def served_authority(self, request: Any) -> Any:
            return urlsplit("//" + request.headers["host"])

        def request_refusal(self, request: Any) -> None:
            return None

        def is_local(self, request: Any) -> bool:
            return False

    app = create_ui_app(
        adapter=FakeAdapter(tmp_path / "source.pdf"),
        network_access=Policy(),
    )
    with TestClient(
        app, base_url="http://192.168.1.20:5051", client=("192.168.1.30", 50000)
    ) as client:
        response = client.get("/remote")
        assert response.status_code == 200
        assert encoded == ["http://192.168.1.20:5051/next/"]
        assert "<svg" in response.text
        assert 'href="http://192.168.1.20:5051/next/"' in response.text
        assert "https://" not in response.text
        assert not client.get("/api/ui").json()["capabilities"]["clients"]
        assert not client.get("/api/ui").json()["capabilities"]["projects"]
        assert client.get("/api/ui").json()["source_open_mode"] == "browser"


def test_lan_policy_enforces_workspace_routes_writes_and_originals(
    tmp_path: Path,
) -> None:
    from research_rag.runtime.network import NetworkAccess

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7 authorized original")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        adapter=adapter,
        network_access=NetworkAccess(5051, lan=True, addresses=("192.168.1.20",)),
    )
    with TestClient(
        app, base_url="http://192.168.1.20:5051", client=("192.168.1.30", 50000)
    ) as client:
        assert client.get("/next/").status_code == 200
        for path in (
            "/api/projects",
            "/api/clients",
            "/api/agent-entry",
            "/api/health",
        ):
            assert client.get(path).status_code == 403
        assert (
            client.get("/remote", headers={"Host": "192.168.1.21:5051"}).status_code
            == 403
        )
        assert client.post("/api/search", json={"query": "evidence"}).status_code == 403
        assert (
            client.post(
                "/api/search",
                json={"query": "evidence"},
                headers={"Origin": "http://evil.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/search",
                json={"query": "evidence"},
                headers={"Origin": "http://192.168.1.20:5051"},
            ).status_code
            == 200
        )
        original = client.get("/api/source-file", params={"path": "evidence.pdf"})
        assert original.status_code == 200
        assert original.content == source.read_bytes()
        assert (
            client.get("/api/source-file", params={"path": "../secret.pdf"}).status_code
            == 404
        )


def test_policy_does_not_relax_shared_control_guard() -> None:
    from starlette.applications import Starlette
    from starlette.requests import Request

    from research_rag.runtime.network import NetworkAccess
    from research_rag.surfaces.workspace.write_guard import write_refusal

    app = Starlette()
    app.state.network_access = NetworkAccess(
        5051, lan=True, addresses=("192.168.1.20",)
    )
    request = Request(
        {
            "type": "http",
            "app": app,
            "method": "POST",
            "scheme": "http",
            "path": "/control/status",
            "query_string": b"",
            "client": ("192.168.1.30", 50000),
            "server": ("192.168.1.20", 5051),
            "headers": [
                (b"host", b"192.168.1.20:5051"),
                (b"origin", b"http://192.168.1.20:5051"),
                (b"content-type", b"application/json"),
            ],
        }
    )
    assert write_refusal(request) == (403, "Writes are refused from another host")


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    return source


def test_workspace_and_normalized_read_operations(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        page = client.get("/")
        stylesheet = client.get("/assets/app.css")
        script = client.get("/assets/app.js")
        profile = client.get("/api/ui")
        health = client.get("/api/health")
        status = client.get("/api/status")
        sources = client.get("/api/sources?categories=theory,history")
        passage = client.get("/api/passages/chunk-1?context_chunks=2")

    assert page.status_code == 200
    assert "<title>Research RAG</title>" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "applyProfile" in script.text
    assert set(profile.json()["capabilities"]) == {
        "clients",
        "projects",
        "agent_entry",
        "updates",
    }
    assert health.json()["project_root"] == "/project"
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert passage.json()["requested_chunk_id"] == "chunk-1"
    # A query string reaches the adapter as text; the operation table reads it.
    assert ("list_sources", {"categories": "theory,history"}) in adapter.calls
    assert ("get_passage", {"chunk_id": "chunk-1", "context_chunks": "2"}) in (
        adapter.calls
    )


def test_source_selection_and_partitions_are_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        adapter=adapter,
    )

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        partitions = client.get("/api/sources?categories_any=theory,history")
        projects = client.get("/api/sources?projects_any=ai-and-fetishism")
        search = client.post(
            "/api/search",
            json={
                "query": "evidence",
                "top_k": 4,
                "categories_any": ["theory"],
                "projects_any": ["ai-and-fetishism"],
                "source_ids": ["src_1"],
                "exclude_source_ids": ["src_2"],
            },
        )
        metadata = client.post(
            "/api/source-metadata",
            json={
                "source_path": "evidence.pdf",
                "metadata": {
                    "categories": ["marxism"],
                    "keywords": ["fetishism"],
                    "project": ["ai-and-fetishism"],
                },
            },
        )

    assert partitions.status_code == 200
    assert projects.status_code == 200
    assert ("list_sources", {"categories_any": "theory,history"}) in adapter.calls
    assert ("list_sources", {"projects_any": "ai-and-fetishism"}) in adapter.calls
    assert search.status_code == 200
    forwarded = next(
        arguments for operation, arguments in adapter.calls if operation == "search"
    )
    assert forwarded["categories_any"] == ["theory"]
    assert forwarded["projects_any"] == ["ai-and-fetishism"]
    assert forwarded["source_ids"] == ["src_1"]
    assert forwarded["exclude_source_ids"] == ["src_2"]
    assert metadata.status_code == 200
    assert (
        "set_source_metadata",
        {
            "source_path": "evidence.pdf",
            "metadata": {
                "categories": ["marxism"],
                "keywords": ["fetishism"],
                "project": ["ai-and-fetishism"],
            },
        },
    ) in adapter.calls


def test_writes_and_source_file_are_constrained(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        search = client.post("/api/search", json={"query": "evidence", "top_k": 5})
        metadata = client.post(
            "/api/source-metadata",
            json={"source_path": "evidence.pdf", "metadata": {"title": "Evidence"}},
        )
        inclusion = client.post(
            "/api/source-inclusion",
            json={"source_path": "evidence.pdf", "included": False},
        )
        ingestion = client.post("/api/ingest", json={})
        served = client.get("/api/source-file?path=evidence.pdf")
        missing = client.get("/api/source-file?path=missing.pdf")
        non_json = client.post("/api/search", content=b"query=test")
        cross_origin = client.post(
            "/api/search",
            headers={"Origin": "https://example.com"},
            json={"query": "test"},
        )

    assert search.status_code == 200
    assert metadata.json()["requires_ingest"] is True
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    assert served.headers["content-type"].startswith("application/pdf")
    assert missing.status_code == 404
    assert non_json.status_code == 415
    assert cross_origin.status_code == 403


def test_force_recompute_is_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        adapter=adapter,
    )

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        ingestion = client.post("/api/ingest", json={"force_recompute": True})

    assert ingestion.status_code == 200
    assert ("ingest", {"force_recompute": True}) in adapter.calls


def test_version_label_defaults_to_empty(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(adapter=FakeAdapter(source))

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        payload = client.get("/api/ui").json()

    assert payload["version_label"] == ""


def test_version_label_is_served_and_rendered(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(
        adapter=FakeAdapter(source), version_label="server 1.2.3 · ui 0.5.0"
    )

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        payload = client.get("/api/ui").json()
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert payload["version_label"] == "server 1.2.3 · ui 0.5.0"
    assert 'id="version-label"' in page.text
    assert "version_label" in script.text


def test_bibliographic_filters_are_forwarded(
    tmp_path: Path,
) -> None:
    """An author, title, or language filter reaches the adapter as sent."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        search = client.post(
            "/api/search",
            json={
                "query": "evidence",
                "top_k": 4,
                "authors_any": ["Crawford", "Gidwani"],
                "titles_any": ["Atlas of AI"],
                "languages_any": ["en", "de"],
            },
        )

    assert search.status_code == 200
    forwarded = next(
        arguments for operation, arguments in adapter.calls if operation == "search"
    )
    assert forwarded["authors_any"] == ["Crawford", "Gidwani"]
    assert forwarded["titles_any"] == ["Atlas of AI"]
    assert forwarded["languages_any"] == ["en", "de"]


def test_the_quotation_rule_is_not_a_footer(tmp_path: Path) -> None:
    """The workspace states no quotation rule beyond the cleaned-text footer.

    A rule that matters on every passage belongs in the documentation and in the
    tool descriptions that an agent reads, not in the corner of every view.
    """

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(adapter=FakeAdapter(source))

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        page = client.get("/")

    assert "not for direct quotation" not in page.text.lower()
    assert "Verify important quotations" not in page.text
    assert "Passages are cleaned text, not verified quotations." in page.text


def test_only_the_local_panels_are_gated_by_the_profile(
    tmp_path: Path,
) -> None:
    """A LAN viewer is not shown the panels that act on this machine."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(adapter=FakeAdapter(source))

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        page = client.get("/")
        javascript = client.get("/assets/app.js")

    assert page.status_code == 200
    assert javascript.status_code == 200
    gated = set(re.findall(r'data-capability="([^"]+)"', page.text))
    assert gated == {"clients", "projects", "agent_entry", "updates"}
    # The language filter and the inventory that names the languages in a
    # project are one capability: a box a reader cannot fill from the status
    # beside it is a box that asks them to know the codes already listed.
    assert 'id="language-filter"' in page.text
    assert 'id="language-chips"' in page.text
    assert 'byId("language-filter").value' in javascript.text
    assert "renderLanguages(status)" in javascript.text


def test_a_generation_is_removed_only_with_a_matching_confirmation(
    tmp_path: Path,
) -> None:
    """The dialog's typed id reaches the adapter as the reader typed it.

    A host that supplied its own confirmation would be a host where one click
    removed a generation nobody chose, which is the only thing the dialog is for.
    """

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(adapter=adapter)
    generation_id = "20260930T191235Z-45608dc5"

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        removed = client.post(
            "/api/generations/remove",
            json={"generation_id": generation_id, "confirm": generation_id},
        )
        assert removed.status_code == 200
    assert adapter.calls == [
        (
            "remove_generation",
            {"generation_id": generation_id, "confirm": generation_id},
        )
    ]


def test_a_generation_is_loaded_by_its_id(
    tmp_path: Path,
) -> None:
    """Loading a generation is a write, so it is same-origin and JSON only."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(adapter=adapter)
    generation_id = "20260930T191235Z-45608dc5"

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        loaded = client.post(
            "/api/generations/use", json={"generation_id": generation_id}
        )
        form = client.post(
            "/api/generations/use",
            content=b"generation_id=x",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        cross_origin = client.post(
            "/api/generations/use",
            headers={"Origin": "https://example.com"},
            json={"generation_id": generation_id},
        )

    assert loaded.status_code == 200
    assert loaded.json()["status"] == "changed"
    assert loaded.json()["generation_id"] == generation_id
    assert form.status_code == 415
    assert cross_origin.status_code == 403
    assert adapter.calls == [("use_generation", {"generation_id": generation_id})]


def test_the_status_view_carries_the_generation_panel_and_its_dialog() -> None:
    """The listing is free in the status payload, so the panel needs only markup."""

    app = create_ui_app(adapter=FakeAdapter(Path("evidence.pdf")))

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        page = client.get("/").text
        javascript = client.get("/assets/app.js").text

    assert 'id="generation-chips"' in page
    assert 'id="generation-dialog"' in page
    assert 'id="generation-load-dialog"' in page
    # The remove button stays disabled until the typed id matches, so the
    # confirmation is a gate rather than a message.
    assert 'id="generation-submit"' in page and "disabled" in page
    assert "renderGenerations(status.generations || [])" in javascript
    assert 'byId("generation-confirm").addEventListener' in javascript
    assert "/api/generations/remove" in javascript
    assert "/api/generations/use" in javascript
    assert 'byId("generation-load-form").addEventListener' in javascript
    # The one a search reads is not offered a removal it would only refuse.
    assert "if (!generation.is_current)" in javascript


def test_a_host_reports_the_projects_it_serves_and_its_own_client_entry(
    tmp_path: Path,
) -> None:
    """The selector and the entry are the host's own answer, forwarded whole.

    Both take no argument this repository could add: which projects exist and
    what a client's configuration says are the host's facts, so a route here that
    named one would be a second place for it to be wrong.
    """

    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        assert client.get("/api/ui").json()["capabilities"]["projects"] is True
        projects = client.get("/api/projects")
        entry = client.get("/api/agent-entry")

    assert projects.status_code == 200
    assert projects.json() == {
        "projects": [
            {
                "project_name": "thesis",
                "project_root": "/project",
                "running": True,
                "url": "http://127.0.0.1:5051",
                "attached_clients": 1,
            },
            {
                "project_name": "archive",
                "project_root": "/other/archive",
                "running": False,
                "url": None,
                "attached_clients": 0,
            },
        ],
        "current": "thesis",
        "message": "",
    }
    # The entry is the host's text, and the text is what a client is handed: a
    # wrapper, a default, or a reformat here would be a client's configuration
    # this repository had edited.
    assert entry.status_code == 200
    assert entry.json() == {
        "entry": (
            '{\n  "mcp": {\n    "example": {"type": "local", '
            '"command": ["example", "mcp"]}\n  }\n}\n'
        )
    }
    assert ("list_projects", {}) in adapter.calls
    assert ("agent_entry", {}) in adapter.calls


def test_settings_are_read_and_written_through_the_adapter(tmp_path: Path) -> None:
    """The panel reads the server's settings and sends back only what changed.

    Nothing here is named or defaulted: the settings, their origins, and the
    cost of changing them are the server's facts, so both directions travel
    exactly as they were produced.
    """

    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        read = client.get("/api/settings")
        written = client.post(
            "/api/settings",
            json={
                "values": {"retrieval.rrf_k": 40},
                "expected_revision": "rev-1",
                "confirm": True,
            },
        )

    assert read.status_code == 200
    assert read.json()["revision"] == "rev-1"
    assert [section["key"] for section in read.json()["sections"]] == ["retrieval"]
    assert read.json()["sections"][0]["settings"][1]["writable"] is False
    assert written.status_code == 200
    assert written.json() == {
        "revision": "rev-2",
        "changed": ["retrieval.rrf_k"],
        "requires_ingest": True,
        "message": "Settings saved.",
    }
    assert ("settings_read", {}) in adapter.calls
    assert (
        "settings_write",
        {
            "values": {"retrieval.rrf_k": 40},
            "expected_revision": "rev-1",
            "confirm": True,
        },
    ) in adapter.calls


def test_a_settings_write_is_same_origin_and_json_only(tmp_path: Path) -> None:
    """Changing a setting changes the server, so it is held to the write rules."""

    adapter = FakeAdapter(_source(tmp_path))
    body = {"values": {"retrieval.rrf_k": 40}, "expected_revision": "rev-1"}

    with TestClient(
        create_ui_app(adapter=adapter),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    ) as client:
        cross_origin = client.post(
            "/api/settings", headers={"Origin": "https://example.com"}, json=body
        )
        form_encoded = client.post(
            "/api/settings",
            content=b"values=%7B%7D",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    assert cross_origin.status_code == 403
    assert form_encoded.status_code == 415
    assert not any(operation == "settings_write" for operation, _ in adapter.calls)


def test_chunk_exclusions_are_listed_and_one_chunk_is_set(tmp_path: Path) -> None:
    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(adapter=adapter)

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, client=LOOPBACK_TEST_CLIENT
    ) as client:
        listed = client.get("/api/chunk-exclusions")
        excluded = client.post(
            "/api/chunk-inclusion",
            json={
                "chunk_id": "chunk-9",
                "included": False,
                "reason": "Repeats the passage beside it.",
            },
        )
        restored = client.post(
            "/api/chunk-inclusion",
            json={"chunk_id": "chunk-9", "included": True},
        )

    assert listed.status_code == 200
    assert listed.json()["exclusions"][0]["chunk_id"] == "chunk-9"
    assert excluded.json()["included"] is False
    # A restore carries no reason: the server has nothing to record for it.
    assert restored.json() == {
        "chunk_id": "chunk-9",
        "included": True,
        "reason": None,
    }
    assert ("list_chunk_exclusions", {}) in adapter.calls


def test_a_chunk_inclusion_write_is_same_origin_and_json_only(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(_source(tmp_path))
    body = {"chunk_id": "chunk-9", "included": False, "reason": "why"}
    form = {"Content-Type": "application/x-www-form-urlencoded"}

    with TestClient(
        create_ui_app(adapter=adapter),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    ) as client:
        cross_origin = client.post(
            "/api/chunk-inclusion", headers={"Origin": "https://example.com"}, json=body
        )
        form_encoded = client.post(
            "/api/chunk-inclusion", content=b"chunk_id=chunk-9", headers=form
        )

    assert cross_origin.status_code == 403
    assert form_encoded.status_code == 415
    assert not any(operation == "set_chunk_inclusion" for operation, _ in adapter.calls)
