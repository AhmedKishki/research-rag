"""The browser workspace: the shared UI over one in-process research service.

The shared UI sends its full optional argument set for every route, so most of
these tests are about what the adapter forwards: an argument this app does not
serve must not reach the service as a value the service would ignore.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_rag.config import (
    MANAGED_CHILD_ENV,
    TOP_LEVEL_ONLY_ENV,
    child_process_environment,
    resolve_config,
)
from research_rag.ui import Workspace, create_ui_app
from research_rag.ultrarag import create_vanilla_transport


class FakeResearchService:
    """Stand in for `ResearchService`, recording every call it is asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> None:
        self.calls.append((name, arguments))

    async def status(self) -> dict[str, Any]:
        self._record("status", {})
        return {
            "ready": True,
            "stale": False,
            "project_root": "/research/project",
            "project_name": "Research project",
            "source_root": "/research/project/sources",
            "generation_id": "generation-1",
            "created_at": "2026-09-17T12:00:00Z",
            "selected_source_count": 1,
            "indexed_source_count": 1,
            "searchable_source_count": 1,
            "excluded_source_count": 0,
            "chunk_count": 1,
            "hybrid_ready": True,
            "generation_upgrade_required": False,
            "upgrade_reasons": [],
            "available_retrieval_methods": ["bm25", "dense", "hybrid"],
            "default_retrieval_method": "hybrid",
        }

    async def list_sources(self) -> dict[str, Any]:
        self._record("list_sources", {})
        return {
            "ready": True,
            "source_count": 1,
            "sources": [
                {
                    "document_id": "doc-1",
                    "source_path": "sources/evidence.pdf",
                    "source_relative_path": "evidence.pdf",
                    "format": "pdf",
                    "title": "Evidence",
                    "authors": ["Researcher"],
                    "year": 2026,
                    "doi": "",
                    "categories": ["theory"],
                    "keywords": ["evidence"],
                }
            ],
            "excluded_source_count": 0,
            "excluded_sources": [],
        }

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        self._record("search", {"query": query, **arguments})
        return {
            "query": query,
            "generation_id": "generation-1",
            "stale": False,
            "retrieval_method": arguments["retrieval_method"],
            "reranked": arguments["rerank"],
            "result_count": 1,
            "hits": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "title": "Evidence",
                    "authors": ["Researcher"],
                    "year": 2026,
                    "source_path": "sources/evidence.pdf",
                    "categories": ["theory"],
                    "keywords": ["evidence"],
                    "locator": {"type": "pdf_page", "page": 1, "page_label": "1"},
                    "citation": "Researcher, Evidence (2026), p. 1",
                    "text": "A cleaned semantic passage.",
                    "direct_quote_safe": False,
                    "component_ranks": {"bm25": 1, "dense": 1},
                    "component_scores": {
                        "dense_cosine_similarity": 0.8,
                        "bm25": None,
                    },
                    "fusion_score": 0.03,
                    "rerank_score": None,
                }
            ],
        }

    async def get_passage(
        self, chunk_id: str, *, context_chunks: int = 1
    ) -> dict[str, Any]:
        self._record(
            "get_passage", {"chunk_id": chunk_id, "context_chunks": context_chunks}
        )
        return {
            "generation_id": "generation-1",
            "requested_chunk_id": chunk_id,
            "context": [
                {
                    "chunk_id": chunk_id,
                    "document_id": "doc-1",
                    "source_path": "sources/evidence.pdf",
                    "title": "Evidence",
                    "locator": {"type": "pdf_page", "page": 1},
                    "citation": "Researcher, Evidence (2026), p. 1",
                    "text": "A cleaned semantic passage.",
                }
            ],
        }

    async def ingest(self, *, force_recompute: bool = False) -> dict[str, Any]:
        self._record("ingest", {"force_recompute": force_recompute})
        return {"status": "ready", "generation_id": "generation-2", "chunk_count": 2}

    async def set_source_inclusion(self, **arguments: Any) -> dict[str, Any]:
        self._record("set_source_inclusion", arguments)
        return {
            "source_relative_path": arguments.get("source_path"),
            "included": arguments.get("included"),
            "message": "Inclusion saved.",
        }

    async def set_source_metadata(self, **arguments: Any) -> dict[str, Any]:
        self._record("set_source_metadata", arguments)
        return {
            "source_relative_path": arguments.get("source_path"),
            "metadata": arguments.get("metadata"),
            "message": "Reviewed metadata saved.",
        }


def _client(project: Path) -> tuple[TestClient, FakeResearchService]:
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = FakeResearchService()
    return TestClient(create_ui_app(config, service=fake)), fake


def test_the_workspace_serves_the_page_and_read_apis(project: Path) -> None:
    client, fake = _client(project)
    with client:
        page = client.get("/")
        assert page.status_code == 200
        assert "UltraRAG MCP" in page.text
        assert "Evidence, with its provenance intact" not in page.text
        assert page.text.index('id="search-view"') < page.text.index(
            'id="status-cards"'
        )
        assert "default-src 'self'" in page.headers["content-security-policy"]

        css = client.get("/assets/app.css")
        javascript = client.get("/assets/app.js")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert javascript.status_code == 200
        assert "loadWorkspace" in javascript.text
        assert client.get("/assets/unknown.js").status_code == 404

        health = client.get("/api/health")
        profile = client.get("/api/ui")
        status = client.get("/api/status")
        sources = client.get("/api/sources?categories=theory,history")
        context = client.get("/api/passages/chunk-1?context_chunks=2")

    assert health.json()["project_root"] == str(project)
    assert profile.json()["application_name"] == "Research RAG"
    assert profile.json()["capabilities"]["metadata"] is True
    assert profile.json()["capabilities"]["bundle_export"] is False
    assert profile.json()["capabilities"]["bundle_import"] is False
    assert profile.json()["capabilities"]["force_recompute"] is True
    assert profile.json()["capabilities"]["source_selection"] is True
    assert profile.json()["capabilities"]["category_partitions"] is True
    assert profile.json()["capabilities"]["project_metadata"] is True
    assert profile.json()["capabilities"]["metadata_filters"] is True
    assert profile.json()["capabilities"]["bibliographic_filters"] is True
    assert profile.json()["capabilities"]["retrieval_modes"] is False
    assert profile.json()["capabilities"]["reranking"] is False
    assert profile.json()["capabilities"]["chunk_settings"] is False
    # The adapter leaves the shared UI's neutral result label in place: the quote
    # rule is stated once in the README, not on every passage a browser renders.
    assert "not for direct quotation" not in json.dumps(profile.json()).lower()
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert context.json()["requested_chunk_id"] == "chunk-1"
    assert ("status", {}) in fake.calls
    # The source inventory takes no filter, so the route's filter arguments are
    # dropped rather than forwarded as values nothing would act on.
    assert ("list_sources", {}) in fake.calls
    assert ("get_passage", {"chunk_id": "chunk-1", "context_chunks": 2}) in fake.calls


def test_the_workspace_forwards_search_and_the_surviving_mutations(
    project: Path,
) -> None:
    client, fake = _client(project)
    with client:
        search = client.post(
            "/api/search",
            json={"query": "research question", "top_k": 5},
        )
        narrowed = client.post(
            "/api/search",
            json={
                "query": "research question",
                "top_k": 3,
                "categories_any": ["Commodity fetishism"],
                "source_ids": ["src_1"],
                "exclude_source_ids": ["src_2"],
            },
        )
        inclusion = client.post(
            "/api/source-inclusion",
            json={
                "source_path": "evidence.pdf",
                "included": False,
                "reason": "Reviewed duplicate",
            },
        )
        ingestion = client.post("/api/ingest", json={"force_recompute": True})
        saved_metadata = client.post(
            "/api/source-metadata",
            json={
                "source_path": "evidence.pdf",
                "metadata": {"categories": ["theory"]},
            },
        )
        retired_export = client.post("/api/bundles/export", json={})
        retired_import = client.post(
            "/api/bundles/import",
            json={
                "bundle_name": "research-generation.research-rag.zip",
                "activate": True,
            },
        )

    assert search.json()["hits"][0]["citation"].endswith("p. 1")
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    assert saved_metadata.status_code == 200
    assert saved_metadata.json()["message"] == "Reviewed metadata saved."
    assert retired_export.status_code == 404
    assert retired_import.status_code == 404
    assert narrowed.json()["hits"][0]["citation"].endswith("p. 1")
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 5,
            "categories_any": None,
            "projects_any": None,
            "keywords": None,
            "source_ids": None,
            "exclude_source_ids": None,
            "retrieval_method": "hybrid",
            "rerank": True,
            "include_staleness": True,
        },
    ) in fake.calls
    assert (
        "set_source_metadata",
        {
            "source_path": "evidence.pdf",
            "metadata": {"categories": ["theory"]},
        },
    ) in fake.calls
    assert (
        "set_source_inclusion",
        {
            "source_path": "evidence.pdf",
            "included": False,
            "reason": "Reviewed duplicate",
        },
    ) in fake.calls
    assert ("ingest", {"force_recompute": True}) in fake.calls


def test_a_search_is_always_hybrid_and_always_reranked(project: Path) -> None:
    """The profile hides both switches, so the workspace fixes them itself."""

    client, fake = _client(project)
    with client:
        rejected_mode = client.post(
            "/api/search",
            json={"query": "q", "retrieval_method": "bm25"},
        )
        rejected_rerank = client.post(
            "/api/search", json={"query": "q", "rerank": True}
        )
        declined_rerank = client.post(
            "/api/search", json={"query": "q", "rerank": False}
        )
        accepted = client.post("/api/search", json={"query": "q"})

    assert rejected_mode.status_code == 400
    assert rejected_rerank.status_code == 400
    assert accepted.status_code == 200
    # A request that asks not to rerank is dropped rather than obeyed: reranking
    # is not optional, and the answer always says whether it happened.
    assert declined_rerank.status_code == 200
    assert all(
        arguments["retrieval_method"] == "hybrid"
        for name, arguments in fake.calls
        if name == "search"
    )
    assert all(
        arguments["rerank"] is True
        for name, arguments in fake.calls
        if name == "search"
    )


def test_an_unserved_search_argument_never_reaches_the_service(
    project: Path,
) -> None:
    """A control the workspace hides must not travel as an ignored argument."""

    from research_rag.ui import _OPERATION_ARGUMENTS, ResearchUIAdapter

    adapter = ResearchUIAdapter(resolve_config(project), FakeResearchService())  # type: ignore[arg-type]

    assert adapter._arguments(
        "search",
        {
            "query": "q",
            "authors_any": ["Crawford"],
            "titles_any": ["Atlas of AI"],
            "categories": ["theory"],
            "retrieval_method": "bm25",
            "rerank": False,
            "top_k": 5,
        },
    ) == {"query": "q", "top_k": 5}
    assert set(_OPERATION_ARGUMENTS) == {
        "status",
        "ingest",
        "search",
        "list_sources",
        "get_passage",
        "set_source_inclusion",
        "set_source_metadata",
    }


def test_an_unknown_operation_is_refused(project: Path) -> None:
    from research_rag.ui import ResearchUIAdapter, UIRequestError

    adapter = ResearchUIAdapter(resolve_config(project), FakeResearchService())  # type: ignore[arg-type]

    with pytest.raises(UIRequestError) as refusal:
        asyncio.run(adapter.call("export_bundle", {}))
    assert refusal.value.status_code == 404


def test_the_workspace_rejects_unsafe_writes_and_source_paths(project: Path) -> None:
    source = project / "sources" / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    client, _fake = _client(project)
    with client:
        served = client.get("/api/source-file?path=evidence.pdf")
        traversal = client.get("/api/source-file?path=../secret.pdf")
        non_json = client.post("/api/search", content=b"query=test")
        cross_origin = client.post(
            "/api/search",
            headers={"Origin": "https://example.com"},
            json={"query": "test"},
        )
        unknown = client.post(
            "/api/search",
            json={"query": "test", "unsupported": True},
        )

    assert served.status_code == 200
    assert served.headers["content-type"].startswith("application/pdf")
    assert traversal.status_code == 400
    assert non_json.status_code == 415
    assert cross_origin.status_code == 403
    assert unknown.status_code == 400


def test_a_service_failure_becomes_a_safe_message(project: Path) -> None:
    class Failing(FakeResearchService):
        async def status(self) -> dict[str, Any]:
            raise RuntimeError("the corpus is on a slow disk")

    config = resolve_config(project, vanilla_executable=sys.executable)
    with TestClient(create_ui_app(config, service=Failing())) as client:  # type: ignore[arg-type]
        response = client.get("/api/status")

    assert response.status_code == 400
    assert "the corpus is on a slow disk" in response.json()["error"]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def _wait_until_ready(workspace: Workspace) -> None:
    for _ in range(300):
        if workspace.ready:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the workspace never became ready: {workspace.error}")


def _get_json(url_host: str, port: int, path: str) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(url_host, port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


async def _assert_the_workspace_serves_and_stops(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    port = _free_loopback_port()
    workspace = Workspace(config, port=port)

    assert workspace.url == f"http://127.0.0.1:{port}"
    assert workspace.ready is False
    assert workspace.error is None

    await workspace.start(service=FakeResearchService())  # type: ignore[arg-type]
    try:
        await _wait_until_ready(workspace)
        # The HTTP call must not run on this loop: the hosted server shares it,
        # so a blocking request here would deadlock the very server it calls.
        status_code, payload = await asyncio.to_thread(
            _get_json, workspace.host, port, "/api/health"
        )
        assert status_code == 200
        assert payload["status"] == "ok"
    finally:
        await workspace.stop()

    assert workspace.ready is False


def test_the_workspace_serves_loopback_and_stops(project: Path) -> None:
    asyncio.run(_assert_the_workspace_serves_and_stops(project))


async def _assert_the_workspace_reports_a_used_port(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = int(busy.getsockname()[1])
        workspace = Workspace(config, port=port)
        await workspace.start(service=FakeResearchService())  # type: ignore[arg-type]

    # A taken port is reported, never raised: the process keeps serving.
    assert workspace.ready is False
    assert workspace.error is not None
    assert str(port) in workspace.error


def test_the_workspace_reports_a_used_port(project: Path) -> None:
    asyncio.run(_assert_the_workspace_reports_a_used_port(project))


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_a_port_outside_the_range_is_refused(project: Path, port: int) -> None:
    from research_rag.config import ConfigurationError

    with pytest.raises(ConfigurationError):
        Workspace(resolve_config(project), port=port)


def test_child_environment_is_marked_and_drops_top_level_settings(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_UI_PORT", "5151")
    monkeypatch.delenv(MANAGED_CHILD_ENV, raising=False)

    environment = child_process_environment()

    for name in TOP_LEVEL_ONLY_ENV:
        assert name not in environment
    assert environment[MANAGED_CHILD_ENV] == "1"
    assert environment["PATH"] == os.environ["PATH"]


def test_the_gateway_transport_carries_the_marker(project: Path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_UI_PORT", "5151")
    monkeypatch.delenv(MANAGED_CHILD_ENV, raising=False)
    config = resolve_config(project, vanilla_executable=sys.executable)

    transport = create_vanilla_transport(config)

    assert transport.env is not None
    for name in TOP_LEVEL_ONLY_ENV:
        assert name not in transport.env
    assert transport.env[MANAGED_CHILD_ENV] == "1"
    assert transport.env["PATH"] == os.environ["PATH"]


async def _assert_the_port_claim_is_exclusive_and_released(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    port = _free_loopback_port()
    first = Workspace(config, port=port)
    second = Workspace(config, port=port)

    await first.start(service=FakeResearchService())  # type: ignore[arg-type]
    try:
        await _wait_until_ready(first)
        # The claim is exclusive because it listens: a socket that is only bound
        # can still be bound again under SO_REUSEADDR, which is the trap a plain
        # probe-before-bind falls into.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))

        await second.start(service=FakeResearchService())  # type: ignore[arg-type]

        # The claim is the bind and the listen, so the loser of a simultaneous
        # start fails before it serves anything and says why instead of reporting
        # nothing: the message is the claim's, not a downstream uvicorn failure.
        assert second.ready is False
        assert second.error is not None
        assert "already in use" in second.error
        assert str(port) in second.error
    finally:
        await first.stop()

    # The verdict is not cached: a released port is claimable again.
    third = Workspace(config, port=port)
    await third.start(service=FakeResearchService())  # type: ignore[arg-type]
    try:
        await _wait_until_ready(third)
        assert third.ready is True
        assert third.error is None
    finally:
        await third.stop()


def test_the_port_claim_is_exclusive_and_released(project: Path) -> None:
    asyncio.run(_assert_the_port_claim_is_exclusive_and_released(project))
