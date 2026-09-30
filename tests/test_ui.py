"""The browser workspace: the shared UI over the app's one service.

The shared UI sends its full optional argument set for every route, so most of
these tests are about what the adapter forwards: an argument this app does not
serve must not reach the service as a value the service would ignore.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_rag.app import ClientRegistry
from research_rag.config import (
    resolve_config,
)
from research_rag.surfaces.ui import RESEARCH_UI_PROFILE, create_ui_app


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
        filtered = client.post(
            "/api/search",
            json={
                "query": "research question",
                "languages_any": ["en"],
                "authors_any": ["Crawford"],
                "titles_any": ["Atlas of AI"],
                "keywords": ["desire", "labour"],
                "projects_any": ["thesis"],
            },
        )
        assert filtered.status_code == 200
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
            "languages_any": None,
            "authors_any": None,
            "titles_any": None,
            "source_ids": None,
            "exclude_source_ids": None,
            "retrieval_method": "hybrid",
            "rerank": True,
            "include_staleness": True,
        },
    ) in fake.calls
    # Every reviewed-metadata layer the service offers reaches it from the
    # browser. A layer that worked in the terminal and was dropped here was a
    # filter a reader could type but not click, which is the disagreement the
    # adapter's argument set exists to prevent.
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 10,
            "categories_any": None,
            "projects_any": ["thesis"],
            "keywords": ["desire", "labour"],
            "languages_any": ["en"],
            "authors_any": ["Crawford"],
            "titles_any": ["Atlas of AI"],
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

    from research_rag.surfaces.ui import _OPERATION_ARGUMENTS, ResearchUIAdapter

    adapter = ResearchUIAdapter(resolve_config(project), FakeResearchService())  # type: ignore[arg-type]

    # The retrieval switches are the app's engine decisions and the workspace
    # hides them, so they are dropped rather than forwarded as arguments the
    # app would ignore.
    assert adapter._arguments(
        "search",
        {
            "query": "q",
            "categories": ["theory"],
            "retrieval_method": "bm25",
            "rerank": False,
            "top_k": 5,
        },
    ) == {"query": "q", "top_k": 5}
    assert adapter._arguments(
        "search",
        {
            "query": "q",
            "authors_any": ["Crawford"],
            "titles_any": ["Atlas of AI"],
            "languages_any": ["en"],
        },
    ) == {
        "query": "q",
        "authors_any": ["Crawford"],
        "titles_any": ["Atlas of AI"],
        "languages_any": ["en"],
    }
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
    from research_rag.surfaces.ui import ResearchUIAdapter, UIRequestError

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


def _registry(*, attached: bool = True) -> ClientRegistry:
    """A real registry with one session in it, seeded the way the app seeds one."""

    registry = ClientRegistry()
    if attached:
        registry._touch("s-1", "reader-agent")
        registry._touch("s-1", "reader-agent")
    return registry


def _client_client(project: Path, registry: ClientRegistry | None) -> TestClient:
    config = resolve_config(project, vanilla_executable=sys.executable)
    return TestClient(
        create_ui_app(
            config,
            service=FakeResearchService(),  # type: ignore[arg-type]
            clients=registry,
        )
    )


def test_the_workspace_lists_the_clients_attached_to_the_app(project: Path) -> None:
    """A person in the browser sees what the command line sees."""

    with _client_client(project, _registry()) as client:
        response = client.get("/api/clients")
    assert response.status_code == 200
    entry = response.json()["clients"][0]
    assert entry["name"] == "reader-agent"
    assert entry["attached"] is True
    assert entry["requests"] == 2


def test_the_workspace_can_end_one_client(project: Path) -> None:
    registry = _registry()
    with _client_client(project, registry) as client:
        response = client.post(
            "/api/clients/s-1/disconnect", json={"reason": "from the workspace"}
        )
    assert response.status_code == 200
    assert response.json()["attached"] is False
    assert response.json()["detached_reason"] == "from the workspace"
    # The same registry the command line reads, so a drop from the browser is a
    # drop the command line can see.
    assert registry.report()[0]["attached"] is False


def test_the_workspace_names_a_session_it_does_not_have(project: Path) -> None:
    with _client_client(project, _registry()) as client:
        response = client.post("/api/clients/nope/disconnect", json={})
    assert response.status_code == 400
    assert "nope" in response.json()["error"]


def test_the_clients_panel_is_offered_because_the_app_is_a_server(
    project: Path,
) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    with TestClient(create_ui_app(config, service=FakeResearchService())) as client:  # type: ignore[arg-type]
        profile = client.get("/api/ui").json()
    assert profile["capabilities"]["clients"] is True
    assert RESEARCH_UI_PROFILE.capabilities.clients is True


def test_a_workspace_with_no_process_behind_it_refuses_rather_than_lying(
    project: Path,
) -> None:
    """A stand-in service is not a server, so it says so instead of reporting none."""

    with _client_client(project, None) as client:
        response = client.get("/api/clients")
    assert response.status_code == 501
    assert "not served by a running app" in response.json()["error"]


def test_the_workspace_hides_the_panel_when_no_registry_is_wired(
    project: Path,
) -> None:
    """The capability stays on, because the app is a server; the route refuses."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    with TestClient(create_ui_app(config, service=FakeResearchService())) as client:  # type: ignore[arg-type]
        assert client.get("/api/ui").json()["capabilities"]["clients"] is True
        assert client.get("/api/clients").status_code == 501


def test_a_workspace_disconnect_is_held_to_the_write_rules(project: Path) -> None:
    """Ending a session is a write, so the same rules the other writes carry apply."""

    registry = _registry()
    with _client_client(project, registry) as client:
        cross_origin = client.post(
            "/api/clients/s-1/disconnect",
            headers={"Origin": "https://example.com"},
            json={"reason": "x"},
        )
        form = client.post(
            "/api/clients/s-1/disconnect",
            content=b"reason=x",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    assert cross_origin.status_code == 403
    assert form.status_code == 415
    assert registry.report()[0]["attached"] is True
