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

    async def set_chunk_inclusion(
        self, chunk_id: str, **arguments: Any
    ) -> dict[str, Any]:
        self._record("set_chunk_inclusion", {"chunk_id": chunk_id, **arguments})
        return {
            "chunk_id": chunk_id,
            "included": arguments.get("included"),
            "reason": arguments.get("reason"),
            "message": "Chunk exclusion saved.",
        }

    async def list_chunk_exclusions(self) -> dict[str, Any]:
        self._record("list_chunk_exclusions", {})
        return {
            "generation_id": "generation-1",
            "excluded_chunk_count": 1,
            "exclusions": [
                {
                    "chunk_id": "chunk-1",
                    "source_relative_path": "evidence.pdf",
                    "locator": "p. 1",
                    "reason": "Reviewed fragment",
                    "excluded_at": "2026-09-30T19:12:35.104Z",
                    "in_current_generation": True,
                }
            ],
            "message": "Every one of the 1 excluded chunks is withheld.",
        }

    async def set_source_metadata(self, **arguments: Any) -> dict[str, Any]:
        self._record("set_source_metadata", arguments)
        return {
            "source_relative_path": arguments.get("source_path"),
            "metadata": arguments.get("metadata"),
            "message": "Reviewed metadata saved.",
        }

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        self._record(
            "remove_generation",
            {"generation_id": generation_id, "confirm": confirm},
        )
        return {"status": "removed", "generation_id": generation_id}

    async def settings_read(self) -> dict[str, Any]:
        self._record("settings_read", {})
        return {"revision": "rev-1", "sections": [], "message": "38 settings."}

    async def settings_write(
        self, values: dict[str, Any], *, expected_revision: str, confirm: bool = False
    ) -> dict[str, Any]:
        self._record(
            "settings_write",
            {
                "values": values,
                "expected_revision": expected_revision,
                "confirm": confirm,
            },
        )
        return {
            "revision": "rev-2",
            "changed": sorted(values),
            "requires_ingest": True,
            "message": "Saved.",
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


def test_the_workspace_carries_a_chunk_decision_and_lists_the_ones_on_file(
    project: Path,
) -> None:
    """The panel beside a search result reaches the same decision the terminal makes.

    A decision about one passage has to be reachable where a reader sees the
    passage, or it is a decision only a person with a shell can make. The listing
    is what makes the decision reversible: without it a reader can add exclusions
    and never see them again.
    """

    from research_rag.surfaces.ui import RESEARCH_UI_PROFILE

    client, fake = _client(project)
    with client:
        excluded = client.post(
            "/api/chunk-inclusion",
            json={
                "chunk_id": "chunk-1",
                "included": False,
                "reason": "Reviewed fragment",
            },
        )
        restored = client.post(
            "/api/chunk-inclusion",
            json={"chunk_id": "chunk-1", "included": True},
        )
        listed = client.get("/api/chunk-exclusions")

    assert excluded.status_code == 200
    assert excluded.json()["included"] is False
    assert restored.json()["included"] is True
    assert listed.status_code == 200
    assert listed.json()["exclusions"][0]["chunk_id"] == "chunk-1"
    assert (
        "set_chunk_inclusion",
        {
            "chunk_id": "chunk-1",
            "included": False,
            "reason": "Reviewed fragment",
        },
    ) in fake.calls
    # A restore carries no reason: the exclusion is gone, so there is nothing left
    # to explain, and the file rejects an entry without one.
    assert (
        "set_chunk_inclusion",
        {"chunk_id": "chunk-1", "included": True, "reason": None},
    ) in fake.calls
    assert ("list_chunk_exclusions", {}) in fake.calls
    assert RESEARCH_UI_PROFILE.capabilities.chunk_exclusion is True


def test_the_workspace_refuses_a_chunk_decision_it_cannot_act_on(
    project: Path,
) -> None:
    """A missing id, a missing flag, and a reason that is not text are refusals.

    Each names the field it is about, because the panel that sent the request is
    the only place the reader can fix it.
    """

    client, fake = _client(project)
    with client:
        for body, expected in (
            ({"included": False, "reason": "x"}, "needs a chunk_id"),
            ({"chunk_id": "chunk-1", "reason": "x"}, "needs a boolean included"),
            ({"chunk_id": "chunk-1", "included": False, "reason": 7}, "must be"),
        ):
            answer = client.post("/api/chunk-inclusion", json=body)
            assert answer.status_code == 400, body
            assert expected in answer.json()["error"], answer.text

    assert not [call for call in fake.calls if call[0] == "set_chunk_inclusion"]


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
        "set_chunk_inclusion",
        "list_chunk_exclusions",
        "set_source_metadata",
        "remove_generation",
        "settings_read",
        "settings_write",
        "list_projects",
        "agent_entry",
    }
    # The account's projects and this project's client entry are both read from
    # what this app already knows, so neither takes an argument the workspace
    # could have asked something for.
    assert _OPERATION_ARGUMENTS["list_projects"] == frozenset()
    assert _OPERATION_ARGUMENTS["agent_entry"] == frozenset()


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


def test_the_workspace_removes_a_generation_only_with_the_id_repeated(
    project: Path,
) -> None:
    """A browser may not turn a listing into a deletion with one click.

    The shared workspace already refuses a confirmation that does not repeat the
    id, so the half that matters here is that the adapter forwards the pair
    rather than defaulting the confirmation. Forwarding one field and inventing
    the other is how a click would come to mean yes.
    """

    from ui_ultra_rag_mcp import UIRequestError

    from research_rag.surfaces.ui import ResearchUIAdapter

    config = resolve_config(project)
    service = FakeResearchService()
    adapter = ResearchUIAdapter(config, service)  # type: ignore[arg-type]
    generation_id = "20260930T191235Z-45608dc5"

    result = asyncio.run(
        adapter.call(
            "remove_generation",
            {"generation_id": generation_id, "confirm": generation_id},
        )
    )

    assert result == {"status": "removed", "generation_id": generation_id}
    assert (
        "remove_generation",
        {"generation_id": generation_id, "confirm": generation_id},
    ) in (service.calls)
    for body in (
        {"generation_id": generation_id},
        {"generation_id": generation_id, "confirm": 7},
        {"confirm": generation_id},
    ):
        with pytest.raises(UIRequestError):
            asyncio.run(adapter.call("remove_generation", body))


def test_the_workspace_serves_the_generation_panel(project: Path) -> None:
    """The panel is on because the app retains generations a rebuild leaves behind."""

    from research_rag.surfaces.ui import RESEARCH_UI_PROFILE

    assert RESEARCH_UI_PROFILE.capabilities.generations is True
    assert RESEARCH_UI_PROFILE.capabilities.clients is True


def test_the_workspace_forwards_a_settings_write_whole(project: Path) -> None:
    """The values, the revision, and the confirmation all reach the service.

    The workspace sends its own field names, and the revision is the server's
    comparison token. Dropping one would let a browser write over a change it
    never saw, and defaulting the confirmation would let a click mean yes on a
    change that forces a rebuild.
    """

    from ui_ultra_rag_mcp import UIRequestError

    from research_rag.surfaces.ui import ResearchUIAdapter

    config = resolve_config(project)
    service = FakeResearchService()
    adapter = ResearchUIAdapter(config, service)  # type: ignore[arg-type]

    result = asyncio.run(
        adapter.call(
            "settings_write",
            {
                "values": {"retrieval.rrf_k": 40},
                "expected_revision": "rev-1",
                "confirm": True,
            },
        )
    )

    assert result == {
        "revision": "rev-2",
        "changed": ["retrieval.rrf_k"],
        "requires_ingest": True,
        "message": "Saved.",
    }
    assert service.calls[-1] == (
        "settings_write",
        {
            "values": {"retrieval.rrf_k": 40},
            "expected_revision": "rev-1",
            "confirm": True,
        },
    )
    for body in (
        {"expected_revision": "rev-1", "confirm": True},
        {"values": {"retrieval.rrf_k": 40}, "confirm": True},
        {"values": {"retrieval.rrf_k": 40}, "expected_revision": ""},
        {
            "values": {"retrieval.rrf_k": 40},
            "expected_revision": "rev-1",
            "confirm": "yes",
        },
    ):
        with pytest.raises(UIRequestError):
            asyncio.run(adapter.call("settings_write", body))


def test_the_workspace_reads_the_settings_the_server_resolved(project: Path) -> None:
    """A read carries no argument of this app's, so none can be forwarded."""

    from research_rag.surfaces.ui import ResearchUIAdapter

    config = resolve_config(project)
    service = FakeResearchService()
    adapter = ResearchUIAdapter(config, service)  # type: ignore[arg-type]

    result = asyncio.run(adapter.call("settings_read", {}))

    assert result["revision"] == "rev-1"
    assert service.calls == [("settings_read", {})]


def test_the_workspace_serves_the_account_record_not_the_served_project(
    project: Path,
) -> None:
    """One installation serves several projects, so the workspace says which.

    The listing is the account's own record rather than this project's descriptor,
    read through the same function `research-rag projects` prints, so a browser
    and a terminal name the same projects and cannot disagree about which are up.
    """

    from research_rag import registry
    from research_rag.registry import account_projects
    from research_rag.surfaces.ui import ResearchUIAdapter

    other = project.parent / "other-project"
    (other / "sources").mkdir(parents=True)
    config = resolve_config(project, vanilla_executable=sys.executable)
    served = ResearchUIAdapter(config, FakeResearchService())  # type: ignore[arg-type]
    # The account holds this project and another one, which is the arrangement a
    # workspace's selector exists for.
    registry.register(config.project_id, config.project_name, config.project_root)
    other_config = resolve_config(other, vanilla_executable=sys.executable)
    registry.register(other_config.project_id, other_config.project_name, other)
    service = FakeResearchService()
    adapter = ResearchUIAdapter(config, service)  # type: ignore[arg-type]

    result = asyncio.run(adapter.call("list_projects", {}))

    assert [entry["project_name"] for entry in result["projects"]] == [
        "other-project",
        "research-project",
    ]
    # The page marks the project it is actually serving, not the first entry.
    assert result["current"] == config.project_name
    assert result["message"] == ""
    for entry in result["projects"]:
        # A project whose app is not up has no URL, and the workspace is told so
        # rather than handed one it could not open.
        assert entry["running"] is False
        assert entry["url"] is None
        assert entry["attached_clients"] == 0
        assert Path(entry["project_root"]).is_absolute()
    # The names, roots, and states are the account record's own, not a second
    # reading of it: the terminal's listing and the browser's are one answer.
    assert [entry["project_name"] for entry in account_projects()["projects"]] == [
        "other-project",
        "research-project",
    ]
    # The service was never asked: a project this app does not serve has no
    # service to answer for it.
    assert service.calls == []
    assert served.config.project_name == config.project_name


def test_the_agent_entry_is_the_text_the_doctor_prints(project: Path) -> None:
    """A client's configuration is one generator, whichever surface produced it.

    A browser that generated its own entry would be a second place for it to be
    wrong, and the reader would have two entries to choose between.
    """

    from research_rag.doctor import mcp_entry_block
    from research_rag.surfaces.ui import ResearchUIAdapter

    config = resolve_config(project)
    service = FakeResearchService()
    adapter = ResearchUIAdapter(config, service)  # type: ignore[arg-type]

    result = asyncio.run(adapter.call("agent_entry", {}))

    assert result == {"entry": mcp_entry_block(config)}
    # The entry names the project rather than its directory, so one entry serves
    # the project on every machine where it was initialised.
    entry = json.loads(result["entry"])["mcp"]["research-rag"]
    assert entry["command"][entry["command"].index("--project-name") + 1] == (
        config.project_name
    )
    assert "--project-root" not in entry["command"]
    assert str(project) not in result["entry"]
    assert service.calls == []
