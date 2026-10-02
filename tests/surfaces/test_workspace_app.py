from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from research_rag.surfaces.workspace import (
    SourceFile,
    UICapabilities,
    UIProfile,
    UIRequestError,
    create_ui_app,
)


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
            "export_bundle": {
                "bundle_name": "project-generation.research-rag.zip",
                "sha256": "abc",
            },
            "import_bundle": {
                "generation_id": "generation-imported",
                "activated": arguments.get("activate", True),
            },
            "memory_status": {
                "scopes": [
                    {
                        "scope": "local",
                        "label": "This project",
                        "directory": "/project/.research-rag",
                        "standing_present": True,
                        "round_count": 1,
                        "latest_round_date": "2026-09-22",
                    },
                    {
                        "scope": "global:ahmed",
                        "label": "Global · ahmed",
                        "directory": "/shared/memory/ahmed",
                        "standing_present": True,
                        "round_count": 0,
                        "latest_round_date": None,
                    },
                ],
            },
            "memory_rounds": {
                "scope": arguments.get("scope"),
                "round_count": 1,
                "truncated": False,
                "rounds": [
                    {
                        "date": "2026-09-22",
                        "time": "14:54:02",
                        "user": "Where does the draft live?",
                        "assistant": "In the project directory.",
                        "source_file": "2026-09-22.md",
                    }
                ],
            },
            "memory_standing": {
                "scope": arguments.get("scope"),
                "content": "# MEMORY\nRemember the draft.\n",
                "sha256": "d1g3st",
            },
            "memory_append": {
                "status": "saved",
                "scope": arguments.get("scope"),
                "written": "/project/.research-rag/project/2026-09-22.md",
            },
            "memory_standing_save": {
                "status": "saved",
                "scope": arguments.get("scope"),
                "sha256": "n3w-digest",
            },
            "settings_read": {
                "revision": "rev-1",
                "sections": [
                    {
                        "key": "retrieval",
                        "title": "Retrieval",
                        "settings": [
                            {
                                "key": "retrieval.rrf_k",
                                "label": "Reciprocal rank fusion k",
                                "value": 60,
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


def _profile(**capabilities: bool) -> UIProfile:
    return UIProfile(
        application_name="Test UltraRAG",
        capabilities=UICapabilities(**capabilities),
    )


class SqlRecordAdapter(FakeAdapter):
    """A corpus fake that also runs statements against the records it stores."""

    def __init__(self, source: Path) -> None:
        super().__init__(source)
        self.statements: list[tuple[str, str, str]] = []

    async def sql_query(self, scope: str, statement: str) -> Mapping[str, Any]:
        self.statements.append((scope, statement, "query"))
        if "current_generation" in statement:
            raise UIRequestError(
                "The generation in use cannot be edited", status_code=409
            )
        return {
            "columns": ["generation_id", "chunk_count"],
            "rows": [["generation-1", 128], ["generation-2", 64]],
            "row_count": 2,
            "statement": statement,
            "scope": scope,
        }

    async def sql_execute(self, scope: str, statement: str) -> Mapping[str, Any]:
        self.statements.append((scope, statement, "execute"))
        return {
            "scope": scope,
            "rows_affected": 1,
            "statement": statement,
            "reindexed": True,
        }


def _sql_host(adapter: Any, *, enabled: bool = True) -> TestClient:
    return TestClient(
        create_ui_app(profile=_profile(sql_console=enabled), adapter=adapter)
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    return source


def test_workspace_and_normalized_read_operations(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(profile=_profile(), adapter=adapter)

    with TestClient(app) as client:
        page = client.get("/")
        stylesheet = client.get("/assets/app.css")
        script = client.get("/assets/app.js")
        profile = client.get("/api/ui")
        health = client.get("/api/health")
        status = client.get("/api/status")
        sources = client.get("/api/sources?categories=theory,history")
        passage = client.get("/api/passages/chunk-1?context_chunks=2")

    assert page.status_code == 200
    assert "UltraRAG MCP" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "applyProfile" in script.text
    assert profile.json()["application_name"] == "Test UltraRAG"
    assert health.json()["project_root"] == "/project"
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert passage.json()["requested_chunk_id"] == "chunk-1"
    assert (
        "list_sources",
        {
            "categories": ["theory", "history"],
            "categories_any": None,
            "projects": None,
            "projects_any": None,
            "keywords": None,
        },
    ) in adapter.calls


def test_source_selection_and_partitions_are_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(
            source_selection=True,
            category_partitions=True,
            project_metadata=True,
        ),
        adapter=adapter,
    )

    with TestClient(app) as client:
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
    assert (
        "list_sources",
        {
            "categories": None,
            "categories_any": ["theory", "history"],
            "projects": None,
            "projects_any": None,
            "keywords": None,
        },
    ) in adapter.calls
    assert (
        "list_sources",
        {
            "categories": None,
            "categories_any": None,
            "projects": None,
            "projects_any": ["ai-and-fetishism"],
            "keywords": None,
        },
    ) in adapter.calls
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
    app = create_ui_app(profile=_profile(), adapter=adapter)

    with TestClient(app) as client:
        search = client.post("/api/search", json={"query": "evidence", "top_k": 5})
        metadata = client.post(
            "/api/source-metadata",
            json={"source_path": "evidence.pdf", "metadata": {"title": "Evidence"}},
        )
        inclusion = client.post(
            "/api/source-inclusion",
            json={"source_path": "evidence.pdf", "included": False},
        )
        ingestion = client.post(
            "/api/ingest",
            json={"chunk_size": 384, "chunk_overlap": 64},
        )
        served = client.get("/api/source-file?path=evidence.pdf")
        missing = client.get("/api/source-file?path=missing.pdf")
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
        unsupported_force = client.post(
            "/api/ingest",
            json={"force_recompute": True},
        )

    assert search.status_code == 200
    assert metadata.json()["requires_ingest"] is True
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    assert served.headers["content-type"].startswith("application/pdf")
    assert missing.status_code == 404
    assert non_json.status_code == 415
    assert cross_origin.status_code == 403
    assert unknown.status_code == 400
    assert unsupported_force.status_code == 400


def test_force_recompute_is_capability_gated_and_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(force_recompute=True),
        adapter=adapter,
    )

    with TestClient(app) as client:
        ingestion = client.post(
            "/api/ingest",
            json={
                "chunk_size": 384,
                "chunk_overlap": 64,
                "force_recompute": True,
            },
        )
        invalid = client.post(
            "/api/ingest",
            json={"force_recompute": "yes"},
        )

    assert ingestion.status_code == 200
    assert invalid.status_code == 400
    assert (
        "ingest",
        {
            "chunk_size": 384,
            "chunk_overlap": 64,
            "force_recompute": True,
        },
    ) in adapter.calls


def test_disabled_capabilities_are_reported_and_enforced(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    profile = _profile(
        sources=False,
        passage_context=False,
        ingestion=False,
        metadata=False,
        source_inclusion=False,
        source_files=False,
        metadata_filters=False,
        source_selection=False,
        category_partitions=False,
        project_metadata=False,
        retrieval_modes=False,
        reranking=False,
        chunk_settings=False,
        generations=False,
    )
    app = create_ui_app(profile=profile, adapter=FakeAdapter(source))

    with TestClient(app) as client:
        ui = client.get("/api/ui")
        sources = client.get("/api/sources")
        ingest = client.post("/api/ingest", json={})
        chunked_ingest = client.post(
            "/api/ingest",
            json={"chunk_size": 384, "chunk_overlap": 64},
        )
        rerank = client.post(
            "/api/search",
            json={"query": "evidence", "rerank": True},
        )
        mode = client.post(
            "/api/search",
            json={"query": "evidence", "retrieval_method": "bm25"},
        )
        partitions = client.post(
            "/api/search",
            json={"query": "evidence", "categories_any": ["theory"]},
        )
        selection = client.post(
            "/api/search",
            json={"query": "evidence", "source_ids": ["src_1"]},
        )
        project_filter = client.post(
            "/api/search",
            json={"query": "evidence", "projects_any": ["ai-and-fetishism"]},
        )
        authors = client.post(
            "/api/search",
            json={"query": "evidence", "authors_any": ["Crawford"]},
        )
        titles = client.post(
            "/api/search",
            json={"query": "evidence", "titles_any": ["Atlas of AI"]},
        )
        source_file = client.get("/api/source-file?path=evidence.pdf")
        export = client.post("/api/bundles/export", json={})
        bundle_import = client.post(
            "/api/bundles/import",
            json={"bundle_name": "project.research-rag.zip"},
        )

    assert ui.json()["capabilities"]["sources"] is False
    assert ui.json()["capabilities"]["retrieval_modes"] is False
    assert ui.json()["capabilities"]["chunk_settings"] is False
    assert ui.json()["capabilities"]["bibliographic_filters"] is False
    assert sources.status_code == 404
    assert ingest.status_code == 404
    assert chunked_ingest.status_code == 400
    assert rerank.status_code == 400
    assert mode.status_code == 400
    assert partitions.status_code == 400
    assert selection.status_code == 400
    assert project_filter.status_code == 400
    assert authors.status_code == 400
    assert titles.status_code == 400
    assert source_file.status_code == 404
    assert export.status_code == 404
    assert bundle_import.status_code == 404


def test_bundle_actions_are_capability_gated_and_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(bundle_export=True, bundle_import=True),
        adapter=adapter,
    )

    with TestClient(app) as client:
        exported = client.post("/api/bundles/export", json={})
        imported = client.post(
            "/api/bundles/import",
            json={
                "bundle_name": "project-generation.research-rag.zip",
                "activate": False,
            },
        )
        unsafe = client.post(
            "/api/bundles/import",
            json={"bundle_name": "bundle.zip", "path": "../bundle.zip"},
        )

    assert exported.json()["bundle_name"].endswith(".research-rag.zip")
    assert imported.json()["activated"] is False
    assert unsafe.status_code == 400
    assert ("export_bundle", {}) in adapter.calls
    assert (
        "import_bundle",
        {
            "bundle_name": "project-generation.research-rag.zip",
            "activate": False,
        },
    ) in adapter.calls


def test_configuration_rejects_ambiguous_adapter_setup(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    adapter = FakeAdapter(source)

    try:
        create_ui_app(profile=_profile())
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("missing adapter was accepted")

    try:
        create_ui_app(
            profile=_profile(),
            adapter=adapter,
            adapter_factory=lambda: None,  # type: ignore[arg-type,return-value]
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous adapter setup was accepted")


def test_version_label_defaults_to_empty(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(profile=_profile(), adapter=FakeAdapter(source))

    with TestClient(app) as client:
        payload = client.get("/api/ui").json()

    assert payload["version_label"] == ""


def test_version_label_is_served_and_rendered(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    profile = UIProfile(
        application_name="Test UltraRAG",
        version_label="server 1.2.3 · ui 0.5.0",
    )
    app = create_ui_app(profile=profile, adapter=FakeAdapter(source))

    with TestClient(app) as client:
        payload = client.get("/api/ui").json()
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert payload["version_label"] == "server 1.2.3 · ui 0.5.0"
    assert 'id="version-label"' in page.text
    assert "version_label" in script.text


def test_a_memory_only_adapter_hides_the_document_workspace(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(
            documents=False,
            sources=False,
            ingestion=False,
            metadata=False,
            source_inclusion=False,
            source_files=False,
            metadata_filters=False,
            reranking=False,
            memory=True,
        ),
        adapter=adapter,
    )

    with TestClient(app) as client:
        payload = client.get("/api/ui").json()
        page = client.get("/")
        search = client.post("/api/search", json={"query": "evidence"})
        sources = client.get("/api/sources")
        memory = client.get("/api/memory")

    assert payload["capabilities"]["documents"] is False
    assert 'data-panel="search" data-capability="documents"' in page.text
    assert 'data-view="memory" data-capability="memory"' in page.text
    assert search.status_code == 404
    assert sources.status_code == 404
    assert memory.status_code == 200


def test_bibliographic_filters_are_capability_gated_and_forwarded(
    tmp_path: Path,
) -> None:
    """An author, title, or language filter travels only where the server implements it."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(profile=_profile(bibliographic_filters=True), adapter=adapter)

    with TestClient(app) as client:
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


def test_a_language_filter_is_refused_where_the_capability_is_off(
    tmp_path: Path,
) -> None:
    """Language is a bibliographic filter, so the same flag gates it.

    A layer the app serves and the workspace accepted silently would narrow a
    search the reader did not know was narrowed, which is the one failure this
    gate exists to prevent.
    """

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(profile=_profile(), adapter=adapter)

    with TestClient(app) as client:
        language = client.post(
            "/api/search", json={"query": "evidence", "languages_any": ["en"]}
        )
        author = client.post(
            "/api/search", json={"query": "evidence", "authors_any": ["Crawford"]}
        )

    assert language.status_code == 400
    assert author.status_code == 400
    assert not any(operation == "search" for operation, _ in adapter.calls)


def test_the_quotation_rule_is_not_a_footer(tmp_path: Path) -> None:
    """The shared UI states no quotation rule of its own.

    A rule that matters on every passage belongs in the documentation and in the
    tool descriptions that an agent reads, not in the corner of every view.
    """

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(profile=_profile(), adapter=FakeAdapter(source))

    with TestClient(app) as client:
        profile = client.get("/api/ui")
        page = client.get("/")
        javascript = client.get("/assets/app.js")

    assert profile.json()["footer_text"] == ""
    assert "not for direct quotation" not in page.text.lower()
    assert "Verify important quotations" not in page.text
    # The footer is emptied and hidden rather than left as an empty band, so a
    # profile that sets no footer shows none.
    assert "footer.hidden = !profile.footer_text" in javascript.text


def test_every_capability_is_declared_in_the_markup_and_named_in_the_loop(
    tmp_path: Path,
) -> None:
    """A control the profile hides must be declared in the markup, not left inert."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    app = create_ui_app(profile=_profile(), adapter=FakeAdapter(source))

    with TestClient(app) as client:
        page = client.get("/")
        javascript = client.get("/assets/app.js")

    assert page.status_code == 200
    assert javascript.status_code == 200
    for capability in (
        "bibliographic_filters",
        "bundle_export",
        "bundle_import",
        "category_partitions",
        "chunk_settings",
        "documents",
        "ingestion",
        "memory",
        "metadata_filters",
        "project_metadata",
        "reranking",
        "retrieval_modes",
        "source_selection",
        "sources",
        "settings",
        "chunk_exclusion",
        "projects",
        "agent_entry",
    ):
        assert f'data-capability="{capability}"' in page.text
    # The search and ingestion payloads send a capability-gated field only when
    # that capability is on, so a hidden control never travels as a value.
    assert 'hasCapability("retrieval_modes")' in javascript.text
    assert 'hasCapability("chunk_settings")' in javascript.text
    assert 'hasCapability("reranking")' in javascript.text
    assert 'hasCapability("bibliographic_filters")' in javascript.text
    # The language filter and the inventory that names the languages in a
    # project are one capability: a box a reader cannot fill from the status
    # beside it is a box that asks them to know the codes already listed.
    assert 'id="language-filter"' in page.text
    assert 'id="language-chips"' in page.text
    assert 'byId("language-filter").value' in javascript.text
    assert "renderLanguages(status)" in javascript.text


def test_memory_view_is_capability_gated_and_forwarded(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(memory=True, memory_writes=True),
        adapter=adapter,
    )

    with TestClient(app) as client:
        status = client.get("/api/memory")
        rounds = client.get("/api/memory/rounds?scope=local&limit=5")
        standing = client.get("/api/memory/standing?scope=global:ahmed")
        appended = client.post(
            "/api/memory/append",
            json={
                "scope": "local",
                "user_message": "Remember this.",
                "assistant_message": "Remembered.",
            },
        )
        saved = client.post(
            "/api/memory/standing",
            json={
                "scope": "local",
                "content": "# MEMORY\nRemember the draft.\n",
                "expected_sha256": "d1g3st",
            },
        )

    assert status.json()["scopes"][0]["scope"] == "local"
    assert rounds.json()["rounds"][0]["assistant"] == "In the project directory."
    assert standing.json()["scope"] == "global:ahmed"
    assert appended.json()["status"] == "saved"
    assert saved.json()["sha256"] == "n3w-digest"
    assert ("memory_rounds", {"scope": "local", "limit": 5}) in adapter.calls
    assert ("memory_standing", {"scope": "global:ahmed"}) in adapter.calls
    assert (
        "memory_append",
        {
            "scope": "local",
            "user_message": "Remember this.",
            "assistant_message": "Remembered.",
        },
    ) in adapter.calls
    assert (
        "memory_standing_save",
        {
            "scope": "local",
            "content": "# MEMORY\nRemember the draft.\n",
            "expected_sha256": "d1g3st",
        },
    ) in adapter.calls


def test_memory_operations_are_capability_gated(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    append = {
        "scope": "local",
        "user_message": "Remember this.",
        "assistant_message": "Remembered.",
    }

    with TestClient(create_ui_app(profile=_profile(), adapter=adapter)) as client:
        hidden = [
            client.get("/api/memory"),
            client.get("/api/memory/rounds?scope=local"),
            client.get("/api/memory/standing?scope=local"),
            client.post("/api/memory/append", json=append),
            client.post(
                "/api/memory/standing",
                json={"scope": "local", "content": "# MEMORY\n"},
            ),
        ]

    with TestClient(
        create_ui_app(profile=_profile(memory=True), adapter=adapter)
    ) as client:
        readable = client.get("/api/memory")
        refused = client.post("/api/memory/append", json=append)

    assert [response.status_code for response in hidden] == [404] * 5
    assert readable.status_code == 200
    assert refused.status_code == 404
    assert all(operation != "memory_append" for operation, _ in adapter.calls)


def test_memory_writes_are_validated(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(
        profile=_profile(memory=True, memory_writes=True),
        adapter=adapter,
    )

    with TestClient(app) as client:
        no_scope = client.post(
            "/api/memory/append",
            json={"user_message": "a", "assistant_message": "b"},
        )
        empty_message = client.post(
            "/api/memory/append",
            json={"scope": "local", "user_message": "   ", "assistant_message": "b"},
        )
        unknown_field = client.post(
            "/api/memory/append",
            json={
                "scope": "local",
                "user_message": "a",
                "assistant_message": "b",
                "path": "../outside",
            },
        )
        empty_content = client.post(
            "/api/memory/standing",
            json={"scope": "local", "content": "  "},
        )
        bad_digest = client.post(
            "/api/memory/standing",
            json={"scope": "local", "content": "# MEMORY\n", "expected_sha256": 7},
        )
        bad_limit = client.get("/api/memory/rounds?scope=local&limit=many")
        wide_limit = client.get("/api/memory/rounds?scope=local&limit=900")
        no_scope_read = client.get("/api/memory/standing")

    assert no_scope.status_code == 400
    assert empty_message.status_code == 400
    assert unknown_field.status_code == 400
    assert empty_content.status_code == 400
    assert bad_digest.status_code == 400
    assert bad_limit.status_code == 400
    assert wide_limit.status_code == 400
    assert no_scope_read.status_code == 400
    assert not [
        operation
        for operation, _ in adapter.calls
        if operation in {"memory_append", "memory_standing_save"}
    ]


def test_memory_labels_are_served_and_rendered(tmp_path: Path) -> None:
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    profile = UIProfile(
        application_name="Test memory",
        memory_label="Agent memory",
        memory_standing_label="What is always remembered",
        memory_rounds_label="Dialogue rounds",
        memory_add_label="Record a round",
        memory_note="Memory stays on this machine.",
        capabilities=UICapabilities(memory=True, memory_writes=True),
    )
    app = create_ui_app(profile=profile, adapter=FakeAdapter(source))

    with TestClient(app) as client:
        payload = client.get("/api/ui").json()
        script = client.get("/assets/app.js")
        page = client.get("/")

    assert payload["memory_label"] == "Agent memory"
    assert payload["memory_standing_label"] == "What is always remembered"
    assert payload["memory_note"] == "Memory stays on this machine."
    assert payload["capabilities"]["memory"] is True
    assert payload["capabilities"]["memory_writes"] is True
    assert 'id="memory-scopes"' in page.text
    assert 'data-panel="memory"' in page.text
    for label in (
        "memory_standing_label",
        "memory_rounds_label",
        "memory_add_label",
    ):
        assert label in script.text
    # Every scope the status reports is rendered, so the view asks for each one.
    assert "memoryScopeCard" in script.text
    assert "/api/memory/rounds?scope=" in script.text


def test_a_generation_is_removed_only_with_a_matching_confirmation(
    tmp_path: Path,
) -> None:
    """The dialog's typed id reaches the adapter, and a mismatch never does.

    A host that supplied its own confirmation would be a host where one click
    removed a generation nobody chose, which is the only thing the dialog is for.
    """

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(profile=_profile(generations=True), adapter=adapter)
    generation_id = "20260930T191235Z-45608dc5"

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["generations"] is True
        removed = client.post(
            "/api/generations/remove",
            json={"generation_id": generation_id, "confirm": generation_id},
        )
        assert removed.status_code == 200
        assert (
            "remove_generation",
            {"generation_id": generation_id, "confirm": generation_id},
        ) in adapter.calls

        for body, expected in (
            ({"generation_id": generation_id, "confirm": "nope"}, "repeat"),
            ({"generation_id": generation_id}, "generation_id and confirm"),
            (
                {"generation_id": generation_id, "confirm": generation_id, "x": 1},
                "generation_id and confirm",
            ),
        ):
            refused = client.post("/api/generations/remove", json=body)
            assert refused.status_code == 400
            assert expected in refused.json()["error"]

    # Every refusal happened in the workspace, so the adapter saw one call.
    assert [name for name, _ in adapter.calls] == ["remove_generation"]


def test_a_generation_cannot_be_removed_where_the_capability_is_off(
    tmp_path: Path,
) -> None:
    """A panel that shows chips and a button is not shown without a way to act."""

    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    adapter = FakeAdapter(source)
    app = create_ui_app(profile=_profile(generations=False), adapter=adapter)
    generation_id = "20260930T191235Z-45608dc5"

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["generations"] is False
        refused = client.post(
            "/api/generations/remove",
            json={"generation_id": generation_id, "confirm": generation_id},
        )

    assert refused.status_code == 404
    assert adapter.calls == []


def test_the_status_view_carries_the_generation_panel_and_its_dialog() -> None:
    """The listing is free in the status payload, so the panel needs only markup."""

    app = create_ui_app(profile=_profile(), adapter=FakeAdapter(Path("evidence.pdf")))

    with TestClient(app) as client:
        page = client.get("/").text
        javascript = client.get("/assets/app.js").text

    assert 'id="generation-chips"' in page
    assert 'data-capability="generations"' in page
    assert 'id="generation-dialog"' in page
    # The remove button stays disabled until the typed id matches, so the
    # confirmation is a gate rather than a message.
    assert 'id="generation-submit"' in page and "disabled" in page
    assert 'hasCapability("generations")' in javascript
    assert "renderGenerations(status.generations || [])" in javascript
    assert 'byId("generation-confirm").addEventListener' in javascript
    assert "/api/generations/remove" in javascript
    # The one a search reads is not offered a removal it would only refuse.
    assert "if (!generation.is_current)" in javascript


def test_a_sql_statement_reads_the_records_the_adapter_stores(tmp_path: Path) -> None:
    """The result envelope reaches the page exactly as the adapter sent it.

    The panel is given columns and rows and nothing else, so any reshaping here
    would be a panel that renders a table the adapter did not return.
    """

    adapter = SqlRecordAdapter(_source(tmp_path))
    statement = "SELECT generation_id, chunk_count FROM generations"

    with _sql_host(adapter) as client:
        assert client.get("/api/ui").json()["capabilities"]["sql_console"] is True
        response = client.post(
            "/api/sql/query",
            json={"scope": "  generations  ", "statement": statement},
        )

    assert response.status_code == 200
    assert response.json() == {
        "columns": ["generation_id", "chunk_count"],
        "rows": [["generation-1", 128], ["generation-2", 64]],
        "row_count": 2,
        "statement": statement,
        "scope": "generations",
    }
    # The statement travels as typed; the scope is trimmed because an identifier
    # carried with stray whitespace is one the adapter does not recognise.
    assert adapter.statements == [("generations", statement, "query")]


def test_a_sql_write_reports_the_records_it_affected(tmp_path: Path) -> None:
    adapter = SqlRecordAdapter(_source(tmp_path))
    statement = "DELETE FROM generations WHERE generation_id = 'generation-1';"

    with _sql_host(adapter) as client:
        response = client.post(
            "/api/sql/execute",
            json={"scope": "generations", "statement": statement},
        )

    assert response.status_code == 200
    assert response.json() == {
        "scope": "generations",
        "rows_affected": 1,
        "statement": statement,
        "reindexed": True,
    }
    assert adapter.statements == [("generations", statement, "execute")]


def test_the_sql_routes_are_absent_when_the_capability_is_off(tmp_path: Path) -> None:
    """A store this library cannot know about is not reachable from a workspace."""

    adapter = SqlRecordAdapter(_source(tmp_path))

    with _sql_host(adapter, enabled=False) as client:
        assert client.get("/api/ui").json()["capabilities"]["sql_console"] is False
        read = client.post(
            "/api/sql/query", json={"scope": "generations", "statement": "SELECT 1"}
        )
        write = client.post(
            "/api/sql/execute",
            json={"scope": "generations", "statement": "DELETE FROM generations"},
        )

    assert read.status_code == 404
    assert write.status_code == 404
    assert adapter.statements == []


def test_a_sql_request_must_carry_a_scope_and_a_statement(tmp_path: Path) -> None:
    """A statement without a scope would be a statement against whatever is open."""

    adapter = SqlRecordAdapter(_source(tmp_path))
    scope = "generations"
    statement = "SELECT 1"

    with _sql_host(adapter) as client:
        no_scope = client.post("/api/sql/query", json={"statement": statement})
        blank_scope = client.post(
            "/api/sql/query", json={"scope": "   ", "statement": statement}
        )
        wrong_scope = client.post(
            "/api/sql/query", json={"scope": 7, "statement": statement}
        )
        no_statement = client.post("/api/sql/query", json={"scope": scope})
        blank_statement = client.post(
            "/api/sql/query", json={"scope": scope, "statement": "\n\t "}
        )
        no_scope_write = client.post("/api/sql/execute", json={"statement": statement})
        unknown_field = client.post(
            "/api/sql/execute",
            json={"scope": scope, "statement": statement, "path": "../outside"},
        )

    assert no_scope.status_code == 400
    assert "scope is required" in no_scope.json()["error"]
    assert blank_scope.status_code == 400
    assert wrong_scope.status_code == 400
    assert no_statement.status_code == 400
    assert "statement is required" in no_statement.json()["error"]
    assert blank_statement.status_code == 400
    assert no_scope_write.status_code == 400
    assert unknown_field.status_code == 400
    assert "Unsupported SQL fields: path" in unknown_field.json()["error"]
    # Every refusal happened in the workspace, so the adapter saw nothing.
    assert adapter.statements == []


def test_a_runaway_statement_stops_at_the_request_layer(tmp_path: Path) -> None:
    """A paste of a script stops here rather than travelling to the store."""

    adapter = SqlRecordAdapter(_source(tmp_path))

    with _sql_host(adapter) as client:
        refused = client.post(
            "/api/sql/query",
            json={"scope": "generations", "statement": "SELECT 1\n" * 20000},
        )
        accepted = client.post(
            "/api/sql/query",
            json={"scope": "generations", "statement": "SELECT 1"},
        )

    assert refused.status_code == 400
    assert "20000 characters" in refused.json()["error"]
    assert accepted.status_code == 200
    assert len(adapter.statements) == 1


def test_an_sql_refusal_keeps_its_own_status_and_reason(tmp_path: Path) -> None:
    """A store that will not be edited has said something a reader can act on.

    Translating that into a generic failure would throw the reason away, so a
    statement route carries the same translation the operation routes do.
    """

    with _sql_host(SqlRecordAdapter(_source(tmp_path))) as client:
        response = client.post(
            "/api/sql/query",
            json={"scope": "generations", "statement": "SELECT current_generation"},
        )

    assert response.status_code == 409
    assert "cannot be edited" in response.json()["error"]


def test_a_sql_statement_is_same_origin_and_json_only(tmp_path: Path) -> None:
    """A statement writes records, so it is held to the write rules."""

    adapter = SqlRecordAdapter(_source(tmp_path))
    body = {"scope": "generations", "statement": "DELETE FROM generations"}
    form = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    with _sql_host(adapter) as client:
        cross_origin = [
            client.post(
                "/api/sql/query", headers={"Origin": "https://example.com"}, json=body
            ),
            client.post(
                "/api/sql/execute", headers={"Origin": "https://example.com"}, json=body
            ),
        ]
        form_encoded = [
            client.post("/api/sql/query", content=b"scope=generations", headers=form),
            client.post("/api/sql/execute", content=b"scope=generations", headers=form),
        ]

    assert [response.status_code for response in cross_origin] == [403, 403]
    assert [response.status_code for response in form_encoded] == [415, 415]
    assert adapter.statements == []


def test_a_host_that_advertises_the_sql_console_and_cannot_answer_says_501(
    tmp_path: Path,
) -> None:
    """A capability without the methods is a host fact, not a traceback."""

    with _sql_host(FakeAdapter(_source(tmp_path))) as client:
        response = client.post(
            "/api/sql/query", json={"scope": "generations", "statement": "SELECT 1"}
        )

    assert response.status_code == 501
    assert "cannot run statements" in response.json()["error"]


def test_a_host_reports_the_projects_it_serves_and_its_own_client_entry(
    tmp_path: Path,
) -> None:
    """The selector and the entry are the host's own answer, forwarded whole.

    Both take no argument this repository could add: which projects exist and
    what a client's configuration says are the host's facts, so a route here that
    named one would be a second place for it to be wrong.
    """

    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(
        profile=_profile(projects=True, agent_entry=True), adapter=adapter
    )

    with TestClient(app) as client:
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


def test_the_projects_and_agent_entry_routes_are_absent_when_the_flags_are_off(
    tmp_path: Path,
) -> None:
    """A host that serves one project has nothing to select and no entry to give."""

    adapter = FakeAdapter(_source(tmp_path))

    with TestClient(
        create_ui_app(
            profile=_profile(projects=False, agent_entry=False), adapter=adapter
        )
    ) as client:
        assert client.get("/api/ui").json()["capabilities"]["projects"] is False
        assert client.get("/api/ui").json()["capabilities"]["agent_entry"] is False
        projects = client.get("/api/projects")
        entry = client.get("/api/agent-entry")

    assert projects.status_code == 404
    assert entry.status_code == 404
    assert adapter.calls == []


def test_settings_are_read_and_written_through_the_adapter(tmp_path: Path) -> None:
    """The panel reads the server's settings and sends back only what changed.

    Nothing here is named or defaulted: the settings, their origins, and the
    cost of changing them are the server's facts, so both directions travel
    exactly as they were produced.
    """

    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(profile=_profile(settings=True), adapter=adapter)

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["settings"] is True
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


def test_the_settings_routes_are_absent_when_the_capability_is_off(
    tmp_path: Path,
) -> None:
    """A server with no settings panel of its own gets no settings surface."""

    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(profile=_profile(settings=False), adapter=adapter)

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["settings"] is False
        read = client.get("/api/settings")
        write = client.post(
            "/api/settings",
            json={"values": {"retrieval.rrf_k": 40}, "confirm": True},
        )

    assert read.status_code == 404
    assert write.status_code == 404
    assert adapter.calls == []


def test_a_settings_write_is_same_origin_and_json_only(tmp_path: Path) -> None:
    """Changing a setting changes the server, so it is held to the write rules."""

    adapter = FakeAdapter(_source(tmp_path))
    body = {"values": {"retrieval.rrf_k": 40}, "expected_revision": "rev-1"}

    with TestClient(
        create_ui_app(profile=_profile(settings=True), adapter=adapter)
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
    app = create_ui_app(profile=_profile(chunk_exclusion=True), adapter=adapter)

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["chunk_exclusion"] is True
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


def test_the_chunk_routes_are_absent_when_the_capability_is_off(tmp_path: Path) -> None:
    adapter = FakeAdapter(_source(tmp_path))
    app = create_ui_app(profile=_profile(chunk_exclusion=False), adapter=adapter)

    with TestClient(app) as client:
        assert client.get("/api/ui").json()["capabilities"]["chunk_exclusion"] is False
        listed = client.get("/api/chunk-exclusions")
        excluded = client.post(
            "/api/chunk-inclusion",
            json={"chunk_id": "chunk-9", "included": False, "reason": "why"},
        )

    assert listed.status_code == 404
    assert excluded.status_code == 404
    assert adapter.calls == []


def test_a_chunk_inclusion_write_is_same_origin_and_json_only(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(_source(tmp_path))
    body = {"chunk_id": "chunk-9", "included": False, "reason": "why"}
    form = {"Content-Type": "application/x-www-form-urlencoded"}

    with TestClient(
        create_ui_app(profile=_profile(chunk_exclusion=True), adapter=adapter)
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
