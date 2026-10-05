"""What a client reads before it calls: the annotations, the schemas, and the answers.

`surfaces/mcp.py` is the only place an agent's tools are declared, and a client
sees three things about them: the annotations that tell it whether to ask first,
the input schemas it builds a call from, and the projection each answer passes
through. These tests read all three the way a client does, through FastMCP, so a
description that grows into a manual or an annotation that lies about a write
fails here rather than in a user session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from research_rag.core.tool_views import present_tool_response
from research_rag.project.config import resolve_config
from research_rag.project.instructions import AGENT_INSTRUCTIONS
from research_rag.project.settings import FULL_TOOL_DETAIL, LEAN_TOOL_DETAIL
from research_rag.surfaces.mcp import AGENT_BRIDGE_NAME, SERVER_NAME, create_mcp

pytestmark = pytest.mark.anyio

TOOLS = (
    "status",
    "ingest",
    "search",
    "find_source",
    "get_passage",
    "set_source_inclusion",
    "set_chunk_inclusion",
    "set_source_metadata",
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Service:
    """A recording service, so no generation, index, or gateway is opened here."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.detail = LEAN_TOOL_DETAIL

    def _record(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((operation, arguments))
        return getattr(self, f"_{operation}")(arguments)

    async def status(self) -> dict[str, Any]:
        return self._record("status", {})

    async def ingest(self, *, force_recompute: bool) -> dict[str, Any]:
        return self._record("ingest", {"force_recompute": force_recompute})

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self._record("search", {"query": query, **arguments})

    async def find_source(self, *, query: str, limit: int) -> dict[str, Any]:
        return self._record("find_source", {"query": query, "limit": limit})

    async def get_passage(self, chunk_id: str) -> dict[str, Any]:
        return self._record("get_passage", {"chunk_id": chunk_id})

    async def set_source_inclusion(
        self, *, source_path: str, included: bool, reason: str | None
    ) -> dict[str, Any]:
        return self._record(
            "set_source_inclusion",
            {"source_path": source_path, "included": included, "reason": reason},
        )

    async def set_chunk_inclusion(
        self, *, chunk_id: str, included: bool, reason: str | None
    ) -> dict[str, Any]:
        return self._record(
            "set_chunk_inclusion",
            {"chunk_id": chunk_id, "included": included, "reason": reason},
        )

    async def set_source_metadata(
        self, *, metadata: dict[str, Any], source_path: str
    ) -> dict[str, Any]:
        return self._record(
            "set_source_metadata",
            {"metadata": metadata, "source_path": source_path},
        )

    def _status(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "ready": True,
            "stale": False,
            "project_name": "example",
            "generation_id": "20260101T000000Z-abcdef",
            "version": {"server": "0.16.0", "installed": "0.16.0"},
            "chunk_count": 1412,
            "categories": [{"category": "research", "searchable_source_count": 2}],
            "generations": [{"generation_id": "20260101T000000Z-abcdef"}],
            "blocked_by": [],
            "degraded": [],
            "message": "Everything is current.",
        }

    def _ingest(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments == {"force_recompute": False}
        return {
            "status": "ready",
            "generation_changed": True,
            "generation_id": "20260101T000000Z-abcdef",
            "document_count": 59,
            "chunk_count": 14072,
            "message": "Indexed 59 documents and 14072 chunks.",
        }

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments["retrieval_method"] == "hybrid"
        assert arguments["rerank"] is True
        return {
            "query": arguments["query"],
            "generation_id": "20260101T000000Z-abcdef",
            "stale": False,
            "generation_upgrade_required": False,
            "reranked": True,
            "rerank_fallback": None,
            "relevance_limited": False,
            "retrieval_method": "hybrid",
            "fusion": {"method": "weighted_reciprocal_rank_fusion", "rrf_k": 60},
            "candidate_count": 9,
            "result_count": 1,
            "filters": {"authors_any": [], "window_is_whole_corpus": True},
            "collapsed_repetitions": {"repetitions_collapsed": 0, "pairs": []},
            "hits": [
                {
                    "rank": 1,
                    "retrieval_rank": 1,
                    "chunk_id": "chk_one",
                    "document_id": "doc_one",
                    "source_id": "src_one",
                    "source_relative_path": "evidence.pdf",
                    "title": "Citable Evidence",
                    "authors": ["A. Researcher"],
                    "year": 2025,
                    "doi": "10.1/example",
                    "locator": {"page": 3, "type": "pdf_page"},
                    "citation": "A. Researcher, Citable Evidence (2025), p. 3",
                    "text": "cleaned semantic text",
                    "text_fidelity": "cleaned_semantic_text",
                    "direct_quote_safe": False,
                    "text_notes": [],
                    "dense_truncated": False,
                    "fusion_score": 0.032,
                    "rerank_score": 4.2,
                }
            ],
            "message": "One passage.",
        }

    def _find_source(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments["query"] == "crawford"
        return {
            "generation_id": "20260101T000000Z-abcdef",
            "query": "crawford",
            "match_count": 1,
            "truncated": False,
            "matches": [
                {
                    "source_id": "src_one",
                    "source_relative_path": "evidence.pdf",
                    "title": "Citable Evidence",
                    "authors": ["A. Researcher"],
                    "exists": True,
                    "included": True,
                    "indexed_in_current_generation": True,
                    "has_reviewed_metadata": False,
                    "searchable": True,
                }
            ],
            "message": "1 source matching 'crawford' is searchable now.",
        }

    def _get_passage(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments["chunk_id"] == "chk_one"
        return {
            "generation_id": "20260101T000000Z-abcdef",
            "requested_chunk_id": "chk_one",
            "context": [
                {
                    "chunk_id": "chk_neighbour",
                    "source_id": "src_one",
                    "source_relative_path": "evidence.pdf",
                    "title": "Citable Evidence",
                    "authors": ["A. Researcher"],
                    "locator": {"page": 4},
                    "text": "the passage before it",
                    "direct_quote_safe": False,
                }
            ],
        }

    def _set_source_inclusion(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments["included"] is False
        assert arguments["reason"] == "Reviewed duplicate."
        return {
            "status": "changed",
            "source_id": "src_two",
            "source_relative_path": arguments["source_path"],
            "source_path": f"sources/{arguments['source_path']}",
            "included": False,
            "reason": arguments["reason"],
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": True,
            "message": "Source exclusion saved and enforced for current retrieval.",
        }

    def _set_chunk_inclusion(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert arguments["included"] is False
        return {
            "status": "changed",
            "chunk_id": arguments["chunk_id"],
            "source_relative_path": "evidence.pdf",
            "locator": "p. 3",
            "included": False,
            "reason": arguments["reason"],
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": False,
            "in_current_generation": True,
            "message": "Chunk exclusion saved and enforced for current retrieval.",
        }

    def _set_source_metadata(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "changed",
            "source_id": "src_one",
            "source_relative_path": arguments["source_path"],
            "source_path": f"sources/{arguments['source_path']}",
            "metadata": arguments["metadata"],
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": False,
            "message": "Reviewed metadata saved and applied to current retrieval.",
        }


APP_STATE = {
    "ui_url": "http://127.0.0.1:5051/",
    "ui_ready": True,
    "ui_error": None,
    "mcp_url": "http://127.0.0.1:5051/mcp",
    "mcp_clients": 3,
}


def _server(project: Path, service: _Service, state: dict[str, Any] | None = None):
    config = resolve_config(project, vanilla_executable=sys.executable)

    async def connect() -> _Service:
        return service

    return create_mcp(
        config,
        connect=connect,
        app_state=lambda: APP_STATE if state is None else state,
    )


def _client(project: Path, service: _Service | None = None, state: dict | None = None):
    return Client(_server(project, service or _Service(), state))


async def test_the_surface_declares_eight_tools_and_one_resource(project: Path) -> None:
    async with _client(project) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()

    assert tuple(tool.name for tool in tools) == TOOLS
    assert tuple(str(item.uri) for item in resources) == ("research://status",)
    assert SERVER_NAME == AGENT_BRIDGE_NAME == "research-rag"


async def test_the_client_handshake_carries_the_instructions(project: Path) -> None:
    async with _client(project) as client:
        assert client.initialize_result.instructions == AGENT_INSTRUCTIONS


async def test_the_serialized_tools_list_stays_inside_its_budget(project: Path) -> None:
    """Descriptions and schemas are read by every client before any answer arrives.

    A client that lists tools pays for all eight at once, so their size is a
    presentation cost rather than a rounding error. The bound is measured on the
    declared surface, so it tracks the text an agent actually receives.
    """

    async with _client(project) as client:
        tools = await client.list_tools()

    listing = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "annotations": tool.annotations,
            }
            for tool in tools
        ],
        default=str,
    ).encode("utf-8")

    assert len(listing) < 12_500, f"tools/list serialized to {len(listing)} bytes"
    for tool in tools:
        assert tool.description, tool.name
        assert len(tool.description) < 900, tool.name
        for name, schema in tool.inputSchema["properties"].items():
            assert schema["description"], f"{tool.name}.{name}"


async def test_no_tool_offers_engine_tuning_to_a_caller(project: Path) -> None:
    """A caller chooses what to ask for, never how the engine answers it."""

    async with _client(project) as client:
        schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}

    everything = json.dumps(schemas)
    for engine_key in (
        "retrieval_method",
        "rerank",
        "rerank_model",
        "rerank_max_candidates",
        "context_chunks",
        "chunk_size",
        "rrf_k",
        "prf",
        "embedding_model",
        "dense",
        "bm25",
    ):
        assert engine_key not in everything, engine_key

    assert tuple(schemas["search"]["properties"]) == (
        "query",
        "top_k",
        "categories_any",
        "projects_any",
        "keywords",
        "languages_any",
        "authors_any",
        "titles_any",
        "source_ids",
        "exclude_source_ids",
    )
    assert set(schemas["search"]["required"]) == {"query"}
    assert schemas["search"]["properties"]["top_k"]["minimum"] == 1
    assert schemas["search"]["properties"]["top_k"]["maximum"] == 50
    assert set(schemas["get_passage"]["properties"]) == {"chunk_id"}
    assert set(schemas["find_source"]["properties"]) == {"query", "limit"}


async def test_the_metadata_schema_states_the_types_and_the_replace_rule(
    project: Path,
) -> None:
    """A free-form mapping is only safe to send when its fields and types are named."""

    async with _client(project) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    schema = tools["set_source_metadata"].inputSchema
    assert set(schema["properties"]) == {"source_path", "metadata"}
    assert set(schema["required"]) == {"source_path", "metadata"}
    description = schema["properties"]["metadata"]["description"]
    for field in (
        "title",
        "doi",
        "year",
        "authors",
        "categories",
        "keywords",
        "project",
        "language",
    ):
        assert f"`{field}`" in description, field
    assert "ISO 639-1" in description
    assert "replaces" in description
    assert "clears it" in description


async def test_the_annotations_tell_a_client_when_to_ask_first(project: Path) -> None:
    """`find_source` is a lookup, and it is not read-only: it syncs the catalog."""

    async with _client(project) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for name in ("status", "search", "get_passage"):
        assert tools[name].annotations.readOnlyHint is True, name
        assert tools[name].annotations.destructiveHint is False, name
    # A lookup that records every source id this project has seen may rewrite
    # `source-catalog.json`, so a client that honours readOnlyHint must not be told
    # it cannot write.
    assert tools["find_source"].annotations.readOnlyHint is False
    assert tools["find_source"].annotations.destructiveHint is False
    assert tools["ingest"].annotations.readOnlyHint is False
    for name in ("set_source_inclusion", "set_chunk_inclusion"):
        assert tools[name].annotations.readOnlyHint is False, name
        assert tools[name].annotations.destructiveHint is True, name
        assert tools[name].annotations.idempotentHint is True, name
    assert tools["set_source_metadata"].annotations.readOnlyHint is False
    assert tools["set_source_metadata"].annotations.destructiveHint is False
    for tool in tools.values():
        assert tool.annotations.openWorldHint is False, tool.name


async def test_the_instructions_explain_no_engine_they_cannot_act_on() -> None:
    """The reader acts on calls and conditions, so the guidance carries those."""

    lowered = AGENT_INSTRUCTIONS.lower()
    for internal in ("bm25", "dense semantic", "cpu", "reciprocal rank", "qdrant"):
        assert internal not in lowered, internal
    for tool_name in TOOLS:
        assert tool_name in AGENT_INSTRUCTIONS, tool_name
    assert "direct_quote_safe" in AGENT_INSTRUCTIONS
    # The reranker falling back leaves the lexical and dense order in place, which is
    # a ranking and not an absence of one.
    assert "unranked" not in lowered
    assert "lexical" in lowered and "dense" in lowered and "rerank_fallback" in lowered
    assert len(AGENT_INSTRUCTIONS.encode("utf-8")) < 3000
    # The command line is a reader's, and this surface must not offer it as a call.
    assert "research-rag " not in AGENT_INSTRUCTIONS


async def test_status_reports_the_workspace_and_nothing_else_of_the_app(
    project: Path,
) -> None:
    """The app's state is the app's: a client count and a port are not the project's."""

    service = _Service()
    async with _client(project, service) as client:
        answer = await client.call_tool("status", {})

    assert answer.data["ready"] is True
    assert answer.data["generation_id"] == "20260101T000000Z-abcdef"
    assert answer.data["ui_url"] == APP_STATE["ui_url"]
    for absent in ("mcp_clients", "mcp_url", "ui_ready", "chunk_count", "categories"):
        assert absent not in answer.data, absent
    # A project with nothing to fix is a verdict, not a confirmation.
    assert "message" not in answer.data

    failing = _Service()
    async with _client(
        project,
        failing,
        {**APP_STATE, "ui_ready": False, "ui_error": "The workspace did not start."},
    ) as client:
        degraded = await client.call_tool("status", {})

    assert degraded.data["ui_error"] == "The workspace did not start."
    assert "ui_ready" not in degraded.data


async def test_a_search_answer_carries_handles_and_never_the_ranking(
    project: Path,
) -> None:
    service = _Service()
    async with _client(project, service) as client:
        answer = await client.call_tool("search", {"query": "heron", "top_k": 3})

    assert answer.data["hits"] == [
        {
            "chunk_id": "chk_one",
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "title": "Citable Evidence",
            "authors": ["A. Researcher"],
            "locator": {"page": 3},
            "text": "cleaned semantic text",
            "direct_quote_safe": False,
        }
    ]
    serialized = json.dumps(answer.data)
    for absent in (
        "query",
        "fusion_score",
        "rerank_score",
        "candidate_count",
        "citation",
        "doi",
        "text_notes",
        "Message",
    ):
        assert absent not in serialized, absent
    # The caller's own next call is reachable from what came back.
    assert service.calls == [
        (
            "search",
            {
                "query": "heron",
                "top_k": 3,
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
        )
    ]


async def test_a_passage_context_carries_the_same_handles_as_a_hit(
    project: Path,
) -> None:
    async with _client(project) as client:
        answer = await client.call_tool("get_passage", {"chunk_id": "chk_one"})

    assert answer.data["context"][0]["direct_quote_safe"] is False
    assert answer.data["context"][0]["chunk_id"] == "chk_neighbour"
    assert answer.data["context"][0]["source_id"] == "src_one"
    assert answer.data["context"][0]["locator"] == {"page": 4}
    assert set(answer.data) == {"generation_id", "context"}


async def test_a_review_decision_answers_in_fields_not_sentences(project: Path) -> None:
    async with _client(project) as client:
        source = await client.call_tool(
            "set_source_inclusion",
            {
                "source_path": "duplicate.pdf",
                "included": False,
                "reason": "Reviewed duplicate.",
            },
        )
        chunk = await client.call_tool(
            "set_chunk_inclusion",
            {
                "chunk_id": "chk_one",
                "included": False,
                "reason": "Misread extraction.",
            },
        )
        metadata = await client.call_tool(
            "set_source_metadata",
            {
                "source_path": "evidence.pdf",
                "metadata": {"title": "Reviewed", "year": 2025},
            },
        )

    assert source.data["included"] is False
    assert source.data["reason"] == "Reviewed duplicate."
    assert "message" not in source.data
    assert chunk.data["included"] is False
    assert "message" not in chunk.data
    assert metadata.data["metadata"] == {"title": "Reviewed", "year": 2025}
    assert "message" not in metadata.data


async def test_the_full_detail_option_keeps_the_whole_payload(project: Path) -> None:
    """The lean answer is the default, and the diagnostic payload is one setting away."""

    config = resolve_config(
        project, vanilla_executable=sys.executable, tool_detail="full"
    )
    service = _Service()

    async def connect() -> _Service:
        return service

    async with Client(
        create_mcp(config, connect=connect, app_state=lambda: APP_STATE)
    ) as client:
        full = await client.call_tool("search", {"query": "heron"})

    lean = present_tool_response("search", full.data, detail=LEAN_TOOL_DETAIL)
    assert config.tool_detail == FULL_TOOL_DETAIL
    assert full.data["hits"][0]["rerank_score"] == 4.2
    assert full.data["hits"][0]["text_fidelity"] == "cleaned_semantic_text"
    assert "rerank_score" not in json.dumps(lean)
    assert len(json.dumps(lean)) * 2 < len(json.dumps(full.data))
