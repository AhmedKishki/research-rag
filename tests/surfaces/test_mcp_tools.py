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
    "synthesise",
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

    async def synthesise(self, question: str, **filters: Any) -> dict[str, Any]:
        return self._record("synthesise", {"question": question, **filters})

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

    def _synthesise(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": "deepseek-flash",
            "generated": True,
            "synthesis": "It dissolved because its members left [S1].",
            "references": "- [S1] Mensah, *Grain Cooperatives*, p. 112.",
            "provenance": "1. S1: verbatim",
            "reasoning": "- Z: it dissolved.",
            "gaps": 0,
            "sources": [{"label": "S1", "chunk_id": "c-1"}],
            "sent": {"endpoint": "https://api.deepseek.com", "chunk_ids": ["c-1"]},
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
        # No method reaches the engine: an agent gets no retrieval mode, so the
        # engine resolves the method the generation can serve.
        assert "retrieval_method" not in arguments
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
                    "title": "Citable Evidence",
                    "authors": ["A. Researcher"],
                    "year": 2025,
                    "doi": "10.1/example",
                    "locator": {"page": 3, "type": "pdf_page"},
                    "citation": "A. Researcher, Citable Evidence (2025), p. 3",
                    "text": "cleaned semantic text",
                    "text_fidelity": "cleaned_semantic_text",
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
                    "title": "Citable Evidence",
                    "authors": ["A. Researcher"],
                    "locator": {"page": 4},
                    "text": "the passage before it",
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


async def test_the_advertised_contract_and_handshake(project: Path) -> None:
    async with _client(project) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        instructions = client.initialize_result.instructions

    assert tuple(tool.name for tool in tools) == TOOLS
    assert tuple(str(item.uri) for item in resources) == ("research://status",)
    assert SERVER_NAME == AGENT_BRIDGE_NAME == "research-rag"
    assert instructions == AGENT_INSTRUCTIONS

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

    schemas = {tool.name: tool.inputSchema for tool in tools}

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
    assert tuple(schemas["synthesise"]["properties"]) == (
        "question",
        "reasoning",
        "verbose",
    )
    assert set(schemas["synthesise"]["required"]) == {"question"}
    assert schemas["synthesise"]["properties"]["verbose"]["default"] is False
    assert set(schemas["get_passage"]["properties"]) == {"chunk_id"}
    assert set(schemas["find_source"]["properties"]) == {"query", "limit"}

    schema = schemas["set_source_metadata"]
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

    tools = {tool.name: tool for tool in tools}

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
    # A synthesis is the one call that reaches past this machine, and it writes
    # nothing a reader keeps.
    assert tools["synthesise"].annotations.openWorldHint is True
    assert tools["synthesise"].annotations.readOnlyHint is True
    assert tools["synthesise"].annotations.idempotentHint is False
    for tool in tools.values():
        if tool.name != "synthesise":
            assert tool.annotations.openWorldHint is False, tool.name

    lowered = instructions.lower()
    for tool_name in TOOLS:
        assert tool_name in instructions, tool_name
    assert "cleaned text, not a transcript" in lowered
    # The reranker falling back leaves the lexical and dense order in place, which is
    # a ranking and not an absence of one.
    assert "unranked" not in lowered
    assert "lexical" in lowered and "dense" in lowered and "rerank_fallback" in lowered
    assert len(instructions.encode("utf-8")) < 3000
    # The command line is a reader's, and this surface must not offer it as a call.
    assert "research-rag " not in instructions


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
            "title": "Citable Evidence",
            "authors": ["A. Researcher"],
            "locator": {"page": 3},
            "text": "cleaned semantic text",
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

    assert "direct_quote_safe" not in answer.data["context"][0]
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


async def test_an_unexpected_failure_is_logged_with_its_traceback(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """FastMCP logs a `ToolError` without a traceback, so the surface logs it."""

    class Failing(_Service):
        async def get_passage(self, chunk_id: str) -> dict[str, Any]:
            raise KeyError("document_id")

    async with _client(project, Failing()) as client:
        answer = await client.call_tool(
            "get_passage", {"chunk_id": "chk_one"}, raise_on_error=False
        )

    assert answer.is_error
    assert "serving terminal" in answer.content[0].text
    (record,) = [item for item in caplog.records if item.exc_info]
    assert isinstance(record.exc_info[1], KeyError)


@pytest.mark.parametrize("verbose", [False, True])
async def test_a_synthesis_shows_provenance_and_reasoning_only_when_verbose(
    project: Path, verbose: bool
) -> None:
    service = _Service()
    async with _client(project, service) as client:
        answer = await client.call_tool(
            "synthesise",
            {
                "question": "Why did it dissolve?",
                "reasoning": "low",
                "verbose": verbose,
            },
        )

    assert answer.data["synthesis"] == "It dissolved because its members left [S1]."
    assert answer.data["references"].startswith("- [S1] Mensah")
    assert ("provenance" in answer.data, "reasoning" in answer.data) == (
        verbose,
        verbose,
    )
    assert service.calls[0][0] == "synthesise"
    assert service.calls[0][1]["question"] == "Why did it dissolve?"
    assert service.calls[0][1]["reasoning"] == "low"
