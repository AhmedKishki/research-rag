"""The operation table: what each surface may forward, checked in one place."""

from __future__ import annotations

import pytest

from research_rag.core.operations import MAXIMUM_TOP_K, arguments
from research_rag.project.policy import ResearchError


def test_a_search_takes_its_defaults_and_no_diagnostic_choice() -> None:
    assert arguments("search", {"query": "labour"}) == {
        "query": "labour",
        "top_k": 10,
        "categories_any": None,
        "projects_any": None,
        "keywords": None,
        "languages_any": None,
        "authors_any": None,
        "titles_any": None,
        "source_ids": None,
        "exclude_source_ids": None,
    }
    for diagnostic in ({"rerank": False}, {"retrieval_method": "bm25"}):
        with pytest.raises(ResearchError, match="does not take"):
            arguments("search", {"query": "labour", **diagnostic})


def test_only_the_command_line_names_a_diagnostic_choice() -> None:
    checked = arguments(
        "search",
        {"query": "labour", "rerank": False, "retrieval_method": "bm25"},
        command_line=True,
    )
    assert checked["rerank"] is False
    assert checked["retrieval_method"] == "bm25"
    # An unnamed diagnostic is not passed, so the engine's own default holds.
    assert "rerank" not in arguments("search", {"query": "q"}, command_line=True)


def test_only_the_command_line_names_sources_to_ingest_or_uningest() -> None:
    """An agent and the workspace build every source; neither takes one out."""

    with pytest.raises(ResearchError, match="does not take: sources"):
        arguments("ingest", {"sources": ["a.pdf"]})
    with pytest.raises(ResearchError, match="command line only"):
        arguments("uningest", {"sources": ["a.pdf"]})
    assert arguments("ingest", {}) == {"force_recompute": False}
    assert arguments("ingest", {"sources": ["a.pdf"]}, command_line=True) == {
        "force_recompute": False,
        "sources": ["a.pdf"],
    }
    assert arguments("uningest", {"sources": "a.pdf,b.pdf"}, command_line=True) == {
        "sources": ["a.pdf", "b.pdf"]
    }


def test_a_query_string_is_read_as_each_arguments_kind() -> None:
    assert (
        arguments("search", {"query": "q", "top_k": "5", "keywords": "a, b,"})["top_k"]
        == 5
    )
    assert arguments("list_sources", {"categories": "theory,history"})[
        "categories"
    ] == ["theory", "history"]
    assert arguments("get_passage", {"chunk_id": "c", "context_chunks": "0"}) == {
        "chunk_id": "c",
        "context_chunks": 0,
    }
    assert arguments("ingest", {"force_recompute": "true"}) == {"force_recompute": True}
    assert arguments("search_stats", {"days": "7"})["days"] == 7.0


@pytest.mark.parametrize(
    ("name", "payload", "refusal"),
    [
        ("search", {"query": "q", "unsupported": True}, "does not take: unsupported"),
        ("ingest", {"chunk_size": 384}, "does not take: chunk_size"),
        ("search", {"query": "  "}, "query must be a non-empty string"),
        ("search", {}, "query is required"),
        ("search", {"query": "q", "top_k": MAXIMUM_TOP_K + 1}, "top_k must be between"),
        ("search", {"query": "q", "top_k": True}, "top_k must be an integer"),
        ("search", {"query": "q", "keywords": ["a", ""]}, "keywords must be a list"),
        ("ingest", {"force_recompute": "yes"}, "force_recompute must be true or false"),
        ("get_passage", {"chunk_id": "c", "context_chunks": 6}, "between 0 and 5"),
        ("use_generation", {}, "generation_id is required"),
        ("use_generation", {"generation_id": 7}, "generation_id must be a string"),
        ("use_generation", {"generation_id": "  "}, "a non-empty string"),
        ("remove_generation", {"generation_id": "g"}, "confirm is required"),
        ("set_source_inclusion", {"source_path": "a.pdf"}, "included is required"),
        ("set_source_metadata", {"metadata": []}, "metadata must be an object"),
        (
            "settings_write",
            {"values": {}, "expected_revision": "r"},
            "values must be an object naming at least one key",
        ),
        ("search_stats", {"days": 0}, "days must be a positive number"),
        ("unknown", {}, "Unknown operation: unknown"),
    ],
)
def test_an_argument_is_refused_by_name(name, payload, refusal) -> None:
    with pytest.raises(ResearchError, match=refusal):
        arguments(name, payload)
