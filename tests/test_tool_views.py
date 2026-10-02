from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from research_rag.config import (
    ConfigurationError,
    resolve_config,
)
from research_rag.service import ResearchError
from research_rag.settings import (
    FULL_TOOL_DETAIL,
    LEAN_TOOL_DETAIL,
)
from research_rag.tool_views import present_tool_response

# Keys the service builds for ranking, extraction, and storage diagnostics.
DIAGNOSTIC_KEYS = {
    "annotations",
    "available_retrieval_methods",
    "candidate_count",
    "candidate_depth",
    "candidate_distinct_reference_count",
    "component_ranks",
    "component_scores",
    "content_kind",
    "dense_fidelity",
    "dense_gate",
    "dense_truncated",
    "distinct_reference_count",
    "doi",
    "document_id",
    "embedding_model",
    "embedding_model_revision",
    "embedding_token_count",
    "filters",
    "fusion",
    "fusion_score",
    "generation_root",
    "grouping",
    "grouping_limited",
    "grouping_skipped_candidate_count",
    "ignored_extensions",
    "last_build_metrics",
    "match_kind",
    "metadata_pending_source_paths",
    "metadata_provenance",
    "metadata_warnings",
    "passages_per_reference",
    "quality_flags",
    "rank",
    "rejected_candidate_examples",
    "rejected_candidates",
    "relevance_limited",
    "relevance_policy",
    "rerank_requested",
    "rerank_score",
    "reranker_model",
    "retrieval",
    "retrieval_method",
    "retrieval_rank",
    "selection_policy",
    "source_path",
    "text_fidelity",
    "withheld_candidates",
    "year",
}


def test_lean_passage_shape() -> None:
    lean = present_tool_response("search", _search_payload(), detail=LEAN_TOOL_DETAIL)

    # The query is the caller's own, the ranking is not published, and neither `stale` nor
    # `reranked` is news when both are what every answer is.
    assert set(lean) == {"generation_id", "hits"}
    assert lean["hits"] == [
        {
            "chunk_id": "chk_one",
            "source_relative_path": "evidence.pdf",
            "authors": ["A. Researcher"],
            "locator": {"page": 3},
            "text": "cleaned semantic text",
        },
        {
            "chunk_id": "chk_two",
            "source_relative_path": "anonymous.pdf",
            "locator": {"section": "chapter.xhtml"},
            "text": "second passage",
        },
    ]

    # The reference is deliberately not citation-ready, and neither the quote-safety rule nor
    # the advisory script note is repeated on every passage.
    for key in (
        "citation",
        "direct_quote_safe",
        "source_id",
        "text_notes",
        "title",
    ):
        assert key not in json.dumps(lean)


def test_lean_response_never_contains_a_diagnostic_key() -> None:
    for operation, payload in _every_tool_payload().items():
        lean = present_tool_response(operation, payload, detail=LEAN_TOOL_DETAIL)
        leaked = _keys(lean) & DIAGNOSTIC_KEYS
        assert not leaked, f"{operation} leaked {sorted(leaked)}"
        assert len(json.dumps(lean, ensure_ascii=False)) < len(
            json.dumps(payload, ensure_ascii=False)
        )


def test_full_detail_returns_the_service_payload_unchanged() -> None:
    payload = _search_payload()

    assert present_tool_response("search", payload, detail=FULL_TOOL_DETAIL) == payload


def test_search_reports_rerank_state_and_unknown_ids() -> None:
    payload = _search_payload(
        reranked=False,
        rerank_fallback={"reason": "reranker_model_unavailable", "effect": "unranked"},
        unknown_source_ids=["src_missing"],
    )
    lean = present_tool_response("search", payload, detail=LEAN_TOOL_DETAIL)

    assert lean["reranked"] is False
    assert lean["rerank_fallback"]["reason"] == "reranker_model_unavailable"
    assert lean["unresolved_source_ids"] == ["src_missing"]
    assert "unresolved_exclude_source_ids" not in lean


def test_search_reports_an_applied_bibliographic_filter() -> None:
    """A filtered answer says which filter it applied, so an empty one explains itself."""

    unfiltered = present_tool_response(
        "search", _search_payload(), detail=LEAN_TOOL_DETAIL
    )
    assert "applied_filters" not in unfiltered

    filtered = present_tool_response(
        "search",
        _search_payload(authors_any=["Crawford"], titles_any=["Atlas of AI"]),
        detail=LEAN_TOOL_DETAIL,
    )
    assert filtered["applied_filters"] == {
        "authors_any": ["Crawford"],
        "titles_any": ["Atlas of AI"],
    }
    assert "categories_any" not in filtered


def test_search_omits_an_upgrade_note_that_is_not_required() -> None:
    lean = present_tool_response("search", _search_payload(), detail=LEAN_TOOL_DETAIL)

    assert "generation_upgrade_required" not in lean
    assert "reference_groups" not in lean

    required = present_tool_response(
        "search",
        _search_payload(generation_upgrade_required=True),
        detail=LEAN_TOOL_DETAIL,
    )
    assert required["generation_upgrade_required"] is True


def test_passage_context_is_lean_and_keeps_no_rank() -> None:
    payload = {
        "generation_id": "20260101T000000Z-abcdef",
        "requested_chunk_id": "chk_one",
        "context": [{key: value for key, value in _hit().items() if key != "rank"}],
    }
    lean = present_tool_response("get_passage", payload, detail=LEAN_TOOL_DETAIL)

    assert set(lean) == {"generation_id", "context"}
    assert "rank" not in lean["context"][0]
    assert "direct_quote_safe" not in lean["context"][0]


def test_status_names_a_blocker_and_its_remedy() -> None:
    lean = present_tool_response(
        "status",
        _status_payload(
            blocked_by=[
                {
                    "check": "vanilla_runtime",
                    "reason": "The tree differs at servers/stray.pyc.",
                    "remedy": "research-rag doctor --project-root /project "
                    "--repair-runtime",
                }
            ],
            degraded=[
                {
                    "check": "lock",
                    "reason": "Process 4321 has held the project since 09:00:00Z.",
                    "remedy": "research-rag --project-root /project stop --servers",
                }
            ],
        ),
        detail=LEAN_TOOL_DETAIL,
    )

    assert lean["blocked_by"] == [
        {
            "check": "vanilla_runtime",
            "reason": "The tree differs at servers/stray.pyc.",
            "remedy": "research-rag doctor --project-root /project --repair-runtime",
        }
    ]
    assert lean["degraded"][0]["check"] == "lock"


def test_a_blocked_surface_names_no_call_an_agent_cannot_make() -> None:
    """A server blocked on the app declares `status` and nothing else.

    Its answer says the project exists and its app is not running, and the remedy is
    a command to run in a terminal. Naming `ingest` alongside that asks for a tool
    this surface does not serve, so the answer carries only the blocker.
    """

    lean = present_tool_response(
        "status",
        _status_payload(
            ready=False,
            project_initialised=True,
            blocked_by=[
                {
                    "check": "app.serving",
                    "reason": "An app runs in a terminal and ends when that "
                    "terminal closes.",
                    "remedy": "research-rag --project 'AI and fetishism' start",
                }
            ],
        ),
        detail=LEAN_TOOL_DETAIL,
    )

    assert lean["ready"] is False
    assert lean["blocked_by"][0]["check"] == "app.serving"
    assert "requires" not in lean

    # A health check that blocks is a different condition: the corpus is there and
    # `ingest` is still the call that serves it.
    blocked_by_health = present_tool_response(
        "status",
        _status_payload(
            ready=False,
            blocked_by=[
                {
                    "check": "vanilla_runtime",
                    "reason": "The tree differs at servers/stray.pyc.",
                    "remedy": "research-rag doctor --repair-runtime",
                }
            ],
        ),
        detail=LEAN_TOOL_DETAIL,
    )

    assert blocked_by_health["requires"] == ["ingest"]


def test_status_keeps_dependencies_out_of_the_answer_when_there_are_none() -> None:
    lean = present_tool_response("status", _status_payload(), detail=LEAN_TOOL_DETAIL)

    assert "blocked_by" not in lean
    assert "degraded" not in lean


def test_status_discloses_nothing_the_payload_did_not_say() -> None:
    """Every string in the lean answer comes from the payload it projects."""

    payload = _status_payload(
        blocked_by=[
            {
                "check": "generation",
                "reason": "No generation exists; call ingest.",
                "remedy": None,
            }
        ]
    )

    lean = present_tool_response("status", payload, detail=LEAN_TOOL_DETAIL)

    values = {value for value in _values(lean)}
    for entry in lean["blocked_by"]:
        assert set(entry) == {"check", "reason", "remedy"}
        assert entry["check"] in payload["blocked_by"][0]["check"]
        assert entry["reason"] in values
    assert "degraded" not in lean
    assert json.loads(json.dumps(lean)) == lean


def _values(payload: object) -> list[str]:
    if isinstance(payload, dict):
        return [item for value in payload.values() for item in _values(value)]
    if isinstance(payload, list):
        return [item for value in payload for item in _values(value)]
    return [payload] if isinstance(payload, str) else []


def test_status_lean_states_the_verdict_and_nothing_else() -> None:
    payload = _status_payload()
    lean = present_tool_response("status", payload, detail=LEAN_TOOL_DETAIL)

    # Two booleans the caller can act on, the generation they describe, and the sentence
    # that explains them.
    assert set(lean) == {"ready", "stale", "generation_id", "message"}
    assert lean["ready"] is True
    assert lean["stale"] is False
    assert lean["generation_id"] == "20260101T000000Z-abcdef"
    assert "requires" not in lean

    legacy = present_tool_response(
        "status",
        _status_payload(
            available_retrieval_methods=["bm25"],
            hybrid_ready=False,
            hybrid_upgrade_required=True,
            generation_upgrade_required=True,
            upgrade_reasons=["bm25_only_generation"],
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert legacy["ready"] is True
    assert legacy["stale"] is False
    assert legacy["requires"] == ["ingest"]
    assert "hybrid_ready" not in legacy
    assert "generation_upgrade_required" not in legacy
    assert "upgrade_reasons" not in legacy
    assert "available_retrieval_methods" not in legacy

    # Inventories answer a question of their own and belong to the full-detail
    # payload, so a status answer stays a statement about the selected generation.
    serialized = json.dumps(lean)
    assert "generations" not in serialized
    for key in (
        "categories",
        "projects",
        "project_name",
        "created_at",
        "chunk_count",
        "discovered_source_count",
        "selected_source_count",
        "indexed_source_count",
        "searchable_source_count",
        "excluded_source_count",
        "retained_generation_count",
        "retained_generation_bytes",
        "generations",
    ):
        assert key not in serialized
        assert key in present_tool_response("status", payload, detail=FULL_TOOL_DETAIL)
    # A dependency that is fine and one that is only listed are both absent: a lean
    # answer discloses a condition, not an inventory of checks.
    assert "blocked_by" not in lean
    assert "degraded" not in lean
    assert "checks" not in lean
    assert "not_checked" not in lean
    assert "excluded_sources" not in lean
    assert "generation_upgrade_required" not in lean
    assert "upgrade_reasons" not in lean
    # The overlay applies at read time, so it is not a condition the caller acts on.
    assert "metadata_overlay_active" not in lean
    assert "metadata_pending_source_paths" not in lean
    assert "metadata_pending_source_count" not in lean
    assert "ingestion_progress" not in lean
    assert "generation_root" not in lean
    assert "version" not in lean
    assert "restart_required" not in lean


def test_status_names_every_call_the_verdict_requires() -> None:
    for overrides in (
        {"stale": True},
        {"generation_upgrade_required": True, "upgrade_reasons": ["retrieval_policy"]},
        {"hybrid_ready": False},
        {"metadata_pending_source_paths": ["a.pdf", "b.pdf"]},
        {"ingestion_progress": {"phase": "embedding", "progress": 0.4}},
        {"stale": True, "changes": {"source_exclusions_changed": True}},
    ):
        lean = present_tool_response(
            "status", _status_payload(**overrides), detail=LEAN_TOOL_DETAIL
        )
        assert lean["requires"] == ["ingest"], overrides

    pending = present_tool_response(
        "status",
        _status_payload(metadata_pending_source_paths=["a.pdf", "b.pdf"]),
        detail=LEAN_TOOL_DETAIL,
    )
    assert pending["ready"] is True
    assert pending["stale"] is False
    assert pending["requires"] == ["ingest"]
    assert "metadata_pending_source_count" not in pending

    restarting = present_tool_response(
        "status",
        _status_payload(
            version={
                "server": "0.15.0",
                "installed": "0.16.0",
                "ui": None,
                "restart_required": True,
            }
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert restarting["requires"] == ["restart_app"]
    assert "version" not in restarting

    both = present_tool_response(
        "status",
        _status_payload(
            stale=True,
            version={
                "server": "0.15.0",
                "installed": "0.16.0",
                "ui": None,
                "restart_required": True,
            },
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert both["requires"] == ["ingest", "restart_app"]


def test_status_counts_available_source_changes_and_names_missing_ones() -> None:
    current = present_tool_response(
        "status", _status_payload(), detail=LEAN_TOOL_DETAIL
    )

    assert "changes" not in current

    stale = present_tool_response(
        "status",
        _status_payload(
            stale=True,
            changes={
                "added": ["new.pdf", "another.pdf", "third.pdf"],
                "removed": ["gone.pdf"],
                "modified": ["edited.pdf"],
                "metadata_changed": True,
                "source_exclusions_changed": False,
            },
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert stale["stale"] is True
    assert stale["requires"] == ["ingest"]
    assert stale["changes"] == {
        "added_source_count": 3,
        "modified_source_count": 1,
        "removed_sources": ["gone.pdf"],
        "metadata_changed": True,
    }
    assert "new.pdf" not in json.dumps(stale)
    assert "edited.pdf" not in json.dumps(stale)


def test_status_before_the_first_ingestion_states_what_is_missing() -> None:
    payload = {
        "ready": False,
        "stale": True,
        "project_name": "example",
        "discovered_source_count": 3,
        "selected_source_count": 3,
        "excluded_source_count": 0,
        "categories": [],
        "projects": [],
        "upgrade_reasons": [],
        "metadata_overlay_active": False,
        "metadata_pending_source_paths": [],
        "ingestion_progress": None,
        "message": "No knowledge-base generation exists; call ingest.",
    }
    lean = present_tool_response("status", payload, detail=LEAN_TOOL_DETAIL)

    # No generation is `ready: false`, the corpus it would hold is `stale: true`, and the
    # call that fixes both is named.
    assert set(lean) == {"ready", "stale", "requires", "message"}
    assert lean["ready"] is False
    assert lean["stale"] is True
    assert lean["requires"] == ["ingest"]
    assert "generations" not in lean
    assert "categories" not in lean
    assert "projects" not in lean
    assert "retained_generation_count" not in lean
    assert "excluded_source_count" not in lean
    assert "discovered_source_count" not in lean
    assert "metadata_overlay_active" not in lean
    assert lean["message"] == "No knowledge-base generation exists; call ingest."


def test_find_source_lean_keeps_handles_and_availability() -> None:
    lean = present_tool_response(
        "find_source", _find_source_payload(), detail=LEAN_TOOL_DETAIL
    )

    # `included` and `exists` appear only where they withhold the source.
    assert lean["query"] == "crawford"
    assert lean["generation_id"] == "20260101T000000Z-abcdef"
    assert lean["match_count"] == 2
    assert "truncated" not in lean
    assert lean["matches"] == [
        {
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "title": "Citable Evidence",
            "authors": ["A. Researcher"],
            "indexed_in_current_generation": True,
            "has_reviewed_metadata": True,
        },
        {
            "source_id": "src_two",
            "source_relative_path": "archive/duplicate.pdf",
            "title": "Citable Evidence",
            "authors": ["A. Researcher"],
            "exists": False,
            "included": False,
            "indexed_in_current_generation": False,
        },
    ]

    capped = present_tool_response(
        "find_source",
        _find_source_payload(match_count=12, truncated=True),
        detail=LEAN_TOOL_DETAIL,
    )
    assert capped["truncated"] is True
    assert capped["match_count"] == 12


def test_find_source_lean_states_an_empty_lookup() -> None:
    lean = present_tool_response(
        "find_source",
        {
            "ready": True,
            "generation_id": "20260101T000000Z-abcdef",
            "query": "nobody",
            "match_count": 0,
            "searchable_match_count": 0,
            "truncated": False,
            "matches": [],
            "message": "No source matches 'nobody'.",
        },
        detail=LEAN_TOOL_DETAIL,
    )

    assert set(lean) == {"generation_id", "query", "message", "matches"}
    assert lean["matches"] == []
    assert "match_count" not in lean


def test_ingest_lean_discloses_anomalies_only_when_they_happened() -> None:
    lean = present_tool_response("ingest", _ingest_payload(), detail=LEAN_TOOL_DETAIL)

    assert lean["status"] == "ready"
    assert lean["generation_changed"] is True
    assert lean["document_count"] == 59
    assert lean["chunk_count"] == 14072
    # What the build reused and rebuilt is cost rather than outcome: a caller cannot
    # act on it.
    for key in (
        "reused_document_count",
        "rebuilt_document_count",
        "reused_chunk_count",
        "rebuilt_chunk_count",
        "created_vector_count",
        "reused_vector_count",
    ):
        assert key not in lean
        assert key in _ingest_payload()
    assert "discarded_corrupt_chunk_count" not in lean
    assert "withheld_chunk_reasons" not in lean
    assert "phase_timings_seconds" not in lean
    assert "embedding_model" not in lean

    loud = present_tool_response(
        "ingest",
        _ingest_payload(
            discarded_corrupt_chunk_count=2,
            withheld_chunk_count=1,
            withheld_chunk_reasons={"corrupt_text": {"count": 1}},
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert loud["discarded_corrupt_chunk_count"] == 2
    assert loud["withheld_chunk_count"] == 1
    assert loud["withheld_chunk_reasons"] == {"corrupt_text": {"count": 1}}


def test_ingest_in_progress_keeps_resume_state() -> None:
    payload = {
        "status": "in_progress",
        "generation_changed": False,
        "build_id": "20260101T000000Z-abcdef",
        "phase": "embedding",
        "progress": {"completed": 64, "total": 128, "unit": "chunks"},
        "parameters": {"chunk_size": 384},
        "created_at": "2026-01-01T00:00:00Z",
        "checkpointed_at": "2026-01-01T00:00:05Z",
        "next_action": "call_ingest_again",
        "message": "Ingestion checkpoint saved; call ingest again.",
    }
    lean = present_tool_response("ingest", payload, detail=LEAN_TOOL_DETAIL)

    assert lean["status"] == "in_progress"
    assert "generation_changed" not in lean
    assert lean["build_id"] == "20260101T000000Z-abcdef"
    assert lean["phase"] == "embedding"
    assert lean["progress"]["unit"] == "chunks"
    assert lean["next_action"] == "call_ingest_again"
    assert "parameters" not in lean
    assert "checkpointed_at" not in lean


def test_retired_tools_have_no_projection() -> None:
    for operation in ("export_bundle", "import_bundle"):
        with pytest.raises(ResearchError):
            present_tool_response(operation, {}, detail=LEAN_TOOL_DETAIL)


def test_inclusion_response_is_lean() -> None:
    inclusion = present_tool_response(
        "set_source_inclusion",
        {
            "status": "changed",
            "source_id": "src_two",
            "source_relative_path": "duplicate.pdf",
            "source_path": "sources/duplicate.pdf",
            "included": False,
            "reason": "Reviewed duplicate.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": True,
            "message": "Source exclusion saved and enforced for current retrieval.",
        },
        detail=LEAN_TOOL_DETAIL,
    )
    assert set(inclusion) == {
        "status",
        "source_id",
        "source_relative_path",
        "included",
        "reason",
        "generation_rebuild_recommended",
        "message",
    }
    assert inclusion["included"] is False
    assert inclusion["reason"] == "Reviewed duplicate."
    assert "effective_immediately" not in inclusion
    assert "source_file_changed" not in inclusion
    assert "source_path" not in inclusion


def test_a_chunk_decision_is_lean_and_says_whether_it_withholds_anything() -> None:
    """A decision about a chunk the current generation does not hold is withholding nothing
    now, and a caller that could not tell that apart would report a removal that removed
    nothing.
    """

    withheld = present_tool_response(
        "set_chunk_inclusion",
        {
            "status": "changed",
            "chunk_id": "chk_one",
            "source_relative_path": "evidence.pdf",
            "locator": "p. 3",
            "included": False,
            "reason": "Misread extraction.",
            "source_file_changed": False,
            "effective_immediately": False,
            "generation_rebuild_recommended": False,
            "in_current_generation": False,
            "message": "Chunk exclusion saved, but this generation does not hold it.",
        },
        detail=LEAN_TOOL_DETAIL,
    )
    assert set(withheld) == {
        "status",
        "chunk_id",
        "included",
        "reason",
        "effective_immediately",
        "in_current_generation",
        "message",
    }
    assert withheld["in_current_generation"] is False
    assert withheld["effective_immediately"] is False
    # The locator and the source path are what the caller's own chunk id already says.
    assert "locator" not in withheld
    assert "source_relative_path" not in withheld
    assert "generation_rebuild_recommended" not in withheld
    assert "source_file_changed" not in withheld

    present = present_tool_response(
        "set_chunk_inclusion",
        {
            "status": "changed",
            "chunk_id": "chk_one",
            "source_relative_path": "evidence.pdf",
            "locator": "p. 3",
            "included": False,
            "reason": "Misread extraction.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": False,
            "in_current_generation": True,
            "message": "Chunk exclusion saved and enforced for current retrieval.",
        },
        detail=LEAN_TOOL_DETAIL,
    )
    assert "in_current_generation" not in present
    assert "effective_immediately" not in present


def test_every_public_tool_has_a_lean_projection() -> None:
    for operation, payload in _every_tool_payload().items():
        lean = present_tool_response(operation, payload, detail=LEAN_TOOL_DETAIL)
        assert isinstance(lean, dict)


def test_unknown_tool_name_fails_loudly() -> None:
    with pytest.raises(ResearchError):
        present_tool_response("no_such_tool", {}, detail=LEAN_TOOL_DETAIL)


def test_tool_detail_defaults_to_lean_and_rejects_unknown_modes(project: Path) -> None:
    assert resolve_config(project, vanilla_executable=sys.executable).tool_detail == (
        LEAN_TOOL_DETAIL
    )
    normalized = resolve_config(
        project,
        vanilla_executable=sys.executable,
        tool_detail=" FULL ",
    )
    assert normalized.tool_detail == FULL_TOOL_DETAIL

    with pytest.raises(ConfigurationError):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            tool_detail="chatty",
        )


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


def _hit() -> dict[str, object]:
    return {
        "rank": 1,
        "retrieval_rank": 1,
        "chunk_id": "chk_one",
        "document_id": "doc_one",
        "source_id": "src_one",
        "source_path": "sources/evidence.pdf",
        "source_relative_path": "evidence.pdf",
        "title": "Citable Evidence",
        "authors": ["A. Researcher"],
        "year": 2025,
        "doi": "10.1/example",
        "categories": ["research"],
        "keywords": ["wetland"],
        "project": ["ai-and-fetishism"],
        "locator": {"page": 3, "page_label": "3", "type": "pdf_page"},
        "citation": "A. Researcher, Citable Evidence (2025), p. 3",
        "text": "cleaned semantic text",
        "text_fidelity": "cleaned_semantic_text",
        "direct_quote_safe": False,
        "text_notes": [],
        "embedding_token_count": 120,
        "dense_truncated": False,
        "content_kind": "prose",
        "annotations": [],
        "quality_flags": [],
        "metadata_provenance": {"title": "reviewed_override"},
        "metadata_warnings": [],
        "match_kind": "hybrid",
        "retrieval_method": "hybrid",
        "component_ranks": {"bm25": 1, "dense": 1},
        "component_scores": {"dense_cosine_similarity": 0.81, "bm25": None},
        "fusion_score": 0.032,
        "rerank_score": 4.2,
    }


def _anonymous_hit() -> dict[str, object]:
    return {
        **_hit(),
        "rank": 2,
        "retrieval_rank": 4,
        "chunk_id": "chk_two",
        "document_id": "doc_two",
        "source_id": "src_two",
        "source_path": "sources/anonymous.pdf",
        "source_relative_path": "anonymous.pdf",
        "title": "anonymous",
        "authors": [],
        "year": None,
        "doi": "",
        "locator": {"section_index": 4, "href": "chapter.xhtml", "type": "epub"},
        "citation": "anonymous, section chapter.xhtml",
        "text": "second passage",
        "text_notes": ["non_latin_dominant"],
        "metadata_warnings": ["title_from_filename"],
        "component_ranks": {"bm25": None, "dense": 4},
        "match_kind": "semantic",
    }


def _search_payload(**overrides: object) -> dict[str, object]:
    filters: dict[str, object] = {
        "categories_all": [],
        "categories_any": [],
        "projects_all": [],
        "projects_any": [],
        "keywords_all": [],
        "authors_any": list(overrides.pop("authors_any", [])),
        "titles_any": list(overrides.pop("titles_any", [])),
        "document_ids": [],
        "source_ids": [],
        "exclude_source_ids": [],
        "unknown_source_ids": list(overrides.pop("unknown_source_ids", [])),
        "unknown_exclude_source_ids": [],
        "active_document_count": 2,
        "note": "Filters narrow the corpus before ranking.",
    }
    payload: dict[str, object] = {
        "query": "cobalt heron amber marsh",
        "generation_id": "20260101T000000Z-abcdef",
        "stale": False,
        "staleness_checked": True,
        "generation_upgrade_required": False,
        "excluded_source_count": 1,
        "filters": filters,
        "retrieval_method": "hybrid",
        "reranked": True,
        "rerank_requested": True,
        "rerank_fallback": None,
        "candidate_depth": 24,
        "candidate_count": 9,
        "candidate_distinct_reference_count": 2,
        "requested_top_k": 6,
        "fusion": {"method": "weighted_reciprocal_rank_fusion", "rrf_k": 60},
        "relevance_policy": {"dense_minimum_cosine_similarity": 0.72},
        "selection_policy": {
            "method": "greedy_source_diversity",
            "source_diversity_penalty": 0.25,
        },
        "rejected_candidates": {"dense_below_threshold": 11},
        "withheld_candidates": {"policy": "corruption_evidence_only", "total": 0},
        "dense_fidelity": {"embedding_maximum_tokens": 512},
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": "52398278842ec682c6f32300af41344b1c0b0bb2",
        "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "reranker_model_revision": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        "result_count": 2,
        "distinct_reference_count": 2,
        "relevance_limited": False,
        "hits": [_hit(), _anonymous_hit()],
    }
    payload.update(overrides)
    return payload


def _status_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ready": True,
        "stale": False,
        "project_root": "/project",
        "project_id": "121304a5-378c-4384-a0c8-9db4476cec47",
        "project_name": "example",
        "source_root": "/project/sources",
        "state_root": "/state",
        "runtime_root": None,
        "portable_root": "/project/.research-rag",
        "model_cache_root": "/cache/models",
        "version": {
            "server": "0.15.0",
            "installed": "0.15.0",
            "restart_required": False,
        },
        "generation_id": "20260101T000000Z-abcdef",
        "created_at": "2026-01-01T00:00:00Z",
        "discovered_source_count": 3,
        "selected_source_count": 2,
        "indexed_source_count": 2,
        "searchable_source_count": 2,
        "excluded_source_count": 1,
        "excluded_sources": [{"source_id": "src_two", "reason": "Reviewed duplicate."}],
        "chunk_count": 12,
        "categories": [{"category": "research", "searchable_source_count": 2}],
        "projects": [{"project": "example", "searchable_source_count": 2}],
        "allowed_formats": [".epub", ".pdf"],
        "ignored_extensions": {".md": 4},
        "default_retrieval_method": "hybrid",
        "available_retrieval_methods": ["bm25", "dense", "hybrid"],
        "hybrid_ready": True,
        "hybrid_upgrade_required": False,
        "generation_upgrade_required": False,
        "upgrade_reasons": [],
        "retrieval": {"available_methods": ["bm25", "dense", "hybrid"]},
        "last_build_metrics": {"reused_chunk_count": 12},
        "ingestion_progress": None,
        "source_exclusion_revision": "a" * 64,
        "metadata_revision": "b" * 64,
        "generation_metadata_revision": "c" * 64,
        "metadata_overlay_active": False,
        "metadata_pending_source_paths": [],
        "generation_metadata_snapshot_outdated": False,
        "changes": {
            "added": [],
            "removed": [],
            "modified": [],
            "metadata_changed": False,
            "source_exclusions_changed": False,
        },
        "generation_root": "/state/generations/20260101T000000Z-abcdef",
        "message": "Everything is current.",
        "generations": [
            {
                "generation_id": "20260101T000000Z-abcdef",
                "is_current": True,
                "created_at": "2026-01-01T00:00:00Z",
                "chunk_count": 12,
                "document_count": 2,
                "schema_version": 5,
                "file_count": 14,
                "size_bytes": 4096,
            }
        ],
        "retained_generation_count": 1,
        "retained_generation_bytes": 4096,
        "checks": [
            {
                "check": "project_identity",
                "state": "ok",
                "reason": "This project owns its state root.",
                "remedy_command": None,
            }
        ],
        "blocked_by": [],
        "degraded": [],
        "not_checked": [],
    }
    payload.update(overrides)
    return payload


def _find_source_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ready": True,
        "generation_id": "20260101T000000Z-abcdef",
        "query": "crawford",
        "match_count": 2,
        "searchable_match_count": 1,
        "truncated": False,
        "matches": [
            {
                "source_id": "src_one",
                "source_relative_path": "evidence.pdf",
                "source_path": "sources/evidence.pdf",
                "title": "Citable Evidence",
                "authors": ["A. Researcher"],
                "exists": True,
                "included": True,
                "indexed_in_current_generation": True,
                "has_reviewed_metadata": True,
                "searchable": True,
            },
            {
                "source_id": "src_two",
                "source_relative_path": "archive/duplicate.pdf",
                "source_path": "sources/archive/duplicate.pdf",
                "title": "Citable Evidence",
                "authors": ["A. Researcher"],
                "exists": False,
                "included": False,
                "indexed_in_current_generation": False,
                "has_reviewed_metadata": False,
                "searchable": False,
            },
        ],
        "message": "1 of the 2 sources matching 'crawford' are searchable now.",
    }
    payload.update(overrides)
    return payload


def _ingest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ready",
        "generation_changed": True,
        "generation_id": "20260101T000000Z-abcdef",
        "generation_root": "/state/generations/20260101T000000Z-abcdef",
        "source_file_count": 63,
        "excluded_source_count": 4,
        "document_count": 59,
        "pdf_count": 58,
        "epub_count": 1,
        "extraction_unit_count": 14072,
        "chunk_count": 14072,
        "content_kind_counts": {"prose": 14000},
        "reused_document_count": 58,
        "rebuilt_document_count": 1,
        "reused_chunk_count": 14044,
        "rebuilt_chunk_count": 28,
        "created_vector_count": 28,
        "reused_vector_count": 14044,
        "discarded_empty_chunk_count": 0,
        "discarded_symbol_only_chunk_count": 0,
        "discarded_corrupt_chunk_count": 0,
        "excluded_corrupt_unit_count": 0,
        "dense_truncated_chunk_count": 0,
        "withheld_chunk_count": 0,
        "withheld_chunk_reasons": {},
        "ignored_extensions": {".md": 61},
        "default_retrieval_method": "hybrid",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "phase_timings_seconds": {"extraction": 2.1},
    }
    payload.update(overrides)
    return payload


def _every_tool_payload() -> dict[str, dict[str, object]]:
    return {
        "status": _status_payload(),
        "ingest": _ingest_payload(),
        "search": _search_payload(),
        "find_source": _find_source_payload(),
        "get_passage": {
            "generation_id": "20260101T000000Z-abcdef",
            "requested_chunk_id": "chk_one",
            "context": [_hit()],
        },
        "set_source_inclusion": {
            "status": "changed",
            "source_id": "src_two",
            "source_relative_path": "duplicate.pdf",
            "source_path": "sources/duplicate.pdf",
            "included": False,
            "reason": "Reviewed duplicate.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": True,
            "message": "Source exclusion saved and enforced for current retrieval.",
        },
        "set_chunk_inclusion": {
            "status": "changed",
            "chunk_id": "chk_one",
            "source_relative_path": "evidence.pdf",
            "locator": "p. 3",
            "included": False,
            "reason": "Misread extraction.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": False,
            "in_current_generation": True,
            "message": "Chunk exclusion saved and enforced for current retrieval.",
        },
    }


def test_metadata_response_is_lean() -> None:
    metadata = present_tool_response(
        "set_source_metadata",
        {
            "status": "changed",
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "source_path": "sources/evidence.pdf",
            "metadata": {"title": "Reviewed", "keywords": ["theory"]},
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": False,
            "message": "Reviewed metadata saved and applied to current retrieval.",
        },
        detail=LEAN_TOOL_DETAIL,
    )
    assert set(metadata) == {
        "status",
        "source_id",
        "source_relative_path",
        "metadata",
        "message",
    }
    assert metadata["metadata"] == {"title": "Reviewed", "keywords": ["theory"]}
    assert metadata["message"].startswith("Reviewed metadata saved")
    assert "source_file_changed" not in metadata
