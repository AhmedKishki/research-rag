"""Projections from service payloads to MCP tool answers.

Every tool answers with the lean projection, which `present_tool_response`
applies. `--tool-detail full` returns the service payload unchanged.

One rule decides what a lean answer carries: a field is here when a caller can
act on it or could not otherwise account for it. `stale`, a blocker, a filter
that removed every source, and a reranker that did not run all qualify. The
query, the timing, the ranking scores, and the counts the caller could make for
itself do not. A field whose value is the ordinary case is left out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .settings import FULL_TOOL_DETAIL, LEAN_TOOL_DETAIL, TOOL_DETAIL_MODES
from .support import ResearchError

__all__ = [
    "FULL_TOOL_DETAIL",
    "LEAN_TOOL_DETAIL",
    "TOOL_DETAIL_MODES",
    "lean_find_source",
    "lean_ingest",
    "lean_passage",
    "lean_passage_context",
    "lean_search",
    "lean_source_inclusion",
    "lean_status",
    "present_tool_response",
]


def _copy(source: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return an allowlisted copy; a key the source lacks stays absent."""

    return {key: source[key] for key in keys if key in source}


def _meaningful(value: Any) -> bool:
    """Whether a value carries information rather than a default."""

    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)) and not value:
        return False
    return not (isinstance(value, int) and not isinstance(value, bool) and value == 0)


def _add(target: dict[str, Any], key: str, value: Any) -> None:
    """Set a key unless its value is empty, null, false, or zero."""

    if _meaningful(value):
        target[key] = value


def _lean_locator(locator: Mapping[str, Any]) -> dict[str, Any]:
    """Return where a passage sits: its page, or its section.

    The printed page label is carried only when it differs from the physical page.
    The locator's kind is dropped, since the passage is not a citation.
    """

    page = locator.get("page")
    if page is not None:
        result: dict[str, Any] = {"page": page}
        label = locator.get("page_label")
        if label is not None and str(label) != str(page):
            result["page_label"] = label
        return result
    for key in ("section_title", "href", "section_index"):
        value = locator.get(key)
        if value is not None:
            return {"section": value}
    return {}


def lean_passage(passage: Mapping[str, Any]) -> dict[str, Any]:
    """Return one passage: its source, its authors, its position, and its text.

    The text is cleaned for retrieval and so never quote-safe, which the tool
    description states once. The advisory script note belongs to the full payload.
    """

    result: dict[str, Any] = {}
    for key in ("chunk_id", "source_relative_path", "authors"):
        _add(result, key, passage.get(key))
    locator = _lean_locator(passage.get("locator") or {})
    if locator:
        result["locator"] = locator
    result["text"] = passage.get("text")
    return result


def lean_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a search answer: the passages, and anything the caller must know.

    Plus the conditions that change what the caller can conclude: `stale` when the
    corpus has moved on, `reranked: false` when the cross-encoder did not run, an
    upgrade note when this generation cannot serve, a filter that emptied the
    answer, and a source id that resolved to nothing.
    """

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    if payload.get("stale"):
        result["stale"] = True
    if payload.get("reranked") is False:
        result["reranked"] = False
    _add(result, "rerank_fallback", payload.get("rerank_fallback"))
    _add(
        result,
        "generation_upgrade_required",
        payload.get("generation_upgrade_required"),
    )
    filters = payload.get("filters") or {}
    _add(result, "unresolved_source_ids", filters.get("unknown_source_ids"))
    _add(
        result,
        "unresolved_exclude_source_ids",
        filters.get("unknown_exclude_source_ids"),
    )
    # The bibliographic filters travel with the answer, not only in the developer
    # payload: a filter that removed every source is why an answer is empty, and
    # an agent reading the answer has otherwise no way to tell that apart from a
    # corpus that holds nothing.
    _add(
        result,
        "applied_filters",
        {
            name: filters[name]
            for name in ("authors_any", "titles_any")
            if filters.get(name)
        },
    )
    result["hits"] = [lean_passage(hit) for hit in payload.get("hits") or []]
    return result


def lean_passage_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the requested passage's neighbours, and the generation they are in."""

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    result["context"] = [lean_passage(item) for item in payload.get("context") or []]
    return result


# The two calls a status verdict can ask for. Each name is the call it maps to:
# `ingest` is a tool this surface serves, and `restart_app` is the client
# restarting the process running it.
INGEST = "ingest"
RESTART_APP = "restart_app"


def _required_actions(payload: Mapping[str, Any]) -> list[str]:
    """Return the calls that close the gap between this generation and the corpus.

    Every condition that needs one is a rebuild: a missing generation, a corpus
    that moved on, a generation built by an older policy, one that predates the
    dense index this tool searches, reviewed sources it has never indexed, an
    exclusion the indexes still hold, and a build that stopped part-way. The
    answer names them once.
    """

    changes = payload.get("changes") or {}
    actions: list[str] = []
    if (
        payload.get("ready") is not True
        or payload.get("stale")
        or payload.get("generation_upgrade_required")
        or payload.get("hybrid_ready") is False
        or payload.get("ingestion_progress")
        or payload.get("metadata_pending_source_paths")
        or changes.get("source_exclusions_changed")
    ):
        actions.append(INGEST)
    if (payload.get("version") or {}).get("restart_required"):
        actions.append(RESTART_APP)
    return actions


def lean_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return whether this project can be searched, and what must happen first.

    `ready` and `stale` are booleans rather than an absence, `requires` names the
    calls that make this generation serve what the project holds, and `message`
    says why in one sentence. `changes`, `ingestion_progress`, `blocked_by`, and
    `degraded` appear only while they hold.

    The corpus counts, the retained generations, the retrieval policy, and the
    per-check detail are the command line's and the workspace's answer. The
    reviewed-metadata overlay is not news: it is applied at read time, and the
    message says so when reviewed sources are still waiting to be indexed.
    """

    result: dict[str, Any] = {
        "ready": bool(payload.get("ready")),
        "stale": bool(payload.get("stale")),
    }
    _add(result, "generation_id", payload.get("generation_id"))
    required = _required_actions(payload)
    if required:
        result["requires"] = required
    if payload.get("stale"):
        changes = payload.get("changes") or {}
        lean_changes: dict[str, Any] = {}
        for key, source_key in (
            ("added_source_count", "added"),
            ("modified_source_count", "modified"),
        ):
            count = len(changes.get(source_key) or [])
            if count:
                lean_changes[key] = count
        # Available sources are counted, never listed. A source the generation
        # has and the directory does not is named, because that is what a
        # researcher acts on.
        _add(lean_changes, "removed_sources", changes.get("removed"))
        _add(lean_changes, "metadata_changed", changes.get("metadata_changed"))
        if lean_changes:
            result["changes"] = lean_changes
    _add(result, "ingestion_progress", payload.get("ingestion_progress"))
    for key in ("blocked_by", "degraded"):
        entries = payload.get(key) or []
        if entries:
            result[key] = [
                _copy(entry, ("check", "reason", "remedy")) for entry in entries
            ]
    result["message"] = payload.get("message")
    return result


def lean_ingest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return what ingestion did, what is left to do, and anything it dropped.

    A build answers with its outcome and its size: whether the generation changed,
    how many documents and chunks it holds, and the next action when work is
    resumable. The counters that report discarded, withheld, or densely truncated
    material appear only when they are not zero. What the build reused, rebuilt, or
    re-embedded is cost rather than outcome, and belongs to `--tool-detail full`.
    """

    result: dict[str, Any] = {}
    _add(result, "status", payload.get("status"))
    _add(result, "generation_changed", payload.get("generation_changed"))
    for key in ("generation_id", "build_id", "phase", "progress"):
        _add(result, key, payload.get(key))
    for key in ("document_count", "chunk_count"):
        _add(result, key, payload.get(key))
    for key in (
        "discarded_empty_chunk_count",
        "discarded_symbol_only_chunk_count",
        "discarded_corrupt_chunk_count",
        "excluded_corrupt_unit_count",
        "dense_truncated_chunk_count",
        "withheld_chunk_count",
    ):
        _add(result, key, payload.get(key))
    _add(result, "withheld_chunk_reasons", payload.get("withheld_chunk_reasons"))
    _add(result, "superseded_build", payload.get("superseded_build"))
    _add(result, "next_action", payload.get("next_action"))
    result["message"] = payload.get("message")
    return result


def lean_source_match(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one source's handle, its bibliography, and whether it is searchable.

    `indexed_in_current_generation` always travels, because whether a source can
    answer a search is what the lookup was made to settle. `included` and `exists`
    appear only when they withhold it.
    """

    result: dict[str, Any] = {}
    for key in ("source_id", "source_relative_path", "title", "authors"):
        _add(result, key, record.get(key))
    if record.get("exists") is False:
        result["exists"] = False
    if record.get("included") is False:
        result["included"] = False
    result["indexed_in_current_generation"] = bool(
        record.get("indexed_in_current_generation")
    )
    if record.get("has_reviewed_metadata"):
        result["has_reviewed_metadata"] = True
    return result


def lean_find_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sources one name resolved to, and how many were withheld.

    A match count above the returned rows means `limit` hid some.
    """

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    result["query"] = payload.get("query")
    _add(result, "match_count", payload.get("match_count"))
    if payload.get("truncated"):
        result["truncated"] = True
    result["matches"] = [
        lean_source_match(record) for record in payload.get("matches") or []
    ]
    result["message"] = payload.get("message")
    return result


def lean_source_inclusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the inclusion decision, its reason, and whether it applies now."""

    result: dict[str, Any] = {}
    for key in ("status", "source_id", "source_relative_path"):
        _add(result, key, payload.get(key))
    for key in ("included", "reason"):
        if key in payload:
            result[key] = payload[key]
    # An exclusion applies at once, which is the ordinary case and so unsaid. A
    # decision that has to wait for a rebuild is news, because the caller must
    # rebuild before the corpus answers differently.
    if payload.get("effective_immediately") is False:
        result["effective_immediately"] = False
    if payload.get("generation_rebuild_recommended"):
        result["generation_rebuild_recommended"] = True
    result["message"] = payload.get("message")
    return result


def lean_source_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return what was saved for one source and whether it applies now."""

    result: dict[str, Any] = {}
    for key in ("status", "source_id", "source_relative_path"):
        _add(result, key, payload.get(key))
    if "metadata" in payload:
        result["metadata"] = payload["metadata"]
    if payload.get("effective_immediately") is False:
        result["effective_immediately"] = False
    if payload.get("generation_rebuild_recommended"):
        result["generation_rebuild_recommended"] = True
    result["message"] = payload.get("message")
    return result


_PROJECTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "status": lean_status,
    "ingest": lean_ingest,
    "search": lean_search,
    "find_source": lean_find_source,
    "get_passage": lean_passage_context,
    "set_source_inclusion": lean_source_inclusion,
    "set_source_metadata": lean_source_metadata,
}


def present_tool_response(
    operation: str,
    payload: Mapping[str, Any],
    *,
    detail: str,
) -> dict[str, Any]:
    if detail == FULL_TOOL_DETAIL:
        return dict(payload)
    projector = _PROJECTORS.get(operation)
    if projector is None:
        raise ResearchError(f"Unknown research tool: {operation}")
    return projector(payload)
