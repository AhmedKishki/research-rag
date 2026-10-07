"""Bounded replies with follow-up IDs, evidence, and actionable conditions.

Keep partial-result disclosures; omit ordinary confirmations,
query echoes, scores, timings, inventories, and inferable counts. The workspace
and explicit full-detail mode retain complete service payloads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..project.policy import ResearchError
from ..project.settings import FULL_TOOL_DETAIL, LEAN_TOOL_DETAIL, TOOL_DETAIL_MODES

__all__ = [
    "FULL_TOOL_DETAIL",
    "LEAN_TOOL_DETAIL",
    "TOOL_DETAIL_MODES",
    "lean_chunk_inclusion",
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
    if _meaningful(value):
        target[key] = value


def _add_message(
    target: dict[str, Any], payload: Mapping[str, Any], *, explaining: bool
) -> None:
    """Keep a reason when needed, not a sentence restating structured fields."""

    if explaining and payload.get("message"):
        target["message"] = payload["message"]


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
    """Keep follow-up handles, attribution, locator, complete text, and safeguards.

    Truncation and context-exclusion markers qualify the evidence; diagnostics
    remain in the full payload.
    """

    result: dict[str, Any] = {}
    for key in ("chunk_id", "source_id", "title", "authors"):
        _add(result, key, passage.get(key))
    locator = _lean_locator(passage.get("locator") or {})
    if locator:
        result["locator"] = locator
    result["text"] = passage.get("text")
    if passage.get("dense_truncated"):
        result["dense_truncated"] = True
    if passage.get("excluded_from_search"):
        result["excluded_from_search"] = True
    return result


# The filters a caller passes, named as the tool names them, and the field each
# one lands in. An agent reports the answer and names the filter that produced
# it, so the name it reads has to be the name it would pass to the next search.
_SEARCH_FILTERS = (
    ("categories_any", "categories_any"),
    ("projects_any", "projects_any"),
    ("keywords", "keywords_all"),
    ("languages_any", "languages_any"),
    ("authors_any", "authors_any"),
    ("titles_any", "titles_any"),
    ("source_ids", "source_ids"),
    ("exclude_source_ids", "exclude_source_ids"),
)


def lean_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep passages and conditions that qualify the search or guide another query.

    Filtered or partial silence must remain distinct from an unqualified corpus miss.
    """

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    if payload.get("stale"):
        result["stale"] = True
    if payload.get("reranked") is False:
        result["reranked"] = False
    _add(result, "rerank_fallback", payload.get("rerank_fallback"))
    _add(result, "retrieval_fallback", payload.get("retrieval_fallback"))
    _add(
        result,
        "generation_upgrade_required",
        payload.get("generation_upgrade_required"),
    )
    if payload.get("relevance_limited"):
        result["relevance_limited"] = True
    _add(
        result,
        "repetitions_collapsed",
        (payload.get("collapsed_repetitions") or {}).get("repetitions_collapsed"),
    )
    filters = payload.get("filters") or {}
    _add(result, "unresolved_source_ids", filters.get("unknown_source_ids"))
    _add(
        result,
        "unresolved_exclude_source_ids",
        filters.get("unknown_exclude_source_ids"),
    )
    # A window that stopped short of the whole corpus is a partial search, and an
    # empty answer over one of those is not evidence the corpus holds nothing.
    if filters.get("window_is_whole_corpus") is False:
        result["search_window_partial"] = True
    # The filters travel with the answer, not only in the developer payload: a
    # filter that emptied the answer is why it is empty, and an agent reading the
    # answer has otherwise no way to tell that apart from a corpus that holds
    # nothing. Each is named as the caller would pass it again.
    _add(
        result,
        "applied_filters",
        {name: filters[field] for name, field in _SEARCH_FILTERS if filters.get(field)},
    )
    result["hits"] = [lean_passage(hit) for hit in payload.get("hits") or []]
    return result


def lean_passage_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the passage asked for with its immediate neighbours, one shape each.

    Excluded neighbours remain context and carry `excluded_from_search: true`;
    do not present them as eligible search results.
    """

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    result["context"] = [lean_passage(item) for item in payload.get("context") or []]
    return result


# The two calls a status verdict can ask for. Each name is the call it maps to:
# `ingest` is a tool this surface serves, and `restart_app` is the client
# restarting the process running it.
INGEST = "ingest"
RESTART_APP = "restart_app"

# The conditions a surface reports about itself rather than about a corpus. Both
# are written by `blocked_answers`, both name a command a reader runs in a
# terminal, and neither is closed by a tool: a server blocked this way declares
# `status` and nothing else, so naming `ingest` would ask for a call that is not
# on this surface. A health check that blocks is not in this set, because the
# corpus is still there and `ingest` is still the call that serves it.
SURFACE_CONDITIONS = frozenset({"project.initialised", "app.serving"})


def _required_actions(payload: Mapping[str, Any]) -> list[str]:
    """Name required generation work or an operator restart.

    Uninitialized or stopped projects instead carry terminal remedies in blocked_by.
    """

    if payload.get("project_initialised") is False:
        return []
    if any(
        str(entry.get("check")) in SURFACE_CONDITIONS
        for entry in payload.get("blocked_by") or ()
        if isinstance(entry, Mapping)
    ):
        return []
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
    says why in one sentence while there is a why. `changes`,
    `ingestion_progress`, `blocked_by`, and `degraded` appear only while they
    hold. `project_initialised` appears only when it is false, which is the one
    answer with no project behind it. A ready, current project with nothing
    required gets the two booleans and its generation, because its own message
    then restates them.

    The corpus counts, the retained generations, the retrieval policy, and the
    per-check detail are the command line's and the workspace's answer. The
    reviewed-metadata overlay is not news: it is applied at read time, and the
    message says so when reviewed sources are still waiting to be indexed.
    """

    ready = bool(payload.get("ready"))
    result: dict[str, Any] = {
        "ready": ready,
        "stale": bool(payload.get("stale")),
    }
    # Stated only when it is false: true everywhere else carries no information,
    # and a reader that has to branch on it needs the one answer where it fails.
    if payload.get("project_initialised") is False:
        result["project_initialised"] = False
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
    _add_message(
        result,
        payload,
        explaining=(
            not ready
            or bool(payload.get("stale"))
            or bool(required)
            or bool(payload.get("blocked_by"))
            or bool(payload.get("degraded"))
        ),
    )
    return result


# The counters that say a build dropped, withheld, or could not fully read
# material. Each is absent from a lean answer while it is zero, so any of them
# appearing is what makes the build's own sentence worth carrying.
_INGEST_DISCLOSURES = (
    "discarded_empty_chunk_count",
    "discarded_symbol_only_chunk_count",
    "discarded_corrupt_chunk_count",
    "excluded_corrupt_unit_count",
    "dense_truncated_chunk_count",
    "withheld_chunk_count",
)


def lean_ingest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return what ingestion did, what is left to do, and anything it dropped.

    A build answers with its outcome and its size: whether the generation changed,
    how many documents and chunks it holds, and the next action when work is
    resumable. The counters that report discarded, withheld, or densely truncated
    material appear only when they are not zero, and the sentence that explains
    them travels with them. `activation_recovered` says a completed build was
    selected after the process restarted, which is a build no call of the
    caller's made. What the build reused, rebuilt, or re-embedded is cost rather
    than outcome, and belongs to `--tool-detail full`.
    """

    result: dict[str, Any] = {}
    _add(result, "status", payload.get("status"))
    _add(result, "generation_changed", payload.get("generation_changed"))
    for key in ("generation_id", "build_id", "phase", "progress"):
        _add(result, key, payload.get(key))
    for key in ("document_count", "chunk_count"):
        _add(result, key, payload.get(key))
    for key in _INGEST_DISCLOSURES:
        _add(result, key, payload.get(key))
    _add(result, "withheld_chunk_reasons", payload.get("withheld_chunk_reasons"))
    _add(result, "superseded_build", payload.get("superseded_build"))
    if payload.get("activation_recovered"):
        result["activation_recovered"] = True
    _add(result, "next_action", payload.get("next_action"))
    _add_message(
        result,
        payload,
        explaining=any(payload.get(key) for key in _INGEST_DISCLOSURES)
        or bool(payload.get("superseded_build"))
        or bool(payload.get("activation_recovered")),
    )
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

    The query is the caller's own and does not travel back. A match count above
    the returned rows means `limit` hid some. The sentence travels only while the
    lookup found nothing to search, because then it is the reason rather than a
    restatement of the rows.
    """

    result: dict[str, Any] = {}
    _add(result, "generation_id", payload.get("generation_id"))
    _add(result, "match_count", payload.get("match_count"))
    if payload.get("truncated"):
        result["truncated"] = True
    matches = [lean_source_match(record) for record in payload.get("matches") or []]
    result["matches"] = matches
    searchable = payload.get("searchable_match_count")
    _add_message(
        result,
        payload,
        explaining=not matches or (searchable is not None and not searchable),
    )
    return result


def _lean_inclusion_decision(
    payload: Mapping[str, Any],
    *,
    identity: tuple[str, ...],
) -> dict[str, Any]:
    """Return an inclusion decision: the subject, the flag, the reason, and the gap.

    One shape serves a source and a chunk, because the decision is the same: a
    reader either has something back in retrieval or has taken it out, and the
    only difference is the identifier that names it. The sentence travels with a
    decision that takes effect later, where it says what the caller must do
    before the corpus answers differently.
    """

    result: dict[str, Any] = {}
    for key in ("status", *identity):
        _add(result, key, payload.get(key))
    for key in ("included", "reason"):
        if key in payload:
            result[key] = payload[key]
    # An exclusion applies at once, which is the ordinary case and so unsaid. A
    # decision that has to wait for a rebuild is news, because the caller must
    # rebuild before the corpus answers differently.
    deferred = payload.get("effective_immediately") is False
    if deferred:
        result["effective_immediately"] = False
    if payload.get("generation_rebuild_recommended"):
        result["generation_rebuild_recommended"] = True
    _add_message(result, payload, explaining=deferred)
    return result


def lean_source_inclusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _lean_inclusion_decision(
        payload, identity=("source_id", "source_relative_path")
    )


def lean_chunk_inclusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the passage decision, its reason, and whether it withholds anything now.

    A decision about a chunk the current generation does not hold says so, because
    a caller that cannot tell the two apart would report a decision as withholding
    a passage it never removed from anything.
    """

    result = _lean_inclusion_decision(payload, identity=("chunk_id",))
    if payload.get("in_current_generation") is False:
        result["in_current_generation"] = False
    return result


def lean_source_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the saved review, and whether it is in force yet.

    `metadata` is the whole review this source now carries, so a cleared entry
    travels as an empty mapping rather than as an absence, and the sentence that
    explains a review that is not in force yet travels with it.
    """

    result: dict[str, Any] = {}
    for key in ("status", "source_id", "source_relative_path"):
        _add(result, key, payload.get(key))
    if "metadata" in payload:
        result["metadata"] = payload["metadata"]
    deferred = payload.get("effective_immediately") is False
    if deferred:
        result["effective_immediately"] = False
    if payload.get("generation_rebuild_recommended"):
        result["generation_rebuild_recommended"] = True
    _add_message(result, payload, explaining=deferred)
    return result


_PROJECTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "status": lean_status,
    "ingest": lean_ingest,
    "search": lean_search,
    "find_source": lean_find_source,
    "get_passage": lean_passage_context,
    "set_source_inclusion": lean_source_inclusion,
    "set_chunk_inclusion": lean_chunk_inclusion,
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
