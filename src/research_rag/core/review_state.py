"""The four decisions a reader records about a source or a passage.

This module owns the writes over the three review files: it loads one entry,
mutates it, writes the file only when the entry actually changed, and composes
the message that says what the change means for the generation on screen. It may
never read a generation's indexes, hold the project lock, or answer a status
question, because a decision is a fact about the review rather than about the
corpus it applies to. A decision about a passage is enforced by the retrieval
filter and never by a rebuild, which is why no message here recommends one.

What moved here from `review.py`: `set_source_inclusion`, `set_chunk_inclusion`,
`list_chunk_exclusions`, `set_source_metadata`, and `_chunk_placements`. The
workflow methods that remain are the ones that hold the lock, resolve the source
selector, and read the review files, and they call the functions here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from ..corpus.sources import SourcePolicyError, normalize_metadata
from ..project.config import ResearchConfig
from ..project.support import (
    ResearchError,
    _document_for_chunk,
    _effective_documents,
    _locator_place,
    _public_document,
    _utc_now,
)
from ..storage.records import (
    StorageError,
    load_metadata_overrides,
    write_chunk_exclusions,
    write_metadata_overrides,
    write_source_exclusions,
)

# Where a named chunk sits in the selected generation, resolved by the workflow
# because the artifact lookup is its own awaitable.
Placements = Callable[
    [Sequence[str], "tuple[Path, dict[str, Any]] | None"],
    Awaitable[dict[str, dict[str, str]]],
]


async def chunk_placements(
    chunk_ids: Sequence[str],
    current: tuple[Path, dict[str, Any]] | None,
    load_chunks: Callable[[Path, dict[str, Any], list[str]], Awaitable[dict[str, Any]]],
    read_metadata: Callable[[], dict[str, dict[str, Any]]],
) -> dict[str, dict[str, str]]:
    """Where each named chunk sits in the selected generation.

    A chunk the generation does not hold is absent from the answer rather
    than refused, because the decision is the reader's and stays on file
    either way, and because a caller has to be able to tell a decision that
    withholds nothing now from one that withholds a passage.
    """

    if current is None or not chunk_ids:
        return {}
    generation_root, manifest = current
    chunks = await load_chunks(generation_root, manifest, sorted(chunk_ids))
    documents_by_id = _effective_documents(manifest, read_metadata())
    return {
        str(chunk_id): {
            "source_relative_path": str(
                _public_document(_document_for_chunk(chunk, documents_by_id)).get(
                    "source_relative_path"
                )
                or ""
            ),
            "locator": _locator_place(dict(chunk.get("locator") or {})),
        }
        for chunk_id, chunk in chunks.items()
    }


def set_source_inclusion(
    config: ResearchConfig,
    selected: dict[str, Any],
    exclusions: dict[str, dict[str, str]],
    *,
    included: bool,
    reason: str | None,
) -> dict[str, Any]:
    """Include or exclude one source without changing the source file."""

    try:
        relative = str(selected["source_relative_path"])
        previous = exclusions.get(relative)
        if included:
            changed = previous is not None
            exclusions.pop(relative, None)
        else:
            normalized_reason = (reason or "").strip()
            if not normalized_reason:
                raise SourcePolicyError(
                    "A non-empty reason is required when excluding a source"
                )
            if (
                not selected["exists"]
                and not selected["indexed_in_current_generation"]
                and previous is None
            ):
                raise SourcePolicyError(
                    "Only an existing or currently indexed PDF or EPUB can "
                    "be excluded: "
                    f"{relative}"
                )
            changed = previous is None or previous.get("reason") != normalized_reason
            if changed:
                exclusions[relative] = {
                    "reason": normalized_reason,
                    "excluded_at": _utc_now(),
                }

        if changed:
            write_source_exclusions(
                config.source_exclusions_path,
                exclusions,
            )

        indexed = bool(selected["indexed_in_current_generation"])
    except (StorageError, SourcePolicyError, ValueError) as exc:
        raise ResearchError(str(exc)) from exc

    effective_immediately = not included or indexed
    if not changed:
        message = (
            "Source is already included."
            if included
            else "Source is already excluded with this reason."
        )
    elif included and not indexed:
        message = (
            "Source inclusion saved. Run ingest before it can appear in "
            "search because it is absent from the current generation."
        )
    elif included:
        message = (
            "Source inclusion saved and restored for current retrieval. "
            "Run ingest to record the change in a new generation."
        )
    else:
        message = (
            "Source exclusion saved and enforced for current retrieval. "
            "The original file was not changed. Run ingest to rebuild the "
            "indexes without it."
        )
    return {
        "status": "changed" if changed else "unchanged",
        "source_id": selected["source_id"],
        "source_relative_path": relative,
        "source_path": selected["source_path"],
        "included": included,
        "reason": None if included else exclusions[relative]["reason"],
        "source_file_changed": False,
        "effective_immediately": effective_immediately,
        "generation_rebuild_recommended": changed,
        "message": message,
    }


async def set_chunk_inclusion(
    config: ResearchConfig,
    current: tuple[Path, dict[str, Any]] | None,
    read_exclusions: Callable[[], dict[str, dict[str, str]]],
    placements: Placements,
    *,
    chunk_id: str,
    included: bool,
    reason: str | None,
) -> dict[str, Any]:
    """Exclude one passage from retrieval, or restore it, without touching the file.

    A decision about a passage is a decision about evidence rather than about
    a document, so it is enforced by the retrieval filter at query time and
    never by a generation: a rebuild keeps the chunk in the indexes and the
    exclusion over it, which is why no call is recommended to close the gap.

    A `chunk_id` is derived from content, so unchanged text keeps it across a
    rebuild and anything else changes it. The generation a decision was
    recorded in travels with it, and a chunk this generation does not hold is
    reported as such rather than silently kept: an ingestion cannot bring a
    removed chunk back, so the answer says which kind of entry this is.
    """

    normalized_id = (chunk_id or "").strip()
    try:
        if current is None:
            raise SourcePolicyError(
                "No knowledge base exists, so no chunk can be named; call ingest first"
            )
        generation_id = str(current[1]["generation_id"])
        # The review file is read only once the knowledge base exists, because a
        # refusal that names it is the answer when there is none.
        exclusions = read_exclusions()
        previous = exclusions.get(normalized_id)
        if included:
            changed = previous is not None
            exclusions.pop(normalized_id, None)
        else:
            normalized_reason = (reason or "").strip()
            if not normalized_reason:
                raise SourcePolicyError(
                    "A non-empty reason is required when excluding a chunk"
                )
            changed = (
                previous is None
                or previous.get("reason") != normalized_reason
                or previous.get("generation_id") != generation_id
            )
            if changed:
                exclusions[normalized_id] = {
                    "reason": normalized_reason,
                    "excluded_at": _utc_now(),
                    "generation_id": generation_id,
                }
        if changed:
            write_chunk_exclusions(
                config.chunk_exclusions_path,
                exclusions,
            )
        saved_reason = None if included else exclusions[normalized_id]["reason"]
        placement = (await placements([normalized_id], current)).get(normalized_id)
    except (StorageError, SourcePolicyError, ValueError) as exc:
        raise ResearchError(str(exc)) from exc

    in_current_generation = placement is not None
    if not changed:
        message = (
            "Chunk is already included."
            if included
            else "Chunk is already excluded with this reason."
        )
    elif included:
        message = (
            "Chunk inclusion saved. The passage answers a search again at "
            "once, and no ingestion is needed for that."
        )
    elif not in_current_generation:
        message = (
            "Chunk exclusion saved, but this generation does not hold the "
            "chunk, so nothing is withheld from the passages you can reach "
            "now. The decision binds any later generation that holds it, and "
            "no ingestion can bring the chunk back."
        )
    else:
        message = (
            "Chunk exclusion saved and enforced for current retrieval. "
            "The original file was not changed, and no ingestion is needed "
            "to keep enforcing it."
        )
    return {
        "status": "changed" if changed else "unchanged",
        "chunk_id": normalized_id,
        "source_relative_path": (
            placement["source_relative_path"] if placement else None
        ),
        "locator": placement["locator"] if placement else None,
        "included": included,
        "reason": saved_reason,
        "source_file_changed": False,
        # An exclusion is applied by the filter rather than by the
        # indexes, so it holds for the generation on screen and for every
        # one built after it without a rebuild.
        "effective_immediately": not included or in_current_generation,
        "generation_rebuild_recommended": False,
        "in_current_generation": in_current_generation,
        "message": message,
    }


def list_chunk_exclusions(
    exclusions: dict[str, dict[str, str]],
    placements: dict[str, dict[str, str]],
    generation_id: str | None,
) -> dict[str, Any]:
    """The passages this project has taken out of retrieval, and why.

    It reviews decisions rather than the corpus: one row per recorded
    decision, each naming where its passage sat when the decision was made.
    A row the selected generation does not hold is reported as such, because
    that is the row a reader has to remove by hand and a rebuild will not
    remove for them.
    """

    records = [
        {
            "chunk_id": chunk_id,
            "source_relative_path": placements.get(chunk_id, {}).get(
                "source_relative_path", ""
            ),
            "locator": placements.get(chunk_id, {}).get("locator", ""),
            "reason": record["reason"],
            "excluded_at": record["excluded_at"],
            "in_current_generation": chunk_id in placements,
        }
        for chunk_id, record in exclusions.items()
    ]
    withheld = sum(1 for entry in records if entry["in_current_generation"])
    if not records:
        message = "No chunk is excluded from retrieval."
    elif withheld == 0:
        message = (
            f"None of the {len(records)} excluded chunks is withheld from "
            "current retrieval: each names a chunk this generation does not "
            "hold, and no ingestion restores a removed chunk."
        )
    elif withheld == len(records):
        message = (
            f"Every one of the {withheld} excluded chunks is withheld from "
            "current retrieval, and no original file was changed."
        )
    else:
        message = (
            f"{withheld} of the {len(records)} excluded chunks are withheld "
            f"from current retrieval; the other {len(records) - withheld} name "
            "a chunk this generation does not hold, and no ingestion restores "
            "a removed chunk."
        )
    return {
        "generation_id": generation_id,
        "excluded_chunk_count": len(records),
        "exclusions": records,
        "message": message,
    }


def set_source_metadata(
    config: ResearchConfig,
    selected: dict[str, Any],
    reviewed: dict[str, Any],
) -> dict[str, Any]:
    """Save reviewed bibliographic metadata for one source.

    The review is authoritative at read time, so a saved change applies to the
    current generation without re-ingesting, and the same JSON file can be
    edited by hand between calls. Only the named source's entry is replaced,
    so every other entry survives. An empty review removes the entry, so
    automatic metadata applies again.
    """

    try:
        relative = str(selected["source_relative_path"])
        normalized = {
            key: value
            for key, value in normalize_metadata(dict(reviewed)).items()
            if value not in ("", [], None)
        }
        overrides = load_metadata_overrides(config.metadata_path)
        previous = overrides.get(relative)
        if normalized:
            overrides[relative] = normalized
        else:
            overrides.pop(relative, None)
        changed = previous != (normalized or None)
        if changed:
            write_metadata_overrides(config.metadata_path, overrides)
        indexed = bool(selected["indexed_in_current_generation"])
    except (StorageError, SourcePolicyError, ValueError) as exc:
        raise ResearchError(str(exc)) from exc

    if not changed:
        message = "Reviewed metadata already matches what was saved."
    elif not normalized:
        message = (
            "Reviewed metadata cleared for this source; automatic "
            "metadata applies again."
        )
    elif indexed:
        message = (
            "Reviewed metadata saved and applied to current retrieval. "
            "The next ingestion records it in a new generation."
        )
    else:
        message = (
            "Reviewed metadata saved. Run ingest before it can appear in "
            "search because the source is absent from the current "
            "generation."
        )
    return {
        "status": "changed" if changed else "unchanged",
        "source_id": selected["source_id"],
        "source_relative_path": relative,
        "source_path": selected["source_path"],
        "metadata": normalized,
        "source_file_changed": False,
        "effective_immediately": bool(normalized) and indexed,
        "generation_rebuild_recommended": False,
        "message": message,
    }
