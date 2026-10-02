"""The portable records a project carries, and the pointer to its generation.

Five hand-editable files live under `<project>/.research-rag`: the reviewed
metadata overlay, the source catalog, the source exclusions, and the passage
exclusions, each read and written here, plus `current.json`, the pointer to the
generation in use. Every reader refuses a file written for a later version of this
package and names the field it could not read, and every writer replaces its file
atomically through `durable_io`.

This module holds schemas and validation, not I/O. The durable-write primitives
live in `durable_io`, and the rule every stored path must satisfy lives in
`normalized_paths`. The primitives are re-exported here for now: a module that
imports `storage` for one of them works, and that re-export is transitional rather
than a second home.

Nothing here writes outside the path it is given, and nothing here reads a setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .durable_io import (
    StorageError,
    atomic_write_json,
    atomic_write_jsonl,
    directory_statistics,
    fsync_directories,
    fsync_directory,
    iter_jsonl,
    read_json,
    read_jsonl,
    write_handoff_jsonl,
    write_jsonl,
)
from .normalized_paths import normalized_relative_path
from .sources import ALLOWED_SOURCE_EXTENSIONS, METADATA_FIELDS, stable_source_id

__all__ = [
    "StorageError",
    "atomic_write_json",
    "atomic_write_jsonl",
    "directory_statistics",
    "fsync_directories",
    "fsync_directory",
    "iter_jsonl",
    "load_chunk_exclusions",
    "load_current_generation",
    "load_metadata_overrides",
    "load_source_catalog",
    "load_source_exclusions",
    "read_json",
    "read_jsonl",
    "write_chunk_exclusions",
    "write_handoff_jsonl",
    "write_jsonl",
    "write_metadata_overrides",
    "write_source_catalog",
    "write_source_exclusions",
]


def load_metadata_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StorageError(f"Unsupported metadata file: {path}")
    _reject_unknown_metadata_version(path, value)
    sources = value.get("sources", {})
    if not isinstance(sources, dict) or any(
        not isinstance(key, str) or not isinstance(item, dict)
        for key, item in sources.items()
    ):
        raise StorageError(f"Invalid source metadata mapping: {path}")
    for source_path in sources:
        if normalized_relative_path(source_path) is None:
            raise StorageError(
                "Metadata source paths must be normalized and relative: "
                f"{source_path!r}"
            )
    return sources


def _reject_unknown_metadata_version(path: Path, value: dict[str, Any]) -> None:
    """Refuse metadata written for a later version of this package.

    `schema_version` alone cannot carry that: a field added to the shape left the
    number at 1, so a server predating the field rejected the whole file with
    `Unsupported metadata fields: language` and named a field rather than the
    cause.

    A writer records the fields it understood, so a file without the key is
    accepted: its contents are validated field by field anyway, and refusing it
    would break every existing project.
    """

    written_with = value.get("written_with")
    if written_with is None:
        return
    if not isinstance(written_with, dict):
        raise StorageError(f"Invalid metadata writer record: {path}")
    fields = written_with.get("metadata_fields")
    if fields is None:
        return
    if not isinstance(fields, list) or any(
        not isinstance(item, str) for item in fields
    ):
        raise StorageError(f"Invalid metadata writer field list: {path}")
    unknown = sorted(set(fields) - set(METADATA_FIELDS))
    if unknown:
        raise StorageError(
            f"{path} was written by a version that understood metadata fields "
            f"this one does not: {', '.join(unknown)}. Update the server that "
            "reads this project before trusting its answers."
        )


def write_metadata_overrides(
    path: Path,
    overrides: dict[str, dict[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "written_with": {
                "metadata_fields": sorted(METADATA_FIELDS),
            },
            "sources": overrides,
        },
    )


def load_source_catalog(path: Path, *, project_id: str) -> dict[str, str]:
    if not path.exists():
        return {}
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("project_id") != project_id
    ):
        raise StorageError(f"Unsupported source catalog: {path}")
    sources = value.get("sources", {})
    if not isinstance(sources, dict):
        raise StorageError(f"Invalid source catalog mapping: {path}")

    # Import lazily to keep these JSON helpers independent of `sources`, while
    # still validating project-derived identity at this trust boundary.

    result: dict[str, str] = {}
    seen_paths: set[str] = set()
    for source_id, record in sources.items():
        if not isinstance(source_id, str) or not isinstance(record, dict):
            raise StorageError(f"Invalid source catalog record: {source_id!r}")
        if set(record) != {"source_relative_path"}:
            raise StorageError(f"Unsupported source catalog fields: {source_id!r}")
        source_path = record.get("source_relative_path")
        if not isinstance(source_path, str):
            raise StorageError(f"Invalid source path for {source_id!r}: {path}")
        relative = normalized_relative_path(source_path)
        if (
            relative is None
            or source_path in seen_paths
            or relative.suffix.casefold() not in ALLOWED_SOURCE_EXTENSIONS
        ):
            raise StorageError(
                "Source catalog paths must be unique, supported, normalized, and "
                f"relative: {source_path!r}"
            )
        try:
            expected_source_id = stable_source_id(project_id, source_path)
        except ValueError as exc:
            raise StorageError(
                f"Invalid source identity in catalog: {source_id!r}"
            ) from exc
        if source_id != expected_source_id:
            raise StorageError(
                f"Source catalog ID does not match its path: {source_id!r}"
            )
        result[source_id] = source_path
        seen_paths.add(source_path)
    return dict(sorted(result.items()))


def write_source_catalog(
    path: Path,
    catalog: dict[str, str],
    *,
    project_id: str,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "project_id": project_id,
            "sources": {
                source_id: {"source_relative_path": source_path}
                for source_id, source_path in sorted(catalog.items())
            },
        },
    )


def load_source_exclusions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StorageError(f"Unsupported source-exclusion file: {path}")
    sources = value.get("sources", {})
    if not isinstance(sources, dict):
        raise StorageError(f"Invalid source-exclusion mapping: {path}")

    normalized: dict[str, dict[str, str]] = {}
    for source_path, record in sources.items():
        if not isinstance(source_path, str) or not source_path.strip():
            raise StorageError(f"Invalid excluded source path: {path}")
        if normalized_relative_path(source_path) is None:
            raise StorageError(
                f"Excluded source path must be normalized and relative: {source_path!r}"
            )
        if not isinstance(record, dict):
            raise StorageError(f"Invalid exclusion record for {source_path!r}: {path}")
        unknown = set(record) - {"reason", "excluded_at"}
        if unknown:
            raise StorageError(
                f"Unsupported exclusion fields for {source_path!r}: "
                f"{', '.join(sorted(unknown))}"
            )
        reason = record.get("reason")
        excluded_at = record.get("excluded_at")
        if not isinstance(reason, str) or not reason.strip():
            raise StorageError(
                f"Exclusion for {source_path!r} requires a non-empty reason: {path}"
            )
        if not isinstance(excluded_at, str) or not excluded_at.strip():
            raise StorageError(
                f"Exclusion for {source_path!r} requires excluded_at: {path}"
            )
        normalized[source_path] = {
            "reason": reason.strip(),
            "excluded_at": excluded_at.strip(),
        }
    return dict(sorted(normalized.items()))


def write_source_exclusions(
    path: Path,
    exclusions: dict[str, dict[str, str]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "sources": dict(sorted(exclusions.items())),
        },
    )


def load_chunk_exclusions(path: Path) -> dict[str, dict[str, str]]:
    """Load the reviewed decisions about single passages.

    An entry names one `chunk_id`, so the file is keyed by an identifier the
    generation owns rather than by a path this project can normalize. The
    recorded `generation_id` makes an entry this generation cannot match
    identifiable by hand.
    """

    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StorageError(f"Unsupported chunk-exclusion file: {path}")
    chunks = value.get("chunks", {})
    if not isinstance(chunks, dict):
        raise StorageError(f"Invalid chunk-exclusion mapping: {path}")

    normalized: dict[str, dict[str, str]] = {}
    for chunk_id, record in chunks.items():
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise StorageError(f"Invalid excluded chunk id: {path}")
        if not isinstance(record, dict):
            raise StorageError(
                f"Invalid chunk exclusion record for {chunk_id!r}: {path}"
            )
        unknown = set(record) - {"reason", "excluded_at", "generation_id"}
        if unknown:
            raise StorageError(
                f"Unsupported chunk exclusion fields for {chunk_id!r}: "
                f"{', '.join(sorted(unknown))}"
            )
        reason = record.get("reason")
        excluded_at = record.get("excluded_at")
        generation_id = record.get("generation_id")
        if not isinstance(reason, str) or not reason.strip():
            raise StorageError(
                f"Chunk exclusion for {chunk_id!r} requires a non-empty reason: {path}"
            )
        if not isinstance(excluded_at, str) or not excluded_at.strip():
            raise StorageError(
                f"Chunk exclusion for {chunk_id!r} requires excluded_at: {path}"
            )
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise StorageError(
                f"Chunk exclusion for {chunk_id!r} requires generation_id: {path}"
            )
        normalized[chunk_id] = {
            "reason": reason.strip(),
            "excluded_at": excluded_at.strip(),
            "generation_id": generation_id.strip(),
        }
    return dict(sorted(normalized.items()))


def write_chunk_exclusions(
    path: Path,
    exclusions: dict[str, dict[str, str]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "chunks": dict(sorted(exclusions.items())),
        },
    )


def load_current_generation(state_root: Path) -> tuple[Path, dict[str, Any]]:
    pointer = read_json(state_root / "current.json")
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise StorageError("Invalid current-generation pointer")
    generation_id = pointer.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise StorageError("Current-generation pointer has no generation_id")
    generation_root = (state_root / "generations" / generation_id).resolve()
    generations_root = (state_root / "generations").resolve()
    try:
        generation_root.relative_to(generations_root)
    except ValueError as exc:
        raise StorageError(
            "Current-generation pointer escapes generation storage"
        ) from exc
    manifest = read_json(generation_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("generation_id") != generation_id:
        raise StorageError("Current generation manifest does not match its pointer")
    return generation_root, manifest
