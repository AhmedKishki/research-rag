from __future__ import annotations

import errno
import json
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any


class StorageError(RuntimeError):
    pass


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        # These errnos mean the platform cannot open a directory as a file
        # descriptor. Anything else is a genuine storage failure such as EIO or a
        # missing parent.
        if exc.errno not in {
            errno.EACCES,
            errno.EINVAL,
            errno.EISDIR,
            errno.ENOTSUP,
            errno.EPERM,
        }:
            raise
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {
                errno.EACCES,
                errno.EBADF,
                errno.EINVAL,
                errno.ENOTSUP,
            }:
                raise
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any, *, fsync_parent: bool = True) -> None:
    """Write JSON atomically.

    Pass ``fsync_parent=False`` to defer the directory fsync when several files
    are committed together. A caller that defers must persist the parents with
    :func:`fsync_directories` before committing state that depends on them: cost
    follows the number of durability operations rather than the size of a
    payload, so a group of related writes costs far less than one per file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if fsync_parent:
            fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_handoff_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write a file that only a peer process reads and no resume path trusts.

    It must be *visible* to another process on this machine, which closing the
    file guarantees. It is rewritten before every use and deleted afterwards,
    so durability would only slow the caller down.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    fsync_parent: bool = True,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_jsonl(temporary, records)
        os.replace(temporary, path)
        if fsync_parent:
            fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directories(paths: Iterable[Path]) -> None:
    """Persist directory entries for writes that deferred their own fsync."""

    for path in dict.fromkeys(paths):
        if path.is_dir():
            fsync_directory(path)


def directory_statistics(path: Path) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for the regular files under a directory.

    Symlinks and unreadable entries are skipped rather than failing the caller.
    This describes retained state; it does not validate it. A directory walk
    is cheap enough to report alongside every retained generation.
    """

    files = 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            total += entry.stat().st_size
        except OSError:
            continue
        files += 1
    return files, total


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageError(f"Required state file does not exist: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"Invalid JSON state file: {path}") from exc


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects without materializing the complete file."""

    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StorageError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise StorageError(
                        f"Expected an object in {path} at line {line_number}"
                    )
                yield value
    except FileNotFoundError as exc:
        raise StorageError(f"Required state file does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise StorageError(f"Invalid UTF-8 in JSONL state file: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


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
        relative = PurePosixPath(source_path)
        if (
            source_path in {"", "."}
            or "\\" in source_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != source_path
        ):
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
    # Import here because `sources` imports this module.
    from .sources import METADATA_FIELDS

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
    from .sources import METADATA_FIELDS

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
    from .sources import ALLOWED_SOURCE_EXTENSIONS, stable_source_id

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
        relative = PurePosixPath(source_path)
        if (
            source_path in {"", "."}
            or "\\" in source_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != source_path
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
        relative = PurePosixPath(source_path)
        if (
            source_path == "."
            or "\\" in source_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != source_path
        ):
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
