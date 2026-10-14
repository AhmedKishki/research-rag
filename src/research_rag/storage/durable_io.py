"""Durable writes, and the reads that go with them.

A write here is atomic and persisted: `atomic_file` writes a temporary name in
the target directory, fsyncs it, renames it over the target, and fsyncs the
parent directory so the rename itself survives a power cut. No other module
repeats it. A caller that writes several files as one commit passes
``fsync_parent=False`` and calls :func:`fsync_directories` once at the end,
because cost follows the number of durability operations rather than the size
of a payload.

This module knows nothing about the app: no record schema, no project layout, no
settings. The record schemas are in `records.py`, and the rule every stored path
must satisfy is in `normalized_paths.py`.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


async def _atomic_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Let a write-side thread finish its atomic unit before propagating cancel."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


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


def file_identity(path: Path) -> dict[str, int]:
    """The stat fields that change whenever a file's bytes may have."""

    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def fsync_file(path: Path) -> None:
    """Persist the contents of a file another writer has already closed."""

    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def replace_durably(temporary: Path, path: Path, *, fsync_parent: bool = True) -> None:
    """Rename a persisted file over `path`, and persist the rename itself."""

    os.replace(temporary, path)
    if fsync_parent:
        fsync_directory(path.parent)


@contextmanager
def atomic_file(
    path: Path,
    mode: str = "w",
    *,
    fsync_parent: bool = True,
    private: bool = False,
) -> Iterator[IO[Any]]:
    """Write a file beside `path`, then persist it and rename it over `path`.

    A reader sees the old file or the new one, never half of either. Text is UTF-8
    with `\n` line endings. A `private` file is created readable by its owner
    alone. A write that raises leaves `path` untouched and no temporary behind.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    text = {} if "b" in mode else {"encoding": "utf-8", "newline": "\n"}
    opener = (lambda name, flags: os.open(name, flags, 0o600)) if private else None
    try:
        with open(temporary, mode, opener=opener, **text) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        replace_durably(temporary, path, fsync_parent=fsync_parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, fsync_parent: bool = True) -> None:
    """Write JSON atomically.

    Pass ``fsync_parent=False`` to defer the directory fsync when several files
    are committed together. A caller that defers must persist the parents with
    :func:`fsync_directories` before committing state that depends on them: cost
    follows the number of durability operations rather than the size of a
    payload, so a group of related writes costs far less than one per file.
    """

    with atomic_file(path, fsync_parent=fsync_parent) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")


def _write_records(
    handle: IO[str], records: Iterable[dict[str, Any]], sort_keys: bool
) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=sort_keys) + "\n")


def write_jsonl(
    path: Path, records: Iterable[dict[str, Any]], *, sort_keys: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        _write_records(handle, records, sort_keys)
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
        _write_records(handle, records, sort_keys=True)


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    fsync_parent: bool = True,
) -> None:
    with atomic_file(path, fsync_parent=fsync_parent) as handle:
        _write_records(handle, records, sort_keys=True)


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
