from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import research_rag.storage.durable_io as durable_io_module
from research_rag.storage.durable_io import atomic_write_json, atomic_write_jsonl
from research_rag.storage.records import (
    StorageError,
    iter_jsonl,
    load_source_catalog,
    read_json,
    read_jsonl,
    write_source_catalog,
)


def test_iter_jsonl_validates_lazily_without_materializing_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\nnot-json\n', encoding="utf-8")

    records = iter_jsonl(path)
    assert next(records) == {"id": 1}
    assert next(records) == {"id": 2}
    with pytest.raises(StorageError, match="line 3"):
        next(records)


def test_read_jsonl_remains_the_materializing_compatibility_helper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_read_json_reports_non_utf8_state_as_invalid(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(StorageError, match="Invalid JSON state file"):
        read_json(path)


def test_iter_jsonl_reports_non_utf8_state_as_invalid(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id": 1}\n\xff\n')

    with pytest.raises(StorageError, match="Invalid UTF-8 in JSONL state file"):
        list(iter_jsonl(path))


def test_source_catalog_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "source-catalog.json"
    project_id = "11111111-2222-3333-4444-555555555555"
    from research_rag.corpus.sources import stable_source_id

    catalog = {
        stable_source_id(project_id, "nested/article.pdf"): "nested/article.pdf",
        stable_source_id(project_id, "book.epub"): "book.epub",
    }

    write_source_catalog(path, catalog, project_id=project_id)

    assert load_source_catalog(path, project_id=project_id) == dict(
        sorted(catalog.items())
    )

    path.write_text(
        '{"schema_version": 1, "project_id": '
        f'"{project_id}", "sources": {{'
        f'"{next(iter(catalog))}": '
        '{"source_relative_path": "../escape.pdf"}}}',
        encoding="utf-8",
    )
    with pytest.raises(StorageError, match="normalized"):
        load_source_catalog(path, project_id=project_id)


@pytest.mark.parametrize("writer", [atomic_write_json, atomic_write_jsonl])
def test_atomic_writers_fsync_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: object,
) -> None:
    calls: list[str] = []
    real_fsync = os.fsync

    def tracking_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        calls.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(durable_io_module.os, "fsync", tracking_fsync)
    path = tmp_path / "state" / "record.json"
    if writer is atomic_write_json:
        atomic_write_json(path, {"id": 1})
    else:
        atomic_write_jsonl(path, [{"id": 1}])

    assert path.is_file()
    assert calls.count("file") == 1
    assert calls.count("directory") == 1
    assert not list(path.parent.glob("*.tmp"))
