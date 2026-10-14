"""The source catalog survives a round trip and refuses what it cannot read."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_rag.core.records import load_source_catalog, write_source_catalog
from research_rag.storage.durable_io import StorageError


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
