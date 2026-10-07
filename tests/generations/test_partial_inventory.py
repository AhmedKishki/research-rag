"""Partial disclosure reads manifest metrics without mutating retained artifacts."""

import json
from types import SimpleNamespace

import pytest

from research_rag.generations.generation_inventory import generation_inventory


@pytest.mark.parametrize("skipped", [[], [{"source_path": "broken.pdf"}]])
def test_inventory_reports_partial_from_manifest_metrics(tmp_path, skipped):
    root = tmp_path / "generations"
    generation = root / "generation"
    generation.mkdir(parents=True)
    manifest = generation / "manifest.json"
    manifest.write_text(json.dumps({"build_metrics": {"skipped_sources": skipped}}))
    original = manifest.read_bytes()
    inventory = generation_inventory(SimpleNamespace(generations_root=root), None)
    record = inventory["generations"][0]
    assert record["partial"] is bool(skipped)
    assert record["skipped_source_count"] == len(skipped)
    assert record["is_current"] is False
    assert manifest.read_bytes() == original
