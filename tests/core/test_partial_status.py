"""Partial coverage remains actionable even when source bytes are unchanged."""

import asyncio
import sys

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config


def test_selected_partial_is_stale_and_requests_retry(project, monkeypatch):
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(config, ultrarag=None)
    monkeypatch.setattr(service, "_generation_upgrade_reasons", lambda manifest: [])
    manifest = {
        "generation_id": "partial",
        "created_at": "2026-10-07T00:00:00Z",
        "documents": [],
        "source_files": [],
        "document_count": 0,
        "chunk_count": 0,
        "build_metrics": {"skipped_sources": [{"source_path": "broken.pdf"}]},
    }
    status = service._status(current=(config.generations_root / "partial", manifest))
    assert status["ready"] is True
    assert status["stale"] is True
    assert "1 sources" in status["message"]
    assert not any(status["changes"][key] for key in ("added", "removed", "modified"))
    assert "retry" in status["message"]


def test_unselected_partial_mentions_manual_load(project, monkeypatch):
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(config, ultrarag=None)
    monkeypatch.setattr(
        service,
        "_generation_inventory",
        lambda current: {
            "generations": [{"generation_id": "partial", "partial": True}]
        },
    )
    status = asyncio.run(service.status())
    assert status["ready"] is False
    assert "retained partial" in status["message"]
    assert "Load" in status["message"]
    assert "retry" in status["message"]
