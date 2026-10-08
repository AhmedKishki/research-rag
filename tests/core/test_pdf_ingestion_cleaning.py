"""PDF recovery reaches ingestion, diagnostics, search, and reuse."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest

from research_rag.core.service import ResearchService
from research_rag.core.tool_views import lean_ingest
from research_rag.corpus import extraction
from research_rag.corpus.sources import scan_sources
from research_rag.project.config import resolve_config
from research_rag.storage.records import read_json, read_jsonl
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


def test_pdf_native_retry_is_automatic_and_keeps_physical_page(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Readable opening page.", "Native amber evidence survives."])
    original = source.read_bytes()
    line_text = extraction._pdf_line_text

    def damaged_reader(spans: list[dict]) -> str:
        text = line_text(spans)
        return "\ufffd\ufffd broken character map" if "Native amber" in text else text

    monkeypatch.setattr(extraction, "_pdf_line_text", damaged_reader)
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )

    async def verify() -> None:
        result = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert result["status"] == "ready"
        assert result["pdf_text_recovery_pages"] == 1
        assert result["recovered_pdf_blocks"] == 1
        assert result["excluded_corrupt_passage_count"] == 0
        assert lean_ingest(result)["recovered_pdf_blocks"] == 1
        manifest = read_json(Path(result["generation_root"]) / "manifest.json")
        assert manifest["documents"][0]["recovered_pdf_blocks"] == 1
        direct_documents, direct_units = extraction.extract_sources(
            scan_sources(config).selected, maximum_unclean_percent=1.0
        )
        assert direct_documents == manifest["documents"]
        assert direct_units == read_jsonl(
            Path(result["generation_root"]) / "corpus/extracted-units.jsonl"
        )
        hits = await service.search("amber", retrieval_method="bm25")
        assert hits["hits"][0]["text"] == "Native amber evidence survives."
        assert hits["hits"][0]["locator"]["page"] == 2
        assert source.read_bytes() == original

    asyncio.run(verify())


def test_clean_pdf_does_not_invoke_native_retry(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "clean.pdf", ["Clean cobalt evidence."])

    def unexpected_retry(*_args):
        raise AssertionError("a clean passage must not pay for native recovery")

    monkeypatch.setattr(extraction, "recover_pdf_blocks", unexpected_retry)
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )
    result = asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))
    assert result["pdf_text_recovery_pages"] == 0
    assert result["cleaned_passage_count"] == 0


def test_pdf_bad_control_glyphs_do_not_disappear_before_screening(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Damaged region marker.\n\nReadable cobalt evidence."],
    )
    line_text = extraction._pdf_line_text

    def damaged_reader(spans: list[dict]) -> str:
        text = line_text(spans)
        return "lost\x01\x02glyphs" if "Damaged region" in text else text

    monkeypatch.setattr(extraction, "_pdf_line_text", damaged_reader)
    monkeypatch.setattr(extraction, "recover_pdf_blocks", lambda *_args: ({}, None))
    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extraction.extract_sources(
        scan_sources(config).selected, maximum_unclean_percent=1.0
    )
    assert documents[0]["excluded_corrupt_passage_count"] == 1
    assert all("lostglyphs" not in unit["contents"] for unit in units)


def test_pdf_layout_controls_remain_spaces(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Layout marker."])
    monkeypatch.setattr(extraction, "_pdf_line_text", lambda _spans: "one\ftwo\vthree.")

    def unexpected_retry(*_args):
        raise AssertionError("layout whitespace is not a damaged glyph")

    monkeypatch.setattr(extraction, "recover_pdf_blocks", unexpected_retry)
    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extraction.extract_sources(
        scan_sources(config).selected, maximum_unclean_percent=1.0
    )
    assert documents[0]["excluded_corrupt_passage_count"] == 0
    assert units[0]["contents"] == "one two three."


def test_geometric_retry_cannot_drop_a_readable_line_in_the_damaged_block(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Damaged region marker.\nReadable cobalt evidence."])
    line_text = extraction._pdf_line_text

    def damaged_reader(spans: list[dict]) -> str:
        text = line_text(spans)
        return "\ufffd\ufffd region marker." if "Damaged region" in text else text

    monkeypatch.setattr(extraction, "_pdf_line_text", damaged_reader)
    monkeypatch.setattr(
        extraction,
        "recover_pdf_blocks",
        lambda *_args: ({0: "Damaged region marker."}, None),
    )
    with pymupdf.open(source) as document:
        blocks = extraction._page_blocks(document[0], source_path=source)
    assert len(blocks) == 1
    assert not blocks[0].recovered
    assert "Readable cobalt evidence." in blocks[0].text


def test_pdf_irreparable_passage_does_not_abort_generation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Damaged region marker.\n\nReadable cobalt evidence."])
    original = source.read_bytes()
    line_text = extraction._pdf_line_text

    def damaged_reader(spans: list[dict]) -> str:
        text = line_text(spans)
        return "\ufffd" * 5000 if "Damaged region" in text else text

    monkeypatch.setattr(extraction, "_pdf_line_text", damaged_reader)
    monkeypatch.setattr(
        extraction,
        "recover_pdf_blocks",
        lambda *_args: ({}, "pdf_text_recovery_failed"),
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )

    async def verify() -> None:
        result = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert result["status"] == "ready"
        assert result["excluded_corrupt_passage_count"] == 1
        assert lean_ingest(result)["excluded_corrupt_passage_count"] == 1
        root = Path(result["generation_root"])
        manifest = read_json(root / "manifest.json")
        decision = manifest["documents"][0]["excluded_corrupt_passages"][0]
        assert decision["locator"]["page"] == 1
        assert "text" not in decision and "contents" not in decision
        assert "replacement_characters" in decision["reasons"]
        assert all(
            "\ufffd" not in item["contents"]
            for item in read_jsonl(root / "chunks/chunks.jsonl")
        )
        hits = await service.search("cobalt", retrieval_method="bm25")
        assert "Readable cobalt evidence" in hits["hits"][0]["text"]
        assert source.read_bytes() == original

    asyncio.run(verify())


def test_pdf_quality_threshold_changes_invalidate_reuse(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Damaged region marker.\n\nReadable cobalt evidence."],
    )
    line_text = extraction._pdf_line_text

    def damaged_reader(spans: list[dict]) -> str:
        text = line_text(spans)
        return "Small \ufffd passage." if "Damaged region" in text else text

    monkeypatch.setattr(extraction, "_pdf_line_text", damaged_reader)
    monkeypatch.setattr(extraction, "recover_pdf_blocks", lambda *_args: ({}, None))
    config = resolve_config(project, vanilla_executable=sys.executable)

    async def verify() -> None:
        first = ResearchService(
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        )
        built = await first.ingest(chunk_size=50, chunk_overlap=10)
        assert built["excluded_corrupt_passage_count"] == 1
        changed = replace(
            config, settings=replace(config.settings, maximum_unclean_percent=100.0)
        )
        second = ResearchService(
            changed,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),  # type: ignore[arg-type]
        )
        assert "passage_cleaning" in (await second.status())["upgrade_reasons"]
        rebuilt = await second.ingest(chunk_size=50, chunk_overlap=10)
        assert rebuilt["generation_id"] != built["generation_id"]
        assert rebuilt["reused_document_count"] == 0
        assert rebuilt["excluded_corrupt_passage_count"] == 0
        manifest = read_json(Path(rebuilt["generation_root"]) / "manifest.json")
        assert manifest["passage_cleaning"]["maximum_unclean_percent"] == 100.0
        unchanged = await second.ingest(chunk_size=50, chunk_overlap=10)
        assert unchanged["status"] == "unchanged"

    asyncio.run(verify())
