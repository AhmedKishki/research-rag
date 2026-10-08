"""PDF substantive-loss totals reach build metrics, lean answers, and source stats."""

from __future__ import annotations

import asyncio
import sys
import tomllib
from pathlib import Path

from research_rag.core.service import ResearchService
from research_rag.core.stats import StatsWorkflow
from research_rag.core.tool_views import lean_ingest
from research_rag.generations.ingestion import _document_loss_totals
from research_rag.project.config import resolve_config
from research_rag.project.settings import SETTINGS_BY_KEY, default_config_file
from research_rag.storage.records import read_json
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

LOSS_FIELDS = (
    "substantive_character_count",
    "discarded_corrupt_character_count",
    "cleaned_corrupt_span_count",
    "partially_cleaned_passage_count",
)


def test_pdf_loss_totals_sum_each_document_field() -> None:
    """A field a source lacks counts as zero rather than a guessed figure."""

    documents = [
        {
            "substantive_character_count": 1000,
            "discarded_corrupt_character_count": 20,
            "cleaned_corrupt_span_count": 3,
            "partially_cleaned_passage_count": 2,
        },
        {"substantive_character_count": 500, "cleaned_corrupt_span_count": 1},
    ]

    assert _document_loss_totals(documents) == {
        "substantive_character_count": 1500,
        "discarded_corrupt_character_count": 20,
        "cleaned_corrupt_span_count": 4,
        "partially_cleaned_passage_count": 2,
    }
    assert _document_loss_totals([]) == dict.fromkeys(LOSS_FIELDS, 0)


def test_lean_ingest_surfaces_pdf_loss_counters_only_when_nonzero() -> None:
    """Each loss counter reaches a lean answer while it is nonzero."""

    quiet = lean_ingest(
        {
            "status": "ready",
            "discarded_corrupt_character_count": 0,
            "cleaned_corrupt_span_count": 0,
            "partially_cleaned_passage_count": 0,
        }
    )
    for key in (
        "discarded_corrupt_character_count",
        "cleaned_corrupt_span_count",
        "partially_cleaned_passage_count",
    ):
        assert key not in quiet

    loud = lean_ingest(
        {
            "status": "ready",
            "discarded_corrupt_character_count": 12,
            "cleaned_corrupt_span_count": 4,
            "partially_cleaned_passage_count": 2,
            "substantive_character_count": 900,
        }
    )
    assert loud["discarded_corrupt_character_count"] == 12
    assert loud["cleaned_corrupt_span_count"] == 4
    assert loud["partially_cleaned_passage_count"] == 2
    # The denominator is always present and restates no loss, so it is not a
    # disclosure a lean answer carries.
    assert "substantive_character_count" not in loud


def test_build_facts_exposes_recorded_pdf_loss_totals_and_rate() -> None:
    """The stats answer reads the build's own loss figures, not a recomputation."""

    facts = StatsWorkflow._build_facts(
        {
            "build_metrics": {
                "substantive_character_count": 1000,
                "discarded_corrupt_character_count": 25,
                "cleaned_corrupt_span_count": 6,
                "partially_cleaned_passage_count": 3,
                "unclean_character_rate": 0.025,
            }
        }
    )

    assert facts["substantive_character_count"] == 1000
    assert facts["discarded_corrupt_character_count"] == 25
    assert facts["cleaned_corrupt_span_count"] == 6
    assert facts["partially_cleaned_passage_count"] == 3
    assert facts["unclean_character_rate"] == 0.025


def test_passage_cleaning_setting_describes_a_per_document_loss_cap() -> None:
    """The setting keeps its name and default, and states the new meaning."""

    setting = SETTINGS_BY_KEY["ingestion.maximum_unclean_percent"]
    assert setting.field == "maximum_unclean_percent"
    assert setting.layer == "identity"
    document = tomllib.loads(default_config_file().read_text(encoding="utf-8"))
    assert document["ingestion"]["maximum_unclean_percent"] == 1.0

    doc = setting.doc
    assert "substantive" in doc
    assert "one PDF" in doc
    assert "100" in doc
    assert "never a PDF with none" in doc
    assert "EPUB" in doc


def test_ingest_records_aggregate_pdf_loss_totals(project: Path) -> None:
    """A build sums each document's loss fields and derives the corpus rate."""

    write_pdf(project / "sources" / "article.pdf", ["Readable cobalt evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),  # type: ignore[arg-type]
    )

    result = asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))
    assert result["status"] == "ready"
    manifest = read_json(Path(result["generation_root"]) / "manifest.json")
    metrics = manifest["build_metrics"]
    totals = _document_loss_totals(manifest["documents"])
    for key, value in totals.items():
        assert metrics[key] == value
    expected_rate = (
        round(
            totals["discarded_corrupt_character_count"]
            / totals["substantive_character_count"],
            6,
        )
        if totals["substantive_character_count"]
        else 0.0
    )
    assert metrics["unclean_character_rate"] == expected_rate
    facts = StatsWorkflow._build_facts(manifest)
    for key in (*LOSS_FIELDS, "unclean_character_rate"):
        assert key in facts
