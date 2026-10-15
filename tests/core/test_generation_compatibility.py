"""Processing provenance stays diagnostic unless stored evidence cannot be served."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest

import research_rag.core.status as status_module
from research_rag.core.status import (
    generation_diagnostics,
    generation_upgrade_reasons,
)
from research_rag.project.config import resolve_config
from research_rag.project.policy import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
)


@pytest.mark.parametrize(
    "field,value,reason,required",
    [
        ("schema_version", 1, "generation_schema", False),
        ("extraction_policy_version", 0, "layout_extraction", False),
        ("cleaning_policy_version", 0, "semantic_cleaning", False),
        ("artifact_policy_version", 0, "generation_artifacts", False),
        ("metadata_storage_policy", "historical", "metadata_storage", False),
        ("retrieval_policy_fingerprint", "historical", "retrieval_policy", False),
        ("project_id", "different-project", "project_identity", True),
    ],
)
def test_policy_drift_is_verbose_only(
    project: Path,
    caplog: pytest.LogCaptureFixture,
    field: str,
    value: int | str,
    reason: str,
    required: bool,
) -> None:
    config = resolve_config(project)
    manifest = {
        "generation_id": "historical-generation",
        "project_id": config.project_id,
        "schema_version": SCHEMA_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "cleaning_policy_version": CLEANING_POLICY_VERSION,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "metadata_storage_policy": METADATA_STORAGE_POLICY,
        "retrieval_policy_fingerprint": "current",
        "source_files": [],
        "retrieval": {
            "available_methods": ["bm25"],
            "fusion": {
                "method": "weighted_reciprocal_rank_fusion",
                "rrf_k": config.settings.rrf_k,
                "bm25_weight": config.settings.bm25_weight,
                "dense_weight": config.settings.dense_weight,
            },
            "relevance_gates": {
                "dense_minimum_cosine_similarity": config.settings.dense_minimum_cosine_similarity,
                "bm25_requires_query_token_overlap": True,
            },
        },
    }
    manifest[field] = value
    before = copy.deepcopy(manifest)
    with caplog.at_level(logging.DEBUG, logger="research_rag.core.status"):
        reasons = generation_upgrade_reasons(config, manifest, "current")
    assert reasons == ([reason] if required else [])
    diagnostics = generation_diagnostics(config, manifest, "current")
    assert diagnostics["policy_differences"] == [reason]
    assert "historical-generation compatibility" in caplog.text
    assert reason in caplog.text
    assert manifest == before


@pytest.mark.parametrize("legacy_pdf", [True, False])
def test_mixed_pdf_provenance_is_diagnostic_and_never_requires_ingestion(
    project: Path, monkeypatch: pytest.MonkeyPatch, legacy_pdf: bool
) -> None:
    config = resolve_config(project)
    current_backend = {"backend": "docling", "fingerprint": "docling-current"}
    document = {"source_relative_path": "existing.pdf"}
    if not legacy_pdf:
        document["extraction_backend"] = current_backend
    manifest = {
        "project_id": config.project_id,
        "schema_version": SCHEMA_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "cleaning_policy_version": CLEANING_POLICY_VERSION,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "metadata_storage_policy": METADATA_STORAGE_POLICY,
        "retrieval_policy_fingerprint": "current",
        "extraction_backend": current_backend,
        "documents": [document],
        "source_files": [{"source_relative_path": "existing.pdf"}],
        "retrieval": {"available_methods": ["bm25"]},
    }
    before = copy.deepcopy(manifest)
    monkeypatch.setattr(
        status_module,
        "available_pdf_backend_record",
        lambda *_: {"fingerprint": "docling-current"},
    )

    diagnostics = generation_diagnostics(config, manifest, "current")

    assert ("extraction_backend" in diagnostics["policy_differences"]) is legacy_pdf
    assert generation_upgrade_reasons(config, manifest, "current") == []
    assert manifest == before
