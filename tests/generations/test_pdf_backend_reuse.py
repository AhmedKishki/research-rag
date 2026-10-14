"""The PDF backend is provenance, never a generation-wide reuse gate.

A backend or adapter change must not discard an otherwise compatible generation,
because its EPUB units and text-keyed vectors are still reusable and an unchanged
PDF is passed through whole. The staged path that keeps an unchanged legacy PDF's
own provenance is exercised in `test_docling_staged_ingestion.py`, and here the
generation-wide rule is shown not to fire.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_rag.core import status as status_module
from research_rag.corpus.pdf_backend import CUSTOM_BACKEND_FINGERPRINT
from research_rag.generations.generation import (
    chunks_are_reusable,
    units_are_reusable,
    vectors_are_reusable,
)
from research_rag.generations.staging import _checkpoint_identity
from research_rag.project.config import resolve_config
from research_rag.project.embeddings import resolve_embedding_model
from research_rag.project.policy import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    REFLOWABLE_EXTRACTION_POLICY_VERSION,
    SCHEMA_VERSION,
)

_MODEL = resolve_embedding_model("BAAI/bge-small-en-v1.5")
_PROJECT_ID = "proj_test"


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "cleaning_policy_version": CLEANING_POLICY_VERSION,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "project_id": _PROJECT_ID,
        "chunking": {
            "backend": "UltraRAG token chunker",
            "tokenizer": "gpt2",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "headers": False,
        },
        "retrieval": {
            "dense": {
                "embedding_model": _MODEL.name,
                "embedding_model_revision": _MODEL.revision,
                "embedding_dimension": _MODEL.dimension,
                "embedding_model_repository": _MODEL.repository,
            }
        },
    }


def _reusable(manifest: dict[str, Any]) -> bool:
    return (
        units_are_reusable(
            manifest,
            schema_version=SCHEMA_VERSION,
            extraction_policy_version=EXTRACTION_POLICY_VERSION,
            cleaning_policy_version=CLEANING_POLICY_VERSION,
            artifact_policy_version=ARTIFACT_POLICY_VERSION,
            project_id=_PROJECT_ID,
        )
        and chunks_are_reusable(
            manifest, chunk_size=100, chunk_overlap=10, chunk_headers=False
        )
        and vectors_are_reusable(manifest, embedding=_MODEL)
    )


def test_legacy_manifest_without_a_backend_stays_reusable() -> None:
    assert _reusable(_manifest()) is True


def test_a_recorded_backend_does_not_gate_the_generation_globally() -> None:
    manifest = _manifest()
    manifest["extraction_backend"] = {
        "backend": "docling",
        "version": "2.135.0",
        "options": {},
        "fingerprint": "pdf-backend:docling:2.135.0:deadbeef",
    }

    # The generation still loads; the PDF-specific refusal happens per source.
    assert _reusable(manifest) is True


def test_checkpoint_identity_changes_with_the_backend() -> None:
    common = {
        "project_id": _PROJECT_ID,
        "inventory": [],
        "exclusion_revision": "none",
        "baseline_generation_id": None,
        "chunk_size": 100,
        "chunk_overlap": 10,
        "chunk_headers": False,
        "force_recompute": False,
        "embedding": _MODEL,
        "maximum_unclean_percent": 1.0,
    }

    custom = _checkpoint_identity(
        **common, pdf_backend_fingerprint=CUSTOM_BACKEND_FINGERPRINT
    )
    docling = _checkpoint_identity(
        **common, pdf_backend_fingerprint="pdf-backend:docling:2.135.0:deadbeef"
    )

    assert custom != docling


@pytest.mark.parametrize("extension", [".epub", ".pdf"])
def test_backend_drift_is_diagnostic_and_pdf_scoped(
    tmp_path, monkeypatch, extension
) -> None:
    config = resolve_config(tmp_path)
    manifest = _manifest()
    manifest["source_files"] = [{"source_relative_path": f"book{extension}"}]
    monkeypatch.setattr(
        status_module, "pdf_backend_available", lambda backend, config: True
    )
    monkeypatch.setattr(
        status_module, "pdf_backend_fingerprint", lambda backend, config: "new"
    )

    # A recorded backend is provenance, so it never names a serving upgrade. It is
    # a diagnostic difference, and only a PDF source carries one.
    assert "extraction_backend" not in status_module.generation_upgrade_reasons(
        config, manifest, "test"
    )
    differences = status_module.generation_policy_differences(config, manifest, "test")
    assert ("extraction_backend" in differences) is (extension == ".pdf")


def test_older_epub_rules_are_a_diagnostic_difference(tmp_path, monkeypatch) -> None:
    config = resolve_config(tmp_path)
    manifest = _manifest()
    manifest["documents"] = [{"format": "pdf"}, {"format": "epub"}]
    monkeypatch.setattr(
        status_module, "pdf_backend_available", lambda backend, config: False
    )

    # The selected generation stays servable; ingest reads the EPUB again.
    assert "reflowable_extraction" not in status_module.generation_upgrade_reasons(
        config, manifest, "test"
    )
    assert "reflowable_extraction" in status_module.generation_policy_differences(
        config, manifest, "test"
    )
    manifest["documents"][1]["reflowable_extraction_policy_version"] = (
        REFLOWABLE_EXTRACTION_POLICY_VERSION
    )
    assert "reflowable_extraction" not in status_module.generation_policy_differences(
        config, manifest, "test"
    )
