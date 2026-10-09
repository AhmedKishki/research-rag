"""The PDF backend is a per-source reuse rule, never a generation-wide gate.

A backend change must not discard an otherwise compatible generation, because its
EPUB units and text-keyed vectors are still reusable. The staged path refuses to
reuse a PDF whose recorded backend differs; that is exercised in
`test_docling_staged_ingestion.py`, and here the generation-wide rule is shown
not to fire.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_rag.core import status as status_module
from research_rag.corpus.pdf_backend import CUSTOM_BACKEND_FINGERPRINT
from research_rag.generations.generation import generation_is_reusable
from research_rag.project.config import resolve_config
from research_rag.project.support import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    SCHEMA_VERSION,
    _checkpoint_identity,
)
from research_rag.retrieval.embeddings import resolve_embedding_model

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
    return generation_is_reusable(
        manifest,
        schema_version=SCHEMA_VERSION,
        extraction_policy_version=EXTRACTION_POLICY_VERSION,
        cleaning_policy_version=CLEANING_POLICY_VERSION,
        artifact_policy_version=ARTIFACT_POLICY_VERSION,
        project_id=_PROJECT_ID,
        chunk_size=100,
        chunk_overlap=10,
        chunk_headers=False,
        embedding=_MODEL,
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
def test_backend_upgrade_reason_only_applies_to_pdf_sources(
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

    reasons = status_module.generation_upgrade_reasons(config, manifest, "test")

    assert ("extraction_backend" in reasons) is (extension == ".pdf")
