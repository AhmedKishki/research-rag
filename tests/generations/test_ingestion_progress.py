from copy import deepcopy
from types import SimpleNamespace

import pytest

from research_rag.generations.ingestion import IngestionWorkflow
from research_rag.project.support import _source_work_key
from research_rag.storage.records import atomic_write_json


@pytest.fixture
def workflow(tmp_path):
    workflow = IngestionWorkflow()
    workflow.config = SimpleNamespace(staging_root=tmp_path / "staging")
    return workflow


@pytest.fixture
def checkpoint():
    return {
        "build_id": "20261007T210000Z-1234abcd",
        "phase": "extraction",
        "source_inventory": [
            {"source_relative_path": path, "format": "pdf"}
            for path in ("a.pdf", "nested/b.pdf")
        ],
        "selected_source_paths": ["a.pdf", "nested/b.pdf"],
        "extracted_source_paths": ["a.pdf"],
        "extraction_work_completed": 7,
        "extraction_work_total": 8,
    }


def state_path(workflow, checkpoint):
    return (
        workflow.config.staging_root
        / checkpoint["build_id"]
        / "work"
        / "sources"
        / _source_work_key("nested/b.pdf")
        / "state.json"
    )


def test_extraction_uses_source_totals_without_changing_checkpoint(
    workflow, checkpoint
):
    path = state_path(workflow, checkpoint)
    path.parent.mkdir(parents=True)
    atomic_write_json(
        path,
        {
            "extraction_stage": "pdf_pages",
            "total": 10,
            "next_index": 4,
            "page_batch_size": 2,
        },
    )
    before = deepcopy(checkpoint)
    result = workflow._ingestion_progress(checkpoint)
    assert result["source"] == "nested/b.pdf"
    assert result["source_progress"] == {
        "completed": 8,
        "total": 12,
        "unit": "pdf_page_batches_or_epub_sections",
    }
    assert result["overall_progress"] == {"completed": 1, "total": 2, "unit": "sources"}
    assert result["eta_seconds"] is None
    assert checkpoint == before


def test_chunking_reads_only_pending_source(workflow, checkpoint):
    checkpoint.update(phase="chunking", chunked_source_paths=["a.pdf"])
    path = state_path(workflow, checkpoint)
    path.parent.mkdir(parents=True)
    atomic_write_json(path, {"chunking_work_total": 9, "chunked_unit_count": 3})
    result = workflow._ingestion_progress(checkpoint)
    assert result["source_progress"] == {
        "completed": 3,
        "total": 9,
        "unit": "extraction_units",
    }
    assert result["overall_progress"]["completed"] == 1


@pytest.mark.parametrize(
    "phase,key",
    [
        ("source_hashing", "source_digests"),
        ("source_revalidation", "revalidation_digests"),
    ],
)
def test_hashing_uses_inventory_not_selected(workflow, checkpoint, phase, key):
    checkpoint.update(
        phase=phase, selected_source_paths=[], **{key: {"a.pdf": "digest"}}
    )
    result = workflow._ingestion_progress(checkpoint)
    assert result["source"] == "nested/b.pdf"
    assert result["overall_progress"] == {"completed": 1, "total": 2, "unit": "sources"}


@pytest.mark.parametrize(
    "unsafe", ["build", "symlink", "oversized", "malformed", "missing"]
)
def test_source_state_is_bounded_and_safe(workflow, checkpoint, unsafe):
    path = state_path(workflow, checkpoint)
    path.parent.mkdir(parents=True)
    if unsafe == "build":
        checkpoint["build_id"] = "../../outside"
    elif unsafe == "symlink":
        path.symlink_to(path.parent / "outside.json")
    elif unsafe == "oversized":
        path.write_text(" " * (64 * 1024 + 1))
    elif unsafe == "malformed":
        path.write_text("not json")
    assert workflow._ingestion_progress(checkpoint)["source_progress"] is None


def test_global_phase_has_no_invented_source_or_eta(workflow, checkpoint):
    checkpoint.update(
        phase="embedding",
        embedded_chunk_count=2,
        chunk_count=5,
        phase_timings_seconds={"embedding": 10},
    )
    result = workflow._ingestion_progress(checkpoint)
    assert result["source"] is None
    assert result["source_progress"] is None
    assert result["overall_progress"] == {"completed": 2, "total": 5, "unit": "chunks"}
    assert result["eta_seconds"] is None
