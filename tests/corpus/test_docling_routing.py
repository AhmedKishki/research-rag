"""Docling routing through the direct and staged build paths, with a fake worker.

No Docling import, no model load, no conversion: the worker result is a recorded
mapping, so these tests exercise only the parent's routing, adaptation, and gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import research_rag.corpus.extraction as extraction_module
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.extraction import (
    ExtractionError,
    docling_pdf_batch,
)
from research_rag.corpus.pdf_backend import DoclingConversionError
from research_rag.corpus.sources import scan_sources, sha256_file
from research_rag.project.config import resolve_config
from tests.conftest import write_pdf
from tests.extraction_helpers import extract_pdf_docling, extract_sources

_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={"docling": "2.135.0"},
)

_PAGES = {
    1: "First page cobalt evidence argues the case in prose.",
    2: "Second page amber evidence continues the argument.",
}


def _canned_result(pages_text: dict[int, str], start: int, end: int) -> dict[str, Any]:
    pages: dict[str, list[dict[str, Any]]] = {}
    for page_no, text in pages_text.items():
        if not (start <= page_no <= end):
            continue
        pages[str(page_no)] = [
            {
                "contents": text,
                "content_kind": "prose",
                "provenance": [
                    {
                        "page_no": page_no,
                        "charspan": [0, len(text)],
                        "bbox": {"l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
                    }
                ],
            }
        ]
    return {
        "version": "2.135.0",
        "options": {"ocr": False, "vlm": False, "table_structure": False},
        "boundary": {"memory.max": str(2 * 1024**3)},
        "pages": pages,
        "diagnostics": {
            "retained_item_count": len(pages),
            "excluded_furniture_item_count": 1,
            "excluded_picture_child_item_count": 0,
        },
        "pdf_metadata": {"title": "Docling PDF", "author": "Ada Example"},
        "page_labels": {str(page): str(page) for page in pages_text},
    }


def _install_fake_worker(
    monkeypatch: pytest.MonkeyPatch, pages_text: dict[int, str] | None = None
) -> None:
    full = dict(pages_text or _PAGES)

    def fake_run(
        *,
        source_path: Path,
        start: int,
        end: int,
        offline: bool,
        model_cache_root: Path | None,
        docling_env: DoclingEnv,
        timeout: float = 180,
        worker: Any = None,
    ) -> dict[str, Any]:
        return _canned_result(full, start, end)

    monkeypatch.setattr(extraction_module, "run_docling_batch", fake_run)


def _source(project: Path):
    write_pdf(
        project / "sources" / "docling.pdf",
        ["first", "second"],
        title="Docling PDF",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    return config, scan_sources(config).selected[0]


def _staged_units(source, digest: str) -> list[dict[str, Any]]:
    """Replay the staged state machine's page-batch loop, in memory."""

    document = None
    units: list[dict[str, Any]] = []
    for start in range(0, 2, 1):
        pages, document, _empty, _removals, _diagnostics = docling_pdf_batch(
            source,
            document,
            start,
            start + 1,
            digest=digest,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
            total_pages=2,
        )
        for page_index in sorted(pages):
            units.extend(pages[page_index])
    return units


def test_staged_and_direct_paths_reach_the_same_units(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(monkeypatch)
    _config, source = _source(project)
    digest = sha256_file(source.path)

    document, direct = extract_pdf_docling(
        source,
        digest,
        page_batch_size=1,
        offline=True,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
        total_pages=2,
    )
    staged = _staged_units(source, digest)

    assert [(unit["id"], unit["contents"]) for unit in direct] == [
        (unit["id"], unit["contents"]) for unit in staged
    ]
    assert [unit["locator"]["page"] for unit in direct] == [1, 2]
    assert all(unit["locator"]["type"] == "pdf_page" for unit in direct)
    assert all(unit["provenance"] for unit in direct)
    assert document["extraction_backend"]["backend"] == "docling"
    assert document["physical_pages"] == 2


def test_docling_units_pass_the_shared_gate_with_the_configured_budget(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(monkeypatch)
    _config, source = _source(project)
    captured: list[float | None] = []

    def spy(
        screen_source, document, units, *, removals=None, maximum_unclean_percent=None
    ):
        captured.append(maximum_unclean_percent)
        return []

    monkeypatch.setattr(extraction_module, "screen_source_units", spy)

    extract_sources(
        [source],
        maximum_unclean_percent=2.0,
        pdf_backend="docling",
        docling_env=_FAKE_ENV,
    )

    # The configured 2% is shared; Docling does not cap it to 1%.
    assert captured == [2.0]


def test_structural_labels_travel_with_a_unit_and_absent_labels_add_no_key(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "labels.pdf", ["one"], title="T")
    config = resolve_config(project, vanilla_executable=sys.executable)
    source = scan_sources(config).selected[0]
    digest = sha256_file(source.path)

    result = _canned_result({1: "Labelled evidence."}, 1, 1)
    labelled = result["pages"]["1"][0]
    labelled["docling_label"] = "section_header"
    labelled["docling_content_layer"] = "body"
    labelled["docling_ref"] = "#/texts/0"
    result["pages"]["1"].append(
        {"contents": "Plain evidence.", "content_kind": "prose", "provenance": []}
    )
    monkeypatch.setattr(extraction_module, "run_docling_batch", lambda **_k: result)

    pages, _document, _empty, _removals, _diagnostics = docling_pdf_batch(
        source,
        None,
        0,
        1,
        digest=digest,
        offline=True,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
        total_pages=1,
    )

    labelled_unit, plain_unit = pages[0]
    assert labelled_unit["docling_label"] == "section_header"
    assert labelled_unit["docling_content_layer"] == "body"
    assert labelled_unit["docling_ref"] == "#/texts/0"
    # A raw unit without a label carries no key for one.
    assert "docling_label" not in plain_unit
    assert "docling_content_layer" not in plain_unit
    assert "docling_ref" not in plain_unit


def test_retired_custom_alias_routes_to_docling_with_the_configured_budget(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(monkeypatch)
    _config, source = _source(project)
    captured: list[float | None] = []

    def spy(
        screen_source, document, units, *, removals=None, maximum_unclean_percent=None
    ):
        captured.append(maximum_unclean_percent)
        return []

    monkeypatch.setattr(extraction_module, "screen_source_units", spy)

    # "custom" is a retired read alias: settings/backend normalize it to the
    # operational Docling backend, and the configured budget is preserved.
    extract_sources(
        [source],
        maximum_unclean_percent=2.0,
        pdf_backend="custom",
        docling_env=_FAKE_ENV,
    )

    assert captured == [2.0]


def test_conversion_failure_is_a_source_local_omission(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, source = _source(project)
    digest = sha256_file(source.path)

    def failing_run(**_kwargs: Any) -> dict[str, Any]:
        raise DoclingConversionError("worker could not read the page")

    monkeypatch.setattr(extraction_module, "run_docling_batch", failing_run)

    with pytest.raises(ExtractionError):
        extract_pdf_docling(
            source,
            digest,
            page_batch_size=1,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
            total_pages=2,
        )


def test_a_blank_requested_page_is_counted_once(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(project / "sources" / "blank.pdf", ["one", "two", "three"], title="T")
    config = resolve_config(project, vanilla_executable=sys.executable)
    source = scan_sources(config).selected[0]
    digest = sha256_file(source.path)
    # The worker emits nothing for page 2, so it carries no page key at all.
    _install_fake_worker(monkeypatch, {1: "Page one only.", 3: "Page three only."})

    pages, _document, empty, removals, _diagnostics = docling_pdf_batch(
        source,
        None,
        0,
        3,
        digest=digest,
        offline=True,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
        total_pages=3,
    )

    assert empty == 1
    assert set(pages) == {0, 2}
    # The blank page is counted as empty once, not again as an image-only page.
    assert removals["image_only_pages"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"unsupported_item_count": 1},
        {"missing_child_ref_count": 1},
        {"missing_caption_ref_count": 2},
    ],
)
def test_an_incomplete_conversion_is_refused_before_success(
    project: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, int]
) -> None:
    _config, source = _source(project)
    digest = sha256_file(source.path)
    result = _canned_result(_PAGES, 1, 1)
    result["diagnostics"] = {**result["diagnostics"], **extra}
    monkeypatch.setattr(extraction_module, "run_docling_batch", lambda **_k: result)

    with pytest.raises(ExtractionError):
        docling_pdf_batch(
            source,
            None,
            0,
            1,
            digest=digest,
            offline=True,
            model_cache_root=None,
            docling_env=_FAKE_ENV,
            total_pages=2,
        )


def test_image_only_pages_forward_to_source_removals(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, source = _source(project)
    digest = sha256_file(source.path)
    result = _canned_result(_PAGES, 1, 1)
    result["image_only_pages"] = 2
    monkeypatch.setattr(extraction_module, "run_docling_batch", lambda **_k: result)

    _pages, _document, _empty, removals, _diagnostics = docling_pdf_batch(
        source,
        None,
        0,
        1,
        digest=digest,
        offline=True,
        model_cache_root=None,
        docling_env=_FAKE_ENV,
        total_pages=2,
    )

    assert removals["image_only_pages"] == 2
