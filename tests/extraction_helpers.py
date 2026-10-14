"""Whole-source extraction for tests: the per-batch calls ingestion makes, in one call.

Ingestion extracts a source batch by checkpointed batch. A test of what extraction
produces wants the finished source, so this composes the same production calls
over a whole source without the checkpoints.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_rag.corpus import extraction
from research_rag.corpus.docling_env import DoclingEnv, DoclingEnvironmentError
from research_rag.corpus.pdf_backend import (
    DEFAULT_PDF_BACKEND,
    effective_maximum_unclean_percent,
    normalize_pdf_backend,
)
from research_rag.corpus.sources import SourceFile, sha256_file


def extract_epub(
    source: SourceFile,
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = digest or sha256_file(source.path)
    record, spine_items = extraction.prepare_epub_extraction(source, digest)
    units: list[dict[str, Any]] = []
    empty_sections = 0
    removals = extraction.empty_removal_counts()
    for spine_index in range(spine_items):
        batch, empty, counts = extraction.extract_epub_spine_item(
            source, record, spine_index
        )
        units.extend(batch)
        empty_sections += int(empty)
        removals = extraction.merge_removal_counts(removals, counts)
    units, section_counts = extraction.strip_non_argument_units(units)
    removals = extraction.merge_removal_counts(removals, section_counts)
    record.update(removals)
    record["extracted_units"] = len(units)
    record["empty_units"] = empty_sections
    if not units:
        raise extraction.ExtractionError(
            extraction._no_text_layer_message(source, "EPUB")
        )
    return record, units


def _pdf_loss_limit(
    backend: str,
    configured: float | None,
) -> float | None:
    """The loss budget the shared gate applies, or None when none was given."""

    if configured is None:
        return None
    return effective_maximum_unclean_percent(backend, float(configured))


def extract_pdf_docling(
    source: SourceFile,
    digest: str | None = None,
    *,
    page_batch_size: int = 8,
    offline: bool = False,
    model_cache_root: Path | None = None,
    docling_env: DoclingEnv,
    total_pages: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Direct Docling extraction of one PDF, page batch by page batch."""

    digest = digest or sha256_file(source.path)
    total = int(total_pages) if total_pages else extraction.pdf_page_count(source)
    document: dict[str, Any] | None = None
    units: list[dict[str, Any]] = []
    empty = 0
    removals = extraction.empty_removal_counts()
    diagnostics: Counter[str] = Counter()
    batch = max(1, int(page_batch_size))
    for start in range(0, total, batch):
        end = min(start + batch, total)
        pages, document, batch_empty, batch_removals, batch_diag = (
            extraction.docling_pdf_batch(
                source,
                document,
                start,
                end,
                digest=digest,
                offline=offline,
                model_cache_root=model_cache_root,
                docling_env=docling_env,
                total_pages=total,
            )
        )
        empty += batch_empty
        removals = extraction.merge_removal_counts(removals, batch_removals)
        for page_index in sorted(pages):
            units.extend(pages[page_index])
        for key, value in batch_diag.items():
            diagnostics[key] += int(value)
    if document is None:
        raise extraction.ExtractionError(
            extraction._no_text_layer_message(source, "PDF")
        )
    document.update(
        {
            "extracted_units": len(units),
            "empty_units": empty,
            **removals,
        }
    )
    document["docling_diagnostics"] = dict(diagnostics)
    if not units:
        raise extraction.ExtractionError(
            extraction._no_text_layer_message(source, "PDF")
        )
    return document, units


def extract_sources(
    sources: tuple[SourceFile, ...],
    source_digests: dict[str, str] | None = None,
    *,
    maximum_unclean_percent: float | None = None,
    pdf_backend: str = DEFAULT_PDF_BACKEND,
    page_batch_size: int = 8,
    offline: bool = False,
    model_cache_root: Path | None = None,
    docling_env: DoclingEnv | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract every source, routing every PDF through the Docling worker.

    ``pdf_backend`` stays a public keyword for a caller that still passes a
    project setting, but it is only normalized and never selects a reader: the
    bundled native reader is retired, so a PDF is always converted by the
    managed Docling worker. ``docling_env`` must be the environment the caller
    resolved, because this path never provisions one. EPUB and MOBI are
    unchanged.
    """

    backend = normalize_pdf_backend(pdf_backend)
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    limit = _pdf_loss_limit(backend, maximum_unclean_percent)
    for source in sources:
        digest = (source_digests or {}).get(source.source_relative_path)
        if source.extension == ".pdf":
            if docling_env is None:
                raise DoclingEnvironmentError(
                    "Docling extraction requires the managed environment, which "
                    "was not resolved before extraction began."
                )
            document, extracted = extract_pdf_docling(
                source,
                digest,
                page_batch_size=page_batch_size,
                offline=offline,
                model_cache_root=model_cache_root,
                docling_env=docling_env,
            )
        elif source.extension == ".epub":
            document, extracted = extract_epub(source, digest)
        elif source.extension == ".mobi":
            document, extracted = extraction.extract_mobi(source, digest)
        else:  # pragma: no cover
            raise extraction.ExtractionError(
                f"Unsupported source format: {source.path}"
            )
        retained = extraction.screen_source_units(
            source,
            document,
            extracted,
            maximum_unclean_percent=limit,
        )
        documents.append(document)
        units.extend(retained)
    return documents, units
