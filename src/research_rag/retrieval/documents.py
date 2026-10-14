"""The documents a generation records, as a reader sees them.

Reviewed metadata is overlaid at read time without rewriting the generation,
a search and a source listing filter by the same predicate, and a passage is
named by its public form and its citation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..corpus.sources import normalize_metadata, stable_source_id
from ..corpus.text_normalization import normalize_inline_text, normalize_reading_text
from ..corpus.text_quality import text_health_reasons, text_script_notes
from ..project.policy import METADATA_STORAGE_POLICY, ResearchError, value_fingerprint
from .artifact_lookup import chunk_text


def _normalized_filter(values: list[str] | None) -> set[str]:
    return {
        normalized.casefold()
        for value in values or []
        if (normalized := normalize_inline_text(str(value)))
    }


def _normalized_scalar(value: Any) -> set[str]:
    """Normalize one scalar metadata value, or nothing when it is empty.

    `title` is a string where `authors` and the other list fields are lists, and
    iterating a string yields its characters.
    """

    text = normalize_inline_text(str(value or ""))
    return {text.casefold()} if text else set()


def _matches_name_filter(supplied: set[str], values: set[str]) -> bool:
    """Whether one of the supplied phrases appears inside one of these values.

    Titles and author names are phrases, not controlled tags, so a substring is the
    right comparison. Both sides arrive casefolded.
    """

    if not supplied:
        return True
    return any(
        any(name in document_value for document_value in values) for name in supplied
    )


def _document_matches_metadata(
    document: dict[str, Any],
    *,
    keywords: set[str],
    categories_any: set[str] = frozenset(),
    projects_any: set[str] = frozenset(),
    languages_any: set[str] = frozenset(),
    authors_any: set[str] = frozenset(),
    titles_any: set[str] = frozenset(),
    categories: set[str] = frozenset(),
    projects: set[str] = frozenset(),
) -> bool:
    """Match one document against the reviewed-metadata filter layers.

    `keywords`, `categories` and `projects` without the `_any` suffix are all-of;
    the `_any` forms, `language` and `authors` are any-of. The project layer is a
    passthrough inside a one-project server and matters when a corpus is copied or
    shared.

    `title` and `authors` are names rather than tags, so they match by
    case-insensitive substring. Reviewed values override extracted ones, so these
    filters read the effective document the query path already holds.
    """

    document_categories = _normalized_filter(document.get("categories"))
    document_keywords = _normalized_filter(document.get("keywords"))
    document_projects = _normalized_filter(document.get("project"))
    document_languages = _normalized_filter(document.get("language"))
    document_authors = _normalized_filter(document.get("authors"))
    document_titles = _normalized_scalar(document.get("title"))
    return (
        keywords.issubset(document_keywords)
        and categories.issubset(document_categories)
        and projects.issubset(document_projects)
        and (not categories_any or not categories_any.isdisjoint(document_categories))
        and (not projects_any or not projects_any.isdisjoint(document_projects))
        and (not languages_any or not languages_any.isdisjoint(document_languages))
        and _matches_name_filter(authors_any, document_authors)
        and _matches_name_filter(titles_any, document_titles)
    )


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata without extraction-related line wrapping."""

    result = dict(document)
    # These fields existed in older immutable generations and are internal
    # implementation details, not reviewed metadata or useful diagnostics.
    result.pop("metadata_confidence", None)
    result.pop("metadata_override_revision", None)
    for field in ("title", "doi"):
        result[field] = normalize_inline_text(str(result.get(field) or ""))
    for field in ("authors", "categories", "keywords", "language", "project"):
        result[field] = [
            normalized
            for value in result.get(field) or []
            if (normalized := normalize_inline_text(str(value)))
        ]
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = list(result.get("metadata_warnings") or [])
    if provenance.get("title") != "reviewed_override" and text_health_reasons(
        result["title"]
    ):
        result["title"] = Path(str(result.get("source_path") or "source")).stem
        warnings.append("corrupt_extracted_title")
    if provenance.get("authors") != "reviewed_override":
        clean_authors = [
            author for author in result["authors"] if not text_health_reasons(author)
        ]
        if len(clean_authors) != len(result["authors"]):
            warnings.append("corrupt_extracted_authors")
        result["authors"] = clean_authors
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    return result


def _canonical_metadata_override(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize one override while preserving explicit empty reviewed values."""

    normalized = normalize_metadata(value)
    result: dict[str, Any] = {}
    for field in ("title", "doi"):
        if field in normalized:
            result[field] = normalize_inline_text(str(normalized[field]))
    for field in ("authors", "categories", "keywords", "language", "project"):
        if field not in normalized:
            continue
        items: list[str] = []
        seen: set[str] = set()
        for raw in normalized[field]:
            item = normalize_inline_text(str(raw))
            key = item.casefold()
            if item and key not in seen:
                items.append(item)
                seen.add(key)
        result[field] = items
    if "year" in normalized:
        result["year"] = normalized["year"]
    return result


def _metadata_snapshot_changed(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> bool:
    stored_revision = manifest.get("metadata_revision")
    if stored_revision is None:
        return bool(metadata)
    return stored_revision != value_fingerprint(metadata)


def _effective_document_metadata(
    document: dict[str, Any],
    override: dict[str, Any],
    *,
    unknown_legacy_snapshot_mismatch: bool = False,
) -> dict[str, Any]:
    """Overlay current reviewed metadata without mutating generation artifacts.

    Generation documents retain the metadata snapshot used while building their
    indexes; the portable reviewed-metadata file is authoritative at read time.

    An older generation does not retain the automatic value a reviewed override hid,
    so removing one uses a deterministic fallback.
    """

    normalized_override = _canonical_metadata_override(override)
    override_revision = value_fingerprint(normalized_override)
    stored_override_revision = document.get("metadata_override_revision")
    if stored_override_revision == override_revision:
        result = dict(document)
        result.pop("metadata_confidence", None)
        return result

    result = dict(document)
    result.pop("metadata_confidence", None)
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = [
        str(item)
        for item in result.get("metadata_warnings") or []
        if item
        not in {
            "authors_missing",
            "title_from_filename",
            "automatic_metadata_unavailable_after_override_removal",
        }
    ]
    removed_reviewed_value = False
    unknown_legacy_snapshot = (
        unknown_legacy_snapshot_mismatch
        and not provenance
        and stored_override_revision != override_revision
    )

    fallbacks: dict[str, Any] = {
        "title": Path(str(result.get("source_path") or "source")).stem,
        "authors": [],
        "year": None,
        "doi": "",
        "language": [],
    }
    for field in ("title", "authors", "year", "doi", "language"):
        if field in normalized_override:
            result[field] = normalized_override[field]
            provenance[field] = "reviewed_override"
        elif provenance.get(field) == "reviewed_override" or unknown_legacy_snapshot:
            result[field] = fallbacks[field]
            provenance[field] = "filename" if field == "title" else "missing"
            removed_reviewed_value = True

    # Categories, keywords, and the project tag have no automatic extraction
    # source, so the current reviewed lists are represented exactly even on
    # old generations.
    for field in ("categories", "keywords", "project"):
        result[field] = list(normalized_override.get(field, []))
        provenance[field] = (
            "reviewed_override" if field in normalized_override else "missing"
        )

    if not result.get("authors"):
        warnings.append("authors_missing")
    if provenance.get("title") == "filename":
        warnings.append("title_from_filename")
    if removed_reviewed_value:
        warnings.append("automatic_metadata_unavailable_after_override_removal")

    result["metadata_provenance"] = provenance
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    result["metadata_override_revision"] = override_revision
    return result


def _effective_documents(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    legacy_snapshot_mismatch = manifest.get(
        "metadata_storage_policy"
    ) != METADATA_STORAGE_POLICY and _metadata_snapshot_changed(manifest, metadata)
    project_id = str(manifest.get("project_id") or "")
    result: dict[str, dict[str, Any]] = {}
    for stored in manifest.get("documents", []):
        relative = str(stored.get("source_relative_path") or "")
        document = _effective_document_metadata(
            stored,
            metadata.get(relative, {}),
            unknown_legacy_snapshot_mismatch=legacy_snapshot_mismatch,
        )
        if not document.get("source_id") and project_id and relative:
            document["source_id"] = stable_source_id(project_id, relative)
        result[str(document["document_id"])] = document
    return result


def _document_for_chunk(
    chunk: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(chunk.get("document_id") or "")
    document = documents_by_id.get(document_id)
    if document is None:
        raise ResearchError(
            f"Current generation chunk references unknown document: {document_id}"
        )
    return document


def _locator_place(locator: Mapping[str, Any]) -> str:
    """Where a passage sits in the original, as the fragment a citation ends with.

    A reader who is told only this is reading a position, not a claim about the
    work, so the printed page label is carried only where it differs from the
    physical page and a section is named rather than numbered when it has a name.
    """

    if locator.get("type") == "pdf_page":
        return f"p. {locator.get('page_label') or locator.get('page')}"
    section = (
        locator.get("href_with_fragment")
        or locator.get("section_title")
        or locator.get("href")
    )
    if section:
        return f"section {section}"
    return f"EPUB section {locator.get('section_index')}"


def _citation(document: dict[str, Any], locator: dict[str, Any]) -> str:
    authors = document.get("authors") or []
    creator = "; ".join(str(item) for item in authors) if authors else ""
    title = str(document.get("title") or document.get("source_path") or "Source")
    year = document.get("year")
    lead = creator or title
    if creator and title:
        lead = f"{creator}, {title}"
    if year:
        lead = f"{lead} ({year})"
    doi = normalize_inline_text(str(document.get("doi") or ""))
    if doi:
        lead = f"{lead}, doi:{doi.removeprefix('doi:')}"

    return f"{lead}, {_locator_place(locator)}"


def _public_passage(
    chunk: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    public_document = _public_document(document)
    locator = dict(chunk.get("locator") or {})
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_id": public_document["source_id"],
        "source_path": public_document["source_path"],
        "title": public_document["title"],
        "authors": public_document["authors"],
        "year": public_document.get("year"),
        "doi": public_document["doi"],
        "language": public_document["language"],
        "categories": public_document["categories"],
        "keywords": public_document["keywords"],
        "project": public_document["project"],
        "locator": locator,
        "citation": normalize_inline_text(_citation(public_document, locator)),
        "text": normalize_reading_text(chunk_text(chunk)),
        "text_fidelity": "cleaned_semantic_text",
        "text_notes": text_script_notes(chunk_text(chunk)),
        "embedding_token_count": chunk.get("embedding_token_count"),
        "dense_truncated": chunk.get("dense_truncated"),
        "content_kind": str(chunk.get("content_kind") or "prose"),
        "annotations": list(chunk.get("annotations") or []),
        "quality_flags": list(chunk.get("quality_flags") or []),
        "metadata_provenance": dict(public_document.get("metadata_provenance") or {}),
        "metadata_warnings": list(public_document.get("metadata_warnings") or []),
    }
