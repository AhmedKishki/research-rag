"""What sources this project knows about, and how it answers for each one.

This module owns the question, not the request: it reads a scan, the selected
generation, and the review files, and returns the records a caller selects a
source from. It may never open a generation, answer a search, or decide whether
a project can be served, and it never imports a surface. It writes exactly one
file, `source-catalog.json`, because an issued source ID outlives the file it
named and a reader must still be able to address it.

What moved here from `review.py`: `_sync_source_catalog`, `_known_sources`,
`_resolve_source_selector`, and `_exclusion_records`, together with the record
builder that four sites duplicated and the `source_directory` fallback that two
sites duplicated. What moved here from `status.py`: nothing, and `status.py`
reads `exclusion_records` from here rather than reaching into `ReviewWorkflow`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..corpus.sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourceFile,
    SourcePolicyError,
    SourceScan,
    stable_source_id,
)
from ..project.config import ResearchConfig, resolve_source_reference
from ..project.support import ResearchError, _document_matches_metadata
from ..storage.records import write_source_catalog


def source_directory(config: ResearchConfig, manifest: dict[str, Any]) -> Path:
    """The directory a manifest's source-relative paths are resolved against.

    A generation may have been built from a source directory that is not the one
    configured now, so its own record wins and the configured directory is the
    fallback for a manifest that records none.
    """

    return Path(
        str(
            manifest.get("source_directory")
            or config.source_root.relative_to(config.project_root)
        )
    )


def unrecorded_source_record(
    config: ResearchConfig,
    directory: Path,
    source_id: str,
    relative: str,
) -> dict[str, Any]:
    """The record for a source no generation has ever indexed.

    The two sites that built this shape, one for a catalogued source and one for
    a path a review record kept addressable, agree on every key and every value,
    so they share this builder.
    """

    return {
        "source_id": source_id,
        "source_relative_path": relative,
        "source_path": (directory / relative).as_posix(),
        "format": Path(relative).suffix.removeprefix(".").casefold(),
        "exists": False,
        "indexed_in_current_generation": False,
    }


def scanned_source_identity(source: SourceFile) -> dict[str, Any]:
    """The four keys a scanned file answers with, whatever record it lands in.

    Two sites need them and they do not agree on the rest of the record: the
    inventory says whether the file exists, and `list_sources` says whether it is
    included. Only these four are the same, so only these four are shared.
    """

    return {
        "source_id": source.source_id,
        "source_relative_path": source.source_relative_path,
        "source_path": source.project_relative_path,
        "format": source.extension.removeprefix("."),
    }


def stored_source_record(
    config: ResearchConfig,
    directory: Path,
    item: dict[str, Any],
    relative: str,
    *,
    indexed_paths: set[str],
) -> dict[str, Any]:
    """The record for a source the selected generation describes.

    It differs from `unrecorded_source_record` in two ways that are kept
    deliberately: it carries every field the manifest recorded for the source,
    and it carries no `format`, because the manifest's own fields own that name.
    """

    return {
        **item,
        "source_id": str(item.get("source_id") or "")
        or stable_source_id(config.project_id, relative),
        "source_relative_path": relative,
        "source_path": str(
            item.get("source_path") or (directory / relative).as_posix()
        ),
        "exists": False,
        "indexed_in_current_generation": relative in indexed_paths,
    }


def sync_source_catalog(
    config: ResearchConfig,
    catalog: dict[str, str],
    scan: SourceScan,
    current: tuple[Path, dict[str, Any]] | None,
    exclusions: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Durably retain every issued opaque source ID and its project path.

    The write is what a reader is allowed to do: the catalog is the only memory a
    project has of a source it has seen and no longer holds, so a file that is
    renamed or deleted between two builds stays addressable by the ID it was given
    while it was there. Without it, the second listing would have forgotten it.
    """

    updated = dict(catalog)
    paths_to_ids = {relative: source_id for source_id, relative in catalog.items()}

    def register(relative: str, recorded_source_id: str | None = None) -> None:
        try:
            expected_source_id = stable_source_id(
                config.project_id,
                relative,
            )
        except SourcePolicyError as exc:
            raise ResearchError(str(exc)) from exc
        if Path(relative).suffix.casefold() not in ALLOWED_SOURCE_EXTENSIONS:
            raise ResearchError(
                f"Source catalog contains an unsupported source path: {relative}"
            )
        if recorded_source_id and recorded_source_id != expected_source_id:
            raise ResearchError(
                f"Source ID does not match its project path: {relative}"
            )
        existing_path = updated.get(expected_source_id)
        existing_id = paths_to_ids.get(relative)
        if existing_path not in {None, relative} or existing_id not in {
            None,
            expected_source_id,
        }:
            raise ResearchError(
                f"Conflicting source identity in project catalog: {relative}"
            )
        updated[expected_source_id] = relative
        paths_to_ids[relative] = expected_source_id

    manifest = current[1] if current is not None else {}
    stored_records = manifest.get("source_files")
    if not isinstance(stored_records, list):
        stored_records = manifest.get("documents", [])
    for item in stored_records:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("source_relative_path") or "")
        if relative:
            register(relative, str(item.get("source_id") or "") or None)
    for relative in set(exclusions) | set(metadata):
        register(relative)
    for source in scan.selected:
        register(source.source_relative_path, source.source_id)

    if updated != catalog:
        write_source_catalog(
            config.source_catalog_path,
            updated,
            project_id=config.project_id,
        )
    return updated


def source_matches_filters(
    record: Mapping[str, Any],
    *,
    categories: set[str],
    categories_any: set[str],
    projects: set[str],
    projects_any: set[str],
    keywords: set[str],
) -> bool:
    """Whether one source record passes the listing's filter layers.

    The same predicate a search applies to a document, so a reader who narrows the
    listing and then searches the result is filtering by one rule rather than two.
    A record that carries no metadata fails every layer that names a value, which
    is what a file that has not been extracted yet does.
    """

    return _document_matches_metadata(
        record,
        keywords=keywords,
        categories=categories,
        categories_any=categories_any,
        projects=projects,
        projects_any=projects_any,
    )


def known_sources(
    config: ResearchConfig,
    catalog: dict[str, str],
    scan: SourceScan,
    current: tuple[Path, dict[str, Any]] | None,
    exclusions: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Every source this project can name, keyed by its source-relative path.

    Reading this inventory is allowed to rewrite `source-catalog.json` when the
    catalog is stale, and that is the price of remembering a source this project has
    seen and no longer holds.
    """

    synced = sync_source_catalog(
        config,
        catalog,
        scan,
        current,
        exclusions,
        metadata,
    )
    manifest = current[1] if current is not None else {}
    directory = source_directory(config, manifest)
    records: dict[str, dict[str, Any]] = {
        relative: unrecorded_source_record(
            config,
            directory,
            source_id,
            relative,
        )
        for source_id, relative in synced.items()
    }
    indexed_paths = {
        str(item.get("source_relative_path") or "")
        for item in manifest.get("documents", [])
        if isinstance(item, dict)
    }
    stored_records = manifest.get("source_files")
    if not isinstance(stored_records, list):
        stored_records = manifest.get("documents", [])
    for item in stored_records:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("source_relative_path") or "")
        if not relative:
            continue
        records[relative] = stored_source_record(
            config,
            directory,
            item,
            relative,
            indexed_paths=indexed_paths,
        )
    # A persisted policy record stays addressable after its file is removed
    # and before it has ever appeared in a generation, so every source_id
    # returned by list_sources is usable by the mutation tools.
    policy_paths = set(exclusions) | set(metadata)
    for relative in policy_paths:
        records.setdefault(
            relative,
            unrecorded_source_record(
                config,
                directory,
                stable_source_id(config.project_id, relative),
                relative,
            ),
        )
    for source in scan.selected:
        records[source.source_relative_path] = {
            **records.get(source.source_relative_path, {}),
            **scanned_source_identity(source),
            "exists": True,
            "indexed_in_current_generation": (
                source.source_relative_path in indexed_paths
            ),
        }
    return records


def resolve_source_selector(
    config: ResearchConfig,
    known: dict[str, dict[str, Any]],
    *,
    source_id: str | None,
    source_path: str | None,
) -> dict[str, Any]:
    """Resolve exactly one stable ID or source-relative compatibility path."""

    if (source_id is None) == (source_path is None):
        raise SourcePolicyError("Provide exactly one of source_id or source_path")
    if source_id is not None:
        normalized_id = source_id.strip()
        if not normalized_id:
            raise SourcePolicyError("source_id must not be empty")
        matches = [
            record
            for record in known.values()
            if record.get("source_id") == normalized_id
        ]
        if not matches:
            raise SourcePolicyError(f"Unknown source_id: {normalized_id}")
        if len(matches) != 1:  # Defensive: deterministic IDs must be unique.
            raise SourcePolicyError(f"Ambiguous source_id: {normalized_id}")
        return matches[0]

    assert source_path is not None
    source = resolve_source_reference(config, source_path)
    relative = source.relative_to(config.source_root).as_posix()
    record = known.get(relative)
    if record is None:
        raise SourcePolicyError(
            f"Unknown PDF or EPUB source_relative_path: {source_path}"
        )
    return record


def exclusion_records(
    config: ResearchConfig,
    scan: SourceScan,
    exclusions: dict[str, dict[str, str]],
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per excluded source, with why and whether it is still withheld."""

    scanned = {item.source_relative_path: item for item in scan.selected}
    indexed_paths = {
        str(document.get("source_relative_path"))
        for document in (manifest or {}).get("documents", [])
    }
    directory = source_directory(config, manifest or {})
    records: list[dict[str, Any]] = []
    for relative, exclusion in exclusions.items():
        source = scanned.get(relative)
        records.append(
            {
                "source_id": (
                    source.source_id
                    if source is not None
                    else stable_source_id(config.project_id, relative)
                ),
                "source_relative_path": relative,
                "source_path": (
                    source.project_relative_path
                    if source is not None
                    else (directory / relative).as_posix()
                ),
                "reason": exclusion["reason"],
                "excluded_at": exclusion["excluded_at"],
                "exists": source is not None,
                "indexed_in_current_generation": relative in indexed_paths,
            }
        )
    return records
