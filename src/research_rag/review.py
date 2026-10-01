from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import resolve_source_reference
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    SourceScan,
    normalize_metadata,
    scan_sources,
    stable_source_id,
)
from .storage import (
    StorageError,
    load_metadata_overrides,
    write_chunk_exclusions,
    write_metadata_overrides,
    write_source_catalog,
    write_source_exclusions,
)
from .support import (
    ResearchError,
    _document_for_chunk,
    _effective_documents,
    _locator_place,
    _public_document,
    _utc_now,
)

# How many sources one lookup returns by default, and the ceiling a caller can ask
# for. The answer is a lookup, not an inventory, so it stays small whatever the
# corpus holds.
DEFAULT_FIND_SOURCE_LIMIT = 10
FIND_SOURCE_MAX_MATCHES = 50


class ReviewWorkflow:
    def _sync_source_catalog(
        self,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
        *,
        exclusions: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Durably retain every issued opaque source ID and its project path."""

        catalog = self._source_catalog()
        updated = dict(catalog)
        paths_to_ids = {relative: source_id for source_id, relative in catalog.items()}

        def register(relative: str, recorded_source_id: str | None = None) -> None:
            try:
                expected_source_id = stable_source_id(
                    self.config.project_id,
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
        for relative in set(
            self._source_exclusions() if exclusions is None else exclusions
        ) | set(self._metadata() if metadata is None else metadata):
            register(relative)
        for source in scan.selected:
            register(source.source_relative_path, source.source_id)

        if updated != catalog:
            write_source_catalog(
                self.config.source_catalog_path,
                updated,
                project_id=self.config.project_id,
            )
        return updated

    def _known_sources(
        self,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
        *,
        exclusions: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        catalog = self._sync_source_catalog(
            scan,
            current,
            exclusions=exclusions,
            metadata=metadata,
        )
        manifest = current[1] if current is not None else {}
        source_directory = Path(
            str(
                manifest.get("source_directory")
                or self.config.source_root.relative_to(self.config.project_root)
            )
        )
        records: dict[str, dict[str, Any]] = {
            relative: {
                "source_id": source_id,
                "source_relative_path": relative,
                "source_path": (source_directory / relative).as_posix(),
                "format": Path(relative).suffix.removeprefix(".").casefold(),
                "exists": False,
                "indexed_in_current_generation": False,
            }
            for source_id, relative in catalog.items()
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
            records[relative] = {
                **item,
                "source_id": str(item.get("source_id") or "")
                or stable_source_id(self.config.project_id, relative),
                "source_relative_path": relative,
                "source_path": str(
                    item.get("source_path") or (source_directory / relative).as_posix()
                ),
                "exists": False,
                "indexed_in_current_generation": relative in indexed_paths,
            }
        # A persisted policy record stays addressable after its file is removed
        # and before it has ever appeared in a generation, so every source_id
        # returned by list_sources is usable by the mutation tools.
        policy_paths = set(
            self._source_exclusions() if exclusions is None else exclusions
        ) | set(self._metadata() if metadata is None else metadata)
        for relative in policy_paths:
            records.setdefault(
                relative,
                {
                    "source_id": stable_source_id(self.config.project_id, relative),
                    "source_relative_path": relative,
                    "source_path": (source_directory / relative).as_posix(),
                    "format": Path(relative).suffix.removeprefix(".").casefold(),
                    "exists": False,
                    "indexed_in_current_generation": False,
                },
            )
        for source in scan.selected:
            existing = records.get(source.source_relative_path, {})
            records[source.source_relative_path] = {
                **existing,
                "source_id": source.source_id,
                "source_relative_path": source.source_relative_path,
                "source_path": source.project_relative_path,
                "format": source.extension.removeprefix("."),
                "exists": True,
                "indexed_in_current_generation": (
                    source.source_relative_path in indexed_paths
                ),
            }
        return records

    def _resolve_source_selector(
        self,
        *,
        source_id: str | None,
        source_path: str | None,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Resolve exactly one stable ID or source-relative compatibility path."""

        if (source_id is None) == (source_path is None):
            raise SourcePolicyError("Provide exactly one of source_id or source_path")
        known = self._known_sources(scan, current)
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
        source = resolve_source_reference(self.config, source_path)
        relative = source.relative_to(self.config.source_root).as_posix()
        record = known.get(relative)
        if record is None:
            raise SourcePolicyError(
                f"Unknown PDF or EPUB source_relative_path: {source_path}"
            )
        return record

    def _exclusion_records(
        self,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        manifest: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scanned = {item.source_relative_path: item for item in scan.selected}
        indexed_paths = {
            str(document.get("source_relative_path"))
            for document in (manifest or {}).get("documents", [])
        }
        source_directory = Path(
            str(
                (manifest or {}).get("source_directory")
                or self.config.source_root.relative_to(self.config.project_root)
            )
        )
        records: list[dict[str, Any]] = []
        for relative, exclusion in exclusions.items():
            source = scanned.get(relative)
            records.append(
                {
                    "source_id": (
                        source.source_id
                        if source is not None
                        else stable_source_id(self.config.project_id, relative)
                    ),
                    "source_relative_path": relative,
                    "source_path": (
                        source.project_relative_path
                        if source is not None
                        else (source_directory / relative).as_posix()
                    ),
                    "reason": exclusion["reason"],
                    "excluded_at": exclusion["excluded_at"],
                    "exists": source is not None,
                    "indexed_in_current_generation": relative in indexed_paths,
                }
            )
        return records

    async def set_source_inclusion(
        self,
        source_path: str | None = None,
        *,
        source_id: str | None = None,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Include or exclude one source without changing the source file."""

        async with self._operation():
            try:
                scan = scan_sources(self.config)
                current = self._load_current_optional()
                selected = self._resolve_source_selector(
                    source_id=source_id,
                    source_path=source_path,
                    scan=scan,
                    current=current,
                )
                relative = str(selected["source_relative_path"])

                exclusions = self._source_exclusions()
                previous = exclusions.get(relative)
                if included:
                    changed = previous is not None
                    exclusions.pop(relative, None)
                else:
                    normalized_reason = (reason or "").strip()
                    if not normalized_reason:
                        raise SourcePolicyError(
                            "A non-empty reason is required when excluding a source"
                        )
                    if (
                        not selected["exists"]
                        and not selected["indexed_in_current_generation"]
                        and previous is None
                    ):
                        raise SourcePolicyError(
                            "Only an existing or currently indexed PDF or EPUB can "
                            "be excluded: "
                            f"{relative}"
                        )
                    changed = (
                        previous is None or previous.get("reason") != normalized_reason
                    )
                    if changed:
                        exclusions[relative] = {
                            "reason": normalized_reason,
                            "excluded_at": _utc_now(),
                        }

                if changed:
                    write_source_exclusions(
                        self.config.source_exclusions_path,
                        exclusions,
                    )

                indexed = bool(selected["indexed_in_current_generation"])
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

            effective_immediately = not included or indexed
            if not changed:
                message = (
                    "Source is already included."
                    if included
                    else "Source is already excluded with this reason."
                )
            elif included and not indexed:
                message = (
                    "Source inclusion saved. Run ingest before it can appear in "
                    "search because it is absent from the current generation."
                )
            elif included:
                message = (
                    "Source inclusion saved and restored for current retrieval. "
                    "Run ingest to record the change in a new generation."
                )
            else:
                message = (
                    "Source exclusion saved and enforced for current retrieval. "
                    "The original file was not changed. Run ingest to rebuild the "
                    "indexes without it."
                )
            return {
                "status": "changed" if changed else "unchanged",
                "source_id": selected["source_id"],
                "source_relative_path": relative,
                "source_path": selected["source_path"],
                "included": included,
                "reason": None if included else exclusions[relative]["reason"],
                "source_file_changed": False,
                "effective_immediately": effective_immediately,
                "generation_rebuild_recommended": changed,
                "message": message,
            }

    async def _chunk_placements(
        self,
        chunk_ids: Sequence[str],
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, dict[str, str]]:
        """Where each named chunk sits in the selected generation.

        A chunk the generation does not hold is absent from the answer rather
        than refused, because the decision is the reader's and stays on file
        either way, and because a caller has to be able to tell a decision that
        withholds nothing now from one that withholds a passage.
        """

        if current is None or not chunk_ids:
            return {}
        generation_root, manifest = current
        lookup = await self._ensure_artifact_lookup(generation_root, manifest)
        chunks = await asyncio.to_thread(lookup.chunks_by_ids, sorted(chunk_ids))
        documents_by_id = _effective_documents(manifest, self._metadata())
        return {
            str(chunk_id): {
                "source_relative_path": str(
                    _public_document(_document_for_chunk(chunk, documents_by_id)).get(
                        "source_relative_path"
                    )
                    or ""
                ),
                "locator": _locator_place(dict(chunk.get("locator") or {})),
            }
            for chunk_id, chunk in chunks.items()
        }

    async def set_chunk_inclusion(
        self,
        chunk_id: str,
        *,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Exclude one passage from retrieval, or restore it, without touching the file.

        A decision about a passage is a decision about evidence rather than about
        a document, so it is enforced by the retrieval filter at query time and
        never by a generation: a rebuild keeps the chunk in the indexes and the
        exclusion over it, which is why no call is recommended to close the gap.

        A `chunk_id` is derived from content, so unchanged text keeps it across a
        rebuild and anything else changes it. The generation a decision was
        recorded in travels with it, and a chunk this generation does not hold is
        reported as such rather than silently kept: an ingestion cannot bring a
        removed chunk back, so the answer says which kind of entry this is.
        """

        normalized_id = (chunk_id or "").strip()
        if not normalized_id:
            raise ResearchError("A chunk decision needs a chunk_id")
        async with self._operation():
            try:
                current = self._load_current_optional()
                if current is None:
                    raise SourcePolicyError(
                        "No knowledge base exists, so no chunk can be named; "
                        "call ingest first"
                    )
                generation_id = str(current[1]["generation_id"])
                exclusions = self._chunk_exclusions()
                previous = exclusions.get(normalized_id)
                if included:
                    changed = previous is not None
                    exclusions.pop(normalized_id, None)
                else:
                    normalized_reason = (reason or "").strip()
                    if not normalized_reason:
                        raise SourcePolicyError(
                            "A non-empty reason is required when excluding a chunk"
                        )
                    changed = (
                        previous is None
                        or previous.get("reason") != normalized_reason
                        or previous.get("generation_id") != generation_id
                    )
                    if changed:
                        exclusions[normalized_id] = {
                            "reason": normalized_reason,
                            "excluded_at": _utc_now(),
                            "generation_id": generation_id,
                        }
                if changed:
                    write_chunk_exclusions(
                        self.config.chunk_exclusions_path,
                        exclusions,
                    )
                saved_reason = None if included else exclusions[normalized_id]["reason"]
                placement = (
                    await self._chunk_placements([normalized_id], current)
                ).get(normalized_id)
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

            in_current_generation = placement is not None
            if not changed:
                message = (
                    "Chunk is already included."
                    if included
                    else "Chunk is already excluded with this reason."
                )
            elif included:
                message = (
                    "Chunk inclusion saved. The passage answers a search again at "
                    "once, and no ingestion is needed for that."
                )
            elif not in_current_generation:
                message = (
                    "Chunk exclusion saved, but this generation does not hold the "
                    "chunk, so nothing is withheld from the passages you can reach "
                    "now. The decision binds any later generation that holds it, and "
                    "no ingestion can bring the chunk back."
                )
            else:
                message = (
                    "Chunk exclusion saved and enforced for current retrieval. "
                    "The original file was not changed, and no ingestion is needed "
                    "to keep enforcing it."
                )
            return {
                "status": "changed" if changed else "unchanged",
                "chunk_id": normalized_id,
                "source_relative_path": (
                    placement["source_relative_path"] if placement else None
                ),
                "locator": placement["locator"] if placement else None,
                "included": included,
                "reason": saved_reason,
                "source_file_changed": False,
                # An exclusion is applied by the filter rather than by the
                # indexes, so it holds for the generation on screen and for every
                # one built after it without a rebuild.
                "effective_immediately": not included or in_current_generation,
                "generation_rebuild_recommended": False,
                "in_current_generation": in_current_generation,
                "message": message,
            }

    async def list_chunk_exclusions(self) -> dict[str, Any]:
        """The passages this project has taken out of retrieval, and why.

        It reviews decisions rather than the corpus: one row per recorded
        decision, each naming where its passage sat when the decision was made.
        A row the selected generation does not hold is reported as such, because
        that is the row a reader has to remove by hand and a rebuild will not
        remove for them.
        """

        async with self._operation():
            current = self._load_current_optional()
            exclusions = self._chunk_exclusions()
            placements = await self._chunk_placements(list(exclusions), current)
            generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )

        records = [
            {
                "chunk_id": chunk_id,
                "source_relative_path": placements.get(chunk_id, {}).get(
                    "source_relative_path", ""
                ),
                "locator": placements.get(chunk_id, {}).get("locator", ""),
                "reason": record["reason"],
                "excluded_at": record["excluded_at"],
                "in_current_generation": chunk_id in placements,
            }
            for chunk_id, record in exclusions.items()
        ]
        withheld = sum(1 for entry in records if entry["in_current_generation"])
        if not records:
            message = "No chunk is excluded from retrieval."
        elif withheld == 0:
            message = (
                f"None of the {len(records)} excluded chunks is withheld from "
                "current retrieval: each names a chunk this generation does not "
                "hold, and no ingestion restores a removed chunk."
            )
        elif withheld == len(records):
            message = (
                f"Every one of the {withheld} excluded chunks is withheld from "
                "current retrieval, and no original file was changed."
            )
        else:
            message = (
                f"{withheld} of the {len(records)} excluded chunks are withheld "
                f"from current retrieval; the other {len(records) - withheld} name "
                "a chunk this generation does not hold, and no ingestion restores "
                "a removed chunk."
            )
        return {
            "generation_id": generation_id,
            "excluded_chunk_count": len(records),
            "exclusions": records,
            "message": message,
        }

    async def set_source_metadata(
        self,
        metadata: dict[str, Any],
        *,
        source_path: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Save reviewed bibliographic metadata for one source.

        The review is authoritative at read time, so a saved change applies to the
        current generation without re-ingesting, and the same JSON file can be
        edited by hand between calls. Only the named source's entry is replaced,
        so every other entry survives. An empty review removes the entry, so
        automatic metadata applies again.
        """

        async with self._operation():
            try:
                scan = scan_sources(self.config)
                current = self._load_current_optional()
                selected = self._resolve_source_selector(
                    source_id=source_id,
                    source_path=source_path,
                    scan=scan,
                    current=current,
                )
                relative = str(selected["source_relative_path"])
                normalized = {
                    key: value
                    for key, value in normalize_metadata(dict(metadata)).items()
                    if value not in ("", [], None)
                }
                overrides = load_metadata_overrides(self.config.metadata_path)
                previous = overrides.get(relative)
                if normalized:
                    overrides[relative] = normalized
                else:
                    overrides.pop(relative, None)
                changed = previous != (normalized or None)
                if changed:
                    write_metadata_overrides(self.config.metadata_path, overrides)
                indexed = bool(selected["indexed_in_current_generation"])
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

            if not changed:
                message = "Reviewed metadata already matches what was saved."
            elif not normalized:
                message = (
                    "Reviewed metadata cleared for this source; automatic "
                    "metadata applies again."
                )
            elif indexed:
                message = (
                    "Reviewed metadata saved and applied to current retrieval. "
                    "The next ingestion records it in a new generation."
                )
            else:
                message = (
                    "Reviewed metadata saved. Run ingest before it can appear in "
                    "search because the source is absent from the current "
                    "generation."
                )
            return {
                "status": "changed" if changed else "unchanged",
                "source_id": selected["source_id"],
                "source_relative_path": relative,
                "source_path": selected["source_path"],
                "metadata": normalized,
                "source_file_changed": False,
                "effective_immediately": bool(normalized) and indexed,
                "generation_rebuild_recommended": False,
                "message": message,
            }

    async def find_source(
        self,
        query: str,
        limit: int = DEFAULT_FIND_SOURCE_LIMIT,
    ) -> dict[str, Any]:
        """Return the sources one name resolves to, and whether each is searchable.

        The caller asks about a filename, a title, or an author, and the answer is
        the handful of sources that resolve it, each with the stable ID and the
        source-relative path the inclusion and metadata operations take. Nothing
        here lists the corpus.

        The lookup covers every source this project can name: the files in the
        source directory, the sources of the selected generation, and the sources
        a review record kept addressable after their file was removed. A source
        absent from the corpus is the one a reader asks about, so it is answered
        rather than hidden.
        """

        term = query.strip()
        if not term:
            raise ResearchError("Provide a filename, title, or author to look up.")
        bounded = max(1, min(int(limit), FIND_SOURCE_MAX_MATCHES))
        needle = term.casefold()
        async with self._operation():
            current = self._load_current_optional()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            exclusions = self._source_exclusions()
            metadata = self._metadata()
            known = self._known_sources(
                scan,
                current,
                exclusions=exclusions,
                metadata=metadata,
            )
            manifest = current[1] if current is not None else {}
            documents = {
                str(document.get("source_relative_path") or ""): _public_document(
                    document
                )
                for document in (
                    _effective_documents(manifest, metadata).values()
                    if current is not None
                    else []
                )
            }
            matches: list[dict[str, Any]] = []
            for relative, record in known.items():
                override = metadata.get(relative, {})
                document = documents.get(relative, {})
                title = str(
                    document.get("title")
                    or override.get("title")
                    or Path(relative).stem
                )
                authors = [
                    str(author)
                    for author in (
                        document.get("authors") or override.get("authors") or []
                    )
                ]
                indexed = bool(record.get("indexed_in_current_generation"))
                if not any(
                    needle in value.casefold()
                    for value in (relative, title, *authors)
                    if value
                ):
                    continue
                matches.append(
                    {
                        "source_id": record["source_id"],
                        "source_relative_path": relative,
                        "title": title,
                        "authors": authors,
                        "exists": bool(record.get("exists")),
                        "included": relative not in exclusions,
                        "indexed_in_current_generation": indexed,
                        "has_reviewed_metadata": relative in metadata,
                        "searchable": indexed and relative not in exclusions,
                    }
                )
            matches.sort(
                key=lambda item: (
                    not item["searchable"],
                    not item["indexed_in_current_generation"],
                    item["source_relative_path"],
                )
            )
            shown = matches[:bounded]
            total = len(matches)
            searchable = sum(1 for item in matches if item["searchable"])
            if total == 0:
                message = (
                    f"No source matches {term!r}. The lookup covers the source "
                    "directory, the selected generation, and every source a review "
                    "record kept addressable."
                )
            elif searchable == 0:
                message = (
                    f"{total} sources match {term!r} and none is searchable: each "
                    "needs ingesting, or a reviewed exclusion is in force."
                )
            else:
                message = (
                    f"Every one of the {total} sources matching {term!r} is "
                    "searchable now."
                    if searchable == total
                    else f"{searchable} of the {total} sources matching {term!r} "
                    "are searchable now."
                )
            return {
                "ready": current is not None,
                "generation_id": (
                    manifest.get("generation_id") if current is not None else None
                ),
                "query": term,
                "match_count": total,
                "searchable_match_count": searchable,
                "truncated": total > len(shown),
                "matches": shown,
                "message": message,
            }

    async def list_sources(self) -> dict[str, Any]:
        async with self._operation():
            current = self._load_current_optional()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            exclusions = self._source_exclusions()
            metadata = self._metadata()
            indexed_paths = {
                str(document.get("source_relative_path") or "")
                for document in (current[1].get("documents", []) if current else [])
                if isinstance(document, dict)
            }
            discovered_sources = [
                {
                    "source_id": source.source_id,
                    "source_relative_path": source.source_relative_path,
                    "source_path": source.project_relative_path,
                    "format": source.extension.removeprefix("."),
                    "included": source.source_relative_path not in exclusions,
                    "indexed_in_current_generation": (
                        source.source_relative_path in indexed_paths
                    ),
                }
                for source in scan.selected
            ]
            known_sources = self._known_sources(
                scan,
                current,
                exclusions=exclusions,
                metadata=metadata,
            )
            known_source_records = [
                {
                    "source_id": record["source_id"],
                    "source_relative_path": relative,
                    "exists": bool(record["exists"]),
                    "included": relative not in exclusions,
                    "indexed_in_current_generation": bool(
                        record["indexed_in_current_generation"]
                    ),
                    "has_reviewed_metadata": relative in metadata,
                }
                for relative, record in sorted(known_sources.items())
            ]
            reviewed_metadata_sources = [
                {
                    "source_id": known_sources[relative]["source_id"],
                    "source_relative_path": relative,
                    "source_path": known_sources[relative]["source_path"],
                    "metadata": override,
                    "indexed_in_current_generation": relative in indexed_paths,
                }
                for relative, override in sorted(metadata.items())
            ]
            if current is None:
                return {
                    "ready": False,
                    "source_count": 0,
                    "sources": [],
                    "discovered_source_count": len(discovered_sources),
                    "discovered_sources": discovered_sources,
                    "known_source_count": len(known_source_records),
                    "known_sources": known_source_records,
                    "excluded_source_count": len(exclusions),
                    "excluded_sources": self._exclusion_records(scan, exclusions),
                    "reviewed_metadata_source_count": len(reviewed_metadata_sources),
                    "reviewed_metadata_sources": reviewed_metadata_sources,
                }
            _generation_root, manifest = current
            documents_by_id = _effective_documents(manifest, metadata)
            sources = [
                _public_document(document)
                for document in documents_by_id.values()
                if document.get("source_relative_path") not in exclusions
            ]
            return {
                "ready": True,
                "generation_id": manifest["generation_id"],
                "source_count": len(sources),
                "sources": sources,
                "discovered_source_count": len(discovered_sources),
                "discovered_sources": discovered_sources,
                "known_source_count": len(known_source_records),
                "known_sources": known_source_records,
                "excluded_source_count": len(exclusions),
                "excluded_sources": self._exclusion_records(
                    scan,
                    exclusions,
                    manifest,
                ),
                "reviewed_metadata_source_count": len(reviewed_metadata_sources),
                "reviewed_metadata_sources": reviewed_metadata_sources,
            }
