"""The review surface: what a reader decides about each source and passage.

`ReviewWorkflow` holds the lock, resolves the caller's selector, and reads the
review files; the decisions themselves live in `review_state`, the question of
which sources a project knows about lives in `source_inventory`, and a refusal
about a project that cannot be served lives in `blocked_answers`. What moved out
of this module is named in each of those.

No source file is ever edited here, and no generation is ever rebuilt to record a
decision: a source exclusion is enforced by the retrieval filter at query time,
and the message beside every answer says whether an ingestion is recommended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .review_state import (
    chunk_placements,
    list_chunk_exclusions,
    set_chunk_inclusion,
    set_source_inclusion,
    set_source_metadata,
)
from .source_inventory import (
    exclusion_records,
    known_sources,
    resolve_source_selector,
    scanned_source_identity,
    sync_source_catalog,
)
from .sources import (
    SourcePolicyError,
    SourceScan,
    scan_sources,
)
from .storage import StorageError
from .support import (
    ResearchError,
    _effective_documents,
    _public_document,
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
        """Durably retain every issued opaque source ID and its project path.

        `ingestion` reaches this through the service, so the method stays here.
        """

        return sync_source_catalog(
            self.config,
            self._source_catalog(),
            scan,
            current,
            (self._source_exclusions() if exclusions is None else exclusions),
            (self._metadata() if metadata is None else metadata),
        )

    def _known_sources(
        self,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
        *,
        exclusions: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return known_sources(
            self.config,
            self._source_catalog(),
            scan,
            current,
            (self._source_exclusions() if exclusions is None else exclusions),
            (self._metadata() if metadata is None else metadata),
        )

    def _resolve_source_selector(
        self,
        *,
        source_id: str | None,
        source_path: str | None,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Resolve exactly one stable ID or source-relative compatibility path."""

        return resolve_source_selector(
            self.config,
            self._known_sources(scan, current),
            source_id=source_id,
            source_path=source_path,
        )

    def _exclusion_records(
        self,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        manifest: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return exclusion_records(self.config, scan, exclusions, manifest)

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
                return set_source_inclusion(
                    self.config,
                    selected,
                    self._source_exclusions(),
                    included=included,
                    reason=reason,
                )
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

    async def _chunk_placements(
        self,
        chunk_ids: Sequence[str],
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, dict[str, str]]:
        """Where each named chunk sits in the selected generation.

        The artifact lookup is this workflow's, because it opens the generation;
        the answer it returns is `review_state`'s.
        """

        async def load_chunks(
            generation_root: Path,
            manifest: dict[str, Any],
            ids: list[str],
        ) -> dict[str, Any]:
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            return await asyncio.to_thread(lookup.chunks_by_ids, ids)

        return await chunk_placements(
            chunk_ids,
            current,
            load_chunks,
            self._metadata,
        )

    async def set_chunk_inclusion(
        self,
        chunk_id: str,
        *,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """The decision `review_state.set_chunk_inclusion` composes, under the lock."""

        # The empty-ID refusal happens before the lock, because it is a statement
        # about the argument rather than about the project.
        if not (chunk_id or "").strip():
            raise ResearchError("A chunk decision needs a chunk_id")
        async with self._operation():
            return await set_chunk_inclusion(
                self.config,
                self._load_current_optional(),
                self._chunk_exclusions,
                self._chunk_placements,
                chunk_id=chunk_id,
                included=included,
                reason=reason,
            )

    async def list_chunk_exclusions(self) -> dict[str, Any]:
        """The decision list `review_state.list_chunk_exclusions` renders."""

        async with self._operation():
            current = self._load_current_optional()
            exclusions = self._chunk_exclusions()
            placements = await self._chunk_placements(list(exclusions), current)
            generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )
        return list_chunk_exclusions(exclusions, placements, generation_id)

    async def set_source_metadata(
        self,
        metadata: dict[str, Any],
        *,
        source_path: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """The answer `review_state.set_source_metadata` composes, under the lock."""

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
                return set_source_metadata(self.config, selected, metadata)
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

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
                    **scanned_source_identity(source),
                    "included": source.source_relative_path not in exclusions,
                    "indexed_in_current_generation": (
                        source.source_relative_path in indexed_paths
                    ),
                }
                for source in scan.selected
            ]
            known = self._known_sources(
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
                for relative, record in sorted(known.items())
            ]
            reviewed_metadata_sources = [
                {
                    "source_id": known[relative]["source_id"],
                    "source_relative_path": relative,
                    "source_path": known[relative]["source_path"],
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
