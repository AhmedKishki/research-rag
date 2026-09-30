"""Readiness, staleness, upgrade reasons, and the generation inventory."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .config import ResearchConfig
from .generation import value_fingerprint
from .health import health_report
from .launcher import ui_launcher_state
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    scan_sources,
    sha256_file,
)
from .storage import StorageError, directory_statistics, read_json
from .support import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    DEFAULT_RETRIEVAL_METHOD,
    EXTRACTION_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
    ResearchError,
    _effective_documents,
    _metadata_inventory,
    _metadata_snapshot_changed,
)
from .version import version_block


def generation_upgrade_reasons(
    config: ResearchConfig,
    manifest: dict[str, Any],
    policy_fingerprint: str,
) -> list[str]:
    """Return the policy mismatches a generation has, without touching disk.

    This is the part of a status report that depends only on the manifest and the
    current policy constants, so a caller that skipped the staleness check can
    still report whether the generation needs an upgrade. The fingerprint is the
    caller's: a process computes it once from the settings it resolved.
    """

    retrieval = manifest.get("retrieval", {})
    dense_policy = retrieval.get("dense", {})
    fusion_policy = retrieval.get("fusion", {})
    relevance_policy = retrieval.get("relevance_gates", {})
    reasons: list[str] = []
    if int(manifest.get("schema_version") or 0) != SCHEMA_VERSION:
        reasons.append("generation_schema")
    if int(manifest.get("extraction_policy_version") or 0) != EXTRACTION_POLICY_VERSION:
        reasons.append("layout_extraction")
    if int(manifest.get("cleaning_policy_version") or 0) != CLEANING_POLICY_VERSION:
        reasons.append("semantic_cleaning")
    if int(manifest.get("artifact_policy_version") or 0) != ARTIFACT_POLICY_VERSION:
        reasons.append("generation_artifacts")
    if manifest.get("metadata_storage_policy") != METADATA_STORAGE_POLICY:
        reasons.append("metadata_storage")
    if manifest.get("project_id") != config.project_id:
        reasons.append("project_identity")
    if (
        dense_policy.get("embedding_model") != config.settings.embedding_model
        or dense_policy.get("embedding_model_revision")
        != config.settings.embedding_model_revision
        or dense_policy.get("embedding_dimension")
        != config.settings.embedding_dimension
    ):
        reasons.append("embedding_model")
    if (
        manifest.get("retrieval_policy_fingerprint") != policy_fingerprint
        or fusion_policy.get("method") != "weighted_reciprocal_rank_fusion"
        or fusion_policy.get("rrf_k") != config.settings.rrf_k
        or fusion_policy.get("bm25_weight") != config.settings.bm25_weight
        or fusion_policy.get("dense_weight") != config.settings.dense_weight
        or relevance_policy.get("dense_minimum_cosine_similarity")
        != config.settings.dense_minimum_cosine_similarity
        or relevance_policy.get("bm25_requires_query_token_overlap") is not True
    ):
        reasons.append("retrieval_policy")
    return reasons


def generation_inventory(
    config: ResearchConfig,
    current_generation_id: str | None,
) -> dict[str, Any]:
    """Describe every retained generation on disk without validating it.

    These are exactly the directories a prune would consider, so `status`
    reports them with their size and file count. A generation whose manifest is
    missing, unreadable, or not JSON is reported with a ``manifest_error``
    instead of raising: the purpose is transparency about what occupies disk,
    and a status call must still answer when one retained generation is
    damaged. This runs only on the read-only ``status`` surface, never on the
    search path, because it walks each generation's files.
    """

    root = config.generations_root
    records: list[dict[str, Any]] = []
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            record: dict[str, Any] = {
                "generation_id": entry.name,
                "is_current": entry.name == current_generation_id,
            }
            try:
                manifest = read_json(entry / "manifest.json")
            except StorageError as exc:
                record["manifest_error"] = str(exc)
            else:
                if isinstance(manifest, dict):
                    record.update(
                        created_at=manifest.get("created_at"),
                        chunk_count=manifest.get("chunk_count"),
                        document_count=manifest.get("document_count"),
                        schema_version=manifest.get("schema_version"),
                    )
                else:
                    record["manifest_error"] = "manifest.json is not a JSON object"
            file_count, size_bytes = directory_statistics(entry)
            record["file_count"] = file_count
            record["size_bytes"] = size_bytes
            records.append(record)
    records.sort(
        key=lambda item: (str(item.get("created_at") or ""), item["generation_id"]),
        reverse=True,
    )
    return {
        "generations": records,
        "retained_generation_count": len(records),
        "retained_generation_bytes": sum(int(item["size_bytes"]) for item in records),
    }


class StatusWorkflow:
    def _generation_upgrade_reasons(self, manifest: dict[str, Any]) -> list[str]:
        return generation_upgrade_reasons(
            self.config,
            manifest,
            self.retrieval_policy_fingerprint,
        )

    def _status(
        self,
        current: tuple[Path, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Summarize the project, reusing a caller already-parsed pointer.

        `search` already holds the selected generation and its manifest, so it
        passes them in rather than making this read and parse them again.
        """

        try:
            scan = scan_sources(self.config)
        except SourcePolicyError as exc:
            raise ResearchError(str(exc)) from exc
        exclusions = self._source_exclusions()
        exclusion_revision = value_fingerprint(exclusions)
        metadata = self._metadata()
        metadata_revision = value_fingerprint(metadata)
        selected = tuple(
            source
            for source in scan.selected
            if source.source_relative_path not in exclusions
        )
        checkpoint_state = self._load_ingestion_checkpoint()
        ingestion_progress = (
            self._ingestion_progress(checkpoint_state[1])
            if checkpoint_state is not None
            else None
        )
        if current is None:
            current = self._load_current_optional()
        if current is None:
            if ingestion_progress is not None:
                message = (
                    "An ingestion checkpoint is available; call ingest again with "
                    "the same settings to continue it."
                )
            elif scan.selected and not selected:
                message = (
                    "No knowledge-base generation exists and all discovered sources "
                    "are excluded; include at least one source before ingesting."
                )
            else:
                message = "No knowledge-base generation exists; call ingest."
            return {
                "ready": False,
                "stale": bool(selected),
                "project_root": str(self.config.project_root),
                "project_id": self.config.project_id,
                "project_name": self.config.project_name,
                "source_root": str(self.config.source_root),
                "state_root": str(self.config.state_root),
                "runtime_root": (
                    str(self.config.runtime_root)
                    if self.config.runtime_root is not None
                    else None
                ),
                "portable_root": str(self.config.portable_root),
                "model_cache_root": str(self.config.model_cache_root),
                "version": version_block(),
                "ui_launcher": ui_launcher_state(
                    self.config.project_root,
                    self.config.portable_root,
                ),
                "discovered_source_count": len(scan.selected),
                "selected_source_count": len(selected),
                "excluded_source_count": len(exclusions),
                "categories": [],
                "projects": [],
                "excluded_sources": self._exclusion_records(scan, exclusions),
                "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                "ignored_extensions": scan.ignored_extensions,
                "language": {
                    "corpus": self.config.settings.language_corpus,
                    "languages": list(self.config.settings.corpus_languages),
                    "bm25_stopwords": self.config.settings.bm25_stopwords_language,
                    "warning": self.config.settings.embedding_language_warning,
                },
                "source_exclusion_revision": exclusion_revision,
                "metadata_revision": metadata_revision,
                "generation_metadata_revision": None,
                "metadata_overlay_active": False,
                "metadata_pending_source_paths": sorted(metadata),
                "generation_metadata_snapshot_outdated": False,
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": [],
                "generation_upgrade_required": False,
                "upgrade_reasons": [],
                "last_build_metrics": None,
                "ingestion_progress": ingestion_progress,
                "message": message,
            }

        generation_root, manifest = current
        manifest_source_files = manifest.get("source_files")
        if not isinstance(manifest_source_files, list):
            manifest_source_files = manifest.get("documents", [])
        current_documents = {
            str(item["source_relative_path"]): item for item in manifest_source_files
        }
        scanned = {item.source_relative_path: item for item in scan.selected}
        added = sorted(set(scanned) - set(current_documents))
        removed = sorted(set(current_documents) - set(scanned))
        modified: list[str] = []
        for relative in sorted(set(scanned) & set(current_documents)):
            source = scanned[relative]
            existing = current_documents[relative]
            if source.size != existing.get("size"):
                modified.append(relative)
                continue
            if source.mtime_ns != existing.get("mtime_ns"):
                recorded_digest = str(existing.get("sha256") or "")
                if not recorded_digest or sha256_file(source.path) != recorded_digest:
                    modified.append(relative)
        metadata_changed = _metadata_snapshot_changed(manifest, metadata)
        stored_exclusion_revision = manifest.get("source_exclusion_revision")
        source_exclusions_changed = (
            stored_exclusion_revision != exclusion_revision
            if stored_exclusion_revision is not None
            else bool(exclusions)
        )
        stale = bool(added or removed or modified or source_exclusions_changed)
        retrieval = manifest.get("retrieval", {})
        available_methods = retrieval.get("available_methods") or ["bm25"]
        hybrid_ready = "hybrid" in available_methods
        upgrade_reasons = self._generation_upgrade_reasons(manifest)
        indexed_source_paths = {
            str(document.get("source_relative_path") or "")
            for document in manifest.get("documents", [])
        }
        metadata_overlay_active = bool(set(metadata) & indexed_source_paths)
        metadata_pending_source_paths = sorted(set(metadata) - indexed_source_paths)
        excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
        exclusion_records = self._exclusion_records(scan, exclusions, manifest)
        if ingestion_progress is not None:
            status_message = (
                "A new generation is in progress; the selected generation remains "
                "searchable. Call ingest again with the same settings to continue."
            )
        elif source_exclusions_changed:
            status_message = (
                "Source exclusions are already enforced by retrieval; run ingest "
                "to rebuild the stored indexes without excluded sources."
            )
        elif upgrade_reasons:
            status_message = (
                "The current generation uses an older extraction or storage schema; "
                "regenerate it with ingest."
            )
        elif added or removed or modified:
            status_message = (
                "Source files differ from the selected generation; call ingest to "
                "index the current source set."
            )
        elif metadata_pending_source_paths:
            status_message = (
                "Reviewed metadata is saved for sources outside the selected "
                "generation; include those sources if needed and ingest to index them."
            )
        elif metadata_changed:
            status_message = (
                "Reviewed metadata differs from the immutable generation snapshot "
                "and is being applied immediately at read time; ingestion is not "
                "required."
            )
        elif hybrid_ready:
            status_message = "Current generation supports hybrid retrieval."
        else:
            status_message = (
                "Current generation is BM25-only; run ingest to build its "
                "project-local dense index."
            )
        effective_documents = _effective_documents(manifest, metadata)
        return {
            "ready": True,
            "stale": stale,
            "project_root": str(self.config.project_root),
            "project_id": self.config.project_id,
            "project_name": self.config.project_name,
            "source_root": str(self.config.source_root),
            "state_root": str(self.config.state_root),
            "runtime_root": (
                str(self.config.runtime_root)
                if self.config.runtime_root is not None
                else None
            ),
            "portable_root": str(self.config.portable_root),
            "model_cache_root": str(self.config.model_cache_root),
            "version": version_block(),
            "ui_launcher": ui_launcher_state(
                self.config.project_root,
                self.config.portable_root,
            ),
            "generation_id": manifest["generation_id"],
            "created_at": manifest["created_at"],
            "discovered_source_count": len(scan.selected),
            "selected_source_count": len(selected),
            "indexed_source_count": manifest["document_count"],
            "searchable_source_count": sum(
                str(document["document_id"]) not in excluded_document_ids
                for document in manifest.get("documents", [])
            ),
            "excluded_source_count": len(exclusions),
            "excluded_sources": exclusion_records,
            "chunk_count": manifest["chunk_count"],
            "categories": _metadata_inventory(
                effective_documents,
                excluded_document_ids,
                field="categories",
                label="category",
            ),
            "projects": _metadata_inventory(
                effective_documents,
                excluded_document_ids,
                field="project",
                label="project",
            ),
            "languages": _metadata_inventory(
                effective_documents,
                excluded_document_ids,
                field="language",
                label="language",
            ),
            "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
            "ignored_extensions": scan.ignored_extensions,
            "language": {
                "corpus": self.config.settings.language_corpus,
                "languages": list(self.config.settings.corpus_languages),
                "bm25_stopwords": self.config.settings.bm25_stopwords_language,
                "warning": self.config.settings.embedding_language_warning,
            },
            "default_retrieval_method": (
                retrieval.get("default_method", "bm25") if hybrid_ready else "bm25"
            ),
            "available_retrieval_methods": available_methods,
            "hybrid_ready": hybrid_ready,
            "hybrid_upgrade_required": not hybrid_ready,
            "generation_upgrade_required": bool(upgrade_reasons),
            "upgrade_reasons": upgrade_reasons,
            "retrieval": retrieval,
            "last_build_metrics": manifest.get("build_metrics"),
            "ingestion_progress": ingestion_progress,
            "source_exclusion_revision": exclusion_revision,
            "metadata_revision": metadata_revision,
            "generation_metadata_revision": manifest.get("metadata_revision"),
            "metadata_overlay_active": metadata_overlay_active,
            "metadata_pending_source_paths": metadata_pending_source_paths,
            "generation_metadata_snapshot_outdated": metadata_changed,
            "changes": {
                "added": added,
                "removed": removed,
                "modified": modified,
                "metadata_changed": metadata_changed,
                "source_exclusions_changed": source_exclusions_changed,
            },
            "generation_root": str(generation_root),
            "message": status_message,
        }

    def _generation_inventory(
        self, current_generation_id: str | None
    ) -> dict[str, Any]:
        return generation_inventory(self.config, current_generation_id)

    async def status(self) -> dict[str, Any]:
        async with self._operation():
            payload = await asyncio.to_thread(self._status)
            payload.update(
                await asyncio.to_thread(
                    self._generation_inventory,
                    payload.get("generation_id"),
                )
            )
            # The dependency report is built from the payload this method already
            # produced, so the health answer and the status answer cannot be taken
            # from two different states of the project.
            report = await asyncio.to_thread(health_report, self.config, payload)
            payload.update(report.as_status_fields())
            return payload
