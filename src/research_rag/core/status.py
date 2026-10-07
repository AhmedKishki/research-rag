"""The one status answer the command line, the workspace, and an agent share.

`StatusWorkflow` decides whether a project can be served and what its answer is.
The answers a blocked project gets live in `blocked_answers`, the walk over
retained generations lives in `generation_inventory`, and the records a status
answer lists for excluded sources are built by `source_inventory.exclusion_records`
rather than by reaching into `ReviewWorkflow`. What moved out of this module is
named in each of those.

The payload is built from the groups below rather than from two literals: the
project identity, the source counts, the corpus vocabulary, the retrieval
methods, and the review state are each written once, and each branch states the
order it emits those groups in. Three keys cannot be grouped because the two
branches interleave them differently: `excluded_source_count`,
`excluded_chunk_count`, and `excluded_sources`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..corpus.sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    SourceScan,
    scan_sources,
    sha256_file,
)
from ..generations.generation_inventory import generation_inventory
from ..project.config import ResearchConfig
from ..project.policy import (
    DEFAULT_RETRIEVAL_METHOD,
    ResearchError,
    value_fingerprint,
)
from ..project.support import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
    _effective_documents,
    _metadata_inventory,
    _metadata_snapshot_changed,
)
from ..runtime.health import health_report
from ..runtime.version import version_block
from .source_inventory import exclusion_records


def generation_upgrade_reasons(
    config: ResearchConfig,
    manifest: dict[str, Any],
    policy_fingerprint: str,
) -> list[str]:
    """Return the policy mismatches a generation has, without touching disk.

    Only the manifest and the current policy constants decide the answer, so a
    caller that skipped the staleness check can still report whether the
    generation needs an upgrade. The fingerprint is the caller's: a process
    computes it once from the settings it resolved.
    """

    retrieval = manifest.get("retrieval", {})
    dense_policy = retrieval.get("dense")
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
    # Legacy BM25-only generations have no embedding identity. A generation that
    # advertises dense retrieval must record a matching identity before queries
    # can be embedded against its vectors, including when that block is missing.
    has_dense = bool(
        {"dense", "hybrid"} & set(retrieval.get("available_methods") or [])
    )
    if (dense_policy or has_dense) and (
        not isinstance(dense_policy, dict)
        or not config.settings.embedding_facts.matches_dense_metadata(dense_policy)
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


# Each builder below is one group of the payload, held together because both
# branches emit it in one piece. A branch composes them in the order it emits the
# keys; the values are the branch's own.


def _base_payload(
    config: ResearchConfig, *, ready: bool, stale: bool
) -> dict[str, Any]:
    """Which project this is, and what this machine makes of it."""

    return {
        "ready": ready,
        "stale": stale,
        "project_root": str(config.project_root),
        "project_id": config.project_id,
        "project_name": config.project_name,
        "source_root": str(config.source_root),
        "state_root": str(config.state_root),
        "runtime_root": (
            str(config.runtime_root) if config.runtime_root is not None else None
        ),
        "portable_root": str(config.portable_root),
        "model_cache_root": str(config.model_cache_root),
        "version": version_block(),
    }


def _selection_payload(scan: SourceScan, selected_count: int) -> dict[str, Any]:
    """How many sources the walk found, and how many survive the exclusions."""

    return {
        "discovered_source_count": len(scan.selected),
        "selected_source_count": selected_count,
    }


def _vocabulary_payload(config: ResearchConfig, scan: SourceScan) -> dict[str, Any]:
    """The corpus vocabulary a reader sets, and the formats the walk accepts."""

    return {
        "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
        "ignored_extensions": scan.ignored_extensions,
        "language": {
            "corpus": config.settings.language_corpus,
            "languages": list(config.settings.corpus_languages),
            "bm25_stopwords": config.settings.bm25_stopwords_language,
            "warning": config.settings.embedding_language_warning,
        },
    }


def _available_methods_payload(
    *,
    default_method: str,
    available_methods: list[str],
) -> dict[str, Any]:
    """The retrieval methods this answer will serve, and the one it defaults to."""

    return {
        "default_retrieval_method": default_method,
        "available_retrieval_methods": available_methods,
    }


def _upgrade_payload(
    *,
    upgrade_required: bool,
    upgrade_reasons: list[str],
) -> dict[str, Any]:
    """Whether the policy moved under this generation, and which policies moved."""

    return {
        "generation_upgrade_required": upgrade_required,
        "upgrade_reasons": upgrade_reasons,
    }


def _build_payload(
    *,
    last_build_metrics: Any,
    ingestion_progress: Any,
) -> dict[str, Any]:
    """What the last build measured, and how far a build still running has come."""

    return {
        "last_build_metrics": last_build_metrics,
        "ingestion_progress": ingestion_progress,
    }


def _review_state_payload(
    *,
    exclusion_revision: str,
    metadata_revision: str,
    generation_metadata_revision: Any,
    overlay_active: bool,
    pending_source_paths: list[str],
    snapshot_outdated: bool,
) -> dict[str, Any]:
    """The review state a reader decided, and how much of it this generation holds.

    Reviewed metadata is a read-time overlay, so it never makes a generation
    stale and never names `ingest`; `snapshot_outdated` says the recorded
    snapshot differs from what is being applied, and nothing else.
    """

    return {
        "source_exclusion_revision": exclusion_revision,
        "metadata_revision": metadata_revision,
        "generation_metadata_revision": generation_metadata_revision,
        "metadata_overlay_active": overlay_active,
        "metadata_pending_source_paths": pending_source_paths,
        "generation_metadata_snapshot_outdated": snapshot_outdated,
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
        """Summarize the project from a caller already-parsed pointer.

        `search` already holds the selected generation and its manifest, and
        passes them in instead of making this read and parse them again.
        """

        try:
            scan = scan_sources(self.config)
        except SourcePolicyError as exc:
            raise ResearchError(str(exc)) from exc
        exclusions = self._source_exclusions()
        exclusion_revision = value_fingerprint(exclusions)
        chunk_exclusions = self._chunk_exclusions()
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
                **_base_payload(self.config, ready=False, stale=bool(selected)),
                **_selection_payload(scan, len(selected)),
                "excluded_source_count": len(exclusions),
                "excluded_chunk_count": len(chunk_exclusions),
                "categories": [],
                "projects": [],
                "excluded_sources": exclusion_records(self.config, scan, exclusions),
                **_vocabulary_payload(self.config, scan),
                **_review_state_payload(
                    exclusion_revision=exclusion_revision,
                    metadata_revision=metadata_revision,
                    generation_metadata_revision=None,
                    overlay_active=False,
                    pending_source_paths=sorted(metadata),
                    snapshot_outdated=False,
                ),
                **_available_methods_payload(
                    default_method=DEFAULT_RETRIEVAL_METHOD,
                    available_methods=[],
                ),
                **_upgrade_payload(upgrade_required=False, upgrade_reasons=[]),
                **_build_payload(
                    last_build_metrics=None,
                    ingestion_progress=ingestion_progress,
                ),
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
        upgrade_reasons = self._generation_upgrade_reasons(manifest)
        # The dense vectors were embedded by the model the generation recorded.
        # When the current embedding model has a different identity, scoring those
        # vectors with it would be a silent comparison of one model's output
        # against another's, so the dense and hybrid methods are not served. BM25
        # needs no query embeddings and keeps working.
        embedding_incompatible = "embedding_model" in upgrade_reasons
        recorded_methods = list(retrieval.get("available_methods") or ["bm25"])
        if embedding_incompatible:
            available_methods = [
                method for method in recorded_methods if method == "bm25"
            ] or ["bm25"]
        else:
            available_methods = recorded_methods
        hybrid_ready = "hybrid" in available_methods
        indexed_source_paths = {
            str(document.get("source_relative_path") or "")
            for document in manifest.get("documents", [])
        }
        metadata_overlay_active = bool(set(metadata) & indexed_source_paths)
        metadata_pending_source_paths = sorted(set(metadata) - indexed_source_paths)
        excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
        records = exclusion_records(self.config, scan, exclusions, manifest)
        # A chunk id is derived from content, so an entry recorded against
        # another generation is not yet a decision about this corpus. It is
        # counted here rather than folded into `stale`, which names `ingest` as
        # the call that closes the gap. `AGENTS.md` states the rule.
        absent_chunk_exclusions = sum(
            1
            for record in chunk_exclusions.values()
            if record["generation_id"] != str(manifest["generation_id"])
        )
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
        elif embedding_incompatible:
            status_message = (
                "The selected generation was built with a different embedding "
                "model, so its dense index cannot be served with the current one. "
                "BM25 remains available; run ingest to rebuild the dense index."
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
        if absent_chunk_exclusions:
            status_message += (
                f" {absent_chunk_exclusions} of {len(chunk_exclusions)} chunk "
                "exclusions were recorded against another generation: a chunk id is "
                "derived from content, so read each one against the current search "
                "before relying on it, and no ingestion restores a removed chunk."
            )
        effective_documents = _effective_documents(manifest, metadata)
        return {
            **_base_payload(self.config, ready=True, stale=stale),
            "generation_id": manifest["generation_id"],
            "created_at": manifest["created_at"],
            **_selection_payload(scan, len(selected)),
            "indexed_source_count": manifest["document_count"],
            "searchable_source_count": sum(
                str(document["document_id"]) not in excluded_document_ids
                for document in manifest.get("documents", [])
            ),
            "excluded_source_count": len(exclusions),
            "excluded_sources": records,
            "excluded_chunk_count": len(chunk_exclusions),
            "chunk_exclusion_other_generation_count": absent_chunk_exclusions,
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
            **_vocabulary_payload(self.config, scan),
            **_available_methods_payload(
                default_method=(
                    retrieval.get("default_method", "bm25") if hybrid_ready else "bm25"
                ),
                available_methods=available_methods,
            ),
            "hybrid_ready": hybrid_ready,
            "hybrid_upgrade_required": not hybrid_ready,
            **_upgrade_payload(
                upgrade_required=bool(upgrade_reasons),
                upgrade_reasons=upgrade_reasons,
            ),
            "retrieval": retrieval,
            **_build_payload(
                last_build_metrics=manifest.get("build_metrics"),
                ingestion_progress=ingestion_progress,
            ),
            **_review_state_payload(
                exclusion_revision=exclusion_revision,
                metadata_revision=metadata_revision,
                generation_metadata_revision=manifest.get("metadata_revision"),
                overlay_active=metadata_overlay_active,
                pending_source_paths=metadata_pending_source_paths,
                snapshot_outdated=metadata_changed,
            ),
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
        async with self._read():
            payload = await asyncio.to_thread(self._status)
            payload.update(
                await asyncio.to_thread(
                    self._generation_inventory,
                    payload.get("generation_id"),
                )
            )
            # Built from the payload this method already produced, so the health
            # answer and the status answer cannot come from two different states
            # of the project.
            report = await asyncio.to_thread(health_report, self.config, payload)
            payload.update(report.as_status_fields())
            return payload
