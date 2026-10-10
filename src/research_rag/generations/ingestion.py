from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus.docling_env import (
    DoclingEnvironmentError,
    ensure_docling_env,
    resolve_docling_env,
)
from ..corpus.extraction import (
    REMOVAL_FIELDS,
    ExtractionError,
    docling_pdf_batch,
    empty_removal_counts,
    extract_epub_spine_item,
    extract_mobi,
    extract_scanned_pdf_pages,
    merge_removal_counts,
    pdf_page_count,
    prepare_epub_extraction,
    prepare_scanned_pdf,
    scan_pdf_pages,
    screen_source_units,
)
from ..corpus.pdf_backend import (
    PDF_BACKEND_DOCLING,
    effective_maximum_unclean_percent,
    normalize_pdf_backend,
    pdf_backend_fingerprint,
    pdf_backend_record,
    recorded_pdf_backend_fingerprint,
)
from ..corpus.pdf_text_recovery import PdfRecoveryCache, close_pdf_recovery_cache
from ..corpus.sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourceFile,
    SourcePolicyError,
    SourceScan,
    scan_sources,
    sha256_file,
)
from ..corpus.text_quality import text_corruption_reasons
from ..project.config import project_command
from ..project.policy import (
    DEFAULT_RETRIEVAL_METHOD,
    RETRIEVAL_METHODS,
    ResearchError,
    _utc_now,
    value_fingerprint,
)
from ..project.settings import SETTINGS_BY_KEY
from ..project.support import (
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    GENERATION_ID_PATTERN,
    INGESTION_IDENTITY_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
    _atomic_to_thread,
    _checkpoint_identity,
    _chunk_text,
    _embedding_text,
    _enrich_chunks,
    _generation_id,
    _hash_with_stable_stat,
    _pdf_batch_count,
    _record_embedding_token_counts,
    _reranker_revision,
    _source_inventory,
    _source_stat_identity,
    _source_work_key,
)
from ..retrieval.artifact_lookup import (
    LOOKUP_RELATIVE_PATH,
    ArtifactLookup,
    ArtifactLookupError,
    ensure_artifact_lookup,
)
from ..retrieval.dense import DENSE_INDEX_PATHS, EXACT_BACKEND_NAME, QDRANT_BACKEND_NAME
from ..storage.records import (
    StorageError,
    atomic_write_json,
    atomic_write_jsonl,
    directory_statistics,
    fsync_directories,
    fsync_directory,
    iter_jsonl,
    read_json,
    read_jsonl,
    write_handoff_jsonl,
)
from .generation import (
    generation_artifacts_are_valid,
    load_reuse_snapshot,
    source_set_matches,
)

INGESTION_CHECKPOINT_VERSION = 1
PENDING_ACTIVATION_VERSION = 1

# The per-document loss a PDF build reports, summed for the whole corpus. The
# substantive count is the denominator a document's own loss rate is measured
# against; the rest are the material it lost. A source extracted before a field
# was recorded contributes zero rather than a guessed figure.
_DOCUMENT_LOSS_FIELDS = (
    "substantive_character_count",
    "discarded_corrupt_character_count",
    "cleaned_corrupt_span_count",
    "partially_cleaned_passage_count",
)


def _document_loss_totals(documents: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Sum the per-source loss fields a build reports for the whole corpus."""

    return {
        key: sum(int(document.get(key) or 0) for document in documents)
        for key in _DOCUMENT_LOSS_FIELDS
    }


class _SourceChangedDuringIngest(RuntimeError):
    pass


class IngestionWorkflow:
    @property
    def _pdf_backend(self) -> str:
        return normalize_pdf_backend(self.config.settings.pdf_backend)

    @property
    def _pdf_backend_fingerprint(self) -> str:
        return pdf_backend_fingerprint(self._pdf_backend, self.config)

    def _pdf_backend_matches_snapshot(
        self,
        manifest: dict[str, Any],
        selected_paths: list[str],
        sources_by_path: dict[str, SourceFile],
    ) -> bool:
        """Whether the snapshot's recorded backend may serve the selected PDFs.

        A PDF's units are backend-specific and are never reused across a backend
        change. An EPUB or a corpus with no selected PDF is unaffected, so the
        generation as a whole stays eligible for source-independent reuse.
        """

        has_pdf = any(
            (source := sources_by_path.get(path)) is not None
            and source.extension == ".pdf"
            for path in selected_paths
        )
        if not has_pdf:
            return True
        return (
            recorded_pdf_backend_fingerprint(manifest) == self._pdf_backend_fingerprint
        )

    def _effective_pdf_loss_limit(self) -> float:
        """The loss budget the shared gate applies for the selected backend."""

        return effective_maximum_unclean_percent(
            self._pdf_backend,
            self.config.settings.maximum_unclean_percent,
        )

    def _select_build_dense_backend(self, chunk_count: int) -> str:
        """Choose the dense backend for a new generation.

        `auto` uses the exact scan while a linear scan is cheaper than
        maintaining an ANN index, and falls back above the documented threshold.
        """

        if self.config.dense_backend == "exact":
            return EXACT_BACKEND_NAME
        if self.config.dense_backend == "qdrant":
            return QDRANT_BACKEND_NAME
        if chunk_count <= self.config.settings.exact_backend_chunk_limit:
            return EXACT_BACKEND_NAME
        return QDRANT_BACKEND_NAME

    def _load_ingestion_checkpoint(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        candidates: list[tuple[Path, dict[str, Any]]] = []
        roots = list(self.config.staging_root.iterdir())
        if self._pending_activation_path.is_file():
            try:
                journal = read_json(self._pending_activation_path)
            except StorageError:
                journal = None
            build_id = (
                str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
            )
            pending_root = self.config.generations_root / build_id
            if build_id and pending_root.is_dir():
                roots.append(pending_root)
        for root in sorted(set(roots)):
            checkpoint_path = root / "checkpoint.json"
            if not root.is_dir() or not checkpoint_path.is_file():
                continue
            try:
                checkpoint = read_json(checkpoint_path)
            except StorageError:
                continue
            if not isinstance(checkpoint, dict):
                continue
            if (
                checkpoint.get("schema_version") != INGESTION_CHECKPOINT_VERSION
                or checkpoint.get("project_id") != self.config.project_id
                or checkpoint.get("build_id") != root.name
            ):
                continue
            self._reconcile_checkpoint_progress(root, checkpoint)
            candidates.append((root, checkpoint))
        if not candidates:
            return None
        return max(candidates, key=lambda item: str(item[1].get("updated_at") or ""))

    @staticmethod
    def _reconcile_checkpoint_progress(
        root: Path,
        checkpoint: dict[str, Any],
        *,
        source_states: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Rebuild split progress counters from committed per-source state."""

        sources_root = root / "work" / "sources"
        if not sources_root.is_dir():
            return
        selected_paths = [
            str(relative) for relative in checkpoint.get("selected_source_paths", [])
        ]
        source_formats = {
            str(record.get("source_relative_path") or ""): str(
                record.get("format") or ""
            )
            for record in checkpoint.get("source_inventory", [])
            if isinstance(record, dict)
        }
        extracted_paths = {
            str(relative) for relative in checkpoint.get("extracted_source_paths", [])
        }
        extraction_total = 0
        extraction_completed = 0
        chunking_total = 0
        chunking_completed = 0

        try:
            for relative in selected_paths:
                state_path = sources_root / _source_work_key(relative) / "state.json"
                if source_states is None:
                    if not state_path.is_file():
                        continue
                    state = read_json(state_path)
                else:
                    state = source_states.get(relative)
                if not isinstance(state, dict):
                    return
                if state.get("extraction_stage") == "skipped":
                    continue

                if not state.get("reused"):
                    extraction_stage = str(state.get("extraction_stage") or "")
                    unit_total = int(state.get("total") or 0)
                    next_index = max(
                        0, min(int(state.get("next_index") or 0), unit_total)
                    )
                    source_format = source_formats.get(relative)
                    if source_format == "pdf":
                        page_batch_size = int(state.get("page_batch_size") or 1)
                        batch_total = _pdf_batch_count(
                            unit_total,
                            page_batch_size,
                        )
                        completed_batches = _pdf_batch_count(
                            next_index,
                            page_batch_size,
                        )
                        if state.get("docling"):
                            # The Docling path has one pass over the page batches,
                            # then finalize.
                            source_total = batch_total + 2
                            if extraction_stage == "docling_pages":
                                source_completed = completed_batches
                            elif extraction_stage == "finalize":
                                source_completed = batch_total + 1
                            elif extraction_stage == "complete":
                                source_completed = source_total - int(
                                    relative not in extracted_paths
                                )
                            else:
                                return
                        else:
                            source_total = (batch_total * 2) + 2
                            if extraction_stage == "pdf_scan":
                                source_completed = completed_batches
                            elif extraction_stage == "pdf_prepare":
                                source_completed = batch_total
                            elif extraction_stage == "pdf_pages":
                                source_completed = batch_total + 1 + completed_batches
                            elif extraction_stage == "finalize":
                                source_completed = (batch_total * 2) + 1
                            elif extraction_stage == "complete":
                                source_completed = source_total - int(
                                    relative not in extracted_paths
                                )
                            else:
                                return
                    elif source_format == "epub" and extraction_stage == "epub_items":
                        source_total = unit_total + 2
                        source_completed = next_index + 1
                    elif source_format == "epub" and extraction_stage == "finalize":
                        source_total = unit_total + 2
                        source_completed = unit_total + 1
                    elif source_format == "epub" and extraction_stage == "complete":
                        source_total = unit_total + 2
                        source_completed = source_total - int(
                            relative not in extracted_paths
                        )
                    else:
                        return
                    extraction_total += source_total
                    extraction_completed += min(source_completed, source_total)

                    if "chunking_work_total" in state:
                        source_chunk_total = int(state["chunking_work_total"])
                        chunking_total += source_chunk_total
                        chunking_completed += min(
                            int(state.get("chunked_unit_count") or 0),
                            source_chunk_total,
                        )
        except (OSError, StorageError, TypeError, ValueError):
            # Leave the durable checkpoint untouched. The normal state machine
            # will diagnose malformed per-source state instead of hiding it.
            return

        checkpoint["extraction_work_total"] = extraction_total
        checkpoint["extraction_work_completed"] = extraction_completed
        checkpoint["chunking_work_total"] = chunking_total
        checkpoint["chunking_work_completed"] = chunking_completed

    def _cleanup_invalid_staging_roots(self) -> None:
        """Remove uncommitted staging entries that cannot be resumed safely."""

        for root in sorted(self.config.staging_root.iterdir()):
            checkpoint: Any = None
            if root.is_dir() and not root.is_symlink():
                checkpoint_path = root / "checkpoint.json"
                if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                    try:
                        checkpoint = read_json(checkpoint_path)
                    except StorageError:
                        checkpoint = None
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version") == INGESTION_CHECKPOINT_VERSION
                and checkpoint.get("project_id") == self.config.project_id
                and checkpoint.get("build_id") == root.name
            ):
                continue

            diagnostic_id = (
                root.name
                if re.fullmatch(r"[A-Za-z0-9._-]+", root.name)
                else f"invalid-{_source_work_key(root.name)}"
            )
            atomic_write_json(
                self.config.failures_root / f"{diagnostic_id}-orphan.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": root.name,
                    "failed_at": _utc_now(),
                    "phase": None,
                    "error": "Removed invalid or checkpointless staging state",
                    "resumable": False,
                },
            )
            if root.is_dir() and not root.is_symlink():
                shutil.rmtree(root)
            else:
                root.unlink(missing_ok=True)
        fsync_directory(self.config.staging_root)

    def _ingestion_progress(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        phase = str(checkpoint.get("phase") or "source_hashing")
        inventory = checkpoint.get("source_inventory", [])
        selected = checkpoint.get("selected_source_paths", [])
        if phase == "source_hashing":
            completed = len(checkpoint.get("source_digests", {}))
            total = len(inventory)
            unit = "sources"
        elif phase == "extraction":
            completed = int(checkpoint.get("extraction_work_completed") or 0)
            total = int(checkpoint.get("extraction_work_total") or 0)
            if not total:
                completed = len(checkpoint.get("extracted_source_paths", []))
                total = len(selected)
                unit = "sources"
            else:
                unit = "pdf_page_batches_or_epub_sections"
        elif phase == "chunking":
            completed = int(checkpoint.get("chunking_work_completed") or 0)
            total = int(checkpoint.get("chunking_work_total") or 0)
            if not total:
                completed = len(checkpoint.get("chunked_source_paths", []))
                total = len(selected)
                unit = "sources"
            else:
                unit = "extraction_units"
        elif phase == "embedding":
            completed = int(checkpoint.get("embedded_chunk_count") or 0)
            total = int(checkpoint.get("chunk_count") or 0)
            unit = "chunks"
        elif phase in {"dense_indexing", "qdrant_indexing"}:
            completed = int(
                checkpoint.get("dense_indexed_count")
                or checkpoint.get("qdrant_indexed_count")
                or 0
            )
            total = int(checkpoint.get("chunk_count") or 0)
            unit = "chunks"
        elif phase == "source_revalidation":
            completed = len(checkpoint.get("revalidation_digests", {}))
            total = len(inventory)
            unit = "sources"
        else:
            completed = int(phase in {"complete"})
            total = 1
            unit = "phase"
        source: str | None = None
        source_progress: dict[str, Any] | None = None
        overall_progress = {"completed": completed, "total": total, "unit": unit}
        if phase in {"source_hashing", "source_revalidation", "extraction", "chunking"}:
            paths = (
                [
                    record["source_relative_path"]
                    for record in inventory
                    if isinstance(record, dict)
                    and isinstance(record.get("source_relative_path"), str)
                ]
                if phase in {"source_hashing", "source_revalidation"}
                else selected
            )
            finished_key = {
                "source_hashing": "source_digests",
                "source_revalidation": "revalidation_digests",
                "extraction": "extracted_source_paths",
                "chunking": "chunked_source_paths",
            }[phase]
            finished = set(checkpoint.get(finished_key) or [])
            overall_progress = {
                "completed": sum(path in finished for path in paths),
                "total": len(paths),
                "unit": "sources",
            }
            source = next((path for path in paths if path not in finished), None)
            if source is not None:
                if phase in {"source_hashing", "source_revalidation"}:
                    source_progress = {"completed": 0, "total": 1, "unit": "sources"}
                else:
                    source_progress = self._source_ingestion_progress(
                        checkpoint, source, phase
                    )
        return {
            "build_id": str(checkpoint["build_id"]),
            "phase": phase,
            "source": source,
            "source_progress": source_progress,
            "overall_progress": overall_progress,
            # Durable phase timings do not record a homogeneous work rate:
            # sources vary in size, and reused work changes batch costs. Do not
            # extrapolate these counters into a build or phase deadline.
            "eta_seconds": None,
            "progress": {
                "completed": completed,
                "total": total,
                "unit": unit,
            },
            "parameters": dict(checkpoint.get("parameters") or {}),
            "created_at": checkpoint.get("created_at"),
            "checkpointed_at": checkpoint.get("updated_at"),
        }

    def _source_ingestion_progress(
        self, checkpoint: dict[str, Any], source: str, phase: str
    ) -> dict[str, Any] | None:
        """Read only the pending source's small, committed staging state."""

        build_id = str(checkpoint.get("build_id") or "")
        if not re.fullmatch(GENERATION_ID_PATTERN, build_id):
            return None
        root = self.config.staging_root / build_id
        source_root = root / "work" / "sources" / _source_work_key(source)
        state_path = source_root / "state.json"
        # Never follow staging symlinks, including intermediate directories.
        if any(
            path.is_symlink()
            for path in (
                self.config.staging_root,
                root,
                root / "work",
                source_root.parent,
                source_root,
                state_path,
            )
        ):
            return None
        try:
            if not state_path.is_file() or state_path.stat().st_size > 64 * 1024:
                return None
            state = read_json(state_path)
            if not isinstance(state, dict) or state.get("reused"):
                return None
            if phase == "chunking":
                total = int(state.get("chunking_work_total") or 0)
                completed = int(state.get("chunked_unit_count") or 0)
                unit = "extraction_units"
            else:
                # Use the same durable extraction-stage accounting as status,
                # restricted to one selected path (no corpus/artifact scan).
                view = {**checkpoint, "selected_source_paths": [source]}
                view.pop("extraction_work_total", None)
                view.pop("extraction_work_completed", None)
                self._reconcile_checkpoint_progress(
                    root, view, source_states={source: state}
                )
                total = int(view.get("extraction_work_total") or 0)
                completed = int(view.get("extraction_work_completed") or 0)
                unit = "pdf_page_batches_or_epub_sections"
            if total <= 0 or completed < 0:
                return None
            return {"completed": min(completed, total), "total": total, "unit": unit}
        except (OSError, StorageError, TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _write_checkpoint(root: Path, checkpoint: dict[str, Any]) -> None:
        checkpoint["updated_at"] = _utc_now()
        atomic_write_json(root / "checkpoint.json", checkpoint)

    def _discard_checkpoint(
        self,
        root: Path,
        checkpoint: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        atomic_write_json(
            self.config.failures_root / f"{checkpoint['build_id']}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generation_id": checkpoint["build_id"],
                "failed_at": _utc_now(),
                "phase": checkpoint.get("phase"),
                "error": reason,
                "resumable": False,
            },
        )
        shutil.rmtree(root, ignore_errors=True)
        fsync_directory(root.parent)

    @property
    def _pending_activation_path(self) -> Path:
        return self.config.state_root / "pending-activation.json"

    def _discard_invalid_activation_journal(
        self,
        journal: Any,
        *,
        reason: str,
    ) -> None:
        """Remove an unreadable journal without trusting its referenced paths."""

        build_id = (
            str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
        )
        valid_build_id = bool(re.fullmatch(GENERATION_ID_PATTERN, build_id))
        project_matches = bool(
            isinstance(journal, dict)
            and journal.get("project_id") == self.config.project_id
        )
        if valid_build_id and project_matches:
            staging_root = self.config.staging_root / build_id
            checkpoint_path = staging_root / "checkpoint.json"
            try:
                checkpoint = (
                    read_json(checkpoint_path) if checkpoint_path.is_file() else None
                )
            except StorageError:
                checkpoint = None
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("project_id") == self.config.project_id
                and checkpoint.get("build_id") == build_id
            ):
                self._discard_checkpoint(staging_root, checkpoint, reason=reason)

        diagnostic_id = build_id if valid_build_id else uuid.uuid4().hex[:16]
        atomic_write_json(
            self.config.failures_root
            / f"{diagnostic_id}-invalid-activation-journal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generation_id": build_id or None,
                "failed_at": _utc_now(),
                "phase": "activation",
                "error": reason,
                "resumable": False,
            },
        )
        self._pending_activation_path.unlink(missing_ok=True)
        fsync_directory(self.config.state_root)

    @staticmethod
    async def _ensure_artifact_lookup(
        root: Path,
        manifest: dict[str, Any],
    ) -> ArtifactLookup:
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ResearchError("Generation manifest has no artifact mapping")
        try:
            chunks_path = root / str(files["chunks"])
            units_path = root / str(files["extracted_units"])
            return await _atomic_to_thread(
                ensure_artifact_lookup,
                chunks_path,
                units_path,
                root / LOOKUP_RELATIVE_PATH,
            )
        except (ArtifactLookupError, KeyError, OSError) as exc:
            raise ResearchError("Generation artifact lookup is unavailable") from exc

    async def _validate_generation_for_activation(
        self,
        root: Path,
        manifest: dict[str, Any],
    ) -> None:
        """Validate portable artifacts and both indexes before pointer changes."""

        lookup = await self._ensure_artifact_lookup(root, manifest)
        if not await _atomic_to_thread(
            generation_artifacts_are_valid,
            root,
            manifest,
            embedding=self.config.settings.embedding_facts,
        ):
            raise ResearchError("Generation portable artifacts failed validation")
        try:
            # The probe points the gateway's one retriever at this generation, so
            # a search running beside the build waits for it and reloads its own.
            async with self._retriever_lock:
                self._loaded_generation = None
                await self.ultrarag.initialize_bm25(
                    root / str(manifest["files"]["chunks"]),
                    root / str(manifest["files"]["bm25_index"]),
                    language=self.config.settings.bm25_stopwords_language,
                )
                chunks_path = root / str(manifest["files"]["chunks"])
                probe = next(iter_jsonl(chunks_path))["contents"]
                bm25_probe = await self.ultrarag.search_bm25(str(probe), 1)
            stored_probe = (
                await _atomic_to_thread(lookup.chunks_by_contents, bm25_probe)
                if bm25_probe
                else {}
            )
            if not bm25_probe or not stored_probe.get(bm25_probe[0]):
                raise RuntimeError("BM25 validation probe returned no stored passage")
            await _atomic_to_thread(
                self._dense_for(manifest).validate_index,
                root / str(manifest["files"]["dense_index"]),
                expected_count=int(manifest["chunk_count"]),
                dimension=self.config.settings.embedding_dimension,
            )
        except Exception as exc:
            raise ResearchError(
                "Generation retrieval indexes failed validation"
            ) from exc

    async def _recover_pending_activation(
        self,
        *,
        expected_identity: str,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Finish a compatible validated activation or discard its orphan."""

        journal_path = self._pending_activation_path
        if not journal_path.exists():
            return None
        try:
            journal = read_json(journal_path)
        except StorageError:
            self._discard_invalid_activation_journal(
                None,
                reason="Pending activation journal contained invalid JSON",
            )
            return None
        build_id = (
            str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
        )
        if (
            not isinstance(journal, dict)
            or journal.get("schema_version") != PENDING_ACTIVATION_VERSION
            or journal.get("project_id") != self.config.project_id
            or not re.fullmatch(GENERATION_ID_PATTERN, build_id)
        ):
            self._discard_invalid_activation_journal(
                journal,
                reason="Pending activation journal fields were invalid",
            )
            return None

        staging_root = self.config.staging_root / build_id
        generation_root = self.config.generations_root / build_id
        current_id = str(current[1]["generation_id"]) if current is not None else None
        if current_id == build_id:
            # The pointer was committed before the process stopped. Clean only
            # activation debris, then let normal ingest validate freshness and
            # repair artifacts if necessary.
            shutil.rmtree(generation_root / "work", ignore_errors=True)
            (generation_root / "checkpoint.json").unlink(missing_ok=True)
            fsync_directory(generation_root)
            journal_path.unlink(missing_ok=True)
            fsync_directory(journal_path.parent)
            self._loaded_generation = None
            return None

        root = generation_root if generation_root.is_dir() else staging_root
        checkpoint_path = root / "checkpoint.json"
        manifest_path = root / "manifest.json"
        checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else None
        manifest = read_json(manifest_path) if manifest_path.is_file() else None
        source_stats = journal.get("revalidation_stats")
        source_paths = {source.source_relative_path for source in scan.selected}
        try:
            source_state_matches = bool(
                isinstance(source_stats, dict)
                and set(source_stats) == source_paths
                and all(
                    _source_stat_identity(source.path)
                    == source_stats[source.source_relative_path]
                    for source in scan.selected
                )
            )
        except OSError:
            source_state_matches = False
        baseline_matches = current_id == journal.get("baseline_generation_id")
        valid = bool(
            isinstance(manifest, dict)
            and (
                (
                    isinstance(checkpoint, dict)
                    and checkpoint.get("build_id") == build_id
                    and checkpoint.get("identity") == expected_identity
                )
                or (root == generation_root and checkpoint is None)
            )
            and journal.get("identity") == expected_identity
            and manifest.get("generation_id") == build_id
            and manifest.get("project_id") == self.config.project_id
            and manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("artifact_policy_version") == ARTIFACT_POLICY_VERSION
            and _source_inventory(scan) == journal.get("source_inventory")
            and value_fingerprint(exclusions)
            == journal.get("source_exclusion_revision")
            and source_state_matches
            and baseline_matches
        )
        if valid:
            manifest_digests = {
                str(item.get("source_relative_path") or ""): str(
                    item.get("sha256") or ""
                )
                for item in manifest.get("source_files", [])
                if isinstance(item, dict)
            }
            valid = manifest_digests == journal.get("source_digests")
        if valid:
            self._loaded_generation = None
            try:
                await self._validate_generation_for_activation(root, manifest)
            except (ResearchError, TypeError):
                valid = False
        if not valid:
            if isinstance(checkpoint, dict):
                self._discard_checkpoint(
                    root,
                    checkpoint,
                    reason=(
                        "Pending activation was superseded by changed inputs, "
                        "selected generation, or incomplete artifacts"
                    ),
                )
            journal_path.unlink(missing_ok=True)
            self._loaded_generation = None
            return None

        shutil.rmtree(root / "work", ignore_errors=True)
        if root == staging_root:
            os.replace(staging_root, generation_root)
            fsync_directory(self.config.generations_root)
            fsync_directory(self.config.staging_root)
        skipped_sources = manifest.get("build_metrics", {}).get("skipped_sources", [])
        # Legacy journals have no intent. Their manifest must still prevent
        # automatic selection of incomplete source coverage.
        select_generation = not skipped_sources and journal.get("select", True) is True
        if select_generation:
            atomic_write_json(
                self.config.current_path,
                {"schema_version": 1, "generation_id": build_id},
            )
        (generation_root / "checkpoint.json").unlink(missing_ok=True)
        fsync_directory(generation_root)
        journal_path.unlink(missing_ok=True)
        fsync_directory(journal_path.parent)
        self._loaded_generation = None
        return {
            "status": "partial" if skipped_sources else "ready",
            "generation_changed": select_generation,
            "current_generation_id": build_id if select_generation else current_id,
            **(
                {
                    "manual_selection_command": project_command(
                        self.config.project_root, "generations", "--use", build_id
                    )
                }
                if not select_generation
                else {}
            ),
            "activation_recovered": True,
            "skipped_sources": skipped_sources,
            "generation_id": build_id,
            "generation_root": str(generation_root),
            "message": (
                "Recovered and selected the completed generation."
                if select_generation
                else "Recovered the partial generation without changing the selected generation."
            ),
        }

    @staticmethod
    def _remove_uncommitted_files(root: Path) -> None:
        """Remove files that are always recreated before their checkpoint commit."""

        for pattern in ("*.tmp", "raw-chunks.jsonl"):
            for path in root.rglob(pattern):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)
        checkpoint_path = root / "checkpoint.json"
        if checkpoint_path.is_file():
            checkpoint = read_json(checkpoint_path)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("phase") == "bm25_indexing"
            ):
                shutil.rmtree(root / "indexes" / "bm25", ignore_errors=True)

    def _create_checkpoint(
        self,
        *,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        exclusion_revision: str,
        baseline_generation_id: str | None,
        chunk_size: int,
        chunk_overlap: int,
        force_recompute: bool,
    ) -> tuple[Path, dict[str, Any]]:
        build_id = _generation_id()
        inventory = _source_inventory(scan)
        identity = _checkpoint_identity(
            project_id=self.config.project_id,
            inventory=inventory,
            exclusion_revision=exclusion_revision,
            baseline_generation_id=baseline_generation_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_headers=self.config.settings.chunk_headers,
            force_recompute=force_recompute,
            embedding=self.config.settings.embedding_facts,
            maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
            pdf_backend_fingerprint=self._pdf_backend_fingerprint,
        )
        now = _utc_now()
        checkpoint: dict[str, Any] = {
            "schema_version": INGESTION_CHECKPOINT_VERSION,
            "build_id": build_id,
            "project_id": self.config.project_id,
            "identity": identity,
            "identity_policy_version": INGESTION_IDENTITY_POLICY_VERSION,
            "metadata_storage_policy": METADATA_STORAGE_POLICY,
            "phase": "source_hashing",
            "created_at": now,
            "updated_at": now,
            "baseline_generation_id": baseline_generation_id,
            "parameters": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_headers": self.config.settings.chunk_headers,
                "maximum_unclean_percent": self.config.settings.maximum_unclean_percent,
                "force_recompute": force_recompute,
            },
            "source_inventory": inventory,
            "selected_source_paths": [
                source.source_relative_path
                for source in scan.selected
                if source.source_relative_path not in exclusions
            ],
            "source_exclusion_revision": exclusion_revision,
            "source_digests": {},
            "extracted_source_paths": [],
            "chunked_source_paths": [],
            "embedded_chunk_count": 0,
            "dense_indexed_count": 0,
            "revalidation_digests": {},
            "revalidation_stats": {},
            "phase_timings_seconds": {},
        }
        root = self.config.staging_root / build_id
        root.mkdir(parents=False, exist_ok=False)
        fsync_directory(self.config.staging_root)
        self._write_checkpoint(root, checkpoint)
        return root, checkpoint

    @staticmethod
    def _source_artifact_root(staging_root: Path, relative_path: str) -> Path:
        return staging_root / "work" / "sources" / _source_work_key(relative_path)

    def _pdf_recovery_cache(self, digest: str) -> PdfRecoveryCache:
        """The native-reader cache for one source, kept across its scan batches.

        Keyed by project and content digest, so a build never reuses a parse of
        another project or of bytes that changed. The cache belongs to the build
        that is scanning; it is released when the source finishes or the build
        ends.
        """

        caches = getattr(self, "_pdf_recovery_caches", None)
        if caches is None:
            caches = {}
            self._pdf_recovery_caches = caches
        key = (self.config.project_id, digest)
        cache = caches.get(key)
        if cache is None:
            cache = {}
            caches[key] = cache
        return cache

    def _release_pdf_recovery_cache(self, digest: str) -> None:
        caches = getattr(self, "_pdf_recovery_caches", None)
        if not caches:
            return
        close_pdf_recovery_cache(caches.pop((self.config.project_id, digest), None))

    def _release_pdf_recovery_caches(self) -> None:
        caches = getattr(self, "_pdf_recovery_caches", None)
        if not caches:
            return
        for cache in caches.values():
            close_pdf_recovery_cache(cache)
        caches.clear()

    @staticmethod
    def _save_vector_batch(
        path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _assemble_vector_batches(
        path: Path,
        batches_root: Path,
        total: int,
        embedding_batch_size: int,
        dimension: int,
    ) -> None:
        """Assemble portable vectors without allocating a second full matrix."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        vectors = None
        try:
            vectors = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(total, dimension),
            )
            for offset in range(0, total, embedding_batch_size):
                batch = np.load(
                    batches_root / f"{offset:012d}.npy",
                    allow_pickle=False,
                    mmap_mode="r",
                )
                vectors[offset : offset + len(batch)] = batch
            vectors.flush()
            del vectors
            vectors = None
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            if vectors is not None:
                del vectors
            temporary.unlink(missing_ok=True)

    def _in_progress_result(
        self,
        checkpoint: dict[str, Any],
        *,
        message: str | None = None,
    ) -> dict[str, Any]:
        progress = self._ingestion_progress(checkpoint)
        return {
            "status": "in_progress",
            "skipped_sources": checkpoint.get("skipped_sources", []),
            "generation_changed": False,
            **progress,
            "next_action": "call_ingest_again",
            "message": message
            or "Ingestion checkpoint saved; call ingest again with the same settings.",
        }

    @staticmethod
    def _add_phase_time(
        checkpoint: dict[str, Any],
        phase: str,
        elapsed: float,
    ) -> None:
        timings = checkpoint.setdefault("phase_timings_seconds", {})
        timings[phase] = float(timings.get(phase) or 0.0) + elapsed

    async def _advance_ingestion(  # noqa: C901 — split is tracked in TODO.md
        self,
        *,
        staging_root: Path,
        checkpoint: dict[str, Any],
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        current: tuple[Path, dict[str, Any]] | None,
        deadline: float,
    ) -> dict[str, Any]:
        parameters = checkpoint["parameters"]
        chunk_size = int(parameters["chunk_size"])
        chunk_overlap = int(parameters["chunk_overlap"])
        force_recompute = bool(parameters["force_recompute"])
        sources_by_path: dict[str, SourceFile] = {
            source.source_relative_path: source for source in scan.selected
        }
        selected_paths = [str(item) for item in checkpoint["selected_source_paths"]]
        snapshot = None
        if not force_recompute:
            snapshot = load_reuse_snapshot(
                current,
                schema_version=SCHEMA_VERSION,
                extraction_policy_version=EXTRACTION_POLICY_VERSION,
                cleaning_policy_version=CLEANING_POLICY_VERSION,
                artifact_policy_version=ARTIFACT_POLICY_VERSION,
                project_id=self.config.project_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_headers=self.config.settings.chunk_headers,
                maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
                load_units=False,
                load_chunks=False,
                load_vectors=False,
                embedding=self.config.settings.embedding_facts,
            )
        units_cache: dict[str, list[dict[str, Any]]] = {}
        portable_vectors: np.ndarray[Any, np.dtype[np.float32]] | None = None

        def budget_expired() -> bool:
            return time.perf_counter() >= deadline

        def record_skip(record: dict[str, str]) -> None:
            relative = record["source_relative_path"]
            skipped = checkpoint.setdefault("skipped_sources", [])
            if not any(item["source_relative_path"] == relative for item in skipped):
                skipped.append(record)
            for key in ("extracted_source_paths", "chunked_source_paths"):
                if relative not in checkpoint[key]:
                    checkpoint[key].append(relative)
            self._write_checkpoint(staging_root, checkpoint)

        async def extract_source(operation: Any, *args: Any, **kwargs: Any) -> Any:
            # Only typed failures from the source-local extractor are omissions.
            # Artifact writes, hashing, chunking and index failures remain fatal.
            try:
                return await _atomic_to_thread(operation, *args, **kwargs)
            except ExtractionError as exc:
                record = {
                    "source_relative_path": relative,
                    "reason": str(exc),
                    # OCR is performed outside this app, so the remedy is the same
                    # for every source-local failure: repair or replace the file.
                    "remedy": "Repair or replace this source, then run research-rag ingest.",
                }
                skipped_state = {
                    **(state or {}),
                    "extraction_stage": "skipped",
                    "skipped_source": record,
                }
                # Commit the source decision first so a lost checkpoint write
                # can recover it without repeating partially extracted work.
                atomic_write_json(state_path, skipped_state)
                self._add_phase_time(
                    checkpoint, "extraction", time.perf_counter() - started
                )
                failed_digest = checkpoint["source_digests"].get(
                    record["source_relative_path"]
                )
                if failed_digest is not None:
                    self._release_pdf_recovery_cache(str(failed_digest))
                record_skip(record)
                return None

        def source_records() -> list[dict[str, Any]]:
            digests = checkpoint["source_digests"]
            return [
                {
                    **record,
                    "sha256": str(digests[record["source_relative_path"]]),
                    "included": record["source_relative_path"] not in exclusions,
                }
                for record in checkpoint["source_inventory"]
            ]

        def revalidation_stats_match(*, require_complete: bool = False) -> bool:
            recorded = checkpoint.get("revalidation_stats") or {}
            if not isinstance(recorded, dict):
                return False
            if require_complete and set(recorded) != set(sources_by_path):
                return False
            try:
                return all(
                    _source_stat_identity(sources_by_path[relative].path) == identity
                    for relative, identity in recorded.items()
                    if relative in sources_by_path
                ) and set(recorded).issubset(sources_by_path)
            except OSError:
                return False

        if checkpoint.get("revalidation_stats") and not revalidation_stats_match():
            raise _SourceChangedDuringIngest(
                "A source changed after its final ingestion hash"
            )

        while True:
            phase = str(checkpoint["phase"])
            if phase == "source_hashing":
                digests = checkpoint["source_digests"]
                pending = [
                    record
                    for record in checkpoint["source_inventory"]
                    if record["source_relative_path"] not in digests
                ]
                if pending:
                    record = pending[0]
                    source = sources_by_path[str(record["source_relative_path"])]
                    started = time.perf_counter()
                    try:
                        digest = await _atomic_to_thread(sha256_file, source.path)
                    except OSError as exc:
                        raise _SourceChangedDuringIngest(
                            "A source became unavailable during hashing"
                        ) from exc
                    self._add_phase_time(
                        checkpoint,
                        "source_hashing",
                        time.perf_counter() - started,
                    )
                    digests[source.source_relative_path] = digest
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue

                records = source_records()
                if (
                    snapshot is not None
                    and not snapshot.manifest.get("build_metrics", {}).get(
                        "skipped_sources"
                    )
                    # A corpus that selects a PDF under a different backend is not
                    # unchanged; an EPUB-only corpus still is.
                    and self._pdf_backend_matches_snapshot(
                        snapshot.manifest, selected_paths, sources_by_path
                    )
                    and source_set_matches(
                        snapshot,
                        records,
                        exclusion_revision=str(checkpoint["source_exclusion_revision"]),
                        retrieval_policy_fingerprint=(
                            self.retrieval_policy_fingerprint
                        ),
                        metadata_storage_policy=METADATA_STORAGE_POLICY,
                    )
                ):
                    manifest = snapshot.manifest
                    try:
                        await self._validate_generation_for_activation(
                            snapshot.root,
                            manifest,
                        )
                    except ResearchError as exc:
                        checkpoint["reuse_rejection_reason"] = str(exc)
                        snapshot = None
                    else:
                        self._loaded_generation = str(manifest["generation_id"])
                        shutil.rmtree(staging_root, ignore_errors=True)
                        return {
                            "status": "unchanged",
                            "skipped_sources": manifest.get("build_metrics", {}).get(
                                "skipped_sources", []
                            ),
                            "generation_changed": False,
                            "generation_id": manifest["generation_id"],
                            "generation_root": str(snapshot.root),
                            "source_file_count": len(scan.selected),
                            "excluded_source_count": len(exclusions),
                            "document_count": manifest["document_count"],
                            "extraction_unit_count": manifest["extraction_unit_count"],
                            "chunk_count": manifest["chunk_count"],
                            "reused_document_count": manifest["document_count"],
                            "rebuilt_document_count": 0,
                            "reused_chunk_count": manifest["chunk_count"],
                            "rebuilt_chunk_count": 0,
                            "reused_vector_count": manifest["chunk_count"],
                            "created_vector_count": 0,
                            "phase_timings_seconds": checkpoint[
                                "phase_timings_seconds"
                            ],
                            "message": (
                                "Inputs match the selected generation; no build "
                                "was needed."
                            ),
                        }
                checkpoint["phase"] = "extraction"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "extraction":
                if snapshot is not None and not snapshot.units_by_document:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_headers=self.config.settings.chunk_headers,
                        maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
                        load_units=True,
                        load_chunks=True,
                        load_vectors=False,
                        embedding=self.config.settings.embedding_facts,
                    )
                completed = set(checkpoint["extracted_source_paths"])
                relative = next(
                    (path for path in selected_paths if path not in completed),
                    None,
                )
                if relative is None:
                    checkpoint["phase"] = "chunking"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                source = sources_by_path[relative]
                source_digest = str(checkpoint["source_digests"][relative])
                artifact_root = self._source_artifact_root(staging_root, relative)
                artifact_root.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                state_path = artifact_root / "state.json"
                state = read_json(state_path) if state_path.exists() else None
                if state is None:
                    previous_source = (
                        snapshot.source_files.get(relative)
                        if snapshot is not None
                        else None
                    )
                    previous_document = (
                        snapshot.documents.get(relative)
                        if snapshot is not None
                        else None
                    )
                    document_id = str(
                        (previous_document or {}).get("document_id") or ""
                    )
                    reusable = bool(
                        snapshot is not None
                        and previous_source is not None
                        and previous_document is not None
                        and previous_source.get("sha256")
                        == checkpoint["source_digests"][relative]
                        and snapshot.manifest.get("metadata_storage_policy")
                        == METADATA_STORAGE_POLICY
                        and document_id in snapshot.units_by_document
                        and document_id in snapshot.chunks_by_document
                        # A PDF's units are backend-specific: never reuse them
                        # across a backend change, whatever else matches. EPUB
                        # units are unaffected and stay reusable.
                        and (
                            source.extension != ".pdf"
                            or recorded_pdf_backend_fingerprint(snapshot.manifest)
                            == self._pdf_backend_fingerprint
                        )
                    )
                    if (
                        reusable
                        and snapshot is not None
                        and previous_document is not None
                    ):
                        document = dict(previous_document)
                        document["size"] = source.size
                        document["mtime_ns"] = source.mtime_ns
                        units = [
                            dict(item)
                            for item in snapshot.units_by_document[document_id]
                        ]
                        atomic_write_json(artifact_root / "document.json", document)
                        atomic_write_jsonl(artifact_root / "units.jsonl", units)
                        state = {
                            "reused": True,
                            "extraction_stage": "complete",
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        atomic_write_json(state_path, state)
                        checkpoint["extracted_source_paths"].append(relative)
                        checkpoint["reused_document_count"] = (
                            int(checkpoint.get("reused_document_count") or 0) + 1
                        )
                    elif source.extension == ".pdf":
                        total = await extract_source(pdf_page_count, source)
                        if total is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        docling = self._pdf_backend == PDF_BACKEND_DOCLING
                        state = {
                            "reused": False,
                            "docling": docling,
                            "extraction_stage": (
                                "docling_pages" if docling else "pdf_scan"
                            ),
                            "next_index": 0,
                            "total": total,
                            "page_batch_size": self.config.settings.pdf_page_batch_size,
                            "removals": empty_removal_counts(),
                            "empty_units": 0,
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0)
                            + (
                                _pdf_batch_count(
                                    total, self.config.settings.pdf_page_batch_size
                                )
                                * (1 if docling else 2)
                            )
                            + 2
                        )
                        atomic_write_json(state_path, state)
                    elif source.extension == ".mobi":
                        prepared = await extract_source(
                            extract_mobi, source, checkpoint["source_digests"][relative]
                        )
                        if prepared is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        document, units = prepared
                        atomic_write_json(artifact_root / "document.json", document)
                        atomic_write_jsonl(
                            artifact_root / "unit-batches" / "00000000.jsonl", units
                        )
                        state = {
                            "reused": False,
                            "extraction_stage": "finalize",
                            "total": 1,
                            "removals": {
                                key: int(document.get(key) or 0)
                                for key in REMOVAL_FIELDS
                            },
                            "empty_units": document["empty_units"],
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0) + 2
                        )
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                        atomic_write_json(state_path, state)
                    else:
                        prepared = await extract_source(
                            prepare_epub_extraction,
                            source,
                            checkpoint["source_digests"][relative],
                        )
                        if prepared is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        document, total = prepared
                        atomic_write_json(artifact_root / "document.json", document)
                        state = {
                            "reused": False,
                            "extraction_stage": "epub_items",
                            "next_index": 0,
                            "total": total,
                            "removals": empty_removal_counts(),
                            "empty_units": 0,
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0)
                            + total
                            + 2
                        )
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                        atomic_write_json(state_path, state)
                    self._add_phase_time(
                        checkpoint,
                        "extraction",
                        time.perf_counter() - started,
                    )
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue

                extraction_stage = str(state["extraction_stage"])
                if extraction_stage == "skipped":
                    self._release_pdf_recovery_cache(source_digest)
                    record_skip(state["skipped_source"])
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                if extraction_stage == "complete":
                    document = read_json(artifact_root / "document.json")
                    units = read_jsonl(artifact_root / "units.jsonl")
                    if (
                        not isinstance(document, dict)
                        or not units
                        or any(
                            unit.get("document_id") != document.get("document_id")
                            or unit.get("source_id") != document.get("source_id")
                            for unit in units
                        )
                    ):
                        raise ResearchError(
                            f"Completed extraction artifacts are invalid: {relative}"
                        )
                    self._release_pdf_recovery_cache(source_digest)
                    checkpoint["extracted_source_paths"].append(relative)
                    count_field = (
                        "reused_document_count"
                        if state.get("reused")
                        else "rebuilt_document_count"
                    )
                    checkpoint[count_field] = int(checkpoint.get(count_field) or 0) + 1
                    if not state.get("reused"):
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                elif extraction_stage == "pdf_scan":
                    index = int(state["next_index"])
                    total = int(state["total"])
                    page_batch_size = int(state.get("page_batch_size") or 1)
                    if index < total:
                        page_scans = await extract_source(
                            scan_pdf_pages,
                            source,
                            index,
                            min(page_batch_size, total - index),
                            self._pdf_recovery_cache(source_digest),
                        )
                        if page_scans is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        expected_indices = list(
                            range(index, min(index + page_batch_size, total))
                        )
                        if [int(scan["page_index"]) for scan in page_scans] != (
                            expected_indices
                        ):
                            raise ResearchError(
                                f"PDF scan batch was incomplete: {relative}"
                            )
                        for page_scan in page_scans:
                            page_index = int(page_scan["page_index"])
                            atomic_write_json(
                                artifact_root / "page-scans" / f"{page_index:08d}.json",
                                page_scan,
                                fsync_parent=False,
                            )
                        fsync_directories([artifact_root / "page-scans"])
                        state["next_index"] = expected_indices[-1] + 1
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                    else:
                        # Every page is scanned; the native-reader cache has no
                        # further use once the scan batches are stored.
                        self._release_pdf_recovery_cache(source_digest)
                        state["extraction_stage"] = "pdf_prepare"
                        state["next_index"] = 0
                    atomic_write_json(state_path, state)
                elif extraction_stage == "pdf_prepare":
                    page_scans = [
                        read_json(artifact_root / "page-scans" / f"{index:08d}.json")
                        for index in range(int(state["total"]))
                    ]
                    prepared = await extract_source(
                        prepare_scanned_pdf,
                        source,
                        checkpoint["source_digests"][relative],
                        page_scans,
                    )
                    if prepared is None:
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    document, repeated = prepared
                    atomic_write_json(artifact_root / "document.json", document)
                    state["repeated_margins"] = repeated
                    state["extraction_stage"] = "pdf_pages"
                    state["next_index"] = 0
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    atomic_write_json(state_path, state)
                elif extraction_stage == "docling_pages":
                    index = int(state["next_index"])
                    total = int(state["total"])
                    page_batch_size = int(state.get("page_batch_size") or 1)
                    end_index = min(index + page_batch_size, total)
                    document_path = artifact_root / "document.json"
                    existing_document = (
                        read_json(document_path) if document_path.exists() else None
                    )
                    docling_env = getattr(
                        self, "_docling_env", None
                    ) or resolve_docling_env(self.config)
                    if docling_env is None:
                        raise DoclingEnvironmentError(
                            "the managed Docling environment was not resolved "
                            "before the PDF batches began."
                        )
                    prepared_batch = await extract_source(
                        docling_pdf_batch,
                        source,
                        existing_document,
                        index,
                        end_index,
                        digest=source_digest,
                        offline=self.config.offline,
                        model_cache_root=self.config.model_cache_root,
                        docling_env=docling_env,
                        total_pages=total,
                    )
                    if prepared_batch is None:
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    pages, document, batch_empty, batch_removals, batch_diagnostics = (
                        prepared_batch
                    )
                    if not document_path.exists():
                        atomic_write_json(document_path, document)
                    accumulated_diagnostics = {
                        str(key): int(value)
                        for key, value in (
                            state.get("docling_diagnostics") or {}
                        ).items()
                    }
                    for key, value in batch_diagnostics.items():
                        accumulated_diagnostics[key] = int(
                            accumulated_diagnostics.get(key) or 0
                        ) + int(value)
                    state["docling_diagnostics"] = accumulated_diagnostics
                    # Write one batch file per page in the range, empty included,
                    # so finalize reads a complete sequence whatever the worker
                    # emitted for a blank page.
                    for page_index in range(index, end_index):
                        atomic_write_jsonl(
                            artifact_root / "unit-batches" / f"{page_index:08d}.jsonl",
                            pages.get(page_index, []),
                            fsync_parent=False,
                        )
                    fsync_directories([artifact_root / "unit-batches"])
                    state["removals"] = merge_removal_counts(
                        state.get("removals"), batch_removals
                    )
                    state["empty_units"] = int(state.get("empty_units") or 0) + int(
                        batch_empty
                    )
                    state["next_index"] = end_index
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    if end_index >= total:
                        state["extraction_stage"] = "finalize"
                        state["next_index"] = 0
                    atomic_write_json(state_path, state)
                elif extraction_stage in {"pdf_pages", "epub_items"}:
                    index = int(state["next_index"])
                    total = int(state["total"])
                    if index >= total:
                        state["extraction_stage"] = "finalize"
                        atomic_write_json(state_path, state)
                        continue
                    document = read_json(artifact_root / "document.json")
                    if extraction_stage == "pdf_pages":
                        page_batch_size = int(state.get("page_batch_size") or 1)
                        end_index = min(index + page_batch_size, total)
                        page_scans = [
                            read_json(
                                artifact_root / "page-scans" / f"{page_index:08d}.json"
                            )
                            for page_index in range(index, end_index)
                        ]
                        page_batches = await extract_source(
                            extract_scanned_pdf_pages,
                            source,
                            document,
                            page_scans,
                            list(state.get("repeated_margins") or []),
                        )
                        if page_batches is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        if [page_index for page_index, *_rest in page_batches] != list(
                            range(index, end_index)
                        ):
                            raise ResearchError(
                                f"PDF extraction batch was incomplete: {relative}"
                            )
                    else:
                        extracted = await extract_source(
                            extract_epub_spine_item,
                            source,
                            document,
                            index,
                        )
                        if extracted is None:
                            if budget_expired():
                                return self._in_progress_result(checkpoint)
                            continue
                        batch, empty, removed = extracted
                        page_batches = [(index, batch, empty, removed)]
                    for page_index, batch, empty, removed in page_batches:
                        state["removals"] = merge_removal_counts(
                            state.get("removals"),
                            removed,
                        )
                        atomic_write_jsonl(
                            artifact_root / "unit-batches" / f"{page_index:08d}.jsonl",
                            batch,
                            fsync_parent=False,
                        )
                        state["empty_units"] = int(state.get("empty_units") or 0) + int(
                            empty
                        )
                    state["next_index"] = (
                        end_index if extraction_stage == "pdf_pages" else index + 1
                    )
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    atomic_write_json(state_path, state, fsync_parent=False)
                    fsync_directories([artifact_root / "unit-batches", artifact_root])
                elif extraction_stage == "finalize":
                    units = []
                    for index in range(int(state["total"])):
                        units.extend(
                            read_jsonl(
                                artifact_root / "unit-batches" / f"{index:08d}.jsonl"
                            )
                        )
                    document = read_json(artifact_root / "document.json")
                    document["empty_units"] = int(state.get("empty_units") or 0)
                    if state.get("docling_diagnostics"):
                        document["docling_diagnostics"] = {
                            str(key): int(value)
                            for key, value in state["docling_diagnostics"].items()
                        }
                    removals = {
                        key: int(value)
                        for key, value in (state.get("removals") or {}).items()
                        if key in REMOVAL_FIELDS
                    }
                    retained = await extract_source(
                        screen_source_units,
                        source,
                        document,
                        units,
                        removals=removals,
                        maximum_unclean_percent=self._effective_pdf_loss_limit(),
                    )
                    if retained is None:
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    atomic_write_json(artifact_root / "document.json", document)
                    atomic_write_jsonl(artifact_root / "units.jsonl", retained)
                    units = retained
                    state["extraction_stage"] = "complete"
                    atomic_write_json(state_path, state)
                    checkpoint["extracted_source_paths"].append(relative)
                    checkpoint["rebuilt_document_count"] = (
                        int(checkpoint.get("rebuilt_document_count") or 0) + 1
                    )
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                else:
                    raise ResearchError(
                        f"Unsupported staged extraction phase: {extraction_stage}"
                    )
                self._add_phase_time(
                    checkpoint,
                    "extraction",
                    time.perf_counter() - started,
                )
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "chunking":
                if snapshot is not None and not snapshot.chunks_by_document:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_headers=self.config.settings.chunk_headers,
                        maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
                        load_units=True,
                        load_chunks=True,
                        load_vectors=False,
                        embedding=self.config.settings.embedding_facts,
                    )
                completed = set(checkpoint["chunked_source_paths"])
                relative = next(
                    (path for path in selected_paths if path not in completed),
                    None,
                )
                if relative is None:
                    checkpoint["phase"] = "assembly"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                artifact_root = self._source_artifact_root(staging_root, relative)
                state = read_json(artifact_root / "state.json")
                document = read_json(artifact_root / "document.json")
                units = units_cache.get(relative)
                if units is None:
                    units = read_jsonl(artifact_root / "units.jsonl")
                    units_cache[relative] = units
                started = time.perf_counter()
                if bool(state.get("reused")) and snapshot is not None:
                    chunks = [
                        dict(item)
                        for item in snapshot.chunks_by_document[
                            str(document["document_id"])
                        ]
                    ]
                    discarded_empty = 0
                    discarded_symbol_only = 0
                    discarded_corrupt = 0
                    checkpoint["reused_chunk_count"] = int(
                        checkpoint.get("reused_chunk_count") or 0
                    ) + len(chunks)
                    state["chunking_stage"] = "complete"
                else:
                    chunked_count = int(state.get("chunked_unit_count") or 0)
                    if "chunking_work_total" not in state:
                        state["chunking_work_total"] = len(units)
                        checkpoint["chunking_work_total"] = int(
                            checkpoint.get("chunking_work_total") or 0
                        ) + len(units)
                        atomic_write_json(artifact_root / "state.json", state)
                        self._write_checkpoint(staging_root, checkpoint)
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    if chunked_count < len(units):
                        batch_units = units[
                            chunked_count : chunked_count
                            + self.config.settings.chunk_batch_units
                        ]
                        requested_ids = {str(unit["id"]) for unit in batch_units}
                        chunking_root = artifact_root / "chunking"
                        input_path = chunking_root / "unit.jsonl"
                        working_path = artifact_root / "raw-chunks.jsonl"
                        # A handoff file is rewritten before every call and
                        # deleted afterwards, so it needs visibility, not
                        # durability.
                        write_handoff_jsonl(input_path, batch_units)
                        working_path.unlink(missing_ok=True)
                        await self.ultrarag.chunk(
                            input_path,
                            working_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        raw_by_unit: defaultdict[str, list[dict[str, Any]]] = (
                            defaultdict(list)
                        )
                        for raw in read_jsonl(working_path):
                            raw_by_unit[str(raw.get("doc_id") or "")].append(raw)
                        working_path.unlink(missing_ok=True)
                        input_path.unlink(missing_ok=True)
                        unknown = [
                            unit_id
                            for unit_id in raw_by_unit
                            if unit_id not in requested_ids
                        ]
                        if unknown:
                            raise ResearchError(
                                "UltraRAG returned an unknown extraction unit: "
                                f"{unknown[0]}"
                            )
                        # Every unit keeps its own durable output file and its own
                        # index, written in unit order, so the ordinal assignment
                        # in `_enrich_chunks` and the resume boundary per unit are
                        # the same as when each unit had its own call.
                        for position, unit in enumerate(batch_units):
                            atomic_write_jsonl(
                                chunking_root / f"{chunked_count + position:08d}.jsonl",
                                raw_by_unit.get(str(unit["id"]), []),
                                fsync_parent=False,
                            )
                        state["chunked_unit_count"] = chunked_count + len(batch_units)
                        checkpoint["chunking_work_completed"] = int(
                            checkpoint.get("chunking_work_completed") or 0
                        ) + len(batch_units)
                        atomic_write_json(
                            artifact_root / "state.json",
                            state,
                            fsync_parent=False,
                        )
                        # The whole batch is on disk before the checkpoint that
                        # claims it is complete.
                        fsync_directories([chunking_root, artifact_root])
                        self._add_phase_time(
                            checkpoint,
                            "chunking",
                            time.perf_counter() - started,
                        )
                        self._write_checkpoint(staging_root, checkpoint)
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    raw_chunks: list[dict[str, Any]] = []
                    for index in range(len(units)):
                        raw_chunks.extend(
                            read_jsonl(
                                artifact_root / "chunking" / f"{index:08d}.jsonl"
                            )
                        )
                    (
                        chunks,
                        discarded_empty,
                        discarded_symbol_only,
                        discarded_corrupt,
                    ) = _enrich_chunks(
                        raw_chunks,
                        units,
                        [document],
                        headers=self.config.settings.chunk_headers,
                    )
                    audited = await _atomic_to_thread(
                        _record_embedding_token_counts,
                        chunks,
                        self.dense.embedding_token_counts,
                        maximum_tokens=(self.config.settings.embedding_maximum_tokens),
                    )
                    checkpoint["dense_token_audit"] = (
                        "counted" if audited else "unavailable"
                    )
                    checkpoint["rebuilt_chunk_count"] = int(
                        checkpoint.get("rebuilt_chunk_count") or 0
                    ) + len(chunks)
                    state["chunking_stage"] = "complete"
                atomic_write_jsonl(artifact_root / "chunks.jsonl", chunks)
                state["discarded_empty_chunks"] = discarded_empty
                state["discarded_symbol_only_chunks"] = discarded_symbol_only
                state["discarded_corrupt_chunks"] = discarded_corrupt
                atomic_write_json(artifact_root / "state.json", state)
                self._add_phase_time(
                    checkpoint,
                    "chunking",
                    time.perf_counter() - started,
                )
                checkpoint["chunked_source_paths"].append(relative)
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "assembly":
                started = time.perf_counter()
                documents: list[dict[str, Any]] = []
                discarded_empty_chunks = 0
                discarded_symbol_only_chunks = 0
                discarded_corrupt_chunks = 0
                for relative in selected_paths:
                    if relative in {
                        item["source_relative_path"]
                        for item in checkpoint.get("skipped_sources", [])
                    }:
                        continue
                    artifact_root = self._source_artifact_root(staging_root, relative)
                    document = read_json(artifact_root / "document.json")
                    state = read_json(artifact_root / "state.json")
                    if not isinstance(document, dict) or not isinstance(state, dict):
                        raise ResearchError("Invalid staged source artifacts")
                    documents.append(document)
                    discarded_empty_chunks += int(
                        state.get("discarded_empty_chunks") or 0
                    )
                    discarded_symbol_only_chunks += int(
                        state.get("discarded_symbol_only_chunks") or 0
                    )
                    discarded_corrupt_chunks += int(
                        state.get("discarded_corrupt_chunks") or 0
                    )
                extracted_path = staging_root / "corpus" / "extracted-units.jsonl"
                if not documents:
                    raise ResearchError(
                        "No clean sources remain; the current generation is unchanged. "
                        + " ".join(
                            item["reason"]
                            for item in checkpoint.get("skipped_sources", [])
                        )
                    )
                chunks_path = staging_root / "chunks" / "chunks.jsonl"

                record_counts = {"units": 0, "chunks": 0}

                def staged_records(
                    filename: str,
                    count_key: str,
                    counts: dict[str, int],
                ) -> Iterator[dict[str, Any]]:
                    for source_relative_path in selected_paths:
                        if source_relative_path in {
                            item["source_relative_path"]
                            for item in checkpoint.get("skipped_sources", [])
                        }:
                            continue
                        path = (
                            self._source_artifact_root(
                                staging_root,
                                source_relative_path,
                            )
                            / filename
                        )
                        for record in iter_jsonl(path):
                            counts[count_key] += 1
                            yield record

                atomic_write_jsonl(
                    extracted_path,
                    staged_records("units.jsonl", "units", record_counts),
                )
                atomic_write_jsonl(
                    chunks_path,
                    staged_records("chunks.jsonl", "chunks", record_counts),
                )
                batches_root = staging_root / "work" / "chunk-batches"
                offset = 0
                batch: list[dict[str, Any]] = []
                for chunk in iter_jsonl(chunks_path):
                    batch.append(chunk)
                    if len(batch) < self.config.settings.embedding_batch_size:
                        continue
                    atomic_write_jsonl(
                        batches_root / f"{offset:012d}.jsonl",
                        batch,
                        fsync_parent=False,
                    )
                    offset += len(batch)
                    batch = []
                if batch:
                    atomic_write_jsonl(
                        batches_root / f"{offset:012d}.jsonl",
                        batch,
                        fsync_parent=False,
                    )
                if batches_root.is_dir():
                    # One directory fsync covers every batch before embedding
                    # resumes from any of them.
                    fsync_directories([batches_root])
                checkpoint["document_count"] = len(documents)
                checkpoint["extraction_unit_count"] = record_counts["units"]
                checkpoint["chunk_count"] = record_counts["chunks"]
                checkpoint["discarded_empty_chunk_count"] = discarded_empty_chunks
                checkpoint["discarded_symbol_only_chunk_count"] = (
                    discarded_symbol_only_chunks
                )
                checkpoint["discarded_corrupt_chunk_count"] = discarded_corrupt_chunks
                checkpoint["excluded_corrupt_unit_count"] = sum(
                    int(document.get("excluded_corrupt_unit_count") or 0)
                    for document in documents
                )
                self._add_phase_time(
                    checkpoint,
                    "assembly",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "embedding"
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "embedding":
                if snapshot is not None and not snapshot.vectors_by_text:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_headers=self.config.settings.chunk_headers,
                        maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
                        load_units=False,
                        load_chunks=True,
                        load_vectors=True,
                        embedding=self.config.settings.embedding_facts,
                    )
                offset = int(checkpoint.get("embedded_chunk_count") or 0)
                total = int(checkpoint["chunk_count"])
                if offset >= total:
                    checkpoint["phase"] = "vector_assembly"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                batch = read_jsonl(
                    staging_root / "work" / "chunk-batches" / f"{offset:012d}.jsonl"
                )
                vectors = np.empty(
                    (len(batch), self.config.settings.embedding_dimension),
                    dtype=np.float32,
                )
                missing_positions: list[int] = []
                missing_texts: list[str] = []
                # What is embedded is the chunk's own text, header included. Vector
                # reuse is keyed on the canonical passage, and a header cannot be
                # reconstructed from that key, so reuse suspends when the two
                # differ: recomputing a vector is always correct, while reusing
                # another chunk's vector is not.
                batch_texts = [_embedding_text(chunk) for chunk in batch]
                reuse_keys = [_chunk_text(chunk) for chunk in batch]
                reusable_vectors = (
                    await _atomic_to_thread(
                        snapshot.vectors_for_texts,
                        reuse_keys,
                    )
                    if snapshot is not None
                    and not force_recompute
                    and reuse_keys == batch_texts
                    else {}
                )
                reused_count = 0
                for index, text in enumerate(batch_texts):
                    reusable_vector = reusable_vectors.get(text)
                    if reusable_vector is None:
                        missing_positions.append(index)
                        missing_texts.append(text)
                    else:
                        vectors[index] = reusable_vector
                        reused_count += 1
                started = time.perf_counter()
                if missing_texts:
                    created = await _atomic_to_thread(
                        self.dense.embed_texts,
                        missing_texts,
                    )
                    if created.shape != (
                        len(missing_positions),
                        self.config.settings.embedding_dimension,
                    ):
                        raise ResearchError(
                            "Dense backend returned an invalid embedding matrix"
                        )
                    for position, vector in zip(
                        missing_positions,
                        created,
                        strict=True,
                    ):
                        vectors[position] = vector
                batch_path = (
                    staging_root / "work" / "vector-batches" / f"{offset:012d}.npy"
                )
                self._save_vector_batch(batch_path, vectors)
                self._add_phase_time(
                    checkpoint,
                    "embedding",
                    time.perf_counter() - started,
                )
                checkpoint["embedded_chunk_count"] = offset + len(batch)
                checkpoint["reused_vector_count"] = (
                    int(checkpoint.get("reused_vector_count") or 0) + reused_count
                )
                checkpoint["created_vector_count"] = int(
                    checkpoint.get("created_vector_count") or 0
                ) + len(missing_positions)
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "vector_assembly":
                started = time.perf_counter()
                total = int(checkpoint["chunk_count"])
                vectors_path = staging_root / "portable" / "embeddings.npy"
                await _atomic_to_thread(
                    self._assemble_vector_batches,
                    vectors_path,
                    staging_root / "work" / "vector-batches",
                    total,
                    self.config.settings.embedding_batch_size,
                    self.config.settings.embedding_dimension,
                )
                self._add_phase_time(
                    checkpoint,
                    "vector_assembly",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "bm25_indexing"
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "bm25_indexing":
                bm25_index_path = staging_root / "indexes" / "bm25"
                shutil.rmtree(bm25_index_path, ignore_errors=True)
                started = time.perf_counter()
                # The build points the gateway's one retriever at the staging
                # index, so a search beside it waits and reloads its own.
                async with self._retriever_lock:
                    self._loaded_generation = None
                    await self.ultrarag.build_bm25(
                        staging_root / "chunks" / "chunks.jsonl",
                        bm25_index_path,
                        language=self.config.settings.bm25_stopwords_language,
                    )
                self._add_phase_time(
                    checkpoint,
                    "bm25_indexing",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "dense_indexing"
                checkpoint["dense_indexed_count"] = 0
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase in {"dense_indexing", "qdrant_indexing"}:
                if portable_vectors is None:
                    portable_vectors = np.load(
                        staging_root / "portable" / "embeddings.npy",
                        allow_pickle=False,
                        mmap_mode="r",
                    )
                total = int(checkpoint["chunk_count"])
                backend_name = str(checkpoint.get("dense_backend") or "")
                if backend_name not in DENSE_INDEX_PATHS:
                    # A checkpoint resumed from before the backend was recorded
                    # restarts this phase cleanly with the selected backend.
                    backend_name = self._select_build_dense_backend(total)
                    checkpoint["dense_backend"] = backend_name
                    checkpoint["dense_index_relative"] = DENSE_INDEX_PATHS[backend_name]
                    checkpoint["dense_indexed_count"] = 0
                    self._write_checkpoint(staging_root, checkpoint)
                backend = self._dense_backends[backend_name]
                dense_index_path = staging_root / DENSE_INDEX_PATHS[backend_name]
                offset = int(checkpoint.get("dense_indexed_count") or 0)
                if offset == 0:
                    shutil.rmtree(dense_index_path, ignore_errors=True)
                    await _atomic_to_thread(
                        backend.initialize_index,
                        dense_index_path,
                        self.config.settings.embedding_dimension,
                    )
                if offset < total:
                    batch = read_jsonl(
                        staging_root / "work" / "chunk-batches" / f"{offset:012d}.jsonl"
                    )
                    started = time.perf_counter()
                    await _atomic_to_thread(
                        backend.upload_index_batch,
                        batch,
                        dense_index_path,
                        portable_vectors[offset : offset + len(batch)],
                        offset=offset,
                    )
                    self._add_phase_time(
                        checkpoint,
                        "dense_indexing",
                        time.perf_counter() - started,
                    )
                    checkpoint["dense_indexed_count"] = offset + len(batch)
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                dense_metadata = await _atomic_to_thread(
                    backend.finalize_index,
                    dense_index_path,
                    expected_count=total,
                    dimension=self.config.settings.embedding_dimension,
                )
                checkpoint["dense_metadata"] = dense_metadata
                checkpoint["phase"] = "source_revalidation"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "source_revalidation":
                revalidated = checkpoint["revalidation_digests"]
                pending = [
                    record
                    for record in checkpoint["source_inventory"]
                    if record["source_relative_path"] not in revalidated
                ]
                if pending:
                    record = pending[0]
                    source = sources_by_path[str(record["source_relative_path"])]
                    started = time.perf_counter()
                    try:
                        digest, stat_identity = await _atomic_to_thread(
                            _hash_with_stable_stat,
                            source.path,
                        )
                    except OSError as exc:
                        raise _SourceChangedDuringIngest(
                            "A source became unavailable during final validation"
                        ) from exc
                    revalidated[source.source_relative_path] = digest
                    checkpoint["revalidation_stats"][source.source_relative_path] = (
                        stat_identity
                    )
                    self._add_phase_time(
                        checkpoint,
                        "source_revalidation",
                        time.perf_counter() - started,
                    )
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                if revalidated != checkpoint["source_digests"]:
                    raise _SourceChangedDuringIngest(
                        "Source bytes changed while ingestion was in progress"
                    )
                try:
                    final_scan = scan_sources(self.config)
                except SourcePolicyError as exc:
                    raise _SourceChangedDuringIngest(str(exc)) from exc
                if _source_inventory(final_scan) != checkpoint["source_inventory"]:
                    raise _SourceChangedDuringIngest(
                        "The source collection changed while ingestion was in progress"
                    )
                if value_fingerprint(self._source_exclusions()) != checkpoint.get(
                    "source_exclusion_revision"
                ):
                    raise _SourceChangedDuringIngest(
                        "Source exclusions changed while ingestion was in progress"
                    )
                selected_now = self._load_current_optional()
                selected_generation_id = (
                    str(selected_now[1]["generation_id"])
                    if selected_now is not None
                    else None
                )
                if selected_generation_id != checkpoint.get("baseline_generation_id"):
                    raise _SourceChangedDuringIngest(
                        "The selected generation changed while ingestion was in progress"
                    )
                checkpoint["phase"] = "finalizing"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "finalizing":
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed after its final ingestion hash"
                    )
                documents = [
                    read_json(
                        self._source_artifact_root(staging_root, relative)
                        / "document.json"
                    )
                    for relative in selected_paths
                    if relative
                    not in {
                        item["source_relative_path"]
                        for item in checkpoint.get("skipped_sources", [])
                    }
                ]
                content_kinds: Counter[str] = Counter()
                withheld_chunk_count = 0
                withheld_chunk_reasons: Counter[str] = Counter()
                audited_chunk_count = 0
                dense_truncated_chunk_count = 0
                maximum_embedding_token_count = 0
                for item in iter_jsonl(staging_root / "chunks" / "chunks.jsonl"):
                    content_kinds[str(item.get("content_kind") or "prose")] += 1
                    reasons = text_corruption_reasons(str(item.get("contents") or ""))
                    if reasons:
                        withheld_chunk_count += 1
                        withheld_chunk_reasons.update(reasons)
                    token_count = item.get("embedding_token_count")
                    if isinstance(token_count, int) and not isinstance(
                        token_count, bool
                    ):
                        audited_chunk_count += 1
                        maximum_embedding_token_count = max(
                            maximum_embedding_token_count, token_count
                        )
                        if item.get("dense_truncated"):
                            dense_truncated_chunk_count += 1
                content_kind_counts = dict(sorted(content_kinds.items()))
                extraction_unit_count = int(checkpoint["extraction_unit_count"])
                chunk_count = int(checkpoint["chunk_count"])
                phase_timings = {
                    key: round(float(value), 6)
                    for key, value in checkpoint["phase_timings_seconds"].items()
                }
                loss_totals = _document_loss_totals(documents)
                substantive_characters = loss_totals["substantive_character_count"]
                build_metrics = {
                    "skipped_sources": checkpoint.get("skipped_sources", []),
                    "skipped_source_count": len(checkpoint.get("skipped_sources", [])),
                    "forced": force_recompute,
                    "resumed": bool(checkpoint.get("resume_count")),
                    "reused_document_count": int(
                        checkpoint.get("reused_document_count") or 0
                    ),
                    "rebuilt_document_count": int(
                        checkpoint.get("rebuilt_document_count") or 0
                    ),
                    "reused_chunk_count": int(
                        checkpoint.get("reused_chunk_count") or 0
                    ),
                    "rebuilt_chunk_count": int(
                        checkpoint.get("rebuilt_chunk_count") or 0
                    ),
                    "reused_vector_count": int(
                        checkpoint.get("reused_vector_count") or 0
                    ),
                    "created_vector_count": int(
                        checkpoint.get("created_vector_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    **{
                        key: sum(int(document.get(key) or 0) for document in documents)
                        for key in (
                            "excluded_corrupt_passage_count",
                            "excluded_symbol_only_unit_count",
                            "excluded_symbol_only_passage_count",
                            "cleaned_passage_count",
                            "pdf_text_recovery_pages",
                            "recovered_pdf_blocks",
                        )
                    },
                    **loss_totals,
                    "unclean_character_rate": (
                        round(
                            loss_totals["discarded_corrupt_character_count"]
                            / substantive_characters,
                            6,
                        )
                        if substantive_characters
                        else 0.0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "withheld_chunk_count": withheld_chunk_count,
                    "withheld_chunk_reasons": dict(
                        sorted(withheld_chunk_reasons.items())
                    ),
                    "dense_token_audit": (
                        "counted"
                        if audited_chunk_count == chunk_count
                        else ("partial" if audited_chunk_count else "unavailable")
                    ),
                    "dense_audited_chunk_count": audited_chunk_count,
                    "dense_truncated_chunk_count": dense_truncated_chunk_count,
                    "embedding_maximum_tokens": self.config.settings.embedding_maximum_tokens,
                    "maximum_embedding_token_count": maximum_embedding_token_count,
                    "phase_timings_seconds": phase_timings,
                }
                records = source_records()
                dense_metadata = dict(checkpoint["dense_metadata"])
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "extraction_policy_version": EXTRACTION_POLICY_VERSION,
                    "cleaning_policy_version": CLEANING_POLICY_VERSION,
                    "artifact_policy_version": ARTIFACT_POLICY_VERSION,
                    "generation_id": checkpoint["build_id"],
                    "project_id": self.config.project_id,
                    "project_name": self.config.project_name,
                    "created_at": _utc_now(),
                    "source_directory": self.config.source_root.relative_to(
                        self.config.project_root
                    ).as_posix(),
                    "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                    "ignored_extensions": scan.ignored_extensions,
                    # Reviewed metadata is a portable read-time overlay. This
                    # revision is observational and never fingerprints derived
                    # index contents or checkpoint compatibility.
                    "metadata_revision": value_fingerprint(self._metadata()),
                    "metadata_storage_policy": METADATA_STORAGE_POLICY,
                    "source_exclusion_revision": checkpoint[
                        "source_exclusion_revision"
                    ],
                    "retrieval_policy_fingerprint": (self.retrieval_policy_fingerprint),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
                    "document_count": len(documents),
                    "extraction_unit_count": extraction_unit_count,
                    "chunk_count": chunk_count,
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "passage_cleaning": {
                        # The configured budget is the identity a rebuild reads;
                        # the effective budget is what the shared gate actually
                        # applied for the selected backend. Both readers use the
                        # project budget. Reuse compares the configured value, so a
                        # legacy manifest with only that key stays compatible.
                        "maximum_unclean_percent": self.config.settings.maximum_unclean_percent,
                        "effective_maximum_unclean_percent": (
                            self._effective_pdf_loss_limit()
                        ),
                        "pdf_backend": self._pdf_backend,
                    },
                    "extraction_backend": pdf_backend_record(
                        self.config.settings.pdf_backend, self.config
                    ),
                    "chunking": {
                        "backend": "Chonkie token chunker (UltraRAG-compatible)",
                        "tokenizer": "gpt2",
                        "unit": "tokens",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "headers": self.config.settings.chunk_headers,
                    },
                    "retrieval": {
                        "default_method": DEFAULT_RETRIEVAL_METHOD,
                        "available_methods": sorted(RETRIEVAL_METHODS),
                        "bm25": {
                            "backend": "bm25s (UltraRAG-compatible)",
                            # The setting the index was built with, not a constant.
                            "language": self.config.settings.bm25_stopwords_language,
                            "tokenizer": "default",
                        },
                        "dense": dense_metadata,
                        "fusion": {
                            "method": "weighted_reciprocal_rank_fusion",
                            "rrf_k": self.config.settings.rrf_k,
                            "bm25_weight": self.config.settings.bm25_weight,
                            "dense_weight": self.config.settings.dense_weight,
                            "minimum_candidates": self.config.settings.minimum_candidates,
                            "maximum_candidates": self.config.settings.maximum_candidates,
                        },
                        "relevance_gates": {
                            "bm25_requires_query_token_overlap": True,
                            "dense_minimum_cosine_similarity": (
                                self.config.settings.dense_minimum_cosine_similarity
                            ),
                        },
                        "reranker": {
                            "optional": True,
                            "runtime": "FastEmbed ONNX Runtime (CPU)",
                            "model": self.config.reranker_model,
                            "model_revision": _reranker_revision(
                                self.config.reranker_model
                            ),
                            "maximum_candidates": self.config.settings.rerank_max_candidates,
                        },
                    },
                    "documents": documents,
                    "source_files": records,
                    "excluded_sources": [
                        {
                            "source_id": source.source_id,
                            "source_relative_path": source.source_relative_path,
                            "source_path": source.project_relative_path,
                            **exclusions[source.source_relative_path],
                        }
                        for source in scan.selected
                        if source.source_relative_path in exclusions
                    ],
                    "files": {
                        "extracted_units": "corpus/extracted-units.jsonl",
                        "chunks": "chunks/chunks.jsonl",
                        "bm25_index": "indexes/bm25",
                        "dense_index": str(
                            checkpoint.get("dense_index_relative")
                            or DENSE_INDEX_PATHS[QDRANT_BACKEND_NAME]
                        ),
                        "portable_embeddings": "portable/embeddings.npy",
                    },
                    "build_metrics": build_metrics,
                }
                generation_root = self.config.generations_root / str(
                    checkpoint["build_id"]
                )
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed while final artifacts were assembled"
                    )
                atomic_write_json(staging_root / "manifest.json", manifest)
                await self._validate_generation_for_activation(
                    staging_root,
                    manifest,
                )
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed while final indexes were validated"
                    )
                select_generation = not build_metrics.get("skipped_sources")
                atomic_write_json(
                    self._pending_activation_path,
                    {
                        "schema_version": PENDING_ACTIVATION_VERSION,
                        "project_id": self.config.project_id,
                        "build_id": checkpoint["build_id"],
                        "select": select_generation,
                        "identity": checkpoint["identity"],
                        "baseline_generation_id": checkpoint.get(
                            "baseline_generation_id"
                        ),
                        "source_inventory": checkpoint["source_inventory"],
                        "source_digests": checkpoint["source_digests"],
                        "revalidation_stats": checkpoint["revalidation_stats"],
                        "source_exclusion_revision": checkpoint[
                            "source_exclusion_revision"
                        ],
                        "created_at": _utc_now(),
                    },
                )
                shutil.rmtree(staging_root / "work", ignore_errors=True)
                os.replace(staging_root, generation_root)
                fsync_directory(self.config.generations_root)
                fsync_directory(self.config.staging_root)
                if select_generation:
                    atomic_write_json(
                        self.config.current_path,
                        {
                            "schema_version": 1,
                            "generation_id": checkpoint["build_id"],
                        },
                    )
                (generation_root / "checkpoint.json").unlink()
                fsync_directory(generation_root)
                self._pending_activation_path.unlink(missing_ok=True)
                fsync_directory(self._pending_activation_path.parent)
                self._loaded_generation = None
                return {
                    "status": "ready" if select_generation else "partial",
                    "generation_changed": select_generation,
                    "current_generation_id": (
                        checkpoint["build_id"]
                        if select_generation
                        else checkpoint.get("baseline_generation_id")
                    ),
                    **(
                        {
                            "manual_selection_command": project_command(
                                self.config.project_root,
                                "generations",
                                "--use",
                                str(checkpoint["build_id"]),
                            )
                        }
                        if not select_generation
                        else {}
                    ),
                    "generation_id": checkpoint["build_id"],
                    "generation_root": str(generation_root),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
                    "excluded_sources": [
                        {
                            "source_id": source.source_id,
                            "source_relative_path": source.source_relative_path,
                            "source_path": source.project_relative_path,
                            **exclusions[source.source_relative_path],
                        }
                        for source in scan.selected
                        if source.source_relative_path in exclusions
                    ],
                    "document_count": len(documents),
                    "pdf_count": sum(item["format"] == "pdf" for item in documents),
                    "epub_count": sum(item["format"] == "epub" for item in documents),
                    "mobi_count": sum(item["format"] == "mobi" for item in documents),
                    "extraction_unit_count": extraction_unit_count,
                    "chunk_count": chunk_count,
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "ignored_extensions": scan.ignored_extensions,
                    "empty_units": sum(int(item["empty_units"]) for item in documents),
                    "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                    "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
                    "embedding_model": self.config.settings.embedding_model,
                    "embedding_model_revision": self.config.settings.embedding_model_revision,
                    **build_metrics,
                }

            raise ResearchError(f"Unsupported ingestion checkpoint phase: {phase}")

    def _generation_root(self, generation_id: str) -> Path:
        """Resolve one generation directory by name, refusing any other name.

        The name arrives from a caller, so it is matched against the pattern the
        builder writes and refused otherwise. Joining a caller's string to a
        directory and resolving the result would work too, and would be the wrong
        shape: this keeps a name that is not a build identifier from ever
        becoming a path, and the check is one line rather than a containment test
        whose edge cases are the interesting part.
        """

        if not re.fullmatch(GENERATION_ID_PATTERN, generation_id):
            raise ResearchError(
                f"{generation_id!r} is not a generation id; the id is the "
                "directory name, such as 20260930T191235Z-45608dc5"
            )
        return self.config.generations_root / generation_id

    def _pending_activation_build_id(self) -> str | None:
        """The build a pending activation is waiting to move into place, if any."""

        if not self._pending_activation_path.exists():
            return None
        try:
            journal = read_json(self._pending_activation_path)
        except (OSError, ValueError):
            return None
        if not isinstance(journal, dict):
            return None
        build_id = journal.get("build_id")
        return build_id if isinstance(build_id, str) and build_id else None

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        """Search a retained generation instead of the current one.

        A build always moves forward, so a reader who wants the previous
        generation has no way back to it. This is that way: it validates the
        target's artifacts and both indexes exactly as an activation does, then
        moves the pointer, so a rollback that lands on a damaged generation
        fails instead of bricking every read surface.
        """

        async with self._operation():
            root = self._generation_root(generation_id)
            if not root.is_dir() or root.is_symlink():
                raise ResearchError(f"No retained generation named {generation_id}")
            current = self._load_current_optional()
            if current is not None and root.name == str(
                current[1].get("generation_id")
            ):
                return {
                    "status": "unchanged",
                    "generation_id": generation_id,
                    "message": f"Generation {generation_id} is already the one in use.",
                }
            pending = self._pending_activation_build_id()
            if pending is not None and pending != generation_id:
                raise ResearchError(
                    f"A pending activation is waiting to switch to {pending}. Run "
                    "ingest to finish or supersede it before moving the pointer "
                    "somewhere else, or it will move the pointer again."
                )
            try:
                manifest = read_json(root / "manifest.json")
            except (OSError, ValueError) as exc:
                raise ResearchError(
                    f"Generation {generation_id} has no readable manifest"
                ) from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("generation_id") != generation_id
            ):
                raise ResearchError(
                    f"Generation {generation_id} does not match its own manifest"
                )
            try:
                await self._validate_generation_for_activation(root, manifest)
            except ResearchError as exc:
                raise ResearchError(
                    f"Generation {generation_id} failed validation and was not "
                    f"selected: {exc}"
                ) from exc
            atomic_write_json(
                self.config.current_path,
                {"schema_version": 1, "generation_id": generation_id},
            )
            fsync_directory(self.config.generations_root)
            self._loaded_generation = None
            return {
                "status": "changed",
                "generation_id": generation_id,
                "previous_generation_id": (
                    None if current is None else str(current[1].get("generation_id"))
                ),
                "effective_immediately": True,
                "message": (
                    f"Search now reads generation {generation_id}. The corpus is "
                    "unchanged; run ingest to build the current sources again."
                ),
            }

    async def remove_generation(
        self,
        generation_id: str,
        *,
        confirm: str,
    ) -> dict[str, Any]:
        """Delete a retained generation that is not the one search reads.

        Removal is permanent and the space is only recoverable by rebuilding, so
        the caller has to repeat the id: a generation named by a listing and
        removed by a copy of that listing is a mistake that cannot be walked
        back, and the second name is the only check that the reader meant this
        one. The current generation is refused because deleting it makes every
        read surface fail, and a generation a pending activation names is
        refused because that activation is about to move it into place.
        """

        async with self._operation():
            root = self._generation_root(generation_id)
            if confirm != generation_id:
                raise ResearchError(
                    f"Pass --confirm {generation_id} to remove that generation"
                )
            if not root.is_dir() or root.is_symlink():
                raise ResearchError(f"No retained generation named {generation_id}")
            current = self._load_current_optional()
            if current is not None and root.name == str(
                current[1].get("generation_id")
            ):
                raise ResearchError(
                    f"Generation {generation_id} is the one search reads, so it "
                    "cannot be removed. Point the project at another with "
                    "'research-rag generations --use OTHER_ID', then remove this "
                    "one."
                )
            pending = self._pending_activation_build_id()
            if pending == generation_id:
                raise ResearchError(
                    f"Generation {generation_id} is waiting to become the one "
                    "search reads. Run ingest to finish it, or remove it after a "
                    "superseding build."
                )
            try:
                file_count, size_bytes = directory_statistics(root)
            except OSError as exc:
                raise ResearchError(
                    f"Generation {generation_id} could not be measured: {exc}"
                ) from exc
            # A read takes no project lock, so one that resolved this generation
            # before it stopped being current may still be reading it.
            await self._wait_for_readers(generation_id)
            try:
                shutil.rmtree(root)
            except OSError as exc:
                raise ResearchError(
                    f"Generation {generation_id} could not be removed: {exc}"
                ) from exc
            fsync_directory(self.config.generations_root)
            return {
                "status": "removed",
                "generation_id": generation_id,
                "freed_file_count": file_count,
                "freed_bytes": size_bytes,
                "recoverable_by": "ingest",
                "message": (
                    f"Removed {generation_id} and freed {size_bytes} bytes. The "
                    "originals are untouched; rebuilding costs one ingestion."
                ),
            }

    async def ingest(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force_recompute: bool = False,
        work_budget_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Create or refresh the generation that search reads.

        An omitted parameter comes from the merged settings, where the
        chunking values live; a caller that passes one gets it checked here.
        """

        settings = self.config.settings
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        if work_budget_seconds is None:
            work_budget_seconds = settings.work_budget_seconds
        size_setting = SETTINGS_BY_KEY["chunking.size"]
        if not size_setting.minimum <= chunk_size <= size_setting.maximum:
            raise ResearchError(
                "chunk_size must be between "
                f"{size_setting.minimum:g} and {size_setting.maximum:g} GPT-2 tokens"
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )
        budget_setting = SETTINGS_BY_KEY["ingestion.work_budget_seconds"]
        # An engine caller may ask for 0 to checkpoint at the next boundary; the
        # configured setting keeps the higher floor, since a whole build with a
        # budget below it would only ever complete one unit per call.
        if not 0 <= work_budget_seconds <= budget_setting.maximum:
            raise ResearchError(
                f"work_budget_seconds must be between 0 and {budget_setting.maximum:g}"
            )

        async with self._operation():
            deadline = time.perf_counter() + work_budget_seconds
            self._cleanup_invalid_staging_roots()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if not scan.selected:
                raise ResearchError(
                    f"No PDF, EPUB or MOBI sources found beneath {self.config.source_root}"
                )
            exclusions = self._source_exclusions()
            selected = tuple(
                source
                for source in scan.selected
                if source.source_relative_path not in exclusions
            )
            if not selected:
                raise ResearchError(
                    "All discovered PDF, EPUB and MOBI sources are excluded; include at "
                    "least one source before ingesting"
                )
            current = self._load_current_optional()
            self._sync_source_catalog(
                scan,
                current,
                exclusions=exclusions,
            )
            baseline_generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )
            exclusion_revision = value_fingerprint(exclusions)
            inventory = _source_inventory(scan)
            # Resolve the managed Docling environment before any fingerprint or
            # checkpoint identity is computed, so a first-use install cannot move
            # the build's identity mid-build. An offline run with no environment
            # fails closed here rather than during a source batch.
            if self._pdf_backend == PDF_BACKEND_DOCLING and any(
                source.extension == ".pdf" for source in selected
            ):
                self._docling_env = await _atomic_to_thread(
                    ensure_docling_env, self.config
                )
            identity = _checkpoint_identity(
                project_id=self.config.project_id,
                inventory=inventory,
                exclusion_revision=exclusion_revision,
                baseline_generation_id=baseline_generation_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_headers=self.config.settings.chunk_headers,
                force_recompute=force_recompute,
                embedding=self.config.settings.embedding_facts,
                maximum_unclean_percent=self.config.settings.maximum_unclean_percent,
                pdf_backend_fingerprint=self._pdf_backend_fingerprint,
            )
            recovered = await self._recover_pending_activation(
                expected_identity=identity,
                scan=scan,
                exclusions=exclusions,
                current=current,
            )
            if recovered is not None:
                return recovered
            existing = self._load_ingestion_checkpoint()
            superseded_build: dict[str, Any] | None = None
            if existing is not None and existing[1].get("identity") != identity:
                superseded_checkpoint = dict(existing[1])
                superseded_build = {
                    "build_id": str(superseded_checkpoint.get("build_id") or ""),
                    "phase": str(superseded_checkpoint.get("phase") or ""),
                    "reason": (
                        "The sources or the ingestion parameters changed since that "
                        "build was checkpointed, so it could not be resumed and was "
                        "discarded."
                    ),
                }
                self._discard_checkpoint(
                    existing[0],
                    superseded_checkpoint,
                    reason=str(superseded_build["reason"]),
                )
                existing = None
            if existing is None:
                staging_root, checkpoint = self._create_checkpoint(
                    scan=scan,
                    exclusions=exclusions,
                    exclusion_revision=exclusion_revision,
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                )
            else:
                staging_root, checkpoint = existing
                self._remove_uncommitted_files(staging_root)
                checkpoint["resume_count"] = (
                    int(checkpoint.get("resume_count") or 0) + 1
                )
                self._write_checkpoint(staging_root, checkpoint)

            try:
                result = await self._advance_ingestion(
                    staging_root=staging_root,
                    checkpoint=checkpoint,
                    scan=scan,
                    exclusions=exclusions,
                    current=current,
                    deadline=deadline,
                )
            except _SourceChangedDuringIngest as exc:
                self._release_pdf_recovery_caches()
                self._discard_checkpoint(
                    staging_root,
                    checkpoint,
                    reason=str(exc),
                )
                fresh_scan = scan_sources(self.config)
                fresh_exclusions = self._source_exclusions()
                if not fresh_scan.selected:
                    raise ResearchError(
                        "All PDF, EPUB and MOBI sources disappeared during ingestion"
                    ) from exc
                if not any(
                    source.source_relative_path not in fresh_exclusions
                    for source in fresh_scan.selected
                ):
                    raise ResearchError(
                        "All discovered sources became excluded during ingestion"
                    ) from exc
                self._sync_source_catalog(
                    fresh_scan,
                    current,
                    exclusions=fresh_exclusions,
                )
                fresh_root, fresh = self._create_checkpoint(
                    scan=fresh_scan,
                    exclusions=fresh_exclusions,
                    exclusion_revision=value_fingerprint(fresh_exclusions),
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                )
                del fresh_root
                return self._in_progress_result(
                    fresh,
                    message=(
                        "Source bytes changed during ingestion; the incompatible "
                        "checkpoint was replaced. Call ingest again."
                    ),
                )
            except asyncio.CancelledError:
                self._release_pdf_recovery_caches()
                checkpoint["last_interruption"] = "cancelled"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except TimeoutError:
                self._release_pdf_recovery_caches()
                checkpoint["last_interruption"] = "timeout"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except Exception as exc:
                self._release_pdf_recovery_caches()
                pending = (
                    read_json(self._pending_activation_path)
                    if self._pending_activation_path.is_file()
                    else None
                )
                if not (
                    isinstance(pending, dict)
                    and pending.get("build_id") == checkpoint.get("build_id")
                ):
                    self._discard_checkpoint(
                        staging_root,
                        checkpoint,
                        reason=str(exc),
                    )
                raise
            if superseded_build is not None:
                # Progress that goes backwards has to be explained: a caller looping
                # on ingest otherwise reads a discarded build as its own mistake
                # and retries the same call, which is what it will do again.
                result["superseded_build"] = superseded_build
                result["message"] = (
                    f"Build {superseded_build['build_id']} was discarded because the "
                    "corpus changed since it was checkpointed, so this build starts "
                    "from the beginning. " + str(result.get("message") or "")
                ).strip()
            if result.get("status") != "in_progress":
                # A source releases its own cache when it finishes, so this only
                # clears one left by an interruption that returned no result.
                self._release_pdf_recovery_caches()
            return result
