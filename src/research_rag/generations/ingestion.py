from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus.docling_env import (
    DoclingEnv,
    ensure_docling_env,
)
from ..corpus.extraction import (
    REMOVAL_FIELDS,
    ExtractionError,
    docling_pdf_batch,
    empty_removal_counts,
    extract_epub_spine_item,
    extract_mobi,
    merge_removal_counts,
    pdf_page_count,
    prepare_epub_extraction,
    screen_source_units,
)
from ..corpus.pdf_backend import (
    docling_backend_record_from_env,
    effective_maximum_unclean_percent,
    legacy_custom_backend_record,
    normalize_pdf_backend,
    pdf_backend_fingerprint,
    pdf_backend_record,
)
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
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    DEFAULT_RETRIEVAL_METHOD,
    EXTRACTION_POLICY_VERSION,
    GENERATION_ID_PATTERN,
    INGESTION_IDENTITY_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    REFLOWABLE_EXTRACTION_POLICY_VERSION,
    RETRIEVAL_METHODS,
    SCHEMA_VERSION,
    ResearchError,
    _utc_now,
    reflowable_document_is_current,
    value_fingerprint,
)
from ..project.rerankers import _reranker_revision
from ..project.settings import SETTINGS_BY_KEY
from ..retrieval.artifact_lookup import (
    LOOKUP_RELATIVE_PATH,
    ArtifactLookup,
    ArtifactLookupError,
    chunk_text,
    ensure_artifact_lookup,
)
from ..retrieval.dense import DENSE_INDEX_PATHS, EXACT_BACKEND_NAME, QDRANT_BACKEND_NAME
from ..storage.durable_io import (
    StorageError,
    _atomic_to_thread,
    atomic_file,
    atomic_write_json,
    atomic_write_jsonl,
    directory_statistics,
    file_identity,
    fsync_directories,
    fsync_directory,
    fsync_file,
    iter_jsonl,
    read_json,
    read_jsonl,
    replace_durably,
    write_handoff_jsonl,
)
from .generation import (
    ReuseSnapshot,
    generation_artifacts_are_valid,
    load_reuse_snapshot,
    selection_matches,
)
from .staging import (
    _checkpoint_identity,
    _embedding_text,
    _enrich_chunks,
    _generation_id,
    _hash_with_stable_stat,
    _pdf_batch_count,
    _record_embedding_token_counts,
    _source_inventory,
    _source_work_key,
)

INGESTION_CHECKPOINT_VERSION = 1
PENDING_ACTIVATION_VERSION = 1

LOGGER = logging.getLogger(__name__)

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


def _staged_chunk_census(chunks_path: Path) -> dict[str, Any]:
    """Count a build's chunks by kind, withheld reason, and dense-token audit."""

    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    withheld = audited = truncated = maximum = 0
    for item in iter_jsonl(chunks_path):
        kinds[str(item.get("content_kind") or "prose")] += 1
        found = text_corruption_reasons(str(item.get("contents") or ""))
        if found:
            withheld += 1
            reasons.update(found)
        token_count = item.get("embedding_token_count")
        if isinstance(token_count, int) and not isinstance(token_count, bool):
            audited += 1
            maximum = max(maximum, token_count)
            truncated += bool(item.get("dense_truncated"))
    return {
        "content_kind_counts": dict(sorted(kinds.items())),
        "withheld_chunk_count": withheld,
        "withheld_chunk_reasons": dict(sorted(reasons.items())),
        "audited_chunk_count": audited,
        "dense_truncated_chunk_count": truncated,
        "maximum_embedding_token_count": maximum,
    }


def _selected_sources(
    scan: SourceScan,
    exclusions: Mapping[str, Any],
    held: frozenset[str],
    selection: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """The sources a build's generation will hold, and those it carries unread.

    A build of every source reads each one that is not excluded, and carries an
    excluded one the selected generation already holds: an exclusion filters what
    a search returns and never takes data out of a generation. A build of named
    sources reads those and carries the rest of what is held; one that takes
    sources out carries everything held but them. A source whose file is gone
    leaves every build.
    """

    named = set(selection["source_paths"]) if selection is not None else set()
    mode = selection["mode"] if selection is not None else None
    selected: list[str] = []
    for source in scan.selected:
        relative = source.source_relative_path
        if mode is None:
            keep = relative not in exclusions or relative in held
        elif mode == "ingest":
            keep = relative in named or relative in held
        else:
            keep = relative in held and relative not in named
        if keep:
            selected.append(relative)
    carried = [
        relative
        for relative in selected
        if (relative in exclusions if mode is None else relative not in named)
    ]
    return selected, carried


def _chunking_outputs(directory: Path) -> list[tuple[int, int, Path]]:
    """A source's staged chunker outputs, as (first unit, end unit, path), in order.

    A batch writes one file named for the units it covers. A build staged before
    batches shared a file holds one per unit, named for that unit alone, and
    resumes from the same listing.
    """

    outputs: list[tuple[int, int, Path]] = []
    for path in directory.glob("*.jsonl") if directory.is_dir() else ():
        start, _, end = path.stem.partition("-")
        if start.isdigit() and (end.isdigit() or not end):
            outputs.append((int(start), int(end) if end else int(start) + 1, path))
    return sorted(outputs)


class _SourceChangedDuringIngest(RuntimeError):
    pass


class _BuildCannotResume(ResearchError):
    """The staged build is unusable, so its checkpoint is discarded."""


# A failure that is not proven to be unusable state keeps the checkpoint, so the
# next call resumes. The same failure in the same phase this many times running
# discards it, so a deterministic fault cannot wedge every later call.
MAXIMUM_REPEATED_FAILURES = 3


#: What a source's chunking discarded, as its staged state records it.
_DISCARDED_CHUNK_KEYS = (
    "discarded_empty_chunks",
    "discarded_symbol_only_chunks",
    "discarded_corrupt_chunks",
)
_NO_DISCARDED_CHUNKS = dict.fromkeys(_DISCARDED_CHUNK_KEYS, 0)


@dataclass
class _Build:
    """One ingest call's hold on a staged build.

    The checkpoint is the durable state a later call resumes from. The reuse
    snapshot, the units read for chunking, and the mapped vectors belong to this
    call and are loaded again by the next one.
    """

    staging_root: Path
    checkpoint: dict[str, Any]
    scan: SourceScan
    exclusions: dict[str, dict[str, str]]
    current: tuple[Path, dict[str, Any]] | None
    deadline: float
    snapshot: ReuseSnapshot | None = None
    # The reuse axes this call has loaded into the snapshot.
    loaded_axes: set[str] = field(default_factory=set)
    units_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    portable_vectors: np.ndarray[Any, np.dtype[np.float32]] | None = None

    @property
    def chunk_size(self) -> int:
        return int(self.checkpoint["parameters"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return int(self.checkpoint["parameters"]["chunk_overlap"])

    @property
    def force_recompute(self) -> bool:
        return bool(self.checkpoint["parameters"]["force_recompute"])

    @cached_property
    def sources_by_path(self) -> dict[str, SourceFile]:
        return {source.source_relative_path: source for source in self.scan.selected}

    @cached_property
    def selected_paths(self) -> list[str]:
        return [str(item) for item in self.checkpoint["selected_source_paths"]]

    @cached_property
    def carried(self) -> frozenset[str]:
        """Sources the build keeps exactly as the selected generation holds them."""

        return frozenset(self.checkpoint.get("carried_source_paths") or ())

    def forced(self, relative: str) -> bool:
        """Whether this source is read again with every reuse disabled."""

        return self.force_recompute and relative not in self.carried

    @cached_property
    def forced_source_ids(self) -> frozenset[str]:
        return frozenset(
            self.sources_by_path[relative].source_id
            for relative in self.selected_paths
            if self.forced(relative) and relative in self.sources_by_path
        )

    def budget_expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def add(self, key: str, amount: int = 1) -> None:
        """Add to one of the checkpoint's running counts."""

        self.checkpoint[key] = int(self.checkpoint.get(key) or 0) + amount

    def add_time(self, phase: str, started: float) -> None:
        timings = self.checkpoint.setdefault("phase_timings_seconds", {})
        timings[phase] = float(timings.get(phase) or 0.0) + (
            time.perf_counter() - started
        )

    def next_source(self, done: str) -> str | None:
        """The first selected source the checkpoint's `done` list does not hold."""

        completed = set(self.checkpoint[done])
        return next(
            (path for path in self.selected_paths if path not in completed), None
        )

    def recorded_sources(
        self, documents: Mapping[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The source records a generation keeps: what it holds, and the excluded.

        A source a generation does not hold and does not exclude is left out, so
        `status` reports it as not yet ingested. A source carried from bytes that
        have changed since keeps the record those bytes had, so `status` reports
        it as modified.
        """

        digests = self.checkpoint["source_digests"]
        selected = set(self.selected_paths)
        records: list[dict[str, Any]] = []
        for record in self.checkpoint["source_inventory"]:
            relative = record["source_relative_path"]
            excluded = relative in self.exclusions
            if relative not in selected and not excluded:
                continue
            kept = {**record, "sha256": str(digests[relative])}
            document = documents.get(relative)
            if (
                document is not None
                and str(document.get("sha256")) != kept["sha256"]
                and self.snapshot is not None
                and relative in self.snapshot.source_files
            ):
                kept = dict(self.snapshot.source_files[relative])
            records.append({**kept, "included": not excluded})
        return records

    def excluded_sources(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": source.source_id,
                "source_relative_path": source.source_relative_path,
                "source_path": source.project_relative_path,
                **self.exclusions[source.source_relative_path],
            }
            for source in self.scan.selected
            if source.source_relative_path in self.exclusions
        ]

    def revalidation_stats_match(self, *, require_complete: bool = False) -> bool:
        recorded = self.checkpoint.get("revalidation_stats") or {}
        if not isinstance(recorded, dict):
            return False
        if require_complete and set(recorded) != set(self.sources_by_path):
            return False
        try:
            return all(
                file_identity(self.sources_by_path[relative].path) == identity
                for relative, identity in recorded.items()
                if relative in self.sources_by_path
            ) and set(recorded).issubset(self.sources_by_path)
        except OSError:
            return False


@dataclass
class _StagedSource:
    """One source on its way through extraction, and where its artifacts are staged."""

    relative: str
    source: SourceFile
    root: Path
    digest: str
    started: float
    state: dict[str, Any] | None = None

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def document_path(self) -> Path:
        return self.root / "document.json"


class IngestionWorkflow:
    @property
    def _pdf_backend(self) -> str:
        return normalize_pdf_backend(self.config.settings.pdf_backend)

    @property
    def _pdf_backend_fingerprint(self) -> str:
        # The checkpoint identity uses the backend's pinned *spec* fingerprint,
        # not the installed component versions. The spec is stable before and
        # after a first-use install, so an environment built mid-build does not
        # move the identity and orphan a resumable checkpoint. The installed
        # versions describe the artifact and live in the manifest instead.
        return pdf_backend_fingerprint(self._pdf_backend, None)

    @staticmethod
    def _document_with_provenance(
        document: dict[str, Any],
        source: SourceFile,
    ) -> dict[str, Any]:
        """A reused document's record, with truthful per-document provenance.

        A selected generation built before per-document provenance existed carries
        no backend on its documents. An unchanged PDF is passed through whole, so
        the new generation records it as the retired bundled reader rather than
        claiming Docling converted it; the manifest's own backend is only the
        target for new work. A document that already names its backend keeps it,
        and a non-PDF has none to add.
        """

        if source.extension != ".pdf":
            return document
        if isinstance(document.get("extraction_backend"), dict):
            return document
        return {**document, "extraction_backend": legacy_custom_backend_record()}

    def _effective_pdf_loss_limit(self) -> float:
        """The loss budget the shared gate applies for the selected backend."""

        return effective_maximum_unclean_percent(
            self._pdf_backend,
            self.config.settings.maximum_unclean_percent,
        )

    def _select_build_dense_backend(self) -> str:
        """Choose the dense backend for a new generation.

        `auto` builds the exact scan at every size. Qdrant local mode loads every
        point each time a client opens and scores by brute force, so it is never
        the faster choice; it is built only when `dense.backend` names it.
        """

        if self.config.dense_backend == "qdrant":
            return QDRANT_BACKEND_NAME
        return EXACT_BACKEND_NAME

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
                        # Docling makes one pass over the page batches, then
                        # finalizes.
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
                    elif source_format in {"epub", "mobi"}:
                        # An EPUB is read one spine item at a time; a MOBI is
                        # read whole, so its work is the read and the finalize.
                        items = unit_total if source_format == "epub" else 0
                        source_total = items + 2
                        if extraction_stage == "epub_items" and source_format == "epub":
                            source_completed = next_index + 1
                        elif extraction_stage == "finalize":
                            source_completed = items + 1
                        elif extraction_stage == "complete":
                            source_completed = source_total - int(
                                relative not in extracted_paths
                            )
                        else:
                            return
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

    def _keep_failed_build(self, root: Path, error: Exception) -> bool:
        """Record a failure on the committed checkpoint, unless the build is unusable.

        The checkpoint on disk is re-read rather than the one in memory, so a
        counter the failed step changed before its own commit is not written. The
        same failure in the same phase `MAXIMUM_REPEATED_FAILURES` times running
        is not kept, so a deterministic fault ends instead of repeating.
        """

        if isinstance(error, _BuildCannotResume | StorageError | ArtifactLookupError):
            return False
        try:
            committed = read_json(root / "checkpoint.json")
            if not isinstance(committed, dict):
                return False
            failure = {
                "phase": str(committed.get("phase") or ""),
                "error": f"{type(error).__name__}: {error}"[:500],
            }
            previous = committed.get("last_failure")
            repeats = 1
            if isinstance(previous, dict) and all(
                previous.get(key) == value for key, value in failure.items()
            ):
                repeats = int(previous.get("repeats") or 0) + 1
            if repeats >= MAXIMUM_REPEATED_FAILURES:
                return False
            committed["last_failure"] = {
                **failure,
                "repeats": repeats,
                "failed_at": _utc_now(),
            }
            self._remove_uncommitted_files(root)
            self._write_checkpoint(root, committed)
        except (OSError, StorageError):
            # A disk that refused the work may refuse this write too; the last
            # committed checkpoint still resumes, and only the count is lost.
            pass
        return True

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
            # The probe points the lexical backend's one retriever at this
            # generation, so a search running beside the build waits for it and
            # reloads its own.
            async with self._retriever_lock:
                self._loaded_generation = None
                await self.lexical.initialize_bm25(
                    root / str(manifest["files"]["chunks"]),
                    root / str(manifest["files"]["bm25_index"]),
                    language=self.config.settings.bm25_stopwords_language,
                )
                chunks_path = root / str(manifest["files"]["chunks"])
                probe = next(iter_jsonl(chunks_path))["contents"]
                bm25_probe = await self.lexical.search_bm25(str(probe), 1)
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
                    file_identity(source.path)
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
        held: frozenset[str],
        selection: dict[str, Any] | None,
    ) -> tuple[Path, dict[str, Any]]:
        build_id = _generation_id()
        inventory = _source_inventory(scan)
        selected, carried = _selected_sources(scan, exclusions, held, selection)
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
            selection=selection,
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
                **({"selection": selection} if selection is not None else {}),
            },
            "source_inventory": inventory,
            "selected_source_paths": selected,
            "carried_source_paths": carried,
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

    async def _ensure_docling_environment(self) -> DoclingEnv:
        """The managed Docling environment, resolved once, only when needed.

        A build that reuses every unchanged PDF never provisions Docling: this is
        reached only when a new or changed PDF actually extracts. Resolving it in
        a thread keeps a first-use install off the event loop, and caching it on
        the service means one conversion build resolves it once.
        """

        env = getattr(self, "_docling_env", None)
        if env is None:
            env = await _atomic_to_thread(ensure_docling_env, self.config)
            self._docling_env = env
        return env

    @staticmethod
    def _rescope_pdf_pages_for_environment(
        artifact_root: Path,
        state: dict[str, Any],
        state_path: Path,
        checkpoint: dict[str, Any],
    ) -> None:
        """Discard a source's unfinished pages produced by a different environment.

        A ready managed environment can change its installed component versions
        while the intended spec stays put. The pages already staged for this source
        were converted by the previous environment, so appending the current
        environment's batches would mix provenance inside one document. Only this
        source's unfinished work is discarded: its staged document and page batches
        are removed and its page cursor is reset, while the finished sources, the
        selected generation, and every source byte stay untouched. The cursor is
        reset durably before the artifacts are removed, so a crash inside the
        removal resumes into this same idempotent path.

        Completed selected-document passthrough is unaffected: a reused source
        never reaches the staged Docling path.
        """

        page_batch_size = int(state.get("page_batch_size") or 1)
        completed_batches = _pdf_batch_count(
            int(state.get("next_index") or 0), page_batch_size
        )
        checkpoint["extraction_work_completed"] = max(
            0,
            int(checkpoint.get("extraction_work_completed") or 0) - completed_batches,
        )
        state["next_index"] = 0
        state["removals"] = empty_removal_counts()
        state["empty_units"] = 0
        state.pop("docling_diagnostics", None)
        atomic_write_json(state_path, state)
        (artifact_root / "document.json").unlink(missing_ok=True)
        shutil.rmtree(artifact_root / "unit-batches", ignore_errors=True)
        fsync_directory(artifact_root)
        LOGGER.warning(
            "ingest: build %s discarded unfinished pages of a partially converted "
            "source because the Docling environment changed; it is converted again "
            "from the first page",
            checkpoint.get("build_id"),
        )

    @staticmethod
    def _save_vector_batch(
        path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> None:
        with atomic_file(path, "wb") as handle:
            np.save(handle, vectors, allow_pickle=False)

    @staticmethod
    def _assemble_vector_batches(
        path: Path,
        batches_root: Path,
        total: int,
        dimension: int,
    ) -> None:
        """Assemble portable vectors without allocating a second full matrix.

        The batches are read as the embedding phase named them, by offset, so a
        batch size changed between two ingest calls cannot misplace one.
        """

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
            offset = 0
            for batch_path in sorted(batches_root.glob("*.npy")):
                if int(batch_path.stem) != offset:
                    raise _BuildCannotResume(
                        f"Vector batch {batch_path.name} does not start at row {offset}"
                    )
                batch = np.load(batch_path, allow_pickle=False, mmap_mode="r")
                vectors[offset : offset + len(batch)] = batch
                offset += len(batch)
            if offset != total:
                raise _BuildCannotResume(
                    f"Vector batches hold {offset} rows, not {total}"
                )
            vectors.flush()
            del vectors
            vectors = None
            fsync_file(temporary)
            replace_durably(temporary, path)
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

    def _assemble_staged_corpus(
        self, staging_root: Path, checkpoint: dict[str, Any]
    ) -> None:
        """Write the build's corpus, chunks, and embedding batches from staged sources.

        One pass over every staged source, so it runs off the event loop. The
        counts it derives are recorded on the checkpoint the caller commits.
        """

        skipped = {
            item["source_relative_path"]
            for item in checkpoint.get("skipped_sources", [])
        } | set(checkpoint.get("dropped_source_paths") or ())
        sources = [
            self._source_artifact_root(staging_root, relative)
            for relative in checkpoint["selected_source_paths"]
            if relative not in skipped
        ]
        documents: list[dict[str, Any]] = []
        discarded: Counter[str] = Counter()
        for artifact_root in sources:
            document = read_json(artifact_root / "document.json")
            state = read_json(artifact_root / "state.json")
            if not isinstance(document, dict) or not isinstance(state, dict):
                raise _BuildCannotResume("Invalid staged source artifacts")
            documents.append(document)
            for key in _DISCARDED_CHUNK_KEYS:
                discarded[key] += int(state.get(key) or 0)
        if not documents:
            raise _BuildCannotResume(
                "No clean sources remain; the current generation is unchanged. "
                + " ".join(
                    item["reason"] for item in checkpoint.get("skipped_sources", [])
                )
            )
        counts = {"units": 0, "chunks": 0}

        def staged_records(filename: str, kind: str) -> Iterator[dict[str, Any]]:
            for artifact_root in sources:
                for record in iter_jsonl(artifact_root / filename):
                    counts[kind] += 1
                    yield record

        chunks_path = staging_root / "chunks" / "chunks.jsonl"
        atomic_write_jsonl(
            staging_root / "corpus" / "extracted-units.jsonl",
            staged_records("units.jsonl", "units"),
        )
        atomic_write_jsonl(chunks_path, staged_records("chunks.jsonl", "chunks"))
        batches_root = staging_root / "work" / "chunk-batches"
        batch_size = self.config.settings.embedding_batch_size
        offset = 0
        batch: list[dict[str, Any]] = []
        for chunk in iter_jsonl(chunks_path):
            batch.append(chunk)
            if len(batch) == batch_size:
                atomic_write_jsonl(
                    batches_root / f"{offset:012d}.jsonl", batch, fsync_parent=False
                )
                offset += len(batch)
                batch = []
        if batch:
            atomic_write_jsonl(
                batches_root / f"{offset:012d}.jsonl", batch, fsync_parent=False
            )
        if batches_root.is_dir():
            # One directory fsync covers every batch before embedding resumes
            # from any of them.
            fsync_directories([batches_root])
        checkpoint["document_count"] = len(documents)
        checkpoint["extraction_unit_count"] = counts["units"]
        checkpoint["chunk_count"] = counts["chunks"]
        checkpoint["discarded_empty_chunk_count"] = discarded["discarded_empty_chunks"]
        checkpoint["discarded_symbol_only_chunk_count"] = discarded[
            "discarded_symbol_only_chunks"
        ]
        checkpoint["discarded_corrupt_chunk_count"] = discarded[
            "discarded_corrupt_chunks"
        ]
        checkpoint["excluded_corrupt_unit_count"] = sum(
            int(document.get("excluded_corrupt_unit_count") or 0)
            for document in documents
        )

    async def _save(self, build: _Build) -> None:
        await _atomic_to_thread(
            self._write_checkpoint, build.staging_root, build.checkpoint
        )

    def _pause_if_due(self, build: _Build) -> dict[str, Any] | None:
        """The in-progress answer once the call's budget is spent, else None."""

        return (
            self._in_progress_result(build.checkpoint)
            if build.budget_expired()
            else None
        )

    async def _commit(self, build: _Build) -> dict[str, Any] | None:
        """Save the step just taken, then stop if the budget is spent."""

        await self._save(build)
        return self._pause_if_due(build)

    async def _enter_phase(self, build: _Build, phase: str) -> None:
        build.checkpoint["phase"] = phase
        await self._save(build)

    async def _reuse_snapshot(self, build: _Build, *axes: str) -> ReuseSnapshot:
        """The selected generation as reuse reads it, with the named axes loaded."""

        settings = self.config.settings
        return await _atomic_to_thread(
            load_reuse_snapshot,
            build.current,
            schema_version=SCHEMA_VERSION,
            extraction_policy_version=EXTRACTION_POLICY_VERSION,
            cleaning_policy_version=CLEANING_POLICY_VERSION,
            artifact_policy_version=ARTIFACT_POLICY_VERSION,
            project_id=self.config.project_id,
            chunk_size=build.chunk_size,
            chunk_overlap=build.chunk_overlap,
            chunk_headers=settings.chunk_headers,
            maximum_unclean_percent=settings.maximum_unclean_percent,
            load_units="units" in axes,
            load_chunks="chunks" in axes,
            load_vectors="vectors" in axes,
            embedding=settings.embedding_facts,
        )

    async def _load_axes(self, build: _Build, *axes: str) -> None:
        """Load the reuse axes a phase reads, once per call."""

        if build.snapshot is not None and not build.loaded_axes.issuperset(axes):
            build.snapshot = await self._reuse_snapshot(build, *axes)
            build.loaded_axes.update(axes)

    async def _record_skip(self, build: _Build, record: dict[str, str]) -> None:
        relative = record["source_relative_path"]
        skipped = build.checkpoint.setdefault("skipped_sources", [])
        if not any(item["source_relative_path"] == relative for item in skipped):
            skipped.append(record)
        for key in ("extracted_source_paths", "chunked_source_paths"):
            if relative not in build.checkpoint[key]:
                build.checkpoint[key].append(relative)
        await self._save(build)

    async def _extract_or_skip(
        self,
        build: _Build,
        staged: _StagedSource,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one source-local extraction step, or omit the source when it fails.

        Only typed failures from the source-local extractor are omissions. Artifact
        writes, hashing, chunking and index failures remain fatal. None means the
        source was omitted and the omission is already saved.
        """

        try:
            return await _atomic_to_thread(operation, *args, **kwargs)
        except ExtractionError as exc:
            record = {
                "source_relative_path": staged.relative,
                "reason": str(exc),
                # OCR is performed outside this app, so the remedy is the same
                # for every source-local failure: repair or replace the file.
                "remedy": "Repair or replace this source, then run research-rag ingest.",
            }
            # Commit the source decision first so a lost checkpoint write can
            # recover it without repeating partially extracted work.
            await _atomic_to_thread(
                atomic_write_json,
                staged.state_path,
                {
                    **(staged.state or {}),
                    "extraction_stage": "skipped",
                    "skipped_source": record,
                },
            )
            build.add_time("extraction", staged.started)
            await self._record_skip(build, record)
            return None

    async def _advance_ingestion(
        self,
        *,
        staging_root: Path,
        checkpoint: dict[str, Any],
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        current: tuple[Path, dict[str, Any]] | None,
        deadline: float,
    ) -> dict[str, Any]:
        """Advance the build from its checkpoint until it ends or the budget is spent.

        Each phase handler takes one durable step and returns the call's answer, or
        None to take the next step.
        """

        build = _Build(staging_root, checkpoint, scan, exclusions, current, deadline)
        if build.force_recompute and not build.carried:
            LOGGER.debug("ingest: force_recompute disables every reuse axis")
        else:
            build.snapshot = await self._reuse_snapshot(build)
        if (
            checkpoint.get("revalidation_stats")
            and not build.revalidation_stats_match()
        ):
            raise _SourceChangedDuringIngest(
                "A source changed after its final ingestion hash"
            )
        handlers = {
            "source_hashing": self._hash_next_source,
            "extraction": self._extract_next_source,
            "chunking": self._chunk_next_source,
            "assembly": self._assemble_corpus,
            "embedding": self._embed_next_batch,
            "vector_assembly": self._assemble_vectors,
            "bm25_indexing": self._index_bm25,
            "dense_indexing": self._index_next_dense_batch,
            "qdrant_indexing": self._index_next_dense_batch,
            "source_revalidation": self._revalidate_next_source,
            "finalizing": self._finalize_build,
        }
        logged_phase: str | None = None
        while True:
            phase = str(checkpoint["phase"])
            # One bounded line per phase entry, re-logged after a resume. No
            # source or passage text is included.
            if phase != logged_phase:
                LOGGER.debug(
                    "ingest: build %s phase %s", checkpoint.get("build_id"), phase
                )
                logged_phase = phase
            handler = handlers.get(phase)
            if handler is None:
                raise _BuildCannotResume(
                    f"Unsupported ingestion checkpoint phase: {phase}"
                )
            result = await handler(build)
            if result is not None:
                return result

    async def _hash_next_source(self, build: _Build) -> dict[str, Any] | None:
        """Hash the next source; once all are hashed, answer a no-op or start a build."""

        digests = build.checkpoint["source_digests"]
        record = next(
            (
                record
                for record in build.checkpoint["source_inventory"]
                if record["source_relative_path"] not in digests
            ),
            None,
        )
        if record is None:
            unchanged = await self._unchanged_generation(build)
            if unchanged is None:
                await self._enter_phase(build, "extraction")
            return unchanged
        source = build.sources_by_path[str(record["source_relative_path"])]
        started = time.perf_counter()
        try:
            digest = await _atomic_to_thread(sha256_file, source.path)
        except OSError as exc:
            raise _SourceChangedDuringIngest(
                "A source became unavailable during hashing"
            ) from exc
        build.add_time("source_hashing", started)
        digests[source.source_relative_path] = digest
        return await self._commit(build)

    async def _unchanged_generation(self, build: _Build) -> dict[str, Any] | None:
        """The no-op answer when the selected generation already holds these inputs.

        A no-op means every reuse axis is compatible: extraction units, chunk text,
        and vectors. A model change or a processing-policy change is a genuine
        incompatibility even when the sources are byte-identical, because the new
        generation must store something the old one does not. Ranking policy is
        deliberately absent: it is a runtime decision and never rewrites a
        generation.
        """

        snapshot = build.snapshot
        if snapshot is None:
            return None
        # A carried source is kept as it is, so only the sources a build reads
        # can make it read them again.
        reading = [path for path in build.selected_paths if path not in build.carried]
        reflowable_current = all(
            reflowable_document_is_current(snapshot.documents[path])
            for path in reading
            if path in snapshot.documents
        )
        if not (
            snapshot.units_reusable
            and snapshot.chunks_reusable
            and snapshot.vectors_reusable
            and reflowable_current
        ):
            LOGGER.debug(
                "ingest: building because a reuse axis moved "
                "(units_reusable=%s chunks_reusable=%s vectors_reusable=%s "
                "reflowable_current=%s)",
                snapshot.units_reusable,
                snapshot.chunks_reusable,
                snapshot.vectors_reusable,
                reflowable_current,
            )
            return None
        manifest = snapshot.manifest
        # A build of every source retries the ones the last build could not read.
        retry = "selection" not in build.checkpoint["parameters"] and bool(
            manifest.get("build_metrics", {}).get("skipped_sources")
        )
        digests = build.checkpoint["source_digests"]
        if (
            retry
            or any(build.forced(path) for path in reading)
            or not selection_matches(
                snapshot,
                selected=build.selected_paths,
                digests={path: str(digests[path]) for path in reading},
                metadata_storage_policy=METADATA_STORAGE_POLICY,
            )
        ):
            # A backend switch or adapter drift alone changes how a *new* PDF
            # would be read, never the evidence already stored for an unchanged
            # source. An incremental build passes an unchanged legacy PDF through
            # whole, preserving its provenance, so the backend is not part of this
            # gate; only the sources held, their bytes, and storage policy are.
            return None
        try:
            await self._validate_generation_for_activation(snapshot.root, manifest)
        except ResearchError as exc:
            # Validation failed, so this generation cannot be served as-is and a
            # build is needed. Its per-axis reuse verdict still holds: keep the
            # snapshot so the extraction and chunk text survive, and only the
            # failing axis (usually a corrupt vector file) is recomputed.
            build.checkpoint["reuse_rejection_reason"] = str(exc)
            LOGGER.debug(
                "ingest: selected generation failed validation; "
                "rebuilding while reusing its valid axes: %s",
                exc,
            )
            return None
        self._loaded_generation = str(manifest["generation_id"])
        await _atomic_to_thread(shutil.rmtree, build.staging_root, ignore_errors=True)
        LOGGER.debug(
            "ingest: build %s no-op; selected generation unchanged",
            build.checkpoint.get("build_id"),
        )
        return {
            "status": "unchanged",
            "skipped_sources": manifest.get("build_metrics", {}).get(
                "skipped_sources", []
            ),
            "generation_changed": False,
            "generation_id": manifest["generation_id"],
            "generation_root": str(snapshot.root),
            "source_file_count": len(build.scan.selected),
            "excluded_source_count": len(build.exclusions),
            "document_count": manifest["document_count"],
            "extraction_unit_count": manifest["extraction_unit_count"],
            "chunk_count": manifest["chunk_count"],
            "reused_document_count": manifest["document_count"],
            "rebuilt_document_count": 0,
            "reused_chunk_count": manifest["chunk_count"],
            "rebuilt_chunk_count": 0,
            "reused_vector_count": manifest["chunk_count"],
            "created_vector_count": 0,
            "phase_timings_seconds": build.checkpoint["phase_timings_seconds"],
            "message": "Inputs match the selected generation; no build was needed.",
        }

    async def _extract_next_source(self, build: _Build) -> dict[str, Any] | None:
        """Take the next source one durable step through extraction."""

        await self._load_axes(build, "units", "chunks")
        relative = build.next_source("extracted_source_paths")
        if relative is None:
            return await self._enter_phase(build, "chunking")
        root = self._source_artifact_root(build.staging_root, relative)
        root.mkdir(parents=True, exist_ok=True)
        staged = _StagedSource(
            relative=relative,
            source=build.sources_by_path[relative],
            root=root,
            digest=str(build.checkpoint["source_digests"][relative]),
            started=time.perf_counter(),
        )
        if staged.state_path.exists():
            staged.state = await _atomic_to_thread(read_json, staged.state_path)
        if staged.state is None:
            return await self._start_extraction(build, staged)
        stage = str(staged.state["extraction_stage"])
        if stage == "skipped":
            await self._record_skip(build, staged.state["skipped_source"])
            return self._pause_if_due(build)
        step = {
            "complete": self._accept_extraction,
            "docling_pages": self._extract_pdf_pages,
            "epub_items": self._extract_epub_item,
            "finalize": self._finalize_extraction,
        }.get(stage)
        if step is None:
            raise _BuildCannotResume(f"Unsupported staged extraction phase: {stage}")
        return await step(build, staged)

    async def _end_extraction_step(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        build.add_time("extraction", staged.started)
        return await self._commit(build)

    async def _start_extraction(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        """Pass an unchanged source through whole, or stage its first read."""

        snapshot = build.snapshot
        previous_source = (
            snapshot.source_files.get(staged.relative) if snapshot is not None else None
        )
        previous_document = (
            snapshot.documents.get(staged.relative) if snapshot is not None else None
        )
        document_id = str((previous_document or {}).get("document_id") or "")
        # A carried source is kept from whatever bytes it was built from; a source
        # the build reads is reused only while its bytes are the same.
        carried = staged.relative in build.carried
        same_bytes = (
            previous_source is not None
            and previous_source.get("sha256")
            == build.checkpoint["source_digests"][staged.relative]
        )
        reusable_units = bool(
            snapshot is not None
            and snapshot.units_reusable
            and previous_source is not None
            and previous_document is not None
            and (same_bytes or carried)
            and not build.forced(staged.relative)
            and snapshot.manifest.get("metadata_storage_policy")
            == METADATA_STORAGE_POLICY
            and document_id in snapshot.units_by_document
            # An EPUB or MOBI read under earlier HTML rules is read again; a PDF
            # is unaffected by those rules, and a carried source is kept as it is.
            and (carried or reflowable_document_is_current(previous_document))
        )
        if not reusable_units and carried and staged.relative in build.exclusions:
            # An excluded source is never read. One whose stored extraction this
            # build cannot keep, after a rule or budget change, leaves the
            # generation, and the build names it.
            build.checkpoint.setdefault("dropped_source_paths", []).append(
                staged.relative
            )
            for key in ("extracted_source_paths", "chunked_source_paths"):
                build.checkpoint[key].append(staged.relative)
            return await self._commit(build)
        if reusable_units and snapshot is not None and previous_document is not None:
            # Units and chunks reuse independently. A header or chunk-size change
            # keeps the units but forces the chunks to be derived again from them.
            await self._reuse_extraction(
                build,
                staged,
                snapshot,
                previous_document,
                reuse_chunks=snapshot.chunks_reusable
                and document_id in snapshot.chunks_by_document,
                same_bytes=same_bytes,
            )
        else:
            start = {
                ".pdf": self._start_pdf,
                ".mobi": self._start_mobi,
            }.get(staged.source.extension, self._start_epub)
            if not await start(build, staged):
                return self._pause_if_due(build)
        return await self._end_extraction_step(build, staged)

    async def _reuse_extraction(
        self,
        build: _Build,
        staged: _StagedSource,
        snapshot: ReuseSnapshot,
        previous_document: dict[str, Any],
        *,
        reuse_chunks: bool,
        same_bytes: bool,
    ) -> None:
        """Carry an unchanged source's extraction forward from the selected generation.

        Byte-identical bytes keep the whole artifact, including a PDF's original
        extraction backend. A backend or adapter policy change alters only how a
        *new* PDF would be read, so an unchanged legacy PDF is passed through as
        legacy rather than re-extracted or relabelled as Docling.
        """

        # The same bytes may sit in a touched file; changed bytes keep the stat of
        # the file the document was read from.
        stat = (
            {"size": staged.source.size, "mtime_ns": staged.source.mtime_ns}
            if same_bytes
            else {}
        )
        document = self._document_with_provenance(
            {**previous_document, **stat}, staged.source
        )
        # Each lookup reads fresh records, so they need no copy.
        units = await _atomic_to_thread(
            snapshot.units_by_document.__getitem__,
            str(previous_document.get("document_id") or ""),
        )
        await _atomic_to_thread(atomic_write_json, staged.document_path, document)
        await _atomic_to_thread(atomic_write_jsonl, staged.root / "units.jsonl", units)
        await _atomic_to_thread(
            atomic_write_json,
            staged.state_path,
            {
                "reused": True,
                "reuse_chunks": reuse_chunks,
                "extraction_stage": "complete",
                **_NO_DISCARDED_CHUNKS,
            },
        )
        build.checkpoint["extracted_source_paths"].append(staged.relative)
        build.add("reused_document_count")

    async def _start_pdf(self, build: _Build, staged: _StagedSource) -> bool:
        total = await self._extract_or_skip(
            build, staged, pdf_page_count, staged.source
        )
        if total is None:
            return False
        # Docling is the sole PDF reader. Its environment is resolved lazily, only
        # when a PDF actually extracts.
        batch_size = self.config.settings.pdf_page_batch_size
        state = {
            "reused": False,
            "extraction_stage": "docling_pages",
            "next_index": 0,
            "total": total,
            "page_batch_size": batch_size,
            "removals": empty_removal_counts(),
            "empty_units": 0,
            **_NO_DISCARDED_CHUNKS,
        }
        build.add("extraction_work_total", _pdf_batch_count(total, batch_size) + 2)
        await _atomic_to_thread(atomic_write_json, staged.state_path, state)
        return True

    async def _start_mobi(self, build: _Build, staged: _StagedSource) -> bool:
        prepared = await self._extract_or_skip(
            build, staged, extract_mobi, staged.source, staged.digest
        )
        if prepared is None:
            return False
        document, units = prepared
        document["reflowable_extraction_policy_version"] = (
            REFLOWABLE_EXTRACTION_POLICY_VERSION
        )
        await _atomic_to_thread(atomic_write_json, staged.document_path, document)
        await _atomic_to_thread(
            atomic_write_jsonl, staged.root / "unit-batches" / "00000000.jsonl", units
        )
        state = {
            "reused": False,
            "extraction_stage": "finalize",
            "total": 1,
            "removals": {key: int(document.get(key) or 0) for key in REMOVAL_FIELDS},
            "empty_units": document["empty_units"],
            **_NO_DISCARDED_CHUNKS,
        }
        build.add("extraction_work_total", 2)
        build.add("extraction_work_completed")
        await _atomic_to_thread(atomic_write_json, staged.state_path, state)
        return True

    async def _start_epub(self, build: _Build, staged: _StagedSource) -> bool:
        prepared = await self._extract_or_skip(
            build, staged, prepare_epub_extraction, staged.source, staged.digest
        )
        if prepared is None:
            return False
        document, total = prepared
        document["reflowable_extraction_policy_version"] = (
            REFLOWABLE_EXTRACTION_POLICY_VERSION
        )
        await _atomic_to_thread(atomic_write_json, staged.document_path, document)
        state = {
            "reused": False,
            "extraction_stage": "epub_items",
            "next_index": 0,
            "total": total,
            "removals": empty_removal_counts(),
            "empty_units": 0,
            **_NO_DISCARDED_CHUNKS,
        }
        build.add("extraction_work_total", total + 2)
        build.add("extraction_work_completed")
        await _atomic_to_thread(atomic_write_json, staged.state_path, state)
        return True

    async def _accept_extraction(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        """Count a source whose extraction a previous call completed."""

        document = await _atomic_to_thread(read_json, staged.document_path)
        units = await _atomic_to_thread(read_jsonl, staged.root / "units.jsonl")
        if (
            not isinstance(document, dict)
            or not units
            or any(
                unit.get("document_id") != document.get("document_id")
                or unit.get("source_id") != document.get("source_id")
                for unit in units
            )
        ):
            raise _BuildCannotResume(
                f"Completed extraction artifacts are invalid: {staged.relative}"
            )
        build.checkpoint["extracted_source_paths"].append(staged.relative)
        reused = bool(staged.state and staged.state.get("reused"))
        build.add("reused_document_count" if reused else "rebuilt_document_count")
        if not reused:
            build.add("extraction_work_completed")
        return await self._end_extraction_step(build, staged)

    async def _extract_pdf_pages(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        """Convert the next batch of a PDF's pages with Docling."""

        state = staged.state or {}
        total = int(state["total"])
        page_batch_size = int(state.get("page_batch_size") or 1)
        existing_document = (
            await _atomic_to_thread(read_json, staged.document_path)
            if staged.document_path.exists()
            else None
        )
        docling_env = await self._ensure_docling_environment()
        current_backend = docling_backend_record_from_env(docling_env)
        if existing_document is not None:
            recorded_backend = existing_document.get("extraction_backend")
            if not (
                isinstance(recorded_backend, dict)
                and recorded_backend.get("fingerprint")
                == current_backend.get("fingerprint")
            ):
                # The staged pages came from a different actual environment (or
                # name no environment at all). Do not append this environment's
                # output onto them: discard only this source's unfinished work
                # and convert it again from the first page.
                await _atomic_to_thread(
                    self._rescope_pdf_pages_for_environment,
                    staged.root,
                    state,
                    staged.state_path,
                    build.checkpoint,
                )
                existing_document = None
        index = int(state["next_index"])
        end_index = min(index + page_batch_size, total)
        prepared_batch = await self._extract_or_skip(
            build,
            staged,
            docling_pdf_batch,
            staged.source,
            existing_document,
            index,
            end_index,
            digest=staged.digest,
            offline=self.config.offline,
            model_cache_root=self.config.model_cache_root,
            docling_env=docling_env,
            total_pages=total,
        )
        if prepared_batch is None:
            return self._pause_if_due(build)
        pages, document, batch_empty, batch_removals, batch_diagnostics = prepared_batch
        if not staged.document_path.exists():
            await _atomic_to_thread(atomic_write_json, staged.document_path, document)
        diagnostics = {
            str(key): int(value)
            for key, value in (state.get("docling_diagnostics") or {}).items()
        }
        for key, value in batch_diagnostics.items():
            diagnostics[key] = int(diagnostics.get(key) or 0) + int(value)
        state["docling_diagnostics"] = diagnostics
        # Write one batch file per page in the range, empty included, so finalize
        # reads a complete sequence whatever the worker emitted for a blank page.
        for page_index in range(index, end_index):
            await _atomic_to_thread(
                atomic_write_jsonl,
                staged.root / "unit-batches" / f"{page_index:08d}.jsonl",
                pages.get(page_index, []),
                fsync_parent=False,
            )
        await _atomic_to_thread(fsync_directories, [staged.root / "unit-batches"])
        state["removals"] = merge_removal_counts(state.get("removals"), batch_removals)
        state["empty_units"] = int(state.get("empty_units") or 0) + int(batch_empty)
        state["next_index"] = end_index
        build.add("extraction_work_completed")
        if end_index >= total:
            state["extraction_stage"] = "finalize"
            state["next_index"] = 0
        await _atomic_to_thread(atomic_write_json, staged.state_path, state)
        return await self._end_extraction_step(build, staged)

    async def _extract_epub_item(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        """Read the next item of an EPUB's spine."""

        state = staged.state or {}
        index = int(state["next_index"])
        if index >= int(state["total"]):
            state["extraction_stage"] = "finalize"
            await _atomic_to_thread(atomic_write_json, staged.state_path, state)
            return None
        document = await _atomic_to_thread(read_json, staged.document_path)
        extracted = await self._extract_or_skip(
            build, staged, extract_epub_spine_item, staged.source, document, index
        )
        if extracted is None:
            return self._pause_if_due(build)
        batch, empty, removed = extracted
        state["removals"] = merge_removal_counts(state.get("removals"), removed)
        await _atomic_to_thread(
            atomic_write_jsonl,
            staged.root / "unit-batches" / f"{index:08d}.jsonl",
            batch,
            fsync_parent=False,
        )
        state["empty_units"] = int(state.get("empty_units") or 0) + int(empty)
        state["next_index"] = index + 1
        build.add("extraction_work_completed")
        await _atomic_to_thread(
            atomic_write_json, staged.state_path, state, fsync_parent=False
        )
        await _atomic_to_thread(
            fsync_directories, [staged.root / "unit-batches", staged.root]
        )
        return await self._end_extraction_step(build, staged)

    async def _finalize_extraction(
        self, build: _Build, staged: _StagedSource
    ) -> dict[str, Any] | None:
        """Screen a source's staged units and keep the ones that pass."""

        state = staged.state or {}
        units = []
        for index in range(int(state["total"])):
            units.extend(
                await _atomic_to_thread(
                    read_jsonl, staged.root / "unit-batches" / f"{index:08d}.jsonl"
                )
            )
        document = await _atomic_to_thread(read_json, staged.document_path)
        document["empty_units"] = int(state.get("empty_units") or 0)
        if state.get("docling_diagnostics"):
            document["docling_diagnostics"] = {
                str(key): int(value)
                for key, value in state["docling_diagnostics"].items()
            }
        retained = await self._extract_or_skip(
            build,
            staged,
            screen_source_units,
            staged.source,
            document,
            units,
            removals={
                key: int(value)
                for key, value in (state.get("removals") or {}).items()
                if key in REMOVAL_FIELDS
            },
            maximum_unclean_percent=self._effective_pdf_loss_limit(),
        )
        if retained is None:
            return self._pause_if_due(build)
        await _atomic_to_thread(atomic_write_json, staged.document_path, document)
        await _atomic_to_thread(
            atomic_write_jsonl, staged.root / "units.jsonl", retained
        )
        state["extraction_stage"] = "complete"
        await _atomic_to_thread(atomic_write_json, staged.state_path, state)
        build.checkpoint["extracted_source_paths"].append(staged.relative)
        build.add("rebuilt_document_count")
        build.add("extraction_work_completed")
        return await self._end_extraction_step(build, staged)

    async def _chunk_next_source(self, build: _Build) -> dict[str, Any] | None:
        """Take the next extracted source one durable step through chunking."""

        await self._load_axes(build, "units", "chunks")
        relative = build.next_source("chunked_source_paths")
        if relative is None:
            return await self._enter_phase(build, "assembly")
        root = self._source_artifact_root(build.staging_root, relative)
        state = await _atomic_to_thread(read_json, root / "state.json")
        document = await _atomic_to_thread(read_json, root / "document.json")
        units = build.units_cache.get(relative)
        if units is None:
            units = build.units_cache[relative] = await _atomic_to_thread(
                read_jsonl, root / "units.jsonl"
            )
        started = time.perf_counter()
        if (
            bool(state.get("reused"))
            # A resumed checkpoint from before the units/chunk split has no flag
            # and reused both, which is the default.
            and state.get("reuse_chunks", True)
            and build.snapshot is not None
        ):
            chunks = await _atomic_to_thread(
                build.snapshot.chunks_by_document.__getitem__,
                str(document["document_id"]),
            )
            if not build.snapshot.vectors_reusable:
                # Reused chunks keep their text, but the token count and truncation
                # flag are specific to the model they were counted with. This build
                # embeds them with a different model, so drop the old model's audit
                # and recount with the model now in use rather than carry stale
                # numbers.
                for chunk in chunks:
                    chunk.pop("embedding_token_count", None)
                    chunk.pop("dense_truncated", None)
                await _atomic_to_thread(
                    _record_embedding_token_counts,
                    chunks,
                    self.dense.embedding_token_counts,
                    maximum_tokens=self.config.settings.embedding_maximum_tokens,
                )
            discarded = [0, 0, 0]
            build.add("reused_chunk_count", len(chunks))
        elif "chunking_work_total" not in state:
            state["chunking_work_total"] = len(units)
            build.add("chunking_work_total", len(units))
            await _atomic_to_thread(atomic_write_json, root / "state.json", state)
            return await self._commit(build)
        elif int(state.get("chunked_unit_count") or 0) < len(units):
            return await self._chunk_unit_batch(build, root, state, units, started)
        else:
            chunks, *discarded = await self._chunks_from_units(
                build, root, units, document
            )
        state["chunking_stage"] = "complete"
        await _atomic_to_thread(atomic_write_jsonl, root / "chunks.jsonl", chunks)
        state.update(zip(_DISCARDED_CHUNK_KEYS, discarded, strict=True))
        await _atomic_to_thread(atomic_write_json, root / "state.json", state)
        build.add_time("chunking", started)
        build.checkpoint["chunked_source_paths"].append(relative)
        build.units_cache.pop(relative, None)
        return await self._commit(build)

    async def _chunk_unit_batch(
        self,
        build: _Build,
        root: Path,
        state: dict[str, Any],
        units: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any] | None:
        """Chunk the source's next batch of units and keep its output as one file."""

        chunked_count = int(state.get("chunked_unit_count") or 0)
        batch_units = units[
            chunked_count : chunked_count + self.config.settings.chunk_batch_units
        ]
        requested_ids = {str(unit["id"]) for unit in batch_units}
        chunking_root = root / "chunking"
        input_path = chunking_root / "unit.jsonl"
        working_path = root / "raw-chunks.jsonl"
        # An attempt at this batch that never committed may have left its output,
        # or a later unit's, behind; the listing must hold committed batches only.
        for start, _end, path in await _atomic_to_thread(
            _chunking_outputs, chunking_root
        ):
            if start >= chunked_count:
                path.unlink(missing_ok=True)
        # A handoff file is rewritten before every call and deleted afterwards, so
        # it needs visibility, not durability.
        await _atomic_to_thread(write_handoff_jsonl, input_path, batch_units)
        working_path.unlink(missing_ok=True)
        await self.lexical.chunk(
            input_path,
            working_path,
            chunk_size=build.chunk_size,
            chunk_overlap=build.chunk_overlap,
        )
        raw_by_unit: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in await _atomic_to_thread(read_jsonl, working_path):
            raw_by_unit[str(raw.get("doc_id") or "")].append(raw)
        working_path.unlink(missing_ok=True)
        input_path.unlink(missing_ok=True)
        unknown = [unit_id for unit_id in raw_by_unit if unit_id not in requested_ids]
        if unknown:
            raise _BuildCannotResume(
                f"Chunker returned an unknown extraction unit: {unknown[0]}"
            )
        # The batch's output is written in unit order, so the ordinal assignment in
        # `_enrich_chunks` is the same as when each unit had its own call. The
        # batch is the redo boundary, and one file is one durability operation.
        end = chunked_count + len(batch_units)
        await _atomic_to_thread(
            atomic_write_jsonl,
            chunking_root / f"{chunked_count:08d}-{end:08d}.jsonl",
            [
                raw
                for unit in batch_units
                for raw in raw_by_unit.get(str(unit["id"]), [])
            ],
            fsync_parent=False,
        )
        state["chunked_unit_count"] = end
        build.add("chunking_work_completed", len(batch_units))
        await _atomic_to_thread(
            atomic_write_json, root / "state.json", state, fsync_parent=False
        )
        # The whole batch is on disk before the checkpoint that claims it is
        # complete.
        await _atomic_to_thread(fsync_directories, [chunking_root, root])
        build.add_time("chunking", started)
        return await self._commit(build)

    async def _chunks_from_units(
        self,
        build: _Build,
        root: Path,
        units: list[dict[str, Any]],
        document: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """A source's chunks, enriched from its units' chunker output and audited."""

        # The committed outputs cover every unit once, in one run from the first.
        raw_chunks: list[dict[str, Any]] = []
        covered = 0
        for start, end, path in await _atomic_to_thread(
            _chunking_outputs, root / "chunking"
        ):
            if start != covered:
                covered = -1
                break
            raw_chunks.extend(await _atomic_to_thread(read_jsonl, path))
            covered = end
        if covered != len(units):
            raise _BuildCannotResume(
                f"Staged chunker output does not cover units 0 to {len(units)} "
                f"once each: {document.get('source_relative_path')}"
            )
        enriched = await _atomic_to_thread(
            _enrich_chunks,
            raw_chunks,
            units,
            [document],
            headers=self.config.settings.chunk_headers,
        )
        chunks = enriched[0]
        audited = await _atomic_to_thread(
            _record_embedding_token_counts,
            chunks,
            self.dense.embedding_token_counts,
            maximum_tokens=self.config.settings.embedding_maximum_tokens,
        )
        build.checkpoint["dense_token_audit"] = "counted" if audited else "unavailable"
        build.add("rebuilt_chunk_count", len(chunks))
        return enriched

    async def _assemble_corpus(self, build: _Build) -> dict[str, Any] | None:
        started = time.perf_counter()
        await _atomic_to_thread(
            self._assemble_staged_corpus, build.staging_root, build.checkpoint
        )
        build.add_time("assembly", started)
        build.checkpoint["phase"] = "embedding"
        return await self._commit(build)

    async def _embed_next_batch(self, build: _Build) -> dict[str, Any] | None:
        """Embed the next batch of chunks, reusing every vector the snapshot holds."""

        await self._load_axes(build, "chunks", "vectors")
        offset = int(build.checkpoint.get("embedded_chunk_count") or 0)
        if offset >= int(build.checkpoint["chunk_count"]):
            return await self._enter_phase(build, "vector_assembly")
        work = build.staging_root / "work"
        batch = await _atomic_to_thread(
            read_jsonl, work / "chunk-batches" / f"{offset:012d}.jsonl"
        )
        dimension = self.config.settings.embedding_dimension
        vectors = np.empty((len(batch), dimension), dtype=np.float32)
        # What is embedded is the chunk's own text, header included. Vector reuse
        # is keyed on the canonical passage, and a header cannot be reconstructed
        # from that key, so reuse suspends when the two differ: recomputing a
        # vector is always correct, while reusing another chunk's vector is not.
        batch_texts = [_embedding_text(chunk) for chunk in batch]
        reuse_keys = [chunk_text(chunk) for chunk in batch]
        reusable_vectors = (
            await _atomic_to_thread(build.snapshot.vectors_for_texts, reuse_keys)
            if build.snapshot is not None and reuse_keys == batch_texts
            else {}
        )
        missing_positions: list[int] = []
        for index, text in enumerate(batch_texts):
            reusable_vector = (
                None
                if str(batch[index].get("source_id")) in build.forced_source_ids
                else reusable_vectors.get(text)
            )
            if reusable_vector is None:
                missing_positions.append(index)
            else:
                vectors[index] = reusable_vector
        started = time.perf_counter()
        if missing_positions:
            created = await _atomic_to_thread(
                self.dense.embed_texts,
                [batch_texts[index] for index in missing_positions],
            )
            if created.shape != (len(missing_positions), dimension):
                raise ResearchError(
                    "Dense backend returned an invalid embedding matrix"
                )
            for position, vector in zip(missing_positions, created, strict=True):
                vectors[position] = vector
        await _atomic_to_thread(
            self._save_vector_batch,
            work / "vector-batches" / f"{offset:012d}.npy",
            vectors,
        )
        build.add_time("embedding", started)
        build.checkpoint["embedded_chunk_count"] = offset + len(batch)
        build.add("reused_vector_count", len(batch) - len(missing_positions))
        build.add("created_vector_count", len(missing_positions))
        return await self._commit(build)

    async def _assemble_vectors(self, build: _Build) -> dict[str, Any] | None:
        started = time.perf_counter()
        await _atomic_to_thread(
            self._assemble_vector_batches,
            build.staging_root / "portable" / "embeddings.npy",
            build.staging_root / "work" / "vector-batches",
            int(build.checkpoint["chunk_count"]),
            self.config.settings.embedding_dimension,
        )
        build.add_time("vector_assembly", started)
        build.checkpoint["phase"] = "bm25_indexing"
        return await self._commit(build)

    async def _index_bm25(self, build: _Build) -> dict[str, Any] | None:
        bm25_index_path = build.staging_root / "indexes" / "bm25"
        await _atomic_to_thread(shutil.rmtree, bm25_index_path, ignore_errors=True)
        started = time.perf_counter()
        # The build points the lexical backend's one retriever at the staging
        # index, so a search beside it waits and reloads its own.
        async with self._retriever_lock:
            self._loaded_generation = None
            await self.lexical.build_bm25(
                build.staging_root / "chunks" / "chunks.jsonl",
                bm25_index_path,
                language=self.config.settings.bm25_stopwords_language,
            )
        build.add_time("bm25_indexing", started)
        build.checkpoint["phase"] = "dense_indexing"
        build.checkpoint["dense_indexed_count"] = 0
        return await self._commit(build)

    async def _index_next_dense_batch(self, build: _Build) -> dict[str, Any] | None:
        """Upload the next batch of vectors to the dense index, then finalize it."""

        checkpoint = build.checkpoint
        if build.portable_vectors is None:
            build.portable_vectors = np.load(
                build.staging_root / "portable" / "embeddings.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
        total = int(checkpoint["chunk_count"])
        backend_name = str(checkpoint.get("dense_backend") or "")
        if backend_name not in DENSE_INDEX_PATHS:
            # A checkpoint resumed from before the backend was recorded restarts
            # this phase cleanly with the selected backend.
            backend_name = self._select_build_dense_backend()
            checkpoint["dense_backend"] = backend_name
            checkpoint["dense_index_relative"] = DENSE_INDEX_PATHS[backend_name]
            checkpoint["dense_indexed_count"] = 0
            await self._save(build)
        backend = self._dense_backends[backend_name]
        dense_index_path = build.staging_root / DENSE_INDEX_PATHS[backend_name]
        dimension = self.config.settings.embedding_dimension
        offset = int(checkpoint.get("dense_indexed_count") or 0)
        if offset == 0:
            await _atomic_to_thread(shutil.rmtree, dense_index_path, ignore_errors=True)
            await _atomic_to_thread(
                backend.initialize_index, dense_index_path, dimension
            )
        if offset < total:
            batch = await _atomic_to_thread(
                read_jsonl,
                build.staging_root / "work" / "chunk-batches" / f"{offset:012d}.jsonl",
            )
            started = time.perf_counter()
            await _atomic_to_thread(
                backend.upload_index_batch,
                batch,
                dense_index_path,
                build.portable_vectors[offset : offset + len(batch)],
                offset=offset,
            )
            build.add_time("dense_indexing", started)
            checkpoint["dense_indexed_count"] = offset + len(batch)
            return await self._commit(build)
        checkpoint["dense_metadata"] = await _atomic_to_thread(
            backend.finalize_index,
            dense_index_path,
            expected_count=total,
            dimension=dimension,
        )
        return await self._enter_phase(build, "source_revalidation")

    async def _revalidate_next_source(self, build: _Build) -> dict[str, Any] | None:
        """Hash the next source again; once all are, confirm no input moved."""

        checkpoint = build.checkpoint
        revalidated = checkpoint["revalidation_digests"]
        record = next(
            (
                record
                for record in checkpoint["source_inventory"]
                if record["source_relative_path"] not in revalidated
            ),
            None,
        )
        if record is not None:
            source = build.sources_by_path[str(record["source_relative_path"])]
            started = time.perf_counter()
            try:
                digest, stat_identity = await _atomic_to_thread(
                    _hash_with_stable_stat, source.path
                )
            except OSError as exc:
                raise _SourceChangedDuringIngest(
                    "A source became unavailable during final validation"
                ) from exc
            revalidated[source.source_relative_path] = digest
            checkpoint["revalidation_stats"][source.source_relative_path] = (
                stat_identity
            )
            build.add_time("source_revalidation", started)
            return await self._commit(build)
        if revalidated != checkpoint["source_digests"]:
            raise _SourceChangedDuringIngest(
                "Source bytes changed while ingestion was in progress"
            )
        try:
            final_scan = await _atomic_to_thread(scan_sources, self.config)
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
            str(selected_now[1]["generation_id"]) if selected_now is not None else None
        )
        if selected_generation_id != checkpoint.get("baseline_generation_id"):
            raise _SourceChangedDuringIngest(
                "The selected generation changed while ingestion was in progress"
            )
        return await self._enter_phase(build, "finalizing")

    async def _finalize_build(self, build: _Build) -> dict[str, Any]:
        """Write the manifest, validate the build, and activate it as a generation."""

        checkpoint = build.checkpoint
        if not build.revalidation_stats_match(require_complete=True):
            raise _SourceChangedDuringIngest(
                "A source changed after its final ingestion hash"
            )
        skipped_paths = {
            item["source_relative_path"]
            for item in checkpoint.get("skipped_sources", [])
        } | set(checkpoint.get("dropped_source_paths") or ())
        documents = await _atomic_to_thread(
            list,
            map(
                read_json,
                [
                    self._source_artifact_root(build.staging_root, relative)
                    / "document.json"
                    for relative in build.selected_paths
                    if relative not in skipped_paths
                ],
            ),
        )
        census = await _atomic_to_thread(
            _staged_chunk_census, build.staging_root / "chunks" / "chunks.jsonl"
        )
        build_metrics = self._build_metrics(build, documents, census)
        counts = {
            "extraction_unit_count": int(checkpoint["extraction_unit_count"]),
            "chunk_count": int(checkpoint["chunk_count"]),
            "content_kind_counts": census["content_kind_counts"],
            **{
                key: int(checkpoint.get(key) or 0)
                for key in (
                    "discarded_empty_chunk_count",
                    "discarded_symbol_only_chunk_count",
                    "discarded_corrupt_chunk_count",
                    "excluded_corrupt_unit_count",
                )
            },
        }
        records = build.recorded_sources(
            {str(document["source_relative_path"]): document for document in documents}
        )
        manifest = self._generation_manifest(
            build, documents, records, counts, build_metrics
        )
        if not build.revalidation_stats_match(require_complete=True):
            raise _SourceChangedDuringIngest(
                "A source changed while final artifacts were assembled"
            )
        await _atomic_to_thread(
            atomic_write_json, build.staging_root / "manifest.json", manifest
        )
        try:
            await self._validate_generation_for_activation(build.staging_root, manifest)
        except ResearchError as exc:
            raise _BuildCannotResume(str(exc)) from exc
        if not build.revalidation_stats_match(require_complete=True):
            raise _SourceChangedDuringIngest(
                "A source changed while final indexes were validated"
            )
        select_generation = not build_metrics.get("skipped_sources")
        generation_root = await self._activate_build(
            build, manifest, select=select_generation
        )
        build_id = checkpoint["build_id"]
        LOGGER.debug(
            "ingest: build %s complete status=%s selected=%s documents=%s chunks=%s",
            build_id,
            "ready" if select_generation else "partial",
            select_generation,
            len(documents),
            counts["chunk_count"],
        )
        return {
            "status": "ready" if select_generation else "partial",
            "generation_changed": select_generation,
            "current_generation_id": (
                build_id
                if select_generation
                else checkpoint.get("baseline_generation_id")
            ),
            **(
                {}
                if select_generation
                else {
                    "manual_selection_command": project_command(
                        self.config.project_root, "generations", "--use", str(build_id)
                    )
                }
            ),
            "generation_id": build_id,
            "generation_root": str(generation_root),
            "source_file_count": len(build.scan.selected),
            "excluded_source_count": len(build.exclusions),
            "excluded_sources": build.excluded_sources(),
            "document_count": len(documents),
            "pdf_count": sum(item["format"] == "pdf" for item in documents),
            "epub_count": sum(item["format"] == "epub" for item in documents),
            "mobi_count": sum(item["format"] == "mobi" for item in documents),
            **counts,
            "ignored_extensions": build.scan.ignored_extensions,
            "empty_units": sum(int(item["empty_units"]) for item in documents),
            "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
            "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
            "embedding_model": self.config.settings.embedding_model,
            "embedding_model_revision": self.config.settings.embedding_model_revision,
            **build_metrics,
        }

    def _build_metrics(
        self,
        build: _Build,
        documents: list[dict[str, Any]],
        census: dict[str, Any],
    ) -> dict[str, Any]:
        """What the build reused, rebuilt, discarded, and withheld, and how long it took."""

        checkpoint = build.checkpoint
        loss_totals = _document_loss_totals(documents)
        substantive_characters = loss_totals["substantive_character_count"]
        audited_chunk_count = census["audited_chunk_count"]

        def count(key: str) -> int:
            return int(checkpoint.get(key) or 0)

        return {
            "skipped_sources": checkpoint.get("skipped_sources", []),
            "skipped_source_count": len(checkpoint.get("skipped_sources", [])),
            # A build that named its sources says which, and whether it read them
            # or took them out; a build of every source names none.
            **(
                {"source_selection": checkpoint["parameters"]["selection"]}
                if "selection" in checkpoint["parameters"]
                else {}
            ),
            **(
                {"dropped_excluded_sources": checkpoint["dropped_source_paths"]}
                if checkpoint.get("dropped_source_paths")
                else {}
            ),
            "forced": build.force_recompute,
            "resumed": bool(checkpoint.get("resume_count")),
            **{
                key: count(key)
                for key in (
                    "reused_document_count",
                    "rebuilt_document_count",
                    "reused_chunk_count",
                    "rebuilt_chunk_count",
                    "reused_vector_count",
                    "created_vector_count",
                    "excluded_corrupt_unit_count",
                )
            },
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
            "discarded_symbol_only_chunk_count": count(
                "discarded_symbol_only_chunk_count"
            ),
            "discarded_corrupt_chunk_count": count("discarded_corrupt_chunk_count"),
            "withheld_chunk_count": census["withheld_chunk_count"],
            "withheld_chunk_reasons": census["withheld_chunk_reasons"],
            "dense_token_audit": (
                "counted"
                if audited_chunk_count == int(checkpoint["chunk_count"])
                else ("partial" if audited_chunk_count else "unavailable")
            ),
            "dense_audited_chunk_count": audited_chunk_count,
            "dense_truncated_chunk_count": census["dense_truncated_chunk_count"],
            "embedding_maximum_tokens": self.config.settings.embedding_maximum_tokens,
            "maximum_embedding_token_count": census["maximum_embedding_token_count"],
            "phase_timings_seconds": {
                key: round(float(value), 6)
                for key, value in checkpoint["phase_timings_seconds"].items()
            },
        }

    def _generation_manifest(
        self,
        build: _Build,
        documents: list[dict[str, Any]],
        records: list[dict[str, Any]],
        counts: dict[str, Any],
        build_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """The manifest that names every artifact of the build and how it was made."""

        checkpoint = build.checkpoint
        settings = self.config.settings
        return {
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
            "ignored_extensions": build.scan.ignored_extensions,
            # Reviewed metadata is a portable read-time overlay. This revision is
            # observational and never fingerprints derived index contents or
            # checkpoint compatibility.
            "metadata_revision": value_fingerprint(self._metadata()),
            "metadata_storage_policy": METADATA_STORAGE_POLICY,
            "source_exclusion_revision": checkpoint["source_exclusion_revision"],
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "source_file_count": len(records),
            "excluded_source_count": len(build.exclusions),
            "document_count": len(documents),
            **counts,
            "passage_cleaning": {
                # The configured budget is the identity a rebuild reads; the
                # effective budget is what the shared gate actually applied for the
                # selected backend. Both readers use the project budget. Reuse
                # compares the configured value, so a legacy manifest with only
                # that key stays compatible.
                "maximum_unclean_percent": settings.maximum_unclean_percent,
                "effective_maximum_unclean_percent": self._effective_pdf_loss_limit(),
                "pdf_backend": self._pdf_backend,
            },
            "extraction_backend": pdf_backend_record(settings.pdf_backend, self.config),
            "chunking": {
                "backend": "Chonkie token chunker (UltraRAG-compatible)",
                "tokenizer": "gpt2",
                "unit": "tokens",
                "chunk_size": build.chunk_size,
                "chunk_overlap": build.chunk_overlap,
                "headers": settings.chunk_headers,
            },
            "retrieval": {
                "default_method": DEFAULT_RETRIEVAL_METHOD,
                "available_methods": sorted(RETRIEVAL_METHODS),
                "bm25": {
                    "backend": "bm25s (UltraRAG-compatible)",
                    # The setting the index was built with, not a constant.
                    "language": settings.bm25_stopwords_language,
                    "tokenizer": "default",
                },
                "dense": dict(checkpoint["dense_metadata"]),
                "fusion": {
                    "method": "weighted_reciprocal_rank_fusion",
                    "rrf_k": settings.rrf_k,
                    "bm25_weight": settings.bm25_weight,
                    "dense_weight": settings.dense_weight,
                    "minimum_candidates": settings.minimum_candidates,
                    "maximum_candidates": settings.maximum_candidates,
                },
                "relevance_gates": {
                    "bm25_requires_query_token_overlap": True,
                    "dense_minimum_cosine_similarity": (
                        settings.dense_minimum_cosine_similarity
                    ),
                },
                "reranker": {
                    "optional": True,
                    "runtime": "FastEmbed ONNX Runtime (CPU)",
                    "model": self.config.reranker_model,
                    "model_revision": _reranker_revision(self.config.reranker_model),
                    "maximum_candidates": settings.rerank_max_candidates,
                },
            },
            "documents": documents,
            "source_files": records,
            "excluded_sources": build.excluded_sources(),
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

    async def _activate_build(
        self, build: _Build, manifest: dict[str, Any], *, select: bool
    ) -> Path:
        """Move the validated build into place as a generation, and select it if asked.

        The activation journal is written first, so a crash at any point after it
        is finished by the next call rather than leaving a half-moved build.
        """

        checkpoint = build.checkpoint
        generation_root = self.config.generations_root / str(checkpoint["build_id"])
        await _atomic_to_thread(
            atomic_write_json,
            self._pending_activation_path,
            {
                "schema_version": PENDING_ACTIVATION_VERSION,
                "project_id": self.config.project_id,
                "build_id": checkpoint["build_id"],
                "select": select,
                "identity": checkpoint["identity"],
                "baseline_generation_id": checkpoint.get("baseline_generation_id"),
                "source_inventory": checkpoint["source_inventory"],
                # What the activated manifest records, which a recovery compares.
                "source_digests": {
                    str(record["source_relative_path"]): str(record["sha256"])
                    for record in manifest["source_files"]
                },
                "revalidation_stats": checkpoint["revalidation_stats"],
                "source_exclusion_revision": checkpoint["source_exclusion_revision"],
                "created_at": _utc_now(),
            },
        )
        await _atomic_to_thread(
            shutil.rmtree, build.staging_root / "work", ignore_errors=True
        )
        await _atomic_to_thread(os.replace, build.staging_root, generation_root)
        await _atomic_to_thread(fsync_directory, self.config.generations_root)
        await _atomic_to_thread(fsync_directory, self.config.staging_root)
        if select:
            await _atomic_to_thread(
                atomic_write_json,
                self.config.current_path,
                {"schema_version": 1, "generation_id": checkpoint["build_id"]},
            )
        (generation_root / "checkpoint.json").unlink()
        await _atomic_to_thread(fsync_directory, generation_root)
        self._pending_activation_path.unlink(missing_ok=True)
        await _atomic_to_thread(fsync_directory, self._pending_activation_path.parent)
        self._loaded_generation = None
        return generation_root

    def _source_selection(
        self,
        mode: str,
        names: list[str],
        *,
        scan: SourceScan,
        exclusions: Mapping[str, Any],
        held: frozenset[str],
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """The sources a command named, as paths, or a refusal naming the problem.

        A source is named by its path as `sources` reports it or by its `src_` id.
        An ingest names sources on disk that are not excluded, since an exclusion
        keeps a source from being ingested. An uningest names sources the selected
        generation holds, whether or not their files remain.
        """

        scanned = {source.source_relative_path for source in scan.selected}
        by_id = {
            source.source_id: source.source_relative_path for source in scan.selected
        }
        for document in current[1].get("documents", []) if current else []:
            if isinstance(document, dict) and document.get("source_id"):
                by_id.setdefault(
                    str(document["source_id"]),
                    str(document.get("source_relative_path") or ""),
                )
        paths: set[str] = set()
        unknown: list[str] = []
        for name in names:
            relative = name if name in scanned | held else by_id.get(name)
            if relative:
                paths.add(relative)
            else:
                unknown.append(name)
        if unknown:
            raise ResearchError(
                f"No source matches {', '.join(map(repr, unknown))}. Name a source "
                "by its path, as `sources` reports it, or by its src_ id."
            )
        if mode == "ingest":
            missing = sorted(paths - scanned)
            if missing:
                raise ResearchError(
                    f"{', '.join(missing)} has no file under the source directory, "
                    "so there is nothing to ingest."
                )
            excluded = sorted(paths & set(exclusions))
            if excluded:
                raise ResearchError(
                    f"{', '.join(excluded)} is excluded, which keeps a source from "
                    "being ingested. Include it first: "
                    + project_command(self.config.project_root, "include", excluded[0])
                )
        else:
            absent = sorted(paths - held)
            if absent:
                raise ResearchError(
                    f"The selected generation does not hold {', '.join(absent)}, so "
                    "there is nothing to uningest."
                )
            selection = {"mode": mode, "source_paths": sorted(paths)}
            if not _selected_sources(scan, exclusions, held, selection)[0]:
                raise ResearchError(
                    "Uningesting these sources would leave the generation empty, so "
                    "nothing was changed."
                )
        return {"mode": mode, "source_paths": sorted(paths)}

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
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create or refresh the generation that search reads.

        Every source that is not excluded is read, unless `sources` names the ones
        to read: then every other source the selected generation holds is carried
        as it is, and `force_recompute` reads only the named ones again. An omitted
        parameter comes from the merged settings, where the chunking values live; a
        caller that passes one gets it checked here.
        """

        return await self._build(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_recompute=force_recompute,
            work_budget_seconds=work_budget_seconds,
            selection=("ingest", sources) if sources else None,
        )

    async def uningest(
        self, *, sources: list[str], work_budget_seconds: int | None = None
    ) -> dict[str, Any]:
        """Build a generation without the named sources, and carry every other one.

        Their chunks and data leave the new generation. The files, their reviewed
        metadata, and their exclusions stay, so an ingest of every source reads a
        source again unless it is excluded. The generation before keeps it, until
        it is removed.
        """

        if not sources:
            raise ResearchError("Name at least one source to uningest")
        return await self._build(
            chunk_size=None,
            chunk_overlap=None,
            force_recompute=False,
            work_budget_seconds=work_budget_seconds,
            selection=("uningest", sources),
        )

    async def _build(
        self,
        *,
        chunk_size: int | None,
        chunk_overlap: int | None,
        force_recompute: bool,
        work_budget_seconds: int | None,
        selection: tuple[str, list[str]] | None,
    ) -> dict[str, Any]:
        settings = self.config.settings
        # A build that names its sources carries the rest as they are, so it
        # chunks with the sizes the selected generation recorded.
        recorded = (
            (self._load_current_optional() or (None, {}))[1].get("chunking") or {}
            if selection is not None
            else {}
        )
        if chunk_size is None:
            chunk_size = recorded.get("chunk_size")
        if chunk_overlap is None:
            chunk_overlap = recorded.get("chunk_overlap")
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
            await _atomic_to_thread(self._cleanup_invalid_staging_roots)
            try:
                scan = await _atomic_to_thread(scan_sources, self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if not scan.selected:
                raise ResearchError(
                    f"No PDF, EPUB or MOBI sources found beneath {self.config.source_root}"
                )
            exclusions = self._source_exclusions()
            if selection is None and all(
                source.source_relative_path in exclusions for source in scan.selected
            ):
                raise ResearchError(
                    "All discovered PDF, EPUB and MOBI sources are excluded; include at "
                    "least one source before ingesting"
                )
            current = self._load_current_optional()
            held = frozenset(
                str(document.get("source_relative_path") or "")
                for document in (current[1].get("documents", []) if current else [])
                if isinstance(document, dict)
            )
            selection_record = (
                None
                if selection is None
                else self._source_selection(
                    *selection,
                    scan=scan,
                    exclusions=exclusions,
                    held=held,
                    current=current,
                )
            )
            await _atomic_to_thread(
                self._sync_source_catalog, scan, current, exclusions=exclusions
            )
            baseline_generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )
            exclusion_revision = value_fingerprint(exclusions)
            inventory = _source_inventory(scan)
            # The managed Docling environment is resolved lazily, only when a new
            # or changed PDF actually extracts. An unchanged corpus returns
            # `unchanged` first, so a reader never pays for provisioning Docling to
            # discover there is nothing to build. The identity uses the backend's
            # stable spec fingerprint, which does not move across a first install.
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
                selection=selection_record,
            )
            recovered = await self._recover_pending_activation(
                expected_identity=identity,
                scan=scan,
                exclusions=exclusions,
                current=current,
            )
            if recovered is not None:
                return recovered
            existing = await _atomic_to_thread(self._load_ingestion_checkpoint)
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
                await _atomic_to_thread(
                    self._discard_checkpoint,
                    existing[0],
                    superseded_checkpoint,
                    reason=str(superseded_build["reason"]),
                )
                existing = None
            if existing is None:
                staging_root, checkpoint = await _atomic_to_thread(
                    self._create_checkpoint,
                    scan=scan,
                    exclusions=exclusions,
                    exclusion_revision=exclusion_revision,
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                    held=held,
                    selection=selection_record,
                )
            else:
                staging_root, checkpoint = existing
                await _atomic_to_thread(self._remove_uncommitted_files, staging_root)
                checkpoint["resume_count"] = (
                    int(checkpoint.get("resume_count") or 0) + 1
                )
                await _atomic_to_thread(
                    self._write_checkpoint, staging_root, checkpoint
                )

            LOGGER.debug(
                "ingest: build %s starting phase=%s resume=%s force=%s",
                checkpoint.get("build_id"),
                checkpoint.get("phase"),
                existing is not None,
                force_recompute,
            )
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
                await _atomic_to_thread(
                    self._discard_checkpoint, staging_root, checkpoint, reason=str(exc)
                )
                fresh_scan = await _atomic_to_thread(scan_sources, self.config)
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
                    held=held,
                    selection=selection_record,
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
                checkpoint["last_interruption"] = "cancelled"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except TimeoutError:
                checkpoint["last_interruption"] = "timeout"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except Exception as exc:
                pending = (
                    read_json(self._pending_activation_path)
                    if self._pending_activation_path.is_file()
                    else None
                )
                if isinstance(pending, dict) and pending.get(
                    "build_id"
                ) == checkpoint.get("build_id"):
                    raise
                if self._keep_failed_build(staging_root, exc):
                    LOGGER.warning(
                        "ingest: build %s stopped in %s; its checkpoint is kept",
                        checkpoint.get("build_id"),
                        checkpoint.get("phase"),
                        exc_info=exc,
                    )
                    raise ResearchError(
                        f"Ingestion stopped during {checkpoint.get('phase')}: {exc}. "
                        "Finished work is kept; run "
                        f"{project_command(self.config.project_root, 'ingest')} "
                        "again to resume once the cause is fixed."
                    ) from exc
                self._discard_checkpoint(staging_root, checkpoint, reason=str(exc))
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
            return result
