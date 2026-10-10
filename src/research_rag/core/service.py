from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout

from ..corpus.sources import (  # noqa: F401
    SourcePolicyError,
    scan_sources,
    stable_source_id,
)

# The storage writers and the source walk are re-exported because tests and
# `scripts/benchmark_write_pattern.py` read them from this module. The code that
# calls them lives in the workflow modules, so a patch intercepting a call targets
# that module, not this one.
# Read by tests through this module; the ingestion workflow imports it separately.
from ..generations.ingestion import IngestionWorkflow
from ..project.config import ResearchConfig

# Re-exported on the same terms as the helpers below: the tests read the digest
# from this module, and one definition of it lives in `policy.py`.
from ..project.policy import (
    ResearchError,
    retrieval_policy_fingerprint,
    value_fingerprint,  # noqa: F401
)
from ..project.settings_document import SettingsWorkflow
from ..project.state_files import LOCK_FILE

# The pure helpers moved to `support`, re-exported so every existing import
# path, including the tests that read them from this module, keeps working.
from ..project.support import (  # noqa: F401
    _WORD,
    ARTIFACT_POLICY_VERSION,
    CLEANING_POLICY_VERSION,
    EXTRACTION_POLICY_VERSION,
    INGESTION_IDENTITY_POLICY_VERSION,
    METADATA_STORAGE_POLICY,
    SCHEMA_VERSION,
    _atomic_to_thread,
    _candidate_flags,
    _canonical_metadata_override,
    _checkpoint_identity,
    _chunk_text,
    _citation,
    _content_tokens,
    _document_for_chunk,
    _document_matches_metadata,
    _effective_document_metadata,
    _effective_documents,
    _embedding_text,
    _enrich_chunks,
    _generation_id,
    _hash_with_stable_stat,
    _is_extraction_artifact,
    _metadata_inventory,
    _metadata_snapshot_changed,
    _normalized_filter,
    _pdf_batch_count,
    _pseudo_relevance_terms,
    _public_document,
    _public_passage,
    _record_embedding_token_counts,
    _record_withheld,
    _requested_ids,
    _reranker_revision,
    _source_diverse_selection,
    _source_inventory,
    _source_stat_identity,
    _source_work_key,
    document_frequencies,
)
from ..retrieval.dense import (
    DENSE_INDEX_PATHS,
    EXACT_BACKEND_NAME,
    QDRANT_BACKEND_NAME,
    DenseBackend,
    LocalQdrantDenseBackend,
    LocalVectorDenseBackend,
)
from ..retrieval.direct import DirectRetrieval
from ..retrieval.search import SearchWorkflow
from ..storage.records import (  # noqa: F401
    StorageError,
    atomic_write_json,
    atomic_write_jsonl,
    fsync_directories,
    load_chunk_exclusions,
    load_current_generation,
    load_metadata_overrides,
    load_source_catalog,
    load_source_exclusions,
    read_json,
    read_jsonl,
    write_handoff_jsonl,
)
from .admission import Admission, AdmissionTimeout
from .review import ReviewWorkflow
from .stats import StatsWorkflow
from .status import StatusWorkflow

# How long a caller waits for another process's project lock before it is told the
# project is busy. Waiting longer does not help: an MCP client gives up long before
# a build ends, and the work it waited for goes on unseen. Reporting the resident
# build is more useful than outlasting the client.
PROJECT_LOCK_TIMEOUT_SECONDS = 20

# How long a search waits for its turn behind the others before it is told to ask
# again. It is longer than a write's wait because a search is short and a queue of
# them clears, where a write waits behind a build that does not.
SEARCH_QUEUE_SECONDS = 45


# The retrieval policy is fixed here: the tool offers exactly one way to search.
# The numbers that shape it live in the settings file, so fusion weights, gates,
# batch sizes, and budgets are tunable without editing code.
class ResearchService(
    IngestionWorkflow,
    ReviewWorkflow,
    SearchWorkflow,
    SettingsWorkflow,
    StatsWorkflow,
    StatusWorkflow,
):
    def __init__(
        self,
        config: ResearchConfig,
        ultrarag: DirectRetrieval | None,
        dense: DenseBackend | None = None,
        *,
        record_searches: bool = True,
    ) -> None:
        self.config = config
        self.ultrarag = ultrarag
        # A measurement builds its own service and runs hundreds of searches, so
        # it turns the counts off rather than filling them with its query set.
        self._records_searches = record_searches
        self._dense_backends: dict[str, DenseBackend]
        if dense is not None:
            # An injected backend serves every recorded kind, so deterministic
            # test doubles stand in for both real backends.
            self._dense_backends = {name: dense for name in DENSE_INDEX_PATHS}
        else:
            self._dense_backends = {
                QDRANT_BACKEND_NAME: LocalQdrantDenseBackend(
                    config.models_root,
                    offline=config.offline,
                    embedding_threads=config.embedding_threads,
                    reranker_model=config.reranker_model,
                    embedding_model=config.settings.embedding_model,
                    embedding_inference_batch_size=(
                        config.settings.embedding_inference_batch_size
                    ),
                ),
                EXACT_BACKEND_NAME: LocalVectorDenseBackend(
                    config.models_root,
                    offline=config.offline,
                    embedding_threads=config.embedding_threads,
                    reranker_model=config.reranker_model,
                    embedding_model=config.settings.embedding_model,
                    embedding_inference_batch_size=(
                        config.settings.embedding_inference_batch_size
                    ),
                ),
            }
        # The primary backend answers model-only calls; indexing, validation, and
        # retrieval resolve the backend that the generation itself recorded.
        self.dense = (
            self._dense_backends[EXACT_BACKEND_NAME]
            if config.dense_backend in {"auto", "exact"}
            else self._dense_backends[QDRANT_BACKEND_NAME]
        )
        # The ranking policy is part of a generation identity, so it is
        # computed once from the settings this process resolved.
        self.retrieval_policy_fingerprint = retrieval_policy_fingerprint(
            config.settings
        )
        # Writes take turns one at a time and searches a few at a time, each caller
        # in rounds, so no one agent, and no build, holds the others out. Both
        # waits are bounded and refuse with the queue they stood in.
        self._writes = Admission(
            1, wait_seconds=PROJECT_LOCK_TIMEOUT_SECONDS, what="operations"
        )
        self._searches = Admission(
            config.settings.search_concurrency,
            wait_seconds=SEARCH_QUEUE_SECONDS,
            what="searches",
        )
        self._project_lock = AsyncFileLock(
            config.state_root / LOCK_FILE,
            # A caller must not sit in silence behind another build. A long build is
            # driven by repeated short calls, so one caller never blocks another
            # for minutes, and waiting here turns a busy project into a client
            # timeout: the MCP client gives up long before the wait ends, and the
            # work continues unseen.
            timeout=PROJECT_LOCK_TIMEOUT_SECONDS,
        )
        self._loaded_generation: str | None = None
        # Reads take neither lock above, so a search, a passage, the source list,
        # and the status answer while a build runs. Two things are still shared
        # with a build. The gateway holds one BM25 retriever, which a build points
        # at the index it is writing, so its use is taken in turn. And a read
        # names the generation it is reading, so a removal waits for it.
        self._retriever_lock = asyncio.Lock()
        self._generation_readers: dict[str, int] = {}
        self._readers_left = asyncio.Condition()
        # Term rarity for pseudo-relevance feedback, as (generation id,
        # function-word set, table). Built on the first search that asks for one,
        # so a process that never enables the feature never makes the pass over
        # the corpus.
        self._document_frequencies: (
            tuple[str, frozenset[str], dict[str, int]] | None
        ) = None

    @asynccontextmanager
    async def _operation(self, *, busy_command: str = "") -> AsyncIterator[None]:
        """Serialize project access across MCP and UI server processes.

        `busy_command` names a command that reports what holds the lock, for the
        operation whose reader is waiting on the answer rather than on the work.
        """

        try:
            async with self._writes.slot():
                try:
                    async with self._project_lock:
                        yield
                except FileLockTimeout as exc:
                    raise ResearchError(
                        "Another research process is working on this project"
                        + self._resident_build_note()
                        + self._busy_remedy(busy_command)
                    ) from exc
        except AdmissionTimeout as exc:
            raise ResearchError(
                "This app is already working on this project"
                + self._resident_build_note()
                + f" ({exc.ahead} calls were ahead of this one)"
                + (
                    self._busy_remedy(busy_command)
                    if busy_command
                    else ". Nothing was written: call this again once that work "
                    "finishes."
                )
            ) from exc

    @asynccontextmanager
    async def _read(self) -> AsyncIterator[_ReadLease]:
        """A read of the selected generation, taken without the project lock.

        Activation swaps `current.json` atomically and leaves the generation it
        replaced on disk, so a read resolves its generation once and reads it
        whole while a build runs beside it. The lease names that generation so a
        removal waits for the read to finish.
        """

        lease = _ReadLease(self)
        try:
            yield lease
        finally:
            await lease.release()

    async def _wait_for_readers(self, generation_id: str) -> None:
        """Return once no read holds ``generation_id``, or refuse after the wait."""

        async with self._readers_left:
            try:
                await asyncio.wait_for(
                    self._readers_left.wait_for(
                        lambda: not self._generation_readers.get(generation_id)
                    ),
                    timeout=PROJECT_LOCK_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise ResearchError(
                    f"A read is still using generation {generation_id}, so nothing "
                    "was removed. Remove it again once that read finishes."
                ) from exc

    @asynccontextmanager
    async def _retriever(
        self, generation_root: Path, manifest: dict[str, Any]
    ) -> AsyncIterator[None]:
        """The gateway's BM25 retriever, loaded with this generation, for one use.

        A build points the retriever at the index it is writing, so a search
        takes it in turn and loads the generation it is reading on the way in.
        """

        try:
            await asyncio.wait_for(
                self._retriever_lock.acquire(), timeout=PROJECT_LOCK_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            raise ResearchError(
                "A build in this app is writing the lexical index"
                + self._resident_build_note()
                + ". Nothing was read: search again once that phase ends."
            ) from exc
        try:
            await self._ensure_loaded(generation_root, manifest)
            yield
        finally:
            self._retriever_lock.release()

    @asynccontextmanager
    async def _search_read(self) -> AsyncIterator[_ReadLease]:
        """A read of the selected generation, taken in turn with the other searches."""

        try:
            async with self._searches.slot(), self._read() as lease:
                yield lease
        except AdmissionTimeout as exc:
            raise ResearchError(
                f"{exc} The app is serving {self._searches.waiting} waiting "
                f"{'search' if self._searches.waiting == 1 else 'searches'} now."
            ) from exc

    @staticmethod
    def _busy_remedy(busy_command: str) -> str:
        if busy_command:
            return (
                f". Nothing was written. Read {busy_command} to see what is "
                "holding it, or wait for it to finish."
            )
        return (
            ". Its work is not lost: call this again once it finishes, or stop "
            "that process first."
        )

    def _resident_build_note(self) -> str:
        """Describe the build another process is running, when one is visible.

        Read-only and best-effort: a caller told the project is busy uses the
        note to see whether the resident build is moving, instead of deciding
        between waiting blind and killing it.
        """

        try:
            roots = sorted(
                self.config.staging_root.iterdir(),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return ""
        for root in roots:
            checkpoint_path = root / "checkpoint.json"
            if not checkpoint_path.is_file():
                continue
            try:
                checkpoint = read_json(checkpoint_path)
            except (OSError, ValueError):
                continue
            if not isinstance(checkpoint, dict):
                continue
            phase = str(checkpoint.get("phase") or "an unknown phase")
            progress = self._ingestion_progress(checkpoint).get("progress") or {}
            completed = progress.get("completed")
            total = progress.get("total")
            if completed is None or not total:
                return f" (build {checkpoint.get('build_id')}, in {phase})"
            return (
                f" (build {checkpoint.get('build_id')}, in {phase}, "
                f"{completed} of {total})"
            )
        return ""

    @staticmethod
    def _dense_backend_name(manifest: dict[str, Any] | None) -> str:
        """
        Return the dense backend a generation recorded.

        Generations written before the backend was recorded used the embedded
        Qdrant index.
        """

        if manifest:
            dense = manifest.get("retrieval", {}).get("dense", {})
            recorded = dense.get("dense_backend")
            if isinstance(recorded, str) and recorded in DENSE_INDEX_PATHS:
                return recorded
        return QDRANT_BACKEND_NAME

    def _dense_for(self, manifest: dict[str, Any] | None) -> DenseBackend:
        return self._dense_backends[self._dense_backend_name(manifest)]

    def _metadata(self) -> dict[str, dict[str, Any]]:
        try:
            values = load_metadata_overrides(self.config.metadata_path)
            normalized = {
                key: _canonical_metadata_override(value)
                for key, value in values.items()
            }
            return {key: value for key, value in normalized.items() if value}
        except (StorageError, SourcePolicyError) as exc:
            raise ResearchError(str(exc)) from exc

    def _source_exclusions(self) -> dict[str, dict[str, str]]:
        try:
            return load_source_exclusions(self.config.source_exclusions_path)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    def _chunk_exclusions(self) -> dict[str, dict[str, str]]:
        try:
            return load_chunk_exclusions(self.config.chunk_exclusions_path)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    def _source_catalog(self) -> dict[str, str]:
        try:
            return load_source_catalog(
                self.config.source_catalog_path,
                project_id=self.config.project_id,
            )
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    @staticmethod
    def _excluded_document_ids(
        manifest: dict[str, Any],
        exclusions: dict[str, dict[str, str]],
    ) -> set[str]:
        return {
            str(document["document_id"])
            for document in manifest.get("documents", [])
            if str(document.get("source_relative_path")) in exclusions
        }

    def _document_ids_for_source_ids(
        self,
        manifest: dict[str, Any],
        source_ids: list[str],
    ) -> tuple[set[str], list[str]]:
        """Resolve stable source IDs to document IDs in the given generation.

        Return the matched document IDs and the requested IDs that resolved to
        nothing in this generation. A source absent from the selected generation,
        or renamed or moved since it was built, cannot resolve: a `source_id` is
        derived from the current normalized relative path.
        """

        if not source_ids:
            return set(), []
        by_source_id: dict[str, set[str]] = {}
        for document in manifest.get("documents", []):
            relative = str(document.get("source_relative_path") or "")
            document_id = str(document.get("document_id") or "")
            if not relative or not document_id:
                continue
            try:
                source_id = stable_source_id(self.config.project_id, relative)
            except SourcePolicyError:
                continue
            by_source_id.setdefault(source_id, set()).add(document_id)
        matched: set[str] = set()
        unknown: list[str] = []
        for requested in source_ids:
            document_ids = by_source_id.get(requested)
            if document_ids is None:
                unknown.append(requested)
            else:
                matched.update(document_ids)
        return matched, unknown

    def _load_current_optional(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.config.current_path.exists():
            return None
        try:
            return load_current_generation(self.config.state_root)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc


class _ReadLease:
    """The generations one read holds, released when the read ends."""

    def __init__(self, service: ResearchService) -> None:
        self._service = service
        self._held: list[str] = []

    def hold(self, generation_id: str) -> None:
        readers = self._service._generation_readers
        readers[generation_id] = readers.get(generation_id, 0) + 1
        self._held.append(generation_id)

    async def release(self) -> None:
        if not self._held:
            return
        readers = self._service._generation_readers
        for generation_id in self._held:
            remaining = readers.get(generation_id, 0) - 1
            if remaining > 0:
                readers[generation_id] = remaining
            else:
                readers.pop(generation_id, None)
        self._held.clear()
        async with self._service._readers_left:
            self._service._readers_left.notify_all()
