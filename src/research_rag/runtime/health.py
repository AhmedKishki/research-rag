"""Named dependency checks for one project, read by `status` and by `doctor`.

`AGENTS.md` states the four states a check may report.
"""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..corpus.docling_env import resolve_docling_env
from ..corpus.pdf_backend import (
    docling_cache_has_models,
    docling_cache_root,
)
from ..project.config import ResearchConfig, project_command, runtime_root_claim_problem
from ..project.state_files import LOCK_FILE, process_alive
from ..retrieval.embeddings import resolve_embedding_model
from ..retrieval.model_cache import snapshot_is_complete, snapshot_path
from ..retrieval.rerankers import RERANKER_REQUIRED_FILES, resolve_reranker_model
from . import version as version_module

OK = "ok"
WARN = "warn"
BLOCKED = "blocked"
UNKNOWN = "unknown"

# A new generation is written while the retained ones stay on disk, so a build
# needs about as much free space as the generations already occupy. A build blocks
# below that, and the report calls the headroom thin below twice it.
CAPACITY_APPROACHING_FACTOR = 2


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    state: str
    reason: str
    remedy_command: str | None = None

    @property
    def blocked(self) -> bool:
        return self.state == BLOCKED

    @property
    def degraded(self) -> bool:
        return self.state == WARN

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "state": self.state,
            "reason": self.reason,
            "remedy_command": self.remedy_command,
        }

    def as_disclosure(self) -> dict[str, Any]:
        """The shape `blocked_by` and `degraded` use in a tool answer."""

        return {
            "check": self.name,
            "reason": self.reason,
            "remedy": self.remedy_command,
        }


@dataclass(frozen=True, slots=True)
class HealthReport:
    checks: tuple[Check, ...]

    def named(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    @property
    def blocked_by(self) -> list[dict[str, Any]]:
        return [check.as_disclosure() for check in self.checks if check.blocked]

    @property
    def degraded(self) -> list[dict[str, Any]]:
        return [check.as_disclosure() for check in self.checks if check.degraded]

    @property
    def not_checked(self) -> list[str]:
        return [check.name for check in self.checks if check.state == UNKNOWN]

    @property
    def has_blocker(self) -> bool:
        return any(check.blocked for check in self.checks)

    def as_status_fields(self) -> dict[str, Any]:
        return {
            "checks": [check.as_dict() for check in self.checks],
            "blocked_by": self.blocked_by,
            "degraded": self.degraded,
            "not_checked": self.not_checked,
        }


def doctor_command(config: ResearchConfig, *arguments: str) -> str:
    return project_command(config.project_root, "doctor", *arguments)


def _command(config: ResearchConfig, *arguments: str) -> str:
    return project_command(config.project_root, *arguments)


def _model_check(
    config: ResearchConfig,
    name: str,
    revision: str,
    label: str,
    *,
    repository: str,
    required_files: tuple[str, ...],
    consequence: str,
    offline_state: str,
) -> Check:
    """One pinned model against the cache it must already be in.

    `offline_state` is what an absent model takes with no download possible, and
    it differs per model: no embedding model, no dense build; no reranker, an
    unranked answer.
    """
    snapshot = snapshot_path(config.model_cache_root, repository, revision)
    if snapshot_is_complete(snapshot, required_files):
        return Check(
            label,
            OK,
            f"The pinned {name} ({repository}@{revision}) is cached at {snapshot}.",
        )
    prefetch = doctor_command(config, "--prefetch-models")
    if not config.offline:
        return Check(
            label,
            WARN,
            f"The pinned {name} ({repository}@{revision}) is not fully cached at {snapshot}; "
            f"the first call that needs it downloads it, and until then {consequence}.",
            prefetch,
        )
    if offline_state == OK:
        return Check(
            label,
            WARN,
            f"The pinned {name} ({repository}@{revision}) is not fully cached at {snapshot}; {consequence}.",
            prefetch,
        )
    return Check(
        label,
        offline_state,
        f"The pinned {name} ({repository}@{revision}) is not fully cached at {snapshot}, and "
        f"offline mode forbids downloading it, so {consequence}.",
        prefetch,
    )


def _docling_backend_check(config: ResearchConfig) -> Check | None:
    """Readiness of the operational PDF reader, without importing Docling.

    Docling is the only PDF extractor, so this reports readiness for the *next*
    PDF build on every project. It never decides whether the selected generation
    can be served: searching an existing generation, including one built by the
    retired bundled reader, never needs Docling. A missing environment or cache
    only warns and is remedied before the next ingest; it never blocks and never
    marks the corpus unserviceable or stale.
    """

    cache = docling_cache_root(config.model_cache_root)
    env = resolve_docling_env(config)
    prefetch = doctor_command(config, "--prefetch-models")
    if env is None:
        # The environment is built lazily on the first real conversion. An
        # online run builds it; an offline run cannot, so it needs provisioning.
        if config.offline:
            reason = (
                "PDF extraction uses Docling, whose managed environment is not "
                "built, and runtime.offline forbids building it. Existing "
                "generations still serve; provision it before the next PDF "
                "ingest."
            )
        else:
            reason = (
                "PDF extraction uses Docling; its managed environment is not "
                "built yet. The first PDF ingest builds it lazily; run the "
                "remedy to provision it ahead of that build."
            )
        return Check("docling_backend", WARN, reason, prefetch)
    if not docling_cache_has_models(config.model_cache_root):
        if config.offline:
            reason = (
                "PDF extraction uses Docling and its environment is built, but "
                f"no models are cached at {cache}. Offline extraction cannot "
                "fetch them, so provision them before the next PDF ingest."
            )
        else:
            reason = (
                "PDF extraction uses Docling and its environment is built, but "
                f"no models are cached at {cache}. The first PDF ingest fetches "
                "them lazily; provision them first to keep that build offline or "
                "to avoid paying the fetch during it."
            )
        return Check("docling_backend", WARN, reason, prefetch)
    return Check(
        "docling_backend",
        OK,
        f"Docling is installed and its models are cached at {cache}.",
    )


def _direct_retrieval_check(config: ResearchConfig) -> Check:
    """Read package metadata; normal retrieval uses these direct libraries."""
    required = {"chonkie": "1.7.0", "tiktoken": "0.14.0"}
    problems = []
    for package in ("chonkie", "tiktoken", "bm25s", "numba", "scipy", "PyStemmer"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed")
        else:
            expected = required.get(package)
            if expected is not None and installed != expected:
                problems.append(
                    f"{package} {installed} is installed; expected {expected}"
                )
    if problems:
        return Check(
            "direct_retrieval",
            BLOCKED,
            "; ".join(problems),
            "uv sync --locked",
        )
    return Check(
        "direct_retrieval",
        OK,
        "Compatible Chonkie and BM25 dependencies are installed; normal retrieval "
        "uses these direct libraries.",
    )


def _chunk_tokenizer_check(config: ResearchConfig) -> Check:
    """Read verified cache inputs without importing the tokenizer or fetching."""
    from ..retrieval.direct import tokenizer_cache_status

    if tokenizer_cache_status(config.model_cache_root):
        return Check("chunk_tokenizer", OK, "The compatible GPT-2 tokenizer is cached.")
    reason = (
        "The GPT-2 tokenizer cache is missing or invalid. Existing BM25/dense "
        "indexes remain searchable without a token-length floor; chunking and "
        "token-length filtering need the cache. "
    )
    reason += (
        "Offline mode forbids fetching it."
        if config.offline
        else "The first operation that needs it downloads it."
    )
    return Check(
        "chunk_tokenizer", WARN, reason, doctor_command(config, "--prefetch-models")
    )


def _lock_check(config: ResearchConfig) -> Check:
    path = config.state_root / LOCK_FILE
    if not path.is_file():
        return Check("lock", OK, f"No other process holds {path}.")
    try:
        recorded = path.read_text(encoding="utf-8").strip()
        held_since = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
            timespec="seconds"
        )
    except OSError as exc:
        return Check("lock", WARN, f"The project lock cannot be read: {exc}")
    owner = int(recorded) if recorded.isdigit() else None
    if owner is None or owner == os.getpid() or not process_alive(owner):
        return Check(
            "lock",
            OK,
            f"{path} names no running process other than this one.",
        )
    return Check(
        "lock",
        WARN,
        f"Process {owner} has held {path} since {held_since}; write calls are "
        "rejected while it holds the project.",
        _command(config, "stop", "--servers"),
    )


def _generation_check(config: ResearchConfig, status: Mapping[str, Any]) -> Check:
    """Weigh the selected generation as serviceable, or name why it is not.

    `upgrade_reasons` carries only serving incompatibility. A generation built
    for another project is not served here, but a rebuild is the last resort: the
    finding blocks and names selecting a matching retained generation, and only
    suggests building when none exists. An embedding-identity mismatch makes
    dense and hybrid scoring compare one model's vectors against another's, so it
    warns and names the choice: restore the recorded model, or explicitly build
    matching vectors. Policy differences that leave the generation serviceable
    are recorded in `generation_diagnostics`, which is not a health finding: the
    corpus answers under its recorded policy, so the check is `ok`.
    """

    reasons = list(status.get("upgrade_reasons") or [])
    if "project_identity" in reasons:
        return Check(
            "generation",
            BLOCKED,
            "The selected generation belongs to a different project and cannot be "
            "served here. Select a retained generation that belongs to this "
            "project; build one only if none is retained.",
            _command(config, "generations"),
        )
    ingest = _command(config, "ingest")
    if not status.get("ready"):
        message = str(status.get("message") or "No knowledge-base generation exists.")
        return Check("generation", BLOCKED, message, ingest)
    if reasons:
        return Check(
            "generation",
            WARN,
            "The selected generation's dense index was built with a different "
            "embedding model, so dense and hybrid scoring would compare one model's "
            "vectors against another's; BM25 still serves. Restore the recorded "
            "embedding model, or ingest to rebuild the dense index with the current "
            "one.",
            ingest,
        )
    if status.get("stale"):
        return Check(
            "generation",
            WARN,
            "The sources differ from the selected generation; ingest indexes them.",
            ingest,
        )
    return Check(
        "generation",
        OK,
        f"The selected generation {status.get('generation_id')} serves this corpus.",
    )


def _capacity_check(config: ResearchConfig, status: Mapping[str, Any]) -> Check:
    try:
        free = shutil.disk_usage(config.state_root).free
    except OSError as exc:
        return Check("capacity", UNKNOWN, f"Free space cannot be read: {exc}")
    required = int(status.get("retained_generation_bytes") or 0)
    if required <= 0:
        return Check(
            "capacity",
            OK,
            f"{free} bytes are free and no generation occupies the state root yet.",
        )
    if free < required:
        return Check(
            "capacity",
            BLOCKED,
            f"{free} bytes are free and a build needs about {required}, the size "
            "of the generations already on disk. The doctor never deletes an "
            "index; free space yourself with 'research-rag remove-generation', or "
            "build elsewhere.",
        )
    if free < required * CAPACITY_APPROACHING_FACTOR:
        return Check(
            "capacity",
            WARN,
            f"{free} bytes are free against the {required} a build needs.",
        )
    return Check(
        "capacity",
        OK,
        f"{free} bytes are free against the {required} a build needs.",
    )


def _code_currency_check() -> Check:
    """The versions are read through their module, so an answer follows the module
    it patched.
    """
    running = version_module.APP_VERSION
    installed = version_module.installed_version()
    drift = version_module.checkout_drift()
    if drift is not None:
        # Two checkouts of this package are the reported cause of a process
        # that answers with metadata it could not have written, and a version
        # comparison cannot see it: both copies report the same version.
        return Check(
            "code_currency",
            WARN,
            drift,
            "Stop this app's processes, then start the checkout you mean.",
        )
    if not version_module.restart_required():
        return Check(
            "code_currency",
            OK,
            f"The running app is the installed version {running}.",
        )
    return Check(
        "code_currency",
        WARN,
        f"The running app started with version {running}; {installed} is "
        "installed. Restart this app so it runs the installed code.",
    )


def _service_check(config: ResearchConfig, reason: str) -> Check:
    """The check that says this project's own state could not be read.

    The corpus, the generation, and the models are all unknown until it can be, so
    the report says that rather than reporting the empty project this run did not
    observe.
    """

    return Check(
        "service",
        BLOCKED,
        f"This project's own state could not be read, so nothing about its "
        f"corpus, its generation, or its indexes is known: {reason}",
        _command(config, "doctor"),
    )


def health_report(
    config: ResearchConfig,
    status: Mapping[str, Any],
    *,
    status_unavailable: str | None = None,
) -> HealthReport:
    """`status` is the payload the caller already produced, so the dependency and
    status answers describe one state.

    `status_unavailable` is why the caller could not produce one. An empty status
    is an answer about a project holding nothing; this is the other case, and the
    report must not present it as the first.
    """

    embedding = resolve_embedding_model(config.settings.embedding_model)
    reranker, reranker_revision = resolve_reranker_model(config.reranker_model)
    root = config.state_root
    # A relocated root carries a marker naming its project; the default root is
    # inside the project, which owns it, and asks for nothing.
    if config.runtime_root is None:
        identity = Check(
            "project_identity",
            OK,
            f"This project keeps its state inside {root} and owns it by "
            f"construction; its identifier is {config.project_id}.",
        )
    else:
        claim = runtime_root_claim_problem(root, config.project_id)
        identity = (
            Check("project_identity", BLOCKED, claim)
            if claim is not None
            else Check(
                "project_identity",
                OK,
                f"This project's state root is {root} and its identifier "
                f"{config.project_id} agrees with the marker there.",
            )
        )
    if not root.is_absolute():
        runtime_root = Check(
            "runtime_root",
            BLOCKED,
            f"The state root is not an absolute path: {root}.",
        )
    elif not root.is_dir():
        runtime_root = Check(
            "runtime_root",
            BLOCKED,
            f"The state root is not a directory: {root}.",
        )
    elif not os.access(root, os.W_OK | os.X_OK):
        runtime_root = Check(
            "runtime_root",
            BLOCKED,
            f"The state root is not writable: {root}.",
            f"chmod u+w {shlex.quote(str(root))}",
        )
    else:
        runtime_root = Check(
            "runtime_root",
            OK,
            f"The state root {root} is writable.",
        )
    docling_backend = _docling_backend_check(config)
    return HealthReport(
        checks=(
            *(
                (_service_check(config, status_unavailable),)
                if status_unavailable
                else ()
            ),
            identity,
            runtime_root,
            _direct_retrieval_check(config),
            _chunk_tokenizer_check(config),
            *((docling_backend,) if docling_backend is not None else ()),
            _model_check(
                config,
                embedding.name,
                embedding.revision,
                "embedding_model",
                repository=embedding.repository,
                required_files=embedding.required_files,
                consequence="no build can be dense",
                offline_state=BLOCKED,
            ),
            _model_check(
                config,
                reranker,
                reranker_revision,
                "reranker_model",
                repository=reranker,
                required_files=RERANKER_REQUIRED_FILES,
                consequence="every search falls back to the unranked order",
                offline_state=OK,
            ),
            _lock_check(config),
            (
                Check(
                    "generation",
                    BLOCKED,
                    "This project's generation is unknown because its state could "
                    "not be read.",
                    _command(config, "ingest"),
                )
                if status_unavailable is not None
                else _generation_check(config, status)
            ),
            _capacity_check(config, status),
            _code_currency_check(),
        )
    )
