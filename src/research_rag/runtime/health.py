"""Named dependency checks for one project, read by `status` and by `doctor`.

`AGENTS.md` states the four states a check may report.
"""

from __future__ import annotations

import os
import shlex
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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

_VANILLA_CACHE_LOCK = threading.Lock()
# The managed runtime is validated by hashing its whole tree, which reads every
# byte of it. One process keeps the answer and re-reads only when the marker
# identifying the installed snapshot changes, so a repeated status costs one
# marker stat.
_VANILLA_CACHE: dict[tuple[Any, ...], Check] = {}


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.as_dict() for check in self.checks],
            "blocked_by": self.blocked_by,
            "degraded": self.degraded,
            "not_checked": self.not_checked,
        }

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


def _marker_key(runtime: Any, root: Path, offline: bool) -> tuple[Any, ...]:
    """The cache key for one managed runtime: path, mode, and the marker's
    `(mtime, size)`.

    An absent snapshot blocks offline and only warns online, so a cached answer
    for one mode is not an answer for the other.
    """
    name = getattr(runtime, "MARKER_FILENAME", ".vanilla-ultra-rag-runtime.json")
    try:
        observed = (root / name).stat()
    except OSError:
        return (str(root), offline, None, None)
    return (str(root), offline, observed.st_mtime_ns, observed.st_size)


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
    """Readiness of the selected optional PDF backend, without importing Docling.

    This reports readiness for the *next* PDF build, not whether the selected
    generation can be served: searching an existing generation never needs
    Docling. A missing backend or cache warns and is remedied before the next
    ingest, and never blocks or marks the corpus unserviceable.
    """

    if config.settings.pdf_backend != "docling":
        return None
    cache = docling_cache_root(config.model_cache_root)
    env = resolve_docling_env(config)
    prefetch = doctor_command(config, "--prefetch-models")
    if env is None:
        # The environment is built lazily on the first real conversion. An
        # online run builds it; an offline run cannot, so it needs provisioning.
        if config.offline:
            reason = (
                "ingestion.pdf_backend=docling is selected but its managed "
                "environment is not built, and runtime.offline forbids building "
                "it. Existing generations still serve; provision it before the "
                "next PDF ingest."
            )
        else:
            reason = (
                "ingestion.pdf_backend=docling is selected; its managed "
                "environment is not built yet. The first PDF ingest builds it "
                "lazily; run the remedy to provision it ahead of that build."
            )
        return Check("docling_backend", WARN, reason, prefetch)
    if not docling_cache_has_models(config.model_cache_root):
        if config.offline:
            reason = (
                "ingestion.pdf_backend=docling is selected and its environment "
                f"is built, but no models are cached at {cache}. Offline "
                "extraction cannot fetch them, so provision them before the next "
                "PDF ingest."
            )
        else:
            reason = (
                "ingestion.pdf_backend=docling is selected and its environment "
                f"is built, but no models are cached at {cache}. The first PDF "
                "ingest fetches them lazily; provision them first to keep that "
                "build offline or to avoid paying the fetch during it."
            )
        return Check("docling_backend", WARN, reason, prefetch)
    return Check(
        "docling_backend",
        OK,
        f"Docling is installed and its models are cached at {cache}.",
    )


def _validate_managed_runtime(
    config: ResearchConfig, root: Path, runtime: Any
) -> Check:
    validator = getattr(runtime, "validate_managed_runtime", None)
    install = "research-rag-runtime"
    if validator is None:
        return Check(
            "vanilla_runtime",
            UNKNOWN,
            "The installed vanilla release does not expose runtime validation.",
        )
    if not root.is_dir():
        if config.offline:
            return Check(
                "vanilla_runtime",
                BLOCKED,
                f"The managed UltraRAG runtime is not installed at {root}, and "
                "offline mode forbids downloading it.",
                install,
            )
        return Check(
            "vanilla_runtime",
            WARN,
            f"The managed UltraRAG runtime is not installed at {root}; the first "
            "call that needs it downloads it.",
            install,
        )
    try:
        validator(root)
    except Exception as exc:  # noqa: BLE001 - any failure is a blocked runtime.
        # Every failure blocks, including an exception type the release did not
        # predict.
        reason = str(exc)
        if getattr(exc, "difference", None) is None:
            reason += (
                " This vanilla release does not name the differing file; run "
                f"'{install} --offline' to see the full message."
            )
        return Check(
            "vanilla_runtime",
            BLOCKED,
            reason,
            doctor_command(config, "--repair-runtime"),
        )
    return Check(
        "vanilla_runtime",
        OK,
        f"The managed UltraRAG runtime at {root} matches the pinned snapshot.",
    )


def _vanilla_runtime_check(config: ResearchConfig) -> Check:
    try:
        from ..gateway import runtime as vanilla

        locator = getattr(vanilla, "managed_runtime_path", None)
    except Exception as exc:  # noqa: BLE001 - a broken install fails to import.
        # A broken install fails on import in more than one way, and the report
        # names that rather than raising it.
        return Check(
            "vanilla_runtime",
            UNKNOWN,
            f"The pinned vanilla runtime package cannot be used: {exc}",
        )
    if locator is None:
        return Check(
            "vanilla_runtime",
            UNKNOWN,
            "The installed vanilla release exposes no managed runtime path, so "
            "this installation cannot be checked.",
        )
    root = locator(config.runtime_cache_root)
    key = _marker_key(vanilla, root, config.offline)
    with _VANILLA_CACHE_LOCK:
        cached = _VANILLA_CACHE.get(key)
    if cached is not None:
        return cached
    check = _validate_managed_runtime(config, root, vanilla)
    with _VANILLA_CACHE_LOCK:
        _VANILLA_CACHE[key] = check
    return check


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
    ingest = _command(config, "ingest")
    if not status.get("ready"):
        message = str(status.get("message") or "No knowledge-base generation exists.")
        return Check("generation", BLOCKED, message, ingest)
    reasons = list(status.get("upgrade_reasons") or [])
    if reasons:
        return Check(
            "generation",
            WARN,
            "The selected generation was built by an older policy ("
            + ", ".join(reasons)
            + "); ingest rebuilds it.",
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
            _vanilla_runtime_check(config),
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
