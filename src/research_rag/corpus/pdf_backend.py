"""The PDF extraction backend: the resource-bounded Docling worker.

``docling`` is the only operational reader: every PDF is converted by a separate
resource-bounded child process, and the parent process never imports Docling.
``custom`` names the retired bundled text-only reader. It is kept only as a read
token so a generation built before the default moved keeps its recorded
provenance and an existing config selecting it still resolves; it is never a
selectable extractor.

The worker runs under ``systemd-run --user --scope`` with a 2 GiB memory cap, no
swap, and a 200% CPU quota. The worker itself verifies those exact cgroup values
before it imports Docling, so a missing or wrong boundary fails closed instead
of loading a model unbounded. A build keeps one warm worker for its page batches
(:class:`DoclingWorker`). Each batch has a wall-clock limit, and its expiry kills
the whole scope, so a hung conversion leaves no orphan.

The backend name, its version, its adapter policy, the versions of its
extraction packages, and its disabled-enrichment options are one fingerprint. It
invalidates extraction reuse only when it changes. A manifest that predates the
field carries the retired ``custom`` fingerprint: that recorded identity is read
as-is, so an existing generation keeps its provenance and stays searchable rather
than being relabelled from the operational default.

An extraction run reads the local cache. With ``runtime.offline = false`` the
worker may fetch a missing model lazily on first use, inside the same verified
boundary; with ``runtime.offline = true`` it may not fetch and a missing model
fails closed. ``doctor --prefetch-models`` remains the explicit way to provision
models ahead of an offline run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import IO, Any, Self

from ..project.config import ResearchConfig, child_process_environment
from ..project.policy import DEFAULT_PDF_BACKEND_FINGERPRINT, ResearchError
from ..project.settings import PDF_BACKEND_CHOICES
from . import scoped_child
from .docling_env import (
    DoclingEnv,
    ensure_docling_env,
    resolve_docling_env,
    spec_fingerprint,
    worker_command,
)

PDF_BACKEND_CUSTOM = "custom"
PDF_BACKEND_DOCLING = "docling"
# The names a caller may configure: `docling` only, because it is the sole
# operational reader. `PDF_BACKEND_CUSTOM` is accepted below as a read alias.
PDF_BACKENDS = PDF_BACKEND_CHOICES
# The values a configured setting may still carry. `custom` is the retired token
# an existing file or environment may hold; it resolves to the operational
# reader and is never a selectable extractor.
_KNOWN_PDF_BACKENDS = (PDF_BACKEND_CUSTOM, PDF_BACKEND_DOCLING)
DEFAULT_PDF_BACKEND = PDF_BACKEND_DOCLING

# The adapter's own emission policy. Bump it when the mapping from a Docling
# payload to units changes, so a recorded generation re-extracts rather than
# reusing units the new adapter would not produce. Version 2 carries the
# structural labels (heading/reference/caption/item) that gate cleaning; a
# generation with the version-1 adapter keeps its own fingerprint and provenance.
DOCLING_ADAPTER_POLICY_VERSION = 2
DOCLING_PACKAGE_REMEDY = (
    "run `research-rag doctor --prefetch-models` with a network connection, or "
    "set runtime.offline=false and run the first PDF ingest online; the managed "
    "environment is built on first use and pins docling and rapidocr together"
)

WALL_TIMEOUT_SECONDS = 180
# The job and reply shape `docling_worker.serve` speaks. The worker keeps its own
# copy, because it runs in an interpreter that cannot import this package.
WORKER_PROTOCOL = 2
# A worker scope lives this long at most. A batch that could outlive it starts a
# fresh worker, so every batch keeps its full wall limit.
WORKER_LIFETIME_SECONDS = 3600
# The tail of a worker's stderr that a failure report quotes.
_STDERR_TAIL_BYTES = 64 * 1024

# The fingerprint a generation with no recorded backend carries. It is the
# legacy default, so recording it does not invalidate an existing corpus. The
# string is owned by `project.policy`, shared with the checkpoint identity.
CUSTOM_BACKEND_FINGERPRINT = DEFAULT_PDF_BACKEND_FINGERPRINT

_DOCLING_OPTIONS = {
    "ocr": False,
    "vlm": False,
    "table_structure": False,
    "enrichments": False,
    "accelerator": "cpu",
    "threads": 2,
}
# One live Docling worker per process: held from a worker's start to its retirement.
_WORKER_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


class PdfBackendError(ResearchError):
    """A backend-infrastructure failure. It aborts the build; it is never an
    omission of a single source."""


class DoclingUnavailableError(PdfBackendError):
    """Docling or its hardened runner is not installed."""


class DoclingResourceError(PdfBackendError):
    """The resource boundary is missing or not the one that was promised."""


class DoclingConversionError(PdfBackendError):
    """The worker failed on the source itself. The caller maps this to the
    source-local omission path."""


class _BatchTimeoutError(DoclingResourceError):
    """A batch outran its wall limit. It is never retried."""


def normalize_pdf_backend(value: Any) -> str:
    """The operational reader a configured value selects.

    ``docling`` is the only operational reader. The retired ``custom`` token is a
    read alias: an existing config or environment selecting it resolves to
    ``docling`` rather than failing, so a project keeps loading without an
    on-disk rewrite. Any other value is refused.
    """

    text = str(value or "").strip().casefold()
    if text not in _KNOWN_PDF_BACKENDS:
        raise ValueError(
            f"ingestion.pdf_backend must be one of {', '.join(PDF_BACKENDS)}: {value!r}"
        )
    return PDF_BACKEND_DOCLING


def effective_maximum_unclean_percent(backend: str, configured: float) -> float:
    """The PDF loss budget the shared gate applies for this backend.

    The project's configured budget is shared unchanged across backends: a project
    that sets 2% keeps 2%, and the default keeps 1%. Selecting Docling changes how
    PDFs are read, never what the source health gate accepts.
    """

    normalize_pdf_backend(backend)
    return float(configured)


def pdf_backend_available(backend: str, config: ResearchConfig | None = None) -> bool:
    """Whether the operational reader can run a future build on this machine.

    Docling can when its managed environment is ready. A missing environment is a
    readiness warning for the next ingest, never a reason to report an existing
    servable generation as needing a rebuild or to refuse it.
    """

    normalize_pdf_backend(backend)
    return config is not None and resolve_docling_env(config) is not None


def legacy_custom_backend_record() -> dict[str, Any]:
    """The recorded identity of a generation built by the retired bundled reader.

    Preserved so a historical artifact keeps its own provenance. This is never a
    selectable extractor and never a target for a new build; it exists only to
    stamp or read what already ran. The value is a constant, so it matches a
    manifest that predates the ``extraction_backend`` field.
    """

    return {
        "backend": PDF_BACKEND_CUSTOM,
        "version": None,
        "adapter_policy_version": None,
        "components": {},
        "options": {},
        "fingerprint": CUSTOM_BACKEND_FINGERPRINT,
    }


def pdf_backend_record(
    backend: str, config: ResearchConfig | None = None
) -> dict[str, Any]:
    """The recorded identity of the operational reader and its fingerprint.

    The value is normalized, so a retired ``custom`` resolves to ``docling``: a
    new build always records the operational reader. Docling folds the managed
    environment's *spec* fingerprint, the adapter policy, every recorded package
    version, and the disabled options in. The spec fingerprint is stable before
    and after the first install, so an environment built at the start of a build
    does not move the checkpoint identity.
    """

    normalize_pdf_backend(backend)
    env = resolve_docling_env(config) if config is not None else None
    components = dict(env.versions) if env is not None else {}
    spec = env.spec_fingerprint if env is not None else spec_fingerprint()
    return _docling_record(components, spec)


def available_pdf_backend_record(
    backend: str, config: ResearchConfig
) -> dict[str, Any] | None:
    """The operational reader's record when it can run here, or None.

    One resolution of the managed environment answers both whether a build could
    read a PDF now and what it would record.
    """

    normalize_pdf_backend(backend)
    env = resolve_docling_env(config)
    return None if env is None else docling_backend_record_from_env(env)


def docling_backend_record_from_env(env: DoclingEnv) -> dict[str, Any]:
    """The recorded Docling identity for an already-resolved environment."""

    return _docling_record(dict(env.versions), env.spec_fingerprint)


def _docling_record(components: dict[str, str | None], spec: str) -> dict[str, Any]:
    version = components.get("docling")
    payload = json.dumps(
        {
            "backend": PDF_BACKEND_DOCLING,
            "spec_fingerprint": spec,
            "adapter_policy_version": DOCLING_ADAPTER_POLICY_VERSION,
            "components": components,
            "options": _DOCLING_OPTIONS,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return {
        "backend": PDF_BACKEND_DOCLING,
        "version": version,
        "adapter_policy_version": DOCLING_ADAPTER_POLICY_VERSION,
        "components": components,
        "options": dict(_DOCLING_OPTIONS),
        "fingerprint": f"pdf-backend:{PDF_BACKEND_DOCLING}:{spec}:{digest}",
    }


def pdf_backend_fingerprint(backend: str, config: ResearchConfig | None = None) -> str:
    return str(pdf_backend_record(backend, config)["fingerprint"])


def recorded_pdf_backend_fingerprint(manifest: dict[str, Any]) -> str:
    """The fingerprint a manifest carries, or the legacy default when absent."""

    record = manifest.get("extraction_backend")
    if isinstance(record, dict):
        fingerprint = record.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            return fingerprint
    return CUSTOM_BACKEND_FINGERPRINT


def _require_systemd_run() -> str:
    executable = shutil.which("systemd-run")
    if executable is None:
        raise DoclingUnavailableError(
            "PDF extraction uses Docling, which needs `systemd-run` to bound the "
            "conversion worker's memory, swap, and CPU. It is not on PATH. "
            "Install a systemd user session, then retry the ingest. "
            "No source was changed and the selected generation is unchanged."
        )
    if shutil.which("systemctl") is None:
        raise DoclingUnavailableError(
            "PDF extraction uses Docling, which needs `systemctl` to terminate "
            "the bounded worker scope. Install a systemd user session, then retry "
            "the ingest."
        )
    return executable


MODE_EXTRACT = "extract"
MODE_PREFETCH = "prefetch"


def docling_cache_root(model_cache_root: Path | None) -> Path:
    """The local Docling model cache, kept beside the shared model cache."""

    if model_cache_root is not None:
        return model_cache_root / "docling"
    return Path.home() / ".cache" / "research-rag" / "docling"


def docling_cache_has_models(model_cache_root: Path | None) -> bool:
    """Whether the local Docling cache holds at least one model file.

    A read-only readiness proxy. It never imports Docling and starts nothing;
    `doctor --prefetch-models` owns provisioning.
    """

    root = docling_cache_root(model_cache_root)
    try:
        return root.is_dir() and any(item.is_file() for item in root.rglob("*"))
    except OSError:
        return False


def _worker_environment(
    mode: str, offline: bool, model_cache_root: Path | None
) -> dict[str, str]:
    environment = child_process_environment(
        purpose="docling-offline" if offline and mode == MODE_EXTRACT else "docling"
    )
    cache = docling_cache_root(model_cache_root)
    environment["HF_HOME"] = str(cache)
    environment["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    # Offline extraction never fetches; an online extraction may fetch lazily on
    # its first call, inside this same boundary, and explicit provisioning always
    # may. Only the offline extraction case disables downloads.
    if mode == MODE_EXTRACT and offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    else:
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
    return environment


def _worker_failure(code: int | None, text: str) -> PdfBackendError:
    """The error a worker's failure code or exit status names."""

    if code == 5:
        return DoclingResourceError(
            f"Docling worker refused the resource boundary: {text}"
        )
    if code == 6:
        return DoclingUnavailableError(text or "Docling is not installed")
    if code == 3:
        return DoclingConversionError(text or "Docling conversion failed")
    return PdfBackendError(
        f"Docling worker failed (exit {code}): {text or 'no diagnostics'}"
    )


def _parse_reply(line: bytes) -> dict[str, Any]:
    try:
        reply = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PdfBackendError(
            f"Docling worker returned unparseable output: {exc}"
        ) from exc
    if not isinstance(reply, dict) or (
        reply.get("ok") and not isinstance(reply.get("result"), dict)
    ):
        raise PdfBackendError("Docling worker returned a non-object result")
    return reply


def _pump_replies(stream: IO[bytes], replies: queue.Queue[bytes | None]) -> None:
    """Queue each reply line, then None once the worker's stdout closes."""

    try:
        for line in iter(stream.readline, b""):
            replies.put(line)
    except (OSError, ValueError):
        pass
    finally:
        replies.put(None)


def _pump_stderr(stream: IO[bytes], tail: bytearray) -> None:
    """Drain the worker's stderr, so it never blocks on a full pipe."""

    with contextlib.suppress(OSError, ValueError):
        descriptor = stream.fileno()
        for chunk in iter(lambda: os.read(descriptor, 4096), b""):
            tail += chunk
            del tail[:-_STDERR_TAIL_BYTES]


class DoclingWorker:
    """One warm Docling worker that serves page batches until it is closed.

    Starting a worker imports Docling and loads its layout model, which costs
    about half as long as converting an eight-page batch. A build therefore keeps
    one worker for every batch it converts. Each batch keeps its own wall limit,
    and a timeout kills the scope.

    A batch's failure is reported only by a worker started for that batch. A
    worker that served an earlier batch and then fails is retired, and the batch
    runs once more in a fresh worker, so a long-lived worker never changes a
    batch's outcome. A timeout is not retried. A batch that could outlive the
    worker's scope also starts a fresh one.

    The process starts on the first batch, not at construction. Close the worker,
    or use it as a context manager, to release its memory.
    """

    def __init__(
        self,
        *,
        docling_env: DoclingEnv,
        offline: bool,
        model_cache_root: Path | None,
        mode: str = MODE_EXTRACT,
    ) -> None:
        self.docling_env = docling_env
        self.offline = bool(offline)
        self.model_cache_root = model_cache_root
        self.mode = mode
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._unit = ""
        self._expires = 0.0
        self._replies: queue.Queue[bytes | None] = queue.Queue()
        self._stderr = bytearray()
        self._pumps: list[threading.Thread] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def run(
        self, job: dict[str, Any], *, timeout: float, description: str
    ) -> dict[str, Any]:
        """Send one job and return the worker's result."""

        with self._lock:
            if self._process is not None and time.monotonic() + timeout > self._expires:
                self._retire()
            warm = self._process is not None
            try:
                return self._exchange(job, timeout, description)
            except _BatchTimeoutError:
                raise
            except PdfBackendError as exc:
                if not warm:
                    raise
                LOGGER.debug(
                    "docling: a warm worker failed on %s (%s); it runs again in a "
                    "fresh worker",
                    description,
                    exc,
                )
            return self._exchange(job, timeout, description)

    def close(self) -> None:
        with self._lock:
            self._retire()

    def _start(self, timeout: float) -> subprocess.Popen[bytes]:
        systemd_run = _require_systemd_run()
        environment = _worker_environment(
            self.mode, self.offline, self.model_cache_root
        )
        if not _WORKER_LOCK.acquire(timeout=max(1.0, timeout)):
            raise DoclingUnavailableError(
                "Another Docling worker is already running in this process."
            )
        lifetime = max(WORKER_LIFETIME_SECONDS, timeout)
        try:
            self._unit = f"research-rag-docling-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            self._expires = time.monotonic() + lifetime
            process = scoped_child.start_in_scope(
                systemd_run,
                worker_command(self.docling_env),
                unit=self._unit,
                lifetime=lifetime,
                env=environment,
            )
        except BaseException:
            _WORKER_LOCK.release()
            raise
        self._process = process
        self._replies = queue.Queue()
        self._stderr = bytearray()
        self._pumps = [
            threading.Thread(
                target=_pump_replies, args=(process.stdout, self._replies), daemon=True
            ),
            threading.Thread(
                target=_pump_stderr, args=(process.stderr, self._stderr), daemon=True
            ),
        ]
        for pump in self._pumps:
            pump.start()
        return process

    def _exchange(
        self, job: dict[str, Any], timeout: float, description: str
    ) -> dict[str, Any]:
        process = self._process or self._start(timeout)
        try:
            assert process.stdin is not None
            try:
                request = json.dumps({**job, "protocol": WORKER_PROTOCOL})
                process.stdin.write(request.encode("utf-8") + b"\n")
                process.stdin.flush()
            except OSError:
                line = None  # The worker exited before it read the job.
            else:
                try:
                    line = self._replies.get(timeout=timeout)
                except queue.Empty:
                    raise _BatchTimeoutError(
                        f"{description} exceeded the {timeout:g}s wall limit; the "
                        "worker scope was killed. Lower "
                        "ingestion.pdf_page_batch_size, then retry the ingest."
                    ) from None
            if line is None:
                raise self._exit_failure(process)
            reply = _parse_reply(line)
            if not reply.get("ok"):
                raise _worker_failure(
                    int(reply.get("code") or 4), str(reply.get("error") or "")
                )
        except BaseException:
            self._retire()
            raise
        return reply["result"]

    def _exit_failure(self, process: subprocess.Popen[bytes]) -> PdfBackendError:
        """The error for a worker that exited without replying."""

        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=15)
        for pump in self._pumps:
            pump.join(timeout=5)
        text = bytes(self._stderr).decode("utf-8", "replace").strip()
        return _worker_failure(process.returncode, text)

    def _retire(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            with contextlib.suppress(OSError):
                assert process.stdin is not None
                process.stdin.close()
            scoped_child.terminate_scope(self._unit, process)
        finally:
            for pump in self._pumps:
                pump.join(timeout=5)
            self._pumps = []
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
            _WORKER_LOCK.release()


def run_docling_batch(
    *,
    source_path: Path,
    start: int,
    end: int,
    offline: bool,
    model_cache_root: Path | None,
    docling_env: DoclingEnv,
    timeout: float = WALL_TIMEOUT_SECONDS,
    worker: DoclingWorker | None = None,
) -> dict[str, Any]:
    """Run one bounded Docling page batch and return the parsed worker result.

    ``start`` and ``end`` are 1-based physical page numbers, inclusive. With
    ``offline`` the worker may not fetch missing models and fails closed; without
    it the worker may fetch them lazily on first use, inside the same boundary.
    ``worker`` is a warm worker started for the same environment and offline
    flag; without one the batch runs in a worker of its own. Raises
    :class:`DoclingResourceError` when the boundary is missing or wrong,
    :class:`DoclingUnavailableError` when the runner or a required model is
    unavailable, and :class:`DoclingConversionError` when the worker failed on
    the source itself.
    """

    job = {
        "mode": MODE_EXTRACT,
        "source_path": str(source_path),
        "start": int(start),
        "end": int(end),
        "offline": bool(offline),
    }
    description = f"Docling conversion of {source_path.name} pages {start}-{end}"
    if worker is None:
        with DoclingWorker(
            docling_env=docling_env,
            offline=offline,
            model_cache_root=model_cache_root,
        ) as own:
            return own.run(job, timeout=timeout, description=description)
    if (worker.mode, worker.docling_env, worker.offline, worker.model_cache_root) != (
        MODE_EXTRACT,
        docling_env,
        bool(offline),
        model_cache_root,
    ):
        raise ValueError("The Docling worker was started for another extraction")
    return worker.run(job, timeout=timeout, description=description)


def prefetch_docling_models(
    config: ResearchConfig,
    *,
    timeout: float = WALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Provision the Docling environment and models through the managed worker.

    Only the explicit `doctor --prefetch-models` path calls this. It builds the
    managed environment if needed, then fetches models, both online and inside
    the verified boundary.
    """

    # This explicit provisioning command opts into network access even when
    # ordinary project ingestion is offline, like the other model prefetches.
    env = ensure_docling_env(config, allow_install=True)
    with DoclingWorker(
        docling_env=env,
        offline=False,
        model_cache_root=config.model_cache_root,
        mode=MODE_PREFETCH,
    ) as worker:
        return worker.run(
            {"mode": MODE_PREFETCH},
            timeout=timeout,
            description="Docling model provisioning",
        )
