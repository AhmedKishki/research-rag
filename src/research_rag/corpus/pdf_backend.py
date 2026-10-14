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
of loading a model unbounded. A wall-clock timeout kills the whole scope, so a
hung conversion leaves no orphan.

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

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

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
_WORKER_LOCK = threading.Lock()


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


def _launch_worker(
    job: dict[str, Any],
    *,
    mode: str,
    offline: bool,
    model_cache_root: Path | None,
    docling_env: DoclingEnv,
    timeout: float,
    description: str,
) -> dict[str, Any]:
    """Start the managed worker, map its exit code, and return its JSON result.

    The managed interpreter runs isolated (``-I``), so no inherited
    ``PYTHONPATH`` or the app's own site-packages reach it; only the shipped
    worker module is pointed at from the app source. Docling comes solely from
    the managed environment.
    """

    systemd_run = _require_systemd_run()
    unit = f"research-rag-docling-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    environment = _worker_environment(mode, offline, model_cache_root)
    if not _WORKER_LOCK.acquire(timeout=max(1.0, timeout)):
        raise DoclingUnavailableError(
            "Another Docling conversion is already running in this process."
        )
    try:
        returncode, stdout, stderr = scoped_child.run_in_scope(
            systemd_run,
            worker_command(docling_env),
            unit=unit,
            timeout=timeout,
            env=environment,
            stdin=json.dumps(job).encode("utf-8"),
        )
    except subprocess.TimeoutExpired as exc:
        raise DoclingResourceError(
            f"{description} exceeded the {timeout:g}s wall limit; the worker "
            "scope was killed. Lower ingestion.pdf_page_batch_size, then retry "
            "the ingest."
        ) from exc
    finally:
        _WORKER_LOCK.release()

    text = stderr.decode("utf-8", "replace").strip()
    if returncode == 5:
        raise DoclingResourceError(
            f"Docling worker refused the resource boundary: {text}"
        )
    if returncode == 6:
        raise DoclingUnavailableError(text or "Docling is not installed")
    if returncode == 3:
        raise DoclingConversionError(text or "Docling conversion failed")
    if returncode != 0:
        raise PdfBackendError(
            f"Docling worker failed (exit {returncode}): {text or 'no diagnostics'}"
        )
    try:
        result = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PdfBackendError(
            f"Docling worker returned unparseable output: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise PdfBackendError("Docling worker returned a non-object result")
    return result


def run_docling_batch(
    *,
    source_path: Path,
    start: int,
    end: int,
    offline: bool,
    model_cache_root: Path | None,
    docling_env: DoclingEnv,
    timeout: float = WALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one bounded Docling page batch and return the parsed worker result.

    ``start`` and ``end`` are 1-based physical page numbers, inclusive. With
    ``offline`` the worker may not fetch missing models and fails closed; without
    it the worker may fetch them lazily on first use, inside the same boundary.
    Raises :class:`DoclingResourceError` when the boundary is missing or wrong,
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
    return _launch_worker(
        job,
        mode=MODE_EXTRACT,
        offline=bool(offline),
        model_cache_root=model_cache_root,
        docling_env=docling_env,
        timeout=timeout,
        description=f"Docling conversion of {source_path.name} pages {start}-{end}",
    )


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
    return _launch_worker(
        {"mode": MODE_PREFETCH},
        mode=MODE_PREFETCH,
        offline=False,
        model_cache_root=config.model_cache_root,
        docling_env=env,
        timeout=timeout,
        description="Docling model provisioning",
    )
