"""The bounded subprocess that serves Docling PDF page batches.

This module is the worker entry point the parent starts under
``systemd-run --user --scope`` with a memory, swap, and CPU bound. It imports
Docling **inside** a :class:`Session`, after it has verified the resource
boundary it was promised, so a missing or wrong boundary fails closed before a
model is loaded. The parent process never imports Docling.

It reads one JSON job per stdin line and writes one JSON reply per stdout line,
until stdin closes. Every job carries ``"protocol": 2``; a job without it came
from an app older than this file and is refused with exit status 2. The first job's mode and offline flag fix the boundary the
worker verifies, and every later job must carry the same ones, so the converter
and its model load once per worker. Extraction jobs carry
``{"mode": "extract", "source_path", "start", "end", "offline"}`` and succeed
with ``{"version", "options", "pages", "diagnostics", "image_only_pages",
"pdf_metadata", "page_labels"}``. An offline extraction may not fetch; an online
one may fetch a missing model lazily on its first call, inside this boundary. A
``{"mode": "prefetch"}`` job, run only by the explicit
``doctor --prefetch-models`` path, provisions the models and succeeds with
``{"version", "mode", "ready"}``.

A reply is ``{"ok": true, "result": ...}`` or
``{"ok": false, "code": ..., "error": ...}``. A failed job ends the worker with
that code as its exit status, so a worker never converts again after a failure.
Stdout carries replies alone: anything a library prints goes to stderr.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

# The worker runs two ways. Imported as `research_rag.corpus.docling_worker` by
# tests, it takes the sibling adapter through its package. Executed directly by
# the managed interpreter (`envpython -I <worker file>`), it has no package, so
# it loads the adapter from the same directory by absolute path. It never adds
# the app's source or site-packages to `sys.path`.
if __package__ in (None, ""):
    _adapter_spec = importlib.util.spec_from_file_location(
        "_research_rag_docling_adapter", Path(__file__).with_name("docling_adapter.py")
    )
    if _adapter_spec is None or _adapter_spec.loader is None:  # pragma: no cover
        raise SystemExit("cannot load the shipped Docling adapter beside the worker")
    _adapter_module = importlib.util.module_from_spec(_adapter_spec)
    _adapter_spec.loader.exec_module(_adapter_module)
    adapt_document = _adapter_module.adapt_document
else:
    from .docling_adapter import adapt_document

MEMORY_MAX_BYTES = 2 * 1024**3
CPU_MAX = ("200000", "100000")
WORKER_NICE = 10

MODE_EXTRACT = "extract"
MODE_PREFETCH = "prefetch"
# The parent's `pdf_backend.WORKER_PROTOCOL`. An app started before the worker
# file changed sends jobs without it, and the worker refuses them rather than
# answer in a shape that app would misread.
PROTOCOL = 2
_PROTOCOL_REMEDY = (
    "The Docling worker file is newer than the research-rag app that started it. "
    "Restart the app, then run ingest again."
)

# Only the explicit provisioning path may reach the network.
_PREFETCH_REMEDY = (
    "run `research-rag doctor --prefetch-models` once with a network connection "
    "to provision the Docling models, then re-run ingestion"
)


class WorkerError(RuntimeError):
    pass


class ResourceBoundaryError(WorkerError):
    pass


class WorkerUnavailableError(WorkerError):
    pass


def _own_cgroup() -> Path:
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - kernel interface always present
        raise ResourceBoundaryError(f"Cannot read /proc/self/cgroup: {exc}") from exc
    for line in text.splitlines():
        if line.startswith("0::"):
            relative = line[3:].strip().lstrip("/")
            return Path("/sys/fs/cgroup") / relative
    raise ResourceBoundaryError("No cgroup v2 entry in /proc/self/cgroup")


def verify_resource_boundary(
    cgroup: Path | None = None,
    *,
    memory_max_bytes: int = MEMORY_MAX_BYTES,
    cpu_max: tuple[str, str] = CPU_MAX,
) -> dict[str, str]:
    """Verify the cgroup bounds the parent promised, before any library import.

    Returns the exact strings read back so a caller can report them. Raises
    :class:`ResourceBoundaryError` when a file is missing or a value differs.
    """

    root = cgroup if cgroup is not None else _own_cgroup()
    try:
        memory_max = (root / "memory.max").read_text(encoding="utf-8").strip()
        swap_max = (root / "memory.swap.max").read_text(encoding="utf-8").strip()
        cpu_raw = (root / "cpu.max").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ResourceBoundaryError(
            f"Resource boundary is not readable under {root}: {exc}"
        ) from exc
    if memory_max != str(memory_max_bytes):
        raise ResourceBoundaryError(
            f"memory.max is {memory_max!r}, expected {memory_max_bytes}"
        )
    if swap_max != "0":
        raise ResourceBoundaryError(f"memory.swap.max is {swap_max!r}, expected '0'")
    parts = cpu_raw.split()
    if parts != [cpu_max[0], cpu_max[1]]:
        raise ResourceBoundaryError(
            f"cpu.max is {cpu_raw!r}, expected {cpu_max[0]} {cpu_max[1]}"
        )
    return {"memory.max": memory_max, "memory.swap.max": swap_max, "cpu.max": cpu_raw}


def _verify_environment(mode: str, offline: bool) -> None:
    """The boundary includes the inherited environment the parent set.

    A worker that was not started through ``child_process_environment`` is a
    misconfiguration. An offline extraction must carry the no-download markers;
    an online extraction and explicit provisioning must not, because both may
    fetch.
    """

    import os

    if os.environ.get("RESEARCH_RAG_MANAGED_CHILD") != "1":
        raise ResourceBoundaryError("Worker was not started as a managed child process")
    offline_markers = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    if mode == MODE_EXTRACT and offline:
        if not offline_markers:
            raise ResourceBoundaryError("Offline extraction must disable downloads")
    elif offline_markers:
        raise ResourceBoundaryError(
            "A fetching worker must not run with downloads disabled"
        )


def apply_and_verify_nice() -> int:
    """Raise this worker's niceness by +10 and verify it before heavy imports.

    A Docling conversion must yield the CPU to interactive work, so the niceness
    is applied and read back before the library loads; a machine that refuses it
    fails closed rather than running unbounded.
    """

    os.nice(WORKER_NICE)
    niceness = os.nice(0)
    if niceness < WORKER_NICE:
        raise ResourceBoundaryError(
            f"Worker niceness is {niceness}, expected at least {WORKER_NICE}"
        )
    return niceness


def _pdf_metadata_and_labels(
    source_path: Path,
    start: int,
    end: int,
) -> tuple[dict[str, Any], dict[int, str], int]:
    """PDF catalog metadata, physical page labels, and image-only page count.

    All from PyMuPDF against the original, before Docling sees it. An image-only
    page carries at least one embedded image and no native text layer; it is not
    assumed to be a full-page raster scan, and a page with an image and a text
    layer is not counted.
    """

    import pymupdf

    try:
        document = pymupdf.open(source_path)
    except Exception as exc:
        raise WorkerError(f"Cannot open PDF source: {source_path}") from exc
    try:
        if document.needs_pass:
            raise WorkerError(f"Password-protected PDF is unsupported: {source_path}")
        metadata = dict(document.metadata or {})
        metadata["language"] = str(getattr(document, "language", "") or "")
        labels: dict[int, str] = {}
        image_only_pages = 0
        for page_no in range(start, end + 1):
            if page_no < 1 or page_no > document.page_count:
                continue
            page = document.load_page(page_no - 1)
            try:
                labels[page_no] = str(page.get_label() or page_no)
            except (RuntimeError, ValueError, IndexError):
                labels[page_no] = str(page_no)
            has_images = False
            has_text = False
            try:
                has_images = bool(page.get_images(full=True))
                has_text = bool(page.get_text("text").strip())
            except (RuntimeError, ValueError):
                has_images = False
                has_text = False
            if has_images and not has_text:
                image_only_pages += 1
        return metadata, labels, image_only_pages
    finally:
        document.close()


_MISSING_MODEL_MARKERS = (
    "offline",
    "not found in cache",
    "cannot find",
    "can't load",
    "couldn't reach",
    "connection",
    "connectionerror",
    "localentrynotfound",
    "cache",
    "download",
)


def _looks_like_a_missing_model(message: str) -> bool:
    lowered = message.casefold()
    return any(marker in lowered for marker in _MISSING_MODEL_MARKERS)


def _status_name(result: Any) -> str:
    status = getattr(result, "status", None)
    if status is None:
        return ""
    name = getattr(status, "name", None)
    if name:
        return str(name).upper()
    return str(status).rsplit(".", 1)[-1].upper()


def conversion_result_problem(result: Any) -> tuple[str, bool] | None:
    """Why a conversion did not fully succeed, and whether a model is missing.

    Only ``SUCCESS`` is acceptable. A ``PARTIAL_SUCCESS`` or ``FAILURE`` result
    must be rejected before its document is exported, so a partly converted page
    is never mistaken for evidence. The second element is true when any error
    names a model or download problem, which the caller maps to the
    provisioning remedy rather than a source omission.
    """

    name = _status_name(result)
    if name in {"", "SUCCESS"}:
        return None
    errors = list(getattr(result, "errors", None) or [])
    details: list[str] = []
    missing_model = False
    for error in errors:
        message = str(
            getattr(error, "error_message", None)
            or getattr(error, "message", None)
            or error
        )
        details.append(message)
        missing_model = missing_model or _looks_like_a_missing_model(message)
    summary = f"status={name}"
    if details:
        summary += "; errors: " + " | ".join(details)
    if not missing_model and _looks_like_a_missing_model(summary):
        missing_model = True
    return summary, missing_model


def _docling_version() -> str:
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError as exc:
        raise WorkerUnavailableError(
            "Docling is not installed in the worker's environment"
        ) from exc


def _missing_model_error(message: str, offline: bool) -> WorkerUnavailableError:
    if offline:
        remedy = (
            "the model is not cached and runtime.offline forbids fetching it. "
            f"{_PREFETCH_REMEDY}"
        )
    else:
        remedy = (
            "the model could not be fetched lazily; check the network or "
            f"{_PREFETCH_REMEDY}"
        )
    return WorkerUnavailableError(f"Docling needs a model. {remedy}. {message}")


def _build_converter():  # pragma: no cover - requires Docling installed
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline = PdfPipelineOptions()
    # OCR, VLM, and every enrichment are off. A scanned body yields no text
    # rather than guessed text, matching this app's text-layer-only contract.
    pipeline.do_ocr = False
    pipeline.do_table_structure = False
    pipeline.do_code_enrichment = False
    pipeline.do_formula_enrichment = False
    pipeline.do_picture_classification = False
    pipeline.do_picture_description = False
    pipeline.do_chart_extraction = False
    pipeline.generate_page_images = False
    pipeline.generate_picture_images = False
    pipeline.generate_table_images = False
    pipeline.enable_remote_services = False
    pipeline.accelerator_options.device = "cpu"
    pipeline.accelerator_options.num_threads = 2
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )


def _require_success(result: Any, offline: bool) -> None:
    problem = conversion_result_problem(result)
    if problem is None:
        return
    summary, missing_model = problem
    if missing_model:
        raise _missing_model_error(summary, offline)
    raise WorkerError(f"Docling conversion did not succeed: {summary}")


def _extract(
    job: dict[str, Any],
    converter: Any,
    version: str,
    boundary: dict[str, str],
    offline: bool,
) -> dict[str, Any]:
    source_path = Path(str(job["source_path"]))
    start = int(job["start"])
    end = int(job["end"])
    if start < 1 or end < start:
        raise WorkerError(f"Invalid page range: {start}..{end}")

    try:
        result = converter.convert(source_path, page_range=(start, end))
    except Exception as exc:
        message = str(exc)
        if _looks_like_a_missing_model(message):
            # A model this run needed is unavailable. That is infrastructure:
            # an offline run names provisioning, an online run names the network.
            raise _missing_model_error(message, offline) from exc
        raise WorkerError(f"Docling conversion failed: {message}") from exc

    _require_success(result, offline)
    document = getattr(result, "document", None)
    if document is None or not hasattr(document, "export_to_dict"):
        raise WorkerError("Docling returned no convertible document")
    payload = document.export_to_dict()
    pages, diagnostics = adapt_document(payload, page_first=start, page_last=end)
    metadata, labels, image_only_pages = _pdf_metadata_and_labels(
        source_path, start, end
    )
    options = {
        "ocr": False,
        "vlm": False,
        "table_structure": False,
        "enrichments": False,
        "accelerator": "cpu",
        "threads": 2,
        "page_range": [start, end],
    }
    return {
        "version": version,
        "options": options,
        "boundary": boundary,
        "pages": {str(page): units for page, units in pages.items()},
        "diagnostics": diagnostics,
        "image_only_pages": image_only_pages,
        "pdf_metadata": metadata,
        "page_labels": {str(page): label for page, label in labels.items()},
    }


def _prefetch(converter: Any, version: str, boundary: dict[str, str]) -> dict[str, Any]:
    """Provision the models extraction will need, through the same pipeline."""

    import tempfile

    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Docling model provisioning")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(document.tobytes())
        temporary = Path(handle.name)
    document.close()
    try:
        result = converter.convert(temporary, page_range=(1, 1))
    except Exception as exc:
        raise WorkerUnavailableError(
            f"Docling model provisioning failed: {exc}. {_PREFETCH_REMEDY}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    _require_success(result, offline=False)
    return {
        "version": version,
        "boundary": boundary,
        "mode": MODE_PREFETCH,
        "ready": True,
    }


class Session:
    """A verified worker's converter, reused for every job it is sent."""

    def __init__(self, mode: str, offline: bool) -> None:
        self.mode = mode
        self.offline = offline
        self.boundary = verify_resource_boundary()
        _verify_environment(mode, offline)
        apply_and_verify_nice()
        self.version = _docling_version()
        self.converter = _build_converter()

    @classmethod
    def for_job(cls, job: dict[str, Any]) -> Session:
        return cls(str(job.get("mode") or MODE_EXTRACT), bool(job.get("offline")))

    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        mode = str(job.get("mode") or MODE_EXTRACT)
        offline = bool(job.get("offline"))
        if (mode, offline) != (self.mode, self.offline):
            raise ResourceBoundaryError(
                f"This worker verified mode={self.mode} offline={self.offline}; "
                f"a mode={mode} offline={offline} job needs a new worker"
            )
        if mode == MODE_PREFETCH:
            return _prefetch(self.converter, self.version, self.boundary)
        return _extract(job, self.converter, self.version, self.boundary, offline)


def run(job: dict[str, Any]) -> dict[str, Any]:
    """Run one job in a new session."""

    return Session.for_job(job).run(job)


# Exit status 2 is an invalid job and 4 an unexpected failure.
_FAILURE_CODES = (
    (ResourceBoundaryError, 5),
    (WorkerUnavailableError, 6),
    (WorkerError, 3),
)


def _failure(exc: Exception) -> tuple[int, str]:
    for kind, code in _FAILURE_CODES:
        if isinstance(exc, kind):
            return code, str(exc)
    return 4, f"Docling worker failed: {type(exc).__name__}: {exc}"


def _reply(replies: TextIO, reply: dict[str, Any]) -> None:
    replies.write(json.dumps(reply, ensure_ascii=False) + "\n")
    replies.flush()


def _refuse(replies: TextIO, message: str) -> int:
    """Refuse an invalid job with exit status 2, which no parent reads as a source
    failure."""

    print(message, file=sys.stderr)
    _reply(replies, {"ok": False, "code": 2, "error": message})
    return 2


def serve(jobs: TextIO, replies: TextIO) -> int:
    """Answer each job line with one reply line, and return the exit status.

    Returns 0 when stdin closes. A failed job returns its code after its reply.
    """

    session: Session | None = None
    for line in iter(jobs.readline, ""):
        if not line.strip():
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            return _refuse(replies, f"Invalid job: {exc}")
        if not isinstance(job, dict):
            return _refuse(replies, "Invalid job: a job is a JSON object")
        if job.get("protocol") != PROTOCOL:
            return _refuse(replies, _PROTOCOL_REMEDY)
        try:
            if session is None:
                session = Session.for_job(job)
            result = session.run(job)
        except Exception as exc:  # noqa: BLE001 - the boundary must fail closed.
            code, message = _failure(exc)
            print(message, file=sys.stderr)
            _reply(replies, {"ok": False, "code": code, "error": message})
            return code
        _reply(replies, {"ok": True, "result": result})
    return 0


def main() -> int:
    # Replies keep the real stdout. A library that prints reaches stderr instead.
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        return serve(sys.stdin, replies)
    finally:
        replies.close()


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
