"""The managed, versioned environment the optional Docling backend runs in.

Docling is never installed into this app's own environment. On first use of
``ingestion.pdf_backend=docling`` — a real PDF conversion, or the explicit
``doctor --prefetch-models`` — a dedicated virtual environment is built under the
runtime cache, pinned to the releases this app was tested against, and the shipped
worker runs in it. The app's environment and the managed one stay separate, so a
custom install never carries Docling and the optional extra is not required.

The environment is versioned by a spec fingerprint (the pins, the interpreter
ABI, and a bootstrap policy version). It is built in a staging directory on the
same filesystem, and a completion manifest is written only after the pinned
versions are read back from the staged environment's own ``dist-info``. Every
installer step runs under a resource-bounded scope whose controller re-verifies
the two-GiB memory cap, the zero swap cap, the 200% CPU quota, and the +10
niceness before it executes the trusted ``uv`` command. The staging directory is
renamed into place atomically, so an interrupted install never leaves a half-ready
environment and never touches one that was already ready.

Installation needs the network. An offline run never installs: a missing
environment fails closed and points at ``doctor --prefetch-models`` or dropping
``runtime.offline``. A ready environment is reused offline without a fetch.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from ..project.config import ResearchConfig, child_process_environment
from ..project.policy import ResearchError

# The pinned releases. `docling` and `rapidocr` are the exact tested pair;
# pymupdf is the worker's own PDF reader, allowed within a bounded range. The
# Docling extraction components move independently of the metapackage, so they
# are required to be present and their exact installed versions are recorded.
DOCLING_PINS: tuple[str, ...] = (
    "docling==2.135.0",
    "rapidocr==3.9.1",
    "pymupdf>=1.26,<2",
)
DOCLING_REQUIRED_PACKAGES = (
    "docling",
    "rapidocr",
    "pymupdf",
    "docling-core",
    "docling-parse",
    "docling-ibm-models",
)
DOCLING_VERSION_SPECS: dict[str, str] = {
    "docling": "==2.135.0",
    "rapidocr": "==3.9.1",
    "pymupdf": ">=1.26,<2",
}
DOCLING_COMPONENT_PACKAGES = DOCLING_REQUIRED_PACKAGES

# Bump when the pins, the interpreter ABI policy, or the worker bootstrap change.
BACKEND_SPEC_VERSION = 1
BOOTSTRAP_POLICY_VERSION = 1
TORCH_BACKEND = "cpu"

INSTALL_TIMEOUT_SECONDS = 600
INSTALL_LOCK_TIMEOUT_SECONDS = INSTALL_TIMEOUT_SECONDS + 60
LOCKS_WAIT_SECONDS = 60

MEMORY_MAX_BYTES = 2 * 1024**3
CPU_QUOTA_PERCENT = 200
CPU_MAX = ("200000", "100000")
INSTALL_NICE = 10
WORKER_NICE = 10

_READY_FILE = "ready.json"
_QUARANTINE_PREFIX = ".quarantine-"
_LOCK_FILE = ".install.lock"

# The in-process guards are separate: the install critical section may run
# installer steps, and each of those takes the run lock. Sharing one lock would
# deadlock the process that already holds it.
_INSTALL_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()

_INSTALL_REMEDY = (
    "run `research-rag doctor --prefetch-models` once with a network connection "
    "or set runtime.offline=false, then run ingestion again"
)


class DoclingEnvironmentError(ResearchError):
    """The managed environment is missing or could not be built.

    This is infrastructure: it aborts the build rather than omitting one source.
    """


@dataclass(frozen=True, slots=True)
class DoclingEnv:
    """A ready managed environment."""

    root: Path
    spec_fingerprint: str
    versions: dict[str, str | None]

    @property
    def python(self) -> Path:
        return self.root / "bin" / "python"

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "spec_fingerprint": self.spec_fingerprint,
            "python": str(self.python),
            "versions": dict(self.versions),
        }


def backend_root(config: ResearchConfig) -> Path:
    """The directory holding every managed backend environment for this install."""

    base = config.runtime_cache_root or config.model_cache_root.parent
    return base / "backend-envs"


def python_abi() -> str:
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


def spec_fingerprint() -> str:
    payload = json.dumps(
        {
            "spec_version": BACKEND_SPEC_VERSION,
            "bootstrap_policy_version": BOOTSTRAP_POLICY_VERSION,
            "torch_backend": TORCH_BACKEND,
            "python_abi": python_abi(),
            "pins": list(DOCLING_PINS),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def env_root(config: ResearchConfig) -> Path:
    return backend_root(config) / spec_fingerprint()


def worker_path() -> Path:
    """The shipped worker file, run directly by the managed interpreter."""

    return Path(__file__).resolve().with_name("docling_worker.py")


def worker_command(env: DoclingEnv) -> list[str]:
    """The isolated command that runs the shipped worker in the managed env.

    The managed interpreter runs ``-I`` and executes the worker *file*, so the
    app's own source or site-packages are never placed on ``sys.path``; Docling
    resolves from the managed environment alone.
    """

    return [str(env.python), "-I", str(worker_path())]


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


# --- version checks ------------------------------------------------------------


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value.split("+", 1)[0])
    return tuple(int(number) for number in numbers) if numbers else (0,)


def version_satisfies(version: str, spec: str) -> bool:
    """Whether ``version`` meets a comma-joined ``==``/``>=``/``<=``/``<``/``>`` spec."""

    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        operator = "=="
        remainder = clause
        for candidate in ("==", ">=", "<=", "<", ">"):
            if clause.startswith(candidate):
                operator, remainder = candidate, clause[len(candidate) :]
                break
        if operator == "==" and version != remainder:
            return False
        if operator == ">=" and not (
            _version_tuple(version) >= _version_tuple(remainder)
        ):
            return False
        if operator == "<=" and not (
            _version_tuple(version) <= _version_tuple(remainder)
        ):
            return False
        if operator == "<" and not (
            _version_tuple(version) < _version_tuple(remainder)
        ):
            return False
        if operator == ">" and not (
            _version_tuple(version) > _version_tuple(remainder)
        ):
            return False
    return True


# --- ready marker and its metadata cross-check ---------------------------------


def _read_ready(root: Path) -> dict[str, Any] | None:
    path = root / _READY_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _site_packages(root: Path) -> Path | None:
    lib = root / "lib"
    if not lib.is_dir():
        return None
    for candidate in sorted(lib.glob("python*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def _read_dist_info_versions(root: Path) -> dict[str, str]:
    """Read installed versions from the env's own ``dist-info``, no process."""

    site = _site_packages(root)
    if site is None:
        return {}
    versions: dict[str, str] = {}
    for info in site.glob("*.dist-info"):
        metadata = info / "METADATA"
        name: str | None = None
        version: str | None = None
        try:
            text = metadata.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if name is None and line.startswith("Name: "):
                name = _normalize_name(line[6:].strip())
            elif version is None and line.startswith("Version: "):
                version = line[9:].strip()
            if name is not None and version is not None:
                break
        if name is not None and version is not None:
            versions[name] = version
    return versions


def _versions_valid(versions: dict[str, Any]) -> bool:
    """Whether recorded versions satisfy the pins and carry every component."""

    for package in DOCLING_REQUIRED_PACKAGES:
        value = versions.get(package)
        if not isinstance(value, str) or not value:
            return False
    for package, spec in DOCLING_VERSION_SPECS.items():
        if not version_satisfies(str(versions[package]), spec):
            return False
    return True


def resolve_docling_env(config: ResearchConfig) -> DoclingEnv | None:
    """The ready managed environment, or None. Read-only: never installs.

    The marker's recorded versions are cross-checked against the environment's
    own ``dist-info``, so a dependency mutated after installation invalidates
    readiness rather than leaving a stale fingerprint.
    """

    root = env_root(config)
    ready = _read_ready(root)
    if ready is None or not ready.get("ready"):
        return None
    if ready.get("spec_fingerprint") != spec_fingerprint():
        return None
    if ready.get("python_abi") != python_abi():
        return None
    python = root / "bin" / "python"
    if not python.is_file():
        return None
    versions = ready.get("versions")
    if not isinstance(versions, dict) or not _versions_valid(versions):
        return None
    actual = _read_dist_info_versions(root)
    for package in DOCLING_REQUIRED_PACKAGES:
        if actual.get(package) != versions.get(package):
            return None
    return DoclingEnv(
        root=root,
        spec_fingerprint=spec_fingerprint(),
        versions={str(key): str(value) for key, value in versions.items()},
    )


def docling_environment_ready(config: ResearchConfig) -> bool:
    return resolve_docling_env(config) is not None


def ensure_docling_env(
    config: ResearchConfig,
    *,
    allow_install: bool | None = None,
    timeout: float = INSTALL_TIMEOUT_SECONDS,
) -> DoclingEnv:
    """Return the ready environment, building it on first use when allowed.

    ``allow_install`` defaults to ``not config.offline``: an offline run never
    fetches and never installs.
    """

    ready = resolve_docling_env(config)
    if ready is not None:
        return ready
    if allow_install is None:
        allow_install = not config.offline
    if not allow_install:
        raise DoclingEnvironmentError(
            "ingestion.pdf_backend=docling needs its managed environment, and "
            f"runtime.offline forbids building it. To provision it, {_INSTALL_REMEDY}."
        )
    with _install_lock(config):
        ready = resolve_docling_env(config)
        if ready is not None:
            return ready
        return install_docling_env(config, timeout=timeout)


@contextlib.contextmanager
def _install_lock(config: ResearchConfig) -> Iterator[None]:
    """One install at a time in this process and across processes, both bounded."""

    root = backend_root(config)
    root.mkdir(parents=True, exist_ok=True)
    if not _INSTALL_LOCK.acquire(timeout=LOCKS_WAIT_SECONDS):
        raise DoclingEnvironmentError(
            "Another operation in this process is building the Docling "
            "environment. Nothing was changed; run it again once that finishes."
        )
    try:
        handle = FileLock(root / _LOCK_FILE, timeout=INSTALL_LOCK_TIMEOUT_SECONDS)
        try:
            handle.acquire()
        except FileLockTimeout as exc:
            raise DoclingEnvironmentError(
                "Another research command is building the Docling environment. "
                "Nothing was changed; run it again once that finishes."
            ) from exc
        try:
            yield
        finally:
            handle.release()
    finally:
        _INSTALL_LOCK.release()


def _install_environment() -> dict[str, str]:
    """The installer's environment: sanitized, no download of a second Python.

    The caller's site-packages and interpreter symlinks are never inherited, and
    no user-supplied path or wheel is honoured.
    """

    environment = child_process_environment()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    ):
        environment.pop(name, None)
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    environment["UV_LINK_MODE"] = "copy"
    environment["UV_NO_PROGRESS"] = "1"
    return environment


def _require_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise DoclingEnvironmentError(
            "installing the managed Docling environment needs `uv` on PATH. "
            "Install uv, or set ingestion.pdf_backend=custom. No source was "
            "changed and the selected generation is unchanged."
        )
    return executable


def _require_systemd_run() -> str:
    executable = shutil.which("systemd-run")
    if executable is None or shutil.which("systemctl") is None:
        raise DoclingEnvironmentError(
            "building the managed Docling environment needs a systemd user "
            "session (`systemd-run` and `systemctl`) to bound the installer. "
            "Install one, or set ingestion.pdf_backend=custom. No source was "
            "changed and the selected generation is unchanged."
        )
    return executable


def _terminate_scope(unit: str, process: subprocess.Popen[Any]) -> None:
    systemctl = shutil.which("systemctl")
    if systemctl is not None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [
                    systemctl,
                    "--user",
                    "kill",
                    "--signal=SIGKILL",
                    "--kill-whom=all",
                    unit,
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), 9)
    with contextlib.suppress(subprocess.TimeoutExpired):  # pragma: no cover
        process.wait(timeout=15)


# The controller runs inside the resource scope, before any trusted command. It
# is stdlib-only, verifies the exact boundary it was promised, raises its own
# niceness to at least 10, and only then execs the command. The parent never uses a
# preexec hook.
CONTROLLER_SOURCE = (
    "import os, sys\n"
    "from pathlib import Path\n"
    f"MEMORY_MAX = {MEMORY_MAX_BYTES}\n"
    f"CPU_MAX = {list(CPU_MAX)!r}\n"
    f"NICE = {INSTALL_NICE}\n"
    "def own():\n"
    "    for line in Path('/proc/self/cgroup').read_text().splitlines():\n"
    "        if line.startswith('0::'):\n"
    "            return Path('/sys/fs/cgroup') / line[3:].strip().lstrip('/')\n"
    "    raise SystemExit('no cgroup v2 entry')\n"
    "root = own()\n"
    "try:\n"
    "    mem = (root / 'memory.max').read_text().strip()\n"
    "    swap = (root / 'memory.swap.max').read_text().strip()\n"
    "    cpu = (root / 'cpu.max').read_text().split()\n"
    "except OSError as exc:\n"
    "    sys.stderr.write(f'installer boundary unreadable: {exc}\\n'); raise SystemExit(5)\n"
    "if mem != str(MEMORY_MAX) or swap != '0' or cpu != list(CPU_MAX):\n"
    "    sys.stderr.write(f'installer boundary mismatch: {mem} {swap} {cpu}\\n'); raise SystemExit(5)\n"
    "current = os.nice(0)\n"
    "if current < NICE:\n"
    "    os.nice(NICE - current)\n"
    "if os.nice(0) < NICE:\n"
    "    sys.stderr.write('installer niceness was not applied\\n'); raise SystemExit(5)\n"
    "os.execv(sys.argv[1], sys.argv[1:])\n"
)


def _run_bounded(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    description: str,
) -> bytes:
    """Run one installer step under the verified boundary and +10 niceness."""

    systemd_run = _require_systemd_run()
    unit = f"research-rag-docling-install-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    wrapped = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        f"--property=MemoryMax={MEMORY_MAX_BYTES}",
        "--property=MemorySwapMax=0",
        f"--property=CPUQuota={CPU_QUOTA_PERCENT}%",
        f"--property=RuntimeMaxSec={timeout:g}",
        "--",
        sys.executable,
        "-I",
        "-c",
        CONTROLLER_SOURCE,
        *command,
    ]
    if not _RUN_LOCK.acquire(timeout=max(1.0, timeout)):
        raise DoclingEnvironmentError(
            "Another Docling installer step is already running in this process."
        )
    try:
        process = subprocess.Popen(
            wrapped,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise DoclingEnvironmentError(
                f"{description} exceeded the {timeout:g}s limit; the installer "
                "scope was killed. Run `research-rag doctor --prefetch-models` "
                "again, or set ingestion.pdf_backend=custom."
            ) from exc
        finally:
            _terminate_scope(unit, process)
    finally:
        _RUN_LOCK.release()
    if process.returncode != 0:
        text = (stderr or b"").decode("utf-8", "replace").strip()
        raise DoclingEnvironmentError(
            f"{description} failed (exit {process.returncode}): "
            f"{text or 'no diagnostics'}"
        )
    return stdout or b""


_VERSION_SCRIPT = (
    "import importlib.metadata as m, json\n"
    f"names = {list(DOCLING_COMPONENT_PACKAGES)!r}\n"
    "out = {}\n"
    "for name in names:\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except Exception:\n"
    "        out[name] = None\n"
    "print(json.dumps(out))\n"
)


def _read_env_versions(python: Path, *, timeout: float) -> dict[str, str | None]:
    stdout = _run_bounded(
        [str(python), "-I", "-c", _VERSION_SCRIPT],
        timeout=timeout,
        env=_install_environment(),
        description="Docling environment validation",
    )
    try:
        versions = json.loads(stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DoclingEnvironmentError(
            f"the staged Docling environment reported unreadable metadata: {exc}"
        ) from exc
    if not isinstance(versions, dict):
        raise DoclingEnvironmentError("the staged environment reported no versions")
    return versions


def _validate_versions(versions: dict[str, Any]) -> dict[str, str | None]:
    """Require the exact pins and every extraction component's version."""

    missing = [
        name
        for name in DOCLING_REQUIRED_PACKAGES
        if not isinstance(versions.get(name), str) or not versions.get(name)
    ]
    if missing:
        raise DoclingEnvironmentError(
            "the staged Docling environment is missing required packages: "
            + ", ".join(missing)
        )
    for package, spec in DOCLING_VERSION_SPECS.items():
        installed = str(versions[package])
        if not version_satisfies(installed, spec):
            raise DoclingEnvironmentError(
                f"the staged Docling environment has {package} {installed}, "
                f"which does not satisfy {spec}"
            )
    return {
        str(key): (str(value) if value else None) for key, value in versions.items()
    }


def _fsync_file(path: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_ready(root: Path, versions: dict[str, str | None]) -> None:
    payload = {
        "ready": True,
        "spec_fingerprint": spec_fingerprint(),
        "python_abi": python_abi(),
        "versions": versions,
        "created_at": datetime.now(UTC).isoformat(),
    }
    temporary = root / f"{_READY_FILE}.{uuid.uuid4().hex[:8]}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / _READY_FILE)
    _fsync_directory(root)


def _quarantine(root: Path) -> None:
    if not root.exists():
        return
    target = root.with_name(f"{_QUARANTINE_PREFIX}{uuid.uuid4().hex[:8]}")
    with contextlib.suppress(OSError):
        os.replace(root, target)


def install_docling_env(
    config: ResearchConfig,
    *,
    timeout: float = INSTALL_TIMEOUT_SECONDS,
) -> DoclingEnv:
    """Build the pinned environment in staging and publish it atomically.

    The caller holds the install lock. A ready environment is re-checked first
    and returned untouched; only an invalid target is quarantined. Each step
    shares one wall-clock deadline, so a first-use install is bounded overall.
    """

    # Under the install lock a valid environment cannot have appeared between
    # the caller's check and here, but re-check so a direct caller is safe.
    existing = resolve_docling_env(config)
    if existing is not None:
        return existing

    uv = _require_uv()
    root = backend_root(config)
    root.mkdir(parents=True, exist_ok=True)
    final = env_root(config)
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise DoclingEnvironmentError(
                "the Docling environment install exceeded its wall-clock "
                "deadline; no ready environment was published."
            )
        return left

    staging = root / f".staging-{spec_fingerprint()}-{uuid.uuid4().hex[:8]}"
    environment = _install_environment()
    try:
        _run_bounded(
            [uv, "venv", "--python", sys.executable, str(staging)],
            timeout=remaining(),
            env=environment,
            description="Docling environment creation",
        )
        python = staging / "bin" / "python"
        _run_bounded(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--torch-backend",
                TORCH_BACKEND,
                *DOCLING_PINS,
            ],
            timeout=remaining(),
            env=environment,
            description="Docling dependency install",
        )
        versions = _validate_versions(_read_env_versions(python, timeout=remaining()))
        if final.exists():
            # Only an invalid target reaches here; a valid one was returned above.
            _quarantine(final)
        _write_ready(staging, versions)
        os.replace(staging, final)
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DoclingEnv(
        root=final, spec_fingerprint=spec_fingerprint(), versions=versions
    )
