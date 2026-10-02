"""Configuration and validation for the vanilla UltraRAG MCP gateway."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .manifest import BASELINE_COMMIT, BASELINE_VERSION, SERVER_SPECS
from .runtime import (
    MARKER_FILENAME,
    RuntimeValidationError,
    validate_managed_runtime,
)


class ConfigurationError(ValueError):
    """Raised when the selected UltraRAG runtime is incompatible."""


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    ultrarag_root: Path
    workspace_root: Path
    python_executable: Path
    log_level: str = "warn"


def _read_package_version(ultrarag_root: Path) -> str:
    pyproject = ultrarag_root / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        return str(data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            f"Cannot read UltraRAG version from {pyproject}"
        ) from exc


def _run_git(ultrarag_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(ultrarag_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(
            f"Cannot validate the UltraRAG Git checkout at {ultrarag_root}"
        ) from exc


def _read_source_revision(ultrarag_root: Path) -> str:
    if (ultrarag_root / ".git").exists():
        commit = _run_git(ultrarag_root, "rev-parse", "HEAD").stdout.strip()
        tracked_status = _run_git(
            ultrarag_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ).stdout
        if tracked_status.strip():
            raise ConfigurationError(
                f"UltraRAG checkout has tracked modifications: {ultrarag_root}"
            )
        return commit

    if (ultrarag_root / MARKER_FILENAME).is_file():
        try:
            validate_managed_runtime(ultrarag_root)
        except RuntimeValidationError as exc:
            raise ConfigurationError(str(exc)) from exc
        return BASELINE_COMMIT

    raise ConfigurationError(
        f"UltraRAG root is neither a Git checkout nor a managed runtime: "
        f"{ultrarag_root}"
    )


def resolve_config(
    ultrarag_root: str | Path,
    workspace_root: str | Path,
    python_executable: str | Path | None = None,
    log_level: str = "warn",
) -> GatewayConfig:
    root = Path(ultrarag_root).expanduser().resolve()
    workspace = Path(workspace_root).expanduser().resolve()
    # Do not resolve the interpreter symlink: a virtual environment's `python`
    # commonly points at the system binary, and resolving it would discard the
    # environment used to install the gateway and UltraRAG dependencies.
    python = Path(python_executable or sys.executable).expanduser().absolute()

    if not root.is_dir():
        raise ConfigurationError(f"UltraRAG root is not a directory: {root}")
    if not python.is_file():
        raise ConfigurationError(f"Python executable does not exist: {python}")

    version = _read_package_version(root)
    if version != BASELINE_VERSION:
        raise ConfigurationError(
            f"Unsupported UltraRAG version {version!r}; expected {BASELINE_VERSION!r}"
        )

    commit = _read_source_revision(root)
    if commit != BASELINE_COMMIT:
        raise ConfigurationError(
            "Unsupported UltraRAG revision "
            f"{commit}; expected {BASELINE_COMMIT}. "
            "Run the deliberate compatibility-update workflow before serving it."
        )

    missing = [
        str(spec.relative_entrypoint)
        for spec in SERVER_SPECS
        if not (root / spec.relative_entrypoint).is_file()
    ]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(f"Missing UltraRAG MCP entrypoints: {joined}")

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "ui-storage").mkdir(exist_ok=True)

    return GatewayConfig(
        ultrarag_root=root,
        workspace_root=workspace,
        python_executable=python,
        log_level=log_level,
    )
