"""`research-rag doctor`: say what is wrong with this installation.

Every repair is a flag. With none of them the command reads: it runs the same
dependency checks `research-rag status` reports, prints one line per check, and
exits nonzero when something is blocked. It writes nothing.

`--prefetch-models` and `--repair-runtime` are the two operations that reach the
network, and neither is implied by the other.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ResearchConfig, project_command
from .embeddings import resolve_embedding_model
from .health import Check, HealthReport, health_report
from .rerankers import resolve_reranker_model

_HASH_IN_MESSAGE = re.compile(r"got\s+([0-9a-f]{8,64})")


class DoctorError(ValueError):
    """Raised when the doctor cannot run the check it was asked to run."""


@dataclass(frozen=True, slots=True)
class DoctorResult:
    """What the doctor printed, and the exit code that follows from it."""

    lines: tuple[str, ...]
    exit_code: int

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _check_line(check: Check) -> str:
    line = f"{check.state:<8} {check.name:<20} {check.reason}"
    if check.remedy_command:
        line += f"  ->  {check.remedy_command}"
    return line


def _observed_hash(root: Path, message: str, runtime: Any) -> str:
    tree_hash = getattr(runtime, "_tree_hash", None)
    if tree_hash is not None:
        try:
            return str(tree_hash(root))[:12]
        except OSError:
            pass
    found = _HASH_IN_MESSAGE.search(message)
    return found.group(1)[:12] if found else "unknown"


def repair_runtime(config: ResearchConfig) -> tuple[str, ...]:
    """Move a mismatched snapshot aside, install the pinned one, and re-validate.

    The evidence is kept rather than deleted: a tree that failed validation is
    what explains the failure, and an operator may want to look inside it.
    """

    from vanilla_ultra_rag_mcp import runtime as vanilla

    root = vanilla.managed_runtime_path(config.runtime_cache_root)
    if not root.is_dir():
        vanilla.install_managed_runtime(config.runtime_cache_root)
        return (
            f"Installed the pinned UltraRAG runtime at {root}.",
            f"It matches the pinned snapshot: {vanilla.validate_managed_runtime(root)}",
        )
    try:
        vanilla.validate_managed_runtime(root)
    except Exception as exc:  # noqa: BLE001 - a mismatch is a repair, whatever it raises.
        observed = _observed_hash(root, str(exc), vanilla)
        commit = root.name.rsplit("-", 1)[-1]
        cache_root = root.parent.parent
        quarantine = cache_root / "quarantine" / f"{commit}-{observed}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            suffix = 1
            while (quarantine.parent / f"{commit}-{observed}-{suffix}").exists():
                suffix += 1
            quarantine = quarantine.parent / f"{commit}-{observed}-{suffix}"
        os.rename(root, quarantine)
        vanilla.install_managed_runtime(config.runtime_cache_root)
        return (
            f"The mismatched runtime is preserved at {quarantine}.",
            (
                "Installed and validated the pinned runtime: "
                f"{vanilla.validate_managed_runtime(root)}"
            ),
        )
    return (f"The managed runtime at {root} already matches the pinned snapshot.",)


def prefetch_models(config: ResearchConfig) -> tuple[str, ...]:
    """Load the pinned models into the configured cache, and report the change."""

    from .dense import _load_cross_encoder, _load_embedder
    from .storage import directory_statistics

    _count, before = directory_statistics(config.model_cache_root)
    embedding = resolve_embedding_model(config.settings.embedding_model)
    reranker, revision = resolve_reranker_model(config.reranker_model)
    _load_embedder(config.model_cache_root, offline=False, model=embedding.name)
    _load_cross_encoder(config.model_cache_root, offline=False, model=reranker)
    _count, after = directory_statistics(config.model_cache_root)
    return (
        f"Cached {embedding.name}@{embedding.revision}.",
        f"Cached {reranker}@{revision}.",
        f"{config.model_cache_root} now holds {after - before} more bytes.",
    )


def _runtime_root_argument(arguments: list[str]) -> str | None:
    """Return one command line's ``--runtime-root`` value, if it carries one."""

    for index, argument in enumerate(arguments):
        if argument == "--runtime-root" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--runtime-root="):
            return argument.split("=", 1)[1]
    return None


def _server_lines(config: ResearchConfig, running: list[tuple[int, str]]) -> list[str]:
    """Report the processes serving this project, and the command that stops them.

    The doctor is not one of them, so it drops itself from the list rather than
    reporting its own process as something to stop.
    """

    servers = [
        (pid, command) for pid, command in running if "doctor" not in command.split()
    ]
    if not servers:
        return [f"Nothing of this app is running for {config.project_root}."]
    lines = [
        (
            f"{len(servers)} process(es) are running for {config.project_root}; "
            "whatever started them decides whether they come back."
        )
    ]
    lines.extend(f"  {pid}  {command}" for pid, command in servers)
    elsewhere = sorted(
        {
            root
            for _pid, command in servers
            if (root := _runtime_root_argument(shlex.split(command)))
            and Path(root).expanduser().resolve() != config.runtime_root
        }
    )
    for root in elsewhere:
        lines.append(
            f"  ->  A server for this project uses --runtime-root {root}; give "
            "the doctor the same flag to report on the state it actually uses."
        )
    lines.append(f"  ->  {project_command(config.project_root, 'stop', '--servers')}")
    return lines


def run_doctor(
    config: ResearchConfig,
    status: dict[str, Any],
    *,
    running: list[tuple[int, str]],
    prefetch: bool = False,
    repair: bool = False,
) -> DoctorResult:
    """Run the checks and the requested operation, and return what to print."""

    if prefetch and repair:
        raise DoctorError(
            "--prefetch-models and --repair-runtime are separate operations; run "
            "one at a time."
        )
    lines: list[str] = []
    if repair:
        lines.extend(repair_runtime(config))
    if prefetch:
        lines.extend(prefetch_models(config))
    report: HealthReport = health_report(config, status)
    lines.extend(_check_line(check) for check in report.checks)
    if report.not_checked:
        lines.append(
            "Not checked: "
            + ", ".join(report.not_checked)
            + ". An unchecked dependency is not a healthy one."
        )
    lines.extend(_server_lines(config, running))
    return DoctorResult(tuple(lines), 1 if report.has_blocker else 0)
