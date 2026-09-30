"""`research-rag doctor`: say what is wrong with this installation.

Every repair is a flag. With none of them the command reads: it runs the same
dependency checks `research-rag status` reports, prints one line per check, and
exits nonzero when something is blocked. It writes nothing — not even a client
entry, which it prints and validates with `--mcp-entry` and `--check-entry`.

`--prefetch-models` and `--repair-runtime` are the two operations that reach the
network, and neither is implied by the other.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
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


def _argument_value(arguments: list[str], name: str) -> str | None:
    """Return one argument's value, whether it was written `--name value` or `--name=value`."""

    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _project_root_argument(arguments: list[str]) -> str | None:
    return _argument_value(arguments, "--project-root")


# The MCP client entry the doctor prints and validates. The app is reached two
# ways, so there are two: a URL entry for a client that can open a socket, which
# points at the app's own port, and a stdio entry for one that cannot, which runs
# the bridge beside the running interpreter.
SERVER_COMMAND = "research-rag"
# A client timeout below this is shorter than an ingestion, and one above this
# is a number nobody waits out.
MINIMUM_ENTRY_TIMEOUT_MS = 60_000
MAXIMUM_ENTRY_TIMEOUT_MS = 86_400_000
_ENTRY_TIMEOUT_MS = 3_600_000


def _project_server_command() -> Path:
    """The console script this environment installed, which an entry should run."""

    return Path(sys.executable).parent / SERVER_COMMAND


def mcp_url_block(config: ResearchConfig) -> str:
    """Return the URL entry, which points at the app rather than at a proxy.

    The port is the one the app claimed and the launcher chose at start, so this
    is the entry to print for a project whose app is already up. When it is not,
    the default is named and the reader is told to start the app first, because
    the port is chosen at start and is not knowable before that.
    """

    from .app import recorded_port
    from .launcher import DEFAULT_UI_PORT

    port = recorded_port(config)
    return (
        json.dumps(
            {
                "mcp": {
                    SERVER_COMMAND: {
                        "type": "local",
                        "url": f"http://127.0.0.1:{port or DEFAULT_UI_PORT}/mcp",
                        "enabled": True,
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )


def mcp_entry_block(config: ResearchConfig) -> str:
    """Return the stdio client entry for the resolved configuration.

    One of two. A client that can open a socket is better served by the URL entry
    below, which reaches the app itself rather than a proxy to it; this one is for
    a client that speaks only stdio, and it names the bridge and the project.
    """

    arguments = [
        str(_project_server_command()),
        "--project-root",
        str(config.project_root),
    ]
    if config.runtime_root is not None:
        arguments.extend(["--runtime-root", str(config.runtime_root)])
    arguments.append("mcp")
    entry = {
        "mcp": {
            SERVER_COMMAND: {
                "type": "local",
                "command": arguments,
                "environment": {},
                "enabled": True,
                "timeout": _ENTRY_TIMEOUT_MS,
            }
        }
    }
    return json.dumps(entry, indent=2) + "\n"


def _strip_jsonc(text: str) -> str:
    """Remove `//` and block comments from a JSON-with-comments document."""

    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    while index < length:
        character = text[index]
        if in_string:
            out.append(character)
            if character == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            out.append(character)
            index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = length if end < 0 else end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        out.append(character)
        index += 1
    return "".join(out)


def _entries(document: Any, path: Path) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise DoctorError(f"The entry file is not a JSON object: {path}")
    for key in ("mcpServers", "mcp"):
        section = document.get(key)
        if isinstance(section, dict):
            return section
    raise DoctorError(
        f"The entry file has neither an mcpServers nor an mcp section: {path}"
    )


def _arguments_of(entry: Any) -> list[str]:
    """Return one entry's command and arguments as a single argv.

    A client that splits them writes the executable as `command` and the
    arguments as `args`; a client that keeps one list writes both in `command`.
    Both describe the same process, so both are read the same way.
    """

    if not isinstance(entry, dict):
        return []
    command = entry.get("command")
    if isinstance(command, str):
        argv = [command]
    elif isinstance(command, list):
        argv = [str(item) for item in command]
    else:
        argv = []
    args = entry.get("args")
    if isinstance(args, list):
        argv.extend(str(item) for item in args)
    return argv


def _runs_this_app(entry: Any) -> bool:
    """Whether one entry runs this app's agent surface over stdio.

    The key is the user's own name for the entry — a project may keep several
    entries and label them any way it likes — so the entry is recognized by what
    it runs, and the `mcp` argument is required: `research-rag ui` and
    `research-rag search` run the same executable and are not agent entries.
    """

    arguments = _arguments_of(entry)
    if not arguments:
        return False
    if Path(arguments[0]).name != SERVER_COMMAND and "research_rag" not in arguments:
        return False
    return "mcp" in arguments


def _entry_url(entry: Any) -> str | None:
    """Return the URL an entry points at, if it is a URL entry."""

    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    return url.strip() if isinstance(url, str) and url.strip() else None


def _check_url_entry(name: str, url: str) -> list[Check]:
    """Check one URL entry, which reaches the app itself rather than a proxy."""

    from .surfaces.mcp import MCP_PATH

    findings: list[Check] = []
    if not url.startswith(("http://", "https://")):
        findings.append(
            Check(
                "entry.url",
                "blocked",
                f"{name} points at {url}, which is not an HTTP address.",
            )
        )
        return findings
    authority = url.split("//", 1)[1]
    host = authority.split("/", 1)[0].split(":", 1)[0]
    if host not in ("127.0.0.1", "localhost", "::1"):
        findings.append(
            Check(
                "entry.url",
                "blocked",
                f"{name} points at {host}, and this app serves loopback only.",
            )
        )
    route = "/" + authority.split("/", 1)[1] if "/" in authority else "/"
    if not route.rstrip("/").endswith(MCP_PATH):
        findings.append(
            Check(
                "entry.url",
                "warn",
                f"{name} points at {route}; the agent surface is at {MCP_PATH}.",
            )
        )
    return findings


def check_entry(config: ResearchConfig, path: str | Path) -> tuple[Check, ...]:
    """Report on one client entry without changing it."""

    entry_path = Path(path).expanduser()
    findings: list[Check] = []
    if not entry_path.is_file():
        raise DoctorError(f"No client entry file at {entry_path}")
    try:
        document = json.loads(_strip_jsonc(entry_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise DoctorError(f"The client entry file cannot be read: {entry_path}: {exc}")
    entries = _entries(document, entry_path)
    mine = [
        name
        for name, entry in entries.items()
        if _runs_this_app(entry) or _entry_url(entry) is not None
    ]
    if not mine:
        raise DoctorError(f"No {SERVER_COMMAND} entry in {entry_path}")
    for name in sorted(mine):
        entry = entries[name]
        url = _entry_url(entry)
        if url is not None:
            findings.extend(_check_url_entry(name, url))
            continue
        arguments = _arguments_of(entry)
        executable = Path(arguments[0]).expanduser() if arguments else Path("")
        if not executable.is_file():
            findings.append(
                Check(
                    "entry.executable",
                    "blocked",
                    f"{name} runs {executable or '(no command)'}, which is not a file.",
                )
            )
        elif executable != _project_server_command():
            findings.append(
                Check(
                    "entry.executable",
                    "warn",
                    f"{name} runs {executable}; this environment provides "
                    f"{_project_server_command()}.",
                )
            )
        root = _project_root_argument(arguments)
        if root is None:
            findings.append(
                Check(
                    "entry.project_root",
                    "blocked",
                    f"{name} names no --project-root, so it cannot serve this project.",
                )
            )
        elif Path(root).expanduser().resolve() != config.project_root:
            findings.append(
                Check(
                    "entry.project_root",
                    "warn",
                    f"{name} serves {root}; this project is {config.project_root}.",
                )
            )
        entry_runtime_root = _runtime_root_argument(arguments)
        configured = str(config.runtime_root) if config.runtime_root else None
        if entry_runtime_root and entry_runtime_root != configured:
            findings.append(
                Check(
                    "entry.runtime_root",
                    "warn",
                    f"{name} passes --runtime-root {entry_runtime_root}; this "
                    f"project resolves to {configured or 'its default location'}.",
                )
            )
        timeout = entry.get("timeout") if isinstance(entry, dict) else None
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            findings.append(
                Check(
                    "entry.timeout",
                    "warn",
                    f"{name} has no numeric timeout; the client default may be "
                    "shorter than one ingestion.",
                )
            )
        elif not MINIMUM_ENTRY_TIMEOUT_MS <= timeout <= MAXIMUM_ENTRY_TIMEOUT_MS:
            findings.append(
                Check(
                    "entry.timeout",
                    "warn",
                    f"{name} allows {int(timeout)} ms, outside "
                    f"{MINIMUM_ENTRY_TIMEOUT_MS}-{MAXIMUM_ENTRY_TIMEOUT_MS}.",
                )
            )
    matching = [
        name
        for name, entry in entries.items()
        if _runs_this_app(entry)
        and (root := _project_root_argument(_arguments_of(entry))) is not None
        and Path(root).expanduser().resolve() == config.project_root
    ]
    if len(matching) > 1:
        findings.append(
            Check(
                "entry.duplicate",
                "warn",
                f"{', '.join(sorted(matching))} all name {config.project_root}. One "
                "app serves a project, so a second entry either reaches the same "
                "app or starts a second one on another port.",
            )
        )
    if not findings:
        findings.append(
            Check(
                "entry",
                "ok",
                f"{entry_path} reaches this project's app with matching paths "
                "and a usable timeout.",
            )
        )
    return tuple(findings)


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
    entry: str | Path | None = None,
    prefetch: bool = False,
    repair: bool = False,
) -> DoctorResult:
    """Run the checks and the requested operation, and return what to print."""

    if prefetch and repair:
        raise DoctorError(
            "--prefetch-models and --repair-runtime are separate operations; run "
            "one at a time."
        )
    if entry is not None and (prefetch or repair):
        raise DoctorError(
            "--check-entry reports on an entry; it cannot run with an operation "
            "that changes the installation."
        )
    lines: list[str] = []
    if entry is not None:
        findings = check_entry(config, entry)
        lines.extend(_check_line(check) for check in findings)
        return DoctorResult(
            tuple(lines), 1 if any(check.blocked for check in findings) else 0
        )
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
