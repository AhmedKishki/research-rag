"""`research-rag doctor`: say what is wrong with this installation.

With no repair flag the command reads: it runs the dependency checks
`research-rag status` reports, prints one line per check, and exits nonzero when
something is blocked. It writes nothing, not even a client entry, which it prints and
validates with `--mcp-entry` and `--check-entry`. `--prefetch-models` and
`--repair-runtime` are the only operations that reach the network, and neither
implies the other.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..project.config import (
    CLI_COMMAND,
    DEFAULT_UI_PORT,
    ResearchConfig,
    project_command,
)
from ..retrieval.embeddings import resolve_embedding_model
from ..retrieval.rerankers import resolve_reranker_model
from .health import WARN, Check, HealthReport, health_report

_HASH_IN_MESSAGE = re.compile(r"got\s+([0-9a-f]{8,64})")


class DoctorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DoctorResult:
    """The printed lines, and the exit code that follows from them."""

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

    The evidence is kept rather than deleted: a tree that failed validation is what
    explains the failure.
    """

    from ..gateway import runtime as vanilla

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
    from ..retrieval.dense import _load_cross_encoder, _load_embedder
    from ..storage.records import directory_statistics

    _count, before = directory_statistics(config.model_cache_root)
    embedding = resolve_embedding_model(config.settings.embedding_model)
    reranker, revision = resolve_reranker_model(config.reranker_model)
    _load_embedder(config.model_cache_root, offline=False, model=embedding.name)
    _load_cross_encoder(config.model_cache_root, offline=False, model=reranker)
    _count, after = directory_statistics(config.model_cache_root)
    return (
        f"Cached {embedding.name} from {embedding.repository}@{embedding.revision}.",
        f"Cached {reranker}@{revision}.",
        f"{config.model_cache_root} now holds {after - before} more bytes.",
    )


def _argument_value(arguments: list[str], name: str) -> str | None:
    """One argument's value, whether written `--name value` or `--name=value`."""

    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _project_root_argument(arguments: list[str]) -> str | None:
    return _argument_value(arguments, "--project-root")


def _project_name_argument(arguments: list[str]) -> str | None:
    return _argument_value(arguments, "--project-name")


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


def _recommended_server_command() -> str:
    """The command a client entry should carry on this machine.

    `install` puts the command on `PATH`, and an entry naming it survives this
    checkout being replaced, so that is what a generated entry uses. A machine
    where the command was never installed gets the absolute path instead, because a
    name no client can resolve is worse than a long one.
    """

    return (
        SERVER_COMMAND
        if shutil.which(SERVER_COMMAND)
        else str(Path(sys.executable).parent / SERVER_COMMAND)
    )


def _project_server_commands() -> set[Path]:
    """Every path that runs this installation's command on this machine.

    `install` puts the command on `PATH` pointing at this checkout's interpreter,
    so a client entry naming either one runs this app. An entry naming any other
    build runs something else, which is what the check is for.
    """

    commands = {Path(sys.executable).parent / SERVER_COMMAND}
    found = shutil.which(SERVER_COMMAND)
    if found is not None:
        commands.add(Path(found))
    return commands


def _resolved_command(value: str) -> Path | None:
    """Where a client entry's command would actually run, or None if it cannot.

    A client execs the first argument and lets the operating system search `PATH`,
    so an entry that names `research-rag` runs the command `install` put there.
    Checking the name as a path instead would block every entry that relies on
    `PATH`, which is the form `install` writes.
    """

    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    found = shutil.which(value)
    return Path(found) if found is not None else None


def mcp_url_block(config: ResearchConfig) -> str:
    """The URL entry, which points at the app rather than at a proxy.

    The port is the one the app claimed at start. When no app is up the default is named
    and the reader is told to start the app first: the port is not knowable before then.
    """
    from .app import recorded_port

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
    """The stdio client entry for the resolved configuration.

    One of two: a client that can open a socket is better served by the URL entry.
    The stdio entry names the project and not its directory, so the same entry
    serves the project on every machine where it was initialised, and the app
    there resolves the name through its own project record.

    The subcommand comes before its own options, because that is the only order
    the parser accepts: `--project-name` belongs to `mcp`, so an entry that puts
    it first has the parser reading the project name as the command, and the
    server closes the connection instead of answering. `check_entry` refuses that
    order for the same reason.
    """

    # The entry carries the project's name and nothing else. A relocated runtime
    # root is where one machine keeps its derived state, and a second machine
    # resolves the same name to a different place entirely.
    arguments = [_recommended_server_command(), "mcp"]
    arguments.extend(["--project-name", config.project_name])
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
    """One entry's command and arguments as a single argv.

    A client may split them into `command` and `args` or keep one list in `command`;
    both describe the same process.
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

    A project may keep several entries and label them any way it likes, so the entry is
    recognized by what it runs. The `mcp` argument is required: `research-rag ui` and
    `research-rag search` run the same executable.
    """

    arguments = _arguments_of(entry)
    if not arguments:
        return False
    if Path(arguments[0]).name != SERVER_COMMAND and "research_rag" not in arguments:
        return False
    return "mcp" in arguments


def _entry_url(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    return url.strip() if isinstance(url, str) and url.strip() else None


def _loopback_url(url: str) -> bool:
    """Whether a URL entry names this machine.

    A client configuration holds every server its owner uses, so an entry
    pointing somewhere else is that server's and this one has nothing to say
    about it. Naming this app over a network address is not a case to report
    either: the app serves loopback only, so such an entry cannot reach it.
    """

    if not url.startswith(("http://", "https://")):
        return False
    authority = url.split("//", 1)[1]
    host = authority.split("/", 1)[0].split(":", 1)[0]
    return host in ("127.0.0.1", "localhost", "::1")


def _check_url_entry(name: str, url: str) -> list[Check]:
    """One loopback URL entry, which reaches the app rather than a proxy."""

    from ..surfaces.mcp import MCP_PATH

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


def _options_before_the_command(arguments: list[str]) -> list[str]:
    """Which of this app's own options an entry puts ahead of the `mcp` command.

    The parser only accepts them after it, so an option in front is read as the
    command name. That closes the connection with an error naming the project's
    own words, which is the hardest kind of failure to see from the entry.
    """

    command = arguments.index("mcp") if "mcp" in arguments else len(arguments)
    return [
        argument
        for argument in arguments[:command]
        if argument in ("--project-name", "--runtime-root")
    ]


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
        if _runs_this_app(entry)
        or (url := _entry_url(entry)) is not None
        and _loopback_url(url)
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
        named = arguments[0] if arguments else ""
        executable = _resolved_command(named)
        misplaced = _options_before_the_command(arguments)
        if misplaced:
            findings.append(
                Check(
                    "entry.argument_order",
                    "blocked",
                    f"{name} puts {' and '.join(misplaced)} before the `mcp` "
                    "command, so the parser reads the project name as the command "
                    "and the server closes the connection. Put `mcp` first.",
                )
            )
        if executable is None:
            findings.append(
                Check(
                    "entry.executable",
                    "blocked",
                    f"{name} runs {named or '(no command)'}, which is not on this "
                    f"machine's PATH.",
                )
            )
        elif executable not in _project_server_commands():
            findings.append(
                Check(
                    "entry.executable",
                    "warn",
                    f"{name} runs {executable}; this environment provides "
                    f"{Path(sys.executable).parent / SERVER_COMMAND}.",
                )
            )
        root = _project_root_argument(arguments)
        project_name = _project_name_argument(arguments)
        if root is not None:
            findings.append(
                Check(
                    "entry.project_name",
                    "blocked",
                    f"{name} names the directory {root}, so it stops working the "
                    "moment it is used on another machine. Name the project "
                    "instead: --project-name "
                    f"{config.project_name}.",
                )
            )
        elif project_name is None:
            findings.append(
                Check(
                    "entry.project_name",
                    "blocked",
                    f"{name} names no project, so it cannot serve {config.project_name}.",
                )
            )
        elif project_name.casefold() != config.project_name.casefold():
            findings.append(
                Check(
                    "entry.project_name",
                    "warn",
                    f"{name} serves the project {project_name}; this one is "
                    f"{config.project_name}.",
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
        and (named := _project_name_argument(_arguments_of(entry))) is not None
        and named.casefold() == config.project_name.casefold()
    ]
    if len(matching) > 1:
        findings.append(
            Check(
                "entry.duplicate",
                "warn",
                f"{', '.join(sorted(matching))} all name {config.project_name}. One "
                "app serves a project, so a second entry either reaches the same "
                "app or starts a second one on another port.",
            )
        )
    if not findings:
        findings.append(
            Check(
                "entry",
                "ok",
                f"{entry_path} reaches the app for {config.project_name} with a "
                "usable timeout.",
            )
        )
    return tuple(findings)


def _runtime_root_argument(arguments: list[str]) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == "--runtime-root" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--runtime-root="):
            return argument.split("=", 1)[1]
    return None


def _server_lines(config: ResearchConfig, running: list[tuple[int, str]]) -> list[str]:
    """The processes serving this project, and the command that stops them.

    The doctor is not one of them, so it drops itself from the list.
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


def desktop_entry_checks() -> list[Check]:
    """Every menu entry this app wrote that no longer opens anything.

    A desktop entry whose project root has been deleted still appears in the
    menu and still starts, and it opens nothing at all, which is a failure with
    no output anywhere else. Only entries carrying this app's marker are read: a
    file the user wrote is theirs.
    """

    from .installation import desktop_entries

    findings: list[Check] = []
    for entry in desktop_entries():
        add = f"`{CLI_COMMAND} install --desktop` writes it"
        if not entry.attached:
            # An entry that opens no terminal serves the app in the
            # background with nothing to close.
            findings.append(
                Check(
                    "desktop_entry",
                    WARN,
                    f"The menu entry {entry.path.name} does not open a terminal "
                    "window, so the app it starts runs with nothing to close.",
                    add,
                )
            )
            continue
        if entry.names_a_project:
            findings.append(
                Check(
                    "desktop_entry",
                    WARN,
                    f"The menu entry {entry.path.name} names a project, so it "
                    "stops working the day that project is renamed or moved. The "
                    "entry written now serves whichever project you choose in the "
                    "window it opens.",
                    add,
                )
            )
            continue
        if not entry.executable.is_file():
            findings.append(
                Check(
                    "desktop_entry",
                    WARN,
                    f"The menu entry {entry.path.name} runs {entry.executable}, "
                    "which is not there, so clicking it starts nothing.",
                    f"`{CLI_COMMAND} install` puts the command back on PATH",
                )
            )
    return findings


def run_doctor(
    config: ResearchConfig,
    status: dict[str, Any],
    *,
    running: list[tuple[int, str]],
    entry: str | Path | None = None,
    prefetch: bool = False,
    repair: bool = False,
    status_unavailable: str | None = None,
) -> DoctorResult:
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
    report: HealthReport = health_report(
        config, status, status_unavailable=status_unavailable
    )
    lines.extend(_check_line(check) for check in report.checks)
    if report.not_checked:
        lines.append(
            "Not checked: "
            + ", ".join(report.not_checked)
            + ". An unchecked dependency is not a healthy one."
        )
    lines.extend(_server_lines(config, running))
    lines.extend(_check_line(check) for check in desktop_entry_checks())
    return DoctorResult(tuple(lines), 1 if report.has_blocker else 0)
