"""The one command line: one research project, in a terminal or a browser.

`research-rag` resolves a project, constructs a `ResearchService`, and calls it in
this process. `ui` hands the browser workspace to the project's own generated
launcher, and `serve` is the foreground workspace host that launcher runs; there
is no second console script and no MCP surface, so the terminal and the browser
answer from the same call on the same payload and cannot drift apart.

Only a command that actually queries opens the vanilla gateway, and it is opened
on the first call rather than chosen from the command line: the BM25 index is
initialized through that gateway when a generation is loaded for querying.
Commands that only read local state — `init`, `config`, `ui`, `sources`,
`status`, `passage`, and the review commands — never start one, so they stay
quick and keep working on a machine where no UltraRAG runtime is installed yet.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from config_ultra_rag_mcp import describe_settings

from .. import bridge
from .. import launcher as launcher_module
from ..app import UI_HOST, App
from ..config import (
    CLI_COMMAND,
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    project_command,
    resolve_config,
)
from ..control import Control, ControlError, connect
from ..launcher import launcher_path, start_app, ui_launcher_state
from ..rerankers import RERANKER_MODEL_CHOICES
from ..service import ResearchService
from ..settings import SETTINGS
from ..support import DEFAULT_RETRIEVAL_METHOD, RETRIEVAL_METHODS, ResearchError
from ..ultrarag import LazyGateway, VanillaUltraRAG

CLI_NAME = CLI_COMMAND
DEFAULT_DEPTH = 10
# What a project's own .gitignore has to keep out of version control: the
# derived state that can be rebuilt, and the machine-local launcher. The
# descriptor, catalogs, and review files are small, portable, and worth keeping.
VERSION_CONTROL_NOTES = (
    ".research-rag/runtime/",
    ".research-rag/bin/",
    # Both products may serve one project while the migration runs, and each
    # generates a launcher link at its own name in the project root.
    "open-research-rag-ui.sh",
    "open-ui.sh",
)
# A process the stop sweep may signal has to name one of these, so a shell or an
# editor that merely mentions the project path is never touched. The MCP server
# this app was seeded from is deliberately absent: during the migration both
# products can serve one project, and stopping this app's workspace must not stop
# a server that a user started on purpose.
SERVICE_MARKERS = ("research_rag", "research-rag")
STOP_GRACE_SECONDS = 5.0


@asynccontextmanager
async def _service(config: ResearchConfig) -> AsyncIterator[ResearchService]:
    """Yield the service over a gateway that opens only if it is spoken to."""

    gateway = LazyGateway(config)
    try:
        yield ResearchService(config, VanillaUltraRAG(gateway, config))
    finally:
        await gateway.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Review evidence from a project's PDF and EPUB corpus, in a "
            "terminal or in a browser."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT", "."),
        help="Project root holding .research-rag (default: the current directory).",
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
        help=(
            "Absolute directory for derived state, when the project itself is on "
            "slow storage. Omit to keep it under .research-rag."
        ),
    )
    parser.add_argument(
        "--model-cache-root",
        default=None,
        help="Shared model cache (default: the user cache for this application).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Require the vanilla runtime and every model to be cached already.",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=None,
        help="ONNX Runtime threads for the embedding model; unset lets the runtime decide.",
    )
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=None,
        help="Dense index backend for a new generation (default: auto).",
    )
    parser.add_argument(
        "--reranker-model",
        choices=RERANKER_MODEL_CHOICES,
        default=None,
        help="Cross-encoder the engine loads for a reranked search.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("RESEARCH_ULTRARAG_CONFIG"),
        help="Extra settings file, layered above the per-user and project files.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override one setting for this command; repeat for more. This reaches "
            "every operation, including ingest and search, because the command "
            "resolves settings in this process."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create = commands.add_parser(
        "init",
        help="Create a project, or add .research-rag to a directory you already have.",
    )
    create.add_argument(
        "--name",
        default=None,
        help=(
            "Name to record for this project. An existing project is renamed only "
            "when this is given; the stable project id never changes."
        ),
    )
    create.add_argument(
        "--sources",
        default=None,
        help="Project-relative source directory to create (default: sources).",
    )

    commands.add_parser(
        "status", help="Report readiness and what changed since the generation."
    )

    refresh = commands.add_parser(
        "ingest", help="Build or refresh the searchable generation."
    )
    refresh.add_argument(
        "--force-recompute",
        action="store_true",
        help="Rebuild every document, chunk, and vector instead of reusing compatible ones.",
    )

    find = commands.add_parser(
        "search", help="Retrieve evidence passages for a question."
    )
    find.add_argument(
        "query", help="Research question, exact phrase, name, or concept."
    )
    find.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Maximum ranked passages to return (default: {DEFAULT_DEPTH}).",
    )
    find.add_argument(
        "--method",
        choices=sorted(RETRIEVAL_METHODS),
        default=DEFAULT_RETRIEVAL_METHOD,
        help="Retrieval half to rank with (default: hybrid, the tool's method).",
    )
    find.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder, which is on by default and is what search costs.",
    )
    find.add_argument(
        "--category",
        action="append",
        help="Keep results carrying any of these categories.",
    )
    find.add_argument(
        "--project",
        action="append",
        help="Keep results carrying any of these project tags.",
    )
    find.add_argument(
        "--keyword",
        action="append",
        help="Keep results carrying every one of these keywords.",
    )
    find.add_argument(
        "--language",
        action="append",
        help="Keep results written in any of these ISO 639 codes.",
    )
    find.add_argument(
        "--author",
        action="append",
        help="Keep results whose source has one of these names among its authors.",
    )
    find.add_argument(
        "--title",
        action="append",
        help="Keep results whose source title contains one of these phrases.",
    )
    find.add_argument(
        "--source-id", action="append", help="Search only these stable source ids."
    )
    find.add_argument(
        "--exclude-source-id",
        action="append",
        help="Search everything except these source ids.",
    )

    commands.add_parser(
        "sources", help="List the project's sources and their review state."
    )

    context = commands.add_parser(
        "passage", help="Read one passage and its immediate neighbours."
    )
    context.add_argument("chunk_id", help="Exact chunk id from a search result.")
    context.add_argument(
        "--context-chunks",
        type=int,
        default=1,
        help="Neighbours to include on each side (0-5, default: 1).",
    )

    for verb, reason_required in (("include", False), ("exclude", True)):
        change = commands.add_parser(
            verb,
            help=f"{verb.capitalize()} one source in retrieval without touching the file.",
        )
        change.add_argument(
            "source",
            nargs="?",
            help="Source path relative to the source directory.",
        )
        change.add_argument(
            "--source-id", help="The stable source id, instead of a path."
        )
        change.add_argument(
            "--reason",
            required=reason_required,
            help=(
                "Why the decision was made."
                if reason_required
                else "Why the decision was made, when it is worth recording."
            ),
        )

    review = commands.add_parser(
        "metadata",
        help="Save reviewed bibliographic metadata for one source.",
    )
    review.add_argument(
        "source", nargs="?", help="Source path relative to the source directory."
    )
    review.add_argument("--source-id", help="The stable source id, instead of a path.")
    review.add_argument("--title", help="Reviewed title.")
    review.add_argument(
        "--author", action="append", help="An author, in the order to list them."
    )
    review.add_argument("--year", type=int, help="Reviewed year of publication.")
    review.add_argument("--doi", help="Reviewed DOI.")
    review.add_argument(
        "--category", action="append", help="A category the work belongs to."
    )
    review.add_argument(
        "--keyword", action="append", help="A keyword the work carries."
    )
    review.add_argument(
        "--language",
        action="append",
        help="A language the work is written in, as an ISO 639 code.",
    )
    review.add_argument("--project", help="The project tag to file the work under.")
    review.add_argument(
        "--clear",
        action="store_true",
        help="Remove the review instead, so automatic metadata applies again.",
    )

    commands.add_parser(
        "config", help="Print the merged settings and where each value came from."
    )

    examine = commands.add_parser(
        "doctor",
        help=(
            "Report what is wrong with this installation: one line per "
            "dependency, with the command that fixes it."
        ),
    )
    examine.add_argument(
        "--prefetch-models",
        action="store_true",
        help="Download the pinned embedding and reranker models into the cache.",
    )
    examine.add_argument(
        "--repair-runtime",
        action="store_true",
        help=(
            "Move a mismatched UltraRAG runtime aside, install the pinned one, "
            "and validate it."
        ),
    )

    browser = commands.add_parser(
        "start", help="Bring this project's app up, and report where it is."
    )
    browser.add_argument(
        "--open", action="store_true", help="Open the workspace in a browser."
    )
    browser.add_argument("--port", type=int, help="Serve on a different port.")

    open_ui = commands.add_parser(
        "ui", help="Bring the app up and open the workspace in a browser."
    )
    open_ui.add_argument(
        "--no-open",
        dest="open",
        action="store_false",
        help="Start it without opening a browser.",
    )
    open_ui.add_argument("--port", type=int, help="Serve on a different port.")

    commands.add_parser(
        "clients", help="List the MCP clients attached to this project's app."
    )

    drop = commands.add_parser(
        "disconnect", help="Disconnect one attached MCP client, ending its session."
    )
    drop.add_argument("session_id", help="The session id reported by `clients`.")
    drop.add_argument("--reason", default=None, help="Why it is being dropped.")

    bridge_command = commands.add_parser(
        "mcp",
        help=(
            "Serve this project's agent surface on stdio, proxied to the running "
            "app, for an MCP client that speaks only stdio."
        ),
    )
    bridge_command.add_argument(
        "--client-name",
        default=None,
        help=(
            "Name this bridge reports itself under, so an app's client list can "
            "tell agents apart."
        ),
    )

    host = commands.add_parser(
        "serve",
        help=(
            "Serve the app in the foreground on one fixed port: the browser "
            "workspace, the agent surface, and the control API together."
        ),
    )
    host.add_argument(
        "--host",
        choices=("127.0.0.1", "localhost", "::1"),
        default=UI_HOST,
        help="Loopback address only (default: 127.0.0.1).",
    )
    host.add_argument(
        "--port",
        type=int,
        default=None,
        help="Serve on this port; omit to claim the first free one at 5051.",
    )

    stop = commands.add_parser(
        "stop",
        help="Stop this project's app, and optionally its processes.",
    )
    stop.add_argument(
        "--servers",
        action="store_true",
        help=(
            "Also stop every process of this app serving this project, including "
            "a build that is still running."
        ),
    )
    return parser


def _project_path(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve()


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The command-line settings layer, shared by every command."""

    return {
        "runtime_root": args.runtime_root,
        "model_cache_root": args.model_cache_root,
        "offline": args.offline,
        "dense_backend": args.dense_backend,
        "embedding_threads": args.embedding_threads,
        "reranker_model": args.reranker_model,
        "config_path": args.config,
        "settings_overrides": args.set_overrides,
    }


def _resolve(args: argparse.Namespace) -> ResearchConfig:
    """Resolve an existing project, reusing the source directory it recorded."""

    project = _project_path(args)
    return resolve_config(
        project,
        source_directory=configured_source_directory(project),
        **_config_kwargs(args),
    )


def _init(args: argparse.Namespace) -> dict[str, Any]:
    """Create a project, or attach the portable state to a directory that exists.

    Both are the same act: a project *is* a directory with `.research-rag` beside
    whatever is already there. Nothing this command does touches a file the user
    wrote, so pointing it at a repository that already holds work adds only the
    portable state, the source directory, and the browser launcher.
    """

    project = _project_path(args)
    created: list[str] = []
    if project.exists() and not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")
    if not project.exists():
        project.mkdir(parents=True)
        created.append("project_root")
    # An existing project records its own source directory, and a project that is
    # not there yet has no descriptor to read, so both are settled before the
    # descriptor is written.
    sources = args.sources or configured_source_directory(project)
    config = resolve_config(
        project,
        source_directory=sources,
        project_name=args.name,
        **_config_kwargs(args),
    )
    if not config.source_root.exists():
        config.source_root.mkdir(parents=True)
        created.append("source_root")
    return {
        "status": "ready",
        "project_root": str(config.project_root),
        "project_id": config.project_id,
        "project_name": config.project_name,
        "source_root": str(config.source_root),
        "created": created,
        "launcher": ui_launcher_state(config.project_root, config.portable_root),
        "keep_out_of_version_control": list(VERSION_CONTROL_NOTES),
        "next_steps": [
            f"Add PDF or EPUB sources to {config.source_root}.",
            project_command(config.project_root, "ingest"),
            project_command(config.project_root, "search", "your question"),
            project_command(config.project_root, "start", "--open"),
        ],
    }


def _project_argument(arguments: list[str]) -> str | None:
    """Return the ``--project-root`` value in one command line, if it has one."""

    for index, argument in enumerate(arguments):
        if argument == "--project-root" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--project-root="):
            return argument.split("=", 1)[1]
    return None


def _service_processes(
    project_root: Path,
    proc_root: Path = Path("/proc"),
) -> list[tuple[int, str]]:
    """Return the ``(pid, command)`` pairs serving this project.

    A process qualifies when it invokes this app by name *and* names this
    project as ``--project-root``. Both halves matter: the name keeps a shell or
    an editor that merely mentions the path out of the sweep, and the path keeps
    another project's process out of it. A relative ``--project-root`` is not
    matched, because resolving it would use this process's directory rather than
    the other one's; every launcher and every invocation passes an absolute path.

    The name has to be a whole argument, not a substring of one. This app's own
    projects all contain ``.research-rag`` in their paths, so a substring test
    matches every process that touches a project — including the vanilla gateway
    the other product in this collection starts, whose ``--workspace-root``
    names the same directory. Sweeping that would kill a gateway serving a
    server this app was told to leave alone.
    """

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    for pid in sorted(int(entry.name) for entry in entries if entry.name.isdigit()):
        if pid == os.getpid():
            continue
        try:
            raw = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        arguments = [
            part for part in raw.decode("utf-8", "replace").split("\0") if part
        ]
        if not _invokes_this_app(arguments, project_root):
            continue
        if _project_argument(arguments) != str(project_root):
            continue
        found.append((pid, " ".join(arguments)))
    return found


def _invokes_this_app(arguments: Sequence[str], project_root: Path) -> bool:
    """Whether one argument names this app as the program being run.

    A console script passes its own path, ``python -m research_rag`` passes the
    module, and either may be a relative name resolved against the other
    process's working directory. Any other argument is a value, not a program:
    a project path under ``.research-rag`` carries the product name in its
    directory and must not be mistaken for one.
    """

    inside = project_root.resolve()
    for index, argument in enumerate(arguments):
        if argument == "-m":
            if index + 1 < len(arguments) and arguments[index + 1] in SERVICE_MARKERS:
                return True
            continue
        if argument in SERVICE_MARKERS:
            return True
        if Path(argument).name not in SERVICE_MARKERS:
            continue
        try:
            resolved = Path(argument).resolve()
        except OSError:
            continue
        if resolved == inside or inside in resolved.parents:
            continue
        return True
    return False


def _alive(pid: int) -> bool:
    """Whether this process still exists and could be signalled."""

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate(pids: list[int]) -> list[int]:
    """Ask these processes to stop and then insist; return the ones killed outright."""

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.1)
    forced = [pid for pid in pids if _alive(pid)]
    for pid in forced:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    return forced


def _stop(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Stop this project's browser view, and with ``--servers`` every process serving it.

    The launcher's own message is captured rather than printed, so a caller gets
    one JSON report on stdout and nothing else.
    """

    script = launcher_path(config.portable_root)
    report: dict[str, Any] = {"project_root": str(config.project_root)}
    if script.is_file():
        stopped = subprocess.run(
            [str(script), "--stop"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["ui_launcher_status"] = stopped.returncode
        report["ui_launcher_output"] = stopped.stdout.strip()
    else:
        report["ui_launcher_status"] = None
    if not args.servers:
        return report
    found = _service_processes(config.project_root)
    report["servers"] = [{"pid": pid, "command": command} for pid, command in found]
    report["forced_pids"] = _terminate([pid for pid, _ in found])
    report["notes"] = [
        "A workspace you started in another terminal belongs to that terminal: this command stops it, and nothing here starts it again.",
        "A build interrupted this way resumes from its checkpoint on the next ingest call.",
    ]
    return report


def _metadata_body(args: argparse.Namespace) -> dict[str, Any]:
    """Build the review body, refusing a request that asks for two things at once."""

    supplied: dict[str, Any] = {
        "title": args.title,
        "authors": args.author,
        "year": args.year,
        "doi": args.doi,
        "language": args.language,
        "categories": args.category,
        "keywords": args.keyword,
        "project": args.project,
    }
    if args.clear and any(value is not None for value in supplied.values()):
        raise ResearchError(
            "--clear removes the review so automatic metadata applies again, "
            "so it cannot be combined with a field."
        )
    if args.clear:
        return {}
    return {key: value for key, value in supplied.items() if value is not None}


class Local:
    """Corpus operations answered in this process, for a project with no app up.

    A project nobody is serving has no gateway running for it, so a command that
    arrives when the app is down is not racing anything: it opens the one
    service it needs, answers, and closes it. When the app *is* up, the same
    command goes to it, so there is never a second service holding the project.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._context = _service(config)
        self._enter: Any = None
        self.service: ResearchService | None = None

    async def __aenter__(self) -> Self:
        self._enter = self._context
        self.service = await self._enter.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._enter is not None:
            await self._enter.__aexit__(*exc)
            self._enter = None

    def _require(self) -> ResearchService:
        if self.service is None:
            raise ResearchError("The local service is not open")
        return self.service

    async def status(self) -> dict[str, Any]:
        return await self._require().status()

    async def ingest(self, *, force_recompute: bool) -> dict[str, Any]:
        return await self._require().ingest(force_recompute=force_recompute)

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return await self._require().search(query, **arguments)

    async def sources(self) -> dict[str, Any]:
        return await self._require().list_sources()

    async def passage(self, chunk_id: str, *, context_chunks: int) -> dict[str, Any]:
        return await self._require().get_passage(
            chunk_id, context_chunks=context_chunks
        )

    async def set_source_inclusion(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self._require().set_source_inclusion(
            source_path=source_path,
            source_id=source_id,
            included=included,
            reason=reason,
        )

    async def set_source_metadata(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._require().set_source_metadata(
            metadata, source_path=source_path, source_id=source_id
        )


class Remote:
    """The running app's operations, awaited.

    `Control` speaks HTTP and is synchronous, because two of the commands that
    use it are synchronous. This is the same interface `Local` implements, so
    `_operate` does not know which one answered.
    """

    def __init__(self, control: Control) -> None:
        self.control = control

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.control.close()

    async def status(self) -> dict[str, Any]:
        return self.control.status()

    async def ingest(self, *, force_recompute: bool) -> dict[str, Any]:
        return self.control.ingest(force_recompute=force_recompute)

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self.control.search(query, **arguments)

    async def sources(self) -> dict[str, Any]:
        return self.control.sources()

    async def passage(self, chunk_id: str, *, context_chunks: int) -> dict[str, Any]:
        return self.control.passage(chunk_id, context_chunks=context_chunks)

    async def set_source_inclusion(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return self.control.set_source_inclusion(
            source_path=source_path,
            source_id=source_id,
            included=included,
            reason=reason,
        )

    async def set_source_metadata(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self.control.set_source_metadata(
            source_path=source_path, source_id=source_id, metadata=metadata
        )


async def _operate(
    args: argparse.Namespace,
    operations: Local | Remote,
) -> dict[str, Any]:
    """Call the one operation this command names, and return its payload."""

    command = args.command
    if command == "status":
        return await operations.status()
    if command == "ingest":
        return await operations.ingest(force_recompute=args.force_recompute)
    if command == "search":
        return await operations.search(
            args.query,
            top_k=args.top_k,
            categories_any=args.category,
            projects_any=args.project,
            keywords=args.keyword,
            languages_any=args.language,
            authors_any=args.author,
            titles_any=args.title,
            source_ids=args.source_id,
            exclude_source_ids=args.exclude_source_id,
            retrieval_method=args.method,
            rerank=not args.no_rerank,
        )
    if command == "sources":
        return await operations.sources()
    if command == "passage":
        return await operations.passage(
            args.chunk_id, context_chunks=args.context_chunks
        )
    if command in {"include", "exclude"}:
        return await operations.set_source_inclusion(
            source_path=args.source,
            source_id=args.source_id,
            included=command == "include",
            reason=args.reason,
        )
    if command == "metadata":
        return await operations.set_source_metadata(
            source_path=args.source,
            source_id=args.source_id,
            metadata=_metadata_body(args),
        )
    raise ResearchError(f"Unknown command: {command}")


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one command printed, and the exit code that follows from it."""

    payload: dict[str, Any] | None = None
    exit_code: int = 0
    text: str | None = None


async def _serve(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:
    """Serve the app in the foreground until this process is stopped.

    The generated launcher runs this, so it takes an explicit port and never
    chooses one: the launcher already made that choice while holding the lock
    that makes it exclusive. A caller running it by hand omits `--port` and gets
    the first free port at or above the default, claimed by binding before
    uvicorn starts so two apps cannot both believe they hold it.
    """

    port = args.port or launcher_module.DEFAULT_UI_PORT
    app = App(config, port=port)
    await app.start()
    if app.error is not None:
        raise ResearchError(app.error)
    print(app.url, flush=True)
    try:
        await app.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await app.stop()
    return CommandResult()


def _start(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Bring the app up, and say where it is and who is attached."""

    outcome = start_app(config, port=args.port, open_browser=args.open)
    if args.open and outcome.get("url"):
        _open_browser(outcome["url"])
    if not outcome.get("running"):
        detail = outcome.get("stderr") or outcome.get("stdout") or ""
        raise ResearchError(
            f"The app for {config.project_root} did not start: {detail} "
            f"(log: {outcome.get('log')})"
        )
    return outcome


def _open_browser(url: str) -> None:
    """Open a browser, and say so when this machine has none to open."""

    import webbrowser

    # A machine with no desktop session has no browser to open, and a browser
    # that fails to start is not a reason to fail the command: the URL is the
    # answer either way.
    with contextlib.suppress(Exception):
        if webbrowser.open(url):
            return
    print(f"Open {url} in a browser.", file=sys.stderr)


def _clients(config: ResearchConfig) -> dict[str, Any]:
    """List the agents attached to the app, or say that there is no app."""

    with _control(config) as control:
        if control is None:
            return {
                "running": False,
                "clients": [],
                "note": "No app is running for this project; start it with "
                f"'{CLI_NAME} start'.",
            }
        return {
            "running": True,
            "url": control.base_url,
            "mcp_url": f"{control.base_url}/mcp",
            "clients": control.clients(),
        }


def _disconnect(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Drop one attached agent, which ends its session."""

    with _control(config) as control:
        if control is None:
            raise ResearchError(
                f"No app is running for {config.project_root}, so no client is "
                f"attached. Start it with '{CLI_NAME} start'."
            )
        return control.disconnect(
            args.session_id, args.reason or "Disconnected by request."
        )


async def _doctor(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:
    """Report the installation, and run only the operation the flags name."""

    from ..doctor import run_doctor

    # The report is built from the same service call the `status` command makes,
    # so the two surfaces cannot disagree about this project.
    async with _service(config) as service:
        status = await service.status()
    result = run_doctor(
        config,
        status,
        running=_service_processes(config.project_root),
        prefetch=args.prefetch_models,
        repair=args.repair_runtime,
    )
    return CommandResult(text=result.text(), exit_code=result.exit_code)


@contextlib.asynccontextmanager
async def _operations(config: ResearchConfig) -> AsyncIterator[Local | Remote]:
    """Reach the app when it is up, and build the one service when it is not."""

    handle = connect(config)
    if handle is not None:
        async with Remote(handle) as remote:
            yield remote
        return
    async with Local(config) as local:
        yield local


def _control(config: ResearchConfig) -> AbstractContextManager[Control | None]:
    """A synchronous handle on the app, for the commands that only read."""

    return nullcontext(connect(config))


async def _run(args: argparse.Namespace) -> CommandResult:
    """Resolve the project, run the named command, and return what to print."""

    if args.command == "init":
        return CommandResult(payload=_init(args))
    config = _resolve(args)
    apply_process_priority(config.nice)
    if args.command == "config":
        print(
            describe_settings(
                SETTINGS, config.settings.as_values(), config.settings_provenance
            )
        )
        return CommandResult()
    if args.command == "start":
        return CommandResult(payload=_start(args, config))
    if args.command == "ui":
        return CommandResult(payload=_start(args, config))
    if args.command == "clients":
        return CommandResult(payload=_clients(config))
    if args.command == "disconnect":
        return CommandResult(payload=_disconnect(args, config))
    if args.command == "serve":
        return await _serve(args, config)
    if args.command == "stop":
        return CommandResult(payload=_stop(args, config))
    if args.command == "doctor":
        return await _doctor(args, config)
    async with _operations(config) as operations:
        payload = await _operate(args, operations)
    return CommandResult(payload=payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "mcp":
        # The bridge owns stdio and its own event loop, so it runs before this
        # process starts one rather than inside it.
        try:
            config = _resolve(args)
        except (
            ConfigurationError,
            ControlError,
            ResearchError,
            OSError,
            ValueError,
        ) as exc:
            raise SystemExit(f"{CLI_NAME}: {exc}") from exc
        bridge.main(config, name=args.client_name)
        return
    try:
        result = asyncio.run(_run(args))
    except (
        ConfigurationError,
        ControlError,
        ResearchError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"{CLI_NAME}: {exc}") from exc
    if result.text is not None:
        sys.stdout.write(result.text)
    elif result.payload is not None:
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    if result.exit_code:
        raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
