"""The one command line: one research project, in a terminal or a browser.

`research-rag` resolves a project, constructs a `ResearchService`, and calls it in
this process. `ui` serves the browser workspace and the agent surface from this
terminal, in the foreground, and there is no other way to bring an app up: a
command that starts one stays in the terminal that ran it, so Ctrl-C and a closed
window both reach it and nothing is left running that nobody is watching. One
console script reaches all of it, and the agent surface lives in `surfaces/mcp.py`
rather than here, so the terminal and the browser answer from the same call on the
same payload.

Only a command that queries opens the vanilla gateway, and it opens on the first
call rather than as a command-line choice: a generation initializes BM25 through
that gateway as it loads for querying. A command that reads only local state never
starts one, so it works with no UltraRAG runtime installed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .. import bridge
from ..app import (
    UI_HOST,
    App,
    _claim_loopback_port,
    _own_tty,
    recorded_pid,
    running_url,
)
from ..config import (
    CLI_COMMAND,
    DEFAULT_UI_PORT,
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    project_command,
    resolve_config,
)
from ..control import Control, ControlError, connect
from ..registry import (
    account_projects,
    detached_from_terminal,
    project_app_state,
    registry_path,
)
from ..registry import register as register_project
from ..registry import resolve as resolve_registered
from ..rerankers import RERANKER_MODEL_CHOICES
from ..service import ResearchService
from ..settings import SETTINGS
from ..settings_document import describe_costs, describe_docs
from ..settings_layers import describe_settings
from ..support import DEFAULT_RETRIEVAL_METHOD, RETRIEVAL_METHODS, ResearchError
from ..tool_views import lean_status
from ..ultrarag import LazyGateway, VanillaUltraRAG

CLI_NAME = CLI_COMMAND
DEFAULT_DEPTH = 10
# The variable a client entry that cannot pass an argument sets instead. It is
# the counterpart of `RESEARCH_RAG_PROJECT_ROOT`, and it exists for the same
# reason: a client that offers only an environment block still has to name a
# project.
PROJECT_NAME_ENV = "RESEARCH_RAG_PROJECT_NAME"
# How far above the default port an attached app looks for a free one. A port a
# reader named is never moved, so a named port that is taken fails and says so
# rather than being served somewhere they did not ask for.
_PORT_ATTEMPTS = 32

# What a project's own .gitignore keeps out of version control: the derived
# state that can be rebuilt. The descriptor, catalogs, and review files are
# small, portable, and worth keeping.
VERSION_CONTROL_NOTES = (".research-rag/runtime/",)
# A process the stop sweep may signal has to name one of these, so a shell or an
# editor that merely mentions the project path is never touched. The MCP server
# this app was seeded from is deliberately absent: during the migration both
# products can serve one project, and stopping this app's workspace must not stop
# a server a user started on purpose.
SERVICE_MARKERS = ("research_rag", "research-rag")
STOP_GRACE_SECONDS = 5.0

# The menu `research-rag help` prints. argparse prints a usage block and a
# per-command one; it cannot print the order of the work, which group a command
# belongs to, or the fact that the three surfaces share one app. Each command
# appears in exactly one group, and
# `test_the_help_menu_accounts_for_every_command` fails when one is added to the
# parser and not here.
HELP_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "First time",
        (
            (
                "init",
                (
                    "Create the project record and .research-rag, or attach to an "
                    "existing directory."
                ),
            ),
            (
                "ingest",
                (
                    "Extract, chunk, embed, and index the PDFs and EPUBs in the "
                    "sources directory. Call it again to resume a build that ran "
                    "out of time."
                ),
            ),
        ),
    ),
    (
        "What state the project is in",
        (
            (
                "status",
                (
                    "Whether the corpus is current, and what changed since the "
                    "generation. A stale answer names the call that closes it."
                ),
            ),
            (
                "sources",
                "The inventory: every file, which are indexed, and why one is not.",
            ),
            (
                "doctor",
                (
                    "What is wrong with this installation: one line per dependency, "
                    "each with the command that fixes it."
                ),
            ),
            (
                "generations",
                "Every generation on disk with its size, and the one search reads.",
            ),
            (
                "remove-generation",
                (
                    "Delete a generation search does not read, after repeating its id. "
                    "Rebuilding costs one ingestion."
                ),
            ),
        ),
    ),
    (
        "Reading",
        (
            (
                "search",
                (
                    "Evidence passages for a question, always hybrid and always "
                    "reranked. `help filters` names the six filter layers and the two "
                    "retrieval switches."
                ),
            ),
            (
                "passage",
                "One passage and its neighbours, for reading around a result.",
            ),
        ),
    ),
    (
        "Deciding about a source",
        (
            (
                "metadata",
                "Reviewed bibliography for one source. It survives a rebuild.",
            ),
            (
                "exclude",
                "Take a source, or one passage with --chunk, out of retrieval. The file stays.",
            ),
            (
                "include",
                "Put an excluded source or passage back into retrieval.",
            ),
        ),
    ),
    (
        "Running the app",
        (
            (
                "start",
                "Bring the app up and report where it is. `ui` also opens a browser.",
            ),
            (
                "projects",
                "Every project this installation knows, and which of them are up.",
            ),
            (
                "clients",
                "The agents attached to this project's app.",
            ),
            (
                "disconnect",
                "End one attached agent's session.",
            ),
            (
                "stop",
                "Stop this project's app and what it started.",
            ),
        ),
    ),
    (
        "Agents and settings",
        (
            (
                "mcp",
                (
                    "The agent surface on stdio for one project, named with "
                    "--project-name and proxied to that project's app. `help agents` "
                    "has the client entry."
                ),
            ),
            (
                "config",
                (
                    "Every effective setting and the layer it came from. It never "
                    "writes. `help settings` names the layers and what a change costs."
                ),
            ),
        ),
    ),
    (
        "This installation",
        (
            (
                "install",
                (
                    "Put this command on the account's PATH, and with --desktop add "
                    "one menu entry for the installation. --uninstall removes what it "
                    "wrote."
                ),
            ),
            (
                "update",
                (
                    "Compare this installation with the latest published release, "
                    "and with --apply move to it: stop every app, put the release "
                    "in place, and print the command that starts each app again. "
                    "The branch head is reported beside the release answer."
                ),
            ),
        ),
    ),
)

# Pages for a subject rather than a command. A command name resolves to that
# command's own usage, so these carry only what no single command can.
HELP_TOPICS: dict[str, str] = {
    "filters": """\
search takes six filter layers, each an --option that may be repeated, and two
options that reach one source at a time. A filter given nothing is not applied.
`search --help` names every option and what it keeps.

The six layers are --category, --project, --keyword, --language, --author, and
--title, and they read the source's reviewed metadata rather than the extracted
text, so a filter matching nothing reports nothing rather than a fallback. Set the
metadata with `metadata` and the filters start working.

--method and --no-rerank decide how retrieval runs, and they are the app's
settings rather than a per-search choice. The agent surface has neither, nor any
method, depth, or reranking argument: it is always hybrid, always reranked, and
its depth is a setting. A method chosen per call is a number a reader cannot
reproduce from the answer they were given.
""",
    "settings": """\
A project's settings resolve in four layers, each overriding the one above it, and
`config` prints every effective value with the layer it came from:

  1. the packaged defaults in default.toml
  2. the per-user file, ~/.config/research-rag/config.toml
  3. the project file, <project>/.research-rag/config.toml
  4. an extra file named by --config PATH

`config` never writes. The browser workspace writes the project file, refusing a
change it cannot validate and naming what the change costs. The two global options
below are per-call and are recorded nowhere:

  --set key=value    override one value for this command, repeatable
  --config PATH      add one more layer for this command

What a key does is written once, as `Setting.doc` in src/research_rag/settings.py.
`config` prints that sentence beside each key and the workspace's Config tab shows
the same one.

What a change costs is computed, not declared, and `config` prints it: each key is
costed by applying a change to it and recomputing what a build records, the
retrieval-policy fingerprint, the recorded chunk settings, and the recorded
models. A key that moves none of them costs nothing and the next search uses the
new value at once, which is what `runtime.tool_detail` and the batch sizes do. A
key that moves the fingerprint or the chunk settings makes search answer `stale`
until `ingest` runs again. A key that changes a model downloads it: a new
embedding model recomputes every vector, and a new reranker is loaded by the next
search.

`doctor` reports which layer a value came from, and the file path it would be
changed in.
""",
    "agents": """\
The app serves MCP at /mcp on its own port, so a client that can open a socket
needs only the URL. A client that speaks only stdio uses `research-rag mcp
--project-name NAME`, which brings that project's app up and proxies to it.

The entry names a project, never a directory, so one entry written on one machine
works on every machine where that project was initialised. The directory is a fact
of each machine: the app resolves the name through this installation's own project
record. `--project-root` and `--project` are refused here, and so is
`RESEARCH_RAG_PROJECT_ROOT`.

A name no project on this machine answers with a connection and one tool: `status`
reports that the project is not initialised and gives the `init` command that
creates it. Run that command and the same entry serves the project.

Let the project print the entry for the machine it runs on, because the port is
chosen at start and a hard-coded URL goes stale the first time it moves:

  research-rag --project-root DIR doctor --mcp-entry
  research-rag --project-root DIR doctor --check-entry <file>

Two ready-to-copy templates ship with the source: mcp_settings.example.json for a
client using an mcpServers object, and kilo-mcp.example.jsonc for one using a
Kilo-style mcp object. Replace the executable path and the project name in either.

Set RESEARCH_RAG_CLIENT_NAME so the app's client list can tell agents apart,
or RESEARCH_RAG_PROJECT_NAME when the client can pass an environment but not
an argument. `clients` lists the agents and `disconnect` ends one; the workspace
shows the same list, so an agent ended in the browser is gone from the terminal
too.

An agent gets eight tools and one resource, and every answer is the lean
projection: a question at a time, no inventory, no scores. `status` is a verdict
naming the call that closes a gap, not the whole corpus state. The full payload is
`status --verbose` here and the workspace there.

A generated sentence citing a passage is not for direct quotation. Take the
quotation from the original file, which the workspace opens beside the passage and
the agent reaches with get_passage.
""",
}


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
        epilog=(
            f"`{CLI_NAME} help` prints the same commands grouped by the work, "
            f"and `{CLI_NAME} help TOPIC` explains a subject."
        ),
    )
    parser.add_argument(
        # None rather than "." so a call naming a registered project has not also
        # named a path. An absent value means the current directory, resolved
        # where it is used.
        "--project-root",
        default=os.environ.get("RESEARCH_RAG_PROJECT_ROOT"),
        help="Project root holding .research-rag (default: the current directory).",
    )
    parser.add_argument(
        "--start-ui",
        dest="start_ui",
        action="store_true",
        help=(
            "Open the workspace in a browser as well as serving it. Serving never "
            "opens one by itself, so a browser appears only when this is passed."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help=(
            "Print this app's version, the installed one, the shared workspace's, "
            "and whether a restart is required. Needs no project."
        ),
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("RESEARCH_RAG_PROJECT"),
        help=(
            "Name or id of a project this installation registered, in place of "
            "--project-root. `research-rag projects` lists them."
        ),
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_RAG_RUNTIME_ROOT"),
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
        default=os.environ.get("RESEARCH_RAG_CONFIG"),
        help="Extra settings file, layered above the per-user and project files.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override one setting for this command; repeat for more. It reaches "
            "every operation, including ingest and search, because the command "
            "resolves settings in this process."
        ),
    )
    # Not `required=True`, because `--version` is an answer of its own and asks
    # for no command. A call with neither prints the same refusal argparse would.
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

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
        "projects",
        help=(
            "List every project this installation registered, and whether its app "
            "is up."
        ),
    )
    # A command name is a valid topic, so the choices are every registered
    # command beside the subject pages. They come from the menu rather than
    # being typed out, so a command added to the parser and the menu is one
    # change, and the drift test below notices one added to neither.
    help_command = commands.add_parser(
        "help",
        help=(
            "Print the menu, or one page of it: a subject, or a command's own "
            "usage. Needs no project."
        ),
    )
    help_command.add_argument(
        "topic",
        nargs="?",
        choices=sorted(
            set(HELP_TOPICS)
            | {name for _, entries in HELP_GROUPS for name, _ in entries}
        ),
        help=(
            "A subject to explain, or a command whose usage to print. Omit for "
            "the whole menu."
        ),
    )

    status = commands.add_parser(
        "status", help="Report readiness and what changed since the generation."
    )
    status.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print the complete status payload instead of the readiness answer: "
            "the corpus counts, the retained generations, the retrieval policy, "
            "and every dependency check."
        ),
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

    generations = commands.add_parser(
        "generations",
        help="List the retained generations, or search one instead of the current one.",
    )
    generations.add_argument(
        "--use",
        dest="use_generation",
        metavar="GENERATION_ID",
        help=(
            "Point the project at this retained generation, after validating its "
            "artifacts and both indexes. Search reads it from the next call."
        ),
    )

    remove = commands.add_parser(
        "remove-generation",
        help="Delete a retained generation that is not the one search reads.",
    )
    remove.add_argument(
        "generation_id", metavar="GENERATION_ID", help="The directory name to remove."
    )
    remove.add_argument(
        "--confirm",
        required=True,
        metavar="GENERATION_ID",
        help=(
            "Repeat the id. Removal is permanent and the space comes back only "
            "from a rebuild."
        ),
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
            help=(
                f"{verb.capitalize()} one source or one passage in retrieval "
                "without touching the file."
            ),
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
            "--chunk",
            metavar="CHUNK_ID",
            help="The chunk id of one passage, instead of a whole source.",
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
        "--mcp-entry",
        action="store_true",
        help=(
            "Print the MCP client entries for the resolved configuration: the "
            "URL entry for a client that can open a socket, and the stdio entry "
            "for one that cannot."
        ),
    )
    examine.add_argument(
        "--check-entry",
        metavar="PATH",
        default=None,
        help="Report on a client entry file without changing it.",
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
        "start", help="Serve this project's app from this terminal, and say where."
    )
    browser.add_argument("--port", type=int, help="Serve on a different port.")

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
            "Serve the agent surface on stdio for the project named by "
            "--project-name, proxied to that project's app."
        ),
    )
    bridge_command.add_argument(
        "--project-name",
        default=os.environ.get(PROJECT_NAME_ENV),
        help=(
            "Name this project was initialised under, which the app on this "
            "machine resolves to a directory. `help agents` has the client entry."
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

    setup = commands.add_parser(
        "install",
        help=(
            "Put this command on the account's PATH, and --desktop give this "
            "installation one menu entry that serves a project in a window."
        ),
    )
    setup.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove what `install` wrote for the same selector, and nothing else.",
    )
    setup.add_argument(
        "--desktop",
        action="store_true",
        help=(
            "Write the one freedesktop entry for this installation instead of the "
            "command on PATH. It opens a terminal window and serves whichever "
            "project you choose there, so no project is named in the entry."
        ),
    )
    setup.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace a file this app did not write. Without it such a file is "
            "reported and left alone."
        ),
    )

    refresh_install = commands.add_parser(
        "update",
        help=(
            "Compare this installation with the latest published release, and "
            "--apply move to it."
        ),
    )
    refresh_install.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the update. Without it nothing is written and the command "
            "only reports."
        ),
    )
    return parser


def _project_path(args: argparse.Namespace) -> Path:
    """Return the project this call names, by registry selector or by path.

    `--project` and `--project-root` are two ways to say the same thing, so a
    call that supplies both is refused rather than resolved by precedence.
    """

    selector = getattr(args, "project", None)
    named_root = getattr(args, "project_root", None)
    if selector and named_root:
        raise ConfigurationError(
            f"--project {selector!r} and --project-root {str(named_root)!r} name "
            "two projects; pass one."
        )
    if selector:
        return resolve_registered(selector).project_root
    return Path(named_root or ".").expanduser().resolve()


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:

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

    A project *is* a directory with `.research-rag` beside whatever is already
    there, and no file the user wrote is touched.
    """

    project = _project_path(args)
    created: list[str] = []
    if project.exists() and not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")
    if not project.exists():
        project.mkdir(parents=True)
        created.append("project_root")
    # An existing project records its own source directory, and a project that is
    # not there yet has no descriptor to read, so both settle before the
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
    try:
        registered = register_project(
            config.project_id, config.project_name, config.project_root
        )
    except (OSError, ResearchError) as exc:
        raise ConfigurationError(
            f"The project at {config.project_root} was created, but this "
            f"installation could not record it ({exc}). Keep addressing it with "
            "--project-root until the record at "
            f"{registry_path()} can be written."
        ) from exc
    return {
        "status": "ready",
        "project_root": str(config.project_root),
        "project_id": config.project_id,
        "project_name": config.project_name,
        "source_root": str(config.source_root),
        "created": created,
        "registered": registered.as_record(),
        "registry_path": str(registry_path()),
        "keep_out_of_version_control": list(VERSION_CONTROL_NOTES),
        "next_steps": [
            f"Add PDF or EPUB sources to {config.source_root}.",
            project_command(config.project_root, "ingest"),
            project_command(config.project_root, "search", "your question"),
            project_command(config.project_root, "start"),
        ],
    }


def _project_argument(arguments: list[str]) -> str | None:

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

    A process qualifies when it invokes this app by name *and* names this project
    as ``--project-root``. A relative ``--project-root`` is not matched, because
    resolving it would use this process's directory rather than the other one's.

    The name must be a whole argument, not a substring of one. This app's own
    projects all contain ``.research-rag`` in their paths, so a substring test
    matches every process that touches a project, including the vanilla gateway
    the other product in this collection starts, whose ``--workspace-root`` names
    the same directory.
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
    process's working directory. A project path under ``.research-rag`` carries
    the product name in its directory and is a value, not a program.
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


def _ask_to_stop(pid: int) -> OSError | None:
    """Ask one app to stop, and report why it could not be asked."""

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return exc
    return None


def _stop_notes(detached: bool) -> list[str]:
    """The sentences `stop` returns, leading with the condition it found.

    A detached app is named first because it is the one that does not end with
    its terminal, so a reader who started one and walked away needs to know that
    the terminal they left is not what is holding the project.
    """

    notes = [
        (
            "An app belongs to the terminal that started it: it ends there, "
            "and this command does not start it again."
        )
    ]
    if detached:
        notes.insert(
            0,
            "This app is serving with no terminal attached, so no terminal holds "
            "it and closing one does not end it. Start it again from a terminal "
            "once this one is stopped.",
        )
    return notes


def _stop(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Stop this project's app, and with ``--servers`` every process serving it.

    An app belongs to the terminal that started it, so this command reports which
    terminal owns it and says plainly that stopping it here leaves that terminal
    without an app. It never starts one afterwards: nothing about an app outliving
    the terminal that began it.
    """

    app_state = project_app_state(config.project_root).get("app") or {}
    report: dict[str, Any] = {
        "project_root": str(config.project_root),
        "running": bool(app_state.get("running")),
        "attached_to": app_state.get("attached_to"),
        "detached": app_state.get("detached"),
        "url": app_state.get("url"),
    }
    # The app is asked to stop, never killed: it closes its own gateway and releases
    # its own lock on the way out, and a terminal that is still there says so.
    pid = recorded_pid(config)
    report["stopped"] = False
    if pid is not None:
        report["stopped"] = _ask_to_stop(pid) is None
        report["pid"] = pid
    if not args.servers:
        report["notes"] = _stop_notes(bool(app_state.get("detached")))
        return report
    found = _service_processes(config.project_root)
    report["servers"] = [{"pid": pid, "command": command} for pid, command in found]
    report["forced_pids"] = _terminate([pid for pid, _ in found])
    report["notes"] = [
        "An app you started in another terminal belongs to that terminal: this command stops it and does not start it again.",
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
            "--clear cannot be combined with a field: it removes the review so "
            "automatic metadata applies again."
        )
    if args.clear:
        return {}
    return {key: value for key, value in supplied.items() if value is not None}


class Local:
    """Corpus operations answered in this process, for a project with no app up.

    When the app *is* up, the same command goes to it, so there is never a second
    service holding the project.
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

    async def set_chunk_inclusion(
        self,
        *,
        chunk_id: str,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self._require().set_chunk_inclusion(
            chunk_id, included=included, reason=reason
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

    async def generations(self) -> dict[str, Any]:
        # The inventory already exists in the status payload, so a listing reads
        # it rather than walking the directory a second time with a second shape.
        status = await self._require().status()
        generations = status.get("generations")
        return {
            "generations": generations if isinstance(generations, list) else [],
            "retained_generation_count": int(
                status.get("retained_generation_count") or 0
            ),
            "retained_generation_bytes": int(
                status.get("retained_generation_bytes") or 0
            ),
            "current_generation_id": status.get("generation_id"),
        }

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return await self._require().use_generation(generation_id)

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return await self._require().remove_generation(generation_id, confirm=confirm)


class Remote:
    """The running app's operations, awaited.

    `Control` speaks HTTP and is synchronous, because two of the commands that
    use it are synchronous. It is the same interface `Local` implements, so
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

    async def set_chunk_inclusion(
        self,
        *,
        chunk_id: str,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return self.control.set_chunk_inclusion(
            chunk_id=chunk_id, included=included, reason=reason
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

    async def generations(self) -> dict[str, Any]:
        return await self.control.generations()

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return await self.control.use_generation(generation_id)

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return await self.control.remove_generation(generation_id, confirm=confirm)


async def _operate(
    args: argparse.Namespace,
    operations: Local | Remote,
) -> dict[str, Any]:
    """Call the one operation this command names, and return its payload."""

    command = args.command
    if command == "status":
        payload = await operations.status()
        # The terminal is an agent's second bounded reader: the same projection,
        # so the two cannot answer differently. `--verbose` is where a person
        # goes for the complete payload the workspace reads.
        return dict(payload) if args.verbose else lean_status(payload)
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
        # One decision about retrievable evidence, two subjects. Each refusal
        # names the subject it is missing or the pair that contradict each other,
        # because a command that silently ignored one of them would record a
        # decision the reader did not make.
        if args.chunk and (args.source or args.source_id):
            raise ResearchError(
                "--chunk decides about one passage and a path or --source-id "
                "about a whole source; name one of them."
            )
        if args.chunk:
            return await operations.set_chunk_inclusion(
                chunk_id=args.chunk,
                included=command == "include",
                reason=args.reason,
            )
        if not args.source and not args.source_id:
            raise ResearchError(
                "Name the passage with --chunk CHUNK_ID, or the source with a "
                "path or --source-id."
            )
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
    if command == "generations":
        if args.use_generation:
            return await operations.use_generation(args.use_generation)
        return await operations.generations()
    if command == "remove-generation":
        return await operations.remove_generation(
            args.generation_id, confirm=args.confirm
        )
    raise ResearchError(f"Unknown command: {command}")


async def _projects(args: argparse.Namespace) -> dict[str, Any]:
    """Report every registered project, and whether an app is serving it.

    The listing is the account's own record read once, so the workspace's
    selector names the same projects this command does. A project whose app is
    not running is reported as such rather than started, because a listing is a
    question about what exists.
    """

    return account_projects()


@dataclass(frozen=True, slots=True)
class CommandResult:
    payload: dict[str, Any] | None = None
    exit_code: int = 0
    text: str | None = None


async def _serve_attached(
    config: ResearchConfig,
    *,
    port: int | None,
    open_browser: bool,
) -> CommandResult:
    """Serve this project in this terminal until the terminal or the app ends.

    Attached is the whole point: the process stays in the foreground and in this
    terminal's process group, so Ctrl-C and a closed window both reach it, and
    the stop below shuts the gateway it opened down rather than orphaning one.

    The port is claimed by binding it, so a second project on the same machine
    walks past a taken one rather than failing where a second app would have
    moved it.
    """

    # A port named by the caller is the one they asked for, and it is tried once:
    # its refusal is the app's own sentence, which names the alternative. A port
    # chosen here is this app's to move, so it walks forward past one that is taken.
    first = port if port is not None else _a_free_port()
    app: App | None = None
    for offset in (0,) if port is not None else range(_PORT_ATTEMPTS):
        app = App(config, port=first + offset)
        await app.start()
        if app.error is None:
            break
    if app is None or app.error is not None:
        # Every candidate was taken. The last refusal is the app's own, naming the
        # port and the fact that a project already serves it.
        raise ResearchError(app.error if app is not None and app.error else "no port")
    if open_browser:
        _open_browser(app.url)
    _attached_banner(config, app)
    with _closing_with_the_terminal() as closing:
        try:
            await _wait_until_stopped(app, closing)
        finally:
            await app.stop()
    sys.stdout.write(f"Stopped. {config.project_name} is no longer served.\n")
    return CommandResult()


async def _wait_until_stopped(app: App, closing: _Closing) -> None:
    """Return when the app stops serving, when the terminal closes, or on Ctrl-C.

    Three things can end an attached run and they are the same ending: the
    serving task finishing on its own, this terminal going away, and the reader
    pressing Ctrl-C. Whichever arrives first, the caller stops the app on the way
    out, so a gateway this app opened is never left serving nothing.
    """

    serving = asyncio.ensure_future(app.wait())
    asked = asyncio.ensure_future(closing.asked.wait())
    try:
        await asyncio.wait((serving, asked), return_when=asyncio.FIRST_COMPLETED)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Only the wait on the closing is cancelled here. The serving task is left
        # for `App.stop`, which asks uvicorn to shut down and then waits for it;
        # cancelling this waiter would put a CancelledError inside that task and
        # turn an orderly stop into a traceback.
        asked.cancel()


def _a_free_port() -> int:
    """Return a port nothing is listening on, or None when the range is full.

    This is a probe, and the app's own claim is the listen, so the answer can be
    taken by something else between the two. The window is the length of one
    socket round trip and the alternative — starting a whole app to see whether it
    binds — is worse: it builds a service and records state for a port this call
    is only going to measure.
    """

    first = DEFAULT_UI_PORT
    for candidate in range(first, first + _PORT_ATTEMPTS):
        try:
            claim = _claim_loopback_port(UI_HOST, candidate)
        except OSError:
            continue
        claim.close()
        return candidate
    raise ResearchError(
        f"No free loopback port in the next {_PORT_ATTEMPTS} from {first}, so the "
        f"app for this project was not started; pass --port to choose one."
    )


def _attached_banner(config: ResearchConfig, app: App) -> None:
    """Say what is running, where it is, and what closes it."""

    lines = [
        f"{config.project_name} — workspace attached to {_own_tty() or 'no terminal'}",
        f"  {app.url}",
        f"  pid {os.getpid()} · log {config.state_root / 'logs' / 'research-rag-ui.log'}",
        "  Ctrl-C stops the app and the gateway it started.",
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


class _Closing:
    """The stop a terminal closing, or a signal, asks for."""

    def __init__(self) -> None:
        self.asked = asyncio.Event()

    def request(self) -> None:
        """Say once that the terminal is going, and ask the app to stop."""

        if not self.asked.is_set():
            sys.stdout.write("\nClosing.\n")
            sys.stdout.flush()
            self.asked.set()


@contextmanager
def _closing_with_the_terminal() -> Iterator[_Closing]:
    """Treat a closed window and a `kill` as the Ctrl-C they are.

    Closing a terminal sends SIGHUP and a `kill` sends SIGTERM. Python's default
    for both ends the process, which would leave the gateway this app opened
    running with nothing serving it, so each is turned into the orderly stop a
    Ctrl-C already performs. A platform that cannot install a handler for one of
    them keeps its default rather than refusing to serve.
    """

    closing = _Closing()
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for number in (signal.SIGHUP, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, closing.request)
        except (NotImplementedError, RuntimeError, ValueError, OSError):
            continue
        installed.append(number)
    try:
        yield closing
    finally:
        for number in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(number)


def _only_project_or_ask() -> Path:
    """The project a bare call serves, chosen by the reader when there is a choice.

    A bare call is the one command a reader types without thinking, so it needs no
    project argument to be useful. Several projects are a decision only the reader
    can make, so it is asked for in the terminal rather than guessed, and a
    terminal that cannot answer — a script, a pipe — is given the list and the
    command to run.

    The list is the one `projects` prints, so the two cannot disagree about what
    this installation holds.
    """

    registered = account_projects()["projects"]
    if not registered:
        raise ConfigurationError(
            "No project is initialised on this machine, so there is no workspace "
            f"to open. Create one with '{CLI_NAME} init --project-root DIR --name "
            "NAME'."
        )
    if len(registered) == 1:
        return Path(str(registered[0]["project_root"]))
    if not sys.stdin.isatty():
        raise ConfigurationError(
            f"This installation holds {len(registered)} projects, so a bare call "
            "cannot choose between them. Name one: "
            + ", ".join(
                f"--project {str(entry['project_name'])!r}" for entry in registered
            )
            + "."
        )
    sys.stdout.write("Which project\n")
    for index, entry in enumerate(registered, start=1):
        running = " — already served" if entry["app"]["running"] else ""
        if entry["app"].get("detached"):
            running = " — served detached, which is a state this app does not serve"
        sys.stdout.write(f"  [{index}] {entry['project_name']}{running}\n")
    sys.stdout.write("  [0] none of these\n")
    sys.stdout.flush()
    answer = input("Number: ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(registered):
        raise ConfigurationError("No project chosen, so nothing was started.")
    return Path(str(registered[int(answer) - 1]["project_root"]))


def _is_detached(config: ResearchConfig) -> bool:
    """Whether this project's app is serving with no terminal attached.

    The app holds itself to being served from the terminal that started it, so a
    serving process that has none is a state a reader is told about rather than
    one that is served around. The answer comes from the live process, so a
    terminal file that was never written cannot make an attached app read as
    detached.
    """

    return detached_from_terminal(config, running_url(config) is not None) is True


def _bare_workspace(args: argparse.Namespace) -> CommandResult:
    """Serve this project's workspace from this terminal, attached to it.

    Nothing opens a browser unless `--start-ui` asked for one, because the command
    line is where this app is worked from and a browser that appears unasked takes
    the reader out of it.

    An app already up for the project is reported rather than started a second
    time, because two apps on one project would each hold the lock and open a
    gateway. Its terminal is not this one, so this call says so and returns the
    reader to their prompt instead of occupying it.
    """

    named = getattr(args, "project", None) or getattr(args, "project_root", None)
    root = _project_path(args) if named else _only_project_or_ask()
    config = resolve_config(
        root, source_directory=configured_source_directory(root), **_config_kwargs(args)
    )
    url = running_url(config)
    if url is not None:
        if getattr(args, "start_ui", False):
            _open_browser(url)
        if _is_detached(config):
            sys.stdout.write(
                f"{config.project_name} is served at {url} by a process with no "
                f"terminal attached, which is a state this app does not serve. "
                f"Ctrl-C here would not stop it. '{CLI_NAME} stop' ends it, and a "
                f"bare call then serves the project from this terminal.\n"
            )
            return CommandResult()
        sys.stdout.write(
            f"{config.project_name} is already served at {url} by a process this "
            f"terminal does not own; Ctrl-C here would not stop it. "
            f"'{CLI_NAME} stop' stops it, and a bare call then serves it here.\n"
        )
        return CommandResult()
    return _serve_attached_sync(config, open_browser=getattr(args, "start_ui", False))


def _serve_attached_sync(
    config: ResearchConfig, *, open_browser: bool
) -> CommandResult:
    """Run the attached app on the running loop of this process."""

    return asyncio.run(_serve_attached(config, port=None, open_browser=open_browser))


async def _start(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:
    """Serve this project from this terminal until the terminal or the app ends.

    Attached is the whole point, so this is the same run the bare call makes: the
    process stays in the foreground and in this terminal's process group, where
    Ctrl-C and a closed window both reach it.

    An app already serving this project is reported rather than started a second
    time, because two apps on one project would each hold the lock and open a
    gateway. Its terminal is not this one, so this call says so and returns the
    reader to their prompt instead of occupying it.
    """

    url = running_url(config)
    if url is not None:
        if args.start_ui:
            _open_browser(url)
        if _is_detached(config):
            sys.stdout.write(
                f"{config.project_name} is served at {url} by a process with no "
                f"terminal attached, which is a state this app does not serve. "
                f"Ctrl-C here would not stop it. '{CLI_NAME} stop' ends it, and "
                f"this command then serves the project here.\n"
            )
            return CommandResult()
        sys.stdout.write(
            f"{config.project_name} is already served at {url} by a process this "
            f"terminal does not own; Ctrl-C here would not stop it. "
            f"'{CLI_NAME} stop' stops it, and this command then serves it here.\n"
        )
        return CommandResult()
    return await _serve_attached(config, port=args.port, open_browser=args.start_ui)


def _open_browser(url: str) -> None:

    import webbrowser

    # A machine with no desktop session has no browser to open, and a browser
    # that fails to start is not a reason to fail the command. The URL is the
    # answer either way.
    with contextlib.suppress(Exception):
        if webbrowser.open(url):
            return
    print(f"Open {url} in a browser.", file=sys.stderr)


def _clients(config: ResearchConfig) -> dict[str, Any]:

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

    with _control(config) as control:
        if control is None:
            raise ResearchError(
                f"No app is running for {config.project_root}, so no client is "
                f"attached. Start it with '{CLI_NAME} start'."
            )
        return control.disconnect(
            args.session_id, args.reason or "Disconnected by request."
        )


def _install(args: argparse.Namespace) -> dict[str, Any]:
    """Make this installation reachable, or take back what it made reachable.

    One command with two selectors rather than two commands: an uninstall has to
    know which of the two the reader meant, and `--desktop` beside `--uninstall`
    answers that.
    """

    from ..installation import (
        install_console_entry,
        install_desktop_entry,
        uninstall_console_entry,
        uninstall_desktop_entry,
    )

    if args.desktop:
        # The entry belongs to this installation rather than to a project, so no
        # project is resolved: resolving one writes its portable state, and adding
        # a menu entry must touch nothing a project owns.
        if args.uninstall:
            report = uninstall_desktop_entry()
        else:
            report = install_desktop_entry(force=args.force)
        return {"command": "install", "desktop": report}
    if args.uninstall:
        report = uninstall_console_entry()
    else:
        report = install_console_entry(force=args.force)
    return {"command": "install", "console": report.as_dict()}


async def _update(args: argparse.Namespace) -> dict[str, Any]:
    """Report what a newer version means, and with ``--apply`` put it in place.

    The check is the default and writes nothing. The comparison is against the
    latest published release, so the version in the answer and the version
    `research-rag --version` prints are the same number. Applying refuses while a
    project lock is held, refuses a checkout with uncommitted work, stops every app
    through that project's own recorded pid, and prints the command that starts
    each one again.
    """

    from .. import release as release_module
    from .. import update as update_module
    from ..registry import load as load_registered
    from ..version import version_block

    offline = bool(args.offline)
    runner = update_module.subprocess_runner
    local = update_module.probe_local(run=runner)
    remote = update_module.probe_remote(local, runner, offline=offline)
    plan = update_module.plan_update(local, remote)
    payload: dict[str, Any] = {
        "command": "update",
        "applied": False,
        # The same four numbers `research-rag --version` prints, from the same
        # functions, so an answer cannot carry one version and the flag another.
        "version": version_block(),
        "install": local.as_dict(),
        "remote": remote.as_dict(),
        "plan": plan.as_dict(),
    }
    if not args.apply and local.is_checkout and local.checkout is not None:
        # On the reporting path only: an applied update moves the checkout, and a
        # tag it read a moment before may be one commit from being irrelevant.
        payload["declared_version_problems"] = list(
            release_module.release_consistency(
                local.checkout, local.declared or local.version, runner
            )
        )
    if plan.blocked:
        raise ResearchError(plan.blocked)
    if not args.apply:
        return payload
    if offline:
        raise ResearchError(
            "--offline cannot update: nothing on the remote was asked. Nothing "
            "was changed. Run this again without --offline."
        )
    if not plan.available:
        # Stopping an app is not free, so a run with nothing to apply changes
        # nothing at all, and says why.
        raise ResearchError("Nothing to apply. " + " ".join(plan.notes))
    projects = [
        update_module.ProjectState(entry.project_root, entry.project_name)
        for entry in load_registered()
        if (entry.project_root / ".research-rag").is_dir()
    ]
    held = update_module.held_projects(projects)
    if held:
        raise ResearchError(
            "Nothing was changed. " + " ".join(one.refusal() for one in held)
        )
    payload["projects"] = [project.as_dict() for project in projects]

    def digests() -> dict[str, dict[str, tuple[int, int]]]:
        return {
            str(project.project_root): update_module.portable_state_digest(
                project.project_root
            )
            for project in projects
        }

    def state_sentence(before_run: dict[str, dict[str, tuple[int, int]]]) -> str:
        changes = update_module.state_changes(before_run, digests())
        if not changes:
            return "no project's portable state changed."
        return "the portable state that changed: " + "; ".join(
            f"{root} ({', '.join(names)})" for root, names in sorted(changes.items())
        )

    before = digests()
    stopped = [update_module.stop_app(project, runner) for project in projects]
    payload["stopped"] = [one.as_dict() for one in stopped]
    try:
        payload["steps"] = update_module.apply_plan(
            plan, run=runner, cwd=local.checkout
        )
    except ResearchError as exc:
        raise ResearchError(
            f"{exc} After the failed step, {state_sentence(before)}"
        ) from exc
    applied = update_module.probe_local(run=runner)
    payload["applied"] = True
    payload["install_after"] = applied.as_dict()
    changes = update_module.state_changes(before, digests())
    payload["project_state_changes"] = changes
    payload["project_state_untouched"] = not changes
    payload["start_again"] = [
        project_command(one.project.project_root, "start")
        for one in stopped
        if one.stopped
    ]
    return payload


async def _doctor(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:

    from ..doctor import mcp_entry_block, mcp_url_block, run_doctor

    if args.mcp_entry:
        # Two entries, because the app is reached two ways. The URL entry names
        # the port the app claimed, which exists only once it is up. The stdio
        # entry is complete either way, because the bridge starts the app itself.
        return CommandResult(
            text=mcp_url_block(config) + "\n" + mcp_entry_block(config)
        )
    entry_check = args.check_entry
    # A whole report is built from the same service call the `status` command
    # makes, so the two surfaces cannot disagree about this project. A check of
    # one entry file needs no project state, so it fetches none.
    status: dict[str, Any] = {}
    if entry_check is None:
        async with _operations(config) as operations:
            status = await operations.status()
    result = run_doctor(
        config,
        status,
        running=_service_processes(config.project_root),
        entry=entry_check,
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

    return nullcontext(connect(config))


def _help_menu() -> str:
    """The whole menu: the order of the work, then the fact a first call needs.

    A description is stored as one sentence and wrapped here, so a wording stays
    readable when the column moves.
    """

    width = max(len(name) for _, entries in HELP_GROUPS for name, _ in entries)
    label = 2 + width + 2
    text_width = max(40, min(78, 96) - label)
    lines = [
        f"{CLI_NAME} — a research knowledge base over a project's own PDFs and EPUBs.",
        "",
        "A project has one app. This command line, the browser workspace, and an",
        "agent's tools are three ways into that one process, so they read one index",
        "and one set of review decisions and cannot disagree.",
        "",
    ]
    for title, entries in HELP_GROUPS:
        lines.append(title)
        for name, description in entries:
            body = textwrap.wrap(description, width=text_width) or [""]
            lines.append(f"  {name.ljust(width)}  {body[0]}")
            lines.extend(f"{' ' * label}{extra}" for extra in body[1:])
        lines.append("")
    lines.append(
        "Each command takes --project-root DIR or --project NAME, except `mcp`,\n"
        "which takes --project-name NAME so a client entry carries no path.\n"
        "`install`, `update`, `help`, and `--version` need no project at all.\n"
        "No command at all opens the workspace in a browser and serves it from\n"
        "this terminal, so Ctrl-C or closing the terminal stops it. It asks\n"
        "which project when this installation holds more than one.\n"
        "One command's own options: `{name} COMMAND --help`.\n"
        "A subject: `{name} help {topics}`.".format(
            name=CLI_NAME, topics=", ".join(sorted(HELP_TOPICS))
        )
    )
    return "\n".join(lines) + "\n"


def _command_usage(parser: argparse.ArgumentParser, name: str) -> str:
    """The usage block one registered subcommand prints, reached by name.

    `help search` and `search --help` are the same document, so this returns the
    subparser's own output rather than a second description that could drift.
    """

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            return action.choices[name].format_help()
    raise KeyError(name)


def _bridge_project_name(args: argparse.Namespace) -> str:
    """Return the project this MCP entry names, and refuse the ones it may not.

    A client entry is copied between machines, and a path in one is true on the
    machine it was written on. The name is resolved here, by this installation,
    so the entry itself names a project and nothing about where it lives.
    """

    for option, value in (
        ("--project-root", getattr(args, "project_root", None)),
        ("--project", getattr(args, "project", None)),
    ):
        if value:
            raise ConfigurationError(
                f"The MCP entry names a project, not a directory: drop {option} "
                "and pass --project-name NAME, which this installation resolves to "
                "the directory the project was initialised at."
            )
    name = getattr(args, "project_name", None)
    if not name or not name.strip():
        raise ConfigurationError(
            "The MCP entry needs --project-name NAME, or "
            f"{PROJECT_NAME_ENV} in the client's environment, naming a project "
            "this installation initialised."
        )
    return name.strip()


async def _run(args: argparse.Namespace) -> CommandResult:

    if args.command == "projects":
        return CommandResult(payload=await _projects(args))
    if args.command == "init":
        return CommandResult(payload=_init(args))
    # Both of these are about the installation rather than a project, so they run
    # before one is resolved: resolving writes the project's portable state, and
    # a reader asking what is installed has no project yet.
    if args.command == "install":
        return CommandResult(payload=_install(args))
    if args.command == "update":
        return CommandResult(payload=await _update(args))
    config = _resolve(args)
    apply_process_priority(config.nice)
    if args.command == "config":
        print(
            describe_settings(
                SETTINGS, config.settings.as_values(), config.settings_provenance
            )
        )
        print(describe_docs())
        print(describe_costs(config.settings))
        return CommandResult()
    if args.command == "start":
        return await _start(args, config)
    if args.command == "clients":
        return CommandResult(payload=_clients(config))
    if args.command == "disconnect":
        return CommandResult(payload=_disconnect(args, config))
    if args.command == "stop":
        return CommandResult(payload=_stop(args, config))
    if args.command == "doctor":
        return await _doctor(args, config)
    async with _operations(config) as operations:
        payload = await _operate(args, operations)
    return CommandResult(payload=payload)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.version:
        # Before the project is resolved, like `help`: a reader asking what is
        # installed has no project yet, and the answer must not fail on a
        # directory that happens to hold one.
        from ..version import version_lines

        sys.stdout.write("\n".join(version_lines()) + "\n")
        return
    if args.command is None:
        # No command is the one thing a reader types without thinking, so it opens
        # the workspace rather than printing a menu: the workspace is what every
        # other command in this app exists to reach, and the app it opens stays in
        # this terminal, so closing the terminal closes it. Before the loop
        # starts, because this call owns one.
        try:
            _bare_workspace(args)
        except (
            ConfigurationError,
            ControlError,
            ResearchError,
            OSError,
            ValueError,
        ) as exc:
            raise SystemExit(f"{CLI_NAME}: {exc}") from exc
        return
    if args.command == "help":
        # Before the project is resolved: the menu is what a reader has
        # precisely when they have no project yet, and it must not fail on a
        # directory that holds one.
        if args.topic is None:
            sys.stdout.write(_help_menu())
        elif args.topic in HELP_TOPICS:
            sys.stdout.write(HELP_TOPICS[args.topic])
        else:
            sys.stdout.write(_command_usage(parser, args.topic))
        return
    if args.command == "mcp":
        # The bridge owns stdio and its own event loop, so it runs before this
        # process starts one. It is handed a project's name, because a client
        # entry that named a directory could not be copied to another machine.
        try:
            project_name = _bridge_project_name(args)
        except (ConfigurationError, ResearchError) as exc:
            raise SystemExit(f"{CLI_NAME}: {exc}") from exc
        bridge.main(project_name, settings=_config_kwargs(args), name=args.client_name)
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
