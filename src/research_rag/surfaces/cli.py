"""The one command line: one research project, in a terminal or a browser.

`research-rag` resolves a project, constructs a `ResearchService`, and calls it in
this process. `ui` hands the browser workspace to the project's own generated
launcher, and `serve` is the foreground workspace host that launcher runs. There
is no second console script and no MCP surface, so the terminal and the browser
answer from the same call on the same payload.

Only a command that queries opens the vanilla gateway, and it opens on the first
call rather than as a command-line choice: the BM25 index is initialized through
that gateway when a generation loads for querying. Commands that only read local
state never start one, so they work with no UltraRAG runtime installed.
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
import textwrap
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from config_ultra_rag_mcp import describe_settings

from .. import bridge
from .. import launcher as launcher_module
from ..app import UI_HOST, App, running_url
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
from ..registry import load as load_registry
from ..registry import register as register_project
from ..registry import registered_at_label, registry_path
from ..registry import resolve as resolve_registered
from ..rerankers import RERANKER_MODEL_CHOICES
from ..service import ResearchService
from ..settings import SETTINGS
from ..settings_document import describe_costs
from ..support import DEFAULT_RETRIEVAL_METHOD, RETRIEVAL_METHODS, ResearchError
from ..tool_views import lean_status
from ..ultrarag import LazyGateway, VanillaUltraRAG

CLI_NAME = CLI_COMMAND
DEFAULT_DEPTH = 10
# The variable a client entry that cannot pass an argument sets instead. It is
# the counterpart of `RESEARCH_ULTRARAG_PROJECT_ROOT`, and it exists for the same
# reason: a client that offers only an environment block still has to name a
# project.
PROJECT_NAME_ENV = "RESEARCH_ULTRARAG_PROJECT_NAME"
# What a project's own .gitignore keeps out of version control: the derived
# state that can be rebuilt, and the machine-local launcher. The descriptor,
# catalogs, and review files are small, portable, and worth keeping.
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
                    "Create the project identity and .research-rag, or attach to an "
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
            (
                "ui",
                "Bring the app up and open the workspace. A first run ends here.",
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
                    "generation. A stale answer names the call that closes it. "
                    "--verbose prints the whole payload."
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
                (
                    "Every generation on disk with its size, and the one search reads. "
                    "With --use, point the project at a retained one."
                ),
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
                (
                    "One passage and its neighbours, for reading around a result. "
                    "Takes the chunk id a search returned."
                ),
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
                (
                    "Take a source, or one passage with --chunk, out of retrieval. "
                    "The file stays."
                ),
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
            (
                "serve",
                "The app in the foreground on one port. The launcher runs this.",
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
                    "--project-name and proxied to that project's app. A client "
                    "entry carries no path, so it works on every machine holding "
                    "the project. `help agents` has the entry."
                ),
            ),
            (
                "config",
                (
                    "Every effective setting and the layer it came from. It prints and "
                    "does not write. `help settings` names the layers and what a "
                    "change costs."
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
                    "Put this command on the account's PATH, and with --desktop give "
                    "one project a menu entry. --uninstall removes what it wrote."
                ),
            ),
            (
                "update",
                (
                    "What a newer version would change, and with --apply the update "
                    "itself: stop every app, pull or upgrade, and print the command "
                    "that starts each one again."
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
switches that reach one source at a time. A filter given nothing is not applied.

  --category C        keep passages whose source carries any of these categories
  --project P         keep passages whose source carries any of these project tags
  --keyword K         keep passages carrying every one of these keywords
  --language L        keep passages from a source in any of these ISO 639 codes
  --author A          keep passages from a source with one of these names
  --title T           keep passages whose source title contains one of these phrases
  --source-id ID      search only these sources
  --exclude-source-id ID
                      search everything except these sources

The category, project, keyword, language, author, and title layers read the
source's reviewed metadata, not the extracted text, so a filter matching nothing
reports nothing rather than a fallback. Set the metadata with `metadata` and the
filters start working.

Two switches decide how retrieval runs. They are the app's settings, not
a per-search choice:

  --method M         bm25, dense, or hybrid; hybrid and reranking are the default
  --no-rerank        skip the cross-encoder, which is on by default and is most of
                     what a search costs

The agent surface has no such switches, and no method, depth, or reranking
argument at all: it is always hybrid, always reranked, and its depth is a
setting. A method chosen per call is a number a reader cannot reproduce from the
answer they were given.
""",
    "settings": """\
A project's settings resolve in four layers, each overriding the one above it,
and `config` prints every effective value with the layer it came from:

  1. the packaged defaults in default.toml
  2. the per-user file, ~/.config/research-ultra-rag-mcp/config.toml
  3. the project file, <project>/.research-rag/config.toml
  4. an extra file named by --config PATH

`config` prints and does not write. The browser workspace writes the project file
for you, refusing a change it cannot validate and naming what the change costs.
The two global options below are per-call and are not recorded anywhere.

  --set key=value    override one value for this command, repeatable, and
                     forgotten when the command ends
  --config PATH      add one more layer for this command

What a change costs is computed, not declared. `config` prints it: each key is
costed by applying a change to it and recomputing what a build records, the
retrieval-policy fingerprint, the recorded chunk settings, and the recorded
models. A key that moves none of them costs nothing and the next search uses the
new value at once, which is what `runtime.tool_detail` and the batch sizes do. A
key that moves the fingerprint or the chunk settings makes search answer `stale`
until `ingest` runs again, which changes the answer to a question already asked.
A key that changes a model downloads it: a new embedding model recomputes every
vector, and a new reranker is loaded by the next search.

`doctor` reports which layer a value came from when a setting does not do what
the project expected, and the file path it would be changed in.
""",
    "agents": """\
The app serves MCP at /mcp on its own port, so a client that can open a socket
needs only the URL. A client that speaks only stdio uses `research-rag mcp
--project-name NAME`, which makes sure that project's app is up and then proxies
to it.

The entry names a project, never a directory, so one entry written on one machine
works on every machine where that project was initialised. The directory is a
fact of each machine: the app resolves the name through this installation's own
project record. `--project-root` and `--project` are refused here, and so is
`RESEARCH_ULTRARAG_PROJECT_ROOT`, because an entry carrying a path in any form
would break the moment it was copied anywhere.

A name no project on this machine answers with a connection and one tool:
`status` reports that the project is not initialised and gives the `init` command
that creates it. Run that command and the same entry serves the project, with no
edit to the entry.

Let the project print the entry for the machine it runs on: the port is chosen at
start, and a hard-coded URL goes stale the first time it moves:

  research-rag --project-root DIR doctor --mcp-entry
  research-rag --project-root DIR doctor --check-entry <file>

To check an entry already in place without editing it. Two ready-to-copy
templates ship with the source: mcp_settings.example.json for a client using an
mcpServers object, and kilo-mcp.example.jsonc for one using a Kilo-style mcp
object. Replace the executable path and the project name in either and the entry
is complete.

Set RESEARCH_ULTRARAG_CLIENT_NAME so the app's client list can tell agents
apart, or RESEARCH_ULTRARAG_PROJECT_NAME when the client can pass an environment
but not an argument. `clients` lists the agents and `disconnect` ends one; the
workspace shows the same list, so an agent ended in the browser is gone from the
terminal too.

An agent gets eight tools and one resource, and every answer is the lean
projection: a question at a time, no inventory, no scores. `status` is a verdict
that names the call closing a gap rather than printing the whole corpus state.
The full payload is `status --verbose` here and the workspace there.

A generated sentence citing a passage is not for direct quotation. Take the
quotation from the original file, which the workspace opens beside the passage
and the agent reaches with get_passage.
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
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT"),
        help="Project root holding .research-rag (default: the current directory).",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT"),
        help=(
            "Name or id of a project this installation registered, in place of "
            "--project-root. `research-rag projects` lists them."
        ),
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
            "from a rebuild, so the second name is the check that this is the one "
            "you meant."
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
            help=(
                "The chunk id of one passage, instead of a whole source. The id a "
                "search returned."
            ),
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
            "Serve the agent surface on stdio for the project named by "
            "--project-name, proxied to that project's app."
        ),
    )
    bridge_command.add_argument(
        "--project-name",
        default=os.environ.get(PROJECT_NAME_ENV),
        help=(
            "Name this project was initialised under, which the app on this "
            "machine resolves to a directory. A client entry carries this name "
            "and no path, so it works on every machine holding the project."
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

    setup = commands.add_parser(
        "install",
        help=(
            "Put this command on the account's PATH, and --desktop give one "
            "project a desktop menu entry."
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
            "Write one freedesktop entry for the named project instead of the "
            "command on PATH. Takes --project-root or --project."
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
        help="Report what a newer version would change, and --apply put it in place.",
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
    there, and nothing here touches a file the user wrote.
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


def _stop(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Stop this project's browser view, and with ``--servers`` every process serving it.

    The launcher's own message is captured rather than printed, so a caller gets
    one JSON report on stdout.
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

    A project whose app is not running is reported as such rather than started,
    because a listing is a question about what exists.
    """

    entries: list[dict[str, Any]] = []
    for project in load_registry():
        entry: dict[str, Any] = {
            "project_name": project.project_name,
            "project_id": project.project_id,
            "project_root": str(project.project_root),
            "registered_at": registered_at_label(project),
            "root_exists": (project.project_root / ".research-rag").is_dir(),
            "app": {"running": False, "url": None, "port": None},
            "attached_clients": 0,
        }
        if entry["root_exists"]:
            try:
                entry.update(_project_app_state(project.project_root))
            except (ConfigurationError, ResearchError) as exc:
                entry["error"] = str(exc)
        entries.append(entry)
    answer: dict[str, Any] = {
        "registry_path": str(registry_path()),
        "project_count": len(entries),
        "projects": entries,
    }
    if not entries:
        answer["message"] = (
            "No project is registered yet. Run 'research-rag --project-root "
            "<path> init' once per project, and this install can address each one "
            "by name."
        )
    return answer


def _project_app_state(project_root: Path) -> dict[str, Any]:

    config = resolve_config(project_root)
    url = running_url(config)
    state: dict[str, Any] = {
        "app": {"running": url is not None, "url": url, "port": _port_of(url)},
        "attached_clients": 0,
    }
    if url is None:
        return state
    try:
        with Control(url, timeout=10.0) as handle:
            verdict = lean_status(handle.status())
            clients = handle.clients()
    except ControlError as exc:
        state["error"] = str(exc)
        return state
    state["attached_clients"] = len(clients)
    for key in ("ready", "stale", "requires"):
        if key in verdict:
            state[key] = verdict[key]
    return state


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    _, _, port = url.rpartition(":")
    return int(port) if port.isdigit() else None


@dataclass(frozen=True, slots=True)
class CommandResult:
    payload: dict[str, Any] | None = None
    exit_code: int = 0
    text: str | None = None


async def _serve(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:
    """Serve the app in the foreground until this process is stopped.

    The generated launcher runs this, so it takes an explicit port and never
    chooses one: the launcher already made that choice while holding the lock
    that makes it exclusive. A caller running it by hand omits `--port` and gets
    the first free port at or above the default.
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
    answers that without a second pair of flags to remember.
    """

    from ..installation import (
        install_console_entry,
        install_desktop_entry,
        uninstall_console_entry,
        uninstall_desktop_entry,
    )

    if args.desktop:
        # Read through the project directory rather than a resolved
        # configuration: resolving one writes the project's portable state, and
        # installing a menu entry must touch nothing the project owns.
        project_root = _project_path(args)
        if args.uninstall:
            report = uninstall_desktop_entry(project_root=project_root)
        else:
            report = install_desktop_entry(project_root=project_root, force=args.force)
        return {"command": "install", "desktop": report}
    if args.uninstall:
        report = uninstall_console_entry()
    else:
        report = install_console_entry(force=args.force)
    return {"command": "install", "console": report.as_dict()}


async def _update(args: argparse.Namespace) -> dict[str, Any]:
    """Report what a newer version means, and with ``--apply`` put it in place.

    The check is the default and writes nothing. Applying refuses while a
    project lock is held, stops every app this installation serves through that
    project's own launcher, and prints the command that starts each one again
    rather than starting it.
    """

    from .. import update as update_module
    from ..registry import load as load_registered

    offline = bool(args.offline)
    runner = update_module.subprocess_runner
    local = update_module.probe_local(run=runner)
    remote = update_module.probe_remote(local, runner, offline=offline)
    plan = update_module.plan_update(local, remote)
    payload: dict[str, Any] = {
        "command": "update",
        "applied": False,
        "install": local.as_dict(),
        "remote": remote.as_dict(),
        "plan": plan.as_dict(),
    }
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
        project_command(one.project.project_root, "ui")
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
        f"Each command takes --project-root DIR or --project NAME, except `mcp`,\n"
        f"which takes --project-name NAME so a client entry carries no path.\n"
        f"`install` needs no project unless --desktop is given, and `update` and\n"
        f"`help` need none.\n"
        f"One command's own options: `{CLI_NAME} COMMAND --help`.\n"
        f"A subject: `{CLI_NAME} help {', '.join(sorted(HELP_TOPICS))}`."
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
                "and pass --project-name NAME. The app on this machine resolves "
                "the name to the directory the project was initialised at, which "
                "is what lets one entry work on every machine holding it."
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
        print(describe_costs(config.settings))
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
    parser = _parser()
    args = parser.parse_args(argv)
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
