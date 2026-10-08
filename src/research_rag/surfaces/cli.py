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
import math
import os
import shutil
import signal
import sys
import textwrap
import time
import unicodedata
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    # The service is reached when a command needs a corpus operation. A command
    # that only reports -- `--version`, `help`, `config`, `doctor`, `install`, and
    # `update` -- must not import the retrieval stack to get there, which is the
    # rule `AGENTS.md` states.
    from ..core.service import ResearchService
    from ..runtime.release import PublishedRelease

from ..project.config import (
    CLI_COMMAND,
    DEFAULT_UI_PORT,
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    project_command,
    resolve_config,
)
from ..project.policy import RETRIEVAL_METHODS, ResearchError
from ..project.registry import (
    account_projects,
    detached_from_terminal,
    project_app_state,
    registry_path,
)
from ..project.registry import register as register_project
from ..project.registry import resolve as resolve_registered
from ..project.settings import SETTINGS
from ..project.settings_document import describe_costs, describe_docs
from ..project.settings_layers import describe_settings
from ..project.state_files import process_alive as alive
from ..retrieval.rerankers import RERANKER_MODEL_CHOICES
from ..runtime import ownership
from ..runtime import process as process_module
from ..runtime.app import (
    UI_HOST,
    App,
    _claim_loopback_port,
    _own_tty,
    has_terminal,
    recorded_pid,
    running_url,
)
from ..runtime.control import Control, ControlError, connect

CLI_NAME = CLI_COMMAND
DEFAULT_DEPTH = 10
# The variable a client entry that cannot pass an argument sets instead. It is
# the counterpart of `RESEARCH_RAG_PROJECT_ROOT`, and it exists for the same
# reason: a client that offers only an environment block still has to name a
# project.
PROJECT_NAME_ENV = "RESEARCH_RAG_PROJECT_NAME"
# The three variables the global options read, named once because `--help` states
# where each default comes from and a literal typed twice is one typo away from
# being wrong in one place only.
PROJECT_ROOT_ENV = "RESEARCH_RAG_PROJECT_ROOT"
PROJECT_ENV = "RESEARCH_RAG_PROJECT"
CONFIG_ENV = "RESEARCH_RAG_CONFIG"
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
                (
                    "Every generation with its size, recorded models and build configuration. "
                    "--use ID loads one for subsequent searches."
                ),
            ),
            (
                "stats",
                (
                    "Which sources and passages searches return at rank one and in "
                    "the top five, and what the corpus holds."
                ),
            ),
            (
                "history",
                "Recent searches that kept their question, to run again. --clear forgets them.",
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
            (
                "chunks",
                "One source's passages in reading order, each with how often searches returned it.",
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
                "Bring the app up and report where it is. --start-ui opens a browser.",
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
                    "--project-name and proxied to that project's app. It starts "
                    "no app of its own. `help agents` has the client entry."
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
                    "It previews the changelog that release was published with and "
                    "installs only on yes, with no as the answer; --yes is for an "
                    "unattended run. The branch head is reported beside the "
                    "release answer."
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

A filter is applied after the corpus is ranked, so an empty answer with a filter
on means nothing the ranking reached matches it, not that no source carries it.
A filter naming one source can therefore find nothing when that source ranks
below the window the answer reports.
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
--project-name NAME`, which proxies to that project's app while it is up and
starts nothing itself: an app belongs to a terminal, so an agent that arrives
when none is serving its project is told which command to run.

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

# `--help` lists the commands flat and `help` lists them grouped, so the same
# command would be described twice and the two could disagree. The menu is the
# canonical text and this takes its first sentence for the flat list, so a
# command added to the parser must reach the menu and nothing else. `help` is
# absent from the menu because a group cannot list the menu it is the contents
# of, so its one line is written here.
_HELP_SUMMARIES: dict[str, str] = {
    name: description.partition(". ")[0].rstrip(".") + "."
    for _, entries in HELP_GROUPS
    for name, description in entries
}
_HELP_SUMMARIES["help"] = (
    "Print the menu, or one page of it: a subject, or a command's own usage."
)

# The lines `--help` shows, in the order a first call needs them. Every one keeps
# a global option in front of the command name, because an option after it is
# read as that command's own, and every path and query is quoted because a
# project directory and a research question both carry spaces. Each line here is
# parsed by `test_the_examples_in_the_help_parse_as_they_are_written`, so an
# example cannot drift from the parser it documents.
HELP_EXAMPLES = """\
A first project, in four commands:

  research-rag --project-root "/path/to/My Project" init --name "My project"
  research-rag --project-root "/path/to/My Project" ingest
  research-rag --project-root "/path/to/My Project" search "what does it say?"
  research-rag --project-root "/path/to/My Project" --start-ui start

Copy the PDFs and EPUBs into "/path/to/My Project/sources" before `ingest`, and
keep the terminal open while `start` is serving it.

A project this installation already registered, named instead of its path:

  research-rag --project "My project" search "what does it say?"

The whole menu, one subject, and one command's own options:

  research-rag help
  research-rag help {topic}
  research-rag search --help

Every global option goes before the command name, and `--project` and
`--project-root` both name the project, so passing one is refused.
"""


@asynccontextmanager
async def _service(config: ResearchConfig) -> AsyncIterator[ResearchService]:
    """Yield the service over a gateway that opens only if it is spoken to."""

    from ..core.service import ResearchService
    from ..retrieval.ultrarag import LazyGateway, VanillaUltraRAG

    gateway = LazyGateway(config)
    try:
        yield ResearchService(config, VanillaUltraRAG(gateway, config))
    finally:
        await gateway.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        # Raw text, because the orientation and the examples below are written as
        # paragraphs and as a list, and the wrapping formatter would fold both into
        # one paragraph. Argument help is still wrapped: only a description and an
        # epilog are read raw.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Review evidence from a project's PDF and EPUB corpus, in a terminal, "
            "in a browser, or through an agent.\n"
            "\n"
            "One project is one app, one index, and one set of review decisions. "
            "This command line, the browser workspace, and an agent's tools are "
            "three ways into that one process, so they read one corpus and cannot "
            "disagree about it.\n"
            "\n"
            "`init` gives a directory its project record, `ingest` builds the "
            "searchable generation from the PDFs and EPUBs in its sources "
            "directory, and `search` or `start` reads what was built. An answer "
            "carries the passage, where it sits, and the original to open beside "
            "it; the conclusion is yours."
        ),
        epilog=HELP_EXAMPLES.format(topic=min(HELP_TOPICS)),
    )
    project = parser.add_argument_group(
        "Choosing a project",
        "One project is named once per command, and these are the two ways.",
    )
    project.add_argument(
        # None rather than "." so a call naming a registered project has not also
        # named a path. An absent value means the current directory, resolved
        # where it is used.
        "--project-root",
        dest="project_root",
        metavar="DIR",
        default=os.environ.get(PROJECT_ROOT_ENV),
        help=(
            "Directory holding this project's `.research-rag`, which `init` "
            "creates. Default: the current directory. Refused together with "
            f"`--project`, and its value is read from {PROJECT_ROOT_ENV} in the "
            "environment."
        ),
    )
    project.add_argument(
        "--project",
        metavar="NAME",
        default=os.environ.get(PROJECT_ENV),
        help=(
            "Name or id this installation registered, in place of "
            f"`--project-root DIR`. `{CLI_NAME} projects` lists them. Refused "
            f"together with `--project-root`, and its value is read from "
            f"{PROJECT_ENV} in the environment."
        ),
    )
    project.add_argument(
        "--runtime-root",
        metavar="DIR",
        default=os.environ.get("RESEARCH_RAG_RUNTIME_ROOT"),
        help=(
            "Absolute directory for derived state, when the project itself is on "
            "slow storage. Default: `.research-rag/runtime` inside the project. "
            "Only derived state moves; the corpus and the review decisions stay."
        ),
    )
    serving = parser.add_argument_group("Serving")
    serving.add_argument(
        "--start-ui",
        dest="start_ui",
        action="store_true",
        help=(
            "Open the workspace in a browser as well as serving it. Serving never "
            "opens one by itself, so a browser appears only when this is passed."
        ),
    )
    models = parser.add_argument_group(
        "Models and retrieval",
        "A new generation is built with these. A search reads the generation\n"
        "that is already built.",
    )
    models.add_argument(
        "--model-cache-root",
        metavar="DIR",
        default=None,
        help=(
            "Directory the pinned models are cached in, shared by every project. "
            "Default: the `research-rag` directory in this account's cache, under "
            "`models`."
        ),
    )
    models.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help=(
            "Refuse the network and require the pinned runtime and every model to "
            "be cached already. `doctor --prefetch-models` caches them."
        ),
    )
    models.add_argument(
        "--embedding-threads",
        metavar="COUNT",
        type=int,
        default=None,
        help="ONNX Runtime threads for the embedding model; unset lets the runtime decide.",
    )
    models.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=None,
        help="Dense index backend for a new generation (default: auto).",
    )
    models.add_argument(
        "--reranker-model",
        metavar="MODEL",
        choices=RERANKER_MODEL_CHOICES,
        default=None,
        help=(
            "Cross-encoder the engine loads for a reranked search. Default: the "
            "one this project's settings name. The pinned models are: "
            f"{', '.join(RERANKER_MODEL_CHOICES)}."
        ),
    )
    layers = parser.add_argument_group(
        "Settings for this command",
        "Both are recorded nowhere: they last exactly as long as this command.",
    )
    layers.add_argument(
        "--config",
        metavar="PATH",
        default=os.environ.get(CONFIG_ENV),
        help=(
            "Extra settings file, layered above the per-user and project files. "
            f"Read from {CONFIG_ENV} in the environment. `help settings` names "
            "every layer and what a change costs."
        ),
    )
    layers.add_argument(
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
    installation = parser.add_argument_group("This installation")
    installation.add_argument(
        "--version",
        action="store_true",
        help=(
            "Print this app's version, the installed one, the shared workspace's, "
            "and whether a restart is required. Needs no project."
        ),
    )
    # Not `required=True`, because `--version` is an answer of its own and asks
    # for no command. A call with neither prints the same refusal argparse would.
    commands = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="Commands",
        description=(
            "The menu `research-rag help` prints groups these by the work they do."
        ),
    )

    def add(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Add one subcommand whose `--help` line is its menu entry's first sentence."""

        return commands.add_parser(name, help=_HELP_SUMMARIES[name], **kwargs)

    create = add(
        "init",
        description=(
            "Give a directory its project record. This creates `.research-rag`, "
            "creates the source directory, and records the project in this "
            "installation's register, so `--project NAME` can name it later. It "
            "never overwrites a file it did not write, and running it on an "
            "existing project reports that project again rather than resetting it."
        ),
    )
    create.add_argument(
        "--name",
        metavar="NAME",
        default=None,
        help=(
            "Name to record for this project. An existing project is renamed only "
            "when this is given; the stable project id never changes. An agent's "
            "client entry carries this name, so choose one that will mean the same "
            "thing on every machine."
        ),
    )
    create.add_argument(
        "--sources",
        metavar="DIR",
        default=None,
        help=(
            "Project-relative directory to create for the originals. Default: "
            "`sources`. An existing project keeps the directory it recorded and "
            "refuses a different one."
        ),
    )

    add(
        "projects",
        description=(
            "List every project this installation registered, with each one's "
            "directory and whether its app is up. This needs no project, because "
            "it is the command that finds them."
        ),
    )
    # A command name is a valid topic, so the choices are every registered
    # command beside the subject pages. They come from the menu rather than
    # being typed out, so a command added to the parser and the menu is one
    # change, and the drift test below notices one added to neither.
    help_command = add(
        "help",
        description=(
            "Print the whole menu, one subject page, or one command's own usage. "
            "Every form works before you have a project."
        ),
    )
    help_command.add_argument(
        "topic",
        metavar="TOPIC",
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

    status = add(
        "status",
        description=(
            "Answer whether this project's corpus is ready, is current, and what "
            "changed since the generation search reads. Without `--verbose` this "
            "is the same bounded answer an agent gets."
        ),
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

    refresh = add(
        "ingest",
        description=(
            "Build the searchable generation from the PDFs and EPUBs in the "
            "source directory: extract, chunk, embed, and index them. A build that "
            "fails or is cancelled leaves the generation search reads in place, "
            "and calling this again resumes the checkpoint."
        ),
    )
    refresh.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Rebuild every document, chunk, and vector instead of reusing the "
            "compatible ones, and write a new generation beside them."
        ),
    )

    find = add(
        "search",
        description=(
            "Answer a question with evidence passages from the selected "
            "generation. Every search is hybrid and reranked. A passage carries "
            "the source path, where it sits in the original, and its cleaned "
            "text, which is never a transcript: open the original for a "
            "quotation. `help filters` names the filter layers."
        ),
    )
    find.add_argument(
        "query",
        metavar="QUERY",
        help=(
            "Research question, exact phrase, name, or concept. Quote it when it "
            "carries spaces."
        ),
    )
    find.add_argument(
        "--top-k",
        metavar="COUNT",
        type=int,
        default=DEFAULT_DEPTH,
        help=(
            f"Maximum ranked passages to return, 1 to 50 (default: {DEFAULT_DEPTH}). "
            "A filter is applied after the ranking, so a filtered answer is often "
            "shorter than this."
        ),
    )
    find.add_argument(
        "--method",
        choices=sorted(RETRIEVAL_METHODS),
        default=None,
        help=(
            "Retrieval half to rank with (default: hybrid, or BM25 with an "
            "upgrade warning when the generation's embedding model is incompatible). "
            "This exists to reproduce a retrieval comparison; no reader-facing "
            "surface offers the choice."
        ),
    )
    find.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Skip the cross-encoder, which is on by default and is what a search "
            "costs. Diagnostic, like --method."
        ),
    )
    filters = find.add_argument_group(
        "Filters",
        "Each is repeatable, each reads the reviewed metadata at query time, and\n"
        "none is applied when given nothing. A filter decides what the answer\n"
        "may contain; it does not change the ranking.",
    )
    filters.add_argument(
        "--category",
        metavar="CATEGORY",
        action="append",
        help="Keep results carrying any of these categories.",
    )
    filters.add_argument(
        "--project",
        metavar="TAG",
        action="append",
        help=(
            "Keep results carrying any of these project tags from the reviewed "
            "metadata. This filters the answer; the global --project chose which "
            "project to search."
        ),
    )
    filters.add_argument(
        "--keyword",
        metavar="KEYWORD",
        action="append",
        help="Keep results carrying every one of these keywords.",
    )
    filters.add_argument(
        "--language",
        metavar="CODE",
        action="append",
        help="Keep results written in any of these ISO 639 codes.",
    )
    filters.add_argument(
        "--author",
        metavar="NAME",
        action="append",
        help=(
            "Keep results whose source has one of these names among its authors, "
            "matched as a case-insensitive substring."
        ),
    )
    filters.add_argument(
        "--title",
        metavar="PHRASE",
        action="append",
        help=(
            "Keep results whose source title contains one of these phrases, "
            "matched as a case-insensitive substring."
        ),
    )
    filters.add_argument(
        "--source-id",
        metavar="SOURCE_ID",
        action="append",
        help=(
            "Search only these stable source ids, as `sources` reports them. "
            "Repeatable."
        ),
    )
    filters.add_argument(
        "--exclude-source-id",
        metavar="SOURCE_ID",
        action="append",
        help="Search everything except these stable source ids. Repeatable.",
    )

    add(
        "sources",
        description=(
            "List this project's source files: which are indexed, which are not "
            "indexed and why, what each one's reviewed metadata says, and what has "
            "been decided about it. This is the inventory the workspace's Sources "
            "view reads, and it is not on the agent surface."
        ),
    )

    stats = add(
        "stats",
        description=(
            "Report how often each source and passage reached rank one and the top "
            "five of a search on this machine, how many searches returned nothing, "
            "how long they took, which searchable sources no search has reached, "
            "and the corpus composition and last build the selected generation "
            "records. Deleting the counts file named in the answer resets them."
        ),
    )
    stats.add_argument(
        "--days",
        type=float,
        metavar="N",
        help="Count only the searches of the last N days (default: all of them).",
    )
    stats.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="How many sources and passages each ranked list names (1-100).",
    )
    stats.add_argument(
        "--largest-by",
        choices=("passages", "size"),
        default="passages",
        help="What the largest sources are ranked by (default passages).",
    )

    history = add(
        "history",
        description=(
            "List recent searches that kept their question and filters, newest "
            "first, with the counts of what each returned. Nothing is kept while "
            "`runtime.search_history` is false. `--clear` forgets every kept "
            "question and leaves the counts."
        ),
    )
    history.add_argument(
        "--limit", type=int, default=20, metavar="N", help="How many to list (1-200)."
    )
    history.add_argument(
        "--days", type=float, metavar="N", help="Only the last N days."
    )
    history.add_argument(
        "--clear",
        action="store_true",
        help="Forget every kept question and filter. The counts stay.",
    )

    chunks = add(
        "chunks",
        description=(
            "List one source's passages in reading order: where each sits, its "
            "cleaned text trimmed, its size in tokens, whether it is excluded, and "
            "how many times a search returned it in the top five and at rank one. "
            "The source is its stable id (`src_...`) or its path as `sources` "
            "reports it."
        ),
    )
    chunks.add_argument(
        "source",
        metavar="SOURCE",
        help="The source's stable id (`src_...`) or its path as `sources` reports it.",
    )
    chunks.add_argument(
        "--page",
        type=int,
        default=1,
        metavar="N",
        help="Which page of passages (default 1).",
    )
    chunks.add_argument(
        "--page-size",
        type=int,
        default=20,
        metavar="N",
        help="Passages per page (1-50).",
    )

    generations = add(
        "generations",
        description=(
            "List the generations on disk with their size, recorded models and "
            "build configuration, and which one search reads. A rebuild writes "
            "a new one and switches to it only when every "
            "index is complete, so the earlier ones stay here to be searched "
            "instead or to be removed."
        ),
    )
    generations.add_argument(
        "--use",
        dest="use_generation",
        metavar="GENERATION_ID",
        help=(
            "Point the project at this retained generation, after validating its "
            "artifacts and both indexes. Search reads it from the next call. One "
            "id is enough here; `remove-generation` asks for it twice."
        ),
    )

    remove = add(
        "remove-generation",
        description=(
            "Delete a retained generation that search is not reading. Removal is "
            "permanent: the space comes back only from a rebuild. A generation "
            "search reads, and one pending activation, are both refused."
        ),
    )
    remove.add_argument(
        "generation_id",
        metavar="GENERATION_ID",
        help="The directory name to remove, as `generations` reports it.",
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

    context = add(
        "passage",
        description=(
            "Read one passage and the passages around it, from the selected "
            "generation. Use it to read around a search hit, and to reach a "
            "neighbouring passage's own text before asking about it."
        ),
    )
    context.add_argument(
        "chunk_id",
        metavar="CHUNK_ID",
        help=(
            "Exact chunk id from a search result. A chunk id belongs to the "
            "generation that returned it, so a rebuild can replace it."
        ),
    )
    context.add_argument(
        "--context-chunks",
        metavar="COUNT",
        type=int,
        default=1,
        help=(
            "Neighbours to include on each side, 0 to 5 (default: 1). 0 returns "
            "the passage alone."
        ),
    )

    for verb, reason_required in (("include", False), ("exclude", True)):
        # Two sentences shared by both verbs, so the difference is the decision
        # and nothing else: `include` puts a decision back, `exclude` records it.
        shared = (
            "The file is never edited: the decision is recorded beside it, every "
            "surface applies it at once, and it holds for the generation on "
            "screen and for every one built after it."
        )
        change = add(
            verb,
            description=(
                "Take one source, or one passage, out of retrieval."
                if verb == "exclude"
                else "Put an excluded source or passage back into retrieval."
            )
            + f" {shared}",
        )
        change.add_argument(
            "source",
            metavar="SOURCE",
            nargs="?",
            help=(
                "Source path relative to the source directory, exactly as "
                "`sources` reports it. Quote it when it carries spaces. Omit it "
                "to name the source with --source-id."
            ),
        )
        change.add_argument(
            "--source-id",
            metavar="SOURCE_ID",
            help=(
                "The stable source id, instead of a path. It survives replacing "
                "the file's bytes, where a path changes when the file is renamed."
            ),
        )
        change.add_argument(
            "--chunk",
            metavar="CHUNK_ID",
            help=(
                "The chunk id of one passage, instead of a whole source. It "
                "cannot be combined with a path or --source-id."
            ),
        )
        change.add_argument(
            "--reason",
            metavar="TEXT",
            required=reason_required,
            help=(
                "Why the decision was made. Required to exclude, optional to "
                "include, and recorded in the file that holds the decision."
                if reason_required
                else "Why the decision was made, when it is worth recording. Recorded "
                "in the file that holds the decision."
            ),
        )

    review = add(
        "metadata",
        description=(
            "Record the bibliography a person checked for one source. The review "
            "applies at the next read, so no ingestion is needed, and it survives "
            "a rebuild.\n"
            "\n"
            "The saved review replaces this source's whole entry, so name every "
            "field you want kept: a field left out stops being reviewed and the "
            "automatic value applies again. An empty string, or an empty repeated "
            "list, drops that field on purpose."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    review.add_argument(
        "source",
        metavar="SOURCE",
        nargs="?",
        help=(
            "Source path relative to the source directory, exactly as `sources` "
            "reports it. Quote it when it carries spaces. Omit it to name the "
            "source with --source-id."
        ),
    )
    review.add_argument(
        "--source-id",
        metavar="SOURCE_ID",
        help=(
            "The stable source id, instead of a path. It survives replacing the "
            "file's bytes, where a path changes when the file is renamed."
        ),
    )
    reviewed = review.add_argument_group("Reviewed fields")
    reviewed.add_argument(
        "--title", metavar="TEXT", help="Reviewed title. An empty string drops it."
    )
    reviewed.add_argument(
        "--author",
        metavar="NAME",
        action="append",
        help=(
            "An author, in the order to list them. Repeatable; blanks and repeats "
            "are dropped, so an empty one drops the whole list."
        ),
    )
    reviewed.add_argument(
        "--year",
        metavar="YEAR",
        type=int,
        help="Reviewed year of publication, 1 to 9999.",
    )
    reviewed.add_argument(
        "--doi", metavar="TEXT", help="Reviewed DOI. An empty string drops it."
    )
    reviewed.add_argument(
        "--category",
        metavar="CATEGORY",
        action="append",
        help=(
            "A category the work belongs to, and a layer `--category` filters on. "
            "Repeatable."
        ),
    )
    reviewed.add_argument(
        "--keyword",
        metavar="KEYWORD",
        action="append",
        help=(
            "A keyword the work carries, and a layer `--keyword` filters on. "
            "Repeatable."
        ),
    )
    reviewed.add_argument(
        "--language",
        metavar="CODE",
        action="append",
        help=(
            "A language the work is written in, as a lowercase ISO 639 code, and "
            "a layer `--language` filters on. Repeatable."
        ),
    )
    reviewed.add_argument(
        "--project",
        metavar="TAG",
        help="The project tag to file the work under, and a layer --project filters on.",
    )
    reviewed.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Remove the review instead, so automatic metadata applies again. It "
            "cannot be combined with a field."
        ),
    )

    add(
        "config",
        description=(
            "Print every effective setting with the layer it came from, what the "
            "key does, and what a change to it costs. This never writes: the "
            "browser workspace writes this project's own config.toml, and a hand "
            "edit there is read as it is. `help settings` names the layers."
        ),
    )

    examine = add(
        "doctor",
        description=(
            "Report what is wrong with this installation, one line per "
            "dependency, each with the command that fixes it. A default run "
            "reads and writes nothing, so it is always safe to run."
        ),
    )
    examine.add_argument(
        "--mcp-entry",
        action="store_true",
        help=(
            "Print the MCP client entries for the resolved configuration: the "
            "URL entry for a client that can open a socket, and the stdio entry "
            "for one that cannot. Print it from the machine it will run on, "
            "because the port is chosen at start."
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
        help=(
            "Download the pinned embedding and reranker models into the cache. "
            "The only other thing that reaches the network is --repair-runtime."
        ),
    )
    examine.add_argument(
        "--repair-runtime",
        action="store_true",
        help=(
            "Move a mismatched UltraRAG runtime aside, install the pinned one, "
            "and validate it. Evidence it discards is moved aside, not deleted."
        ),
    )

    browser = add(
        "start",
        description=(
            "Serve this project's app from this terminal and report the address. "
            "Ctrl-C, a closed window, and `kill` all end it, so the terminal is "
            "where the app lives: this never leaves one running in the "
            "background, and it never opens a browser unless --start-ui was "
            "passed. An app already up for this project is reported and left "
            "alone rather than started a second time."
        ),
    )
    browser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        help=(
            f"Serve on this port. A port you name is never moved: one that is "
            f"taken fails and says so. Default: the first free port at or above "
            f"{DEFAULT_UI_PORT}."
        ),
    )

    add(
        "clients",
        description=(
            "List the MCP clients attached to this project's app, by the name each "
            "one declared and the program it runs in. One agent is one row however "
            "many sessions it opened. A client that has said nothing for a quarter "
            "of an hour is forgotten, so this lists the clients that are here "
            "rather than every session the machine has run. The workspace's MCP "
            "tab shows the same list and can end one there."
        ),
    )

    drop = add(
        "disconnect",
        description=(
            "End one attached client's session, before its requests reach the "
            "tools. This is the same decision the workspace's MCP tab makes, so "
            "an agent ended in the browser is gone from the terminal too."
        ),
    )
    drop.add_argument(
        "session_id",
        metavar="ID",
        help=(
            "The session id `clients` reports. A connection that has not opened a "
            "session yet carries none and cannot be named here."
        ),
    )
    drop.add_argument(
        "--reason",
        metavar="TEXT",
        default=None,
        help="Why it is being dropped. Reported with the client.",
    )

    bridge_command = add(
        "mcp",
        description=(
            "Serve the agent surface on stdio for one project and proxy it to that "
            "project's app.\n"
            "\n"
            "Keep `mcp` before its own options: an option in front of it is read "
            "as the command name and the server closes the connection instead of "
            "answering. This names a project, never a directory, so one entry "
            "works on every machine where that project was initialised, and "
            "--project-root and --project are refused here. It starts no app: an "
            "app belongs to a terminal, so this either proxies to one or answers "
            "with the command that starts it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bridge_command.add_argument(
        "--project-name",
        metavar="NAME",
        default=os.environ.get(PROJECT_NAME_ENV),
        help=(
            "Name this project was initialised under, which the app on this "
            f"machine resolves to a directory. Read from {PROJECT_NAME_ENV} in "
            "the client's environment when this is omitted. `help agents` has "
            "the client entry."
        ),
    )
    bridge_command.add_argument(
        "--client-name",
        metavar="NAME",
        default=None,
        help=(
            "Name this bridge reports itself under, so an app's client list can "
            "tell agents apart. A client that sets nothing appears as "
            "`stdio-bridge`."
        ),
    )

    stop = add(
        "stop",
        description=(
            "Stop this project's app, from any terminal, and report which terminal "
            "owned it. It never starts one afterwards: an app belongs to the "
            "terminal that started it and ends there."
        ),
    )
    stop.add_argument(
        "--servers",
        action="store_true",
        help=(
            "Also stop every process of this app serving this project, including "
            "a build that is still running. A build interrupted this way resumes "
            "from its checkpoint on the next `ingest`."
        ),
    )

    setup = add(
        "install",
        description=(
            "Make this installation reachable by name, or take back what it made "
            "reachable. One command with two selectors rather than two commands, "
            "because an uninstall has to know which of the two was meant.\n"
            "\n"
            "This writes only under this account's own directories and touches no "
            "project's state. It writes no serving process: the app is started by "
            "`start`, or by clicking the menu entry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup.add_argument(
        "--uninstall",
        action="store_true",
        help=(
            "Remove what `install` wrote for the same selector, and nothing else. "
            "A file it did not write is reported and left alone."
        ),
    )
    setup.add_argument(
        "--desktop",
        action="store_true",
        help=(
            "Write the one freedesktop entry for this installation instead of the "
            "command on PATH. It opens a terminal window and serves whichever "
            "project you choose there, so no project is named in the entry, and "
            "closing the window stops the app it started."
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

    refresh_install = add(
        "update",
        description=(
            "Compare this installation with the latest published release, show the "
            "changelog that release was published with, and say what applying "
            "would do. The comparison is against a release tag, not a branch "
            "head, so a checkout ahead of the latest release reports unreleased "
            "work and changes nothing. In a terminal it asks once whether to "
            "install the release it previewed, and no is the answer; nothing is "
            "installed without that answer. Without a terminal it only reports, "
            "so a script reads the JSON and nothing changes. Needs no project."
        ),
    )
    refresh_install.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the update: refuse while a build holds a project lock or the "
            "checkout has uncommitted work, naming both, then stop every app this "
            "installation serves, move to the release, and print the command that "
            "starts each app again. It does not start them for you. In a "
            "terminal it asks first and installs only on yes; without a terminal "
            "it refuses unless --yes says the approval was given already."
        ),
    )
    refresh_install.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Say that the approval was given already, for an unattended run that "
            "must not be asked. With --apply it installs the release the check "
            "previewed without asking; without --apply it reports and changes "
            "nothing. It is never a yes nobody was asked for: the question is "
            "asked at most once, and any answer other than yes declines."
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
        if not ownership.invokes_this_app(arguments, project_root):
            continue
        if ownership.project_argument(arguments) != str(project_root):
            continue
        found.append((pid, " ".join(arguments)))
    return found


def _terminate(project_root: Path, pids: list[int]) -> list[int]:
    """Ask these processes to stop and then insist; return the ones killed outright.

    Each signal goes through `ownership`, so the sweep only reaches a process it
    has proved is this project's app, and never one it descends from.
    """

    for pid in pids:
        ownership.ask_to_stop(pid, project_root)
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.1)
    forced = [pid for pid in pids if alive(pid)]
    for pid in forced:
        ownership.ask_to_stop(pid, project_root, number=signal.SIGKILL)
    return forced


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
    # Asked, never killed: the app closes its own gateway and releases its own lock
    # on the way out. A pid record names a number, so `ownership` establishes that
    # the process behind it is this project's app before anything is signalled.
    pid = recorded_pid(config)
    report["stopped"] = False
    report["stop_refusal"] = None
    if pid is not None:
        outcome = ownership.ask_to_stop(pid, config.project_root)
        report["stopped"] = outcome.asked
        report["stop_refusal"] = None if outcome.asked else outcome.reason
        report["pid"] = pid
    if not args.servers:
        report["notes"] = _stop_notes(bool(app_state.get("detached")))
        return report
    found = _service_processes(config.project_root)
    report["servers"] = [{"pid": pid, "command": command} for pid, command in found]
    report["forced_pids"] = _terminate(config.project_root, [pid for pid, _ in found])
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

    async def stats(
        self, *, days: float | None, top: int, largest_by: str
    ) -> dict[str, Any]:
        return await self._require().search_stats(
            since_days=days, top=top, largest_by=largest_by
        )

    async def history(self, *, limit: int, days: float | None) -> dict[str, Any]:
        return await self._require().search_history(limit=limit, since_days=days)

    async def clear_history(self) -> dict[str, Any]:
        return await self._require().clear_search_history()

    async def source_chunks(
        self, *, source: str, page: int, page_size: int
    ) -> dict[str, Any]:
        return await self._require().source_chunks(
            source_id=source if source.startswith("src_") else None,
            source_path=None if source.startswith("src_") else source,
            page=page,
            page_size=page_size,
        )

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
        return await asyncio.to_thread(self.control.status)

    async def ingest(self, *, force_recompute: bool) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.control.ingest, force_recompute=force_recompute
        )

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self.control.search(query, **arguments)

    async def sources(self) -> dict[str, Any]:
        return self.control.sources()

    async def stats(
        self, *, days: float | None, top: int, largest_by: str
    ) -> dict[str, Any]:
        return self.control.stats(days=days, top=top, largest_by=largest_by)

    async def history(self, *, limit: int, days: float | None) -> dict[str, Any]:
        return self.control.history(limit=limit, days=days)

    async def clear_history(self) -> dict[str, Any]:
        return self.control.clear_history()

    async def source_chunks(
        self, *, source: str, page: int, page_size: int
    ) -> dict[str, Any]:
        return self.control.source_chunks(source=source, page=page, page_size=page_size)

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
        return self.control.generations()

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return self.control.use_generation(generation_id)

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return self.control.remove_generation(generation_id, confirm=confirm)


INGEST_PROGRESS_INTERVAL = 2.0
INGEST_HEARTBEAT_INTERVAL = 15.0
PASSAGE_ETA_PHASES = frozenset({"embedding", "dense_indexing", "qdrant_indexing"})


class _PhaseETA:
    """Measure current-phase throughput, with bounded display updates."""

    def __init__(self) -> None:
        self.key: tuple[Any, ...] | None = None
        self.start = 0.0
        self.initial = 0
        self.previous = 0
        self.advances = 0
        self.last_progress = 0.0
        self.displayed: float | None = None
        self.display_time = 0.0
        self.stalled = False
        self.sources = False
        self.activity: tuple[Any, ...] | None = None

    def observe(self, progress: dict[str, Any], now: float) -> float | None:
        activity = _ingestion_progress_key({"ingestion_progress": progress})
        if activity != self.activity:
            self.activity = activity
            self.last_progress = now
        self.sources = progress.get("phase") in {"extraction", "chunking"}
        counts = progress.get("overall_progress" if self.sources else "progress")
        if (
            (not self.sources and progress.get("phase") not in PASSAGE_ETA_PHASES)
            or not isinstance(counts, dict)
            or counts.get("unit") != ("sources" if self.sources else "chunks")
            or not math.isfinite(now)
        ):
            self.key = None
            return None
        completed, total = counts.get("completed"), counts.get("total")
        if (
            type(completed) is not int
            or type(total) is not int
            or not 0 <= completed <= total
            or total <= 0
        ):
            self.key = None
            return None
        key = (
            progress.get("build_id"),
            progress.get("phase"),
            total,
            counts.get("unit"),
        )
        if key != self.key or completed < self.previous or now < self.start:
            self.key = key
            self.start = now
            self.initial = completed
            self.previous = completed
            self.advances = 0
            self.last_progress = now
            self.displayed = None
            self.display_time = now
            self.stalled = False
            return None
        if completed > self.previous:
            self.advances += 1
            self.last_progress = now
        self.previous = completed
        self.stalled = now - self.last_progress >= 60
        sampled = completed - self.initial >= 3 if self.sources else self.advances >= 2
        if not sampled or now <= self.start or completed == total:
            self.displayed = None
            return None
        return (total - completed) * (now - self.start) / (completed - self.initial)

    def label(self, estimate: float | None, now: float) -> str:
        if estimate is None:
            return "ETA —"
        if self.sources:
            # Power-of-two minute ranges are intentionally coarse for unequal books.
            upper = 2 ** max(1, math.ceil(math.log2(max(1, estimate / 60))))
            candidate = float(upper * 60)
        else:
            bucket = 60 if estimate >= 60 else 10
            candidate = float(max(bucket, math.ceil(estimate / bucket) * bucket))
        if self.displayed is None or (
            now - self.display_time >= 60
            and (candidate >= self.displayed * 1.5 or candidate <= self.displayed / 1.5)
        ):
            self.displayed = candidate
            self.display_time = now
        minutes = int(self.displayed / 60)
        if self.sources:
            return f"ETA {minutes // 2}–{minutes}m"
        return f"ETA ~{minutes}m" if minutes else f"ETA ~{int(self.displayed)}s"


def _ingestion_progress_line(
    status: dict[str, Any], estimator: _PhaseETA | None = None
) -> str | None:
    """Render a compact human-readable snapshot, not the backend payload."""
    progress = status.get("ingestion_progress")
    if not isinstance(progress, dict) or not progress:
        if estimator is not None:
            estimator.key = None
        return None
    now = time.monotonic()
    estimate = estimator.observe(progress, now) if estimator is not None else None
    phase = str(progress.get("phase") or "unknown")
    labels = {
        "source_hashing": "Hashing sources",
        "source_revalidation": "Rechecking sources",
        "dense_indexing": "Building dense index",
        "qdrant_indexing": "Building dense index",
        "bm25_indexing": "Building search index",
    }
    fields = [labels.get(phase, phase.replace("_", " ").capitalize())]

    def count(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        completed, total = value.get("completed"), value.get("total")
        if type(completed) is not int or type(total) is not int or total <= 0:
            return None
        unit = str(value.get("unit") or "items")
        if unit == "phase":
            return None
        units = {
            "pdf_page_batches_or_epub_sections": "extraction batches/sections",
            "extraction_units": "extraction units",
        }.get(unit, unit.replace("_", " "))
        return f"{completed:,}/{total:,} {units}"

    overall = progress.get("overall_progress")
    aggregate = count(overall) or count(progress.get("progress"))
    if aggregate:
        fields.append(aggregate)
    source = progress.get("source")
    if isinstance(source, str) and source:
        # Source names are data, not terminal commands or multiline log records.
        safe_source = "".join(char if char.isprintable() else " " for char in source)
        fields.append(safe_source)
    # Backend/source deadlines cannot describe measured global passage work.
    counts = progress.get("progress")
    passage_counts = (
        phase in PASSAGE_ETA_PHASES
        and isinstance(counts, dict)
        and counts.get("unit") == "chunks"
        and type(counts.get("completed")) is int
        and type(counts.get("total")) is int
        and 0 <= counts["completed"] <= counts["total"]
        and counts["total"] > 0
    )
    fields.append(estimator.label(estimate, now) if estimator else "ETA —")
    if passage_counts and counts["completed"] == counts["total"]:
        fields.append("Finalizing")
    if estimator is not None and estimator.stalled and estimator.key is not None:
        fields.append("Stalled: no progress for ≥60s")
    return " | ".join(fields)


def _ingestion_progress_key(status: dict[str, Any]) -> tuple[Any, ...] | None:
    progress = status.get("ingestion_progress")
    if not isinstance(progress, dict):
        return None
    counters = []
    for name in ("progress", "overall_progress", "source_progress"):
        value = progress.get(name)
        counters.append(
            (value.get("completed"), value.get("total"), value.get("unit"))
            if isinstance(value, dict)
            else None
        )
    return (
        progress.get("build_id"),
        progress.get("phase"),
        progress.get("source"),
        *counters,
    )


def _terminal_progress_line(line: str, columns: int) -> str:
    """Clip by terminal cell width, reserving one cell against line wrapping."""
    remaining = max(0, columns - 1)
    result: list[str] = []
    for char in line:
        width = (
            0
            if unicodedata.combining(char)
            else (2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1)
        )
        if width > remaining:
            break
        result.append(char)
        remaining -= width
    return "".join(result)


def _ingestion_progress_frame(status: dict[str, Any], line: str, columns: int) -> str:
    """Draw progress across the current phase, never within a source."""
    progress = status["ingestion_progress"]
    fields = line.split(" | ")
    header = fields[0]
    overall = progress.get("overall_progress")
    if isinstance(overall, dict) and overall.get("unit") == "sources":
        header += f" | {overall.get('completed', 0)}/{overall.get('total', 0)} sources"
    source = progress.get("source")
    if isinstance(source, str) and source:
        header += " | " + "".join(
            char if char.isprintable() else " " for char in source
        )
    counts = (
        overall
        if isinstance(overall, dict) and overall.get("unit") == "sources"
        else progress.get("progress")
    )
    completed = counts.get("completed") if isinstance(counts, dict) else None
    total = counts.get("total") if isinstance(counts, dict) else None
    valid = (
        type(completed) is int
        and type(total) is int
        and total > 0
        and 0 <= completed <= total
        and counts.get("unit") != "phase"
    )
    percentage = round(100 * completed / total) if valid else None
    eta = next(
        (field for field in fields if field.startswith("ETA ")),
        "",
    )
    suffix = f" {percentage:3d}%" if percentage is not None else "  --%"
    if eta:
        suffix += f" | {eta}"
    for field in fields:
        if field.startswith(("Stalled:", "Finalizing")):
            suffix += f" | {field}"
    bar_width = max(0, min(30, columns - len(suffix) - 5))
    filled = round(bar_width * completed / total) if valid else 0
    bar = f"[{'#' * filled}{'-' * (bar_width - filled)}]{suffix}"
    return (
        _terminal_progress_line(header, columns)
        + "\n"
        + _terminal_progress_line(bar, columns)
    )


async def _ingest_with_progress(
    operations: Local | Remote, *, force_recompute: bool
) -> dict[str, Any]:
    """Poll read-only status while the single ingestion request is outstanding."""
    ingest = asyncio.create_task(operations.ingest(force_recompute=force_recompute))
    status_task: asyncio.Task[dict[str, Any]] | None = None
    estimator = _PhaseETA()
    previous_key: tuple[Any, ...] | None = None
    last_printed = 0.0
    redraw = (
        bool(getattr(sys.stderr, "isatty", lambda: False)())
        and os.environ.get("TERM") != "dumb"
    )
    drawn = False
    try:
        while not ingest.done():
            done, _ = await asyncio.wait({ingest}, timeout=INGEST_PROGRESS_INTERVAL)
            if done:
                break
            status_task = asyncio.create_task(operations.status())
            await asyncio.wait(
                {ingest, status_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if ingest.done():
                break
            # Progress is advisory; a failed read must not abandon a build.
            status_result = (await asyncio.gather(status_task, return_exceptions=True))[
                0
            ]
            line = (
                _ingestion_progress_line(status_result, estimator)
                if isinstance(status_result, dict)
                else None
            )
            key = (
                _ingestion_progress_key(status_result)
                if isinstance(status_result, dict)
                else None
            )
            now = time.monotonic()
            changed = key != previous_key
            if line is not None and (
                changed or now - last_printed >= INGEST_HEARTBEAT_INTERVAL
            ):
                # Only an unchanged heartbeat claims the status endpoint answered;
                # it does not claim that the worker made progress.
                if redraw:
                    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
                    frame = _ingestion_progress_frame(status_result, line, columns)
                    sys.stderr.write(
                        ("\r\x1b[1A" if drawn else "")
                        + "\r\x1b[2K"
                        + frame.replace("\n", "\n\r\x1b[2K", 1)
                    )
                    sys.stderr.flush()
                    drawn = True
                else:
                    print(
                        line
                        if changed
                        else f"{line} | status checked {time.strftime('%H:%M:%S')}",
                        file=sys.stderr,
                        flush=True,
                    )
                previous_key = key
                last_printed = now
            status_task = None
        return await ingest
    finally:
        if drawn:
            sys.stderr.write("\n")
            sys.stderr.flush()
        # Drain an already-started HTTP read before Remote closes its client.
        # Do not start a final status request or print after ingestion returns.
        if status_task is not None:
            await asyncio.gather(status_task, return_exceptions=True)
        if not ingest.done():
            ingest.cancel()
            await asyncio.gather(ingest, return_exceptions=True)


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
        from ..core.tool_views import lean_status

        return dict(payload) if args.verbose else lean_status(payload)
    if command == "ingest":
        return await _ingest_with_progress(
            operations, force_recompute=args.force_recompute
        )
    if command == "search":
        search: dict[str, Any] = {
            "top_k": args.top_k,
            "categories_any": args.category,
            "projects_any": args.project,
            "keywords": args.keyword,
            "languages_any": args.language,
            "authors_any": args.author,
            "titles_any": args.title,
            "source_ids": args.source_id,
            "exclude_source_ids": args.exclude_source_id,
            "rerank": not args.no_rerank,
        }
        # A named --method is a diagnostic choice. Leaving it off asks the engine
        # for the method the generation can serve, which is BM25 with a
        # disclosure when only the dense half is unavailable.
        if args.method is not None:
            search["retrieval_method"] = args.method
        return await operations.search(args.query, **search)
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
    if command == "stats":
        return await operations.stats(
            days=args.days, top=args.top, largest_by=args.largest_by
        )
    if command == "history":
        if args.clear:
            return await operations.clear_history()
        return await operations.history(limit=args.limit, days=args.days)
    if command == "chunks":
        return await operations.source_chunks(
            source=args.source, page=args.page, page_size=args.page_size
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


# How often a serving app checks that the terminal it was started from is still
# its controlling terminal.
_TERMINAL_CHECK_SECONDS = 2.0


def _terminal_attached() -> bool:
    """Whether this process has a controlling terminal.

    The live process answers first, as `/proc/self/stat` reports it. Where there
    is no procfs, standard input standing on a terminal is the next best answer.
    """

    answer = has_terminal(os.getpid())
    if answer is not None:
        return answer
    return _own_tty() is not None


async def _terminal_lost() -> None:
    """Return when the controlling terminal this process started with is gone.

    A closed window sends SIGHUP, which the closing below already handles. A
    process the shell disowned, or one whose session leader exited, gets no
    signal: it only loses its terminal, so the loss is checked for directly.
    """

    # Polled: the kernel announces a lost terminal to a process outside the
    # foreground group with nothing an event loop can wait on.
    while True:
        if not _terminal_attached():
            return
        await asyncio.sleep(_TERMINAL_CHECK_SECONDS)


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

    # A process with no terminal would serve until `stop`, which is the detached
    # state this app does not serve. It is refused before anything is started.
    if not _terminal_attached():
        raise ResearchError(
            f"This process has no terminal, so {config.project_name} was not "
            "served. The app is served from a terminal and stops when that "
            f"terminal closes: run '{CLI_NAME} start' in one. Serving detached is "
            "not available."
        )
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
    # Only a terminal that was there can be lost; a run with none was refused.
    watched = [asyncio.ensure_future(_terminal_lost())] if _terminal_attached() else []
    try:
        await asyncio.wait(
            (serving, asked, *watched), return_when=asyncio.FIRST_COMPLETED
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Only the waits on the closing and the terminal are cancelled here. The
        # serving task is left for `App.stop`, which asks uvicorn to shut down and
        # then waits for it; cancelling this waiter would put a CancelledError
        # inside that task and turn an orderly stop into a traceback.
        asked.cancel()
        for waiter in watched:
            waiter.cancel()


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

    from ..runtime.installation import (
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


def _interactive_terminal() -> bool:
    """Whether a reader can be asked here, rather than a pipe or a log read.

    Both ends matter. The question is written to stderr and the answer is read
    from stdin, so one terminal and one file is a prompt in a log rather than a
    reader, and that is a check that prints and installs nothing.
    """

    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _fetch_published_release(url: str, timeout: float) -> str:
    """The one network request this command makes, named so a test can replace it.

    It reads the published release of this app's public repository and carries no
    token, key, or header that could. `--offline` never reaches it, and nothing is
    installed because of what it returns: it is the changelog a reader approves.
    """

    from ..runtime.release import fetch_text

    return fetch_text(url, timeout)


def _ask_to_install(
    published: PublishedRelease,
    notes: Sequence[str],
    *,
    target: str,
    stdin: Any = None,
    stderr: Any = None,
) -> bool:
    """Show the published changelog and ask once, with no as the answer.

    Everything is written to stderr, so stdout stays the machine-readable report
    the other flags promise, and the answer is one line. Anything that is not
    `y` or `yes` declines: an empty line, EOF, and Ctrl-C are the same no.
    Declining is an answer, not a failure, so it changes nothing and returns.
    """

    from ..runtime import release as release_module

    source = sys.stdin if stdin is None else stdin
    sink = sys.stderr if stderr is None else stderr
    lines = [f"A published release is available: {target}."]
    if published.found and published.release is not None:
        lines.extend(release_module.changelog_lines(published.release))
    else:
        lines.append(f"No changelog was published with it. {published.detail}.")
    lines.extend(("", *notes))
    sink.write("\n".join(lines) + "\nInstall this release? [y/N]: ")
    sink.flush()
    try:
        answer = source.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        answer = ""
    return answer.strip().lower() in {"y", "yes"}


async def _update(args: argparse.Namespace) -> dict[str, Any]:
    """Report what a newer version means, preview its changelog, install it approved.

    The check is the default and writes nothing. The comparison is against the
    latest published release, so the version in the answer and the version
    `research-rag --version` prints are the same number, and the changelog shown
    beside that answer is the body the maintainer published with the release.

    An install is the one thing here a reader approves: a terminal is asked once
    with the changelog in front of it and no as the default, `--apply` asks the
    same question, and `--yes` says the approval was given already. A terminal
    that cannot be asked is refused rather than assumed, and an answer that is not
    yes stops nothing and writes nothing. Applying also refuses a build that holds
    a project lock, refuses a checkout with uncommitted work, stops every app
    through that project's own recorded pid, and prints the command that starts
    each one again.
    """

    from ..project.registry import load as load_registered
    from ..runtime import update as update_module
    from ..runtime.version import version_block

    offline = bool(args.offline)
    runner = process_module.subprocess_runner
    local = update_module.probe_local(run=runner)
    remote = update_module.probe_remote(local, runner, offline=offline)
    plan = update_module.plan_update(local, remote)
    payload: dict[str, Any] = {
        "command": "update",
        "applied": False,
        # The two keys a caller reads instead of deriving: whether a release is
        # waiting to be installed, and where this installation stands against
        # what the remote publishes. A branch head is never either of them.
        "available": plan.available,
        "decision": plan.release_position,
        "notes": list(plan.notes),
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
        from ..runtime import release as release_module

        payload["declared_version_problems"] = list(
            release_module.release_consistency(
                local.checkout, local.declared or local.version, runner
            )
        )
    if plan.blocked:
        raise ResearchError(plan.blocked)
    if args.apply and offline:
        raise ResearchError(
            "--offline cannot update: nothing on the remote was asked. Nothing "
            "was changed. Run this again without --offline."
        )
    if args.apply and not plan.available:
        # Stopping an app is not free, so a run with nothing to apply changes
        # nothing at all, and says why.
        raise ResearchError("Nothing to apply. " + " ".join(plan.notes))

    # What the reader would approve, read before anything is asked or changed.
    published = update_module.published_release_for(
        local, runner, offline=offline, fetch=_fetch_published_release
    )
    notes = list(update_module.preview_notes(plan, published))
    payload["published"] = published.as_dict()
    payload["notes"] = notes
    target = (
        f"release {plan.release_version}, tagged {plan.release_tag}"
        if local.is_checkout
        else f"version {plan.remote_revision}"
    )
    projects = [
        update_module.ProjectState(entry.project_root, entry.project_name)
        for entry in load_registered()
        if (entry.project_root / ".research-rag").is_dir()
    ]
    held = update_module.held_projects(projects)
    interactive = _interactive_terminal()
    # A terminal that shows a preview is offered the install once; `--yes` is not
    # an answer to a question nobody asked, so it never asks and never offers.
    apply_requested = bool(args.apply) or (
        interactive and plan.available and not args.yes
    )
    refusal = "Nothing was changed. " + " ".join(one.refusal() for one in held)
    if args.apply and held:
        raise ResearchError(refusal)
    if held:
        payload["projects"] = [project.as_dict() for project in projects]
        payload["held"] = [one.as_dict() for one in held]
        notes.append(
            f"{len(held)} project(s) are being worked on, so an update would have "
            "to stop them first."
        )
    approval = update_module.approval_for(
        apply_requested=apply_requested,
        preselected=bool(args.yes),
        interactive=interactive,
    )
    payload["approval"] = approval.as_dict()
    if approval.method == update_module.UNAVAILABLE:
        raise ResearchError(update_module.approval_refusal())
    if approval.asked and plan.available:
        # The question goes to stderr with the changelog above it, and stdout
        # carries the report either way.
        approval = update_module.answered(
            approval, _ask_to_install(published, notes, target=target)
        )
        payload["approval"] = approval.as_dict()
    if not apply_requested:
        return payload
    if not approval.granted:
        payload["notes"] = [*notes, "Declined, so nothing was changed."]
        return payload
    expected_version = plan.release_version or plan.remote_revision
    if not published.found or published.version != expected_version:
        raise ResearchError(
            "Nothing was changed. The published GitHub release and changelog "
            f"must match the approved version {expected_version}. "
            "Run 'research-rag update' again after release information is available."
        )
    # An approved release is the one previewed, not whatever is newest by now.
    moved = update_module.target_moved(local, plan, runner, offline=offline)
    if moved:
        raise ResearchError(moved)
    if held:
        raise ResearchError(refusal)
    payload["projects"] = [project.as_dict() for project in projects]
    payload["previewed_version"] = plan.release_version or plan.remote_revision

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

    from ..runtime.doctor import mcp_entry_block, mcp_url_block, run_doctor

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
    unavailable: str | None = None
    if entry_check is None:
        try:
            async with _operations(config) as operations:
                status = await operations.status()
        except ImportError as exc:
            # Without the retrieval stack there is no corpus answer to report, so
            # the report is told which of its checks that leaves unknown.
            unavailable = (
                f"{exc.name or exc} is not importable, so this project's corpus "
                "cannot be read. Install this project's dependencies to answer "
                "anything about its evidence."
            )
    result = run_doctor(
        config,
        status,
        running=_service_processes(config.project_root),
        entry=entry_check,
        prefetch=args.prefetch_models,
        repair=args.repair_runtime,
        status_unavailable=unavailable,
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
        "Given no command at all, it serves the workspace from this terminal, so\n"
        "Ctrl-C or closing the terminal stops it, and it asks which project when\n"
        "this installation holds more than one. Serving never opens a browser:\n"
        "--start-ui is what asks for one.\n"
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
        from ..runtime.version import version_lines

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
        from . import bridge

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
