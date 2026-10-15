"""The command reference: the grouped menu, the subject pages, and the examples."""

from __future__ import annotations

import argparse
import textwrap

from ...project.config import (
    CLI_COMMAND,
)

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
                    "Extract, chunk, embed, and index the PDFs, EPUBs and MOBIs in the "
                    "sources directory, or only the sources named. Call it again to "
                    "resume a build that ran out of time."
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
            (
                "uningest",
                (
                    "Take sources' chunks and data out of the generation. The files "
                    "stay, and an ingest of every source reads them again unless "
                    "they are excluded."
                ),
            ),
        ),
    ),
    (
        "Running the app",
        (
            (
                "start",
                (
                    "Bring the app up and report where it is. --start-ui opens a browser; "
                    "start --lan enables home-network HTTP access without login."
                ),
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
                "lan",
                "Enable or disable LAN browser access immediately for this app run.",
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
                    "Every effective setting and the layer it came from, and "
                    "--write to change this project's config.toml. `help settings` "
                    "names the layers and what a change costs."
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

`config --write key=value` writes the project file, and so does the browser
workspace: both refuse a change they cannot validate and name what it costs. A
running app takes either at once, and a hand edit to any settings file at its next
call. The two global options below are per-call and are recorded nowhere:

  --set key=value    override one value for this command, repeatable
  --config PATH      add one more layer for this command

What a key does is written once, as `Setting.doc` in src/research_rag/project/settings.py.
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

Copy the PDFs, EPUBs and MOBIs into "/path/to/My Project/sources" before `ingest`, and
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


def _help_menu() -> str:
    """The whole menu: the order of the work, then the fact a first call needs.

    A description is stored as one sentence and wrapped here, so a wording stays
    readable when the column moves.
    """

    width = max(len(name) for _, entries in HELP_GROUPS for name, _ in entries)
    label = 2 + width + 2
    text_width = max(40, min(78, 96) - label)
    lines = [
        f"{CLI_COMMAND} — a research knowledge base over a project's own PDFs, EPUBs and MOBIs.",
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
            name=CLI_COMMAND, topics=", ".join(sorted(HELP_TOPICS))
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
