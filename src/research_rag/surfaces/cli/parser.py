"""The argument parser: every global option and every command's own options."""

from __future__ import annotations

import argparse
import os
from typing import Any

from ...project.config import (
    CLI_COMMAND,
    DEFAULT_UI_PORT,
)
from ...project.policy import RETRIEVAL_METHODS
from ...project.rerankers import RERANKER_MODEL_CHOICES
from .help import _HELP_SUMMARIES, HELP_EXAMPLES, HELP_GROUPS, HELP_TOPICS
from .projects import PROJECT_NAME_ENV

DEFAULT_DEPTH = 10


# The three variables the global options read, named once because `--help` states
# where each default comes from and a literal typed twice is one typo away from
# being wrong in one place only.
PROJECT_ROOT_ENV = "RESEARCH_RAG_PROJECT_ROOT"


PROJECT_ENV = "RESEARCH_RAG_PROJECT"


CONFIG_ENV = "RESEARCH_RAG_CONFIG"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_COMMAND,
        # Raw text, because the orientation and the examples below are written as
        # paragraphs and as a list, and the wrapping formatter would fold both into
        # one paragraph. Argument help is still wrapped: only a description and an
        # epilog are read raw.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Review evidence from a project's PDF, EPUB and MOBI corpus, in a terminal, "
            "in a browser, or through an agent.\n"
            "\n"
            "One project is one app, one index, and one set of review decisions. "
            "This command line, the browser workspace, and an agent's tools are "
            "three ways into that one process, so they read one corpus and cannot "
            "disagree about it.\n"
            "\n"
            "`init` gives a directory its project record, `ingest` builds the "
            "searchable generation from the PDFs, EPUBs and MOBIs in its sources "
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
            f"`--project-root DIR`. `{CLI_COMMAND} projects` lists them. Refused "
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
    serving.add_argument(
        "--take-over",
        dest="take_over",
        action="store_true",
        help=(
            "When another terminal, or none, holds this project's app, stop it and "
            "serve the project from this terminal on the same port, without asking. "
            "An interactive terminal is asked instead. A write in progress, such "
            "as an ingest build, is never interrupted."
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
            "Refuse the network and require every needed model and tokenizer to "
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
            "every dependency check, and the generation diagnostics — the policy "
            "differences a rebuild would apply, which do not make the generation "
            "unserviceable."
        ),
    )

    refresh = add(
        "ingest",
        description=(
            "Build the searchable generation from the PDFs, EPUBs and MOBIs in the "
            "source directory: extract, chunk, embed, and index them. Name sources "
            "to read only those; every other source the generation holds is kept "
            "as it is. An excluded source is never read. A build that fails or is "
            "cancelled leaves the generation search reads in place, and calling "
            "this again resumes the checkpoint."
        ),
    )
    refresh.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help=(
            "A source to read, by its path as `sources` reports it or by its src_ "
            "id. Name none to read every source that is not excluded."
        ),
    )
    refresh.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Read the named sources, or every source, again instead of reusing "
            "their documents, chunks, and vectors, and write a new generation "
            "beside the current one."
        ),
    )

    take_out = add(
        "uningest",
        description=(
            "Build a generation without these sources: their chunks and data leave "
            "it, and every other source is kept as it is. The files, their "
            "reviewed metadata, and their exclusions stay, so an ingest of every "
            "source reads them again unless they are excluded. The generation "
            "before keeps them until it is removed."
        ),
    )
    take_out.add_argument(
        "sources",
        nargs="+",
        metavar="SOURCE",
        help="A source to take out, by its path as `sources` reports it or by its src_ id.",
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
    _add_search_filters(find)

    answer = add(
        "answer",
        description=(
            "Answer a question by white-box synthesis. DeepSeek searches this "
            "project as often as the question needs and submits a derivation; the "
            "app renders every sentence from the exact wording of a retrieved "
            "passage, and marks a link no passage states as a gap. The question "
            "and every passage retrieved are sent to DeepSeek with the account's "
            "key, which `api-key` sets. The answer and its references print by "
            "default; --verbose adds the provenance, the reasoning, the sources, "
            "and what was sent."
        ),
    )
    answer.add_argument(
        "question",
        metavar="QUESTION",
        help="The question to answer. Quote it when it carries spaces.",
    )
    answer.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Also print the provenance of each sentence, the reasoning, each "
            "source with its chunk id and locator, the searches, and every "
            "passage sent."
        ),
    )
    _add_search_filters(answer)

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

    settings = add(
        "config",
        description=(
            "Print every effective setting with the layer it came from, what the "
            "key does, and what a change to it costs. With --write, change this "
            "project's own config.toml through the same check the workspace uses: "
            "a running app takes the value at once, and a key a higher layer sets "
            "is refused. A running app reads a hand edit at its next call. "
            "`help settings` names the layers."
        ),
    )
    settings.add_argument(
        "--write",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="Write one value to this project's config.toml; repeatable.",
    )
    settings.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Accept a change that alters what the next build records. Without it, "
            "such a change is refused with its cost."
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
            "Download the pinned embedding and reranker models into the cache, "
            "and provision the managed Docling environment. This is the only "
            "doctor operation that reaches the network or fetches anything."
        ),
    )
    examine.add_argument(
        "--repair-runtime",
        action="store_true",
        help=(
            "Deprecated informational no-op. The external UltraRAG runtime is no "
            "longer used, so this repairs nothing, reaches no network, and "
            "changes no file: it only reports that normal retrieval uses the "
            "required direct libraries. Kept for compatibility."
        ),
    )

    browser = add(
        "start",
        description=(
            "Serve this project's app from this terminal and report the address. "
            "Ctrl-C, a closed window, and `kill` all end it, so the terminal is "
            "where the app lives: this never leaves one running in the "
            "background, and it never opens a browser unless --start-ui was "
            "passed. An app already up for this project is never started a "
            "second time: an interactive terminal is asked whether to take it "
            "over, and --take-over restarts it here without asking."
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
    browser.add_argument(
        "--lan",
        action="store_true",
        help=(
            "Expose this project's workspace over HTTP to private IPv4 devices on "
            "your trusted home network, without login. Open the website's Remote "
            "Access page for phone URLs and QR codes. Control and MCP stay local. "
            "Do not forward this port to the internet."
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
    lan = add(
        "lan",
        description="Change LAN browser access immediately for this app run. HTTP has no authentication or encryption; use only a trusted home network. Control and MCP remain local.",
    )
    lan.add_argument(
        "action",
        choices=("enable", "disable"),
        help="Enable trusted home-LAN HTTP access or return to loopback-only serving without restarting.",
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
    credentials = add(
        "api-key",
        description=(
            "Keep the account's DeepSeek key, which an answer sends with every "
            "request. It lives in credentials.json beside the account's settings, "
            "readable by this account alone, and RESEARCH_RAG_DEEPSEEK_API_KEY "
            "overrides it. Every project on the account answers with it. Needs no "
            "project."
        ),
    )
    credentials.add_argument(
        "action",
        choices=("set", "clear", "status"),
        help=(
            "set reads the key from a hidden prompt, or one line of stdin when no "
            "terminal is attached, so it never lands in shell history. clear "
            "removes it. status says whether one is set and where, never what."
        ),
    )
    return parser


def _add_search_filters(parser: argparse.ArgumentParser) -> None:
    """The reviewed-metadata filters a search and an answer both take."""

    filters = parser.add_argument_group(
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
