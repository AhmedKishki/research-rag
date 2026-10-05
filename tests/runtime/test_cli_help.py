"""`--help` is the documentation a reader reaches first, so it is tested like one.

Three properties hold here and would each hide a real defect on their own:

- every registered command can print its own help, and every argument in it says
  what it is for, so no option is a bare name;
- the examples in `--help` parse, so a reader who copies one gets an answer rather
  than a refusal, and no example names a command or a placement that does not
  exist;
- a refusal is one sentence on stderr with no traceback, so a mistyped command and
  an uninitialised project are both readable rather than stack traces.

Nothing here starts the gateway, downloads a model, or reads a project: `--help`
must answer before a project exists, and a test that needed one would be testing
the wrong thing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research_rag.surfaces.cli import (
    _HELP_SUMMARIES,
    CLI_NAME,
    HELP_GROUPS,
    HELP_TOPICS,
    _metadata_body,
    _parser,
    _project_path,
)

# Aliased on its own line because isort gives an aliased member its own statement.
from research_rag.surfaces.cli import _operate as operate

ROOT = Path(__file__).resolve().parents[2]
CONSOLE_SCRIPT = ROOT / ".venv" / "bin" / CLI_NAME
EXAMPLE_LINE = re.compile(r"^ {2}(?P<command>.+)$", re.MULTILINE)
# A dest an argparse metavar would otherwise print raw, uppercased. Each is an
# internal name rather than something a reader is asked to supply.
RAW_METAVARS = frozenset(
    {
        "PROJECT_ROOT",
        "SET_OVERRIDES",
        "SESSION_ID",
        "USE_GENERATION",
        "START_UI",
        "MODEL_CACHE_ROOT",
        "RUNTIME_ROOT",
        "EMBEDDING_THREADS",
        "CONTEXT_CHUNKS",
        "TOP_K",
    }
)
# The global options that take a value, so an example's value is not read as the
# command name it follows.
_GLOBAL_VALUE_OPTIONS = (
    "--project-root",
    "--project",
    "--runtime-root",
    "--model-cache-root",
    "--config",
)


def _subparsers() -> argparse._SubParsersAction:
    """The one action holding every registered subcommand."""

    for action in _parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("the parser registers no commands")


def _commands() -> list[str]:
    return list(_subparsers().choices)


def _options(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Every option and positional the parser takes, in its own order.

    argparse's own `--help` is left out: its sentence belongs to argparse.
    """

    return [
        action
        for action in parser._actions
        if not isinstance(action, argparse._SubParsersAction)
        and action.dest not in {argparse.SUPPRESS, "help"}
    ]


def _named_command(words: list[str]) -> str | None:
    """The subcommand one example names, or None for a bare invocation."""

    index = 0
    while index < len(words):
        word = words[index]
        if word in _GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        return word
    return None


def _example_lines(epilog: str) -> list[str]:
    """The runnable lines the epilog shows, each indented by two spaces."""

    return [match.group("command") for match in EXAMPLE_LINE.finditer(epilog)]


def _run(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the console script itself, so what is tested is what is installed."""

    return subprocess.run(
        [str(CONSOLE_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**os.environ, "COLUMNS": "80"},
        timeout=120,
        check=False,
    )


def test_the_top_level_help_orients_a_reader_before_it_lists_commands() -> None:
    """`--help` answers two questions: what this is, and what to type first."""

    text = _parser().format_help()

    assert "init" in text and "ingest" in text, "the workflow is not in --help"
    assert "--start-ui" in text, "the browser option is not in --help"
    # The two project selectors are refused together, and a reader who does not
    # know that loses a command to a refusal about two options.
    assert "`--project` and\n`--project-root`" in text or (
        "--project-root" in text and "refused together" in text
    )
    for group in ("Choosing a project", "Commands", "Serving"):
        assert group in text, f"--help has no {group!r} section"


def test_every_registered_command_prints_its_own_help() -> None:
    """A command that cannot print its usage is a command whose reader is lost."""

    for name, subparser in _subparsers().choices.items():
        text = subparser.format_help()
        assert text.startswith(f"usage: {CLI_NAME} {name}"), name
        assert len(text.splitlines()) > 3, f"{name}: its help is a bare usage line"


def test_every_argument_says_what_it_is_for() -> None:
    """An option with no sentence is an option a reader has to guess at."""

    for name, subparser in _subparsers().choices.items():
        for action in _options(subparser):
            assert action.help, f"{name} {action.dest} has no help"
            assert not action.help.startswith("=="), f"{name} {action.dest}"
            assert action.help.strip().endswith("."), f"{name} {action.dest}"
    for action in _options(_parser()):
        assert action.help, f"the global option {action.dest} has no help"
        assert action.help.strip().endswith("."), action.dest


def test_no_metavar_prints_an_internal_name() -> None:
    """`--project-root PROJECT_ROOT` asks the reader for something internal."""

    for name, subparser in _subparsers().choices.items():
        for action in _options(subparser):
            shown = action.metavar or action.dest.upper()
            assert shown not in RAW_METAVARS, f"{name} shows {shown}"
            assert shown == shown.upper(), f"{name}: {shown} is not a placeholder"


def test_every_command_names_a_metavar_or_takes_no_value() -> None:
    """A value-taking option with a raw metavar is the same defect, by another name."""

    for name, subparser in _subparsers().choices.items():
        for action in _options(subparser):
            if action.nargs == 0 or isinstance(
                action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
            ):
                assert not action.metavar, (
                    f"{name}: a flag with a metavar {action.metavar}"
                )
                continue
            assert action.metavar or action.dest, name


def test_the_help_names_no_command_that_does_not_exist() -> None:
    """`ui` was a command once, and naming it sends a reader looking for it.

    Only prose that tells a reader what to type is scanned. An argument's help
    names paths and settings, where a backticked word is a directory rather than
    a command, so scanning those would forbid `` `models` `` and teach nothing.
    """

    parser = _parser()
    # `kill` is the shell's, and this app serves no command by that name.
    known = set(_commands()) | set(HELP_TOPICS) | {CLI_NAME, "kill"}
    prose = [parser.epilog or "", parser.description or ""]
    prose.extend(
        subparser.description or "" for subparser in _subparsers().choices.values()
    )
    for text in prose:
        for cited in re.findall(r"`([a-z][a-z-]+)`", text):
            assert cited in known, cited


def test_the_examples_in_the_help_parse_as_they_are_written() -> None:
    """A copied example that is refused is worse than no example."""

    parser = _parser()
    lines = _example_lines(parser.epilog or "")
    assert len(lines) >= 5, "the help shows too few examples to be a guide"

    for line in lines:
        words = shlex.split(line)
        assert words[0] == CLI_NAME, line
        try:
            parser.parse_args(words[1:])
        except SystemExit as refusal:
            # `--help` prints and exits zero; anything else exiting is a refusal.
            assert words[-1] in {"--help", "-h"}, line
            assert refusal.code == 0, line
        assert _named_command(words[1:]) in set(_commands()), line


def test_the_examples_keep_every_global_option_in_front_of_the_command() -> None:
    """An option after the command name is read as that command's own option."""

    for line in _example_lines(_parser().epilog or ""):
        words = shlex.split(line)[1:]
        command = _named_command(words)
        assert command in _commands(), line
        assert command not in words[: words.index(command)], line


def test_the_help_states_that_mcp_starts_nothing_and_needs_a_name() -> None:
    """The two facts a client entry gets wrong: an invented path, and autostart."""

    text = _subparsers().choices["mcp"].format_help()

    assert "starts no app" in text
    assert "--project-root" in text and "refused" in text
    assert "--project-name" in text
    assert "`mcp` before its own options" in text


def test_the_help_states_that_a_browser_needs_start_ui() -> None:
    """Serving never opens a browser, and a bare call is not the way to ask."""

    assert "Serving never opens one by itself" in _parser().format_help()
    assert "--start-ui" in _subparsers().choices["start"].format_help()


def test_the_help_states_what_the_metadata_review_replaces() -> None:
    """The review is the whole entry, so a reader must know before naming a field."""

    text = _subparsers().choices["metadata"].format_help()

    assert "replaces this source's whole entry" in text
    assert "--clear" in text
    assert "language" in text.lower() and "639" in text


def test_the_metadata_and_search_filters_are_not_the_project_selector() -> None:
    """`--project` picks the project once and filters a tag in an answer.

    The two share a name and nothing else, so the usage line is where a reader
    tells them apart: the selector takes a registered `NAME`, the filter takes a
    reviewed `TAG`.
    """

    parser = _parser()
    selector = next(action for action in parser._actions if action.dest == "project")
    assert selector.metavar == "NAME"

    for name in ("search", "metadata"):
        tag = next(
            action
            for action in _subparsers().choices[name]._actions
            if action.dest == "project"
        )
        assert tag.metavar == "TAG", name
        assert tag.metavar != selector.metavar, name


def _flat(text: str) -> str:
    """The help as one line, so a test checks a phrase rather than a wrap."""

    return " ".join(text.split())


def test_the_update_help_documents_the_preview_and_the_approval() -> None:
    """Three things a reader cannot guess: the changelog, the question, the answer."""

    text = _flat(_subparsers().choices["update"].format_help())

    assert "changelog" in text.lower(), (
        "the published changelog is not in update --help"
    )
    assert "no is the answer" in text, "the default answer is not stated"
    assert "--yes" in text and "unattended" in text
    assert "--apply" in text and "installs only on yes" in text
    assert "Without a terminal it only reports" in text


def test_the_update_help_states_that_nothing_is_installed_unasked() -> None:
    """`--yes` answers a question, so it never becomes one that was not asked."""

    text = _flat(_subparsers().choices["update"].format_help())

    assert "nothing is installed without that answer" in text
    assert "without --apply it reports and changes nothing" in text
    assert "any answer other than yes declines" in text
    # A flag that takes no value carries no placeholder a reader could fill in.
    yes = next(
        action
        for action in _subparsers().choices["update"]._actions
        if action.dest == "yes"
    )
    assert yes.nargs == 0
    assert not yes.metavar


def test_the_menu_entry_for_update_names_the_changelog_and_the_yes() -> None:
    """The menu is the reference, so it states what the reader is asked."""

    entry = next(
        description
        for _, entries in HELP_GROUPS
        for name, description in entries
        if name == "update"
    )

    assert "changelog" in entry.lower()
    assert "yes" in entry
    assert _HELP_SUMMARIES["update"] in _flat(_parser().format_help())


def test_every_command_has_exactly_one_line_in_the_flat_listing() -> None:
    """`--help` and `help` describe one command once, so they cannot disagree."""

    for name in _commands():
        assert name in _HELP_SUMMARIES, f"{name} has no line in --help"
    assert set(_HELP_SUMMARIES) == set(_commands())


def test_a_command_is_in_one_menu_group_and_no_other() -> None:
    listed = [entry for _, entries in HELP_GROUPS for entry, _ in entries]
    assert len(listed) == len(set(listed)), "a command is in the menu twice"


def test_a_summary_is_the_first_sentence_of_the_menu_entry() -> None:
    """The derivation, checked: a rewritten menu must not leave `--help` behind."""

    for _, entries in HELP_GROUPS:
        for name, description in entries:
            first = description.partition(". ")[0].rstrip(".") + "."
            assert _HELP_SUMMARIES[name] == first, name


def test_every_registered_command_is_in_the_menu_and_every_topic_is_offered() -> None:
    """`help` is the reference, so it holds every command and every subject."""

    listed = {name for _, entries in HELP_GROUPS for name, _ in entries}
    assert listed | {"help"} == set(_commands())

    topic = next(
        action
        for action in _subparsers().choices["help"]._actions
        if action.dest == "topic"
    )
    assert set(topic.choices) == set(HELP_TOPICS) | listed


def test_a_typo_is_refused_on_stderr_with_a_usage_line() -> None:
    """A mistyped command is answered with what to type, not with a stack."""

    result = _run("serach", "a question", cwd=ROOT)

    assert result.returncode == 2
    assert result.stdout == "", "a refusal wrote to stdout, which automation reads"
    assert "usage: " + CLI_NAME in result.stderr
    assert "Traceback" not in result.stderr
    assert "invalid choice" in result.stderr


def test_an_invalid_choice_is_refused_on_stderr() -> None:
    """The choices are in the refusal, so a reader does not go and find them."""

    result = _run("search", "a question", "--method", "semantic", cwd=ROOT)

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "hybrid" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_argument_is_refused_on_stderr() -> None:
    result = _run("search", cwd=ROOT)

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_prints_without_a_project_and_without_downloading_anything() -> None:
    """`--help` answers before a project exists: a reader has none yet."""

    result = _run("--help", cwd=ROOT)

    assert result.returncode == 0
    assert "Traceback" not in result.stderr
    assert result.stderr == ""
    # Nothing that reaches the network or writes a project can have run.
    assert "Traceback" not in result.stdout


def test_every_command_help_prints_without_a_project() -> None:
    for name in _commands():
        result = _run(name, "--help", cwd=ROOT)
        assert result.returncode == 0, f"{name} --help exited {result.returncode}"
        assert result.stdout.startswith(f"usage: {CLI_NAME} {name}"), name
        assert result.stderr == "", f"{name} --help wrote to stderr: {result.stderr}"


def test_the_mcp_entry_refuses_a_path_on_stderr_and_leaves_stdout_clean() -> None:
    """stdio carries the protocol, so a refusal there must not become payload."""

    result = _run("--project-root", "/tmp/some-project", "mcp", cwd=ROOT)

    assert result.returncode != 0
    assert result.stdout == "", "the bridge's stdout is the protocol channel"
    assert "--project-name" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_mcp_entry_without_a_name_is_refused_with_the_option_that_fixes_it() -> (
    None
):
    environment = {key: value for key, value in os.environ.items()}
    environment.pop("RESEARCH_RAG_PROJECT_NAME", None)
    result = subprocess.run(
        [str(CONSOLE_SCRIPT), "mcp"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**environment, "COLUMNS": "80"},
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--project-name" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_project_answers_with_the_command_that_creates_one(
    tmp_path: Path,
) -> None:
    """A directory with nothing built is a named condition, not a traceback.

    The command resolves a project the way any other call does, so it creates the
    portable state a project needs; that is why this runs in a temporary
    directory rather than somewhere real.
    """

    result = _run("--project-root", str(tmp_path), "status", cwd=ROOT)

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert payload["requires"] == ["ingest"]
    # Every gap names the call that closes it, which is the contract a reader
    # depends on and the reason this is not an empty answer.
    assert all(
        "remedy" in gap and "ingest" in gap["remedy"] for gap in payload["blocked_by"]
    )


def test_a_missing_project_root_is_refused_with_the_reason(tmp_path: Path) -> None:
    """A path that is not a directory is refused by name, on stderr, once."""

    result = _run("--project-root", str(tmp_path / "no-such-dir"), "status", cwd=ROOT)

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "not a directory" in result.stderr


def test_an_incomplete_mutation_is_refused_with_the_subject_it_is_missing() -> None:
    """`include` and `metadata` name what is missing rather than failing silently."""

    from research_rag.project.policy import ResearchError

    args = _parser().parse_args(["include"])
    with pytest.raises(ResearchError, match="Name the passage"):
        _operate(args)

    args = _parser().parse_args(
        ["exclude", "a.pdf", "--chunk", "chk_1", "--reason", "duplicate"]
    )
    with pytest.raises(ResearchError, match="name one of them"):
        _operate(args)

    args = _parser().parse_args(["metadata", "a.pdf", "--clear", "--year", "2003"])
    with pytest.raises(ResearchError, match="--clear cannot be combined"):
        _metadata_body(args)


def test_both_project_selectors_together_are_refused_with_one_to_pass() -> None:
    from research_rag.project.config import ConfigurationError

    args = _parser().parse_args(
        ["--project-root", "/tmp/one", "--project", "other", "status"]
    )
    with pytest.raises(ConfigurationError, match="pass one"):
        _project_path(args)


class _NullOperations:
    """Records nothing and answers nothing, so a refusal happens before a call."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the refusal must come before {name}")


def _operate(args: argparse.Namespace) -> dict[str, Any]:
    """The one dispatcher `_run` reaches, with nothing that could answer a call."""

    return asyncio.run(operate(args, _NullOperations()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
