"""A command that no longer exists, documented, is invisible to the suite and visible
to the first reader who follows the quickstart.

The same holds for a capability removed with its description still in the quickstart,
a project file left undocumented, and a document naming a path that has moved.

The command reference is `research-rag help`: one menu holding every registered
command exactly once, one page per subject, and each command's own usage. The
README is the way in, so its examples are parsed against the same parser rather
than compared as prose.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CLI_HELP = subprocess.run(
    ["uv", "run", "research-rag", "--help"],
    capture_output=True,
    text=True,
    check=True,
    cwd=ROOT,
).stdout
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)
# A global option that takes a value, so the value is not read as the command.
_GLOBAL_VALUE_OPTIONS = (
    "--project-root",
    "--project",
    "--runtime-root",
    "--model-cache-root",
    "--config",
)


def _installed_commands() -> set[str]:
    """Every subcommand the one console command registers."""

    from research_rag.surfaces.cli.parser import _parser

    return {
        name
        for action in _parser()._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    }


def _readme_command_lines() -> list[list[str]]:
    """Every `research-rag ...` line the README's own examples show a reader running."""

    lines: list[list[str]] = []
    for block in BASH_BLOCK.findall(README):
        for line in block.splitlines():
            words = shlex.split(line, comments=True)
            if not words or words[0] != "research-rag":
                continue
            lines.append(words[1:])
    return lines


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


def test_the_help_is_the_command_reference_and_holds_every_command() -> None:
    """Every installed command is in the menu exactly once, so the reference is whole.

    A command the menu omits is a command a reader cannot find, and a command in
    the menu twice is a menu that has drifted from the parser.
    """

    from research_rag.surfaces.cli.help import HELP_GROUPS

    listed = [name for _, entries in HELP_GROUPS for name, _ in entries]
    # `help` is reached by name and cannot list itself in a group it is the contents of.
    assert set(listed) | {"help"} == _installed_commands()
    assert len(listed) == len(set(listed)), "a command is in the menu twice"
    for title, entries in HELP_GROUPS:
        assert title, "a group has no heading"
        for name, description in entries:
            assert name and description.strip(), f"{title}: an entry is blank"


def test_every_command_the_readme_shows_is_one_the_parser_accepts() -> None:
    """An example a reader can copy but not run is the quickstart's only real defect."""

    from research_rag.surfaces.cli.parser import _parser

    parser = _parser()
    commands: set[str | None] = set()
    for words in _readme_command_lines():
        try:
            parser.parse_args(words)
        except SystemExit as refusal:
            # argparse refuses a line by exiting rather than by raising.
            pytest.fail(f"{' '.join(words)} does not parse: exit {refusal.code}")
        commands.add(_named_command(words))

    assert commands, "the quickstart shows no command at all"
    for command in commands - {None}:
        assert command in _installed_commands(), f"{command!r} is not a command"


def test_the_readme_points_at_the_help_reference_for_the_rest() -> None:
    """The README is a way in; the reference is where every other command lives."""

    from research_rag.project.config import CLI_COMMAND as CLI_NAME
    from research_rag.surfaces.cli.help import HELP_TOPICS

    assert f"{CLI_NAME} help" in README
    assert f"{CLI_NAME} COMMAND --help" in README
    assert f"{CLI_NAME} help TOPIC" in README
    for topic in HELP_TOPICS:
        assert topic in README, topic


def test_the_readme_states_what_this_app_needs_before_it_runs() -> None:
    """Requirements a reader discovers by failing are not requirements."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    low, _, high = project["project"]["requires-python"].partition(",")

    def _minor(bound: str) -> int:
        return int(bound.lstrip("><=").split(".")[1])

    # Both bounds are minor versions of one minor series, and the upper one is
    # exclusive: `>=3.11,<3.13` supports 3.11 and 3.12.
    for version in range(_minor(low), _minor(high)):
        assert f"3.{version}" in README, f"Python 3.{version} is supported but unstated"
    for requirement in ("uv", "git", "--offline"):
        assert requirement in README.lower(), requirement
    assert "download" in README.lower()
    assert "internet" in README.lower() or "network" in README.lower()


def test_the_readme_offers_the_three_front_ends_to_one_app() -> None:
    """A browser, an agent, and a terminal reach one process, and all three are shown."""

    for front_end in (
        "start",  # the app, and the workspace it serves
        "--start-ui",  # that workspace in a browser
        "search",  # the command line
        "mcp",  # the agent surface
    ):
        assert front_end in README, front_end
    assert "stdio" in README
    # The workspace is served from the terminal that started it, so the reader is
    # told to keep that terminal open rather than to look for a daemon.
    assert "terminal" in README


def test_the_readme_keeps_the_original_the_quote_authority() -> None:
    """Cleaned text is not a transcript, and a reader must be sent to the source."""

    assert "transcript" in README
    assert "locator" in README
    assert "sources" in README
    assert "cleaned" in README


def test_the_cli_help_lists_every_installed_command() -> None:
    for command in sorted(_installed_commands()):
        assert command in CLI_HELP, f"{command!r} is installed but absent from --help"


def test_the_distribution_installs_nothing_named_for_another_product() -> None:
    """The app command and two retired-command migration shims use this name.

    A script named for another project would be one a client could not tell from
    that project's. The shims neither expose upstream tools nor install a runtime.
    """

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert list(project["project"]["scripts"]) == [
        "research-rag",
        "research-rag-gateway",
        "research-rag-runtime",
    ]
    assert project["project"]["name"] == "research-rag"
    assert project["project"]["version"]


def test_every_package_data_pattern_names_a_shipped_file() -> None:
    """A pattern that matches nothing ships nothing, whatever the comment beside it says."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]

    for package, patterns in package_data.items():
        directory = ROOT / "src" / package.replace(".", "/")
        for pattern in patterns:
            assert list(directory.glob(pattern)), f"{package}: {pattern}"


def test_the_package_and_lockfile_have_no_ultrarag_install_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    locked = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    dependencies = [
        *project["project"].get("dependencies", []),
        *(
            dependency
            for extra in project["project"].get("optional-dependencies", {}).values()
            for dependency in extra
        ),
    ]

    assert all(
        not dependency.casefold().startswith("ultrarag") for dependency in dependencies
    )
    assert all(
        package["name"].casefold() != "ultrarag" for package in locked["package"]
    )


def test_the_upstream_credit_survives_in_the_readme_and_the_notice() -> None:
    """A fork must keep the attribution, the licence, and the disclaimer."""

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    for text in (README, notice):
        assert "UltraRAG" in text
        assert "https://github.com/OpenBMB/UltraRAG" in text
    assert "https://ultrarag.openbmb.cn/" in notice
    for credit in ("THUNLP", "NEUIR", "OpenBMB", "AI9stars"):
        assert credit in notice, credit
    assert "Apache License" in notice
    assert "not an official UltraRAG release" in notice
    assert "AGPL" in notice


def test_the_storage_contract_names_every_project_file_the_app_writes() -> None:
    """A file that exists but is undocumented is a file nobody dares delete."""

    storage = (ROOT / "STORAGE.md").read_text(encoding="utf-8")

    for name in (
        "project.json",
        "source-catalog.json",
        "source-metadata.json",
        "source-exclusions.json",
        "chunk-exclusions.json",
        "current.json",
        "manifest.json",
        "extracted-units.jsonl",
        "chunks.jsonl",
        "embeddings.npy",
        "artifact-lookup.sqlite3",
    ):
        assert name in storage, name


def test_the_readme_points_at_the_storage_contract_instead_of_duplicating_it() -> None:
    assert "STORAGE.md" in README


def test_every_agent_guidance_rule_points_at_a_file_that_exists() -> None:
    """A contract naming a document that was deleted is worse than no contract."""

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    for name in (
        "README.md",
        "STORAGE.md",
        "FEATURES.md",
        "MEASUREMENTS.md",
        "ROADMAP.md",
        "TODO.md",
        "NOTICE",
        "LICENSE",
    ):
        assert name in agents, name
        assert (ROOT / name).exists(), name


def test_a_shipped_client_template_names_a_project_after_the_command() -> None:
    """The two client templates must put `--project-name` after the `mcp` subcommand.

    An entry that puts it first has the parser read the project name as the command: the
    server exits before it can answer, and the client reports a closed connection with no
    cause.
    """

    import json

    from research_rag.runtime.doctor import _strip_jsonc

    for name in ("mcp_settings.example.json", "kilo-mcp.example.jsonc"):
        document = json.loads(_strip_jsonc((ROOT / name).read_text(encoding="utf-8")))
        servers = document.get("mcpServers") or document["mcp"]
        entry = next(iter(servers.values()))
        # One template splits the command and its arguments, the other lists them.
        arguments = (
            entry["command"]
            if isinstance(entry.get("command"), list)
            else [entry["command"], *entry["args"]]
        )

        assert arguments[1] == "mcp", name
        assert arguments.index("--project-name") > arguments.index("mcp"), name
        assert "--project-root" not in arguments, name
        assert arguments[-1] != "mcp", name


def test_a_subject_page_names_a_command_and_the_help_menu_prints() -> None:
    """The menu and the subject pages are the same document argparse cannot print."""

    from research_rag.surfaces.cli.help import HELP_TOPICS, _help_menu

    menu = _help_menu()
    assert "help" in menu
    for topic, page in HELP_TOPICS.items():
        assert topic in menu, f"the menu does not offer the {topic!r} page"
        assert page.strip(), f"{topic}: a page is empty"
