"""A command that no longer exists, documented, is invisible to the suite and visible
to the first reader who follows the manual.

The same holds for a capability removed with its description still in the manual, a
project file left undocumented, and a document naming a path that has moved.
"""

from __future__ import annotations

import ast
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


def _installed_commands() -> set[str]:
    """Every subcommand the one console command registers."""

    import argparse

    from research_rag.surfaces.cli import _parser

    return {
        name
        for action in _parser()._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    }


def test_the_readme_documents_every_installed_command() -> None:
    """A manual that omits a command, or names a removed one, misleads everyone."""

    for command in _installed_commands():
        assert command in README, f"{command!r} is installed but undocumented"


def test_the_cli_help_lists_every_installed_command() -> None:
    for command in sorted(_installed_commands()):
        assert command in CLI_HELP, f"{command!r} is installed but absent from --help"


def test_the_distribution_installs_nothing_named_for_another_product() -> None:
    """Three scripts, all this product's: the command, the gateway, the runtime.

    The gateway is a child this app spawns rather than a separate product, and the
    runtime script is what `doctor --repair-runtime` names. A script named for
    another project would be one a client could not tell from that project's.
    """

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert list(project["project"]["scripts"]) == [
        "research-rag",
        "research-rag-gateway",
        "research-rag-runtime",
    ]
    assert project["project"]["name"] == "research-rag"
    assert project["project"]["version"]


def test_the_product_is_described_as_a_server_with_three_front_ends() -> None:
    """It is a server product that also has a browser, and the manual must say so."""

    assert "research-rag" in README
    for front_end in ("start", "clients", "disconnect", "mcp", "ui"):
        assert front_end in README, front_end
    assert "mcp" in README.lower()
    # Configuring an agent is a capability a workspace-only product cannot document.
    assert "stdio" in README


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


def test_the_agent_surface_ships_with_its_instructions_and_its_projection() -> None:
    """The agent surface is a feature, so the two things it needs are checked."""

    assert not (ROOT / "AGENT_GUIDE.md").exists()
    instructions = (ROOT / "src/research_rag/project/instructions.py").read_text(
        encoding="utf-8"
    )
    projection = (ROOT / "src/research_rag/core/tool_views.py").read_text(
        encoding="utf-8"
    )
    assert "AGENT_INSTRUCTIONS" in instructions
    assert "present_tool_response" in projection


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


def test_the_no_mcp_surface_rule_holds_in_the_source_as_well_as_the_contract() -> None:
    """A rule stated in `AGENTS.md` and broken in a decorator is not a rule."""

    package = ROOT / "src" / "research_rag"
    declared: dict[str, str] = {}

    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for decorator in getattr(node, "decorator_list", []):
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                parts: list[str] = []
                while isinstance(target, ast.Attribute):
                    parts.append(target.attr)
                    target = target.value
                if isinstance(target, ast.Name):
                    parts.append(target.id)
                parts.reverse()
                if parts[:1] == ["app"] and parts[1:2] in (["tool"], ["resource"]):
                    declared[path.name] = ".".join(parts)

    assert declared == {}, f"an MCP surface reappeared: {declared}"


def test_the_process_lives_in_the_app_and_not_in_a_surface() -> None:
    """A surface that started a process would be a second instance of the app."""

    package = ROOT / "src/research_rag"

    assert not (package / "server.py").exists()
    assert not (package / "transport.py").exists()
    assert not (package / "verify.py").exists()
    assert not (package / "surfaces" / "server.py").exists()
    assert (package / "runtime" / "app.py").is_file()
    assert (package / "surfaces" / "bridge.py").is_file()
    assert (package / "surfaces" / "mcp.py").is_file()


@pytest.mark.parametrize(
    "document",
    [
        "README.md",
        "AGENTS.md",
        "FEATURES.md",
        "MEASUREMENTS.md",
        "ROADMAP.md",
        "TODO.md",
        "STORAGE.md",
    ],
)
def test_no_document_names_a_surface_that_moved(document: str) -> None:
    """A document naming a path that no longer exists sends a reader looking for it."""

    text = (ROOT / document).read_text(encoding="utf-8")

    for retired in (
        "research_rag/cli.py",
        "research_rag/ui.py",
        "research_rag/server.py",
    ):
        assert retired not in text, f"{document} still names {retired}"
    assert "create_research_transport" not in text, document


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


def test_the_help_menu_accounts_for_every_command() -> None:
    """Every installed command is in the hand-written menu, and in it exactly once.

    The menu is hand-written because argparse cannot group commands by the work they do,
    and a hand-written list drifts as soon as a command is added.
    """

    from research_rag.surfaces.cli import HELP_GROUPS

    listed = [name for _, entries in HELP_GROUPS for name, _ in entries]
    # `help` is reached by name and cannot list itself in a group it is the contents of.
    assert set(listed) | {"help"} == _installed_commands()
    assert len(listed) == len(set(listed)), "a command is in the menu twice"


def test_every_menu_entry_says_what_the_command_is_for() -> None:
    """A name in the menu with no description is a name the reader must look up."""

    from research_rag.surfaces.cli import HELP_GROUPS

    for title, entries in HELP_GROUPS:
        assert title, "a group has no heading"
        for name, description in entries:
            assert name and description, f"{title}: an entry is blank"
            assert not description[0].islower(), (
                f"{name}: a description must start with a capital"
            )
            assert description.endswith("."), f"{name}: a description is not a sentence"


def test_a_subject_page_names_a_command_and_the_help_menu_prints() -> None:
    """The menu and the subject pages are the same document argparse cannot print."""

    from research_rag.surfaces.cli import HELP_TOPICS, _help_menu

    menu = _help_menu()
    assert "help" in menu
    for topic, page in HELP_TOPICS.items():
        assert topic in menu, f"the menu does not offer the {topic!r} page"
        assert page.strip().endswith("."), f"{topic}: a page is not a finished answer"
    assert "one app" in menu
