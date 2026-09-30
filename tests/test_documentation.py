"""The documents must describe the software that is installed.

The failure this file prevents is specific: an install that works, documented in
a way that leads a reader to a command that no longer exists, or a capability that
was removed with its description still in the manual. Both are invisible to the
test suite and visible to the first person who follows the documentation.
"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CLI_HELP = subprocess.run(
    ["uv", "run", "research-rag", "--help"],
    capture_output=True,
    text=True,
    check=True,
    cwd=ROOT,
).stdout


def _installed_commands() -> set[str]:
    """Every subcommand the one console command actually registers."""

    import argparse

    from research_rag.cli import _parser

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

    # The bare name `research-ultra-rag-mcp` stays in the manual, because it is
    # the shared settings directory this app reads; what must not survive is any
    # of the three console scripts the app no longer installs.
    for retired in (
        "research-ultra-rag-mcp --",
        "research-ultra-rag-ui",
        "research-ultra-rag-verify",
    ):
        assert retired not in README, f"{retired!r} is documented but not installed"


def test_the_cli_help_lists_every_installed_command() -> None:
    for command in sorted(_installed_commands()):
        assert command in CLI_HELP, f"{command!r} is installed but absent from --help"


def test_the_distribution_and_the_documentation_agree_on_one_command() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert list(project["project"]["scripts"]) == ["research-rag"]
    assert project["project"]["name"] == "research-rag"
    assert project["project"]["version"]


def test_the_product_is_described_as_a_workspace_and_not_an_mcp_server() -> None:
    assert "research-rag" in README
    for client_surface in ("MCP client", "mcpServers", "mcp_settings", "client entry"):
        assert client_surface not in README, client_surface


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


def test_the_agent_guide_became_a_storage_contract() -> None:
    """A no-MCP product owes its reader a state format, not an agent manual."""

    assert not (ROOT / "AGENT_GUIDE.md").exists()
    storage = (ROOT / "STORAGE.md").read_text(encoding="utf-8")

    assert "research-rag" in storage
    assert not (ROOT / "src/research_rag/instructions.py").exists()
    assert not (ROOT / "src/research_rag/tool_views.py").exists()


def test_the_storage_contract_names_every_project_file_the_app_writes() -> None:
    """A file that exists but is undocumented is a file nobody dares delete."""

    storage = (ROOT / "STORAGE.md").read_text(encoding="utf-8")

    for name in (
        "project.json",
        "source-catalog.json",
        "source-metadata.json",
        "source-exclusions.json",
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


def test_the_documented_mcp_boundaries_are_actually_gone() -> None:
    """Three modules existed only to serve the MCP surface; they must not return."""

    package = ROOT / "src" / "research_rag"

    for name in (
        "server.py",
        "transport.py",
        "verify.py",
        "tool_views.py",
        "instructions.py",
    ):
        assert not (package / name).exists(), name


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
def test_no_document_credits_the_removed_projection(document: str) -> None:
    """`tool_views` and the lean answer are gone; a reader must not chase them."""

    text = (ROOT / document).read_text(encoding="utf-8")

    for retired in ("tool_views", "create_research_transport", "SERVER_INSTRUCTIONS"):
        assert retired not in text, f"{document} still documents {retired}"
    assert "--tool-detail" not in text, document
