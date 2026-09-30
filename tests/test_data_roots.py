"""The names this app shares with the MCP server it was seeded from.

Both products read and write the same projects while the migration runs, so four
locations carry the MCP server's name rather than this app's, and three of them
are invisible in a diff: a renamed user settings directory is read by neither
product, a renamed model cache re-downloads about 150 MB on the first build, and
a renamed launcher state file lets two launchers stop each other's process.
Renaming any of them is a one-line change that costs a user their settings, their
models, or a running workspace, and nothing fails until then.

Every constant here is asserted, so a later rename has to delete its assertion
and state the migration that makes it safe. It is the same review either way.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from research_rag import config as config_module
from research_rag import launcher as launcher_module
from research_rag import settings as settings_module
from research_rag.settings import SETTINGS

ROOT = Path(__file__).resolve().parent.parent
LEGACY = "research-ultra-rag-mcp"
LEGACY_ENVIRONMENT_PREFIX = "RESEARCH_ULTRARAG_"


def test_the_user_settings_directory_is_the_mcp_servers() -> None:
    assert settings_module.USER_CONFIG_DIRECTORY == LEGACY


def test_the_settings_environment_prefix_is_the_mcp_servers() -> None:
    assert settings_module.SETTINGS_ENVIRONMENT_PREFIX == LEGACY_ENVIRONMENT_PREFIX


def test_the_model_cache_default_is_the_mcp_servers() -> None:
    """Both models live under that directory, and only a build needs them."""

    source = (ROOT / "src/research_rag/config.py").read_text(encoding="utf-8")
    assert f'user_cache_path("{LEGACY}", appauthor=False)' in source


def test_the_project_state_root_names_neither_product() -> None:
    """The roots both products write to carry no product name at all."""

    assert (
        Path(".research-rag") / "config.toml" == settings_module.PROJECT_CONFIG_RELATIVE
    )
    assert config_module._RUNTIME_MARKER == ".research-ultra-rag-runtime.json"
    assert "research_rag" not in config_module._RUNTIME_MARKER
    assert "research-rag" not in config_module._RUNTIME_MARKER


def test_the_generated_launcher_and_its_link_are_the_apps_own() -> None:
    """The MCP server's launcher sits in the same project under the same names."""

    assert launcher_module.LAUNCHER_NAME == "open-research-rag-ui.sh"
    assert launcher_module.LINK_NAME == "open-research-rag-ui.sh"
    assert launcher_module.LAUNCHER_NAME != "open-ui.sh"


def test_the_launcher_state_files_are_the_apps_own() -> None:
    """A shared pid file lets one product stop the other's process."""

    template = launcher_module._TEMPLATE
    for name in ("pid", "port", "lock"):
        assert f"research-rag-ui.{name}" in template
        assert f"open-ui.{name}" not in template
    assert "logs/research-rag-ui.log" in template
    assert "logs/open-ui.log" not in template


def test_the_launcher_runs_this_apps_own_command() -> None:
    assert launcher_module.UI_COMMAND == "research-rag"
    assert launcher_module.SERVE_COMMAND == "serve"
    template = launcher_module._TEMPLATE
    assert '"$UI_COMMAND" @SERVE@' in template
    assert "research-ultra-rag" not in template


def test_the_inert_tool_detail_key_is_still_declared() -> None:
    """A settings file the MCP server wrote may still name it, and the stack
    refuses an undeclared key in every layer."""

    keys = {setting.key for setting in SETTINGS}
    assert "runtime.tool_detail" in keys

    defaults = tomllib.loads(
        (ROOT / "src/research_rag/default.toml").read_text(encoding="utf-8")
    )
    assert "tool_detail" in defaults["runtime"]


def test_no_module_reads_the_inert_tool_detail_key() -> None:
    """Nothing projects an answer any more, so the key must select nothing."""

    package = ROOT / "src/research_rag"
    readers = [
        path.name
        for path in sorted(package.glob("*.py"))
        if "tool_detail" in path.read_text(encoding="utf-8")
        and path.name not in {"settings.py", "config.py"}
    ]
    assert readers == [], f"modules still reading tool_detail: {readers}"
