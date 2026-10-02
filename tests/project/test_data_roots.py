"""Every shared location is named for this app, and nothing else may claim it.

A renamed user settings directory, model cache, or state file strands the one that
was written, and nothing fails until a command needs it: a settings file under a
directory nothing reads is ignored, a model cache under a new path re-downloads
both models on the first build, and a state file under a name another product also
writes lets one product stop the other's process.

The app's own names must stay its own, so a second product can never stop this one's
process or land beside it in a project.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import research_rag.project.config as config_module
import research_rag.project.settings as settings_module
import research_rag.runtime.app as app_module
from research_rag.project.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[2]


def test_the_user_settings_directory_is_this_apps_own() -> None:
    """One account directory holds this app's settings and its project register."""

    assert settings_module.USER_CONFIG_DIRECTORY == "research-rag"


def test_the_settings_environment_prefix_is_this_apps_own() -> None:
    """Every `RESEARCH_RAG_*` variable this app reads is one of these."""

    assert settings_module.SETTINGS_ENVIRONMENT_PREFIX == "RESEARCH_RAG_"


def test_the_model_cache_default_is_this_apps_own() -> None:
    """Both models live under that directory, and only a build needs them."""

    source = (ROOT / "src/research_rag/project/config.py").read_text(encoding="utf-8")
    assert 'user_cache_path("research-rag", appauthor=False)' in source


def test_no_setting_or_environment_variable_carries_another_products_name() -> None:
    """A reader finding one would look for a directory that does not exist."""

    for setting in SETTINGS:
        assert "ULTRARAG" not in setting.env, setting.key
        assert "ULTRARAG" not in setting.key.upper()


def test_the_project_state_root_is_named_for_this_app() -> None:
    assert (
        Path(".research-rag") / "config.toml" == settings_module.PROJECT_CONFIG_RELATIVE
    )
    assert config_module._RUNTIME_MARKER == ".research-rag-runtime.json"


def test_the_running_state_files_are_this_apps_own() -> None:
    """A shared pid file lets one product stop the other's process."""

    assert app_module.PID_FILE == "research-rag-ui.pid"
    assert app_module.PORT_FILE == "research-rag-ui.port"
    assert app_module.TTY_FILE == "research-rag-ui.tty"
    for name in (app_module.PID_FILE, app_module.PORT_FILE, app_module.TTY_FILE):
        assert "open-ui" not in name


def test_a_project_records_where_its_state_lives_only_when_relocated(
    tmp_path: Path,
) -> None:
    """A relocated runtime root is machine-local, so the record is not portable."""

    portable = tmp_path / "project" / ".research-rag"
    portable.mkdir(parents=True)
    record = portable / config_module.RUNTIME_ROOT_RECORD

    config_module.record_runtime_root(portable, None)
    assert not record.exists()
    assert config_module.recorded_runtime_root(portable) is None

    config_module.record_runtime_root(portable, tmp_path / "elsewhere" / "runtime")
    try:
        assert config_module.recorded_runtime_root(portable) == (
            tmp_path / "elsewhere" / "runtime"
        )
    finally:
        config_module.record_runtime_root(portable, None)


def test_the_agent_answer_detail_is_declared_and_used() -> None:
    """The one setting that is neither project, corpus, nor machine tunable.

    It selects the projection an agent's answers are built with, so the test pins the
    reader as well as the key.
    """

    keys = {setting.key for setting in SETTINGS}
    assert "runtime.tool_detail" in keys

    defaults = tomllib.loads(
        (ROOT / "src/research_rag/project/default.toml").read_text(encoding="utf-8")
    )
    assert defaults["runtime"]["tool_detail"] == "lean"

    readers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("src/research_rag/**/*.py")
        if "tool_detail" in path.read_text(encoding="utf-8")
    )
    # `surfaces/cli.py` names the key in the `help settings` page rather than reading
    # it, so the assertion guards the reader set: a module that behaves differently on
    # the key is added here deliberately, and the agent surface stays the one that acts
    # on its value.
    assert readers == [
        "src/research_rag/project/config.py",
        "src/research_rag/project/settings.py",
        "src/research_rag/surfaces/cli.py",
        "src/research_rag/surfaces/mcp.py",
    ], readers
