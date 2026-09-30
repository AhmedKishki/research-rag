from __future__ import annotations

import tomllib
from pathlib import Path

from research_rag import version as version_module
from research_rag.version import (
    APP_VERSION,
    installed_version,
    restart_required,
    version_block,
    version_label,
)


def test_version_block_reports_the_running_and_installed_versions() -> None:
    block = version_block()

    assert block["app"] == APP_VERSION
    assert block["installed"] == installed_version()
    assert isinstance(block["ui"], str)
    assert block["restart_required"] is False


def test_version_label_names_the_app_and_the_shared_ui() -> None:
    label = version_label()

    assert label.startswith("research-rag ")
    assert "UI " in label


def test_declared_version_matches_the_installed_distribution() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        declared = tomllib.load(handle)["project"]["version"]

    # A version bumped without `uv sync` is exactly the drift this guards against.
    assert installed_version() == declared


def test_restart_is_required_when_the_process_is_older(monkeypatch) -> None:
    monkeypatch.setattr(version_module, "APP_VERSION", "0.0.1")

    assert restart_required() is True
    assert version_block()["restart_required"] is True
