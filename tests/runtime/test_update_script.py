"""`scripts/update.sh` delegates every flag it keeps to `research-rag update`.

The check runs offline, so this test never reaches a network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update.sh"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *arguments], capture_output=True, text=True, check=False
    )


def test_the_script_is_executable_and_documents_the_flags_it_keeps() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111

    result = _run("--help")

    assert result.returncode == 0
    for flag in ("--check", "--offline"):
        assert flag in result.stdout
    assert "research-rag update" in result.stdout


def test_check_offline_delegates_and_changes_nothing() -> None:
    result = _run("--check", "--offline")

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["command"] == "update"
    assert report["applied"] is False
    assert report["install"]["shape"] == "checkout"
    assert report["remote"]["reachable"] is False
    assert "--offline" in report["remote"]["detail"]
    assert report["plan"]["would_run"] == []


def test_the_script_rejects_unknown_options() -> None:
    result = _run("--nonsense")

    assert result.returncode == 2
    assert "unknown option" in result.stderr
