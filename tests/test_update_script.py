from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update.sh"


def test_update_script_is_executable_and_documented() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111

    result = subprocess.run(
        [str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "restart" in result.stdout.lower()


def test_update_script_check_offline_changes_nothing() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--check", "--offline"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "declared in pyproject:" in result.stdout
    assert "not fetched" in result.stdout


def test_update_script_rejects_unknown_options() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--nonsense"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "unknown option" in result.stderr
