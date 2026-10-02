"""An installer that replaces a file it did not create is the fault this command must not
have.

Both write into the account's own directories, which the account fixture points at a
throwaway tree.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

import research_rag.surfaces.cli as cli_module
from research_rag.project.config import ConfigurationError
from research_rag.runtime import installation
from research_rag.runtime.doctor import desktop_entry_checks
from research_rag.surfaces.cli import _init, _parser

FAKE_SCRIPT = "#!/bin/sh\n# a console script standing in for the real one\n"


def _args(*arguments: str) -> Any:
    return _parser().parse_args(list(arguments))


def _console_script(tmp_path: Path, name: str = "console") -> Path:
    directory = tmp_path / name / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "research-rag"
    script.write_text(FAKE_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    return script


def _initialised(project: Path) -> Path:
    _init(_args("--project-root", str(project), "init", "--name", "My Thesis"))
    return project


def _cli(*arguments: str) -> dict[str, Any]:
    result = asyncio.run(cli_module._run(_args(*arguments)))
    assert result.payload is not None
    return result.payload


def _desktop_cli(*arguments: str) -> dict[str, Any]:
    """Run a command with the account's command installed, as a reader would.

    The menu entry runs the command `install` puts on the `PATH`, so writing one without
    it would produce an entry pointing at nothing.
    """

    _cli("install")
    return _cli(*arguments)


def test_the_command_runs_this_installation_from_anywhere(tmp_path: Path) -> None:
    """It is a wrapper naming the interpreter rather than a link to the console script inside
    the virtual environment: `uv sync` recreates that file, and a link to it answers "No
    such file or directory" for a command the reader installed.
    """

    import sys

    payload = _desktop_cli("install")

    command = Path(payload["console"]["path"])
    body = command.read_text(encoding="utf-8")
    assert command.name == "research-rag"
    assert command.parent == installation.account_bin_directory()
    assert body.startswith("#!/bin/sh")
    assert installation.WRAPPER_MARKER in body
    assert f'exec "{Path(sys.executable).expanduser().absolute()}"' in body
    assert '-m research_rag "$@"' in body
    assert body.endswith("\n")
    assert command.stat().st_mode & 0o111, "the command must be executable"


def test_the_command_actually_runs(tmp_path: Path) -> None:
    import sys

    _desktop_cli("install")
    result = subprocess.run(
        [str(installation.account_command_path()), "--version"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert Path(sys.executable) is not None


def test_installing_twice_changes_nothing(tmp_path: Path) -> None:
    _desktop_cli("install")
    command = installation.account_command_path()
    first = command.read_text(encoding="utf-8")

    payload = _desktop_cli("install")

    assert payload["console"]["state"] == "already_installed"
    assert command.read_text(encoding="utf-8") == first


def test_a_foreign_command_on_the_path_is_reported_and_left_alone(
    tmp_path: Path,
) -> None:
    command = installation.account_command_path()
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="is not the command"):
        _cli("install")

    assert "exit 0" in command.read_text(encoding="utf-8")


def test_force_replaces_a_foreign_command_on_the_path(tmp_path: Path) -> None:
    command = installation.account_command_path()
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    payload = _cli("install", "--force")

    assert payload["console"]["state"] == "replaced"
    assert installation.WRAPPER_MARKER in command.read_text(encoding="utf-8")


def test_uninstall_removes_only_the_command_this_installation_wrote(
    tmp_path: Path,
) -> None:
    _cli("install")

    payload = _cli("install", "--uninstall")

    assert payload["console"]["state"] == "removed"
    assert not installation.account_command_path().exists()


def test_uninstall_refuses_a_command_that_is_not_this_ones(tmp_path: Path) -> None:
    command = installation.account_command_path()
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="will not delete it"):
        _cli("install", "--uninstall")

    assert command.is_file()


def test_uninstalling_an_absent_command_reports_it(tmp_path: Path) -> None:
    payload = _cli("install", "--uninstall")

    assert payload["console"]["state"] == "absent"


def test_a_project_path_with_spaces_survives_the_round_trip(tmp_path: Path) -> None:
    project = _initialised(tmp_path / "My Thesis")

    assert _init(_args("--project-root", str(project), "init")) is not None


def test_an_entry_with_a_quoted_exec_is_read_back_as_one_path() -> None:
    path = "/home/ahmed/My Thesis/.venv/bin/research-rag"

    entry = installation.desktop_entry_from_text(
        Path("/tmp/research-rag.desktop"),
        f"{installation.ENTRY_MARKER}\nExec={installation.exec_argument(path)}\n",
    )

    assert entry is not None
    assert entry.executable == Path(path)


def test_the_console_command_needs_no_project(tmp_path: Path) -> None:
    payload = _desktop_cli("install")

    assert payload["console"]["state"] in {"created", "already_installed"}
    assert payload["console"]["path"].endswith("research-rag")


def test_the_menu_entry_serves_a_project_in_a_terminal_window(
    tmp_path: Path,
) -> None:
    """A menu entry that serves the app in the background puts a process on the machine that
    no window, no prompt, and no Ctrl-C reaches, so asking for a terminal window and naming
    no project keeps the start visible.
    """

    payload = _desktop_cli("install", "--desktop")

    entry = Path(payload["desktop"]["entry"]["path"])
    fields = dict(
        line.split("=", 1)
        for line in entry.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "["))
    )
    assert entry.name == "research-rag.desktop"
    assert fields["Type"] == "Application"
    assert fields["Terminal"] == "true"
    assert fields["Name"] == "research-rag"
    assert Path(fields["Exec"]).name == "research-rag"
    assert "--project" not in fields["Exec"]
    assert "Utility" in fields["Categories"]
    assert Path(fields["Icon"]).is_absolute()
    assert Path(fields["Icon"]).is_file()
    # A value the desktop is never told about makes an entry it never resolves as started.
    assert "StartupNotify" not in fields
    assert "StartupWMClass" not in fields


def test_one_entry_serves_every_project_this_installation_holds(
    tmp_path: Path,
) -> None:
    _initialised(tmp_path / "thesis")
    _initialised(tmp_path / "archive")

    _desktop_cli("install", "--desktop")

    entries = installation.desktop_entries()
    assert [entry.path.name for entry in entries] == ["research-rag.desktop"]


def test_an_older_per_project_entry_is_replaced_by_the_one(tmp_path: Path) -> None:
    """A menu listing four copies of the same application is the old arrangement."""

    _desktop_cli("install", "--desktop")
    applications = installation.account_applications_directory()
    older = applications / "research-rag-my-thesis.desktop"
    older.write_text(
        f"{installation.ENTRY_MARKER}\n"
        "Type=Application\n"
        "Name=My Thesis\n"
        "Exec=/home/ahmed/My Thesis/open-research-rag-ui.sh --open\n"
        "Terminal=false\n",
        encoding="utf-8",
    )

    payload = _desktop_cli("install", "--desktop")

    assert payload["desktop"]["superseded"] == [str(older)]
    assert not older.exists()
    assert [entry.path.name for entry in installation.desktop_entries()] == [
        "research-rag.desktop"
    ]


def test_the_entry_is_valid_by_the_desktops_own_validator(tmp_path: Path) -> None:
    """A menu entry the desktop cannot parse is not an entry."""

    payload = _desktop_cli("install", "--desktop")

    result = subprocess.run(
        ["desktop-file-validate", payload["desktop"]["entry"]["path"]],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_icon_is_written_by_the_command_as_a_constant(tmp_path: Path) -> None:
    payload = _desktop_cli("install", "--desktop")

    icon = Path(payload["desktop"]["icon"]["path"])

    assert icon.read_text(encoding="utf-8").startswith("<?xml")
    assert installation.account_icon_path() == icon


def test_installing_the_entry_twice_writes_nothing_the_second_time(
    tmp_path: Path,
) -> None:
    _desktop_cli("install", "--desktop")
    first = installation.desktop_entry_path().read_text(encoding="utf-8")

    payload = _desktop_cli("install", "--desktop")

    assert payload["desktop"]["entry"]["state"] == "unchanged"
    assert installation.desktop_entry_path().read_text(encoding="utf-8") == first


def test_a_desktop_entry_this_app_did_not_write_is_left_alone(tmp_path: Path) -> None:
    entry = installation.desktop_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=Mine\nExec=/bin/true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="did not write"):
        _desktop_cli("install", "--desktop")

    assert "Name=Mine" in entry.read_text(encoding="utf-8")


def test_force_replaces_a_foreign_entry_but_uninstall_will_not_remove_it(
    tmp_path: Path,
) -> None:
    entry = installation.desktop_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=Mine\nExec=/bin/true\n",
        encoding="utf-8",
    )

    payload = _desktop_cli("install", "--desktop", "--force")
    assert payload["desktop"]["entry"]["state"] == "replaced"
    assert "Terminal=true" in entry.read_text(encoding="utf-8")

    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=Mine\nExec=/bin/true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="did not write"):
        _desktop_cli("install", "--desktop", "--uninstall")


def test_the_entry_writes_nothing_a_project_owns(tmp_path: Path) -> None:
    """Adding a menu entry must not touch a project's portable state."""

    project = _initialised(tmp_path / "thesis")
    before = {
        path: path.stat().st_mtime_ns
        for path in sorted(project.rglob("*"))
        if path.is_file()
    }

    _desktop_cli("install", "--desktop")

    after = {
        path: path.stat().st_mtime_ns
        for path in sorted(project.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_uninstall_removes_the_entry_and_the_icon_it_wrote(tmp_path: Path) -> None:
    _desktop_cli("install", "--desktop")
    entry = installation.desktop_entry_path()
    icon = installation.account_icon_path()

    payload = _desktop_cli("install", "--desktop", "--uninstall")

    assert payload["desktop"]["entry"]["state"] == "removed"
    assert not entry.exists()
    assert not icon.exists()
    assert installation.desktop_entries() == ()


def test_uninstalling_an_absent_entry_reports_it_rather_than_failing(
    tmp_path: Path,
) -> None:
    payload = _desktop_cli("install", "--desktop", "--uninstall")

    assert payload["desktop"]["entry"]["state"] == "absent"


def test_the_doctor_names_an_entry_written_by_an_older_build(tmp_path: Path) -> None:
    """The old entry served the app in the background, and the doctor must say so."""

    _desktop_cli("install", "--desktop")
    entry = installation.desktop_entry_path()
    text = entry.read_text(encoding="utf-8").replace("Terminal=true", "Terminal=false")
    entry.write_text(text, encoding="utf-8")

    findings = desktop_entry_checks()

    assert [check.name for check in findings] == ["desktop_entry"]
    assert findings[0].state == "warn"
    assert "does not open a terminal window" in findings[0].reason
    assert "install --desktop" in findings[0].remedy_command


def test_the_doctor_names_an_entry_that_pins_one_project(tmp_path: Path) -> None:
    """A rename or a move breaks an entry that names a project."""

    _desktop_cli("install", "--desktop")
    entry = installation.desktop_entry_path()
    text = entry.read_text(encoding="utf-8")
    entry.write_text(
        text.replace(
            "Exec=",
            'Exec=/home/ahmed/.local/bin/research-rag --project "My Thesis"\n#',
            1,
        ).lstrip("#"),
        encoding="utf-8",
    )

    findings = desktop_entry_checks()

    assert findings, "an entry that names a project must be reported"
    assert "names a project" in findings[0].reason


def test_an_entry_in_the_ordinary_shape_is_not_reported(tmp_path: Path) -> None:
    _desktop_cli("install", "--desktop")

    assert not desktop_entry_checks()


def test_the_doctor_reads_only_the_entries_this_app_wrote(tmp_path: Path) -> None:
    entry = installation.desktop_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=Mine\nExec=/bin/true\n",
        encoding="utf-8",
    )

    assert not desktop_entry_checks()


def test_an_entry_whose_command_is_missing_is_named(tmp_path: Path) -> None:
    """The command is pointed at a path that was never there rather than removed from the
    machine: the entry is a claim about this reader's account, and a test has no business
    deleting a real file to find out whether a check works.
    """

    _desktop_cli("install", "--desktop")
    entry = installation.desktop_entry_path()
    entry.write_text(
        entry.read_text(encoding="utf-8").replace(
            f"Exec={installation.exec_argument(str(installation.account_command_path()))}",
            "Exec=/nowhere/research-rag",
        ),
        encoding="utf-8",
    )

    findings = desktop_entry_checks()

    assert findings and "not there" in findings[0].reason
