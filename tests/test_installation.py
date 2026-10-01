"""Reaching the installation: the command on `PATH`, and the desktop menu entry.

Both write into the account's own directories, which the account fixture points
at a throwaway tree, so nothing here touches the reader's real command path or
their menu. Every refusal is tested as well as every write: an installer that
replaces a file it did not create is the fault this command must not have.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import research_rag.surfaces.cli as cli_module
from research_rag import installation
from research_rag.config import ConfigurationError
from research_rag.doctor import desktop_entry_checks
from research_rag.surfaces.cli import _init, _parser

FAKE_SCRIPT = "#!/bin/sh\n# a console script standing in for the real one\n"


def _args(*arguments: str) -> Any:
    return _parser().parse_args(list(arguments))


def _console_script(tmp_path: Path, name: str = "console") -> Path:
    """An executable file that can stand for an interpreter's console script."""

    directory = tmp_path / name / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "research-rag"
    script.write_text(FAKE_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    return script


def _initialised(project: Path) -> Path:
    _init(_args("--project-root", str(project), "init", "--name", "My Thesis"))
    return project


def _desktop_cli(*arguments: str) -> dict[str, Any]:
    result = asyncio.run(cli_module._run(_args(*arguments)))
    assert result.payload is not None
    return result.payload


def test_the_command_is_linked_where_a_shell_already_looks(tmp_path: Path) -> None:
    """The link goes in the account's binary directory, and points at the script."""

    bin_directory = tmp_path / "bin"
    source = _console_script(tmp_path)

    report = installation.install_console_entry(
        bin_directory=bin_directory, source=source
    )

    destination = bin_directory / "research-rag"
    assert destination.is_symlink()
    assert destination.readlink() == source
    assert report.state == "created"
    assert str(destination) in report.message
    assert str(source) in report.message


def test_installing_twice_changes_nothing(tmp_path: Path) -> None:
    """A second run is the answer, not a rewrite of a working link."""

    bin_directory = tmp_path / "bin"
    source = _console_script(tmp_path)
    installation.install_console_entry(bin_directory=bin_directory, source=source)
    before = (bin_directory / "research-rag").lstat()

    report = installation.install_console_entry(
        bin_directory=bin_directory, source=source
    )

    assert report.state == "already_installed"
    assert "nothing was changed" in report.message
    assert (bin_directory / "research-rag").lstat().st_mtime_ns == before.st_mtime_ns


def test_a_foreign_command_on_the_path_is_reported_and_left_alone(
    tmp_path: Path,
) -> None:
    """Someone else's script is not this command's to replace."""

    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    foreign = bin_directory / "research-rag"
    foreign.write_text("#!/bin/sh\necho another tool\n", encoding="utf-8")
    foreign.chmod(0o755)

    with pytest.raises(ConfigurationError, match="left alone"):
        installation.install_console_entry(
            bin_directory=bin_directory, source=_console_script(tmp_path)
        )

    assert foreign.read_text(encoding="utf-8") == "#!/bin/sh\necho another tool\n"
    assert not foreign.is_symlink()


def test_a_missing_console_script_names_the_two_tools_that_install_one(
    tmp_path: Path,
) -> None:
    """The command cannot be linked from nothing, and says what to run instead."""

    with pytest.raises(ConfigurationError) as refused:
        installation.install_console_entry(
            bin_directory=tmp_path / "bin", source=tmp_path / "absent" / "research-rag"
        )

    assert "uv tool install" in str(refused.value)
    assert "pipx install" in str(refused.value)


def test_uninstall_removes_only_the_link_this_command_wrote(tmp_path: Path) -> None:
    bin_directory = tmp_path / "bin"
    source = _console_script(tmp_path)
    installation.install_console_entry(bin_directory=bin_directory, source=source)

    report = installation.uninstall_console_entry(
        bin_directory=bin_directory, source=source
    )

    assert report.state == "removed"
    assert not (bin_directory / "research-rag").exists()
    # A second run has nothing to say beyond that there is nothing there.
    assert (
        installation.uninstall_console_entry(
            bin_directory=bin_directory, source=source
        ).state
        == "absent"
    )


def test_uninstall_refuses_a_command_that_is_not_this_commands_link(
    tmp_path: Path,
) -> None:
    """A file the command did not write is the reader's, whatever it holds."""

    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    elsewhere = tmp_path / "somewhere" / "research-rag"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(FAKE_SCRIPT, encoding="utf-8")
    foreign = bin_directory / "research-rag"
    foreign.symlink_to(elsewhere)

    with pytest.raises(ConfigurationError, match="did not create"):
        installation.uninstall_console_entry(
            bin_directory=bin_directory, source=_console_script(tmp_path)
        )

    assert foreign.is_symlink()


def test_uninstall_refuses_a_regular_file_it_never_wrote(tmp_path: Path) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    foreign = bin_directory / "research-rag"
    foreign.write_text(FAKE_SCRIPT, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not a symlink"):
        installation.uninstall_console_entry(
            bin_directory=bin_directory, source=_console_script(tmp_path)
        )

    assert foreign.is_file()


def test_a_project_path_with_spaces_survives_the_round_trip(tmp_path: Path) -> None:
    """The quoting a desktop entry needs, and the reading of it back."""

    path = '/home/a b/"c"/100%'

    assert installation.exec_argument(path) == '"/home/a b/\\"c\\"/100%%"'
    assert installation.exec_tokens(installation.exec_argument(path)) == [path]


def test_an_entry_with_a_quoted_exec_is_read_back_as_one_path() -> None:
    path = "/home/ahmed/My Thesis/open-research-rag-ui.sh"

    entry = installation.desktop_entry_from_text(
        Path("/tmp/research-rag-my-thesis.desktop"),
        f"{installation.ENTRY_MARKER}\nExec={installation.exec_argument(path)} --open\n",
    )

    assert entry is not None
    assert entry.launcher == Path(path)
    assert entry.project_root == Path("/home/ahmed/My Thesis")


def test_a_renamed_project_leaves_one_entry(tmp_path: Path) -> None:
    """One entry per project, and a rename moves it rather than adding one."""

    project = _initialised(tmp_path / "thesis")
    _desktop_cli("--project-root", str(project), "install", "--desktop")
    first = installation.desktop_entries()[0]
    _init(_args("--project-root", str(project), "init", "--name", "The Thesis"))

    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")

    entries = installation.desktop_entries()
    assert [entry.path.name for entry in entries] == ["research-rag-the-thesis.desktop"]
    assert payload["desktop"]["superseded"] == [str(first.path)]
    assert not first.path.exists()


def test_the_console_command_needs_no_project(tmp_path: Path) -> None:
    """`install` answers for the installation, so it resolves no project."""

    payload = _desktop_cli("install")

    assert payload["console"]["state"] in {"created", "already_installed"}
    assert payload["console"]["path"].endswith("research-rag")


def test_the_desktop_entry_names_the_project_and_its_launcher(tmp_path: Path) -> None:
    """One entry, the fields a menu needs, and an `Exec` that opens the workspace."""

    project = _initialised(tmp_path / "thesis")

    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")

    applications = Path(payload["desktop"]["entry"]["path"]).parent
    entry = applications / installation.desktop_entry_filename("My Thesis")
    assert entry.read_text(encoding="utf-8").splitlines()[0] == "[Desktop Entry]"
    fields = dict(
        line.split("=", 1)
        for line in entry.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "["))
    )
    assert fields["Type"] == "Application"
    assert fields["Terminal"] == "false"
    assert fields["Name"] == "My Thesis"
    assert fields["Comment"] == "Open the workspace for My Thesis."
    assert fields["Exec"] == f"{project / installation.LINK_NAME} --open"
    assert "Utility" in fields["Categories"]
    assert Path(fields["Icon"]).is_absolute()
    assert Path(fields["Icon"]).is_file()
    # Nothing here sets these, and a value the desktop is never told about makes
    # an entry it never resolves as started.
    assert "StartupNotify" not in fields
    assert "StartupWMClass" not in fields


@pytest.mark.skipif(
    shutil.which("desktop-file-validate") is None,
    reason="desktop-file-validate is not installed",
)
def test_the_entry_is_valid_by_the_desktops_own_validator(tmp_path: Path) -> None:
    """A menu entry the desktop cannot parse is not an entry."""

    project = _initialised(tmp_path / "My Thesis")
    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")

    result = subprocess.run(
        ["desktop-file-validate", payload["desktop"]["entry"]["path"]],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_icon_is_written_by_the_command_as_a_constant(tmp_path: Path) -> None:
    project = _initialised(tmp_path / "thesis")

    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")

    icon = Path(payload["desktop"]["icon"]["path"])
    assert icon.read_text(encoding="utf-8") == installation.ICON_SVG
    assert icon.suffix == ".svg"
    assert "icons" in icon.parts and "scalable" in icon.parts


def test_installing_the_entry_twice_writes_nothing_the_second_time(
    tmp_path: Path,
) -> None:
    project = _initialised(tmp_path / "thesis")
    _desktop_cli("--project-root", str(project), "install", "--desktop")
    entry = installation.desktop_entries()[0]
    before = entry.path.stat().st_mtime_ns

    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")

    assert payload["desktop"]["entry"]["state"] == "unchanged"
    assert payload["desktop"]["icon"]["state"] == "unchanged"
    assert entry.path.stat().st_mtime_ns == before


def test_a_desktop_entry_this_app_did_not_write_is_left_alone(
    tmp_path: Path,
) -> None:
    """A hand-written or hand-edited entry is the reader's file."""

    project = _initialised(tmp_path / "thesis")
    applications = installation.account_applications_directory()
    applications.mkdir(parents=True, exist_ok=True)
    entry_path = applications / installation.desktop_entry_filename("My Thesis")
    hand_written = "[Desktop Entry]\nType=Application\nName=My Thesis\n"
    entry_path.write_text(hand_written, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="did not write"):
        _desktop_cli("--project-root", str(project), "install", "--desktop")

    assert entry_path.read_text(encoding="utf-8") == hand_written
    assert installation.desktop_entries() == ()


def test_force_replaces_a_foreign_entry_but_uninstall_will_not_remove_it(
    tmp_path: Path,
) -> None:
    """`--force` says yes to writing; removal still needs proof of ownership."""

    project = _initialised(tmp_path / "thesis")
    applications = installation.account_applications_directory()
    applications.mkdir(parents=True, exist_ok=True)
    entry_path = applications / installation.desktop_entry_filename("My Thesis")
    entry_path.write_text("[Desktop Entry]\nName=My Thesis\n", encoding="utf-8")

    payload = _desktop_cli(
        "--project-root", str(project), "install", "--desktop", "--force"
    )

    assert payload["desktop"]["entry"]["state"] == "replaced"
    assert installation.entry_is_ours(entry_path.read_text(encoding="utf-8"))
    entry_path.write_text("[Desktop Entry]\nName=My Thesis\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="did not write"):
        _desktop_cli(
            "--project-root", str(project), "install", "--desktop", "--uninstall"
        )
    assert entry_path.is_file()


def test_two_projects_that_reduce_to_one_file_name_are_refused(tmp_path: Path) -> None:
    first = _initialised(tmp_path / "My Thesis")
    second = tmp_path / "copy"
    _init(_args("--project-root", str(second), "init", "--name", "my  thesis"))
    _desktop_cli("--project-root", str(first), "install", "--desktop")

    with pytest.raises(ConfigurationError, match="same file name"):
        _desktop_cli("--project-root", str(second), "install", "--desktop")

    assert installation.desktop_entries()[0].project_root == first


def test_the_entry_names_no_project_state_that_writing_it_creates(
    tmp_path: Path,
) -> None:
    """Installing an entry reads a project; it must not write one."""

    project = tmp_path / "thesis"
    project.mkdir()
    (project / "sources").mkdir()
    (project / installation.LINK_NAME).write_text("#!/bin/sh\n", encoding="utf-8")
    before = sorted(path.name for path in project.iterdir())

    _desktop_cli("--project-root", str(project), "install", "--desktop")

    assert sorted(path.name for path in project.iterdir()) == before
    assert not (project / ".research-rag").exists()


def test_a_project_without_a_launcher_is_refused_by_name(tmp_path: Path) -> None:
    """An entry that opens nothing is not written, and the remedy is named."""

    project = tmp_path / "not-a-project"
    project.mkdir()

    with pytest.raises(ConfigurationError) as refused:
        _desktop_cli("--project-root", str(project), "install", "--desktop")

    assert "open-research-rag-ui.sh" in str(refused.value)
    assert "init" in str(refused.value)
    assert installation.desktop_entries() == ()


def test_uninstall_removes_the_entry_and_the_icon_it_wrote(tmp_path: Path) -> None:
    project = _initialised(tmp_path / "thesis")
    payload = _desktop_cli("--project-root", str(project), "install", "--desktop")
    icon = Path(payload["desktop"]["icon"]["path"])

    removed = _desktop_cli(
        "--project-root", str(project), "install", "--desktop", "--uninstall"
    )

    assert removed["desktop"]["entry"]["state"] == "removed"
    assert removed["desktop"]["icon"]["state"] == "removed"
    assert not icon.exists()
    assert installation.desktop_entries() == ()


def test_the_icon_stays_while_another_project_has_an_entry(tmp_path: Path) -> None:
    """One icon serves every entry, so it goes only with the last one."""

    first = _initialised(tmp_path / "first")
    second = tmp_path / "second"
    _init(_args("--project-root", str(second), "init", "--name", "second"))
    for project in (first, second):
        _desktop_cli("--project-root", str(project), "install", "--desktop")

    payload = _desktop_cli(
        "--project-root", str(first), "install", "--desktop", "--uninstall"
    )

    assert payload["desktop"]["icon"]["state"] == "kept"
    assert len(installation.desktop_entries()) == 1


def test_removing_the_project_is_followed_by_the_doctor_naming_the_entry(
    tmp_path: Path,
) -> None:
    """The entry survives its project, and the doctor says it opens nothing."""

    project = _initialised(tmp_path / "thesis")
    _desktop_cli("--project-root", str(project), "install", "--desktop")
    for path in sorted(project.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    project.rmdir()

    findings = desktop_entry_checks()

    assert [check.name for check in findings] == ["desktop_entry"]
    assert findings[0].state == "warn"
    assert str(project) in findings[0].reason
    assert "install --desktop --uninstall" in str(findings[0].remedy_command)


def test_the_doctor_reads_only_the_entries_this_app_wrote(tmp_path: Path) -> None:
    """A menu full of other applications is not this command's business."""

    applications = installation.account_applications_directory()
    applications.mkdir(parents=True, exist_ok=True)
    (applications / "research-rag-elsewhere.desktop").write_text(
        "[Desktop Entry]\nExec=/nowhere/open-research-rag-ui.sh --open\n",
        encoding="utf-8",
    )

    assert desktop_entry_checks() == []


def test_an_entry_whose_project_is_still_there_is_not_reported(
    tmp_path: Path,
) -> None:
    project = _initialised(tmp_path / "thesis")
    _desktop_cli("--project-root", str(project), "install", "--desktop")

    assert desktop_entry_checks() == []
