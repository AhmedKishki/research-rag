"""The one command that puts this installation on the account's `PATH`.

A shell script written into the account's binary directory, and `wrapper_document`
is where the reason it is not a link is stated. Nothing here reads a project and
nothing here writes one: `install` needs no project, and resolving one writes its
portable state.

A file this command wrote carries the marker naming the command that wrote it, and
a file without that marker is reported and left alone. A person may put their own
script on the `PATH` under this name, and an uninstall that removed it would take
a working command away from them.

`wrapper_document`, `install_console_entry`, `uninstall_console_entry`,
`console_script_path`, `account_command_path`, `account_bin_directory`, the
marker, and the report and file helpers both halves share moved here from
`installation.py`, which keeps the desktop entry and imports them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_bin_dir

from .config import CLI_COMMAND, ConfigurationError

CONSOLE_COMMAND = CLI_COMMAND
# The line every command this installation writes carries, so an uninstall can
# tell its own file from a script that happened to take the name. The text is
# fixed rather than built from the command name: an install already on a machine
# carries it, and a reworded marker would leave that command unremovable.
WRAPPER_MARKER = "# Written by research-rag install."


@dataclass(frozen=True, slots=True)
class WriteReport:
    path: Path
    state: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "state": self.state,
            "message": self.message,
        }


def _absolute(directory: Path | None, fallback: Path) -> Path:
    return Path(directory if directory is not None else fallback).expanduser()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def account_bin_directory() -> Path:
    """The account directory a shell already searches for a command."""

    return Path(user_bin_dir())


def account_command_path() -> Path:
    """Where `research-rag install` puts the command a shell already looks for.

    The menu entry runs this rather than the interpreter's console script, so one
    address serves both the shell and the menu, and re-running `install` moves it.
    """

    return account_bin_directory() / CONSOLE_COMMAND


def console_script_path() -> Path:
    """The console script this interpreter provided."""

    return Path(sys.executable).expanduser().absolute().parent / CONSOLE_COMMAND


def wrapper_document(interpreter: Path) -> str:
    """The one command that puts this installation on the account's `PATH`.

    A shell script rather than a link to the interpreter's console script, because
    that script is one file among thousands inside a virtual environment a
    `uv sync` will happily recreate: a link to it breaks the moment the
    environment is rebuilt, and the reader is left with "No such file or
    directory" for a command they installed. This names the interpreter, which
    survives, and asks it for the module, which is where the command lives.
    """

    return (
        "#!/bin/sh\n"
        "# Written by research-rag install. Runs this installation from wherever\n"
        "# you are; `research-rag install --uninstall` removes this file.\n"
        f'exec "{interpreter}" -m research_rag "$@"\n'
    )


def install_console_entry(
    *,
    bin_directory: Path | None = None,
    interpreter: Path | None = None,
    force: bool = False,
) -> WriteReport:
    """Put this installation on the account's `PATH`.

    A second run changes nothing and says the command is already installed.
    """

    destination = _absolute(bin_directory, account_bin_directory()) / CONSOLE_COMMAND
    running = (
        Path(interpreter).expanduser().absolute()
        if interpreter
        else Path(sys.executable).expanduser().absolute()
    )
    document = wrapper_document(running)
    existing = _read_text(destination)
    state = "created"
    if existing is not None:
        if existing != document:
            if not force:
                raise ConfigurationError(
                    f"{destination} exists and is not the command this "
                    "installation writes, so it was left alone. Read it, and pass "
                    f"--force to replace it with one that runs {running}."
                )
            state = "replaced"
        else:
            state = "already_installed"
    if state != "already_installed":
        if destination.is_dir():
            raise ConfigurationError(
                f"{destination} is a directory, so it was left alone. Remove it, "
                "and run this again."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8", newline="\n")
        destination.chmod(0o755)
    return WriteReport(
        destination,
        state,
        f"{destination} runs this installation through {running}.",
    )


def uninstall_console_entry(
    *,
    bin_directory: Path | None = None,
) -> WriteReport:
    """Remove the command this installation wrote, and nothing else.

    Only a file carrying this command's own marker is removed: a script someone
    else placed on the `PATH` is theirs, and a name collision here would make a
    working command disappear.
    """

    destination = _absolute(bin_directory, account_bin_directory()) / CONSOLE_COMMAND
    existing = _read_text(destination)
    if existing is None:
        return WriteReport(
            destination, "absent", f"There is nothing at {destination} to remove."
        )
    if WRAPPER_MARKER not in existing:
        raise ConfigurationError(
            f"{destination} is not the command this installation writes, so this "
            "command did not create it and will not delete it. Remove it yourself "
            "if it is stale."
        )
    destination.unlink()
    return WriteReport(destination, "removed", f"Removed the command {destination}.")


__all__ = [
    "CONSOLE_COMMAND",
    "WRAPPER_MARKER",
    "WriteReport",
    "account_bin_directory",
    "account_command_path",
    "console_script_path",
    "install_console_entry",
    "uninstall_console_entry",
    "wrapper_document",
]
