"""Reach this installation from a shell and from a desktop menu.

Two artifacts live outside any project, and both are the account's own: a link to
the console script in the account's binary directory, and one freedesktop entry
per project in the account's applications directory. Neither is project state, so
nothing here reads or writes `<project>/.research-rag`, and neither needs a
project to exist.

Every file written here carries the marker naming the command that wrote it, and
a file without that marker is reported and left alone. A person may replace the
command on `PATH` or hand-edit a menu entry, and an installer that overwrites
either without being told is the fault the rule prevents. `--uninstall` removes
exactly what `install` wrote and refuses the rest.

The console script is linked rather than copied so the checkout's `uv sync` and
an installed copy both keep working: a link resolves to whatever the interpreter
that wrote it provides, and a copy would go stale at the first sync.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_applications_dir, user_bin_dir, user_data_dir

from .config import CLI_COMMAND, ConfigurationError

CONSOLE_COMMAND = CLI_COMMAND
# The line every command this installation writes carries, so an uninstall can
# tell its own file from a script that happened to take the name.
WRAPPER_MARKER = "# Written by research-rag install."
PORTABLE_DIRECTORY = ".research-rag"
DESCRIPTOR_NAME = "project.json"
# One entry for the application, not one per project: the projects are inside it,
# and a menu listing four copies of the same name would be the same problem the
# per-project entries had.
DESKTOP_FILENAME = "research-rag.desktop"
ENTRY_MARKER = "# Written by research-rag install --desktop"
ENTRY_GROUP = "[Desktop Entry]"
ICON_RELATIVE = "icons/hicolor/scalable/apps/research-rag.svg"
CATEGORIES = "Utility;"
# The two commands that put this app on a machine's PATH in the first place.
# Neither is guessed at runtime: a refusal names them so a reader can run one.
ALTERNATIVES = (
    "uv tool install git+https://github.com/AhmedKishki/research-rag.git",
    "pipx install git+https://github.com/AhmedKishki/research-rag.git",
)

# The icon is a constant rather than a packaged asset: one file this command
# writes is one fewer thing to declare, install, and keep in step.
ICON_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">
  <rect x="4" y="5" width="27" height="38" rx="3" fill="#f6f3ec" stroke="#33454f" stroke-width="2"/>
  <g stroke="#8fa1ab" stroke-width="2" stroke-linecap="round">
    <path d="M10 14h15M10 20h15M10 26h10"/>
  </g>
  <circle cx="31" cy="31" r="9" fill="#2f6f8f" stroke="#f6f3ec" stroke-width="2"/>
  <path d="M37.5 37.5 43 43" stroke="#33454f" stroke-width="3" stroke-linecap="round"/>
</svg>
"""


@dataclass(frozen=True, slots=True)
class WriteReport:
    """What one account-level file holds now."""

    path: Path
    state: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "state": self.state,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DesktopEntry:
    """One menu entry this app wrote, and the command it runs."""

    path: Path
    executable: Path
    attached: bool
    names_a_project: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exec": str(self.executable),
            "attached": self.attached,
            "names_a_project": self.names_a_project,
        }


def account_bin_directory() -> Path:
    """The account directory a shell already searches for a command."""

    return Path(user_bin_dir())


def account_applications_directory() -> Path:
    """Where the desktop looks for application entries."""

    return Path(user_applications_dir())


def account_icon_path() -> Path:
    """The one icon every entry this app writes points at."""

    return Path(user_data_dir()) / ICON_RELATIVE


def console_script_path() -> Path:
    """The console script this interpreter provided."""

    return Path(sys.executable).expanduser().absolute().parent / CONSOLE_COMMAND


def _absolute(directory: Path | None, fallback: Path) -> Path:
    return Path(directory if directory is not None else fallback).expanduser()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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


_RESERVED = re.compile(r"""[\s"'\\%<>|&;$()*,?#~`]""")


def exec_argument(value: str) -> str:
    """One Exec argument as a desktop entry reads it."""

    if not _RESERVED.search(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def exec_tokens(value: str) -> list[str]:
    """The arguments one Exec line names, honouring the quoting above.

    `%%` is a literal percent sign rather than a field code, so a path holding
    one reads back as the path it was written from.
    """

    tokens: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if quoted and character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if character.isspace() and not quoted:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(character)
    if current:
        tokens.append("".join(current))
    return [token.replace("%%", "%") for token in tokens]


def recorded_project_name(project_root: Path) -> str:
    """The name a project directory recorded, or its directory name."""

    descriptor = project_root.expanduser().resolve() / ".research-rag" / "project.json"
    try:
        document = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return project_root.expanduser().resolve().name
    name = document.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return project_root.expanduser().resolve().name


def account_command_path() -> Path:
    """Where `research-rag install` puts the command a shell already looks for.

    The menu entry runs this rather than the interpreter's console script, so one
    address serves both the shell and the menu, and re-running `install` moves it.
    """

    return account_bin_directory() / CONSOLE_COMMAND


def desktop_entry_document(*, command: Path, icon: Path) -> str:
    """Render the one freedesktop entry for this installation.

    The entry asks for a terminal window and runs the app attached in it, naming
    no project: a click opens the workspace and, when the installation holds
    several projects, asks which one in that window. A click that left the app
    running with nothing to close is the one start path that must not exist, and
    an entry per project would be four copies of the same click.

    `StartupNotify` and `StartupWMClass` are absent because nothing here sets or
    honours them, and a value the desktop cannot be told about makes an entry it
    never resolves as started.
    """

    if not icon.is_absolute():
        raise ConfigurationError(
            f"The icon must be an absolute path for a desktop entry, not {icon}."
        )
    return "\n".join(
        (
            ENTRY_GROUP,
            ENTRY_MARKER,
            "Type=Application",
            f"Name={CLI_COMMAND}",
            "GenericName=Research knowledge base",
            "Comment=Open this installation's workspace in a terminal window.",
            f"Exec={exec_argument(str(command))}",
            f"Icon={icon}",
            "Terminal=true",
            f"Categories={CATEGORIES}",
            "",
        )
    )


def entry_is_ours(text: str) -> bool:
    """Whether one entry carries the marker this command writes."""

    return ENTRY_MARKER in text


def desktop_entry_from_text(path: Path, text: str) -> DesktopEntry | None:
    """The entry one file names, or None when this app did not write it."""

    if not entry_is_ours(text):
        return None
    command = ""
    for line in text.splitlines():
        if line.startswith("Exec="):
            command = line.removeprefix("Exec=")
            break
    tokens = exec_tokens(command)
    if not tokens:
        return None
    return DesktopEntry(
        path=path,
        executable=Path(tokens[0]),
        attached=any(line.strip() == "Terminal=true" for line in text.splitlines()),
        names_a_project="--project" in tokens,
    )


def desktop_entry_path(applications: Path | None = None) -> Path:
    """Where the one entry this app writes lives."""

    return _absolute(applications, account_applications_directory()) / DESKTOP_FILENAME


def desktop_entries(
    applications: Path | None = None,
) -> tuple[DesktopEntry, ...]:
    """Every entry this command wrote, the one it owns plus any older per-project ones."""

    directory = _absolute(applications, account_applications_directory())
    candidates = {directory / DESKTOP_FILENAME}
    with contextlib.suppress(OSError):
        candidates.update(directory.glob("research-rag-*.desktop"))
    found: list[DesktopEntry] = []
    for path in sorted(candidates):
        text = _read_text(path)
        if text is None:
            continue
        entry = desktop_entry_from_text(path, text)
        if entry is not None:
            found.append(entry)
    return tuple(found)


def write_icon(path: Path, *, force: bool = False) -> WriteReport:
    """Write the icon the entries point at, once, and never over a foreign file."""

    path = path.expanduser()
    existing = _read_text(path)
    if existing == ICON_SVG:
        return WriteReport(path, "unchanged", f"{path} already holds this icon.")
    if existing is not None and not force:
        raise ConfigurationError(
            f"{path} exists and is not the icon this app writes, so it was left "
            "alone. Read it, and pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ICON_SVG, encoding="utf-8", newline="\n")
    return WriteReport(
        path,
        "replaced" if existing is not None else "created",
        f"Wrote {path}.",
    )


def install_desktop_entry(
    *,
    applications: Path | None = None,
    icon: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write the one menu entry for this installation, and the icon it names.

    Nothing is written until the refusal has been passed: an entry this app did
    not write may have been hand-edited. No project is named and no project is
    read, because installing a menu entry must touch nothing a project owns.
    """

    directory = _absolute(applications, account_applications_directory())
    icon_path = _absolute(icon, account_icon_path())
    entry_path = directory / DESKTOP_FILENAME
    command = account_command_path()
    if not command.is_file():
        raise ConfigurationError(
            f"{command} is not there, so the menu entry would start nothing. Run "
            f"`{CLI_COMMAND} install` once, then run this again."
        )
    document = desktop_entry_document(command=command, icon=icon_path)
    existing = _read_text(entry_path)
    state = "created"
    if existing is not None:
        if not entry_is_ours(existing):
            if not force:
                raise ConfigurationError(
                    f"{entry_path} is a desktop entry this app did not write, so "
                    "it was left alone. Read it, and pass --force to replace it."
                )
            state = "replaced"
        elif existing == document:
            state = "unchanged"
    icon_report = write_icon(icon_path, force=force)
    if state in {"created", "replaced"}:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(document, encoding="utf-8", newline="\n")
    # An older build wrote one entry per project. They all start this same
    # application, and four copies of it in one menu is the problem they were, so
    # they go whether or not this run had anything else to write.
    superseded = [
        entry.path for entry in desktop_entries(directory) if entry.path != entry_path
    ]
    for path in superseded:
        path.unlink()
    return {
        "entry": WriteReport(
            entry_path, state, f"{entry_path} opens {CLI_COMMAND} in a terminal window."
        ).as_dict(),
        "icon": icon_report.as_dict(),
        "superseded": [str(path) for path in superseded],
    }


def uninstall_desktop_entry(
    *,
    applications: Path | None = None,
    icon: Path | None = None,
) -> dict[str, Any]:
    """Remove this installation's menu entry, and the icon it names.

    Entries written by an older build, one per project, go with it: they are the
    same application in the same menu.
    """

    directory = _absolute(applications, account_applications_directory())
    icon_path = _absolute(icon, account_icon_path())
    entry_path = directory / DESKTOP_FILENAME
    entries = desktop_entries(directory)
    existing = _read_text(entry_path)
    if existing is None:
        return {
            "entry": WriteReport(
                entry_path,
                "absent",
                f"There is no menu entry at {entry_path} to remove.",
            ).as_dict(),
            "icon": WriteReport(
                icon_path, "absent", f"There is no {icon_path}."
            ).as_dict(),
            "superseded": [],
        }
    if not entry_is_ours(existing):
        raise ConfigurationError(
            f"{entry_path} is a desktop entry this app did not write, so it was "
            "left alone. Delete it yourself if it is stale."
        )
    entry_path.unlink()
    superseded = [entry.path for entry in entries if entry.path != entry_path]
    for path in superseded:
        path.unlink()
    if icon_path.is_file():
        icon_path.unlink()
        icon_report = WriteReport(icon_path, "removed", f"Removed {icon_path}.")
    else:
        icon_report = WriteReport(icon_path, "absent", f"There is no {icon_path}.")
    return {
        "entry": WriteReport(
            entry_path, "removed", f"Removed the menu entry {entry_path}."
        ).as_dict(),
        "icon": icon_report.as_dict(),
        "superseded": [str(path) for path in superseded],
    }
