"""The one freedesktop entry that puts this installation in the desktop menu.

The entry is the account's own rather than any project's: it names no project, so
a click opens the workspace and asks which one when this installation holds
several. Nothing here reads or writes `<project>/.research-rag`, and nothing needs
a project to exist.

Every file written here carries the marker naming the command that wrote it, and a
file without that marker is reported and left alone. A person may hand-edit a menu
entry or replace the icon it names, and an installer that overwrites either without
being told is the fault the rule prevents. `--uninstall` removes exactly what
`install` wrote and refuses the rest, which is why these refusals raise
`ConfigurationError` from `config.py`: `install` resolves no project, so a foreign
file here is the account's own configuration, and that is what the class names.

The console script moved to `install_entry.py`, which owns the wrapper, where it
lands on the `PATH`, and the report both halves return. The `ALTERNATIVES` copy
that stood here was never read here; `tool_ownership.py` owns those commands.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_applications_dir, user_data_dir

from ..project.config import CLI_COMMAND, ConfigurationError

# The command line imports these from this module, so they stay bound here, and
# `tests/test_installation.py` reads two of them as its attributes.
from .install_entry import (  # noqa: F401
    WRAPPER_MARKER,
    WriteReport,
    _absolute,
    _read_text,
    account_bin_directory,
    account_command_path,
    install_console_entry,
    uninstall_console_entry,
)

# One entry for the application, not one per project: the projects are inside it,
# and a menu listing four copies of the same name would be the same problem the
# per-project entries had.
DESKTOP_FILENAME = "research-rag.desktop"
ENTRY_MARKER = "# Written by research-rag install --desktop"
ENTRY_GROUP = "[Desktop Entry]"
ICON_RELATIVE = "icons/hicolor/scalable/apps/research-rag.svg"
CATEGORIES = "Utility;"

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
class DesktopEntry:
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


def account_applications_directory() -> Path:
    """Where the desktop looks for application entries."""

    return Path(user_applications_dir())


def account_icon_path() -> Path:
    """The one icon every entry this app writes points at."""

    return Path(user_data_dir()) / ICON_RELATIVE


_RESERVED = re.compile(r"""[\s"'\\%<>|&;$()*,?#~`]""")


def exec_argument(value: str) -> str:
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
        else:
            # This app's own entry that no longer matches what it writes, which is
            # what an update that moved the command leaves behind. It is replaced,
            # and saying so is the difference between a reader who knows the entry
            # moved and one who reads "created" over a file that already existed.
            state = "replaced"
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
