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

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_applications_dir, user_bin_dir, user_data_dir

from .config import CLI_COMMAND, ConfigurationError, project_command
from .launcher import LINK_NAME

CONSOLE_COMMAND = CLI_COMMAND
PORTABLE_DIRECTORY = ".research-rag"
DESCRIPTOR_NAME = "project.json"
DESKTOP_PREFIX = "research-rag-"
DESKTOP_SUFFIX = ".desktop"
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
    """One menu entry this app wrote, and the project root its Exec names."""

    path: Path
    launcher: Path

    @property
    def project_root(self) -> Path:
        return self.launcher.parent

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exec": f"{self.launcher} --open",
            "project_root": str(self.project_root),
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


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _console_state(destination: Path, source: Path) -> str:
    """One word for what the account's command path already holds.

    `absent`, `current`, and `foreign` are the three answers, and only the last
    one may not be overwritten without `--force`.
    """

    if not source.is_file() or not os.access(source, os.X_OK):
        raise ConfigurationError(
            f"This interpreter has no executable console script at {source}, so "
            f"there is nothing to put on PATH. Install the command itself with "
            f"`{ALTERNATIVES[0]}` or `{ALTERNATIVES[1]}`, then run this again."
        )
    if destination == source or _same_file(destination, source):
        return "current"
    if destination.is_symlink():
        try:
            target = Path(os.readlink(destination)).expanduser()
        except OSError:
            return "foreign"
        return "current" if _same_file(target, source) else "foreign"
    if destination.exists():
        return "foreign"
    return "absent"


def install_console_entry(
    *,
    bin_directory: Path | None = None,
    source: Path | None = None,
    force: bool = False,
) -> WriteReport:
    """Link this interpreter's console script into the account's binary directory.

    A second run changes nothing and says the command is already installed.
    """

    destination = _absolute(bin_directory, account_bin_directory()) / CONSOLE_COMMAND
    script = Path(source).expanduser().absolute() if source else console_script_path()
    state = _console_state(destination, script)
    if state == "current":
        return WriteReport(
            destination,
            "already_installed",
            f"{destination} already runs {script}; nothing was changed.",
        )
    if state == "foreign":
        if not force:
            raise ConfigurationError(
                f"{destination} exists and is not a link to this interpreter's "
                f"console script, so it was left alone. Read it, and pass "
                f"--force to replace it with a link to {script}."
            )
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(script)
    return WriteReport(
        destination,
        "replaced" if state == "foreign" else "created",
        f"{destination} now runs {script}.",
    )


def uninstall_console_entry(
    *,
    bin_directory: Path | None = None,
    source: Path | None = None,
) -> WriteReport:
    """Remove the link this command wrote, and nothing else."""

    destination = _absolute(bin_directory, account_bin_directory()) / CONSOLE_COMMAND
    script = Path(source).expanduser().absolute() if source else console_script_path()
    if not destination.is_symlink():
        if destination.exists():
            raise ConfigurationError(
                f"{destination} is not a symlink, so this command did not create "
                "it and will not delete it. Remove it yourself if it is stale."
            )
        return WriteReport(
            destination, "absent", f"There is nothing at {destination} to remove."
        )
    target = Path(os.readlink(destination)).expanduser()
    if not _same_file(target, script):
        raise ConfigurationError(
            f"{destination} links to {target} rather than to {script}, so this "
            "command did not create it and will not delete it. Remove it "
            "yourself if it is stale."
        )
    destination.unlink()
    return WriteReport(
        destination, "removed", f"Removed the link {destination} -> {target}."
    )


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


def desktop_entry_filename(project_name: str) -> str:
    """The file one project's entry takes, readable beside several others."""

    slug = re.sub(r"[^a-z0-9]+", "-", project_name.casefold()).strip("-")
    return f"{DESKTOP_PREFIX}{slug or 'project'}{DESKTOP_SUFFIX}"


def recorded_project_name(project_root: Path) -> str:
    """The name a project recorded, or its directory name when it recorded none."""

    descriptor = project_root / PORTABLE_DIRECTORY / DESCRIPTOR_NAME
    text = _read_text(descriptor)
    if text is not None:
        try:
            document = json.loads(text)
        except ValueError:
            document = None
        name = document.get("name") if isinstance(document, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return project_root.name


def project_launcher(project_root: Path) -> Path:
    """The project-root launcher a menu entry runs."""

    return project_root / LINK_NAME


def desktop_entry_document(
    *,
    project_name: str,
    launcher: Path,
    icon: Path,
) -> str:
    """Render one freedesktop entry.

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
            f"Name={project_name}",
            "GenericName=Research knowledge base",
            f"Comment=Open the workspace for {project_name}.",
            f"Exec={exec_argument(str(launcher))} --open",
            f"Icon={icon}",
            "Terminal=false",
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
    for line in text.splitlines():
        if not line.startswith("Exec="):
            continue
        tokens = exec_tokens(line.removeprefix("Exec="))
        if tokens:
            return DesktopEntry(path, Path(tokens[0]))
    return None


def desktop_entries(
    applications: Path | None = None,
) -> tuple[DesktopEntry, ...]:
    """Every menu entry this command wrote, in file-name order."""

    directory = _absolute(applications, account_applications_directory())
    try:
        candidates = sorted(directory.glob(f"{DESKTOP_PREFIX}*{DESKTOP_SUFFIX}"))
    except OSError:
        return ()
    found: list[DesktopEntry] = []
    for path in candidates:
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
    project_root: Path,
    project_name: str | None = None,
    applications: Path | None = None,
    icon: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one menu entry for one project, and the icon it points at.

    Nothing is written until both refusals have been passed: a project with no
    launcher would produce an entry that opens nothing, and an entry this app did
    not write may have been hand-edited.
    """

    root = project_root.expanduser().resolve()
    name = project_name or recorded_project_name(root)
    launcher = project_launcher(root)
    if not launcher.is_file():
        raise ConfigurationError(
            f"{root} has no workspace launcher at {launcher}, so an entry that "
            f"opens nothing would be written. Run `{project_command(root, 'init')}` "
            "once, then run this again."
        )
    directory = _absolute(applications, account_applications_directory())
    icon_path = _absolute(icon, account_icon_path())
    entry_path = directory / desktop_entry_filename(name)
    document = desktop_entry_document(
        project_name=name, launcher=launcher, icon=icon_path
    )
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
        else:
            mine = desktop_entry_from_text(entry_path, existing)
            if mine is not None and mine.project_root != root:
                raise ConfigurationError(
                    f"{entry_path} is this app's entry for {mine.project_root}, and "
                    f"{root} reduces to the same file name. Record a distinct name "
                    f"with `{project_command(root, 'init', '--name', 'NAME')}`."
                )
            if existing == document:
                state = "unchanged"
    icon_report = write_icon(icon_path, force=force)
    if state in {"created", "replaced"}:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(document, encoding="utf-8", newline="\n")
    # A renamed project has a new file name, and one entry per project is the
    # rule: this app's own entry for this same root goes with it.
    superseded = [
        candidate.path
        for candidate in desktop_entries(directory)
        if candidate.project_root == root and candidate.path != entry_path
    ]
    for path in superseded:
        path.unlink()
    entry_report = WriteReport(
        entry_path,
        state,
        f"{entry_path} opens {launcher} --open.",
    )
    return {
        "project_root": str(root),
        "project_name": name,
        "entry": entry_report.as_dict(),
        "icon": icon_report.as_dict(),
        "superseded": [str(path) for path in superseded],
    }


def uninstall_desktop_entry(
    *,
    project_root: Path,
    project_name: str | None = None,
    applications: Path | None = None,
    icon: Path | None = None,
) -> dict[str, Any]:
    """Remove one project's menu entry, and the icon when nothing else needs it.

    A project root that has been deleted still has its entry, so the entry is
    also found through the root its own Exec names.
    """

    root = project_root.expanduser().resolve()
    name = project_name or recorded_project_name(root)
    directory = _absolute(applications, account_applications_directory())
    icon_path = _absolute(icon, account_icon_path())
    entry_path = directory / desktop_entry_filename(name)
    existing = _read_text(entry_path)
    if existing is None:
        for candidate in desktop_entries(directory):
            if candidate.project_root == root:
                entry_path = candidate.path
                existing = _read_text(entry_path)
                break
    if existing is None:
        entry_report = WriteReport(
            entry_path, "absent", f"There is no menu entry for {root} to remove."
        )
    elif not entry_is_ours(existing):
        raise ConfigurationError(
            f"{entry_path} is a desktop entry this app did not write, so it was "
            "left alone. Delete it yourself if it is stale."
        )
    else:
        entry_path.unlink()
        entry_report = WriteReport(
            entry_path, "removed", f"Removed the menu entry {entry_path}."
        )
    remaining = [
        entry for entry in desktop_entries(directory) if entry.path != entry_path
    ]
    icon_report: WriteReport
    if remaining:
        icon_report = WriteReport(
            icon_path,
            "kept",
            f"{icon_path} is still used by {len(remaining)} menu entries.",
        )
    elif icon_path.is_file():
        icon_path.unlink()
        icon_report = WriteReport(icon_path, "removed", f"Removed {icon_path}.")
    else:
        icon_report = WriteReport(icon_path, "absent", f"There is no {icon_path}.")
    return {
        "project_root": str(root),
        "project_name": name,
        "entry": entry_report.as_dict(),
        "icon": icon_report.as_dict(),
    }
