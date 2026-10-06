"""Hand a file to the desktop's own viewer, instead of sending it to a browser.

A source the workspace lists is already on this machine, so opening it is a
request to the desktop rather than a download. The app only runs on a loopback
port, so the machine that serves the workspace is the machine the reader is at.
Where there is no desktop to hand it to, a headless shell or a session over ssh,
the caller is told so and falls back to showing the file in the browser.

The path is one the caller has already authorised as a source file, and it is
passed as an argument, never through a shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..project.config import child_process_environment


class ViewerUnavailable(RuntimeError):
    """There is no desktop viewer to open this file with, and why."""


def _command(path: Path) -> list[str]:
    if sys.platform == "darwin":
        return ["open", str(path)]
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise ViewerUnavailable(
                "This app has no desktop session to open a file in, so it is "
                "shown in the browser instead."
            )
        opener = shutil.which("xdg-open")
        if opener is None:
            raise ViewerUnavailable(
                "xdg-open is not installed, so the file is shown in the browser "
                "instead."
            )
        return [opener, str(path)]
    raise ViewerUnavailable(
        f"Opening in a desktop viewer is not supported on {sys.platform}, so the "
        "file is shown in the browser instead."
    )


def open_in_default_viewer(path: Path) -> str:
    """Open `path` in the desktop's default program and return that program's name."""

    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
        return "the default program"
    command = _command(path)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=child_process_environment(),
        )
    except OSError as exc:
        raise ViewerUnavailable(f"{command[0]} could not start: {exc}") from exc
    # The opener exits once it has handed the file over; reaping it keeps it from
    # lingering as a zombie of this long-running app.
    threading.Thread(target=process.wait, daemon=True).start()
    return Path(command[0]).name
