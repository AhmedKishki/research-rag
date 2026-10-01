"""Version reporting for the running app, its environment, and the shared UI."""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as _distribution_version
from pathlib import Path

DISTRIBUTION_NAME = "research-rag"
UI_DISTRIBUTION_NAME = "ui-ultra-rag-mcp"
UNKNOWN_VERSION = "0.0.0+unknown"


def distribution_version(name: str) -> str | None:
    """Return an installed distribution's version, or None when it is not installed."""

    try:
        return _distribution_version(name)
    except PackageNotFoundError:
        return None


def distribution_files(name: str) -> Path:
    """Return an installed distribution's metadata directory."""

    return Path(str(distribution(name)._path))


# Read once, when this module is first imported: a running process keeps naming
# the version it started with, so an update that only changed the environment is
# visible as a difference instead of being reported as already current.
APP_VERSION = distribution_version(DISTRIBUTION_NAME) or UNKNOWN_VERSION


def installed_version() -> str:
    """Return the version installed in the environment right now."""

    return distribution_version(DISTRIBUTION_NAME) or UNKNOWN_VERSION


def ui_version() -> str | None:
    """Return the installed shared-UI version, or None when it is not installed."""

    return distribution_version(UI_DISTRIBUTION_NAME)


def restart_required() -> bool:
    """Whether this process started with a different version than is installed now."""

    return installed_version() != APP_VERSION


def checkout_revision() -> str | None:
    """Return the commit the loaded package came from, or None when unknowable.

    A version number cannot tell two checkouts apart: two copies of the same
    version may sit 63 commits apart, and only the commit identifies which one is
    answering.

    The installed distribution's ``direct_url.json`` records the commit when it
    was installed from a checkout, the only case worth reporting.
    """

    try:
        recorded = distribution_files(DISTRIBUTION_NAME)
    except Exception:  # noqa: BLE001 - a broken environment must not raise here.
        return None
    direct_url = recorded / "direct_url.json"
    if not direct_url.is_file():
        return None
    try:
        payload = json.loads(direct_url.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    commit = payload.get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def working_tree_revision() -> str | None:
    """Return the commit checked out beside this process, or None.

    The loaded package is one directory and the checkout is another, and the
    difference between them is the fault this reports: a server answering from a
    copy of the code that is no longer the one on disk.
    """

    package_root = Path(__file__).resolve().parents[2]
    for candidate in (package_root, *package_root.parents):
        head = candidate / ".git"
        if not head.exists():
            continue
        try:
            revision = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = revision.stdout.strip()
        return commit or None
    return None


def checkout_drift() -> str | None:
    """Return a description when the running code is not the checked-out code.

    Two checkouts of this package are the reported cause of a server that
    answered with metadata it could not have written: the process was serving a
    copy of the code while the directory beside it had moved on. A version
    comparison cannot see that, because both copies report the same version.
    The install location can, because the copy that answers and the copy on disk
    resolve to different paths.
    """

    installed_root = install_origin()
    if installed_root is None:
        return None
    checkout_root = nearest_checkout()
    if checkout_root is None or checkout_root == installed_root:
        return None
    return (
        f"The running app was installed from {installed_root} but the "
        f"checked-out copy is {checkout_root}. The two differ, so the answers "
        "come from code that is not the code on disk."
    )


def install_origin() -> Path | None:
    """Return the directory this package was installed from, when it says."""

    try:
        recorded = distribution_files(DISTRIBUTION_NAME)
    except Exception:  # noqa: BLE001 - a broken environment must not raise here.
        return None
    direct_url = recorded / "direct_url.json"
    if not direct_url.is_file():
        return None
    try:
        payload = json.loads(direct_url.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = payload.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    return Path(url.removeprefix("file://")).resolve()


def nearest_checkout() -> Path | None:
    """Return the nearest git checkout at or above the running package."""

    package_root = Path(__file__).resolve().parents[2]
    for candidate in (package_root, *package_root.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def version_block() -> dict[str, object]:
    """Describe the running, installed, and UI versions for `status`."""

    return {
        "app": APP_VERSION,
        "installed": installed_version(),
        "ui": ui_version(),
        "restart_required": restart_required(),
    }


def version_label() -> str:
    """Return the short header label the workspace shows under the project name."""

    parts = [f"{DISTRIBUTION_NAME} {APP_VERSION}"]
    ui = ui_version()
    if ui is not None:
        parts.append(f"UI {ui}")
    return " · ".join(parts)
