"""How this app was installed on this machine, and the command that upgrades it.

Two tools install a distribution, `uv` and `pipx`, and neither is assumed. Each is
asked whether it claims this distribution, and the one that answers is the one
every upgrade command names. Guessing from `PATH` order is how an update ends up
running a command the tool that owns the install does not take.

Nothing here reads a project, and nothing here writes. This module holds the tool
vocabulary, the detection, the command each tool takes, and the refusal that
names both install commands instead of leaving a reader to work one out.
"""

from __future__ import annotations

import json
import re

from .process import CommandResult, Runner
from .version import DISTRIBUTION_NAME

UV = "uv"
PIPX = "pipx"
# The two commands that put this app on a machine in the first place. A refusal
# names them so a reader can run one instead of being told a guess is wrong.
ALTERNATIVES = (
    "uv tool install git+https://github.com/AhmedKishki/research-rag.git",
    "pipx install git+https://github.com/AhmedKishki/research-rag.git",
)


def _owns_tool(tool: str, run: Runner) -> tuple[str, str] | None:
    """What one tool says it installed from, or None when it claims nothing.

    The second element is the specifier the tool records for the install, or an
    empty string when it records none. It is empty rather than guessed because a
    reader is told what the tool said, not what it might have meant.
    """

    if tool == UV:
        result = run(["uv", "tool", "list", "--show-version-specifiers"])
        if not result.ok:
            return None
        for line in result.stdout.splitlines():
            if not line.startswith(f"{DISTRIBUTION_NAME} "):
                continue
            # uv prints `[required: SPEC]`; earlier releases printed `(from SPEC)`.
            specifier = re.search(r"(?:\[required: (.+)\]|\(from (.+)\))\s*$", line)
            return DISTRIBUTION_NAME, (
                (specifier.group(1) or specifier.group(2)) if specifier else ""
            )
        return None
    result = run(["pipx", "list", "--json"])
    if not result.ok:
        return None
    try:
        document = json.loads(result.stdout or "{}")
    except ValueError:
        return None
    venvs = document.get("venvs") if isinstance(document, dict) else None
    if not isinstance(venvs, dict):
        return None
    for venv in venvs.values():
        metadata = venv.get("metadata") if isinstance(venv, dict) else None
        main = metadata.get("main_package") if isinstance(metadata, dict) else None
        packages = main.get("package") if isinstance(main, dict) else None
        if packages and DISTRIBUTION_NAME in packages:
            return DISTRIBUTION_NAME, str(main.get("package_or_url") or "")
    return None


def owning_tool(run: Runner) -> tuple[str | None, str | None]:
    """The tool that installed this distribution, found by asking each one."""

    for tool in (UV, PIPX):
        claimed = _owns_tool(tool, run)
        if claimed is not None:
            return tool, claimed[1]
    return None, None


def outdated_command(tool: str) -> tuple[str, ...]:
    """The command one tool takes to say what it would upgrade to."""

    if tool == UV:
        return (UV, "tool", "list", "--outdated")
    return (
        PIPX,
        "runpip",
        DISTRIBUTION_NAME,
        "index",
        "versions",
        DISTRIBUTION_NAME,
    )


def available_version(tool: str, result: CommandResult) -> str | None:
    """The version a tool prints as available, read out of what it printed.

    Each tool reports an upgrade in its own words and neither is translated into
    the other's, because a version this module cannot find is a version it does
    not report.
    """

    if tool == UV:
        found = re.search(r"upgrade available to:\s*v?([0-9][^\s`]*)", result.stdout)
    else:
        found = re.search(r"Available versions:\s*([0-9][^,\s]*)", result.stdout)
    return found.group(1) if found else None


def git_source(specifier: str) -> str | None:
    """The repository a `git+` install specifier names, without its ref.

    A release is installed from this repository at its tag, so the ref the
    original install named, if any, is not carried over.
    """

    if not specifier.startswith("git+"):
        return None
    url = specifier.removeprefix("git+").split("#", 1)[0]
    head, separator, ref = url.rpartition("@")
    # An `@` inside the authority is a user, not a ref.
    if separator and "/" not in ref and "/" in head.split("://", 1)[-1]:
        url = head
    return url or None


def install_release_arguments(tool: str, url: str, tag: str) -> tuple[str, ...]:
    """The arguments that install one release tag, replacing this installation."""

    target = f"git+{url}@{tag}"
    return (
        ("tool", "install", "--force", target)
        if tool == UV
        else ("install", "--force", target)
    )


def upgrade_arguments(tool: str) -> tuple[str, ...]:
    """The arguments the owning tool takes to upgrade this distribution."""

    return (
        ("tool", "upgrade", DISTRIBUTION_NAME)
        if tool == UV
        else ("upgrade", DISTRIBUTION_NAME)
    )


def unowned_refusal() -> str:
    """What an installation no tool claims is told, with both commands beside it."""

    return (
        "This installation belongs to no tool uv or pipx recognises. "
        f"Install it with `{ALTERNATIVES[0]}` or `{ALTERNATIVES[1]}`, or "
        "update the checkout it came from by hand."
    )


__all__ = [
    "ALTERNATIVES",
    "PIPX",
    "UV",
    "available_version",
    "git_source",
    "install_release_arguments",
    "outdated_command",
    "owning_tool",
    "unowned_refusal",
    "upgrade_arguments",
]
