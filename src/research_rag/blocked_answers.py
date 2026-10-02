"""The status answers a project that cannot be served right now.

This module owns one refusal shape and the prose that goes with it. It may never
read a generation, walk a source directory, or import a surface: a project with
no corpus behind it has no inventory to report, and an answer that carried one
would answer a question the caller did not ask. `status.py` does not re-export
these; `surfaces/mcp.py` and the bridge's own tests import them from here, so
there is one refusal rather than two that can drift.

What moved here from `status.py`: `blocked_status`, `uninitialised_status`, and
`not_served_status`.
"""

from __future__ import annotations

from typing import Any

from .config import (
    ResearchConfig,
    initialise_command,
    project_selection_command,
)


def blocked_status(
    reason: str, *, check: str, remedy: str, initialised: bool
) -> dict[str, Any]:
    """The status answer for a project this machine cannot serve right now.

    One shape for every condition that stops a corpus being answered: `blocked_by`
    names the check that failed, the sentence that explains it, and the command
    that closes it, so an agent reads the same object whether the project was never
    created or the app that serves it is not running.

    The reason names the project, because nothing else in this answer does: an
    agent's answer carries no corpus inventory, and a client that may be
    configured against several projects has to learn which one is missing.
    """

    return {
        "ready": False,
        "stale": False,
        "project_initialised": initialised,
        "blocked_by": [
            {
                "check": check,
                "reason": reason,
                "remedy": remedy,
            }
        ],
        "message": reason,
    }


def uninitialised_status(project_name: str, reason: str) -> dict[str, Any]:
    """The status answer for a project this installation has not initialised.

    An agent's client entry names a project, and the machine it runs on decides
    which directory that name reaches. Where the machine holds no such project,
    this is the whole answer, and it is `blocked_status` with this project's own
    condition and remedy.
    """

    return blocked_status(
        reason,
        check="project.initialised",
        remedy=initialise_command(project_name),
        initialised=False,
    )


def not_served_status(config: ResearchConfig, reason: str) -> dict[str, Any]:
    """The status answer for a project that exists but has no app serving it.

    The project is there and its corpus is readable from disk, so the condition is
    the app rather than the project, and the remedy is the command that starts one
    in a terminal.
    """

    return blocked_status(
        reason,
        check="app.serving",
        remedy=project_selection_command(config.project_name, "start"),
        initialised=True,
    )
