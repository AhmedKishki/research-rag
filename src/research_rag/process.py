"""The one way this app runs an external command.

`git`, `uv`, and `pipx` are not children of this app: they hold no project and
serve no UI, so they inherit the environment unchanged rather than the
managed-child environment a server of this app would be handed. Every call
arrives through an injected runner, so a test can put any tool anywhere it
likes, including out of reach, and the decision it checks is the one the
command makes in production.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# How long an external command may take before it is a failure worth reporting.
COMMAND_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    """One completed subprocess call."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


Runner = Callable[..., CommandResult]


def subprocess_runner(argv: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
    """Run one external command and report what it said."""

    try:
        completed = subprocess.run(
            [str(part) for part in argv],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.SubprocessError as exc:
        return CommandResult(124, "", str(exc))
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)
