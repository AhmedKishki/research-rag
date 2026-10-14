"""One child process confined to a transient systemd scope.

The Docling worker and the Docling environment installer both run this way:
memory capped, swap off, CPU limited, wall time bounded, and the whole scope
killed when the call returns, so no descendant outlives it. `docling_worker.py`
checks the same memory cap from inside the scope and keeps its own copy of the
number, because it runs in an interpreter that cannot import this package.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from typing import Any

from ..project.config import child_process_environment

MEMORY_MAX_BYTES = 2 * 1024**3
CPU_QUOTA_PERCENT = 200


def run_in_scope(
    systemd_run: str,
    command: list[str],
    *,
    unit: str,
    timeout: float,
    env: dict[str, str],
    stdin: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    """Run `command` in scope `unit` and return its exit code, stdout, and stderr.

    Raises `subprocess.TimeoutExpired` after the wall limit, once the scope is
    killed. The scope is killed on every return, including a successful one.
    """

    process = subprocess.Popen(
        [
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            f"--property=MemoryMax={MEMORY_MAX_BYTES}",
            "--property=MemorySwapMax=0",
            f"--property=CPUQuota={CPU_QUOTA_PERCENT}%",
            f"--property=RuntimeMaxSec={timeout:g}",
            "--",
            *command,
        ],
        stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    finally:
        terminate_scope(unit, process)
    return process.returncode, stdout or b"", stderr or b""


def terminate_scope(unit: str, process: subprocess.Popen[Any]) -> None:
    """Kill the whole transient scope, then the process group, with no orphan."""

    systemctl = shutil.which("systemctl")
    if systemctl is not None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [
                    systemctl,
                    "--user",
                    "kill",
                    "--signal=SIGKILL",
                    "--kill-whom=all",
                    unit,
                ],
                capture_output=True,
                env=child_process_environment(purpose="systemctl"),
                timeout=15,
                check=False,
            )
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), 9)
    with contextlib.suppress(subprocess.TimeoutExpired):  # pragma: no cover
        process.wait(timeout=15)
