"""Serving: the app this terminal starts and keeps, the stop sweep, and the
clients attached to a running app.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import (
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...project.config import (
    CLI_COMMAND,
    DEFAULT_UI_PORT,
    ResearchConfig,
    configured_source_directory,
    resolve_config,
)
from ...project.policy import ResearchError
from ...project.state_files import LOCK_FILE
from ...project.state_files import process_alive as alive
from ...runtime import ownership
from ...runtime.account import detached_from_terminal, project_app_state
from ...runtime.app import (
    UI_HOST,
    App,
    _claim_loopback_port,
    _own_tty,
    has_terminal,
    recorded_pid,
    recorded_port,
    running_url,
)
from ...runtime.control import connect
from .projects import _config_kwargs, _only_project_or_ask, _project_path

# How far above the default port an attached app looks for a free one. A port a
# reader named is never moved, so a named port that is taken fails and says so
# rather than being served somewhere they did not ask for.
_PORT_ATTEMPTS = 32


# A process the stop sweep may signal has to name one of these, so a shell or an
# editor that merely mentions the project path is never touched. The MCP server
# this app was seeded from is deliberately absent: during the migration both
# products can serve one project, and stopping this app's workspace must not stop
# a server a user started on purpose.
STOP_GRACE_SECONDS = 5.0
# How long a takeover waits for the app it asked to stop to exit and free its port.
TAKE_OVER_WAIT_SECONDS = 30.0


def _service_processes(
    project_root: Path,
    proc_root: Path = Path("/proc"),
) -> list[tuple[int, str]]:
    """Return the ``(pid, command)`` pairs serving this project.

    A process qualifies when it invokes this app by name *and* names this project
    as ``--project-root``. A relative ``--project-root`` is not matched, because
    resolving it would use this process's directory rather than the other one's.

    The name must be a whole argument, not a substring of one. This app's own
    projects all contain ``.research-rag`` in their paths, so a substring test
    matches unrelated processes that touch the project. Only the app entrypoint
    and its matching ``--project-root`` prove ownership.
    """

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    for pid in sorted(int(entry.name) for entry in entries if entry.name.isdigit()):
        if pid == os.getpid():
            continue
        try:
            raw = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        arguments = [
            part for part in raw.decode("utf-8", "replace").split("\0") if part
        ]
        if not ownership.invokes_this_app(arguments, project_root):
            continue
        if ownership.option_value(arguments, "--project-root") != str(project_root):
            continue
        found.append((pid, " ".join(arguments)))
    return found


def _terminate(project_root: Path, pids: list[int]) -> list[int]:
    """Ask these processes to stop and then insist; return the ones killed outright.

    Each signal goes through `ownership`, so the sweep only reaches a process it
    has proved is this project's app, and never one it descends from.
    """

    for pid in pids:
        ownership.ask_to_stop(pid, project_root)
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.1)
    forced = [pid for pid in pids if alive(pid)]
    for pid in forced:
        ownership.ask_to_stop(pid, project_root, number=signal.SIGKILL)
    return forced


def _stop_notes(detached: bool) -> list[str]:
    """The sentences `stop` returns, leading with the condition it found.

    A detached app is named first because it is the one that does not end with
    its terminal, so a reader who started one and walked away needs to know that
    the terminal they left is not what is holding the project.
    """

    notes = [
        (
            "An app belongs to the terminal that started it: it ends there, "
            "and this command does not start it again."
        )
    ]
    if detached:
        notes.insert(
            0,
            "This app is serving with no terminal attached, so no terminal holds "
            "it and closing one does not end it. Start it again from a terminal "
            "once this one is stopped.",
        )
    return notes


def _stop(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Stop this project's app, and with ``--servers`` every process serving it.

    An app belongs to the terminal that started it, so this command reports which
    terminal owns it and says plainly that stopping it here leaves that terminal
    without an app. It never starts one afterwards: nothing about an app outliving
    the terminal that began it.
    """

    app_state = project_app_state(config.project_root).get("app") or {}
    report: dict[str, Any] = {
        "project_root": str(config.project_root),
        "running": bool(app_state.get("running")),
        "attached_to": app_state.get("attached_to"),
        "detached": app_state.get("detached"),
        "url": app_state.get("url"),
    }
    # Asked, never killed: the app runs its own cleanup and releases its own lock
    # on the way out. A pid record names a number, so `ownership` establishes that
    # the process behind it is this project's app before anything is signalled.
    pid = recorded_pid(config)
    report["stopped"] = False
    report["stop_refusal"] = None
    if pid is not None:
        outcome = ownership.ask_to_stop(
            pid, config.project_root, port=recorded_port(config)
        )
        report["stopped"] = outcome.asked
        report["stop_refusal"] = None if outcome.asked else outcome.reason
        report["pid"] = pid
    if not args.servers:
        report["notes"] = _stop_notes(bool(app_state.get("detached")))
        return report
    found = _service_processes(config.project_root)
    report["servers"] = [{"pid": pid, "command": command} for pid, command in found]
    report["forced_pids"] = _terminate(config.project_root, [pid for pid, _ in found])
    report["notes"] = [
        "An app you started in another terminal belongs to that terminal: this command stops it and does not start it again.",
        "A build interrupted this way resumes from its checkpoint on the next ingest call.",
    ]
    return report


@dataclass(frozen=True, slots=True)
class CommandResult:
    payload: dict[str, Any] | None = None
    exit_code: int = 0
    text: str | None = None


# How often a serving app checks that the terminal it was started from is still
# its controlling terminal.
_TERMINAL_CHECK_SECONDS = 2.0


def _terminal_attached() -> bool:
    """Whether this process has a controlling terminal.

    The live process answers first, as `/proc/self/stat` reports it. Where there
    is no procfs, standard input standing on a terminal is the next best answer.
    """

    answer = has_terminal(os.getpid())
    if answer is not None:
        return answer
    return _own_tty() is not None


async def _terminal_lost() -> None:
    """Return when the controlling terminal this process started with is gone.

    A closed window sends SIGHUP, which the closing below already handles. A
    process the shell disowned, or one whose session leader exited, gets no
    signal: it only loses its terminal, so the loss is checked for directly.
    """

    # Polled: the kernel announces a lost terminal to a process outside the
    # foreground group with nothing an event loop can wait on.
    while True:
        if not _terminal_attached():
            return
        await asyncio.sleep(_TERMINAL_CHECK_SECONDS)


async def _serve_attached(
    config: ResearchConfig,
    *,
    port: int | None,
    open_browser: bool,
    lan: bool = False,
) -> CommandResult:
    """Serve this project in this terminal until the terminal or the app ends.

    Attached is the whole point: the process stays in the foreground and in this
    terminal's process group, so Ctrl-C and a closed window both reach it, and
    the stop below releases the app's owned resources.

    The port is claimed by binding it, so a second project on the same machine
    walks past a taken one rather than failing where a second app would have
    moved it.
    """

    # A process with no terminal would serve until `stop`, which is the detached
    # state this app does not serve. It is refused before anything is started.
    if not _terminal_attached():
        raise ResearchError(
            f"This process has no terminal, so {config.project_name} was not "
            "served. The app is served from a terminal and stops when that "
            f"terminal closes: run '{CLI_COMMAND} start' in one. Serving detached is "
            "not available."
        )
    # A port named by the caller is the one they asked for, and it is tried once:
    # its refusal is the app's own sentence, which names the alternative. A port
    # chosen here is this app's to move, so it walks forward past one that is taken.
    first = port if port is not None else _a_free_port()
    app: App | None = None
    for offset in (0,) if port is not None else range(_PORT_ATTEMPTS):
        app = App(config, port=first + offset, **({"lan": True} if lan else {}))
        await app.start()
        if app.error is None:
            break
    if app is None or app.error is not None:
        # Every candidate was taken. The last refusal is the app's own, naming the
        # port and the fact that a project already serves it.
        raise ResearchError(app.error if app is not None and app.error else "no port")
    if open_browser:
        _open_browser(app.url)
    _attached_banner(config, app)
    with _closing_with_the_terminal() as closing:
        try:
            await _wait_until_stopped(app, closing)
        finally:
            await app.stop()
    sys.stdout.write(f"Stopped. {config.project_name} is no longer served.\n")
    return CommandResult()


async def _wait_until_stopped(app: App, closing: _Closing) -> None:
    """Return when the app stops serving, when the terminal closes, or on Ctrl-C.

    Three things can end an attached run and they are the same ending: the
    serving task finishing on its own, this terminal going away, and the reader
    pressing Ctrl-C. Whichever arrives first, the caller stops the app on the way
    out, so no owned serving task is left running.
    """

    serving = asyncio.ensure_future(app.wait())
    asked = asyncio.ensure_future(closing.asked.wait())
    # Only a terminal that was there can be lost; a run with none was refused.
    watched = [asyncio.ensure_future(_terminal_lost())] if _terminal_attached() else []
    try:
        await asyncio.wait(
            (serving, asked, *watched), return_when=asyncio.FIRST_COMPLETED
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Only the waits on the closing and the terminal are cancelled here. The
        # serving task is left for `App.stop`, which asks uvicorn to shut down and
        # then waits for it; cancelling this waiter would put a CancelledError
        # inside that task and turn an orderly stop into a traceback.
        asked.cancel()
        for waiter in watched:
            waiter.cancel()


def _a_free_port() -> int:
    """Return a port nothing is listening on, or None when the range is full.

    This is a probe, and the app's own claim is the listen, so the answer can be
    taken by something else between the two. The window is the length of one
    socket round trip and the alternative — starting a whole app to see whether it
    binds — is worse: it builds a service and records state for a port this call
    is only going to measure.
    """

    first = DEFAULT_UI_PORT
    for candidate in range(first, first + _PORT_ATTEMPTS):
        try:
            claim = _claim_loopback_port(UI_HOST, candidate)
        except OSError:
            continue
        claim.close()
        return candidate
    raise ResearchError(
        f"No free loopback port in the next {_PORT_ATTEMPTS} from {first}, so the "
        f"app for this project was not started; pass --port to choose one."
    )


def _attached_banner(config: ResearchConfig, app: App) -> None:
    """Say what is running, where it is, and what closes it."""

    lines = [
        f"{config.project_name} — workspace attached to {_own_tty() or 'no terminal'}",
        f"  {app.url}",
        f"  pid {os.getpid()} · log {config.state_root / 'logs' / 'research-rag-ui.log'}",
        "  Ctrl-C stops the app and releases its retrieval backend.",
    ]
    if getattr(app, "network_access", None) is not None and app.network_access.lan:
        lines.append(
            f"  Home-network HTTP access has no login. Open {app.url}/remote "
            "for phone URLs and QR codes; do not port-forward this port."
        )
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


class _Closing:
    """The stop a terminal closing, or a signal, asks for."""

    def __init__(self) -> None:
        self.asked = asyncio.Event()

    def request(self) -> None:
        """Say once that the terminal is going, and ask the app to stop."""

        if not self.asked.is_set():
            sys.stdout.write("\nClosing.\n")
            sys.stdout.flush()
            self.asked.set()


@contextmanager
def _closing_with_the_terminal() -> Iterator[_Closing]:
    """Treat a closed window and a `kill` as the Ctrl-C they are.

    Closing a terminal sends SIGHUP and a `kill` sends SIGTERM. Python's default
    for both ends the process without app cleanup, so each becomes the orderly stop a
    Ctrl-C already performs. A platform that cannot install a handler for one of
    them keeps its default rather than refusing to serve.
    """

    closing = _Closing()
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for number in (signal.SIGHUP, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, closing.request)
        except (NotImplementedError, RuntimeError, ValueError, OSError):
            continue
        installed.append(number)
    try:
        yield closing
    finally:
        for number in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(number)


def _is_detached(config: ResearchConfig) -> bool:
    """Whether this project's app is serving with no terminal attached.

    The app holds itself to being served from the terminal that started it, so a
    serving process that has none is a state a reader is told about rather than
    one that is served around. The answer comes from the live process, so a
    terminal file that was never written cannot make an attached app read as
    detached.
    """

    return detached_from_terminal(config, running_url(config) is not None) is True


def _project_busy(config: ResearchConfig) -> bool:
    """Whether a write, such as an ingest build, holds the project now.

    The project lock is probed and released at once, never held, so the app this
    process then serves takes it as any app does.
    """

    from filelock import FileLock, Timeout

    lock = FileLock(str(config.state_root / LOCK_FILE))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


def _held_by(config: ResearchConfig) -> str:
    """Who serves the project, as the app's live process reports it."""

    if _is_detached(config):
        return "a process with no terminal attached"
    terminal = (project_app_state(config.project_root).get("app") or {}).get(
        "attached_to"
    )
    owner = f" ({terminal})" if terminal else ""
    return f"a process this terminal does not own{owner}"


def _confirmed(question: str) -> bool:
    """Ask a yes-or-no question on this terminal. Anything but yes is no."""

    sys.stdout.write(question)
    sys.stdout.flush()
    try:
        answer = sys.stdin.readline()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return False
    return answer.strip().casefold() in {"y", "yes"}


def _take_over(
    config: ResearchConfig, url: str, *, asked: bool, again: str
) -> CommandResult | int | None:
    """Stop the app another terminal holds, so this terminal can serve the project.

    A terminal that did not start the app cannot stop it with Ctrl-C, so the app
    is restarted here instead of being adopted. `--take-over` or a yes at the
    prompt asks for that; without either the condition is reported, with the
    command that takes it over. A write in progress is never interrupted. The old
    app is stopped through the same proof `stop` uses and given time to exit
    cleanly. Returns the port the old app served on, so clients keep their
    address; None when it had none recorded; or the result that ends this call.
    """

    holder = _held_by(config)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    report = (
        f"{config.project_name} is already served at {url} by {holder}; Ctrl-C "
        f"here would not stop it. '{CLI_COMMAND} --take-over{again}' restarts it "
        f"in this terminal, and '{CLI_COMMAND} stop' stops it.\n"
    )
    if not asked and not interactive:
        sys.stdout.write(report)
        return CommandResult()
    if _project_busy(config):
        sys.stdout.write(
            f"{config.project_name} is served at {url} by {holder}, and a write "
            f"such as an ingest build holds the project, so it was left running. "
            f"Call this again once the write finishes, or '{CLI_COMMAND} stop' "
            f"stops it now and an interrupted build resumes from its checkpoint.\n"
        )
        return CommandResult(exit_code=1)
    if not asked and not _confirmed(
        f"{config.project_name} is already served at {url} by {holder}; Ctrl-C "
        f"here would not stop it. Take it over? It restarts in this terminal, "
        f"and attached agents reconnect. [y/N] "
    ):
        sys.stdout.write(report)
        return CommandResult()
    pid, port = recorded_pid(config), recorded_port(config)
    if pid is not None:
        outcome = ownership.ask_to_stop(pid, config.project_root, port=port)
        if not outcome.asked:
            sys.stdout.write(
                f"{config.project_name} was not taken over: {outcome.reason}\n"
            )
            return CommandResult(exit_code=1)
        deadline = time.monotonic() + TAKE_OVER_WAIT_SECONDS
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if alive(pid):
            sys.stdout.write(
                f"{config.project_name}'s app (pid {pid}) is still stopping, so "
                f"this terminal did not start a second one. Call this again once "
                f"'{CLI_COMMAND} projects' shows it down.\n"
            )
            return CommandResult(exit_code=1)
    return port


def _bare_workspace(args: argparse.Namespace) -> CommandResult:
    """Serve this project's workspace from this terminal, attached to it.

    Nothing opens a browser unless `--start-ui` asked for one, because the command
    line is where this app is worked from and a browser that appears unasked takes
    the reader out of it.

    An app already up for the project is never served a second time, because two
    apps on one project would contend for its lock and indexes. Its terminal is
    not this one, so this call offers to take it over, and otherwise says so and
    returns the reader to their prompt instead of occupying it.
    """

    named = getattr(args, "project", None) or getattr(args, "project_root", None)
    root = _project_path(args) if named else _only_project_or_ask()
    config = resolve_config(
        root, source_directory=configured_source_directory(root), **_config_kwargs(args)
    )
    open_browser = getattr(args, "start_ui", False)
    port: int | None = None
    url = running_url(config)
    if url is not None:
        taken = _take_over(
            config, url, asked=getattr(args, "take_over", False), again=""
        )
        if isinstance(taken, CommandResult):
            if open_browser:
                _open_browser(url)
            return taken
        port = taken
    return _serve_attached_sync(config, port=port, open_browser=open_browser)


def _serve_attached_sync(
    config: ResearchConfig, *, port: int | None, open_browser: bool
) -> CommandResult:
    """Run the attached app on the running loop of this process."""

    return asyncio.run(_serve_attached(config, port=port, open_browser=open_browser))


async def _start(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:
    """Serve this project from this terminal until the terminal or the app ends.

    Attached is the whole point, so this is the same run the bare call makes: the
    process stays in the foreground and in this terminal's process group, where
    Ctrl-C and a closed window both reach it.

    An app already serving this project is never served a second time, because
    two apps on one project would contend for its lock and indexes. Its terminal
    is not this one, so this call offers to take it over, and otherwise says so
    and returns the reader to their prompt instead of occupying it.
    """

    port = args.port
    url = running_url(config)
    if url is not None:
        # Nothing runs on this loop before the app does, so the question is asked
        # on it: a thread blocked on the terminal would outlive a Ctrl-C.
        taken = _take_over(
            config, url, asked=getattr(args, "take_over", False), again=" start"
        )
        if isinstance(taken, CommandResult):
            if args.start_ui:
                _open_browser(url)
            return taken
        port = port if port is not None else taken
    return await _serve_attached(
        config,
        port=port,
        open_browser=args.start_ui,
        **({"lan": True} if getattr(args, "lan", False) else {}),
    )


def _open_browser(url: str) -> None:

    import webbrowser

    # A machine with no desktop session has no browser to open, and a browser
    # that fails to start is not a reason to fail the command. The URL is the
    # answer either way.
    with contextlib.suppress(Exception):
        if webbrowser.open(url):
            return
    print(f"Open {url} in a browser.", file=sys.stderr)


def _clients(config: ResearchConfig) -> dict[str, Any]:

    with nullcontext(connect(config)) as control:
        if control is None:
            return {
                "running": False,
                "clients": [],
                "note": "No app is running for this project; start it with "
                f"'{CLI_COMMAND} start'.",
            }
        return {
            "running": True,
            "url": control.base_url,
            "mcp_url": f"{control.base_url}/mcp",
            "clients": control.clients(),
        }


def _disconnect(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:

    with nullcontext(connect(config)) as control:
        if control is None:
            raise ResearchError(
                f"No app is running for {config.project_root}, so no client is "
                f"attached. Start it with '{CLI_COMMAND} start'."
            )
        return control.disconnect(
            args.session_id, args.reason or "Disconnected by request."
        )
