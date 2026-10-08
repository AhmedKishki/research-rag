"""The terminal owns the app's lifetime.

A closed window ends the app and the gateway with it, the app records the same running
state a detached one does, and a bare call never starts a second app for a project
that already has one.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import research_rag.runtime.app as app_module
import research_rag.runtime.update as update_module
from research_rag.project import registry, state_files
from research_rag.project.config import ConfigurationError, resolve_config
from research_rag.project.policy import ResearchError
from research_rag.project.state_files import recorded_pid
from research_rag.runtime.app import (
    PID_FILE,
    PORT_FILE,
    TTY_FILE,
    App,
    has_terminal,
    recorded_port,
)
from research_rag.surfaces import cli

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _initialised(root: Path, name: str) -> Any:
    """Register a project the way `init` does."""

    config = resolve_config(root, project_name=name)
    registry.register(config.project_id, config.project_name, root)
    return config


def _a_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _no_arguments(*arguments: str) -> Any:
    """The namespace the real parser builds, so a test cannot invent an option."""

    return cli._parser().parse_args(list(arguments))


async def test_an_app_records_the_port_and_pid_it_serves(project: Path) -> None:
    config = _initialised(project, "Attached")
    app = App(config, port=_a_free_port())

    await app.start()
    try:
        assert recorded_port(config) == app.port
        assert (config.state_root / PID_FILE).read_text().strip() == str(os.getpid())
    finally:
        await app.stop()

    assert recorded_port(config) is None
    assert not (config.state_root / PID_FILE).exists()
    assert not (config.state_root / TTY_FILE).exists()


async def test_an_app_that_stops_leaves_a_later_apps_record_alone(
    project: Path,
) -> None:
    """An app is given the project only once the previous one has let it go, so a record
    naming another pid means something took over in between, and deleting it here would
    leave that process unreachable to `stop`.
    """

    config = _initialised(project, "Attached")
    app = App(config, port=_a_free_port())
    await app.start()
    (config.state_root / PORT_FILE).write_text("5099\n", encoding="utf-8")
    (config.state_root / PID_FILE).write_text("1\n", encoding="utf-8")

    await app.stop()

    assert (config.state_root / PID_FILE).read_text().strip() == "1"
    assert (config.state_root / PORT_FILE).read_text().strip() == "5099"


# --- the terminal owns the lifetime -------------------------------------------


async def test_an_app_that_never_serves_would_stop_ends_when_the_closing_asks(
    project: Path,
) -> None:
    """The wait ends on the terminal going away, which is the property being built."""

    config = _initialised(project, "Attached")
    app = App(config, port=_a_free_port())
    closing = cli._Closing()
    waiting = asyncio.ensure_future(cli._wait_until_stopped(app, closing))
    await asyncio.sleep(0)

    closing.request()
    await asyncio.wait_for(waiting, timeout=5)

    assert closing.asked.is_set()


async def test_an_app_ends_when_its_terminal_goes_without_a_signal(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disowned process gets no SIGHUP when its terminal closes.

    It loses its controlling terminal instead, and that loss ends the wait the
    same way a closed window does, so no serving process outlives its terminal.
    """

    answers = iter([True, True, False])
    monkeypatch.setattr(cli, "_terminal_attached", lambda: next(answers, False))
    monkeypatch.setattr(cli, "_TERMINAL_CHECK_SECONDS", 0.01)
    config = _initialised(project, "Disowned")
    app = App(config, port=_a_free_port())

    await asyncio.wait_for(cli._wait_until_stopped(app, cli._Closing()), timeout=5)


async def test_a_process_with_no_terminal_is_not_served(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background job, a desktop launcher, or an agent's shell has no terminal.

    Serving from one leaves a process only `stop` ends, so it is refused before
    an app is built or a port is claimed.
    """

    def never(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no app may be built without a terminal")

    monkeypatch.setattr(cli, "_terminal_attached", lambda: False)
    monkeypatch.setattr(cli, "App", never)
    config = _initialised(project, "Terminalless")

    with pytest.raises(ResearchError, match="has no terminal"):
        await cli._serve_attached(config, port=None, open_browser=False)
    assert not (config.state_root / PID_FILE).exists()


async def test_a_closing_asks_only_once() -> None:
    closing = cli._Closing()

    closing.request()
    closing.request()

    assert closing.asked.is_set()


async def test_the_terminal_signals_are_handled_while_attached_and_released_after() -> (
    None
):
    """A closed window sends SIGHUP and a kill sends SIGTERM; both must reach the app."""

    before = {
        number: signal.getsignal(number) for number in (signal.SIGHUP, signal.SIGTERM)
    }
    with cli._closing_with_the_terminal() as closing:
        inside = {
            number: signal.getsignal(number)
            for number in (signal.SIGHUP, signal.SIGTERM)
        }
        signal.raise_signal(signal.SIGHUP)
        # The loop reads its own pipe on the next pass, so one turn is not enough
        # for the signal to arrive as a callback.
        await asyncio.sleep(0.05)
        assert closing.asked.is_set()
    after = {
        number: signal.getsignal(number) for number in (signal.SIGHUP, signal.SIGTERM)
    }

    assert inside != before, "no handler was installed for either signal"
    assert after == before, "a handler outlived the attached run"


# project_app_state probes the live control endpoint with its production timeout.
@pytest.mark.integration
async def test_a_project_state_names_the_terminal_its_app_is_attached_to(
    project: Path,
) -> None:
    """A reader deciding whether Ctrl-C here stops that app needs the difference."""

    config = _initialised(project, "Attached")
    assert _attached_state(config) is None

    app = App(config, port=_a_free_port())
    await app.start()
    try:
        (config.state_root / TTY_FILE).write_text("/dev/pts/7\n", encoding="utf-8")
        assert _attached_state(config) == "/dev/pts/7"
    finally:
        await app.stop()

    assert _attached_state(config) is None


def _attached_state(config: Any) -> str | None:
    return registry.project_app_state(config.project_root)["app"]["attached_to"]


# --- the bare call -------------------------------------------------------------


@pytest.mark.parametrize("arguments", [("start",), ("start", "--lan")])
def test_start_without_selector_resolves_registry_not_cwd(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    _initialised(project, "Chosen")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_only_project_or_ask", lambda: project)
    assert cli._resolve(_no_arguments(*arguments)).project_root == project


@pytest.mark.parametrize("selector", ["--project", "--project-root"])
def test_start_explicit_selector_bypasses_picker(
    project: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    _initialised(project, "Chosen")

    def refuse() -> Path:
        raise AssertionError("explicit selectors must not prompt")

    monkeypatch.setattr(cli, "_only_project_or_ask", refuse)
    value = "Chosen" if selector == "--project" else str(project)
    assert cli._resolve(_no_arguments(selector, value, "start")).project_root == project


@pytest.mark.parametrize(
    ("keys", "expected"),
    [(b"\x1b[B\r", 1), (b"\x1b[A\r", 1), (b"\r", 0), (b"\x1b", None), (b"\x03", None)],
)
def test_project_picker_reads_pty_keys_and_restores_terminal(
    monkeypatch: pytest.MonkeyPatch, keys: bytes, expected: int | None
) -> None:
    import io
    import pty
    import termios
    import threading

    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    ready = threading.Event()
    result: list[Any] = []

    class Output(io.StringIO):
        def flush(self) -> None:
            ready.set()

    source = os.fdopen(os.dup(slave), "r")
    monkeypatch.setattr(cli.sys, "stdin", source)
    monkeypatch.setattr(cli.sys, "stdout", Output())

    def pick() -> None:
        try:
            result.append(
                cli._pick_registered_project(
                    [{"project_name": "First"}, {"project_name": "Second"}]
                )
            )
        except ConfigurationError as exc:
            result.append(exc)

    worker = threading.Thread(target=pick, daemon=True)
    try:
        worker.start()
        assert ready.wait(3)
        os.write(master, keys)
        worker.join(3)
        assert not worker.is_alive()
        assert termios.tcgetattr(slave) == before
        if expected is None:
            assert isinstance(result[0], ConfigurationError)
            assert "No project chosen" in str(result[0])
        else:
            assert result == [expected]
    finally:
        if worker.is_alive():
            os.write(master, b"\x03")
            worker.join(3)
        source.close()
        os.close(master)
        os.close(slave)


def test_a_bare_call_with_one_project_opens_it_in_a_browser(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "Only One")
    served: dict[str, Any] = {}

    def _serve(config: Any, *, open_browser: bool) -> cli.CommandResult:
        served["name"] = config.project_name
        served["open_browser"] = open_browser
        return cli.CommandResult()

    monkeypatch.setattr(cli, "_serve_attached_sync", _serve)

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_pick_registered_project", lambda entries: 0)

    cli._bare_workspace(_no_arguments())

    # Serving never opens a browser on its own: the command line is where this app is
    # worked from, so a browser appears only when `--start-ui` asks for one.
    assert served == {"name": "Only One", "open_browser": False}


def test_noninteractive_single_project_requires_explicit_selector(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "Only One")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(ConfigurationError, match="interactive terminal") as refusal:
        cli._only_project_or_ask()
    assert "research-rag --project 'Only One' start" in str(refusal.value)


def test_picker_restores_terminal_after_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import pty
    import termios

    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    source = os.fdopen(os.dup(slave), "r")
    monkeypatch.setattr(cli.sys, "stdin", source)
    monkeypatch.setattr(cli.sys, "stdout", io.StringIO())

    def fail(fd: int, size: int) -> bytes:
        raise OSError("read failed")

    monkeypatch.setattr(os, "read", fail)
    try:
        with pytest.raises(OSError, match="read failed"):
            cli._pick_registered_project([{"project_name": "First"}])
        assert termios.tcgetattr(slave) == before
    finally:
        source.close()
        os.close(master)
        os.close(slave)


def test_a_bare_call_with_several_projects_asks_which(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "First")
    other = tmp_path / "second"
    (other / "sources").mkdir(parents=True)
    _initialised(other, "Second")
    served: dict[str, str] = {}

    def _serve(config: Any, *, open_browser: bool) -> cli.CommandResult:
        served["name"] = config.project_name
        return cli.CommandResult()

    monkeypatch.setattr(cli, "_serve_attached_sync", _serve)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_pick_registered_project", lambda entries: 1)

    cli._bare_workspace(_no_arguments())

    assert served["name"] == "Second"


def test_a_bare_call_that_cannot_ask_is_given_the_commands(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script must not be left waiting for a reader who is not there."""

    _initialised(project, "First")
    other = tmp_path / "second"
    (other / "sources").mkdir(parents=True)
    _initialised(other, "Second")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)

    with pytest.raises(ConfigurationError, match="interactive terminal") as refusal:
        cli._bare_workspace(_no_arguments())

    assert "Second" in str(refusal.value)


def test_a_bare_call_does_not_serve_after_picker_cancellation(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "First")
    other = tmp_path / "second"
    (other / "sources").mkdir(parents=True)
    _initialised(other, "Second")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)

    def cancel(entries: Any) -> int:
        raise ConfigurationError("No project chosen")

    monkeypatch.setattr(cli, "_pick_registered_project", cancel)

    with pytest.raises(ConfigurationError, match="No project chosen"):
        cli._bare_workspace(_no_arguments())


def test_a_bare_call_reports_an_app_already_up(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two apps on one project would each hold the lock and open a gateway."""

    _initialised(project, "Already")
    opened: list[str] = []
    monkeypatch.setattr(cli, "running_url", lambda config: "http://127.0.0.1:5051")
    monkeypatch.setattr(cli, "_open_browser", opened.append)

    def _refuse(config: Any, *, open_browser: bool) -> cli.CommandResult:
        raise AssertionError("a second app must not be started")

    monkeypatch.setattr(cli, "_serve_attached_sync", _refuse)

    cli._bare_workspace(_no_arguments("--project", "Already"))

    out = capsys.readouterr().out
    # Reporting the app another terminal owns never opens a browser either.
    assert opened == []
    assert "already served" in out
    assert "Ctrl-C here would not stop it" in out
    assert "stop" in out


def test_a_bare_call_honours_a_named_project(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--project` settles the choice, so the question is not asked."""

    _initialised(project, "First")
    other = tmp_path / "second"
    (other / "sources").mkdir(parents=True)
    _initialised(other, "Second")
    served: dict[str, str] = {}

    def _serve(config: Any, *, open_browser: bool) -> cli.CommandResult:
        served["name"] = config.project_name
        return cli.CommandResult()

    monkeypatch.setattr(cli, "_serve_attached_sync", _serve)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    cli._bare_workspace(_no_arguments("--project", "Second"))

    assert served["name"] == "Second"


def test_a_bare_call_with_no_project_says_how_to_make_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "account_projects", lambda: {"projects": []})

    with pytest.raises(ConfigurationError, match="init --project-root"):
        cli._bare_workspace(_no_arguments())


# --- the detached app ----------------------------------------------------------


def _stat(root: Path, pid: int, tty_nr: int, comm: str = "research-rag") -> None:
    """Write the /proc/<pid>/stat a process with this controlling terminal has."""

    process = root / str(pid)
    process.mkdir(parents=True, exist_ok=True)
    (process / "stat").write_text(
        f"{pid} ({comm}) S 1 {pid} {pid} {tty_nr} -1 4194560 0 0 0 0 0 0 0 0 20 0 3 0 1000\n",
        encoding="utf-8",
    )


def test_a_process_with_no_controlling_terminal_is_detached(tmp_path: Path) -> None:
    _stat(tmp_path, 4242, tty_nr=0)

    assert has_terminal(4242, proc_root=tmp_path) is False


def test_a_process_with_a_controlling_terminal_is_attached(tmp_path: Path) -> None:
    _stat(tmp_path, 4242, tty_nr=34816)

    assert has_terminal(4242, proc_root=tmp_path) is True


def test_a_command_name_in_parentheses_does_not_shift_the_terminal(
    tmp_path: Path,
) -> None:
    """`comm` is in parentheses and may hold a space or one, so fields are counted
    from after its final `)` rather than by splitting the line."""

    _stat(tmp_path, 4242, tty_nr=0, comm="my (odd) name")

    assert has_terminal(4242, proc_root=tmp_path) is False


def test_a_process_that_cannot_be_asked_is_neither(
    tmp_path: Path,
) -> None:
    """Unknown is not the same as attached, and must never read as it."""

    assert has_terminal(999999, proc_root=tmp_path) is None


def test_an_app_with_no_terminal_is_reported_detached(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _initialised(project, "Orphan")
    monkeypatch.setattr(app_module, "recorded_pid", lambda _config: 4242)
    monkeypatch.setattr(app_module, "has_terminal", lambda pid: False)

    assert registry.detached_from_terminal(config, running=True) is True


def test_an_app_with_a_terminal_is_not_reported_detached(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _initialised(project, "Held")
    monkeypatch.setattr(app_module, "recorded_pid", lambda _config: 4242)
    monkeypatch.setattr(app_module, "has_terminal", lambda pid: True)

    assert registry.detached_from_terminal(config, running=True) is False


def test_no_app_at_all_is_not_detached(project: Path) -> None:
    config = _initialised(project, "Down")

    assert registry.detached_from_terminal(config, running=False) is False


def _served(config: Any, monkeypatch: pytest.MonkeyPatch, detached: bool) -> None:
    monkeypatch.setattr(cli, "running_url", lambda _config: "http://127.0.0.1:5051")
    monkeypatch.setattr(
        cli, "detached_from_terminal", lambda _config, _running: detached
    )


def test_a_bare_call_names_a_detached_app_and_its_remedy(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _initialised(project, "Orphan")
    _served(resolve_config(project), monkeypatch, detached=True)
    monkeypatch.setattr(cli, "_serve_attached_sync", lambda *_a, **_k: None)

    cli._bare_workspace(_no_arguments("--project-root", str(project)))

    out = capsys.readouterr().out
    assert "no terminal attached" in out
    assert f"'{cli.CLI_NAME} stop'" in out


def test_a_bare_call_blames_another_terminal_only_when_one_owns_it(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _initialised(project, "Held")
    _served(resolve_config(project), monkeypatch, detached=False)

    cli._bare_workspace(_no_arguments("--project-root", str(project)))

    out = capsys.readouterr().out
    assert "this terminal does not own" in out
    assert "no terminal attached" not in out


def test_stop_reports_a_detached_app_and_names_what_ends_it(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _initialised(project, "Orphan")
    monkeypatch.setattr(
        cli,
        "project_app_state",
        lambda _root: {"app": {"running": True, "attached_to": None, "detached": True}},
    )
    monkeypatch.setattr(cli, "recorded_pid", lambda _config: None)

    report = cli._stop(_no_arguments("--project-root", str(project), "stop"), config)

    assert report["detached"] is True
    assert any("no terminal attached" in note for note in report["notes"])


def test_stop_does_not_blame_a_terminal_that_owns_nothing(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _initialised(project, "Held")
    monkeypatch.setattr(
        cli,
        "project_app_state",
        lambda _root: {
            "app": {"running": True, "attached_to": "/dev/pts/3", "detached": False}
        },
    )
    monkeypatch.setattr(cli, "recorded_pid", lambda _config: None)

    report = cli._stop(_no_arguments("--project-root", str(project), "stop"), config)

    assert report["detached"] is False
    assert not any("no terminal attached" in note for note in report["notes"])


def test_a_pid_record_that_names_no_process_is_not_signed(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero and a negative number name a process group, so `stop` must not signal one.

    `os.kill(0, SIGTERM)` reaches every process in the caller's own group, which
    includes the shell that ran `stop`. A pid file holding either is not a process
    this project owns, so it reads as no app rather than as one to end.
    """

    config = _initialised(project, "Zero")
    config.state_root.mkdir(parents=True, exist_ok=True)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(cli, "project_app_state", lambda _root: {"app": {}})
    monkeypatch.setattr(cli.os, "kill", lambda pid, number: sent.append((pid, number)))

    for recorded in ("0", "-1", "", "not-a-pid"):
        (config.state_root / PID_FILE).write_text(f"{recorded}\n", encoding="utf-8")
        assert recorded_pid(config.state_root) is None, recorded
        report = cli._stop(
            _no_arguments("--project-root", str(project), "stop"), config
        )
        assert report["running"] is False, recorded
        assert report.get("stopped") is False, recorded

    assert sent == []


def test_stop_will_not_signal_an_unrelated_live_process(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale record naming a live process is refused, and that process survives.

    This is the record a crashed app or a copied project leaves behind: the number
    is alive, so the liveness probe passes, and the process behind it belongs to the
    reader. `stop` reports the refusal and leaves it running.
    """

    config = _initialised(project, "Stale")
    config.state_root.mkdir(parents=True, exist_ok=True)
    reader = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
    )
    try:
        (config.state_root / PID_FILE).write_text(f"{reader.pid}\n", encoding="utf-8")
        monkeypatch.setattr(cli, "project_app_state", lambda _root: {"app": {}})

        report = cli._stop(
            _no_arguments("--project-root", str(project), "stop"), config
        )

        assert report["stopped"] is False
        assert "does not run this app" in report["stop_refusal"]
        assert reader.poll() is None
    finally:
        reader.kill()
        reader.wait(timeout=30)


def test_an_update_never_signals_a_record_that_names_a_group(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`update --apply` signals the same record, so the guard is the reader's alone."""

    state = state_files.ProjectState(project_root=project, project_name="Zero")
    state.state_root.mkdir(parents=True, exist_ok=True)
    (state.state_root / state_files.PID_FILE).write_text("0\n", encoding="utf-8")

    stopped = update_module.stop_app(state, lambda *_args, **_kwargs: None)

    assert stopped.stopped is False
    assert "no app was running" in stopped.detail
