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
from pathlib import Path
from typing import Any

import pytest

from research_rag import registry
from research_rag.app import PID_FILE, PORT_FILE, TTY_FILE, App, recorded_port
from research_rag.config import ConfigurationError, resolve_config
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

    cli._bare_workspace(_no_arguments())

    assert served == {"name": "Only One", "open_browser": True}


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
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

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

    with pytest.raises(ConfigurationError, match="cannot choose") as refusal:
        cli._bare_workspace(_no_arguments())

    assert "Second" in str(refusal.value)


def test_a_bare_call_declines_a_number_outside_the_list(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "First")
    other = tmp_path / "second"
    (other / "sources").mkdir(parents=True)
    _initialised(other, "Second")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "9")

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

    cli._bare_workspace(_no_arguments())

    out = capsys.readouterr().out
    assert opened == ["http://127.0.0.1:5051"]
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
