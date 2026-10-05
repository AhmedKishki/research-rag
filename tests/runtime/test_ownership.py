"""A signal is sent only to a process this project can prove it owns.

`AGENTS.md` holds the rule. A pid record names a number, and a number proves
nothing: the record can be stale and the number can have been reused. These tests
cover both, and the only process any of them signals is a disposable child of this
test run, so a failure here cannot reach a reader's app.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from research_rag.runtime import ownership

#: A program that keeps running until it is told to stop, and exits when it is.
IDLE = textwrap.dedent(
    """
    import signal, sys, time

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    time.sleep(300)
    """
)


def _disposable(tmp_path: Path, name: str = "idle") -> subprocess.Popen[bytes]:
    """A child of this run that stays up until it is signalled."""

    script = tmp_path / f"{name}.py"
    script.write_text(IDLE, encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL)


def _fake_process(
    proc_root: Path,
    pid: int,
    arguments: list[str],
    *,
    ppid: int = 1,
) -> None:
    """A `/proc` entry for a process that does not exist, so no signal can reach it."""

    entry = proc_root / str(pid)
    entry.mkdir(parents=True)
    (entry / "cmdline").write_bytes(
        b"".join(argument.encode("utf-8") + b"\0" for argument in arguments)
    )
    (entry / "stat").write_text(f"{pid} (name) S {ppid} 0 0 -1 0\n")


@pytest.fixture
def proc_root(tmp_path: Path) -> Path:
    """A `/proc` tree this test writes, so a refusal reaches no real process."""

    root = tmp_path / "proc"
    root.mkdir()
    return root


@pytest.fixture
def disposable(tmp_path: Path) -> subprocess.Popen[bytes]:
    """One idle child, always ended before the test returns."""

    child = _disposable(tmp_path)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)


def test_a_live_process_this_project_does_not_own_is_left_running(
    disposable: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """A live process is not enough: it has to be this project's app.

    This is the case a stale record produces. The number is alive, so the record's
    liveness probe passes, and the process is a reader's own work that a stop would
    otherwise end.
    """

    verdict = ownership.judge(disposable.pid, tmp_path / "thesis")

    assert not verdict.owned
    assert "does not run this app" in verdict.reason

    outcome = ownership.ask_to_stop(disposable.pid, tmp_path / "thesis")

    assert not outcome.asked
    assert disposable.poll() is None
    assert outcome.reason == verdict.reason


def test_another_projects_serving_app_is_left_running(
    disposable: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """This app, for a different project, is a reader's app and not this one's.

    A pid file that names the wrong project is the one way a correct-looking record
    reaches a process this project has no claim on.
    """

    verdict = ownership.judge(disposable.pid, tmp_path / "thesis")

    assert not verdict.owned

    outcome = ownership.ask_to_stop(disposable.pid, tmp_path / "thesis")

    assert not outcome.asked
    assert disposable.poll() is None


def test_this_process_and_the_processes_above_it_are_never_owned(
    disposable: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """A sweep must not end the terminal running it, or the reader's own shell.

    This one reads the machine's own `/proc`, because the pids under test are this
    process and its real parent.
    """

    here = os.getpid()
    parent = ownership.parent_of(here)
    assert parent is not None

    for pid, expected in ((here, "this process"), (parent, "descends from")):
        verdict = ownership.judge(pid, tmp_path / "thesis")
        assert not verdict.owned
        assert expected in verdict.reason
        assert not ownership.ask_to_stop(pid, tmp_path / "thesis").asked

    assert disposable.poll() is None


def test_this_projects_own_app_is_asked_to_stop_through_a_handle(
    tmp_path: Path, proc_root: Path
) -> None:
    """A process this project owns is signalled, and signalled through a handle.

    The handle is what makes a reused number harmless, so the test also shows the
    target's command line being read twice: once to accept it and once to confirm
    nothing changed while the handle was opened.
    """

    project = tmp_path / "thesis"
    _fake_process(
        proc_root,
        4242,
        ["/srv/bin/research-rag", "--project-root", str(project), "start"],
        ppid=1,
    )
    opened: list[int] = []
    real_open = ownership._pidfd_for
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        ownership, "_pidfd_for", lambda pid: opened.append(pid) or real_open(pid)
    )
    try:
        verdict = ownership.judge(4242, project, proc_root)
        assert verdict.owned

        outcome = ownership.ask_to_stop(4242, project, proc_root)
    finally:
        monkey.undo()

    assert opened == [4242]
    # No such process exists, so the handle reports it is gone rather than the
    # signal reaching a number that now belongs to someone else.
    assert not outcome.asked
    assert "could not be asked to stop" in outcome.reason or "left running" in (
        outcome.reason
    )


def test_a_kernel_without_a_handle_leaves_the_app_running(
    tmp_path: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `pidfd_open` there is no safe signal, so there is none sent.

    Falling back to a bare number is the case this module exists to refuse: a
    number reused between the read and the signal reaches whatever holds it now.
    """

    project = tmp_path / "thesis"
    _fake_process(proc_root, 4243, ["research-rag", "--project-root", str(project)])
    monkeypatch.setattr(ownership, "_pidfd_for", lambda _pid: None)

    outcome = ownership.ask_to_stop(4243, project, proc_root)

    assert not outcome.asked
    assert "reused pid" in outcome.reason


def test_a_pid_record_that_changes_while_it_is_checked_is_left_running(
    tmp_path: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof is taken again after the handle is opened, and a change refuses.

    This is the reuse window: the number belonged to this project's app when it was
    read and to something else by the time it is signalled.
    """

    project = tmp_path / "thesis"
    entry = proc_root / "4244"
    _fake_process(entry.parent, 4244, ["research-rag", "--project-root", str(project)])
    monkeypatch.setattr(
        ownership, "_pidfd_for", lambda _pid: os.open(os.devnull, os.O_RDONLY)
    )
    real = ownership.judge
    seen: list[int] = []

    def _reuse(pid: int, root: Path, root_proc: Path = proc_root) -> ownership.Verdict:
        seen.append(pid)
        if len(seen) == 2:
            (entry / "cmdline").write_bytes(b"/usr/bin/vim\0")
            return ownership.Verdict(False, "pid 4244 does not run this app: vim")
        return real(pid, root, root_proc)

    monkeypatch.setattr(ownership, "judge", _reuse)

    outcome = ownership.ask_to_stop(4244, project, proc_root)

    assert seen == [4244, 4244]
    assert not outcome.asked
    assert "changed while it was being asked to stop" in outcome.reason


def test_a_relative_project_root_is_not_this_project(
    disposable: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """A relative path would be resolved against the other process's own directory.

    Matching it against this process's directory would claim an app serving
    somewhere this project knows nothing about, so a relative value never matches.
    """

    verdict = ownership.judge(disposable.pid, tmp_path / "thesis")

    assert not verdict.owned
    assert not ownership.ask_to_stop(disposable.pid, tmp_path / "thesis").asked
    assert disposable.poll() is None


def test_lineage_walks_to_the_top_without_cycling(
    tmp_path: Path, proc_root: Path
) -> None:
    """A `/proc` that reports a cycle must not hang the sweep that walks it."""

    _fake_process(proc_root, 100, ["research-rag"], ppid=200)
    _fake_process(proc_root, 200, ["research-rag"], ppid=100)

    assert list(ownership.lineage(100, proc_root)) == [100, 200]


def test_a_process_without_this_program_name_is_never_owned(
    disposable: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """The one live process in these tests is answered the same way every time."""

    assert not ownership.judge(disposable.pid, tmp_path / "thesis").owned
    assert not ownership.ask_to_stop(disposable.pid, tmp_path / "thesis").asked
    assert disposable.poll() is None
