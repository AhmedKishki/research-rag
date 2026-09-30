"""Project-root workspace launcher generation and its link."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research_rag.config import resolve_config
from research_rag.launcher import (
    LINK_TARGET,
    default_ui_command,
    ensure_ui_launcher,
    launcher_path,
    launcher_script,
    ui_launcher_state,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "research-project"
    (root / "sources").mkdir(parents=True)
    return root


def _alive(pid: int) -> bool:
    """True while the process exists and is not a zombie."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


# A stub UI that behaves like the real one where the launcher can see it: it
# answers on the port it was given, which is what the launcher waits for before
# it records a pid and a port.
_BINDING_STUB = (
    'PORT=""\n'
    'while [ "$#" -gt 0 ]; do\n'
    '  case "$1" in\n'
    '    --port) PORT="$2"; shift ;;\n'
    "  esac\n"
    "  shift\n"
    "done\n"
    'python3 -c "import socket, sys, time; s = socket.socket(); '
    "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(1); time.sleep(60)\" "
    '"$PORT" &\n'
    "wait\n"
)


def test_initialisation_creates_the_launcher_and_the_root_link(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    resolve_config(project, vanilla_executable=sys.executable)

    script = launcher_path(project / ".research-rag")
    link = project / "open-research-rag-ui.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    assert link.is_symlink()
    assert os.readlink(link) == LINK_TARGET

    body = script.read_text(encoding="utf-8")
    assert str(project.resolve()) in body
    assert 'setsid "$UI_COMMAND"' in body
    # The launcher passes explicit flags only: no top-level-only variable travels
    # through it, and the server it starts reads none from the environment.
    assert "RESEARCH_ULTRARAG_UI_PORT" not in body

    state = ui_launcher_state(project, project / ".research-rag")
    assert state["script_present"] is True
    assert state["link_state"] == "linked"
    assert state["link_target"] == LINK_TARGET


def test_existing_launcher_and_foreign_root_entry_are_never_overwritten(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    script = launcher_path(portable)
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n# hand-written\n", encoding="utf-8")
    link = project / "open-research-rag-ui.sh"
    link.write_text("# a real file the user owns\n", encoding="utf-8")

    result = ensure_ui_launcher(
        project_root=project,
        portable_root=portable,
        state_root=portable / "runtime",
        project_name="research-project",
    )

    assert script.read_text(encoding="utf-8") == "#!/bin/sh\n# hand-written\n"
    assert link.is_file() and not link.is_symlink()
    assert result["script_state"] == "present"
    assert result["link_state"] == "foreign_file"
    assert result["note"]

    file_state = ui_launcher_state(project, portable)
    assert file_state["link_state"] == "foreign_file"


def test_relocated_runtime_root_is_embedded_in_the_launcher(tmp_path: Path) -> None:
    project = _project(tmp_path)
    runtime_root = tmp_path / "relocated-runtime"
    resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_root=runtime_root,
    )

    body = launcher_path(project / ".research-rag").read_text(encoding="utf-8")
    assert f'RUNTIME_ROOT="{runtime_root.resolve()}"' in body
    assert "--runtime-root" in body


def test_ui_launcher_state_reports_absent_and_broken_links(tmp_path: Path) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    portable.mkdir(parents=True)

    assert ui_launcher_state(project, portable)["link_state"] == "absent"

    link = project / "open-research-rag-ui.sh"
    link.symlink_to(LINK_TARGET)
    assert ui_launcher_state(project, portable)["link_state"] == "broken_link"

    (portable / "bin").mkdir()
    launcher_path(portable).write_text("#!/bin/sh\n", encoding="utf-8")
    assert ui_launcher_state(project, portable)["link_state"] == "linked"


def test_launcher_script_renders_project_specific_values() -> None:
    body = launcher_script(
        project_root=Path("/tmp/example-project"),
        state_root=Path("/tmp/example-project/.research-rag/runtime"),
        project_name="example-project",
        port=5099,
    )

    assert body.startswith("#!/bin/sh\n")
    assert "example-project" in body
    assert "PORT=5099" in body
    assert 'RUNTIME_ROOT=""' in body
    assert 'PROJECT_ROOT="/tmp/example-project"' in body
    # The console script lives beside the running interpreter, not on a user's
    # PATH, so the launcher must embed an absolute command.
    assert f'UI_COMMAND="{default_ui_command()}"' in body
    assert os.path.isabs(default_ui_command())


def test_launcher_script_honors_an_explicit_ui_command() -> None:
    body = launcher_script(
        project_root=Path("/tmp/example-project"),
        state_root=Path("/tmp/example-project/.research-rag/runtime"),
        project_name="example-project",
        ui_command="/opt/research/bin/research-rag",
    )

    assert 'UI_COMMAND="/opt/research/bin/research-rag"' in body


@pytest.mark.skipif(
    shutil.which("setsid") is None or not Path("/proc").is_dir(),
    reason="the launcher process-group test needs Linux, /proc, and setsid",
)
def test_launcher_starts_and_stops_the_whole_process_group(tmp_path: Path) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    state = portable / "runtime"
    state.mkdir(parents=True)
    stub_directory = tmp_path / "stub-bin"
    stub_directory.mkdir()
    child_pid_file = tmp_path / "stub-child.pid"
    argument_file = tmp_path / "stub-arguments.txt"
    stub = stub_directory / "research-rag"
    stub.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$STUB_ARGUMENT_FILE"\n'
        "sleep 30 &\n"
        'printf \'%s\\n\' "$!" > "$STUB_CHILD_FILE"\n'
        f"{_BINDING_STUB}",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    ensure_ui_launcher(
        project_root=project,
        portable_root=portable,
        state_root=state,
        project_name="research-project",
        ui_command=str(stub),
    )
    script = project / "open-research-rag-ui.sh"
    environment = {
        **os.environ,
        "PATH": f"{stub_directory}{os.pathsep}{os.environ['PATH']}",
        "STUB_CHILD_FILE": str(child_pid_file),
        "STUB_ARGUMENT_FILE": str(argument_file),
    }

    try:
        started = subprocess.run(
            [str(script), "--port", "5099"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert started.returncode == 0, started.stderr
        assert "127.0.0.1:5099" in started.stdout
        arguments = argument_file.read_text(encoding="utf-8")
        assert "--project-root" in arguments
        assert "--port 5099" in arguments

        pid = int((state / "research-rag-ui.pid").read_text(encoding="utf-8").strip())
        child = int(child_pid_file.read_text(encoding="utf-8").strip())
        assert _alive(pid)
        assert _alive(child)

        stopped = subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert stopped.returncode == 0, stopped.stderr
        for _ in range(40):
            if not _alive(pid) and not _alive(child):
                break
            time.sleep(0.25)
        assert not _alive(pid)
        assert not _alive(child)
        assert not (state / "research-rag-ui.pid").exists()

        idle = subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert idle.returncode == 0
        assert "No app is running" in idle.stdout
    finally:
        subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            timeout=60,
            check=False,
        )


def _stub_launcher(
    tmp_path: Path,
    *,
    stub_body: str,
    name: str = "research-project",
) -> tuple[Path, Path, dict[str, str]]:
    """Generate a launcher whose UI command is a stub script."""

    project = tmp_path / name
    (project / "sources").mkdir(parents=True)
    portable = project / ".research-rag"
    state = portable / "runtime"
    state.mkdir(parents=True)
    stub_directory = tmp_path / f"stub-bin-{name}"
    stub_directory.mkdir()
    stub = stub_directory / "research-rag"
    stub.write_text(f"#!/bin/sh\n{stub_body}", encoding="utf-8")
    stub.chmod(0o755)
    ensure_ui_launcher(
        project_root=project,
        portable_root=portable,
        state_root=state,
        project_name=name,
        ui_command=str(stub),
    )
    return (
        project / "open-research-rag-ui.sh",
        state,
        {
            **os.environ,
            "PATH": f"{stub_directory}{os.pathsep}{os.environ['PATH']}",
        },
    )


def test_a_stale_port_lock_is_taken_over_rather_than_blocking(tmp_path: Path) -> None:
    """A launcher killed mid-choice must not lock its project out."""

    script, state, environment = _stub_launcher(
        tmp_path,
        stub_body=_BINDING_STUB,
    )
    lock = state / "research-rag-ui.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999999", encoding="utf-8")

    try:
        started = subprocess.run(
            [str(script), "--port", "5097"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert started.returncode == 0, started.stderr
        assert _alive(int((state / "research-rag-ui.pid").read_text(encoding="utf-8")))
        assert not lock.exists()
    finally:
        subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            timeout=60,
            check=False,
        )


def test_a_ui_that_exits_immediately_is_retried_and_leaves_no_record(
    tmp_path: Path,
) -> None:
    """A failed start must not leave a pid or a port naming something absent."""

    script, state, environment = _stub_launcher(tmp_path, stub_body="exit 0\n")
    started = subprocess.run(
        [str(script), "--port", "5096"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert started.returncode != 0
    assert "did not stay up" in started.stderr
    assert not (state / "research-rag-ui.pid").exists()
    assert not (state / "research-rag-ui.port").exists()
    assert not (state / "research-rag-ui.lock").exists()


def test_two_launchers_serve_two_ports_and_stop_independently(
    tmp_path: Path,
) -> None:
    """The port choice is atomic, so no launcher can take another's URL."""

    first_script, first_state, first_env = _stub_launcher(
        tmp_path, stub_body=_BINDING_STUB, name="first"
    )
    second_script, second_state, second_env = _stub_launcher(
        tmp_path, stub_body=_BINDING_STUB, name="second"
    )

    first = subprocess.Popen(
        [str(first_script)], env=first_env, stdout=subprocess.PIPE, text=True
    )
    second = subprocess.Popen(
        [str(second_script)], env=second_env, stdout=subprocess.PIPE, text=True
    )
    try:
        assert first.wait(timeout=60) == 0
        assert second.wait(timeout=60) == 0
        first_port = (first_state / "research-rag-ui.port").read_text(encoding="utf-8")
        second_port = (second_state / "research-rag-ui.port").read_text(
            encoding="utf-8"
        )
        assert first_port != second_port
        assert _alive(
            int((first_state / "research-rag-ui.pid").read_text(encoding="utf-8"))
        )
        assert _alive(
            int((second_state / "research-rag-ui.pid").read_text(encoding="utf-8"))
        )
    finally:
        for script, environment in (
            (first_script, first_env),
            (second_script, second_env),
        ):
            subprocess.run(
                [str(script), "--stop"],
                env=environment,
                capture_output=True,
                timeout=60,
                check=False,
            )


def test_stop_refuses_a_pid_that_is_not_this_projects_app(tmp_path: Path) -> None:
    """A reused pid number is not a reason to signal an unrelated process."""

    script, state, environment = _stub_launcher(tmp_path, stub_body="sleep 30\n")
    bystander = subprocess.Popen(["sleep", "30"])
    (state / "research-rag-ui.pid").write_text(str(bystander.pid), encoding="utf-8")
    (state / "research-rag-ui.port").write_text("5095", encoding="utf-8")
    try:
        stopped = subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert stopped.returncode == 0
        assert "is not this project's app" in stopped.stderr
        assert bystander.poll() is None
        assert not (state / "research-rag-ui.pid").exists()
        assert not (state / "research-rag-ui.port").exists()
    finally:
        bystander.terminate()
        bystander.wait(timeout=30)


@pytest.mark.integration
def test_the_generated_launcher_starts_and_stops_the_real_workspace(
    project: Path,
) -> None:
    """The generated script, run as written, must serve and stop this project.

    Every other test in this file replaces the launcher's body with a stub, so
    none of them can see that the command the script builds is one the installed
    parser accepts. That gap is what let the options sit after the subcommand
    through: the script ran, the port never opened, and every test passed.
    """

    config = resolve_config(project, vanilla_executable=sys.executable)
    generated = ensure_ui_launcher(
        project_root=config.project_root,
        portable_root=config.portable_root,
        state_root=config.state_root,
        project_name=config.project_name,
        runtime_root=config.runtime_root,
    )
    script = launcher_path(config.portable_root)
    assert generated["link_state"] in {"linked", "created"}
    state = config.state_root
    log = state / "logs" / "research-rag-ui.log"

    started = subprocess.run(
        [script], capture_output=True, text=True, timeout=180, check=False
    )
    try:
        assert started.returncode == 0, started.stderr or log.read_text(
            encoding="utf-8"
        )
        assert "Started the app for this project" in started.stdout
        assert "URL: http://127.0.0.1:" in started.stdout
        assert (state / "research-rag-ui.pid").is_file()
        port = int((state / "research-rag-ui.port").read_text(encoding="utf-8"))
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            body = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
        assert response.status == 200
        assert body["project_root"] == str(project)
    finally:
        stopped = subprocess.run(
            [script, "--stop"], capture_output=True, text=True, timeout=180, check=False
        )
        assert stopped.returncode == 0, stopped.stderr
    assert not (state / "research-rag-ui.pid").exists()
