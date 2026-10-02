"""A client entry is copied between machines, so nothing in it names a directory.

The bridge resolves a project name through the account's own record, and a machine
holding no such project answers with a connection and a verdict rather than refusing
the session.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client

import research_rag.surfaces.bridge as bridge_module
from research_rag.core.blocked_answers import not_served_status, uninitialised_status
from research_rag.project import registry
from research_rag.project.config import resolve_config
from research_rag.project.support import ResearchError
from research_rag.surfaces import bridge
from research_rag.surfaces.cli import main
from research_rag.surfaces.mcp import create_blocked_mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the account's project record at a throwaway directory."""

    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    return home / "config" / "research-rag" / "projects.json"


def _initialised(root: Path, name: str) -> str:
    """Register a project the way `init` does."""

    config = resolve_config(root, project_name=name)
    registry.register(config.project_id, config.project_name, root)
    return config.project_name


def test_a_registered_name_resolves_to_its_own_project(
    account: Path, project: Path
) -> None:
    _initialised(project, "AI and fetishism")

    config, reason = bridge.resolve_project("ai and fetishism")

    assert reason is None
    assert config is not None
    assert config.project_root == project.resolve()
    assert config.project_name == "AI and fetishism"


def test_a_name_nothing_on_this_machine_holds_names_no_project(
    account: Path, project: Path
) -> None:
    """The fix for a name that resolves nowhere is initialising it, not a path."""

    _initialised(project, "Something Else")

    config, reason = bridge.resolve_project("ai and fetishism")

    assert config is None
    assert reason is not None
    assert "ai and fetishism" in reason
    assert "init" in reason
    assert str(project) not in reason


def test_a_record_whose_directory_is_gone_says_where_it_pointed(
    account: Path, project: Path
) -> None:
    """A record is a pointer, and a project can be moved while it stays."""

    _initialised(project, "AI and fetishism")
    (project / ".research-rag" / "project.json").unlink()

    config, reason = bridge.resolve_project("AI and fetishism")

    assert config is None
    assert reason is not None
    assert str(project.resolve()) in reason
    assert "initialises one there" in reason


def test_a_name_two_projects_share_is_refused(account: Path, project: Path) -> None:
    """No amount of initialising settles which of two directories an entry means."""

    _initialised(project, "Shared Name")
    other = project.parent / "another"
    (other / "sources").mkdir(parents=True)
    registry.register("pid-two", "Shared Name", other)

    with pytest.raises(ResearchError, match="more than one project"):
        bridge.resolve_project("Shared Name")


async def test_an_uninitialised_project_answers_status_and_nothing_else() -> None:
    """One tool, one condition, and the command that closes it."""

    server = create_blocked_mcp(
        "ai-and-fetishism",
        "No project is initialised under that name here.",
        config=None,
    )

    async with Client(server) as client:
        tools = {tool.name for tool in await client.list_tools()}
        answer = await client.call_tool("status", {})

    assert tools == {"status"}
    assert answer.data["ready"] is False
    assert answer.data["project_initialised"] is False
    # Nothing here can be ingested into, so no call is asked of the agent.
    assert "requires" not in answer.data
    blocker = answer.data["blocked_by"][0]
    assert blocker["check"] == "project.initialised"
    assert "init" in blocker["remedy"]
    assert "ai-and-fetishism" in blocker["remedy"]


async def test_the_uninitialised_surface_serves_the_status_resource() -> None:
    server = create_blocked_mcp(
        "ai-and-fetishism",
        "No project is initialised under that name here.",
        config=None,
    )

    async with Client(server) as client:
        answer = await client.read_resource("research://status")

    assert "ai-and-fetishism" in answer[0].text


def test_the_bridge_proxies_a_project_an_app_is_serving(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "AI and fetishism")
    served: dict[str, object] = {}

    class _Proxy:
        """Stands in for the transport, so no socket is opened here."""

        def run(self, **kwargs: object) -> None:
            served["run"] = kwargs

    monkeypatch.setattr(bridge, "is_serving", lambda config: True)
    monkeypatch.setattr(
        bridge,
        "connect",
        lambda config, **_: nullcontext(
            SimpleNamespace(base_url="http://127.0.0.1:5051")
        ),
    )
    monkeypatch.setattr(
        bridge,
        "build_proxy",
        lambda url, *, name, identity: (
            served.update(url=url, name=name, identity=identity) or _Proxy()
        ),
    )

    bridge.run("AI and fetishism", name="an-agent")

    assert served["url"] == "http://127.0.0.1:5051/mcp"
    assert served["name"] == "an-agent"
    # The app cannot read the client that started the bridge, so the bridge sends
    # what it inherited: the project it was asked for and the process it is.
    assert served["identity"]["project"] == "AI and fetishism"
    assert served["identity"]["pid"] == os.getpid()
    assert served["run"] == {"transport": "stdio", "show_banner": False}


def test_the_bridge_answers_instead_of_refusing_a_missing_project(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialised(project, "Something Else")
    served: dict[str, object] = {}

    monkeypatch.setattr(
        bridge_module,
        "serve_blocked",
        lambda name, reason, *, config: served.update(
            name=name, reason=reason, config=config
        ),
    )

    bridge.run("ai and fetishism")

    assert served["name"] == "ai and fetishism"
    assert "init" in str(served["reason"])
    assert served["config"] is None


def test_the_bridge_answers_a_project_no_app_is_serving_and_starts_nothing(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app belongs to a terminal, so a client that finds none is told where."""

    _initialised(project, "AI and fetishism")
    served: dict[str, object] = {}

    monkeypatch.setattr(bridge, "is_serving", lambda config: False)
    monkeypatch.setattr(
        bridge_module,
        "serve_blocked",
        lambda name, reason, *, config: served.update(
            name=name, reason=reason, config=config
        ),
    )
    monkeypatch.setattr(
        bridge,
        "build_proxy",
        lambda url, *, name, identity: pytest.fail(
            "no proxy may be built with no app serving"
        ),
    )

    bridge.run("AI and fetishism")

    assert served["name"] == "AI and fetishism"
    assert served["config"] is not None
    payload = not_served_status(
        served["config"],
        "An app runs in a terminal and ends when that terminal closes.",
    )
    assert payload["project_initialised"] is True
    assert payload["blocked_by"][0]["check"] == "app.serving"
    # The remedy names the project, so no machine's directory reaches an agent.
    assert "start" in payload["blocked_by"][0]["remedy"]
    assert "AI and fetishism" in payload["blocked_by"][0]["remedy"]


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"VSCODE_PID": "991"}, "Visual Studio Code"),
        ({"TERM_PROGRAM": "vscode"}, "Visual Studio Code"),
        ({"TERM_PROGRAM": "WezTerm"}, "WezTerm"),
        ({"TERM": "xterm-256color"}, "a xterm-256color terminal"),
        ({"TERM": "dumb"}, ""),
        ({}, ""),
    ],
)
def test_a_bridge_reports_the_program_it_was_started_by(
    environment: dict[str, str], expected: str
) -> None:
    """A client entry names a command, so the app cannot tell one agent from another."""

    identity = bridge.client_identity("AI and fetishism", environ=environment)

    assert identity["host"]["program"] == expected


def test_a_bridge_reports_a_remote_shell_and_its_own_project() -> None:
    identity = bridge.client_identity(
        "AI and fetishism",
        environ={"SSH_CONNECTION": "10.0.0.2 51000 10.0.0.9 22", "TMUX": "/tmp/tmux"},
    )

    assert identity["project"] == "AI and fetishism"
    assert identity["pid"] == os.getpid()
    assert identity["host"]["ssh"] is True
    assert identity["host"]["tmux"] is True


def test_the_mcp_command_hands_the_bridge_a_name(
    account: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, object] = {}

    def _run(project_name: str, **kwargs: object) -> None:
        served.update(project_name=project_name, kwargs=kwargs)

    monkeypatch.setattr(bridge_module, "run", _run)

    main(["mcp", "--project-name", "AI and fetishism", "--client-name", "an-agent"])

    assert served["project_name"] == "AI and fetishism"
    assert served["kwargs"]["name"] == "an-agent"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--project-root", "/tmp/whatever", "mcp", "--project-name", "any"],
        ["--project", "any", "mcp", "--project-name", "any"],
    ],
)
def test_a_path_in_an_mcp_entry_is_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as refusal:
        main(arguments)

    assert "--project-name" in str(refusal.value)


def test_an_mcp_entry_with_no_project_name_is_refused() -> None:
    with pytest.raises(SystemExit, match="RESEARCH_RAG_PROJECT_NAME"):
        main(["mcp"])


def test_a_client_that_can_only_set_an_environment_still_names_a_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some clients offer an environment block and no argument list."""

    served: dict[str, object] = {}

    def _run(project_name: str, **kwargs: object) -> None:
        served.update(project_name=project_name)

    monkeypatch.setattr(bridge_module, "run", _run)
    monkeypatch.setenv("RESEARCH_RAG_PROJECT_NAME", "ai-and-fetishism")

    main(["mcp"])

    assert served["project_name"] == "ai-and-fetishism"


def test_the_status_answer_names_the_command_that_creates_the_project() -> None:
    """The verdict and the remedy are built once, where a status answer is built."""

    answer = uninitialised_status("ai-and-fetishism", "Nothing here yet.")

    assert json.dumps(answer)
    assert answer["blocked_by"][0]["remedy"] == (
        "research-rag --project-root /path/to/project init --name ai-and-fetishism"
    )
