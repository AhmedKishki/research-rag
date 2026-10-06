"""A client entry is copied between machines, so nothing in it names a directory.

The bridge resolves a project name through the account's own record, and a machine
holding no such project answers with a connection and a verdict rather than refusing
the session.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Self

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.providers.proxy import ProxyClient

import research_rag.surfaces.bridge as bridge_module
from research_rag.core.blocked_answers import uninitialised_status
from research_rag.project import registry
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from research_rag.surfaces import bridge
from research_rag.surfaces.cli import main
from research_rag.surfaces.mcp import create_blocked_mcp, create_mcp
from tests.surfaces.test_mcp_tools import _Service

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


class _Switch:
    """The app, as the bridge sees it: up or down, and who it answers as."""

    def __init__(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.up = False
        self.drops = 0
        self.calls: list[str] = []
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = _Service()
        outer = self

        async def connect() -> _Service:
            outer.calls.append("served")
            return service

        class _Flaky(ProxyClient):  # type: ignore[type-arg]
            """A session that the app closes under the caller."""

            async def __aenter__(self) -> Self:
                if outer.drops:
                    outer.drops -= 1
                    raise ConnectionResetError("the app went away mid-call")
                return await super().__aenter__()

        self.server = create_mcp(config, connect=connect, app_state=dict)
        monkeypatch.setattr(
            bridge_module.Backends,
            "_serving_url",
            lambda _backends, _config: "http://127.0.0.1:5051/mcp" if self.up else None,
        )
        monkeypatch.setattr(
            bridge_module.Backends,
            "_upstream",
            lambda _backends, _url, _config: _Flaky(self.server),
        )


def _bridge(project: Path) -> object:
    return bridge.build_bridge(
        "AI and fetishism",
        name="an-agent",
        identity={"project": "AI and fetishism", "pid": os.getpid()},
    )


async def test_the_bridge_holds_every_tool_whether_or_not_an_app_is_up(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order the agent and the app start in decides nothing.

    The bridge used to choose once, at launch: an app that was not up then left a
    stub with one tool for the life of the process, whatever started afterwards.
    """

    _initialised(project, "AI and fetishism")
    switch = _Switch(project, monkeypatch)

    async with Client(_bridge(project)) as client:
        down = {tool.name for tool in await client.list_tools()}
        blocked = (await client.call_tool("status")).data
        with pytest.raises(ToolError, match="start"):
            await client.call_tool("search", {"query": "labour"})
        assert switch.calls == []

        switch.up = True
        up = {tool.name for tool in await client.list_tools()}
        answered = (await client.call_tool("search", {"query": "labour"})).data
        status = (await client.call_tool("status")).data

        switch.up = False
        with pytest.raises(ToolError, match="start"):
            await client.call_tool("get_passage", {"chunk_id": "chk_one"})

    assert down == up and len(down) == 8
    assert blocked["blocked_by"][0]["check"] == "app.serving"
    assert "AI and fetishism" in blocked["blocked_by"][0]["remedy"]
    assert answered["hits"][0]["chunk_id"] == "chk_one"
    assert status["ready"] is True


async def test_a_read_dropped_mid_call_is_repeated_and_a_write_is_not(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection that closed is named, and only a read is tried again."""

    _initialised(project, "AI and fetishism")
    switch = _Switch(project, monkeypatch)
    switch.up = True

    async with Client(_bridge(project)) as client:
        switch.drops = 1
        repeated = (await client.call_tool("search", {"query": "labour"})).data
        switch.drops = 1
        with pytest.raises(ToolError, match="still up") as write:
            await client.call_tool(
                "set_source_inclusion",
                {"source_path": "a.pdf", "included": False},
            )
        assert "status" in str(write.value)

    assert repeated["hits"][0]["chunk_id"] == "chk_one"


async def test_a_project_initialised_after_the_bridge_started_is_found(
    account: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`init` run after the agent connected needs no reconnect to be seen."""

    async with Client(_bridge(project)) as client:
        before = (await client.call_tool("status")).data
        _initialised(project, "AI and fetishism")
        after = (await client.call_tool("status")).data

    assert before["blocked_by"][0]["check"] == "project.initialised"
    assert after["blocked_by"][0]["check"] == "app.serving"


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
