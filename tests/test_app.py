"""The running app: one port, three front ends, and the clients attached to it.

These tests exercise a real `App` on a real loopback port, because the things that
break in this design are all things a unit test with a fake transport cannot see:
whether the agent endpoint answers a session at all once it is mounted rather than
served, whether the workspace can find the adapter its own lifespan installs, and
whether a disconnect actually ends a live session rather than only recording an
intent.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from research_rag.app import (
    CLIENT_NAME_HEADER,
    App,
    ClientRegistry,
    recorded_port,
    running_url,
)
from research_rag.config import resolve_config

# One app per test would claim a port each time; the fixtures below share one.
OPERATIONS = (
    "status",
    "ingest",
    "search",
    "find_source",
    "get_passage",
    "set_source_inclusion",
    "set_chunk_inclusion",
    "set_source_metadata",
)
RESOURCES = ("research://status",)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def get(url: str, path: str) -> tuple[int, Any]:
    """One HTTP request, off this event loop.

    The app is served on this loop, so a blocking request made on it would
    deadlock the very server it is calling.
    """

    _, _, port = url.rpartition(":")
    connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=30)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        try:
            return response.status, json.loads(body)
        except ValueError:
            return response.status, body
    finally:
        connection.close()


def post(url: str, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    connection = http.client.HTTPConnection(
        "127.0.0.1", int(url.rpartition(":")[2]), timeout=30
    )
    try:
        encoded = json.dumps(payload).encode("utf-8")
        connection.request(
            "POST", path, body=encoded, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


def post_ordered(
    url: str, path: str, payload: dict[str, Any], headers: dict[str, str]
) -> tuple[int, Any]:
    """One POST whose headers are exactly what the caller asked for.

    The gates this app puts on a settings write are header gates, so a test that
    could not set a header would not be testing them.
    """

    connection = http.client.HTTPConnection(
        "127.0.0.1", int(url.rpartition(":")[2]), timeout=30
    )
    try:
        encoded = json.dumps(payload).encode("utf-8")
        connection.request("POST", path, body=encoded, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


async def wait_until_ready(app: App) -> None:
    for _ in range(400):
        if app.ready:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the app never became ready: {app.error}")


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeService:
    """Stands in for `ResearchService` so these tests open no gateway.

    The app builds its own service; replacing it is what keeps a test about
    surfaces and clients from also being a test about a model download.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"operation": name, "arguments": arguments}

    async def status(self) -> dict[str, Any]:
        return self._record("status", {})

    async def ingest(self, *, force_recompute: bool = False) -> dict[str, Any]:
        return self._record("ingest", {"force_recompute": force_recompute})

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self._record("search", {"query": query, **arguments})

    async def find_source(self, query: str, **arguments: Any) -> dict[str, Any]:
        return self._record("find_source", {"query": query, **arguments})

    async def get_passage(
        self, chunk_id: str, *, context_chunks: int = 1
    ) -> dict[str, Any]:
        return self._record(
            "get_passage", {"chunk_id": chunk_id, "context_chunks": context_chunks}
        )

    async def set_source_inclusion(self, **arguments: Any) -> dict[str, Any]:
        return self._record("set_source_inclusion", arguments)

    async def set_source_metadata(self, **arguments: Any) -> dict[str, Any]:
        return self._record("set_source_metadata", arguments)

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return self._record("use_generation", {"generation_id": generation_id})

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return self._record(
            "remove_generation", {"generation_id": generation_id, "confirm": confirm}
        )

    async def settings_read(self) -> dict[str, Any]:
        return self._record("settings_read", {})

    async def settings_write(
        self, values: dict[str, Any], *, expected_revision: str, confirm: bool = False
    ) -> dict[str, Any]:
        return self._record(
            "settings_write",
            {
                "values": values,
                "expected_revision": expected_revision,
                "confirm": confirm,
            },
        )


async def test_the_agent_surface_answers_on_the_workspaces_port(project: Path) -> None:
    """One port, one service, and the agent gets the same operations as always."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        async with Client(app.mcp_url, timeout=30) as client:
            assert tuple(tool.name for tool in await client.list_tools()) == OPERATIONS
            assert tuple(str(item.uri) for item in await client.list_resources()) == (
                RESOURCES
            )
            answer = await client.call_tool("status", {})
            # The app's own state travels with the project's, so an agent knows
            # where the workspace is and how many other agents are attached.
            assert answer.data["ui_url"] == app.url
            assert answer.data["mcp_url"].endswith("/mcp")
    finally:
        await app.stop()
    assert app.ready is False


async def test_the_workspace_answers_on_the_same_port(project: Path) -> None:
    """The workspace is a route on the app, not a process beside it."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(get, app.url, "/api/health")
        assert code == 200
        assert body["project_root"] == str(project)
        code, body = await asyncio.to_thread(get, app.url, "/api/ui")
        assert code == 200
        assert body["application_name"] == "Research RAG"
    finally:
        await app.stop()


async def test_the_control_api_and_the_agent_surface_share_one_service(
    project: Path,
) -> None:
    """A passage the command line retrieves is the passage the agent retrieves."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    fake = FakeService()
    app.service = fake  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post, app.url, "/control/search", {"query": "heron", "top_k": 3}
        )
        assert code == 200
        # The control API fills every filter with its omitted default, so the two
        # surfaces hand the service the same call rather than the same intent.
        through_control = body["arguments"]

        async with Client(app.mcp_url, timeout=30) as client:
            await client.call_tool("search", {"query": "heron", "top_k": 3})

        # The same arguments, from two surfaces, through one service: a second
        # service would appear here as a second recorded call per operation.
        assert [name for name, _ in fake.calls] == ["search", "search"]
        through_tool = dict(fake.calls[1][1])
        for optional in (
            "categories_any",
            "projects_any",
            "keywords",
            "languages_any",
            "authors_any",
            "titles_any",
            "source_ids",
            "exclude_source_ids",
        ):
            # The agent's tool declares the filters it accepts; a surface that
            # forwarded the ones it does not would be sending values nothing acts on.
            assert optional not in through_tool or through_tool[optional] is None
        assert through_tool["query"] == through_control["query"]
        assert through_tool["top_k"] == through_control["top_k"]
    finally:
        await app.stop()


async def test_a_control_settings_write_reaches_the_one_service(project: Path) -> None:
    """The command line and the workspace change one project's settings."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    fake = FakeService()
    app.service = fake  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(get, app.url, "/control/settings")
        assert code == 200
        assert body["operation"] == "settings_read"
        code, body = await asyncio.to_thread(
            post,
            app.url,
            "/control/settings",
            {
                "values": {"retrieval.rrf_k": 40},
                "expected_revision": "rev-1",
                "confirm": True,
            },
        )
        assert code == 200
        assert body["arguments"]["values"] == {"retrieval.rrf_k": 40}
        assert body["arguments"]["confirm"] is True
        assert [name for name, _ in fake.calls] == ["settings_read", "settings_write"]
    finally:
        await app.stop()


async def test_a_control_settings_write_is_refused_from_another_site(
    project: Path,
) -> None:
    """A settings write changes what the next build records, so origin matters."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    fake = FakeService()
    app.service = fake  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post_ordered,
            app.url,
            "/control/settings",
            {"values": {"retrieval.rrf_k": 40}, "expected_revision": "r"},
            {
                "Content-Type": "application/json",
                "Sec-Fetch-Site": "cross-site",
            },
        )
        assert code == 403
        assert "Cross-origin" in body["error"]
        # A page on another site must not be able to make the app record anything.
        assert fake.calls == []

        host = app.url.rpartition(":")[2]
        code, body = await asyncio.to_thread(
            post_ordered,
            app.url,
            "/control/settings",
            {"values": {"retrieval.rrf_k": 40}, "expected_revision": "r"},
            {
                "Content-Type": "application/json",
                "Origin": f"http://elsewhere.example:{host}",
            },
        )
        assert code == 403
        assert fake.calls == []
    finally:
        await app.stop()


async def test_a_control_settings_write_requires_a_json_body(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    fake = FakeService()
    app.service = fake  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post_ordered,
            app.url,
            "/control/settings",
            {"values": {"retrieval.rrf_k": 40}, "expected_revision": "r"},
            {"Content-Type": "text/plain"},
        )
        assert code == 415
        assert "application/json" in body["error"]
        assert fake.calls == []
    finally:
        await app.stop()


async def test_a_control_settings_write_takes_no_path_in_its_body(
    project: Path,
) -> None:
    """The file a write lands in is the project's own, and no body may move it."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    fake = FakeService()
    app.service = fake  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post,
            app.url,
            "/control/settings",
            {
                "values": {"retrieval.rrf_k": 40},
                "expected_revision": "r",
                "confirm": True,
                "path": "/etc/research-rag/config.toml",
            },
        )
        assert code == 400
        assert "path" in body["error"]
        assert fake.calls == []
    finally:
        await app.stop()


async def test_the_agent_surface_declares_the_schema_an_agent_reads(
    project: Path,
) -> None:
    """The names are pinned elsewhere; this pins what an agent is actually told.

    A tool that keeps its name while losing a parameter is invisible to a name
    check and fatal to a client. A tool that stops saying it is destructive turns
    a call an agent must confirm into one it may fire without asking.
    """

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        async with Client(app.mcp_url, timeout=30) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert tuple(tools) == OPERATIONS
            assert tuple(tools["search"].inputSchema["properties"]) == (
                "query",
                "top_k",
                "categories_any",
                "projects_any",
                "keywords",
                "languages_any",
                "authors_any",
                "titles_any",
                "source_ids",
                "exclude_source_ids",
            )
            assert set(tools["search"].inputSchema["required"]) == {"query"}
            assert set(tools["ingest"].inputSchema["properties"]) == {"force_recompute"}
            # A passage decision takes its identifier, its flag, and its reason,
            # and nothing else: no generation to name, no way to exclude a batch
            # in one call, and no reason-free way to remove a passage.
            assert set(tools["set_chunk_inclusion"].inputSchema["properties"]) == {
                "chunk_id",
                "included",
                "reason",
            }
            assert set(tools["set_chunk_inclusion"].inputSchema["required"]) == {
                "chunk_id",
                "included",
            }
            assert "included" in tools["set_source_inclusion"].inputSchema["required"]
            # Only the two operations that change a project's review state are
            # announced as writes, and only the exclusion is announced as
            # destructive: a client that prompts on the wrong one of these
            # either nags a reader or fires a removal unasked.
            assert tools["status"].annotations.readOnlyHint is True
            assert tools["search"].annotations.readOnlyHint is True
            assert tools["get_passage"].annotations.readOnlyHint is True
            assert tools["ingest"].annotations.readOnlyHint is False
            assert tools["set_source_metadata"].annotations.readOnlyHint is False
            assert tools["set_source_inclusion"].annotations.destructiveHint is True
            assert tools["set_chunk_inclusion"].annotations.destructiveHint is True
            assert tools["set_chunk_inclusion"].annotations.readOnlyHint is False
            assert tools["set_source_metadata"].annotations.destructiveHint is False
            instructions = client.initialize_result.instructions or ""
            assert "status" in instructions
    finally:
        await app.stop()


async def test_the_control_api_moves_a_generation_the_way_the_service_does(
    project: Path,
) -> None:
    """The control API is a request path, not a second writer.

    The confirmation crosses the wire because the command line already had to
    repeat the id; a control surface that dropped the repeat would delete a
    generation a person did not choose.
    """

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    service = FakeService()
    app.service = service  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        selected = await asyncio.to_thread(
            post,
            app.url,
            "/control/generations/use",
            {"generation_id": "20260930T191235Z-45608dc5"},
        )
        assert selected[0] == 200
        removed = await asyncio.to_thread(
            post,
            app.url,
            "/control/generations/remove",
            {
                "generation_id": "20260930T191235Z-45608dc5",
                "confirm": "20260930T191235Z-45608dc5",
            },
        )
        assert removed[0] == 200
    finally:
        await app.stop()
    assert ("use_generation", {"generation_id": "20260930T191235Z-45608dc5"}) in (
        service.calls
    )
    assert (
        "remove_generation",
        {
            "generation_id": "20260930T191235Z-45608dc5",
            "confirm": "20260930T191235Z-45608dc5",
        },
    ) in service.calls


async def test_the_control_api_refuses_a_generation_request_it_cannot_serve(
    project: Path,
) -> None:
    """A removal that arrives without a confirmation is refused, not defaulted."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    service = FakeService()
    app.service = service  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        for path, payload, expected in (
            ("/control/generations/use", {"generation_id": ""}, "generation_id"),
            ("/control/generations/use", {}, "generation_id"),
            ("/control/generations/remove", {"generation_id": "x"}, "confirm"),
            (
                "/control/generations/remove",
                {"generation_id": "x", "confirm": 7},
                "confirm",
            ),
        ):
            code, body = await asyncio.to_thread(post, app.url, path, payload)
            assert code == 400, path
            assert expected in body["error"]
    finally:
        await app.stop()
    assert not [name for name, _ in service.calls if "generation" in name]


async def test_the_control_api_refuses_a_search_it_cannot_serve(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post, app.url, "/control/search", {"query": "  "}
        )
        assert code == 400
        assert "query" in body["error"]
    finally:
        await app.stop()


async def test_the_control_api_names_a_client_it_does_not_have(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        code, body = await asyncio.to_thread(
            post, app.url, "/control/clients/nope/disconnect", {"reason": "because"}
        )
        assert code == 400
        assert "nope" in body["error"]
    finally:
        await app.stop()


async def test_the_control_api_answers_the_account_and_the_client_entry(
    project: Path,
) -> None:
    """The command line can reach what the workspace's selector and entry read.

    Both are the app's own answers rather than a command-line reconstruction, so a
    terminal naming a project and a browser naming the same one cannot differ.
    """

    from research_rag import registry
    from research_rag.app import PID_FILE, PORT_FILE
    from research_rag.control import Control
    from research_rag.doctor import mcp_entry_block

    config = resolve_config(project, vanilla_executable=sys.executable)
    registry.register(config.project_id, config.project_name, config.project_root)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        # The launcher is what records the address; an app started in process has
        # to be recorded the same way before a probe can find it, and this is the
        # pair of files the launcher writes.
        (config.state_root / PORT_FILE).write_text(str(app.port), encoding="utf-8")
        (config.state_root / PID_FILE).write_text(str(os.getpid()), encoding="utf-8")

        code, body = await asyncio.to_thread(get, app.url, "/control/projects")
        assert code == 200
        assert [entry["project_name"] for entry in body["projects"]] == [
            config.project_name
        ]
        # This project's own app is the one serving the request, and it is named
        # with the address a reader can open.
        assert body["projects"][0]["app"] == {
            "attached_to": None,
            "running": True,
            "url": app.url,
            "port": app.port,
        }
        assert body["projects"][0]["attached_clients"] == 0

        code, entry = await asyncio.to_thread(get, app.url, "/control/agent-entry")
        assert code == 200
        assert entry == {"entry": mcp_entry_block(config)}

        # `Control` is synchronous, so it is used off the loop: the listing a
        # reader asks for probes each project's own app, and a probe made on the
        # loop would wait for an app that is itself waiting for the probe.
        def through_control() -> tuple[dict[str, Any], dict[str, Any]]:
            with Control(app.url) as handle:
                return handle.projects(), handle.agent_entry()

        projects, entry_through_control = await asyncio.to_thread(through_control)
        assert projects["project_count"] == body["project_count"]
        assert entry_through_control == entry
    finally:
        await app.stop()


async def test_a_named_client_is_listed_once_and_can_be_disconnected(
    project: Path,
) -> None:
    """Connect and disconnect is the part of the contract a reader can watch."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    app = App(config, port=free_port())
    app.service = FakeService()  # type: ignore[assignment]
    await app.start()
    try:
        await wait_until_ready(app)
        transport = StreamableHttpTransport(
            app.mcp_url, headers={CLIENT_NAME_HEADER: "reader-agent"}
        )
        async with Client(transport) as client:
            answer = await client.call_tool("status", {})
            # The MCP SDK opens a notification stream before the session exists,
            # so a sighting is recorded too; it is a connection, not a client.
            assert answer.data["mcp_clients"] == 1

            code, listing = await asyncio.to_thread(get, app.url, "/control/clients")
            assert code == 200
            attached = [entry for entry in listing["clients"] if entry["attached"]]
            assert [entry["name"] for entry in attached] == ["reader-agent"]
            session = attached[0]["session_id"]

            code, dropped = await asyncio.to_thread(
                post,
                app.url,
                f"/control/clients/{session}/disconnect",
                {"reason": "Disconnected by request."},
            )
            assert code == 200
            assert dropped["disconnected"]["attached"] is False

            # The app stops counting it the moment it is dropped, which is what
            # a reader watches. The gate test below is where the refusal itself
            # is pinned, because the wording an MCP SDK reports for a dead
            # session is the SDK's own and is not a contract here.
            code, listing = await asyncio.to_thread(get, app.url, "/control/clients")
            assert code == 200
            entry = next(
                item for item in listing["clients"] if item["session_id"] == session
            )
            assert entry["attached"] is False
            assert entry["detached_reason"] == "Disconnected by request."
    finally:
        await app.stop()


def test_a_port_claim_is_exclusive_and_released(project: Path) -> None:
    """The claim is the bind and the listen, so a race is a loser rather than a pair."""

    config = resolve_config(project, vanilla_executable=sys.executable)
    port = free_port()

    async def scenario() -> None:
        first = App(config, port=port)
        first.service = FakeService()  # type: ignore[assignment]
        await first.start()
        await wait_until_ready(first)
        try:
            second = App(config, port=port)
            second.service = FakeService()  # type: ignore[assignment]
            await second.start()
            assert second.ready is False
            assert second.error is not None
            assert "already in use" in second.error
            assert str(port) in second.error
        finally:
            await first.stop()

        third = App(config, port=port)
        third.service = FakeService()  # type: ignore[assignment]
        await third.start()
        try:
            await wait_until_ready(third)
            assert third.error is None
        finally:
            await third.stop()

    asyncio.run(scenario())


def test_a_port_outside_the_range_is_refused(project: Path) -> None:
    from research_rag.config import ConfigurationError

    config = resolve_config(project)
    for port in (0, 65536, -1):
        with pytest.raises(ConfigurationError):
            App(config, port=port)


def test_the_registry_counts_a_client_and_not_its_connections() -> None:
    """A sighting is not a session, and counting both reports one agent twice."""

    registry = ClientRegistry()

    class Request:
        """One request, with the peer a real connection would arrive from."""

        def __init__(self, session: str | None, name: str, port: int) -> None:
            self.headers = {
                "mcp-session-id": session or "",
                CLIENT_NAME_HEADER: name,
            }
            self.client = type("Peer", (), {"host": "127.0.0.1", "port": port})()

    registry.observe(Request(None, "agent", 5000))  # initialize
    registry.observe(Request(None, "agent", 5001))  # the notification stream
    registry.observe(Request("s-1", "agent", 5000))

    report = registry.report()
    # Two connections, one client: the sightings are folded onto the session by
    # the name the client declared, and the row says how many streams it has.
    assert len(report) == 1
    assert report[0]["attached"] is True
    assert report[0]["pending"] is False
    assert report[0]["streams"] == 3
    assert report[0]["requests"] == 3
    assert len(registry.attached) == 1


def test_two_clients_with_different_names_stay_two_rows() -> None:
    registry = ClientRegistry()

    class Request:
        def __init__(self, session: str | None, name: str, port: int) -> None:
            self.headers = {
                "mcp-session-id": session or "",
                CLIENT_NAME_HEADER: name,
            }
            self.client = type("Peer", (), {"host": "127.0.0.1", "port": port})()

    registry.observe(Request(None, "reader", 5000))
    registry.observe(Request(None, "reader", 5001))
    registry.observe(Request("s-1", "reader", 5000))
    registry.observe(Request(None, "writer", 5002))
    registry.observe(Request("s-2", "writer", 5002))

    assert [(entry["name"], entry["streams"]) for entry in registry.report()] == [
        ("reader", 3),
        ("writer", 2),
    ]


async def test_a_dropped_session_never_reaches_the_tools() -> None:
    """The gate is the only place a disconnect can be enforced."""

    from research_rag.app import ClientGate

    reached: list[str] = []

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["path"])

    registry = ClientRegistry()
    registry._touch("s-9", "agent")
    registry.disconnect("s-9", "Disconnected by request.")
    gate = ClientGate(downstream, registry)

    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "POST",
        "headers": [(b"mcp-session-id", b"s-9")],
    }
    sent: list[dict[str, Any]] = []
    body = bytearray()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        body.extend(message.get("body") or b"")

    await gate(scope, receive, send)

    assert reached == []
    assert sent[0]["status"] == 404
    assert b"disconnected from the app" in bytes(body)


def test_the_registry_refuses_to_disconnect_a_connection() -> None:
    registry = ClientRegistry()
    registry._touch("127.0.0.1:1", "", pending=True)

    with pytest.raises(Exception, match="no session yet"):
        registry.disconnect("127.0.0.1:1", "because")


def test_a_project_with_no_app_reports_no_app(project: Path) -> None:
    """`status` and `clients` must be able to say "down" without starting one."""

    config = resolve_config(project)
    assert recorded_port(config) is None
    assert running_url(config) is None
