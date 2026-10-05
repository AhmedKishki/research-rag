"""The clients view, the SQL console, the sidebar that holds them, and its styles.

A host may serve this workspace and still not be a server — a stdio-only process
has no other client to report — so both the capability and the two routes are off
until a host turns them on, and a host that turns them on without the methods is
told so rather than raising.

A store this library cannot know about is the same case: the SQL panel ships
hidden and only an adapter that declares `sql_console` fills it, because the
scopes it offers and the statements it will run are the adapter's decisions.

One installation serving several projects is the same case again: the selector
and the client entry are the host's facts, so both ship behind a flag and neither
is composed here.

The navigation is a sidebar rather than a strip, and the page is built on a
spacing and type scale rather than on values typed into each rule. Both are here
because a reader who cannot see the page described as cramped is not served by an
assertion about a class name.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_rag.surfaces.workspace import (
    SourceFile,
    UICapabilities,
    UIProfile,
    UIRequestError,
    create_ui_app,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
# The test transport reports itself as `testclient`, and its Host is
# `testserver`; a real request arrives from a loopback peer addressed to a
# loopback name, which is what the write guard requires.
LOOPBACK_TEST_CLIENT = ("127.0.0.1", 50000)


class ClientControlAdapter:
    """An adapter that can report and end the clients attached to its host."""

    def __init__(self) -> None:
        self.dropped: list[tuple[str, str | None]] = []

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def call(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if operation == "status":
            return {"ready": False, "project_root": "/p", "project_name": "p"}
        if operation == "list_sources":
            return {"ready": False, "source_count": 0, "sources": []}
        raise UIRequestError(f"Research operation {operation!r} is not available", 404)

    async def source_file(self, source_path: str) -> SourceFile:  # pragma: no cover
        raise UIRequestError("not used")

    async def list_clients(self) -> list[Mapping[str, Any]]:
        return [
            {
                "session_id": "s-1",
                "name": "reader-agent",
                "label": "reader-agent",
                "transport": "stdio",
                "declared_name": "reader-agent",
                "attached": True,
                "requests": 7,
                "sessions": 3,
                "idle_seconds": 0.2,
                "detached_reason": None,
                "peer": "127.0.0.1:5000",
                "user_agent": "",
                "identity": {
                    "agent": "reader-agent",
                    "project": "p",
                    "pid": 991,
                    "cwd": "/home/reader/p",
                    "host": {
                        "program": "Visual Studio Code",
                        "term": "",
                        "ssh": False,
                        "tmux": False,
                    },
                },
            }
        ]

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]:
        self.dropped.append((session_id, reason))
        if session_id == "gone":
            raise UIRequestError("No client is attached with session gone")
        return {"session_id": session_id, "attached": False, "reason": reason}


def _host(adapter: Any, *, enabled: bool = True) -> TestClient:
    return TestClient(
        create_ui_app(
            profile=UIProfile(
                application_name="Client host",
                navigation_label="Views",
                capabilities=UICapabilities(clients=enabled, retrieval_modes=False),
            ),
            adapter=adapter,
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


def _project_host() -> TestClient:
    """A host serving several projects from one installation."""

    class ProjectHost(ClientControlAdapter):
        async def call(
            self, operation: str, arguments: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            if operation == "list_projects":
                return {
                    "projects": [
                        {
                            "project_name": "thesis",
                            "project_root": "/projects/thesis",
                            "running": True,
                            "url": "http://127.0.0.1:5051",
                            "attached_clients": 1,
                        },
                        {
                            "project_name": "archive",
                            "project_root": "/projects/archive",
                            "running": False,
                            "url": None,
                            "attached_clients": 0,
                        },
                    ],
                    "current": "thesis",
                    "message": "",
                }
            if operation == "agent_entry":
                return {"entry": '{\n  "mcp": {}\n}\n'}
            return await super().call(operation, arguments)

    return TestClient(
        create_ui_app(
            profile=UIProfile(
                application_name="Project host",
                project_start_command="research-rag --project {project} ui",
                capabilities=UICapabilities(
                    clients=True, projects=True, agent_entry=True
                ),
            ),
            adapter=ProjectHost(),
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


class UpdateHost(ClientControlAdapter):
    """A host that answers the release question and records what it was asked.

    Every operation is listed so a request this page has no business making shows
    up as an unknown operation rather than passing unnoticed.
    """

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.writes: list[str] = []
        self.payload = dict(
            payload
            or {
                "installed_version": "1.1.0",
                "update_available": True,
                "offline": False,
                "release": {
                    "version": "1.2.0",
                    "tag_name": "v1.2.0",
                    "name": "research-rag 1.2.0",
                    "published_at": "2026-10-03T00:00:00Z",
                    "html_url": (
                        "https://github.com/AhmedKishki/research-rag"
                        "/releases/tag/v1.2.0"
                    ),
                    "body": "## Fixed\n\n- One passage\n- Another",
                },
                "apply_command": "research-rag update --apply",
                "message": None,
            }
        )

    async def call(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append(operation)
        if operation == "check_updates":
            return self.payload
        return await super().call(operation, arguments)

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]:  # pragma: no cover - a write this page must not make
        self.writes.append("disconnect_client")
        return await super().disconnect_client(session_id, reason)


def _update_host(
    payload: Mapping[str, Any] | None = None,
) -> tuple[TestClient, UpdateHost]:
    """A host that serves the update view and nothing else it cannot answer."""

    adapter = UpdateHost(payload)
    return (
        TestClient(
            create_ui_app(
                profile=UIProfile(
                    application_name="Update host",
                    navigation_label="Views",
                    capabilities=UICapabilities(updates=True, retrieval_modes=False),
                ),
                adapter=adapter,
            ),
            base_url=LOOPBACK_BASE_URL,
            client=LOOPBACK_TEST_CLIENT,
        ),
        adapter,
    )


def client_css() -> str:
    with _host(ClientControlAdapter()) as client:
        return client.get("/assets/app.css").text


def test_a_host_can_report_its_attached_clients() -> None:
    with _host(ClientControlAdapter()) as client:
        response = client.get("/api/clients")
    assert response.status_code == 200
    clients = response.json()["clients"]
    assert [entry["name"] for entry in clients] == ["reader-agent"]
    assert clients[0]["requests"] == 7
    assert clients[0]["sessions"] == 3


def test_a_host_can_end_one_session_from_the_workspace() -> None:
    adapter = ClientControlAdapter()
    with _host(adapter) as client:
        response = client.post(
            "/api/clients/s-1/disconnect", json={"reason": "from the workspace"}
        )
        assert response.status_code == 200
        assert response.json()["attached"] is False
    assert adapter.dropped == [("s-1", "from the workspace")]


def test_the_clients_routes_are_absent_when_the_capability_is_off() -> None:
    """A memory-only or stdio-only host has nothing to report, so it says nothing."""

    adapter = ClientControlAdapter()
    with _host(adapter, enabled=False) as client:
        assert client.get("/api/clients").status_code == 404
        assert client.post("/api/clients/s-1/disconnect", json={}).status_code == 404
    assert adapter.dropped == []


def test_a_workspace_disconnect_is_same_origin_and_json_only() -> None:
    """Ending a session is a write, so it is held to the write rules."""

    adapter = ClientControlAdapter()
    with _host(adapter) as client:
        cross_origin = client.post(
            "/api/clients/s-1/disconnect",
            headers={"Origin": "https://example.com"},
            json={"reason": "x"},
        )
        form = client.post(
            "/api/clients/s-1/disconnect",
            content=b"reason=x",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    assert cross_origin.status_code == 403
    assert form.status_code == 415
    assert adapter.dropped == []


def test_a_host_that_advertises_clients_and_cannot_answer_says_501() -> None:
    """A capability without the methods is a host fact, not a traceback."""

    class Partial:
        async def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

        async def call(
            self, operation: str, arguments: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            if operation == "status":
                return {"ready": False, "project_root": "/p", "project_name": "p"}
            if operation == "list_sources":
                return {"ready": False, "source_count": 0, "sources": []}
            raise UIRequestError("no", 404)

    with _host(Partial()) as client:
        response = client.get("/api/clients")
    assert response.status_code == 501
    assert "cannot report" in response.json()["error"]


def test_a_host_refusal_keeps_its_own_status_and_reason() -> None:
    """A host that refuses a client request has said something a reader can act on.

    Translating it into a generic failure would throw the reason away, so the two
    client routes carry the same translation the operation routes do.
    """

    with _host(ClientControlAdapter()) as client:
        response = client.post("/api/clients/gone/disconnect", json={})
    assert response.status_code == 400
    assert "No client is attached" in response.json()["error"]


def test_the_disconnect_reason_must_be_a_string() -> None:
    adapter = ClientControlAdapter()
    with _host(adapter) as client:
        response = client.post("/api/clients/s-1/disconnect", json={"reason": 7})
    assert response.status_code == 400
    assert adapter.dropped == []


def test_the_mcp_view_carries_the_clients_panel() -> None:
    """The panel ships inside the MCP tab and is revealed by the capability.

    A tab is one panel's way in, so the tab carries the same capability the panel
    does: a tab whose panel the host does not serve would be a way into nothing.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert 'data-panel="mcp" data-capability="clients"' in page.text
    assert 'data-view="mcp" data-capability="clients"' in page.text
    assert 'id="client-chips"' in page.text
    assert 'id="agent-endpoint"' in page.text
    # The endpoint is free in the status payload, so the block reads it there
    # rather than asking the host a second question.
    assert 'id="agent-url"' in page.text
    assert "status.mcp_url" in script.text
    assert "This host reports no MCP endpoint" in script.text


def test_the_client_list_is_folded_away_and_says_how_many_clients_it_holds() -> None:
    """A machine runs a client per agent session, so the list is not the panel.

    The count is on the summary, because a reader who came for the count should
    not have to open the list to read it, and a reader who came for the list
    opens it with one click.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/")
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert '<details id="client-details" class="client-drawer">' in page.text
    assert 'id="client-count"' in page.text
    assert 'id="client-detail"' in page.text
    # A <details> with no open attribute starts closed, so nothing else has to
    # close it on load.
    assert "client-details" in page.text
    assert "open" not in page.text.split('id="client-details"')[1].split(">")[0]
    assert 'byId("client-count").textContent' in script
    assert 'byId("client-detail").textContent' in script
    assert ".client-drawer summary {" in css
    assert ".client-facts .fact-value," in css


def test_a_client_row_shows_who_the_client_is_and_where_it_is_running() -> None:
    """A name, and the program, directory, and process behind it."""

    with _host(ClientControlAdapter()) as client:
        script = client.get("/assets/app.js").text

    # The label is what the host says the client is, never a session id alone.
    assert "client.label || client.name" in script
    for fact in (
        "host.program",
        "host.ssh",
        "host.tmux",
        "identity.cwd",
        "identity.project",
        "identity.pid",
    ):
        assert fact in script
    # A stdio bridge reaches the app through a Python HTTP client, so its user
    # agent names that library rather than the agent behind it.
    assert 'client.transport !== "stdio" && client.user_agent' in script
    # Each fact carries its own label rather than being joined into one line of
    # separators, because a reader looking for the directory was reading for the
    # directory and not for the process id beside it.
    assert 'facts.push(["Program", host.program]);' in script
    assert 'facts.push(["Directory", identity.cwd]);' in script
    assert 'factList(facts, "fact-list client-facts")' in script
    # One client holds every session it opened, and the session id is shown whole
    # because it is what `disconnect` is given.
    assert '`${client.sessions || 1} open · id ${client.session_id || ""}`' in script
    # The host's own words say how a client is named, so a shared page cannot
    # name a variable only this host has.
    assert "profile.client_naming_hint" in script


class SqlConsoleHost:
    """A host that serves the workspace without exposing a store at all.

    No statement method, because this half of the capability is about what the
    page ships: the panel, its gating, and the envelope it renders.
    """

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def call(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if operation == "status":
            return {
                "ready": True,
                "project_root": "/p",
                "project_name": "p",
                "sql_scopes": [{"scope": "generations", "label": "Generations"}],
            }
        if operation == "list_sources":
            return {"ready": True, "source_count": 0, "sources": []}
        raise UIRequestError(f"Operation {operation!r} is not available", 404)

    async def source_file(self, source_path: str) -> SourceFile:  # pragma: no cover
        raise UIRequestError("not used")


def _sql_host(*, enabled: bool) -> TestClient:
    return TestClient(
        create_ui_app(
            profile=UIProfile(
                application_name="SQL host",
                capabilities=UICapabilities(sql_console=enabled),
            ),
            adapter=SqlConsoleHost(),
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


def test_the_status_view_carries_the_sql_console_panel() -> None:
    """The panel ships hidden and is revealed by the capability, not by a route."""

    with _sql_host(enabled=True) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert (
        'id="sql-console" class="panel-section" data-capability="sql_console" hidden'
        in page.text
    )
    assert 'id="sql-scope"' in page.text
    assert 'id="sql-statement"' in page.text
    assert 'id="sql-results"' in page.text
    assert 'hasCapability("sql_console")' in script.text
    # The listing of scopes is free in the status payload, so the panel needs no
    # read route of its own and the select is filled from what the host reported.
    assert "renderSqlConsole(status)" in script.text
    assert "status.sql_scopes || []" in script.text


def test_the_sql_console_is_off_until_an_adapter_declares_it() -> None:
    """A library cannot know which store an adapter keeps, so it offers none."""

    with _sql_host(enabled=False) as client:
        ui = client.get("/api/ui").json()
        page = client.get("/").text

    assert ui["capabilities"]["sql_console"] is False
    assert 'data-capability="sql_console" hidden' in page


def test_the_sql_console_renders_the_envelope_the_server_returned() -> None:
    """A grid is drawn from the columns and rows the adapter sent, and nothing else."""

    with _sql_host(enabled=True) as client:
        script = client.get("/assets/app.js").text
        page = client.get("/").text

    for field in (
        "payload.columns",
        "payload.rows",
        "payload.row_count",
        "payload.truncated",
    ):
        assert field in script
    assert "/api/sql/query" in script
    assert "renderSqlResult(result)" in script
    assert 'node("table", "sql-table")' in script
    # An execute reports what it changed, and a reindexed write is said to be
    # one, because the reader's next search reads different records.
    assert "result.rows_affected" in script
    assert "The server reindexed the change." in script
    # A write says so next to its own control rather than in a dialog the
    # capability cannot be disabled from.
    assert (
        "Execute changes stored records, and the server reindexes the change." in page
    )


def test_the_sql_console_says_when_there_is_no_scope_to_run_against() -> None:
    """A panel with an empty scope list has nothing to offer but says so."""

    with _sql_host(enabled=True) as client:
        script = client.get("/assets/app.js").text

    assert "This server reports no SQL scope" in script
    assert "message.hidden = scopes.length > 0" in script
    assert "select.disabled = !scopes.length" in script
    # Both controls wait for a scope and a statement, so neither sends an
    # empty request the route would only refuse.
    assert 'byId("sql-scope").value && byId("sql-statement").value.trim()' in script
    assert 'byId("sql-run-button").disabled = !ready' in script
    assert 'byId("sql-execute-button").disabled = !ready' in script
    # The run button is a submit, and clearing the busy state re-enables every
    # submit; the gate is re-applied there so a reader is not left with a live
    # button that could only be refused.
    busy = script.split("function setBusy(")[1].split("\n}\n")[0]
    assert "syncSqlControls();" in busy


class SettingsHost:
    """A host that reports two settings: one it can change and one it cannot.

    The second arrives unwritable, the way a value taken from the environment or
    the command line does, so the panel is drawn against both shapes.
    """

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def call(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if operation == "status":
            return {"ready": True, "project_root": "/p", "project_name": "p"}
        if operation == "list_sources":
            return {"ready": True, "source_count": 0, "sources": []}
        if operation == "settings_read":
            return {
                "revision": "rev-1",
                "sections": [
                    {
                        "key": "retrieval",
                        "title": "Retrieval",
                        "settings": [
                            {
                                "key": "retrieval.rrf_k",
                                "label": "Reciprocal rank fusion k",
                                "value": 60,
                                "default": 60,
                                "defaulted": False,
                                "kind": "int",
                                "layer": "retrieval",
                                "origin": "project",
                                "writable": True,
                                "cost": {
                                    "level": "none",
                                    "message": "Applies to the next search.",
                                },
                            },
                            {
                                "key": "retrieval.model",
                                "label": "Embedding model",
                                "value": "bge-small",
                                "default": "BAAI/bge-small-en-v1.5",
                                "defaulted": True,
                                "kind": "str",
                                "layer": "retrieval",
                                "origin": "environment",
                                "writable": False,
                                "cost": {
                                    "level": "model",
                                    "message": "Re-embeds the corpus.",
                                },
                            },
                        ],
                    }
                ],
                "default_file": "/opt/research-rag/default.toml",
                "message": "Settings read.",
            }
        if operation == "list_chunk_exclusions":
            return {"exclusions": [], "message": "No chunk is excluded."}
        raise UIRequestError(f"Operation {operation!r} is not available", 404)

    async def source_file(self, source_path: str) -> SourceFile:  # pragma: no cover
        raise UIRequestError("not used")


def _panel_host() -> TestClient:
    return TestClient(
        create_ui_app(
            profile=UIProfile(
                application_name="Settings host",
                capabilities=UICapabilities(settings=True, chunk_exclusion=True),
            ),
            adapter=SettingsHost(),
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


def test_the_config_tab_carries_the_settings_panel_and_its_confirmation() -> None:
    """The settings panel is the whole Config tab, not one block of the status."""

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert 'data-panel="config" data-capability="settings"' in page
    assert 'data-view="config" data-capability="settings"' in page
    assert 'id="settings-panel" class="settings-panel"' in page
    for element in ("settings-form", "settings-sections", "settings-submit"):
        assert f'id="{element}"' in page
    # A change that costs a regeneration is confirmed by a typed word rather than
    # by a click, so the dialog is part of what the panel ships.
    assert 'id="settings-confirm-dialog"' in page
    assert 'id="settings-confirm-word"' in page
    assert 'hasCapability("settings")' in script
    # Settings are the whole Config view. They are not repeated inside the status
    # view, so a reader who edits one sees one copy of every value.
    status_view = page.split('id="status-view"', 1)[1].split('id="config-view"', 1)[0]
    assert "settings-form" not in status_view
    assert "settings-panel" not in status_view


def test_the_settings_panel_reads_the_shape_the_server_sends() -> None:
    """A field follows the setting's kind, and an unwritable one is disabled.

    A range or a list of choices narrows the field when the server declares one
    and changes nothing when it does not; the value is still parsed as its kind
    and the decision stays the server's.
    """

    with _panel_host() as client:
        script = client.get("/assets/app.js").text

    assert 'control.type = "checkbox"' in script
    assert 'control.type = "number"' in script
    assert 'control.type = "text"' in script
    assert "control.disabled = !setting.writable" in script
    assert "Set by ${setting.origin" in script
    assert 'setting.origin || "default"' in script
    # Only a changed key travels, and only the revision the page loaded. The
    # comparison is against what the page loaded, not against the string the
    # value serialises to, so a setting with no value is not always an edit.
    assert "if (settingChanged(setting, control)) values[key] = parsed;" in script
    assert "return settingChanged(setting, control);" in script
    assert "return String(value).trim();" in script
    assert "expected_revision: state.settingsRevision" in script
    # A refused write is shown once and not retried.
    assert "showSettingsResult(result)" in script
    assert "requires_ingest" in script
    assert "notice.textContent = result.requires_ingest" in script
    assert "/api/settings" in script


def test_the_config_panel_states_the_default_once_and_on_every_row() -> None:
    """The file the defaults come from is named once, and each row gives its own.

    The path is the same for every key, so a row carrying it would repeat one
    fact thirty-eight times and read as thirty-eight places to look.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert 'id="settings-defaults"' in page
    assert "payload.default_file" in script
    assert "Every default below is the one ${payload.default_file} declares." in script
    # The row states the value the app starts from, beside the value in force.
    assert "`Default: ${settingValueLabel(setting.default)}`" in script
    # An origin that is the default is not shown at all: the row already says
    # what the default is, and a layer that supplied nothing needs no badge.
    assert "if (setting.defaulted) return null;" in script
    assert ".setting-default {" in css


def test_a_costly_settings_change_asks_for_its_word_before_it_is_sent() -> None:
    """A regeneration or a model change is named, and the typed word is a gate."""

    with _panel_host() as client:
        script = client.get("/assets/app.js").text

    assert 'return level === "regeneration" || level === "model";' in script
    assert "cost.message" in script
    assert '? "model"' in script
    assert ': "ingest";' in script
    assert "Type ${word} to confirm" in script
    gate = script.split('byId("settings-confirm-word").addEventListener')[1]
    assert "pending.word" in gate


def test_the_workspace_carries_the_chunk_exclusion_controls() -> None:
    """Every hit and every context passage offers the exclusion, and the list restores."""

    with _panel_host() as client:
        page = client.get("/")
        script = client.get("/assets/app.js").text

    assert (
        'id="chunk-exclusion-summary" class="panel-section"'
        ' data-capability="chunk_exclusion"' in page.text
    )
    assert 'id="chunk-exclusion-list"' in page.text
    assert 'id="chunk-dialog"' in page.text
    assert 'id="chunk-error"' in page.text
    assert 'hasCapability("chunk_exclusion")' in script
    assert "actions.append(chunkAction(hit))" in script
    assert "/api/chunk-exclusions" in script
    assert "/api/chunk-inclusion" in script
    # A chunk the server already excluded is offered a restore instead.
    assert "state.chunkExclusions.has(chunk.chunk_id)" in script
    assert "This chunk is not in the current generation." in script
    # The whole-source exclusion is a separate control and stays as it was.
    assert 'id="exclusion-dialog"' in page.text
    assert 'hasCapability("source_inclusion")' in script
    assert "/api/source-inclusion" in script


def test_the_tab_bar_is_the_whole_navigation() -> None:
    """One nav item per job, and an item the host cannot serve is not in the sidebar.

    A nav item is a way into one panel, so every item carries the capability that
    decides whether its panel exists, and a sidebar holding no item is removed
    rather than left as a rule above nothing.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    items = re.findall(r'data-view="([^"]+)"\s+data-capability="([^"]+)"', page)
    assert items == [
        ("search", "documents"),
        ("sources", "sources"),
        ("status", "documents"),
        ("config", "settings"),
        ("mcp", "clients"),
        ("memory", "memory"),
        ("updates", "updates"),
    ]
    # Every item has a panel, and every panel is reached by exactly one item.
    panels = re.findall(r'data-panel="([^"]+)"\s+data-capability="([^"]+)"', page)
    assert panels == items
    assert 'byId("workspace-nav").hidden = !visibleNavItems.length' in script
    # A profile that leaves no item shows no panel rather than the panel of an
    # item that is gone.
    assert "switchView(activeItem ? activeItem.dataset.view : null);" in script
    # No id is declared twice: a second copy of a panel is a second place for it
    # to be wrong, and a duplicated id would silently pick the first one.
    identifiers = re.findall(r'\bid="([^"]+)"', page)
    assert len(identifiers) == len(set(identifiers))


def test_the_sidebar_navigates_without_a_tab_strip() -> None:
    """Navigation is a vertical column of items, and it says which view is current.

    A tab strip above the content is what made the page feel crammed, so the
    navigation is a sidebar: a list of items, each an ordinary button with an
    inline icon, and `aria-current` marking the one in view.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert '<nav id="workspace-nav" class="workspace-sidebar"' in page
    assert 'data-view="search" data-capability="documents"' in page
    assert 'class="sidebar-list"' in page
    # One item per view, and the item is a button rather than a tab role.
    assert page.count('<button class="nav-item') == 7
    assert 'role="tab"' not in page
    assert "aria-selected" not in page
    # The current view is marked as the current page, and only one is.
    assert page.count('aria-current="page"') == 1
    assert (
        'item.querySelector(".nav-item")?.setAttribute("aria-current", "page")'
        in script
    )
    assert (
        'else item.querySelector(".nav-item")?.removeAttribute("aria-current")'
        in script
    )
    # The item the markup marks active is the item the profile loop looks for, so
    # the first view is the one the column shows rather than no view at all.
    assert 'class="sidebar-item is-active" data-view="search"' in page
    assert (
        'const activeItem = visibleNavItems.find((item) => item.classList.contains("is-active"));'
        in script
    )
    # An icon is inline SVG rather than an icon font or a fetched file, so the
    # page adds no request and no dependency.
    assert page.count('<svg class="nav-item-icon"') == 7
    assert "http://" not in page
    # The column is keyboard operable with a visible focus ring, and the project
    # is repeated as a quiet footer beneath the views.
    assert ":focus-visible {" in client_css()
    assert "outline: 2px solid var(--accent);" in client_css()
    assert 'class="sidebar-footer"' in page
    assert 'id="sidebar-project-name" class="sidebar-footer-name"' in page
    assert 'byId("sidebar-project-name").textContent = projectName' in script


def test_the_sidebar_becomes_a_drawer_below_the_large_breakpoint() -> None:
    """Below 992px the column is a drawer, and it holds nothing against Escape.

    A drawer that trapped focus would be a second keyboard trap in a page that
    is otherwise operable, so Escape and the scrim both close it.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
    css = client_css()

    assert 'id="nav-toggle"' in page
    assert 'class="nav-toggle"' in page
    assert 'aria-expanded="false"' in page
    assert 'aria-controls="workspace-nav"' in page
    assert 'id="nav-scrim" class="nav-scrim" hidden' in page
    # The toggle is hidden while the column is fixed and shown while it is not.
    assert "@media (max-width: 991.98px) {" in css
    assert ".nav-toggle {\n  display: none;" in css
    assert "transform: translateX(-100%);" in css
    assert ".workspace-sidebar.is-open {\n    transform: none;" in css
    # Escape closes it and hands the focus back, and no listener holds focus.
    assert 'if (event.key === "Escape") closeNav({ restoreFocus: true });' in script
    assert 'addEventListener("keydown"' in script
    assert "trapFocus" not in script
    assert 'byId("nav-scrim").addEventListener("click", () => closeNav());' in script


def test_the_workspace_declares_a_spacing_and_type_scale() -> None:
    """Space is a scale rather than a number typed into each rule.

    The complaint was congestion, and ad-hoc small values are what produced it,
    so the two scales are declared once and named throughout.
    """

    css = client_css()

    assert ":root {" in css
    for token in (
        "--space-3xs",
        "--space-2xs",
        "--space-xs",
        "--space-sm",
        "--space-md",
        "--space-lg",
        "--space-xl",
        "--text-xs",
        "--text-sm",
        "--text-base",
        "--text-md",
        "--text-lg",
        "--text-xl",
        "--leading-normal",
        "--leading-prose",
        "--measure",
        "--layout-max",
    ):
        assert f"{token}:" in css
    # Body text reads at 1.5 and a long description reads higher than that.
    assert "--leading-normal: 1.5;" in css
    assert "--leading-prose: 1.7;" in css
    assert "body {" in css
    # A readable measure for prose and for the settings text.
    assert ".passage-text," in css
    assert ".setting-doc {" in css
    assert "max-width: var(--measure);" in css
    # Nothing clips: a settings row wraps, and a truncated ellipsis is gone.
    assert ".setting-row {" in css
    assert "text-overflow: ellipsis;" not in css
    assert "white-space: nowrap;" in css  # only on the connection state and the select
    assert ".nav-item {" in css
    assert "min-height: var(--control-height);" in css


def test_the_workspace_reads_in_light_and_dark() -> None:
    """Both schemes name the same tokens, and the accent is this package's own.

    UltraRAG's radii, shadows, fonts, and breakpoints are the vocabulary; its
    surfaces and its accent green and accent blue are not. The palette is cream
    and burnt orange, and borrowing either upstream colour as an accent would
    imply a product relationship this package does not have.
    """

    with _host(ClientControlAdapter()) as client:
        css = client.get("/assets/app.css").text
        page = client.get("/").text

    assert "@media (prefers-color-scheme: dark) {" in css
    assert '<meta name="color-scheme" content="light dark">' in page
    # Every token the light scheme declares is overridden for the dark one.
    light, _, dark = css.partition("@media (prefers-color-scheme: dark) {")
    dark_block = dark.split("\n}\n", 1)[0]
    for token in (
        "--bg-body",
        "--bg-surface",
        "--bg-sidebar",
        "--bg-input",
        "--text-primary",
        "--text-secondary",
        "--text-tertiary",
        "--border-subtle",
        "--accent",
    ):
        assert token in light
        assert f"{token}:" in dark_block
    # The palette is cream and burnt orange, not a white page under a violet
    # accent, and the dark scheme is the same hues at the other end.
    for value in (
        "--bg-body: #f0eee6",
        "--bg-surface: #e8e5db",
        "--bg-sidebar: #eae7dd",
        "--text-primary: #1c1b19",
        "--text-secondary: #5d5b55",
        "--border-subtle: #d9d5c8",
        "--accent: #b8592f",
    ):
        assert value in light
    for value in (
        "--bg-body: #1f1e1c",
        "--accent: #e39070",
    ):
        assert value in dark_block
    # The geometry is still UltraRAG's own.
    for value in (
        "--radius-sm: 6px",
        "--radius-md: 12px",
        "--radius-lg: 16px",
    ):
        assert value in light
    assert '"Inter"' in light
    assert '"JetBrains Mono"' in light
    # Neither upstream product colour is adopted as an accent. The file names
    # both only to record why they were not taken, so the check is on the
    # declaration rather than on the whole stylesheet.
    declarations = "\n".join(
        line for line in css.splitlines() if line.strip().startswith("--accent")
    )
    assert "--accent: #b8592f;" in declarations
    assert "#10a37f" not in declarations
    assert "#2563eb" not in declarations
    # The breakpoints are the ones UltraRAG declares.
    for breakpoint in ("991.98px", "767.98px", "575.98px"):
        assert breakpoint in css


def test_every_pair_that_carries_text_meets_wcag_aa() -> None:
    """Cream and orange are checked, not assumed.

    The light accent is darkened to #b8592f and the tertiary text to #676458 for
    exactly this: at the lighter orange and tertiary that read best on cream, white
    on the accent reaches 4.23:1 and the tertiary on the surface reaches 4.37:1.
    """

    with _host(ClientControlAdapter()) as client:
        css = client.get("/assets/app.css").text
    light, _, dark = css.partition("@media (prefers-color-scheme: dark) {")

    schemes = {
        "light": _tokens(light),
        "dark": _tokens(dark.split("\n}\n", 1)[0]),
    }
    for name, tokens in schemes.items():
        for pair, need in _TEXT_PAIRS:
            assert _contrast(tokens[pair[0]], tokens[pair[1]]) >= need, (
                f"{name}: {pair[0]} on {pair[1]}"
            )


def _tokens(block: str) -> dict[str, str]:
    """The hex tokens of one scheme, keyed by name without the leading dashes."""

    found: dict[str, str] = {}
    for line in block.splitlines():
        name, _, value = line.strip().partition(":")
        value = value.strip()
        if value.startswith("#"):
            found[name.removeprefix("--")] = value.split(";")[0].strip()
    return found


def _contrast(foreground: str, background: str) -> float:
    def channel(pair: str) -> float:
        raw = int(pair, 16) / 255.0
        return raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4

    def luminance(value: str) -> float:
        digits = value.lstrip("#")
        red, green, blue = (
            channel(digits[position : position + 2]) for position in (0, 2, 4)
        )
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lighter, darker = sorted(
        (luminance(foreground), luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


_TEXT_PAIRS = (
    (("text-primary", "bg-card"), 4.5),
    (("text-primary", "bg-surface"), 4.5),
    (("text-primary", "bg-body"), 4.5),
    (("text-primary", "bg-sidebar"), 4.5),
    (("text-primary", "bg-input"), 4.5),
    (("text-secondary", "bg-card"), 4.5),
    (("text-secondary", "bg-surface"), 4.5),
    (("text-secondary", "bg-sidebar"), 4.5),
    (("text-secondary", "bg-body"), 4.5),
    (("text-tertiary", "bg-card"), 4.5),
    (("text-tertiary", "bg-surface"), 4.5),
    (("text-tertiary", "bg-sidebar"), 4.5),
    (("accent-ink", "bg-card"), 4.5),
    (("accent-ink", "accent-soft"), 4.5),
    (("on-accent", "accent"), 4.5),
    (("danger", "bg-card"), 4.5),
    (("danger", "danger-soft"), 4.5),
    (("warning", "bg-card"), 4.5),
    (("warning", "warning-soft"), 4.5),
    (("ready", "bg-card"), 4.5),
    (("accent", "bg-card"), 3.0),
    (("accent-strong", "bg-card"), 3.0),
    (("accent", "bg-surface"), 3.0),
    (("accent", "bg-sidebar"), 3.0),
    (("accent", "bg-input"), 3.0),
)


def test_the_chrome_does_not_print_but_the_passages_do() -> None:
    """A passage is printed to be transcribed, so the column and header do not."""

    css = client_css()

    assert "@media print {" in css
    printed = css.split("@media print {", 1)[1]
    for selector in (".workspace-sidebar", ".site-header", ".nav-toggle", ".nav-scrim"):
        assert selector in printed
    assert "display: none !important;" in printed
    # The evidence itself survives, with its line breaks and its citation.
    assert ".passage-text," in printed
    assert "white-space: pre-wrap;" in printed


def test_a_settings_row_carries_its_description_value_origin_and_cost() -> None:
    """A row has room for everything the server said about one setting.

    The description, the range, and the list of choices arrive with the setting
    and are all optional: a host that sends none of them draws the row it always
    did, and a host that sends all of them gets all of them.
    """

    with _panel_host() as client:
        script = client.get("/assets/app.js").text
    css = client_css()

    assert 'node("p", "setting-doc", inlineText(setting.doc))' in script
    assert "settingValueLabel(setting.value)" in script
    assert 'setting.origin || "default"' in script
    assert "cost.message" in script
    # A range becomes the field's own bounds, and a list becomes a select over
    # exactly the values the server offered.
    assert "control.min = String(setting.minimum)" in script
    assert "control.max = String(setting.maximum)" in script
    assert 'node("select", "setting-input setting-select")' in script
    assert 'node("option", "", settingValueLabel(choice))' in script
    # The variable that would override the value is named beside it.
    assert "setting.env" in script
    assert "overrides this value" in script
    # Every field except a checkbox and a bool keeps the control its kind names,
    # and an unwritable one stays disabled.
    assert 'control.type = "checkbox"' in script
    assert 'control.type = "number"' in script
    assert 'control.type = "text"' in script
    assert "control.disabled = !setting.writable" in script
    # The row is a grid that wraps rather than one that truncates.
    assert ".setting-row {" in css
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 18rem);" in css
    assert ".setting-doc {" in css


def test_a_host_that_sends_no_setting_details_is_drawn_as_before() -> None:
    """An optional key that is absent leaves no blank row and no broken control."""

    with _panel_host() as client:
        script = client.get("/assets/app.js").text

    # Every one of them is guarded, so a host sending none still renders.
    assert "if (setting.doc) field.append" in script
    assert "if (cost.message) {" in script
    assert "if (setting.env) {" in script
    assert 'if (!choices.length || kind === "bool") return null;' in script
    # A missing range leaves no min and no max attribute.
    assert "if (setting.minimum !== undefined && setting.minimum !== null)" in script
    # A value the page did not load is still drawn, as the key with no value.
    assert "settingValueLabel(value)" in script
    assert 'return "none";' in script


def test_a_host_that_serves_no_panel_leaves_every_nav_item_out() -> None:
    """A capability off hides its nav item, and no panel is left standing behind it.

    The sidebar is the navigation, so an item that outlived its panel would be a
    way into a page this host cannot answer, and a profile with nothing on offers
    no panel rather than the one whose item happened to be active.
    """

    adapter = ClientControlAdapter()
    client = TestClient(
        create_ui_app(
            profile=UIProfile(
                application_name="Bare host",
                capabilities=UICapabilities(
                    documents=False,
                    sources=False,
                    settings=False,
                    clients=False,
                    memory=False,
                ),
            ),
            adapter=adapter,
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )
    with client:
        capabilities = client.get("/api/ui").json()["capabilities"]
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    for capability in ("documents", "sources", "settings", "clients", "memory"):
        assert capabilities[capability] is False
    # Every panel and the status summary are gated on the same capabilities the
    # nav items are, so nothing this host cannot answer has a way in.
    for panel, capability in (
        ("search", "documents"),
        ("sources", "sources"),
        ("status", "documents"),
        ("config", "settings"),
        ("mcp", "clients"),
        ("memory", "memory"),
        ("updates", "updates"),
    ):
        assert re.search(
            rf'data-panel="{panel}"\s+data-capability="{capability}"', page
        )
    assert 'class="view-panel system-summary"' in page
    # The loop that hides by capability is the only thing that decides an item,
    # and what it leaves is handled rather than ignored.
    hide = script.split("function applyProfile(")[1].split("\n}\n")[0]
    assert "element.hidden = !hasCapability(element.dataset.capability);" in hide
    assert "switchView(activeItem ? activeItem.dataset.view : null);" in hide
    assert 'byId("workspace-nav").hidden = !visibleNavItems.length;' in hide
    # A column with nothing in it is removed, and the button that opens it goes
    # with the column rather than opening an empty drawer.
    assert 'byId("nav-toggle").hidden = !visibleNavItems.length;' in hide


def test_the_header_carries_a_projects_selector_the_host_supplies() -> None:
    """Each project is named, its state is shown, and switching is never silent.

    One installation serves several projects, so a workspace has to say which one
    it is. Selecting another is offered that project's own workspace, and a
    project with no app is offered the command that starts it, because it has no
    address at all.
    """

    with _project_host() as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert 'id="project-selector"' in page.text
    assert 'data-capability="projects"' in page.text
    assert 'id="project-select"' in page.text
    assert 'id="project-open"' in page.text
    assert 'id="project-start-command"' in page.text
    assert 'id="project-copy-command"' in page.text
    # The selector is the first thing in the sidebar and above the views, because
    # changing the workspace is what a reader reaches for before choosing a view.
    sidebar = page.text.split('class="workspace-sidebar"', 1)[1]
    assert sidebar.index('id="project-selector"') < sidebar.index(
        'class="sidebar-list"'
    )
    assert page.text.index("</header>") < page.text.index("project-selector")
    assert 'hasCapability("projects")' in script.text
    assert "/api/projects" in script.text
    # The current project is marked in the list, and every entry says whether its
    # app is up rather than leaving the reader to try a URL.
    assert "project.project_name === state.currentProject" in script.text
    assert 'project.running ? "app up" : "app not running"' in script.text
    # The control says that another project is a different page, and never
    # repoints this one at it.
    assert "Opening another project opens its own workspace in a new tab." in page.text
    assert 'window.open(project.url, "_blank", "noopener")' in script.text
    # The start command is the host's own, taken from the profile, because a
    # command composed here is one the reader could paste and fail on.
    assert "state.profile?.project_start_command" in script.text
    assert 'template.replace("{project}", project.project_name)' in script.text
    assert 'copyText(byId("project-start-command").textContent' in script.text


def test_the_mcp_tab_carries_a_copyable_client_entry() -> None:
    """The entry is the host's own text, offered for copying and nothing more."""

    with _project_host() as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert 'id="agent-entry-summary" class="panel-section"' in page.text
    assert 'data-capability="agent_entry"' in page.text
    assert 'id="agent-entry" class="standing-document"' in page.text
    assert 'id="agent-entry-copy"' in page.text
    assert 'hasCapability("agent_entry")' in script.text
    assert "/api/agent-entry" in script.text
    # The text reaches the block and the clipboard unedited, and no generator is
    # built here: a second one would be a second place for a client's
    # configuration to be wrong.
    assert (
        'byId("agent-entry").textContent = readableEntry(state.agentEntry)'
        in script.text
    )
    assert 'copyText(state.agentEntry, "Client entry copied.")' in script.text
    # It is the stdio entry, so the block says which client it is for.
    assert "for a client that cannot open a socket" in page.text.lower()


def test_a_panel_holds_only_the_blocks_that_belong_to_it() -> None:
    """Each job is reached from one place, and no panel is a second copy of another.

    The complaint was a page that put every management block under every tab, so
    a reader who opened Config also got the generation list and the clients. Each
    block now sits in the panel its work belongs to, and the test reads them back
    out of the markup rather than trusting the navigation order.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    panels = {
        name.removesuffix("-view"): body
        for name, body in re.findall(
            r'<section id="([a-z-]+-view)"(?:\s[^>]*)?>(.*?)</section>\s*(?=<section|<dialog|</main)',
            page,
            flags=re.DOTALL,
        )
    }
    assert set(panels) == {
        "search",
        "sources",
        "status",
        "config",
        "mcp",
        "memory",
        "updates",
    }
    # The filters and the lists that fill them are one job, and both are in the
    # search view: a box whose values are listed three panels away is a box that
    # asks the reader to know what it already knows.
    for element in (
        'id="filter-section"',
        'id="filter-fields"',
        'id="partition-chips"',
        'id="project-chips"',
        'id="language-chips"',
    ):
        assert element in panels["search"]
    # The build, its generations, the passage decisions, and the record console
    # are all facts about what is stored, so they are one status view.
    for element in (
        'id="status-cards"',
        'id="status-message"',
        'id="status-facts"',
        'id="generation-summary"',
        'id="chunk-exclusion-summary"',
        'id="sql-console"',
    ):
        assert element in panels["status"]
    # Settings stay the whole Config view, and the clients the whole MCP view.
    assert 'id="settings-panel"' in panels["config"]
    for element in (
        'id="client-summary"',
        'id="agent-endpoint"',
        'id="agent-entry-summary"',
    ):
        assert element in panels["mcp"]
    # The source inventory is the only thing in the sources view besides the
    # sources it has decided to leave out.
    assert 'id="source-list"' in panels["sources"]
    assert 'id="excluded-section"' in panels["sources"]
    # Nothing is repeated: one id, one panel.
    for panel in panels.values():
        for element in (
            'id="settings-form"',
            'id="generation-chips"',
            'id="client-chips"',
        ):
            assert element not in panel or element in panel
    # The status reader still asks for what it draws.
    assert "renderGenerations(status.generations || [])" in script


def test_a_filter_group_says_whether_every_value_or_only_one_must_match() -> None:
    """All-semantics and any-semantics decide the result and are not visible from
    the field name, so each group states which one it applies in one sentence."""

    with _panel_host() as client:
        page = client.get("/").text

    assert "Every value must match" in page
    assert "At least one value must match" in page
    # The two meanings sit on the two kinds of field, and the sentence is inside
    # the group whose fields it describes rather than above all of them.
    all_of_them = page.split('data-capability="metadata_filters"', 1)[1]
    assert "Every value must match" in all_of_them.split("</fieldset>", 1)[0]
    any_of_them = page.split('data-capability="category_partitions"', 1)[1]
    assert "At least one value must match" in any_of_them.split("</fieldset>", 1)[0]
    # A list the reader picks from is named as such, and an empty field is stated
    # to match everything rather than to match nothing.
    assert "Select partitions from the list below." in page
    assert "A field left empty" in page and "matches" in page
    assert "Empty searches every source." in page
    assert "Empty excludes nothing." in page


def test_a_value_taken_from_a_list_opens_the_fields_and_is_counted() -> None:
    """A value chosen from a list is invisible until the fields are open.

    Each list counts what it holds, choosing one opens the drawer the value went
    into, and the count on the section heading and the sentence on the summary are
    computed once so they cannot disagree.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    for element in (
        'id="filter-count"',
        'id="filter-summary-note"',
        'id="filter-fields"',
    ):
        assert element in page
    assert "function syncFilterSummary()" in script
    assert 'byId("filter-count").textContent' in script
    assert 'byId("filter-summary-note").textContent' in script
    assert "revealFilterField(field);" in script
    assert "drawer.open = true;" in script
    # The language list fills the language box beside it, so selecting from it is
    # a filter action like every other one.
    assert 'byId("language-chips").addEventListener("click", handleAction);' in script
    # Typing is counted too, not only choosing from a list.
    assert "FILTER_FIELDS.forEach((field) => {" in script
    assert 'byId(field).addEventListener("input", syncFilterSummary);' in script


def test_the_page_says_when_reviewed_decisions_take_effect() -> None:
    """Three moments a reader gets wrong, each stated where it is acted on.

    Reviewed metadata is applied at read time and a passage exclusion is enforced
    by every search, so neither waits for a rebuild. Saying otherwise sends a
    reader to run an ingestion that changes nothing.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert "applies to the next search" in page
    assert "no rebuild is needed" in page
    assert "Metadata saved. It applies to the next search." in script
    assert "Search changes as soon as this is saved." in page
    assert "restoring one brings it back in every generation that holds it" in page
    # The old wording told a reader to wait for an ingestion that no longer
    # restores anything, and it must not come back in another block.
    assert "Metadata changes become searchable after a new ingestion." not in page
    assert "Retrieval changes for the next generation." not in page
    assert "returns it to the next generation" not in page


def test_a_build_that_can_be_continued_says_so_and_one_that_cannot_says_nothing() -> (
    None
):
    """Whether a build resumes is the host's own fact about its own pipeline.

    A shared page cannot claim it on behalf of a host that does not checkpoint,
    so the sentence arrives with the profile and the block is empty until it does.
    """

    with _host(ClientControlAdapter()) as client:
        ui = client.get("/api/ui").json()
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    # The slot exists on every profile; what a shared page must not do is claim
    # checkpointing for a host that says nothing about it.
    assert not ui.get("ingest_resume_note")
    assert 'id="ingest-note" class="form-note dialog-note" hidden' in page
    assert 'ingestNote.textContent = profile.ingest_resume_note || "";' in script
    assert "ingestNote.hidden = !profile.ingest_resume_note;" in script
    # What the page can state on its own is that the generation in use stays.
    assert "The current generation" in page
    assert "remains active unless the complete build succeeds" in page


def test_identifiers_are_shown_whole_behind_one_labelled_disclosure() -> None:
    """An identifier is shortened for a badge and never for a copy.

    A generation id, a passage id, and a per-component score are what a reader
    checks when a result looks wrong, so each is shown in full, labelled, and
    selectable inside one disclosure rather than compressed under the passage.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert (
        'byId("generation-label").textContent = status.generation_id || "None yet";'
        in script
    )
    assert 'rows.push(["Passage identifier", hit.chunk_id]);' in script
    assert 'scorePair("Fusion score", hit.fusion_score)' in script
    assert "Scores and identifiers" in script
    assert "Path and identifier" in script
    assert 'factList(rows, "fact-list score-facts")' in script
    # The compaction helper survives only where a value is a caption rather than
    # an identifier a reader copies.
    assert "function compactId(value)" in script
    assert "compactId(hit.chunk_id)" not in script
    # One disclosure style serves both, so the two look like the same control.
    assert ".detail-drawer," in css and ".identifier-drawer {" in css
    assert ".fact-list {" in css
    assert 'id="status-facts" class="fact-list"' in page
    assert 'id="status-details" class="detail-drawer"' in page


def test_the_status_view_keeps_the_servers_own_sentence_when_nothing_is_wrong() -> None:
    """The server's message carries the sentences the cards cannot.

    Among them: that reviewed metadata is being applied at read time, and that
    some passage exclusions were recorded against another generation. Dropping it
    whenever nothing is urgent hides exactly those.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert 'id="status-message" class="status-message" hidden' in page
    assert 'message.textContent = status.message || "";' in script
    assert "message.hidden = !status.message;" in script
    # It sits above the detail drawer, because it is read before either.
    status_view = page.split('id="status-cards"', 1)[1]
    assert status_view.index('id="status-message"') < status_view.index(
        'id="status-details"'
    )
    # A warning is never folded away: the notice and the sentence are separate.
    assert 'id="status-notice" class="notice" hidden' in page
    assert ".status-message {" in client_css()


def test_the_page_carries_no_private_project_or_source_names() -> None:
    """The placeholders in a shared page are written for any installation.

    A category or project name typed into a placeholder ships one reader's corpus
    to every other reader of the same package.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text

    for name in ("ai-and-fetishism", "fetishism", "Crawford", "Gidwani", "Atlas of AI"):
        assert name not in page


def test_a_badge_takes_a_colour_only_when_the_code_asks_it_to() -> None:
    """The generation in use is green, a blocked one is red, and nothing else is.

    A rule that painted every badge with the danger colour made a neutral count
    read as a fault, which is worse than no colour at all.
    """

    with _panel_host() as client:
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert 'node("span", "state-badge state-badge-current", "In use")' in script
    assert 'node("span", "state-badge", "Retained")' in script
    assert ".state-badge-current {" in css
    assert "border-color: var(--ready);" in css
    assert ".state-badge-blocked {" in css
    assert "color: var(--danger);" in css
    # Only a state the code marks is coloured, and the danger colour is reached
    # through the blocked class rather than through the base badge.
    assert '"state-badge state-badge-blocked"' in script
    assert (
        re.search(r"^\.state-badge \{[^}]*var\(--danger", css, flags=re.MULTILINE)
        is None
    )


def test_the_client_entry_is_indented_for_reading_and_copied_as_it_was_sent() -> None:
    """Readable on screen, byte for byte on the clipboard.

    The two are different promises, so the display path re-indents and the copy
    path does not touch the entry at all.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert "function readableEntry(entry)" in script
    assert "JSON.stringify(JSON.parse(entry), null, 2)" in script
    # An entry that is not JSON is shown as the server sent it rather than refused.
    assert "return entry;" in script
    assert (
        'byId("agent-entry").textContent = readableEntry(state.agentEntry);' in script
    )
    assert 'copyText(state.agentEntry, "Client entry copied.")' in script
    # The sentence beside the button describes the two promises, not one.
    assert "shown exactly as it produced it" not in page
    assert "The copy button sends exactly what the server produced." in page


def test_a_byline_and_a_locator_report_only_what_the_payload_carries() -> None:
    """No `undefined`, and no format the payload never claimed.

    A source with a publication year and no reviewed year shows that year, and a
    locator carrying a page but no declared kind reads as a page rather than
    wearing a format label nothing in the payload supports.
    """

    with _panel_host() as client:
        script = client.get("/assets/app.js").text

    assert "const year = source.year ?? source.publication_year;" in script
    assert "parts.push(String(year));" in script
    # Every value that reaches a label passes a presence test first.
    assert "if (year !== null && year !== undefined && String(year).trim()) {" in script
    assert 'String(index).trim() !== ""' in script
    assert "EPUB section" not in script
    # A locator arrives either as an object the server shaped or as the string it
    # already formatted, and a string is passed through rather than mined.
    assert (
        'if (typeof locator === "string") return locator.trim() || "Source passage";'
        in script
    )
    assert (
        "locator.page_label ?? locator.page ?? pageData.page_label ?? pageData.page"
        in script
    )
    assert (
        'locator.type === "pdf_page" || (locator.type === undefined && hasPage)'
        in script
    )
    assert 'return "Passage location not reported";' in script


def test_a_neighbour_the_corpus_left_out_says_so_on_the_passage() -> None:
    """The context dialog marks a passage no query returns.

    The server refuses to serve an excluded passage as the one asked for, so the
    neighbours it marks are the only way one reaches this dialog, and each is
    labelled rather than presented as a hit.
    """

    with _panel_host() as client:
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert "const excluded = passage.excluded_from_search === true;" in script
    assert 'excluded ? " context-excluded" : ""' in script
    assert '"context-excluded-warning"' in script
    assert "Excluded from search." in script
    assert ".context-excluded {" in css
    assert ".context-passage .context-excluded-warning {" in css
    # Changing what a search reads stays an explicit step taken in the dialog, and
    # a passage already excluded is not offered a toggle beside it.
    assert 'hasCapability("chunk_exclusion") && !excluded' in script
    assert "item.append(chunkExcludeButton(passage));" in script
    # A locator longer than its card wraps rather than being clipped, because a
    # clipped locator is one the reader cannot use.
    assert "flex: 0 1 auto;" in css
    assert "max-width: 100%;" in css
    assert "overflow-wrap: anywhere;" in css


def test_a_chosen_view_takes_the_focus_and_the_narrow_window_keeps_it_reachable() -> (
    None
):
    """The focus follows the choice, and every control keeps a pointer's size.

    A view panel takes the focus when a reader chooses it from the column, so a
    keyboard reader lands on the view rather than on the sidebar they just left,
    and the first render does not take the focus from wherever the page opened.
    """

    with _host(ClientControlAdapter()) as client:
        script = client.get("/assets/app.js").text
    css = client_css()

    assert "switchView(item.dataset.view, { moveFocus: true });" in script
    assert "panel.tabIndex = -1;" in script
    assert "opened.focus({ preventScroll: true });" in script
    # Every breakpoint UltraRAG declares is still declared, and each of them
    # changes the workspace padding so a narrow window is not a wide layout
    # squeezed.
    for breakpoint in ("991.98px", "767.98px", "575.98px"):
        assert f"@media (max-width: {breakpoint}) {{" in css
    assert ".workspace {\n    padding: var(--space-sm);" in css
    # A labelled pair becomes two lines rather than two squeezed columns.
    assert ".fact-list {\n    grid-template-columns: minmax(0, 1fr);" in css
    # A control a reader taps is at least the size of a finger on every list.
    assert ".pick-button {" in css
    assert ".client-drawer summary {" in css
    assert css.count("min-height: var(--control-height);") >= 4
    # The new components are declared once, in the place they belong, rather than
    # appended as an override block at the end of the file.
    for selector in (
        ".panel-section {",
        ".record-list,",
        ".client-list {",
        ".fact-list {",
        ".filter-group {",
        ".status-message {",
    ):
        assert selector in css


def test_the_updates_view_is_gated_on_a_capability_the_page_never_invents() -> None:
    """The view ships behind `updates` and adds no way in for a host that lacks it.

    A page that asked about releases on its own would be a capability the host
    never granted, so the item, the panel, and the one request that follows the
    capability all share the same name.
    """

    with _host(ClientControlAdapter()) as client:
        ui = client.get("/api/ui").json()
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert ui["capabilities"].get("updates") is not True
    assert 'data-view="updates" data-capability="updates"' in page
    assert 'id="updates-view" class="view-panel" data-panel="updates"' in page
    assert 'if (!hasCapability("updates")) return;' in script
    assert 'if (hasCapability("updates")) {' in script
    # The check runs after the workspace has drawn and is never awaited, so a
    # slow or unreachable release feed cannot hold up a search.
    assert "void checkUpdates();" in script


def test_the_updates_view_shows_what_the_server_reported_as_text() -> None:
    """Installed version, tag, date, and the full notes, none of it rendered markup.

    Release notes arrive as Markdown. They are written into the page as text
    content, so nothing here can turn a heading, a link, or a tag into markup a
    reader would act on.
    """

    client, _adapter = _update_host()
    with client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    for element in (
        "update-check-button",
        "update-message",
        "update-detail",
        "update-release-tag",
        "update-facts",
        "update-notes",
        "update-declined",
    ):
        assert f'id="{element}"' in page
    # The notes block is written with textContent and never with markup.
    assert 'byId("update-notes").textContent = release?.body' in script
    assert 'byId("update-dialog-notes").textContent = release.body' in script
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert ".release-notes {" in client_css()
    assert "white-space: pre-wrap;" in client_css()
    # The URL is only rendered when it is the forge the project publishes from.
    assert 'const RELEASE_HOSTS = ["github.com"];' in script
    assert 'if (parsed.protocol !== "https:") return "";' in script
    assert 'if (!RELEASE_HOSTS.includes(parsed.hostname)) return "";' in script


def test_reading_and_declining_an_update_writes_nothing() -> None:
    """The page asks, shows, and stops.

    Checking is a read. Declining is a dismissal. Copying is a clipboard write.
    None of them posts anything, applies anything, or starts a process, because
    an update replaces the very app serving this page.
    """

    client, adapter = _update_host()
    with client:
        response = client.get("/api/updates")
        script = client.get("/assets/app.js").text

    assert response.status_code == 200
    payload = response.json()
    assert payload["installed_version"] == "1.1.0"
    assert payload["update_available"] is True
    assert payload["apply_command"] == "research-rag update --apply"
    assert adapter.calls == ["check_updates"]
    # The only request this page can make about an update is that read, and it is
    # made once per check: there is no polling and nothing that would apply it.
    assert script.count('await api("/api/updates")') == 1
    updates_source = script.split(
        "// --------------------------------------------------------------- updates --"
    )[1]
    updates_source = updates_source.split("function renderSqlConsole")[0]
    for forbidden in (
        '"/api/updates", {',
        "apply_update",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "Popen",
        "subprocess",
        "exec(",
        "spawn(",
        'createElement("a")',
    ):
        assert forbidden not in updates_source
    # Declining records the version in the page and closes the dialog. Nothing
    # leaves the browser.
    assert "function declineUpdate() {" in script
    assert "state.updates.declinedVersion = version;" in script
    assert 'byId("update-dialog").close();' in script
    # A decline holds only against that version, and only until a reader asks
    # for a check themselves.
    assert "return version !== state.updates.declinedVersion;" in script
    assert 'state.updates.declinedVersion = "";' in script
    assert adapter.writes == []


def test_a_declined_update_is_not_asked_about_again_and_a_new_one_is() -> None:
    """The page remembers the version it was told no to, and forgets on a recheck."""

    client, _adapter = _update_host()
    with client:
        script = client.get("/assets/app.js").text

    assert "function shouldPromptUpdate(payload) {" in script
    assert 'if (updateOutcome(payload) !== "available") return false;' in script
    assert "if (!version) return false;" in script
    assert "You chose not to update to ${release.version} for now." in script
    assert "Ask again by checking for updates." in script


def test_continuing_in_a_terminal_shows_the_command_and_says_nothing_is_installed() -> (
    None
):
    """The handoff is a command to read, not an installation to report as done.

    The dialog offers exactly two answers and names them plainly. The command
    appears only after the reader chooses to continue, and the sentence beside it
    says the command asks for approval again.
    """

    client, _adapter = _update_host()
    with client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert '<dialog id="update-dialog" class="dialog dialog-wide">' in page
    assert 'id="update-choice" class="update-actions"' in page
    assert 'id="update-handoff" class="update-handoff" hidden' in page
    assert 'id="update-not-now"' in page
    assert ">Not now</button>" in page
    assert ">Continue in terminal</button>" in page
    assert 'id="update-command" class="update-command"' in page
    assert 'id="update-command-copy"' in page
    # Nothing has happened yet, and the command asks again.
    assert "Nothing has been installed." in page
    assert "asks for your approval again" in page
    # A global installation stops the projects it serves, so the handoff says so
    # rather than leaving the reader to find out from a failed install.
    assert "projects it is serving" in page
    # The button reveals the command and takes the choice away.
    assert 'byId("update-handoff").hidden = false;' in script
    assert 'byId("update-choice").hidden = true;' in script


def test_a_release_that_could_not_be_checked_is_never_reported_as_up_to_date() -> None:
    """Three answers, and the failed one says so in its own words.

    A reader told "up to date" after a failed check stops looking for an update
    that is there, so the offline and unavailable cases carry their own sentence
    and the server's reason is shown beside it.
    """

    offline = {
        "installed_version": "1.1.0",
        "update_available": False,
        "offline": True,
        "release": None,
        "message": "could not check",
    }
    unavailable = {**offline, "offline": False}
    client, _adapter = _update_host()
    with client:
        script = client.get("/assets/app.js").text

    assert 'if (payload?.update_available) return "available";' in script
    assert 'if (payload?.offline) return "offline";' in script
    assert (
        'return payload?.release === null && payload?.message ? "unavailable" : "none";'
        in script
    )
    assert (
        "could not reach the release feed, so whether a newer release exists is unknown"
        in script
    )
    assert "The release feed could not be checked." in script
    assert "checked and reported no newer release." in script
    # Nothing anywhere claims to be up to date.
    assert "up to date" not in script
    assert "up to date" not in client.get("/").text
    # The payload for both negative answers is the contract the server sends.
    assert offline["release"] is None
    assert offline["update_available"] is False
    assert unavailable["message"] == "could not check"


def test_an_available_release_is_flagged_in_the_column_rather_than_announced() -> None:
    """One word in the sidebar, not a banner across the page.

    The badge says there is something to read and nothing about what it is; the
    view says what it is.
    """

    client, _adapter = _update_host()
    with client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert 'id="update-nav-badge" class="nav-item-badge" hidden>new</span>' in page
    assert 'badge.hidden = outcome !== "available";' in script
    assert ".nav-item-badge {" in client_css()
    # No banner, and nothing that polls for a release and could reopen a dialog a
    # reader already dismissed.
    assert "setInterval" not in script
    assert script.count('await api("/api/updates")') == 1


def test_declining_and_copying_an_update_send_no_request_at_all(tmp_path: Path) -> None:
    """The two answers a reader can give reach the network zero times.

    The script is loaded into a stub document and driven, so the claim is about
    what the page does rather than about what it says it does: declining closes
    the dialog and records the version, copying fills the clipboard, and neither
    reaches `fetch` at all.
    """

    harness = tmp_path / "harness.mjs"
    harness.write_text(
        """
import { readFileSync } from "node:fs";
import vm from "node:vm";

const script = readFileSync(process.argv[2], "utf8");

function element() {
  const node = {
    textContent: "",
    hidden: false,
    title: "",
    value: "",
    className: "",
    dataset: {},
    style: {},
    children: [],
    attributes: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = kids; },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    removeAttribute() {},
    focus() {},
    showModal() {},
    close() { this.closed = true; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
  };
  return node;
}

const registry = new Map();
const document = {
  getElementById(id) {
    if (!registry.has(id)) registry.set(id, element());
    return registry.get(id);
  },
  createElement: element,
  createTextNode: (text) => ({ textContent: text }),
  querySelectorAll() { return []; },
  querySelector() { return null; },
  addEventListener() {},
  title: "",
};

const requests = [];
const context = {
  document,
  navigator: { clipboard: { writeText: async (text) => { context.__clipboard = text; } } },
  fetch: async (path, options = {}) => {
    requests.push({ path, method: options.method || "GET" });
    return { ok: true, status: 200, json: async () => ({}) };
  },
  Intl,
  URL,
  FormData: class {},
  setTimeout: () => 0,
  setInterval: () => 0,
  window: {
    setTimeout: () => 0,
    setInterval: () => 0,
    matchMedia: () => ({ addEventListener() {} }),
  },
};
context.globalThis = context;
vm.createContext(context);
// `const` at module scope stays inside it, so the handles this harness drives
// are published explicitly rather than read off the context.
const handle = ["state", "renderUpdates", "declineUpdate", "fillUpdateDialog", "copyText"]
  .map(function (name) { return ";globalThis." + name + " = " + name + ";"; })
  .join("");
vm.runInContext(script + handle, context);

const payload = {
  installed_version: "1.1.0",
  update_available: true,
  offline: false,
  release: {
    version: "1.2.0",
    tag_name: "v1.2.0",
    name: "research-rag 1.2.0",
    published_at: "2026-10-03T00:00:00Z",
    html_url: "https://github.com/AhmedKishki/research-rag/releases/tag/v1.2.0",
    body: "## Fixed\\n\\n- One passage",
  },
  apply_command: "research-rag update --apply",
  message: null,
};

// One read stands in for the check the page made before the dialog opened.
requests.push({ path: "/api/updates", method: "GET" });
context.state.profile = { capabilities: { updates: true } };
context.state.updates.payload = payload;
vm.runInContext("renderUpdates(state.updates.payload)", context);
const afterRender = requests.length;
context.declineUpdate();
const afterDecline = requests.length;
vm.runInContext("fillUpdateDialog(state.updates.payload)", context);
context.document.getElementById("update-command-copy").textContent = "";
const command = context.document.getElementById("update-command").textContent;
await vm.runInContext(
  `(async () => {
     document.getElementById("update-command-copied").hidden = false;
     await copyText(document.getElementById("update-command").textContent, "copied");
   })()`,
  context,
);
process.stdout.write(JSON.stringify({
  afterRender,
  afterDecline,
  total: requests.length,
  command,
  clipboard: context.__clipboard || "",
  declined: context.state.updates.declinedVersion,
  dialogClosed: Boolean(context.document.getElementById("update-dialog").closed),
  notesWrittenAsText: context.document.getElementById("update-notes").textContent,
  badgeHidden: context.document.getElementById("update-nav-badge").hidden,
  releaseUrlRow: context.document.getElementById("update-facts").children
    .flatMap((node) => (node.children.length ? node.children : [node]))
    .map((node) => node.textContent).filter((text) => text.startsWith("https")),
}));
""",
        encoding="utf-8",
    )
    asset = (
        Path(__file__).parents[2] / "src/research_rag/surfaces/workspace/static/app.js"
    )
    if shutil.which("node") is None:
        pytest.skip("node is not installed, so the page cannot be driven here.")
    completed = subprocess.run(
        ["node", str(harness), str(asset)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"the page harness failed: {completed.stderr.strip()[:400]}")
    result = json.loads(completed.stdout)

    # Rendering the answer is a read the page already made, and nothing more.
    assert result["afterRender"] == 1
    # Declining reaches the network zero times, closes the dialog, and remembers
    # the version it was told no to.
    assert result["afterDecline"] == 1
    assert result["dialogClosed"] is True
    assert result["declined"] == "1.2.0"
    # Copying is a clipboard write and nothing else.
    assert result["total"] == 1
    assert result["command"] == "research-rag update --apply"
    assert result["clipboard"] == "research-rag update --apply"
    # The notes were written as text and the release page is the forge's own.
    assert "## Fixed" in result["notesWrittenAsText"]
    assert result["badgeHidden"] is False
    assert result["releaseUrlRow"] == [
        "https://github.com/AhmedKishki/research-rag/releases/tag/v1.2.0"
    ]
