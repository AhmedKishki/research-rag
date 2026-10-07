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

import hashlib
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
    assert "syncGatedSubmits();" in busy
    gate_sync = script.split("function syncGatedSubmits(")[1].split("\n}\n")[0]
    assert "syncSqlControls();" in gate_sync


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
    gate = script.split("function syncSettingsConfirmSubmit(")[1].split("\n}\n")[0]
    assert 'byId("settings-confirm-submit").disabled' in gate
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
    assert "actions.push(chunkAction(hit))" in script
    assert "/api/chunk-exclusions" in script
    assert "/api/chunk-inclusion" in script
    # A chunk the server already excluded is offered a restore instead.
    assert "state.chunkExclusions.has(chunk.chunk_id)" in script
    assert "Not in the current generation" in script
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
        ("stats", "stats"),
        ("config", "settings"),
        ("mcp", "clients"),
        ("memory", "memory"),
        ("updates", "updates"),
    ]
    # Every item has a panel, and every panel but the detail view of one source is
    # reached by exactly one item. The source view is reached from a name, and its
    # sidebar item stays marked.
    panels = re.findall(r'data-panel="([^"]+)"\s+data-capability="([^"]+)"', page)
    assert [panel for panel in panels if panel[0] != "source"] == items
    assert ("source", "sources") in panels
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
    assert page.count('<button class="nav-item') == 8
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
    # A view the fragment names comes first, and the marked item is the fallback.
    assert "visibleNavItems.find((item) => item.dataset.view === routed)" in script
    assert (
        '|| visibleNavItems.find((item) => item.classList.contains("is-active"));'
        in script
    )
    # An icon is inline SVG rather than an icon font or a fetched file, so the
    # page adds no request and no dependency.
    assert page.count('<svg class="nav-item-icon"') == 8
    assert "http://" not in page
    # The column is keyboard operable with a visible focus ring.
    assert ":focus-visible {" in client_css()
    assert "outline: 2px solid var(--accent);" in client_css()


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
    assert 'if (event.key !== "Escape") return;' in script
    assert "closeNav({ restoreFocus: true });" in script
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


def test_search_result_text_is_justified() -> None:
    """A result's passage is set as a justified block, and nothing else is.

    The context dialog and the memory rounds keep their own alignment, so the
    change is scoped to the search result rather than to every passage.
    """

    css = client_css()

    rule = css.split(".passage-text {", 1)[1].split("}", 1)[0]
    assert "text-align: justify;" in rule


def test_the_content_is_centred_at_a_readable_width() -> None:
    """The content block is capped and centred rather than pinned to the left.

    A wide window used to leave the column flush against the sidebar with all
    the spare room to its right, so the block is capped at the page width and
    centred in the space that remains.
    """

    css = client_css()

    workspace = css.split(".workspace {", 1)[1].split("}", 1)[0]
    assert "width: min(var(--layout-max), 100%);" in workspace
    assert "grid-column: 2;" in workspace
    assert "margin-inline: auto;" in workspace


def test_the_app_and_the_project_are_named_once_in_the_header() -> None:
    """The header names the app and the project, and nothing repeats them.

    The project used to be named again in a sidebar footer, which gave a reader
    two places to read one name.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert 'id="application-name" class="brand-name"' in page
    assert page.count('id="application-name"') == 1
    assert 'id="project-name" class="project-name"' in page
    assert page.count('id="project-name"') == 1
    assert "sidebar-project-name" not in page
    assert "sidebar-project-name" not in script
    assert "sidebar-footer" not in page
    assert "sidebar-footer" not in client_css()


def test_the_cleaned_passage_notice_is_shown_once_per_results_view() -> None:
    """The label belongs to the results view, not to every passage in it.

    Repeating it on each card added a line to every result to say one thing
    about all of them.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    assert 'id="results-notice" class="semantic-text-label" hidden' in page
    assert page.count('class="semantic-text-label"') == 1
    # The label is written once, and no card carries its own copy.
    assert 'node("div", "semantic-text-label"' not in script
    assert 'const notice = byId("results-notice");' in script
    assert "notice.textContent" in script
    assert "result_text_label" in script
    assert "notice.hidden = !count;" in script
    # A cleared view takes the notice with the summary and the results.
    assert 'byId("results-notice").hidden = true;' in script


def test_the_page_and_its_assets_are_revalidated_against_a_stale_cache() -> None:
    """The page and its assets share one revision, so neither may be reused stale.

    An updated page served against a cached script throws before it draws, so
    the page and every asset are revalidated on each load. API reads are not
    stored at all, so a project's records are never read from a cache.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")
        api = client.get("/api/ui")

    assert page.headers["cache-control"] == "no-cache"
    assert script.headers["cache-control"] == "no-cache"
    assert api.headers["cache-control"] == "no-store"


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
        ("stats", "stats"),
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
        "source",
        "status",
        "stats",
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
        'id="health-section"',
        'id="generation-summary"',
        'id="rebuild-button"',
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
    assert "an empty field matches everything" in page
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
    assert "revealFilterDrawer();" in script
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
    assert 'if (hasCapability("updates") && !state.updates.checked) {' in script
    # The check runs once a page, after the workspace has drawn, and is never
    # awaited, so a slow or unreachable release feed cannot hold up a search.
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
    setAttribute(name, value) { this.attributes[name] = value; },
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


# A stub document the page script runs in. Each element answers a selector with a
# child of its own and a `closest` with a parent of its own, so a function under
# test can reach the nodes it writes to and the test can read them back.
_PAGE_HARNESS = r"""
import { readFileSync } from "node:fs";
import vm from "node:vm";

const script = readFileSync(process.argv[2], "utf8");
const scenario = readFileSync(process.argv[3], "utf8");

function element(tagName = "div") {
  const lookups = new Map();
  const parents = new Map();
  const classes = new Set();
  return {
    tagName,
    textContent: "",
    hidden: false,
    disabled: false,
    title: "",
    value: "",
    className: "",
    dataset: {},
    style: {},
    children: [],
    opened: false,
    attributes: {},
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      remove(...names) { names.forEach((name) => classes.delete(name)); },
      toggle(name, force) {
        const on = force === undefined ? !classes.has(name) : Boolean(force);
        if (on) classes.add(name);
        else classes.delete(name);
        return on;
      },
      contains(name) { return classes.has(name); },
    },
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = kids; },
    addEventListener() {},
    removeEventListener() {},
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute() {},
    focus() {},
    scrollIntoView() {},
    getBoundingClientRect() { return { top: 0 }; },
    showModal() { this.opened = true; },
    close() { this.opened = false; },
    querySelector(selector) {
      if (!lookups.has(selector)) lookups.set(selector, element());
      return lookups.get(selector);
    },
    querySelectorAll() { return []; },
    closest(selector) {
      if (!parents.has(selector)) parents.set(selector, element());
      return parents.get(selector);
    },
  };
}

const registry = new Map();
const radio = element("input");
radio.value = "hybrid";
// The sidebar's views, so navigation and routes have items to mark.
const navItems = ["search", "sources", "status", "config", "mcp", "updates"].map((view) => {
  const item = element("li");
  item.dataset.view = view;
  return item;
});
navItems[0].classList.add("is-active");
const document = {
  getElementById(id) {
    if (!registry.has(id)) registry.set(id, element());
    return registry.get(id);
  },
  createElement: element,
  createElementNS: (_namespace, tag) => element(tag),
  createDocumentFragment: () => element("fragment"),
  createTextNode: (text) => ({ textContent: text, children: [] }),
  querySelectorAll(selector) {
    if (selector.includes("retrieval_method")) return [radio];
    if (selector === ".sidebar-item") return navItems;
    return [];
  },
  querySelector() { return null; },
  addEventListener() {},
  title: "",
};

function text(node) {
  return [node.textContent || "", ...(node.children || []).map(text)].join(" ").trim();
}

function find(node, predicate) {
  if (predicate(node)) return node;
  for (const child of node.children || []) {
    const found = find(child, predicate);
    if (found) return found;
  }
  return null;
}

const requests = [];
const entries = [];
const location = { hash: "" };
const context = {
  document,
  text,
  find,
  requests,
  navigator: { clipboard: { writeText: async () => {} } },
  fetch: async (path, options = {}) => {
    requests.push({ path, method: options.method || "GET", body: options.body || null });
    return { ok: true, status: 200, json: async () => ({}) };
  },
  Intl,
  URL,
  URLSearchParams,
  setTimeout: () => 0,
  window: {
    innerHeight: 900,
    location,
    history: {
      entries,
      pushState(_state, _title, hash) { entries.push(["push", hash]); location.hash = hash; },
      replaceState(_state, _title, hash) { entries.push(["replace", hash]); location.hash = hash; },
    },
    setTimeout: () => 0,
    open() {},
    matchMedia: () => ({ addEventListener() {} }),
  },
  result: {},
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(script, context);
await vm.runInContext(scenario, context);
process.stdout.write(JSON.stringify(context.result));
"""


@pytest.mark.parametrize("outcome", ["ready", "in_progress", "partial", "error"])
def test_live_ingestion_polling(tmp_path: Path, outcome: str) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(async () => {
  globalThis.performance = { now: () => 1000 };
  const timers = new Map();
  let next = 0;
  const listeners = new Map();
  window.setTimeout = (callback, delay) => { timers.set(++next, { callback, delay }); return next; };
  window.clearTimeout = (id) => timers.delete(id);
  window.addEventListener = (name, callback) => listeners.set(name, callback);
  window.removeEventListener = (name) => listeners.delete(name);
  globalThis.AbortController = class {
    constructor() { this.signal = { aborted: false }; }
    abort() { this.signal.aborted = true; }
  };
  state.capabilities = {};
  state.status = null;
  let finish;
  let statusFinish;
  let signal;
  let loads = 0;
  let clears = 0;
  const messages = [];
  loadWorkspace = async () => { loads++; };
  clearResults = () => { clears++; };
  toast = (message) => messages.push(message);
  api = (path, options) => {
    requests.push(path);
    if (path === '/api/ingest') return new Promise((resolve, reject) => { finish = { resolve, reject }; });
    signal = options.signal;
    return new Promise((resolve) => { statusFinish = resolve; });
  };
  const run = ingest({ preventDefault() {} });
  await ingest({ preventDefault() {} });
  const first = [...timers.values()].find((entry) => entry.delay === 2000);
  timers.clear();
  const tick = first.callback();
  const during = { requests: [...requests], loads, timers: timers.size };
  statusFinish({ ingestion_progress: {
    phase: 'extraction', source: 'Book.pdf',
    source_progress: { completed: 2, total: 8, unit: 'pages' },
    overall_progress: { completed: 1, total: 3, unit: 'sources' }, eta_seconds: 120,
  }});
  await tick;
  const progress = byId('busy-message').textContent;
  const bar = { hidden: byId('ingestion-progress').hidden,
    value: byId('ingestion-progress').value, max: byId('ingestion-progress').max };
  const second = [...timers.values()].find((entry) => entry.delay === 2000);
  timers.clear();
  const late = second.callback();
  if (OUTCOME === 'error') finish.reject(new Error('Build failed'));
  else finish.resolve({ status: OUTCOME, generation_id: 'gen_abc', chunk_count: 12 });
  await run;
  const after = byId('busy-message').textContent;
  statusFinish({ ingestion_progress: { phase: 'late' } });
  await late;
  result = { during, progress, bar, barHidden: byId('ingestion-progress').hidden,
    messages, loads, clears, busy: state.busy,
    aborted: signal.aborted, timers: timers.size, listeners: listeners.size,
    lateIgnored: byId('busy-message').textContent === after };
})();
""".replace("OUTCOME", json.dumps(outcome)),
    )
    assert result["during"]["requests"] == ["/api/ingest", "/api/status"]
    assert result["during"]["loads"] == 0
    assert "Book.pdf" in result["progress"]
    assert "pages" not in result["progress"]
    assert "1 / 3 sources" in result["progress"]
    assert result["bar"] == {"hidden": False, "value": 1, "max": 3}
    assert result["barHidden"]
    assert "ETA" not in result["progress"]
    assert result["aborted"] and result["lateIgnored"]
    assert not result["busy"]
    assert result["timers"] == result["listeners"] == 0
    assert result["loads"] == (0 if outcome == "error" else 1)
    assert result["clears"] == (1 if outcome == "ready" else 0)
    message = result["messages"][0]
    if outcome == "in_progress":
        assert "still in progress" in message and "is ready" not in message
    elif outcome == "partial":
        assert "Partial generation" in message
        assert "not selected" in message and "Load" in message
        assert "is ready" not in message
    elif outcome == "error":
        assert message == "Build failed"
    else:
        assert "is ready with 12 passages" in message


def test_source_phase_ingestion_progress_and_eta(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(() => {
  const estimate = phaseEtaEstimator();
  const status = { build_id: 'a', phase: 'extraction', source: 'Book.pdf',
    overall_progress: { completed: 5, total: 20, unit: 'sources' },
    progress: { completed: 100, total: 100, unit: 'pages' },
    source_progress: { completed: 100, total: 100, unit: 'pages' } };
  const tick = (completed, time) => {
    status.overall_progress.completed = completed;
    renderIngestionProgress(status);
    return estimate(status, time);
  };
  const samples = [tick(5, 0), tick(6, 1000), tick(7, 2000), tick(8, 3000)];
  const summary = liveIngestionSummary(status, samples[3]);
  const stalled = tick(8, 6000);
  const resets = [];
  status.phase = 'embedding'; resets.push(tick(8, 7000));
  // Three files in one poll suffice; page changes and source names do not reset it.
  status.source = 'Other.pdf'; status.progress.completed = 0;
  const batch = tick(11, 10000);
  status.build_id = 'b'; resets.push(tick(11, 11000));
  status.overall_progress.total = 30; resets.push(tick(11, 12000));
  resets.push(tick(2, 13000));
  resets.push(tick(3, 12000));
  resets.push(tick(31, 14000));
  const invalidHidden = byId('ingestion-progress').hidden;
  status.overall_progress = null;
  status.progress = { completed: 2, total: 10, unit: 'chunks' };
  renderIngestionProgress(status);
  const fallback = { hidden: byId('ingestion-progress').hidden,
    value: byId('ingestion-progress').value, max: byId('ingestion-progress').max };
  renderIngestionProgress({});
  result = { samples, summary, stalled, batch, resets, invalidHidden, fallback,
    missingHidden: byId('ingestion-progress').hidden };
})();
""",
    )
    assert result["samples"] == [None] * 4
    assert result["stalled"] is None
    assert result["batch"] is None
    assert result["resets"] == [None] * 6
    assert result["invalidHidden"] and result["missingHidden"]
    assert result["fallback"] == {"hidden": False, "value": 2, "max": 10}
    assert result["summary"] == "extraction · Book.pdf · 8 / 20 sources"


def test_phase_local_ingestion_eta(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(() => {
  const estimate = phaseEtaEstimator();
  const status = { build_id: 'a', phase: 'embedding', eta_seconds: null,
    progress: { completed: 10, total: 100, unit: 'chunks' },
    overall_progress: { completed: 999, total: 1000 } };
  const tick = (completed, time) => {
    status.progress.completed = completed;
    return estimate(status, time);
  };
  const samples = [tick(10, 0), tick(10, 1000), tick(20, 2000), tick(30, 4000)];
  const summary = liveIngestionSummary(status, samples[3]);
  const stalled = tick(30, 8000);
  const resets = [];
  resets.push(tick(5, 9000));
  tick(10, 10000); tick(15, 11000);
  status.phase = 'dense_indexing'; resets.push(tick(20, 12000));
  tick(25, 13000); tick(30, 14000);
  status.build_id = 'b'; resets.push(tick(35, 15000));
  tick(40, 16000); tick(45, 17000);
  status.progress.total = 200; resets.push(tick(50, 18000));
  tick(55, 19000); tick(60, 20000);
  status.progress = null; resets.push(estimate(status, 21000));
  status.progress = { completed: 65, total: 200, unit: 'chunks' };
  resets.push(estimate(status, 22000));
  const estimating = liveIngestionSummary(status);
  status.eta_seconds = 120;
  result = { samples, stalled, resets, summary, estimating,
    backend: liveIngestionSummary(status, 999),
    fresh: phaseEtaEstimator()(status, 23000) };
})();
""",
    )
    assert result["samples"] == [None, None, None, 14]
    assert result["stalled"] == 28
    assert result["resets"] == [None] * 6
    assert "Phase ETA about 15 sec" in result["summary"]
    assert "Phase ETA estimating" in result["estimating"]
    assert "Phase ETA about 16 min 40 sec" in result["backend"]
    assert result["fresh"] is None


@pytest.mark.parametrize("phase", ["embedding", "dense_indexing", "qdrant_indexing"])
def test_passage_eta_scope_completion_and_rounding(tmp_path: Path, phase: str) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(() => {
  const estimate = phaseEtaEstimator();
  const status = { build_id: 'a', phase: PHASE, source: 'First.pdf',
    progress: { completed: 40, total: 100, unit: 'chunks' } };
  const first = estimate(status, 0);
  status.progress.completed = 50;
  const second = estimate(status, 10000);
  status.source = 'Second.pdf';
  status.progress.completed = 60;
  const third = estimate(status, 20000);
  status.progress.completed = 100;
  const complete = estimate(status, 30000);
  const completeSummary = liveIngestionSummary(status, 0);
  const excluded = ['hashing', 'extraction', 'chunking'].map(phase =>
    liveIngestionSummary({ ...status, phase, eta_seconds: 120 }, 120));
  status.progress = { completed: 1, total: 10, unit: 'pages' };
  const pages = liveIngestionSummary(status, 120);
  status.progress = { completed: 1, unit: 'chunks' };
  const unknown = liveIngestionSummary(status, 120);
  result = { first, second, third, complete, completeSummary, excluded, pages, unknown,
    rounded: [0.1, 54, 58, 61, 64].map(formatPhaseEta) };
})();
""".replace("PHASE", json.dumps(phase)),
    )
    assert result["first"] is None and result["second"] is None
    assert result["third"] == 40
    assert result["complete"] is None
    for summary in [
        result["completeSummary"],
        *result["excluded"],
        result["pages"],
        result["unknown"],
    ]:
        assert "ETA" not in summary
    assert result["rounded"] == ["5 sec", "55 sec", "1 min", "1 min", "1 min 5 sec"]


def test_ingestion_poll_failure_and_pagehide_cleanup(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(async () => {
  const timers = new Map();
  const listeners = new Map();
  let next = 0;
  window.setTimeout = (callback, delay) => { timers.set(++next, { callback, delay }); return next; };
  window.clearTimeout = (id) => timers.delete(id);
  window.addEventListener = (name, callback) => listeners.set(name, callback);
  window.removeEventListener = (name) => listeners.delete(name);
  globalThis.AbortController = class {
    constructor() { this.signal = { aborted: false }; }
    abort() { this.signal.aborted = true; }
  };
  let signal;
  api = async (path, options) => { signal = options.signal; throw new Error('Unavailable'); };
  byId('busy-message').textContent = 'Building';
  const stop = pollIngestionProgress();
  const first = [...timers.values()][0];
  timers.clear();
  await first.callback();
  const retried = [...timers.values()].some((entry) => entry.delay === 2000);
  listeners.get('pagehide')();
  stop();
  result = { retried, message: byId('busy-message').textContent,
    timers: timers.size, listeners: listeners.size, aborted: signal.aborted,
    fallback: liveIngestionSummary({ phase: 'embedding', progress: { completed: 3, total: 10, unit: 'chunks' } }) };
})();
""",
    )
    assert result["retried"] and result["aborted"]
    assert result["message"] == "Building"
    assert result["timers"] == result["listeners"] == 0
    assert result["fallback"] == "embedding · 3 / 10 chunks · Phase ETA estimating…"


def _drive_page(tmp_path: Path, scenario: str) -> dict[str, Any]:
    """Run the page script under the stub document, then one scenario after it."""

    if shutil.which("node") is None:
        pytest.skip("node is not installed, so the page cannot be driven here.")
    harness = tmp_path / "harness.mjs"
    harness.write_text(_PAGE_HARNESS, encoding="utf-8")
    steps = tmp_path / "scenario.js"
    steps.write_text(scenario, encoding="utf-8")
    asset = (
        Path(__file__).parents[2] / "src/research_rag/surfaces/workspace/static/app.js"
    )
    completed = subprocess.run(
        ["node", str(harness), str(asset), str(steps)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"the page harness failed: {completed.stderr.strip()[:600]}")
    return json.loads(completed.stdout)


_CURRENT_STATUS = """
const base = {
  ready: true,
  stale: false,
  generation_upgrade_required: false,
  hybrid_ready: true,
  indexed_source_count: 106,
  selected_source_count: 106,
  searchable_source_count: 106,
  chunk_count: 24829,
  available_retrieval_methods: ["hybrid"],
  last_build_metrics: { reused_vector_count: 24089, created_vector_count: 740 },
};
"""


def test_the_build_button_offers_the_build_the_corpus_needs(tmp_path: Path) -> None:
    """A current corpus is offered no build in the header, and nothing it is
    offered there reuses nothing.

    The header button used to read "Regenerate" on a current corpus and send
    `force_recompute`, which discards every reusable extraction and vector: the
    most expensive build was the most prominent control on every view. An
    ingestion already reuses only what is unchanged, so a stale, outdated, or
    empty corpus is served by an ordinary one, and a checkpointed build is resumed
    with the flag it started under.
    """

    result = _drive_page(
        tmp_path,
        _CURRENT_STATUS
        + """
state.profile = { capabilities: { ingestion: true, force_recompute: true, documents: true } };
const plans = {};
const cases = {
  current: base,
  stale: { ...base, stale: true },
  upgrade: { ...base, generation_upgrade_required: true, upgrade_reasons: ["layout_extraction"] },
  empty: { ...base, ready: false },
  checkpoint: {
    ...base,
    ingestion_progress: {
      phase: "embedding",
      progress: { completed: 10, total: 20, unit: "chunks" },
      parameters: { force_recompute: true },
    },
  },
};
for (const [name, status] of Object.entries(cases)) {
  renderStatus(status);
  const header = document.getElementById("ingest-button");
  plans[name] = {
    hidden: header.hidden,
    label: header.textContent,
    force: state.ingestPlan ? state.ingestPlan.force : null,
    rebuildHidden: document.getElementById("rebuild-button").hidden,
    noticeHidden: document.getElementById("status-notice").hidden,
    noticeAction: document.getElementById("notice-action").textContent,
    notice: document.getElementById("status-notice-text").textContent,
  };
}
renderStatus({ ...base, stale: true });
openIngest(state.ingestPlan);
const dialog = document.getElementById("ingest-dialog");
const cost = document.getElementById("ingest-cost");
result.ordinary = {
  force: state.forceRecompute,
  costHidden: cost.hidden,
  title: dialog.querySelector("h2").textContent,
};
openIngest(REBUILD_PLAN);
result.rebuild = {
  force: state.forceRecompute,
  costHidden: cost.hidden,
  cost: cost.textContent,
  title: dialog.querySelector("h2").textContent,
  opened: dialog.opened,
};
result.plans = plans;
""",
    )

    plans = result["plans"]
    # A current corpus has nothing for an ingestion to do, so the header offers
    # nothing; the rebuild is on the Status view, named for what it costs.
    assert plans["current"]["hidden"] is True
    assert plans["current"]["force"] is None
    assert plans["current"]["rebuildHidden"] is False
    assert plans["current"]["noticeHidden"] is True
    # Every other state is an ordinary ingestion, which reuses what is unchanged.
    for name, label in (
        ("stale", "Ingest changes"),
        ("upgrade", "Upgrade generation"),
        ("empty", "Create generation"),
    ):
        assert plans[name]["hidden"] is False
        assert plans[name]["label"] == label
        assert plans[name]["force"] is False
        # The notice offers the same build the header does.
        assert plans[name]["noticeAction"] == label
    # Nothing to rebuild before the first build, nor while a build can resume.
    assert plans["empty"]["rebuildHidden"] is True
    assert plans["checkpoint"]["rebuildHidden"] is True
    # A checkpointed build resumes under the flag it started with, and says where
    # it stopped.
    assert plans["checkpoint"]["label"] == "Resume build"
    assert plans["checkpoint"]["force"] is True
    assert "embedding phase at 10 of 20 chunks" in plans["checkpoint"]["notice"]
    assert "Regenerate" not in plans["upgrade"]["notice"]

    assert result["ordinary"] == {
        "force": False,
        "costHidden": True,
        "title": "Ingest changes",
    }
    # Only the named rebuild sends the flag, and its dialog states the cost from
    # the counts the server reported.
    rebuild = result["rebuild"]
    assert rebuild["force"] is True
    assert rebuild["opened"] is True
    assert rebuild["title"] == "Rebuild from scratch"
    assert rebuild["costHidden"] is False
    assert "all 106 sources are extracted again" in rebuild["cost"]
    assert "reused 24,089 vectors and created 740" in rebuild["cost"]


def test_the_status_view_shows_what_the_health_checks_found(tmp_path: Path) -> None:
    """A degraded project is not reported as ready.

    The status payload carried `checks`, `degraded`, and `blocked_by`, and the
    page drew none of them while the header said "ready". A condition now shows
    its reason and the server's own remedy with a button that copies it exactly,
    every check sits behind one disclosure, and the header counts what needs
    action.
    """

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: {} };
const remedy = "research-rag --project-root /p doctor --prefetch-models";
const warning = { check: "embedding_model", reason: "The pinned model is not cached.", remedy };
renderHealth({
  checks: [
    { check: "lock", state: "ok", reason: "The lock is free.", remedy_command: null },
    { check: "embedding_model", state: "warn", reason: "The pinned model is not cached.", remedy_command: remedy },
  ],
  degraded: [warning],
  blocked_by: [],
});
const section = document.getElementById("health-section");
const conditions = document.getElementById("health-conditions").children;
const copy = find(conditions[0], (node) => node.dataset && node.dataset.action === "copy-remedy");
result.degraded = {
  hidden: section.hidden,
  summary: document.getElementById("health-summary").textContent,
  conditions: conditions.length,
  condition: text(conditions[0]),
  conditionClass: conditions[0].className,
  copied: copy ? copy.dataset.value : null,
  note: document.getElementById("health-check-note").textContent,
  rows: document.getElementById("health-checks").children.length,
};
result.pill = connectionHealth({ degraded: [warning], blocked_by: [] });
result.blockedPill = connectionHealth({
  degraded: [],
  blocked_by: [{ check: "app.serving", reason: "No app is serving.", remedy: "research-rag start" }],
}).slice(0, 2);
result.healthyPill = connectionHealth({ degraded: [], blocked_by: [] });

renderHealth({ checks: [{ check: "lock", state: "ok", reason: "free" }] });
result.healthy = {
  hidden: section.hidden,
  summary: document.getElementById("health-summary").textContent,
  listHidden: document.getElementById("health-conditions").hidden,
};
renderHealth({ ready: true });
result.absent = section.hidden;
""",
    )

    degraded = result["degraded"]
    assert degraded["hidden"] is False
    assert degraded["summary"] == "1 to act on"
    assert degraded["conditions"] == 1
    assert "Warning" in degraded["condition"]
    assert "Embedding model" in degraded["condition"]
    assert "The pinned model is not cached." in degraded["condition"]
    assert "health-condition-warn" in degraded["conditionClass"]
    # The copy button carries the server's command exactly.
    assert (
        degraded["copied"] == "research-rag --project-root /p doctor --prefetch-models"
    )
    assert degraded["note"] == "1 of 2 passed"
    assert degraded["rows"] == 2
    assert result["pill"] == [
        "warn",
        "Local · 1 warning",
        "The pinned model is not cached.",
    ]
    assert result["blockedPill"] == ["blocked", "Local · 1 blocked"]
    assert result["healthyPill"] == ["ready", "Local · ready", ""]
    # A healthy project says so once, in the badge; a host with no checks shows
    # nothing.
    assert result["healthy"]["hidden"] is False
    assert result["healthy"]["summary"] == "All passed"
    assert result["healthy"]["listHidden"] is True
    assert result["absent"] is True


def test_a_list_that_cannot_narrow_a_search_is_not_drawn(tmp_path: Path) -> None:
    """One value every searchable source carries filters nothing.

    A project tag on all 106 sources and a single language were drawn as lists to
    pick from, above the results. A list now appears only when choosing from it
    changes what a search reads.
    """

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { category_partitions: true, project_metadata: true } };
state.status = { searchable_source_count: 106 };
const group = (id) => document.getElementById(id).closest(".pick-group");
group("partition-chips").dataset.capability = "category_partitions";
group("project-chips").dataset.capability = "project_metadata";
renderInventory("project-chips", [{ project: "all", searchable_source_count: 106 }], "project", "project-filter");
result.everySource = group("project-chips").hidden;
renderInventory("partition-chips", [{ category: "some", searchable_source_count: 40 }], "category", "partition-filter");
result.someSources = group("partition-chips").hidden;
renderInventory("partition-chips", [], "category", "partition-filter");
result.none = group("partition-chips").hidden;
state.profile = { capabilities: {} };
renderInventory("partition-chips", [
  { category: "a", searchable_source_count: 4 },
  { category: "b", searchable_source_count: 5 },
], "category", "partition-filter");
result.capabilityOff = group("partition-chips").hidden;
""",
    )

    assert result == {
        "everySource": True,
        "someSources": False,
        "none": True,
        "capabilityOff": True,
    }


def test_the_results_follow_the_query_and_the_filters_fold_into_one_drawer() -> None:
    """A search's results start under its query, not under the filter lists.

    The explanation, the fields, and the lists a reader picks from were stacked
    between the query and the results, so 22 category buttons stood above the
    first passage. All three now live inside the one filter drawer.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    search = page.split('id="search-view"', 1)[1].split("</main>", 1)[0]
    drawer = search.split('id="filter-fields"', 1)[1].split("</section>", 1)[0]
    for element in (
        'class="form-note measure filter-intro"',
        'id="filter-count"',
        'id="filter-picks"',
        'id="partition-chips"',
        'id="project-chips"',
        'id="language-chips"',
    ):
        assert element in drawer
    # Nothing but the closed drawer stands between the query and the results.
    between = search.split("</form>", 1)[1].split('id="search-summary"', 1)[0]
    assert between.count("<section") == 1
    assert between.count('class="filter-drawer"') == 1
    assert "<h2" not in between.split('id="filter-fields"', 1)[0]
    # The page scrolls to the results only when they are out of sight.
    assert "window.innerHeight * 0.75" in script


def test_a_source_card_names_the_reach_of_each_exclusion() -> None:
    """Two adjacent links said "Exclude" and "Exclude from search".

    One recorded a project-wide exclusion that every surface reads; the other
    filled a filter for one search on another view. The second was the one
    painted red, because the rule coloured whichever link came last.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert '"Exclude from project…",' in script
    assert '"action-button action-button-danger",' in script
    assert 'button("Search only this source", "only-source"' in script
    assert 'button("Search without this source", "exclude-from-search"' in script
    assert 'button("Exclude", "exclude-source"' not in script
    assert 'button("Exclude from search"' not in script
    # A filter chosen on Sources opens the search it filters.
    assert 'searchWithSource("include-source-filter", value)' in script
    assert 'switchView("search", { moveFocus: true });' in script
    # The colour follows the class, never the position.
    assert ".action-button-danger {" in css
    assert ".source-card-actions .action-button:last-child" not in css
    assert "<h2>Exclude this source from the project?</h2>" in page
    assert ">Exclude from project</button>" in page


def test_the_sources_view_shows_five_sources_a_page(tmp_path: Path) -> None:
    """A collection is read a page at a time, and the filter reads all of it.

    Every source was drawn at once, one card under another, so a 106-source
    project was one column a reader scrolled through to find one source. The
    list now holds five a page, one window's worth, with a pager under it; the
    filter still matches across every source and starts its matches from their
    first page, and a refresh returns the reader to the page they were on.
    """

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: {} };
const sources = Array.from({ length: 23 }, (_, index) => ({
  title: `Source ${index + 1}`,
  authors: index === 22 ? ["Needle"] : ["Author"],
  source_relative_path: `s${index + 1}.pdf`,
  source_id: `src_${index}`,
  document_id: `doc_${index}`,
}));
const excluded = Array.from({ length: 12 }, (_, index) => ({
  source_relative_path: `x${index + 1}.pdf`,
  reason: "duplicate",
}));
const list = document.getElementById("source-list");
const range = document.getElementById("source-range");
const pager = document.getElementById("source-pager");
const labels = () => pager.children.map((item) => item.textContent);
const read = () => ({ cards: list.children.length, range: range.textContent, pages: labels() });

renderSources({ ready: true, sources, excluded_sources: excluded });
result.first = {
  ...read(),
  pagerHidden: pager.hidden,
  previousDisabled: pager.children[0].disabled,
  current: pager.children.find((item) => item.dataset.value === "1").dataset.list,
};
goToSourcePage("sources", 5);
result.last = { ...read(), nextDisabled: pager.children[pager.children.length - 1].disabled };

document.getElementById("source-filter").value = "needle";
filterSources();
result.filtered = { ...read(), pagerHidden: pager.hidden, page: state.sourcePages.sources };
document.getElementById("source-filter").value = "nothing like it";
filterSources();
result.unmatched = { text: text(list), rangeHidden: range.hidden };

document.getElementById("source-filter").value = "";
goToSourcePage("sources", 2);
renderSources({ ready: true, sources, excluded_sources: excluded });
result.refreshed = read().range;
renderSources({ ready: true, sources: sources.slice(0, 5), excluded_sources: [] });
result.shrunk = { ...read(), pagerHidden: pager.hidden, page: state.sourcePages.sources };

renderSources({ ready: true, sources, excluded_sources: excluded });
result.excluded = {
  cards: document.getElementById("excluded-list").children.length,
  pagerHidden: document.getElementById("excluded-pager").hidden,
};
result.window = pagerNumbers(6, 11);
result.short = pagerNumbers(1, 3);
""",
    )

    assert result["first"] == {
        "cards": 5,
        "range": "Showing 1–5 of 23 sources",
        "pages": ["Previous", "1", "2", "…", "5", "Next"],
        "pagerHidden": False,
        "previousDisabled": True,
        "current": "sources",
    }
    assert result["last"] == {
        "cards": 3,
        "range": "Showing 21–23 of 23 sources",
        "pages": ["Previous", "1", "…", "4", "5", "Next"],
        "nextDisabled": True,
    }
    # The filter matches across every page and starts its own matches at one.
    assert result["filtered"] == {
        "cards": 1,
        "range": "Showing 1 of 1 matching source, 23 in all",
        "pages": [],
        "pagerHidden": True,
        "page": 1,
    }
    assert result["unmatched"] == {
        "text": "No source matches this filter.",
        "rangeHidden": True,
    }
    # A refresh keeps the page, and a page that no longer exists is clamped.
    assert result["refreshed"] == "Showing 6–10 of 23 sources"
    assert result["shrunk"]["cards"] == 5
    assert result["shrunk"]["pagerHidden"] is True
    assert result["shrunk"]["page"] == 1
    # The excluded list pages the same way.
    assert result["excluded"] == {"cards": 5, "pagerHidden": False}
    # A long pager shows its ends and the current page's neighbours.
    assert result["window"] == [1, None, 5, 6, 7, None, 11]
    assert result["short"] == [1, 2, 3]


def test_the_project_picker_offers_only_a_project_that_is_not_on_screen(
    tmp_path: Path,
) -> None:
    """A picker over one project, or an "Open" for the project already open,
    offered a second tab of the same workspace."""

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { projects: true } };
const here = { project_name: "here", running: true, url: "http://127.0.0.1:5051" };
const there = { project_name: "there", running: true, url: "http://127.0.0.1:5052" };
renderProjects({ projects: [here], current: "here", message: "" });
result.single = document.getElementById("project-selector").hidden;
renderProjects({ projects: [here, there], current: "here", message: "" });
result.several = document.getElementById("project-selector").hidden;
result.openForCurrent = !document.getElementById("project-open").hidden;
document.getElementById("project-select").value = "there";
showProjectActions();
result.openForOther = !document.getElementById("project-open").hidden;
""",
    )

    assert result == {
        "single": True,
        "several": False,
        "openForCurrent": False,
        "openForOther": True,
    }


def test_the_workspace_calls_a_passage_a_passage(tmp_path: Path) -> None:
    """The page said "chunk" in some places and "passage" in others.

    The server's messages keep "chunk", the word its agent tools use, so the page
    states each decision and the exclusion list in its own words from the fields
    the server returned. An empty list says so once rather than twice.
    """

    with _panel_host() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
    for retired in (
        "Exclude this chunk",
        "Restore this chunk",
        "Chunk exclusions",
        "Exclude chunk<",
        "No chunk is excluded",
        "Chunk restored.",
    ):
        assert retired not in page
        assert retired not in script

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { chunk_exclusion: true } };
renderChunkExclusions({ exclusions: [], message: "No chunk is excluded from retrieval." });
const message = document.getElementById("chunk-exclusion-message");
const list = document.getElementById("chunk-exclusion-list");
result.empty = { messageHidden: message.hidden, list: text(list) };
renderChunkExclusions({
  exclusions: [
    { chunk_id: "c1", in_current_generation: true, reason: "repeats" },
    { chunk_id: "c2", in_current_generation: false, reason: "repeats" },
  ],
  message: "1 of the 2 excluded chunks are withheld.",
});
result.mixed = { messageHidden: message.hidden, message: message.textContent, list: text(list) };
result.decisions = [
  passageDecisionMessage({ status: "changed", in_current_generation: true }, false),
  passageDecisionMessage({ status: "changed", in_current_generation: false }, false),
  passageDecisionMessage({ status: "changed", in_current_generation: true }, true),
  passageDecisionMessage({ status: "unchanged" }, false),
];
""",
    )

    assert result["empty"] == {
        "messageHidden": True,
        "list": "No passage is excluded from search.",
    }
    assert result["mixed"]["messageHidden"] is False
    assert result["mixed"]["message"].startswith(
        "1 of 2 excluded passages are withheld from current search"
    )
    assert "chunk" not in result["mixed"]["message"]
    assert "Not in the current generation" in result["mixed"]["list"]
    assert result["decisions"] == [
        "Passage excluded from search. The original file is unchanged.",
        "Passage exclusion saved. This generation does not hold the passage, so no current result changes.",
        "Passage restored to search.",
        "This passage is already excluded with this reason.",
    ]


def test_the_generations_are_one_table_with_their_total_size(tmp_path: Path) -> None:
    """Ten builds were ten cards of six labelled lines each; a table compares
    them on one screen and says what they hold on disk together."""

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { generations: true } };
state.status = { retained_generation_bytes: 3 * 1024 * 1024 };
renderGenerations([
  { generation_id: "g-new", is_current: true, created_at: "2026-10-05T08:38:11Z", chunk_count: 24829, document_count: 106, size_bytes: 2 * 1024 * 1024 },
  { generation_id: "g-old", is_current: false, created_at: "2026-10-02T08:08:53Z", chunk_count: 24483, document_count: 102, size_bytes: 1024 * 1024, manifest_error: "truncated" },
]);
const container = document.getElementById("generation-chips");
const table = container.children[1].children[0];
const rows = table.children[1].children;
result.total = container.children[0].textContent;
result.columns = table.children[0].children[0].children.map((cell) => text(cell));
result.rows = rows.length;
result.current = text(rows[0]);
result.currentRemovable = Boolean(find(rows[0], (item) => item.textContent === "Remove"));
result.old = text(rows[1]);
result.oldRemovable = Boolean(find(rows[1], (item) => item.textContent === "Remove"));
""",
    )

    assert result["total"] == "2 builds use 3.0 MB on disk."
    assert result["columns"] == [
        "Generation",
        "Built",
        "Passages",
        "Sources",
        "Size",
        "Action",
    ]
    assert result["rows"] == 2
    assert "g-new" in result["current"] and "In use" in result["current"]
    assert "24,829" in result["current"] and "2.0 MB" in result["current"]
    assert result["currentRemovable"] is False
    assert "Retained" in result["old"]
    assert "Manifest unreadable: truncated" in result["old"]
    assert result["oldRemovable"] is True


def test_a_retained_generation_offers_load_and_a_damaged_one_does_not(
    tmp_path: Path,
) -> None:
    """Load is offered on a valid non-current row, never on the current one or a
    row whose manifest could not be read, and the recorded configuration is
    stated as what the build recorded."""

    result = _drive_page(
        tmp_path,
        r"""
state.profile = { capabilities: { generations: true } };
const chunking = { backend: "UltraRAG token chunker", tokenizer: "gpt2", unit: "tokens", chunk_size: 512, chunk_overlap: 64 };
const retrieval = {
  default_method: "hybrid",
  available_methods: ["bm25", "dense", "hybrid"],
  bm25: { backend: "UltraRAG BM25", language: "en", tokenizer: "default" },
  dense: { backend: "portable float32 vectors (exact cosine scan)", embedding_model: "BAAI/bge-small-en-v1.5", embedding_model_revision: "rev-1", embedding_dimension: 384 },
  reranker: { optional: true, model: "BAAI/bge-reranker-base", model_revision: "rev-2" },
};
renderGenerations([
  { generation_id: "g-current", is_current: true, created_at: "2026-10-05T08:38:11Z", chunk_count: 10, document_count: 2, size_bytes: 1024, chunking, retrieval, retrieval_policy_fingerprint: "fp-1", extraction_policy_version: 3, cleaning_policy_version: 2, artifact_policy_version: 1 },
  { generation_id: "g-old", is_current: false, created_at: "2026-10-02T08:08:53Z", chunk_count: 9, document_count: 2, size_bytes: 1024, chunking, retrieval, retrieval_policy_fingerprint: "fp-2", extraction_policy_version: 3, cleaning_policy_version: 2, artifact_policy_version: 1 },
  { generation_id: "g-damaged", is_current: false, manifest_error: "truncated", size_bytes: 10 },
]);
const container = document.getElementById("generation-chips");
const body = container.children[1].children[0].children[1];
const actionLabels = (row) => row.children[row.children.length - 1].children.map((button) => button.textContent);
result.currentActions = actionLabels(body.children[0]);
result.oldActions = actionLabels(body.children[1]);
result.damagedActions = actionLabels(body.children[2]);
result.damagedText = text(body.children[2]);
const row = body.children[1];
result.summary = find(row, (item) => item.className === "recorded-config-summary").textContent;
result.buttons = row.children[row.children.length - 1].children.map((button) => [button.textContent, button.disabled]);
setBusy(true, "Loading the generation…");
result.busyButtons = row.children[row.children.length - 1].children.map((button) => button.disabled);
setBusy(false);
const config = find(row, (item) => item.className === "recorded-config");
result.configTitle = config.children[0].textContent;
result.configNote = config.children[1].textContent;
result.facts = text(config.children[2]);
result.blocks = config.children.filter((item) => item.tagName === "pre").map((item) => JSON.parse(item.textContent));
const policyLabel = config.children.find((item) => item.textContent === "Policy facts (recorded)");
result.policies = text(config.children[config.children.indexOf(policyLabel) + 1]);
""",
    )

    assert result["currentActions"] == []
    assert result["oldActions"] == ["Load", "Remove"]
    assert result["damagedActions"] == ["Remove"]
    assert "Not possible" in result["damagedText"]
    assert "Embedding model: BAAI/bge-small-en-v1.5" in result["summary"]
    assert "Embedding revision: rev-1" in result["summary"]
    assert "Embedding dimension: 384" in result["summary"]
    assert "Reranker model: BAAI/bge-reranker-base" in result["summary"]
    assert "Reranker revision: rev-2" in result["summary"]
    assert "Chunk size: 512" in result["summary"]
    assert "Chunk overlap: 64" in result["summary"]
    assert "Chunk tokenizer: gpt2" in result["summary"]
    assert "BM25 language: en" in result["summary"]
    assert "Retrieval methods: bm25, dense, hybrid" in result["summary"]
    assert result["buttons"] == [["Load", False], ["Remove", False]]
    # A generation action is held while another is running, so two operations
    # cannot change which generation a search reads at once.
    assert result["busyButtons"] == [True, True]
    assert result["configTitle"] == "Recorded build configuration"
    assert "not the project's active settings" in result["configNote"]
    assert result["blocks"][0]["chunk_size"] == 512
    assert result["blocks"][1]["dense"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert "Retrieval policy fingerprint fp-2" in result["policies"]
    assert "Not recorded" not in result["policies"]


def test_a_generations_recorded_config_is_text_not_markup(tmp_path: Path) -> None:
    """A manifest that carried markup is shown as the text it is, never drawn."""

    result = _drive_page(
        tmp_path,
        r"""
state.profile = { capabilities: { generations: true } };
const payload = "<img src=x onerror=alert(1)>";
renderGenerations([{ generation_id: "g-x", is_current: false, chunking: { tokenizer: payload }, retrieval: { dense: { embedding_model: payload } } }]);
const row = document.getElementById("generation-chips").children[1].children[0].children[1].children[0];
result.summary = find(row, (item) => item.className === "recorded-config-summary").textContent;
const found = [];
const walk = (item) => { if (item.tagName === "img") found.push(item); (item.children || []).forEach(walk); };
walk(row);
result.images = found.length;
""",
    )

    assert "<img src=x onerror=alert(1)>" in result["summary"]
    assert result["images"] == 0


def test_a_generation_that_recorded_nothing_says_so(tmp_path: Path) -> None:
    """A legacy record with no chunking or retrieval never borrows today's
    defaults; every fact it did not record is named as missing."""

    result = _drive_page(
        tmp_path,
        r"""
state.profile = { capabilities: { generations: true } };
renderGenerations([
  { generation_id: "g-legacy", is_current: false, chunking: null, retrieval: null },
  { generation_id: "g-partial", is_current: false, retrieval: { available_methods: ["bm25"] } },
]);
const body = document.getElementById("generation-chips").children[1].children[0].children[1];
const summaryOf = (row) => find(row, (item) => item.className === "recorded-config-summary").textContent;
result.legacy = summaryOf(body.children[0]);
result.partial = summaryOf(body.children[1]);
const config = find(body.children[0], (item) => item.className === "recorded-config");
result.legacyFacts = text(config.children[2]);
const policyLabel = config.children.find((item) => item.textContent === "Policy facts (recorded)");
result.legacyPolicies = text(config.children[config.children.indexOf(policyLabel) + 1]);
result.legacyBlocks = config.children.filter((item) => item.tagName === "pre").map((item) => item.textContent);
""",
    )

    assert "Embedding model: Not recorded" in result["legacy"]
    assert "Reranker model: Not recorded" in result["legacy"]
    assert "Chunk size: Not recorded" in result["legacy"]
    assert "BM25 language: Not recorded" in result["legacy"]
    assert "Retrieval methods: Not recorded" in result["legacy"]
    assert "Not recorded" in result["legacyFacts"]
    assert "Not recorded" in result["legacyPolicies"]
    assert result["legacyBlocks"] == ["null", "null"]
    assert "Embedding model: Not recorded" in result["partial"]
    assert "Retrieval methods: bm25" in result["partial"]


@pytest.mark.parametrize("outcome", ["changed", "unchanged"])
def test_loading_a_generation_confirms_then_refreshes_the_workspace(
    tmp_path: Path,
    outcome: str,
) -> None:
    """A load asks once, changes the selected generation, clears the old results,
    and refreshes the whole status so the corpus facts follow the selection."""

    result = _drive_page(
        tmp_path,
        r"""
(async () => {
state.profile = { capabilities: { generations: true, sources: true } };
const asked = [];
fetch = async (path, options = {}) => {
  asked.push({ path, method: options.method || "GET", body: options.body || null });
  if (path === "/api/generations/use") {
    return { ok: true, status: 200, json: async () => ({ status: "changed", generation_id: "g-old", message: "Search now reads generation g-old." }) };
  }
  if (path === "/api/status") {
    return { ok: true, status: 200, json: async () => ({ ready: true, generation_id: "g-old", available_retrieval_methods: ["bm25"], message: "Current generation is BM25-only." }) };
  }
  return { ok: true, status: 200, json: async () => ({}) };
};
openGenerationLoad("g-old");
result.dialogOpen = document.getElementById("generation-load-dialog").opened;
result.name = document.getElementById("generation-load-name").textContent;
document.getElementById("generation-load-id").value = "g-old";
document.getElementById("search-summary").hidden = false;
const connection = document.getElementById("connection-state");
connection.append(document.createElement("span"));
connection.lastElementChild = connection.children[0];
await loadGeneration({ preventDefault() {} });
result.dialogClosed = !document.getElementById("generation-load-dialog").opened;
result.cleared = document.getElementById("search-summary").hidden;
const loadCall = asked.find((call) => call.path === "/api/generations/use");
result.loadBody = JSON.parse(loadCall.body);
result.loadMethod = loadCall.method;
result.refreshed = asked.some((call) => call.path === "/api/status" && call.method === "GET");
result.sources = asked.some((call) => call.path === "/api/sources");
result.toast = text(document.getElementById("toast-region"));
})();
""".replace('status: "changed"', f"status: {json.dumps(outcome)}"),
    )

    assert result["dialogOpen"] is True
    assert result["name"] == "g-old"
    assert result["dialogClosed"] is True
    assert result["loadMethod"] == "POST"
    assert result["loadBody"] == {"generation_id": "g-old"}
    assert result["refreshed"] is True
    assert result["sources"] is True
    assert result["cleared"] is True
    assert "Search now reads generation g-old." in result["toast"]


def test_a_refused_load_keeps_the_dialog_and_does_not_refresh(
    tmp_path: Path,
) -> None:
    """The server's refusal is shown beside the choice that caused it, and the
    page does not pretend the selection happened."""

    result = _drive_page(
        tmp_path,
        r"""
(async () => {
state.profile = { capabilities: { generations: true } };
const asked = [];
fetch = async (path, options = {}) => {
  asked.push(path);
  return { ok: false, status: 400, json: async () => ({ error: "Generation g-old failed validation and was not selected: damaged." }) };
};
openGenerationLoad("g-old");
document.getElementById("generation-load-id").value = "g-old";
await loadGeneration({ preventDefault() {} });
result.dialogOpen = document.getElementById("generation-load-dialog").opened;
result.statusCalls = asked.filter((path) => path === "/api/status").length;
result.errorHidden = document.getElementById("generation-load-error").hidden;
result.error = document.getElementById("generation-load-error").textContent;
result.toastCount = document.getElementById("toast-region").children.length;
result.currentResults = document.getElementById("search-summary").hidden;
})();
""",
    )

    assert result["dialogOpen"] is True
    assert result["statusCalls"] == 0
    assert result["errorHidden"] is False
    assert result["error"].startswith("Not possible: ")
    assert "failed validation" in result["error"]
    # The refusal is shown where the choice was made, not also as a toast.
    assert result["toastCount"] == 0
    # The current generation and what a search returned are untouched.
    assert result["currentResults"] is False


def test_small_layout_and_keyboard_fixes_hold() -> None:
    """Three small faults: a page number broke onto two lines beside a long
    title, Ctrl+Enter did nothing in the query box, and the clients summary put
    a session count beside the words "Who is attached"."""

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        css = client.get("/assets/app.css").text

    assert ".result-card-header .locator-badge {\n  flex: 0 0 auto;" in css
    assert 'aria-keyshortcuts="Control+Enter Meta+Enter"' in page
    assert "Ctrl+Enter searches from the query box." in page
    assert "(event.ctrlKey || event.metaKey)" in script
    assert 'byId("search-form").requestSubmit();' in script
    assert "Who is attached" not in page
    assert '<span class="drawer-title">Clients</span>' in page


def test_a_section_opens_with_at_most_one_sentence() -> None:
    """Most sections opened with two to four sentences a reader had already read.

    Each introduction under a view's heading now says one thing. The SQL console
    is exempt: this app never shows it, and its host writes its own wording.
    """

    with _panel_host() as client:
        page = client.get("/").text

    views = page.split("<main", 1)[1].split("</main>", 1)[0]
    views = re.sub(r'<div id="sql-console".*?</form>', "", views, flags=re.DOTALL)
    intros = [
        " ".join(body.split())
        for body in re.findall(
            r'<p class="form-note measure[^"]*"[^>]*>(.*?)</p>', views, flags=re.DOTALL
        )
    ]
    assert len(intros) >= 10
    for intro in intros:
        assert len(re.findall(r"[.!?](?:\s|$)", intro)) <= 1, intro


def test_each_view_is_a_page_that_back_forward_and_reload_return_to(
    tmp_path: Path,
) -> None:
    """A reload lost the search, and Back left the workspace.

    The view and what it shows now live in the fragment: a submitted search and
    a page of sources each add a history entry, a reload draws what the
    fragment names, and returning to a search already on screen does not run
    it again.
    """

    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = {
  application_name: "Research RAG",
  capabilities: { sources: true, documents: true, category_partitions: true },
};
const topK = document.getElementById("top-k");
topK.options = ["5", "8", "12", "20"].map((value) => ({ value }));
topK.value = "8";
const searches = () => requests.filter((request) => request.path === "/api/search");
const sources = Array.from({ length: 23 }, (_, index) => ({
  title: `Source ${index + 1}`,
  source_relative_path: `s${index + 1}.pdf`,
  source_id: `src_${index}`,
  document_id: `doc_${index}`,
}));
renderSources({ ready: true, sources, excluded_sources: [] });

window.location.hash = "#/sources?page=2&filter=source";
await applyRoute();
result.sources = {
  view: activeView(),
  page: state.sourcePages.sources,
  filter: document.getElementById("source-filter").value,
  range: document.getElementById("source-range").textContent,
  hash: window.location.hash,
  title: document.title,
};

window.location.hash = "#/sources?page=9";
await applyRoute();
result.clamped = window.location.hash;

window.location.hash = "#/search?q=labour&k=12&categories_any=marxism";
await applyRoute();
result.reloaded = {
  view: activeView(),
  query: document.getElementById("query").value,
  requests: searches().map((request) => JSON.parse(request.body)),
  title: document.title,
};

switchView("status");
recordRoute();
result.statusHash = window.location.hash;
window.location.hash = "#/search?q=labour&k=12&categories_any=marxism";
await applyRoute();
result.searchesAfterBack = searches().length;

document.getElementById("query").value = "automation";
await search({ preventDefault() {} });
result.submitted = window.location.hash;
result.searchesAfterSubmit = searches().length;

window.location.hash = "#/<script>";
await applyRoute();
result.unknown = window.location.hash;
result.history = window.history.entries;
})()
""",
    )

    assert result["sources"] == {
        "view": "sources",
        "page": 2,
        "filter": "source",
        "range": "Showing 6–10 of 23 sources",
        "hash": "#/sources?page=2&filter=source",
        "title": "sources · page 2 · Research RAG",
    }
    # A page past the end is drawn as the last page, and the fragment says so.
    assert result["clamped"] == "#/sources?page=5"
    # A reload of a search runs it once, with what the fragment carries.
    reloaded = result["reloaded"]
    assert reloaded["view"] == "search"
    assert reloaded["query"] == "labour"
    assert reloaded["requests"] == [
        {"query": "labour", "top_k": 12, "categories_any": ["marxism"]}
    ]
    assert reloaded["title"] == "search “labour” · Research RAG"
    # Leaving and coming back to the search on screen costs no second search.
    assert result["statusHash"] == "#/status"
    assert result["searchesAfterBack"] == 1
    # A submitted search is a history entry of its own.
    assert result["submitted"] == "#/search?q=automation&k=12&categories_any=marxism"
    assert result["searchesAfterSubmit"] == 2
    assert ["push", "#/status"] in result["history"]
    assert ["push", result["submitted"]] in result["history"]
    # A fragment that names no view is replaced by the view drawn.
    assert result["unknown"] == "#/search"


def test_a_load_draws_nothing_the_profile_has_not_allowed() -> None:
    """A refresh showed the page's raw markup before the profile arrived.

    The retrieval switches, the rerank box, Export, Import, and a Memory view
    this app does not serve were drawn for a moment and then removed, so every
    reload flashed a workspace that is not this one. Each capability-gated
    element now starts hidden, and the app's name waits for the profile.
    """

    with _panel_host() as client:
        page = client.get("/").text

    gated = re.findall(r"<[a-z]+\b[^>]*\bdata-capability=\"[^\"]+\"[^>]*>", page)
    assert len(gated) > 20
    for tag in gated:
        assert re.search(r"\shidden(?:\s|>)", tag), tag
    assert '<div id="application-name" class="brand-name"></div>' in page
    assert "UltraRAG MCP</div>" not in page


def test_the_page_names_the_exact_assets_it_was_written_with() -> None:
    """The page and its script come from one revision, whatever a cache holds.

    A script cached by a process that sent no revalidation header ran against a
    newer page and threw `Cannot set properties of null`. The page now names
    each asset by a digest of its bytes, so a different script is a different
    address and cannot be reused in its place.
    """

    with _panel_host() as client:
        page = client.get("/").text
        for filename in ("app.css", "app.js"):
            served = client.get(f"/assets/{filename}").content
            digest = hashlib.sha256(served).hexdigest()[:16]
            assert f'"/assets/{filename}?v={digest}"' in page
            stamped = client.get(f"/assets/{filename}?v={digest}")
            assert stamped.status_code == 200
            assert stamped.content == served
    assert '"/assets/app.js"' not in page


def test_a_long_list_starts_folded_and_keeps_the_readers_choice(
    tmp_path: Path,
) -> None:
    """A list of more than five starts closed behind its heading and count.

    Ten generations, twenty passage exclusions, or thirty-nine settings opened
    in full pushed everything under them off the screen. A short list still
    opens, and a fold the reader opened stays open across a refresh.
    """

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { generations: true } };
const fold = document.getElementById("generation-fold");
fold.id = "generation-fold";
const builds = (count) =>
  Array.from({ length: count }, (_, index) => ({ generation_id: `g${index}`, is_current: !index }));
renderGenerations(builds(3));
result.short = fold.open;
renderGenerations(builds(8));
result.long = fold.open;
result.count = document.getElementById("generation-count").textContent;
state.folds.set("generation-fold", true);
renderGenerations(builds(8));
result.chosen = fold.open;

const setting = (key) => ({ key, label: key, value: 1, default: 1, kind: "int", writable: true });
renderSettings({
  sections: [
    { key: "short", title: "Short", settings: [setting("a.one")] },
    { key: "long", title: "Long", settings: ["b", "c", "d", "e", "f", "g"].map(setting) },
  ],
});
result.settings = document
  .getElementById("settings-sections")
  .children.map((section) => [section.tagName, section.dataset.fold, section.open]);
""",
    )

    assert result["short"] is True
    assert result["long"] is False
    assert result["count"] == "8"
    assert result["chosen"] is True
    assert result["settings"] == [
        ["details", "settings-short", True],
        ["details", "settings-long", False],
    ]


def test_each_cards_actions_sit_behind_one_menu(tmp_path: Path) -> None:
    """Eight results carried forty links, five to a card, saying the same thing.

    Each card now has one menu holding its actions, and a result no longer
    prints its citation under the title, byline, and locator it repeats; the
    menu still copies it whole.
    """

    result = _drive_page(
        tmp_path,
        """
state.profile = {
  capabilities: {
    passage_context: true,
    chunk_exclusion: true,
    source_files: true,
    metadata: true,
    source_selection: true,
    source_inclusion: true,
  },
};
state.sources = [
  { document_id: "d1", source_relative_path: "a.pdf", source_id: "src_1", title: "A Title" },
];
const actionsIn = (node, inside = false, found = []) => {
  const inMenu = inside || node.className === "card-menu";
  if (node.dataset?.action) found.push([node.dataset.action, inMenu]);
  (node.children || []).forEach((child) => actionsIn(child, inMenu, found));
  return found;
};
const hit = resultCard({
  rank: 1,
  title: "A Title",
  chunk_id: "c1",
  document_id: "d1",
  text: "The passage.",
  citation: "Author, A Title (2000), p. 3",
  source_path: "a.pdf",
  locator: { page: 3 },
  doi: "10.1/x",
  authors: ["Author"],
  year: 2000,
});
result.hit = actionsIn(hit);
result.hitText = text(hit);
result.source = actionsIn(sourceCard(state.sources[0]));
""",
    )

    assert result["hit"] == [
        ["copy-passage", True],
        ["copy-citation", True],
        ["show-context", True],
        ["open-source", True],
        ["exclude-chunk", True],
    ]
    assert "Author, A Title (2000), p. 3" not in result["hitText"]
    assert "Author · 2000" in result["hitText"]
    assert result["source"] == [
        ["open-source", True],
        ["edit-metadata", True],
        ["only-source", True],
        ["exclude-from-search", True],
        ["exclude-source", True],
    ]


def test_a_setting_names_its_default_only_where_it_differs(tmp_path: Path) -> None:
    """Most rows said the same number twice, as the value and as the default."""

    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: {} };
const row = (value) =>
  text(settingRow({ key: "k", label: "K", value, default: 8, kind: "int", writable: true }, "0-k"));
result.same = row(8);
result.changed = row(12);
""",
    )

    assert "Value: 8" in result["same"]
    assert "Default:" not in result["same"]
    assert "Value: 12" in result["changed"]
    assert "Default: 8" in result["changed"]


def test_a_search_asks_for_any_number_of_results_up_to_the_servers_limit(
    tmp_path: Path,
) -> None:
    """Four fixed choices became a number the reader types, ten by default."""

    with _panel_host() as client:
        page = client.get("/").text
    assert (
        '<input id="top-k" name="top_k" type="number" min="1" max="50" step="1" '
        'value="10"' in page
    )

    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { capabilities: { documents: true } };
result.valid = ["1", "10", "37", "50", "0", "51", "4.5", "ten", ""].map(validTopK);
window.location.hash = "#/search?q=labour&k=37";
await applyRoute();
result.routed = document.getElementById("top-k").value;
window.location.hash = "#/search?q=labour&k=500";
await applyRoute();
result.clamped = document.getElementById("top-k").value;
const before = requests.length;
document.getElementById("top-k").value = "0";
await runSearch();
result.refusedRequests = requests.length - before;
document.getElementById("top-k").value = "23";
await runSearch();
result.sent = JSON.parse(requests[requests.length - 1].body).top_k;
})()
""",
    )

    assert result["valid"] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["routed"] == "37"
    assert result["clamped"] == "10"
    assert result["refusedRequests"] == 0
    assert result["sent"] == 23


def test_each_stat_is_a_card_with_a_scope_of_its_own(tmp_path: Path) -> None:
    """A card asks the server for its own scope, and links go where their names say."""

    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { capabilities: { stats: true, sources: true, passage_context: true } };
const stats = {
  searches: {
    search_count: 12,
    zero_result_count: 3,
    mean_result_count: 7.5,
    median_elapsed_ms: 840,
    p95_elapsed_ms: 2400,
    searches_by_day: [{ day: "2026-10-06", count: 9 }, { day: "2026-10-05", count: 3 }],
  },
  sources: [
    { source_id: "src_1", title: "Atlas of AI", top_five: 9, rank_one: 4, in_corpus: true },
    { source_id: "src_2", title: "Capital", top_five: 3, rank_one: 0, in_corpus: true },
  ],
  passages: [
    { chunk_id: "chk_1", source_id: "src_1", title: "Atlas of AI", top_five: 5, rank_one: 3,
      in_current_generation: true, in_corpus: true, locator: { page: 7 } },
  ],
  unreached_source_count: 1,
  unreached_sources: [{ source_id: "src_9", title: "Unread Book" }],
  corpus: {
    source_count: 3, passage_count: 400, pdf_page_count: 900,
    passages_per_source: { minimum: 20, median: 100, maximum: 280 },
    largest_by: "passages",
    largest_sources: [
      { source_id: "src_2", title: "Capital", passage_count: 280, text_bytes: 910000, physical_pages: 410, in_corpus: true },
      { source_id: "src_1", title: "Atlas of AI", passage_count: 90, text_bytes: 2400000, physical_pages: null, in_corpus: true },
    ],
    formats: [{ value: "pdf", count: 3 }], languages: [{ value: "en", count: 3 }],
    decades: [{ value: "2020s", count: 2 }, { value: "2010s", count: 1 }],
    categories: [{ value: "marxism", count: 2 }, { value: "labour", count: 1 }],
    authors: [{ value: "Crawford", count: 1 }, { value: "Marx", count: 1 }],
    missing_metadata: { authors: 0, year: 1, categories: 2 },
  },
  last_build: { created_at: "2026-10-05T08:36:50Z", seconds: 1545,
    phase_seconds: { embedding: 1150, extraction: 340 }, reused_vector_count: 0, created_vector_count: 26532 },
  generations: { count: 2, bytes: 1048576 },
};
const history = {
  recording: true, count: 1,
  searches: [{ searched_at: "2026-10-06T09:00:00Z", query: "labour and automation",
    filters: { categories_any: ["marxism"] }, requested_top_k: 12, result_count: 12,
    elapsed_ms: 900, caller: "agent" }],
};
const asked = [];
fetch = async (path, options = {}) => {
  asked.push(path);
  const body = path.startsWith("/api/stats/history/clear") ? { cleared: 1, message: "Forgot 1." }
    : path.startsWith("/api/stats/history") ? history : stats;
  return { ok: true, status: 200, json: async () => body };
};
await loadStats();
const body = (id) => state.statPanels.get(id).body;
const collect = (node, found = []) => {
  if ((node.className || "").includes("entity-link")) found.push([node.textContent, node.href]);
  (node.children || []).forEach((child) => collect(child, found));
  return found;
};
result.figures = text(body("searches"));
result.sources = text(body("sources"));
result.sourceLinks = collect(body("sources"));
result.passageLinks = collect(body("passages"));
result.historyLinks = collect(body("history"));
result.hasUnreachedCard = state.statPanels.has("unreached-list");
result.corpus = text(body("corpus"));
result.largest = text(body("largest"));
result.largestLinks = collect(body("largest"));
result.people = text(body("people"));
result.cardIds = [...state.statPanels.keys()];
result.usage = text(body("usage"));
result.svg = find(body("usage"), (node) => node.tagName === "svg").attributes;
result.passageColumns = body("passages").children[0].children[0].children[0].children[0].children.map((cell) => cell.textContent);
result.build = text(body("build"));
result.counts = ["sources", "passages", "history"].map((id) => state.statPanels.get(id).count.textContent);
result.firstRequests = [...new Set(asked)].sort();

asked.length = 0;
state.statScopes.sources = { days: 7, top: 5 };
await refreshCard(STAT_CARDS.find((card) => card.id === "sources"));
state.statScopes.history = { days: 1, top: 20 };
await refreshCard(STAT_CARDS.find((card) => card.id === "history"));
state.statScopes.largest = { days: null, top: 5, by: "size" };
await refreshCard(STAT_CARDS.find((card) => card.id === "largest"));
state.statScopes.corpus = { days: null, top: 5 };
await refreshCard(STAT_CARDS.find((card) => card.id === "corpus"));
result.scoped = [...asked];

asked.length = 0;
await clearHistory();
result.cleared = [...asked];
})()
""",
    )

    # The four headline figures share one card and one scope.
    for fact in (
        "7.5 passages on average",
        "25% of searches",
        "840 ms",
        "95th percentile 2.4 s",
        "of 3 searchable sources",
    ):
        assert fact in result["figures"]
    assert result["cardIds"] == [
        "searches",
        "sources",
        "passages",
        "history",
        "usage",
        "largest",
        "people",
        "corpus",
        "build",
    ]
    assert result["hasUnreachedCard"] is False
    assert "Atlas of AI" in result["sources"]
    assert ["Atlas of AI", "#/source?id=src_1"] in result["sourceLinks"]
    assert ["Page 7", "#/passage?id=chk_1"] in result["passageLinks"]
    assert ["Atlas of AI", "#/source?id=src_1"] in result["passageLinks"]
    # The source and the place in it are columns of their own.
    assert result["passageColumns"] == [
        "Source",
        "Passage",
        "Top five",
        "Rank one",
        "",
    ]
    # A kept search is run again from its own address, filters and size included.
    assert result["historyLinks"] == [
        [
            "labour and automation",
            "#/search?q=labour+and+automation&k=12&categories_any=marxism",
        ]
    ]
    assert "20 fewest · 100 median · 280 most" in result["corpus"]
    assert "1 sources" in result["corpus"] and "2 sources" in result["corpus"]
    assert "2020s" in result["corpus"]
    assert "Formats" not in result["corpus"] and "Languages" not in result["corpus"]
    assert "12 search invocations · per day · UTC" in result["usage"]
    assert result["svg"]["role"] == "img"
    assert result["svg"]["viewBox"] == "0 0 800 260"
    assert ["Capital", "#/source?id=src_2"] in result["largestLinks"]
    # Text size is shown for every source, and an EPUB has no pages to show.
    assert "280" in result["largest"] and "410" in result["largest"]
    assert "2.3 MB" in result["largest"] and "888.7 KB" in result["largest"]
    assert "marxism" in result["people"] and "Crawford" in result["people"]
    assert "26,532 embedded" in result["build"]
    assert result["counts"] == ["2", "1", "1"]
    # Cards that chose the same scope share one answer; each scope is its own ask.
    assert result["firstRequests"] == [
        "/api/stats/history?limit=10",
        "/api/stats?top=10",
        "/api/stats?top=5",
    ]
    # Ranking the largest sources is the one scope that reaches the server; a
    # distribution is shortened in the page, and shares the answer a card of the
    # same scope already asked for.
    assert result["scoped"] == [
        "/api/stats?days=7&top=5",
        "/api/stats/history?days=1&limit=20",
        "/api/stats?top=5&largest_by=size",
    ]
    assert result["cleared"][0] == "/api/stats/history/clear"
    assert result["cleared"][1].startswith("/api/stats/history?")


def test_usage_graph_counts_quiet_intervals_and_groups_long_histories(
    tmp_path: Path,
) -> None:
    result = _drive_page(
        tmp_path,
        """
result.daily = usageSeries([
  { day: "2026-10-06", count: 4 }, { day: "2026-10-04", count: 2 },
]);
result.scoped = usageSeries([{ day: "2026-10-06", count: 4 }], { days: 7 }, new Date("2026-10-06T12:00:00Z"));
result.usageScopes = STAT_CARDS.find((card) => card.id === "usage").timeScopes.map((scope) => scope.days);
result.weekly = usageSeries([
  { day: "2026-07-01", count: 2 }, { day: "2026-09-01", count: 3 },
]);
result.monthly = usageSeries([
  { day: "2024-12-31", count: 7 }, { day: "2026-10-06", count: 4 },
]);
result.empty = text(renderUsage({ searches: { searches_by_day: [] } }));
const chart = renderUsage({ searches: { searches_by_day: [
  { day: "2026-10-06", count: 4 }, { day: "2026-10-04", count: 2 },
] } });
const svg = find(chart, (node) => node.tagName === "svg");
result.bars = svg.children.filter((node) => node.tagName === "rect").map((node) => node.attributes);
result.data = text(find(chart, (node) => node.tagName === "details"));
result.description = text(find(chart, (node) => node.tagName === "desc"));
const one = renderUsage({ searches: { searches_by_day: [{ day: "2026-10-06", count: 1 }] } });
result.one = text(one);
""",
    )
    assert result["daily"] == {
        "interval": "day",
        "buckets": [
            {"day": "2026-10-04", "count": 2},
            {"day": "2026-10-05", "count": 0},
            {"day": "2026-10-06", "count": 4},
        ],
    }
    assert result["weekly"]["interval"] == "week"
    assert len(result["scoped"]["buckets"]) == 8
    assert result["scoped"]["buckets"][0] == {"day": "2026-09-29", "count": 0}
    assert result["scoped"]["buckets"][-1] == {"day": "2026-10-06", "count": 4}
    assert result["usageScopes"] == [None, 30, 7]
    assert result["weekly"]["buckets"][0] == {"day": "2026-06-29", "count": 2}
    assert result["weekly"]["buckets"][-1] == {"day": "2026-08-31", "count": 3}
    assert result["monthly"]["interval"] == "month"
    assert len(result["monthly"]["buckets"]) == 23
    assert sum(bucket["count"] for bucket in result["monthly"]["buckets"]) == 11
    assert "No search invocations" in result["empty"]
    assert [float(bar["height"]) for bar in result["bars"]] == [95, 0, 190]
    assert "2026-10-05" in result["data"] and "Invocations" in result["data"]
    assert "2026-10-05: 0 invocations" in result["description"]
    assert "0.5" not in result["one"]


def test_largest_sources_names_an_old_running_api_instead_of_empty_columns(
    tmp_path: Path,
) -> None:
    result = _drive_page(
        tmp_path,
        """
result.legacy = text(renderLargest({ largest_sources: [
  { title: "Capital", source_relative_path: "capital.pdf", passage_count: 123 },
] }));
result.noGeneration = text(renderLargest(null));
result.empty = text(renderLargest({ largest_by: "passages", largest_sources: [] }));
""",
    )
    assert "older statistics API" in result["legacy"]
    assert "Restart it from its starting terminal" in result["legacy"]
    assert "Refreshing the browser alone" in result["legacy"]
    assert "no generation" in result["noGeneration"]
    assert result["empty"] == "Nothing to count yet."


@pytest.mark.parametrize("old_fails", [False, True])
def test_stat_cards_ignore_obsolete_scope_responses(
    tmp_path: Path, old_fails: bool
) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { capabilities: { stats: true } };
buildStatsBoard();
const pending = [];
fetch = (path) => new Promise((resolve, reject) => pending.push({ path, resolve, reject }));
const card = STAT_CARDS.find((entry) => entry.id === "sources");
const panel = state.statPanels.get("sources");
const old = refreshCard(card);
state.statScopes.sources = { days: 7, top: 5 };
const current = refreshCard(card);
result.loading = text(panel.body);
result.busy = panel.panel.attributes["aria-busy"];
pending[1].resolve({ ok: true, json: async () => ({ sources: [
  { source_id: "new", title: "Current scope", top_five: 1, rank_one: 1 },
] }) });
await current;
OLD_RESPONSE
await old;
result.body = text(panel.body);
result.count = panel.count.textContent;
result.done = panel.panel.attributes["aria-busy"];
result.requests = pending.map((entry) => entry.path);
})()
""".replace(
            "OLD_RESPONSE",
            'pending[0].reject(new Error("Old error"));'
            if old_fails
            else """pending[0].resolve({ ok: true, json: async () => ({ sources: [
  { source_id: "old", title: "Obsolete scope", top_five: 99, rank_one: 99 },
] }) });""",
        ),
    )
    assert result["loading"] == "Loading…"
    assert result["busy"] == "true" and result["done"] == "false"
    assert "Current scope" in result["body"]
    assert "Obsolete scope" not in result["body"] and "Old error" not in result["body"]
    assert result["count"] == "1"
    assert result["requests"] == ["/api/stats?top=10", "/api/stats?days=7&top=5"]


def test_a_source_opens_in_the_desktop_viewer_and_falls_back_to_the_browser(
    tmp_path: Path,
) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { capabilities: { source_files: true } };
const opened = [];
const toasts = [];
window.open = (url) => opened.push(url);
const posted = [];
fetch = async (path, options = {}) => {
  posted.push([path, options.method || "GET", options.body || null]);
  if (result.headless) {
    return { ok: false, status: 501, json: async () => ({ error: "no desktop" }) };
  }
  return { ok: true, status: 200, json: async () => ({ opened: true, viewer: "xdg-open", filename: "a b.pdf" }) };
};
await openSource("books/a b.pdf");
result.desktop = { posted: [...posted], opened: [...opened], toast: text(document.getElementById("toast-region")) };
result.headless = true;
posted.length = 0;
await openSource("books/a b.pdf");
result.browser = { posted: [...posted], opened: [...opened] };
})()
""",
    )

    assert result["desktop"]["posted"] == [
        ["/api/open-source", "POST", '{"source_path":"books/a b.pdf"}']
    ]
    # Nothing is downloaded or shown in a tab while the desktop took the file.
    assert result["desktop"]["opened"] == []
    assert result["browser"]["opened"] == ["/api/source-file?path=books%2Fa%20b.pdf"]


def test_a_name_links_to_its_source_and_its_place_to_the_passage(
    tmp_path: Path,
) -> None:
    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: { sources: true, passage_context: true, source_files: true } };
state.sources = [];
const collect = (node, found = []) => {
  if ((node.className || "").includes("entity-link")) found.push([node.className.trim(), node.textContent, node.href]);
  (node.children || []).forEach((child) => collect(child, found));
  return found;
};
result.hit = collect(resultCard({
  rank: 1, title: "Atlas of AI", source_id: "src_1", chunk_id: "chk_1",
  document_id: "d1", locator: { page: 3 }, text: "The passage.", authors: ["Crawford"],
}));
result.source = collect(sourceCard({
  title: "Atlas of AI", source_id: "src_1", document_id: "d1",
  source_relative_path: "a.pdf", format: "pdf",
}));
state.profile = { capabilities: {} };
result.unlinked = collect(resultCard({
  rank: 1, title: "Atlas of AI", source_id: "src_1", chunk_id: "chk_1",
  locator: { page: 3 }, text: "The passage.",
}));
""",
    )

    assert result["hit"] == [
        ["entity-link", "Atlas of AI", "#/source?id=src_1"],
        ["entity-link locator-badge", "Page 3", "#/passage?id=chk_1"],
    ]
    assert result["source"] == [["entity-link", "Atlas of AI", "#/source?id=src_1"]]
    # A host that serves neither view draws plain text, not a link to nowhere.
    assert result["unlinked"] == []


def test_the_source_and_passage_views_are_pages_of_their_own(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
const tick = async () => { for (let index = 0; index < 40; index += 1) await Promise.resolve(); };
state.profile = { application_name: "Research RAG", capabilities: { sources: true, passage_context: true } };
const asked = [];
fetch = async (path) => {
  asked.push(path);
  if (path.startsWith("/api/source-chunks")) {
    return { ok: true, status: 200, json: async () => ({
      page: 2, pages: 5,
      source: { source_id: "src_1", title: "Atlas of AI", authors: ["Crawford"], year: 2021,
        source_relative_path: "a.pdf", format: "pdf", passage_count: 93, top_five: 4, rank_one: 1,
        physical_pages: 40, withheld_units: 2, unclean_character_rate: 0.004 },
      chunks: [
        { chunk_id: "chk_21", ordinal: 21, locator: { page: 9 }, text: "First words", truncated: true,
          characters: 900, embedding_token_count: 210, dense_truncated: false, content_kind: "prose",
          quality_flags: [], excluded: false, top_five: 3, rank_one: 1 },
        { chunk_id: "chk_22", ordinal: 22, locator: { page: 9 }, text: "More words", truncated: false,
          characters: 400, content_kind: "prose", quality_flags: ["very_short"], excluded: true,
          top_five: 0, rank_one: 0 },
      ],
    }) };
  }
  return { ok: true, status: 200, json: async () => ({ context: [] }) };
};
window.location.hash = "#/source?id=src_1&page=2";
await applyRoute();
await tick();
result.source = {
  view: state.view,
  asked: asked.filter((path) => path.startsWith("/api/source-chunks")),
  hash: window.location.hash,
  title: document.getElementById("source-view-title").textContent,
  documentTitle: document.title,
  chunks: text(document.getElementById("source-chunks")),
  cards: text(document.getElementById("source-stat-cards")),
  pager: document.getElementById("source-chunk-pager").children.map((item) => item.textContent),
};
goToSourcePage("chunks", 3);
await tick();
result.paged = {
  asked: asked.filter((path) => path.startsWith("/api/source-chunks")).pop(),
  hash: window.location.hash,
};

window.location.hash = "#/passage?id=chk_21";
await applyRoute();
await tick();
result.passage = {
  view: state.view,
  hash: window.location.hash,
  asked: asked.filter((path) => path.startsWith("/api/passages")),
  dialog: document.getElementById("context-dialog").opened,
};

navigateTo("#/passage?id=chk_22");
result.pushed = state.passagePushed;
window.location.hash = "#/source";
await applyRoute();
result.withoutId = state.view;
})()
""",
    )

    source = result["source"]
    assert source["view"] == "source"
    assert source["asked"] == ["/api/source-chunks?source_id=src_1&page=2&page_size=20"]
    assert source["hash"] == "#/source?id=src_1&page=2"
    assert source["title"] == "Atlas of AI"
    assert source["documentTitle"] == "Atlas of AI · page 2 · Research RAG"
    assert "#21" in source["chunks"] and "Page 9" in source["chunks"]
    assert "First words…" in source["chunks"] and "More words…" not in source["chunks"]
    assert "Top five 3 · rank one 1" in source["chunks"]
    assert "Not returned by any search" in source["chunks"]
    assert "Excluded from search" in source["chunks"]
    assert "210 tokens" in source["chunks"] and "very short" in source["chunks"]
    assert "93" in source["cards"]
    assert source["pager"] == ["Previous", "1", "2", "3", "…", "5", "Next"]
    assert result["paged"] == {
        "asked": "/api/source-chunks?source_id=src_1&page=3&page_size=20",
        "hash": "#/source?id=src_1&page=3",
    }
    assert result["passage"] == {
        "view": "passage",
        "hash": "#/passage?id=chk_21",
        "asked": ["/api/passages/chk_21?context_chunks=1"],
        "dialog": True,
    }
    assert result["pushed"] is True
    # A source address with no source in it is not a view of anything.
    assert result["withoutId"] == "sources"
