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

import re
from collections.abc import Mapping
from typing import Any

from starlette.testclient import TestClient

from research_rag.surfaces.workspace import (
    SourceFile,
    UICapabilities,
    UIProfile,
    UIRequestError,
    create_ui_app,
)


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
                "attached": True,
                "requests": 7,
                "streams": 2,
                "idle_seconds": 0.2,
                "detached_reason": None,
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
        )
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
        )
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
    assert clients[0]["streams"] == 2


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
        )
    )


def test_the_status_view_carries_the_sql_console_panel() -> None:
    """The panel ships hidden and is revealed by the capability, not by a route."""

    with _sql_host(enabled=True) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert (
        'id="sql-console" class="partition-summary" data-capability="sql_console" hidden'
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
        )
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
    # It left the status section, and nothing is left behind there to show it
    # twice.
    assert page.index('id="settings-panel"') < page.index("system-summary")


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
        'id="chunk-exclusion-summary" class="partition-summary"'
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

    items = re.findall(r'data-view="([^"]+)" data-capability="([^"]+)"', page)
    assert items == [
        ("search", "documents"),
        ("sources", "sources"),
        ("config", "settings"),
        ("mcp", "clients"),
        ("memory", "memory"),
    ]
    # Every item has a panel, and every panel is reached by exactly one item.
    panels = re.findall(r'data-panel="([^"]+)" data-capability="([^"]+)"', page)
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
    assert page.count('<button class="nav-item') == 5
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
    assert page.count('<svg class="nav-item-icon"') == 5
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
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 15rem);" in css
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
        )
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
        ("config", "settings"),
        ("mcp", "clients"),
        ("memory", "memory"),
    ):
        assert f'data-panel="{panel}" data-capability="{capability}"' in page
    assert 'class="system-summary" data-capability="documents"' in page
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

    assert 'id="agent-entry-summary" class="partition-summary"' in page.text
    assert 'data-capability="agent_entry"' in page.text
    assert 'id="agent-entry" class="standing-document"' in page.text
    assert 'id="agent-entry-copy"' in page.text
    assert 'hasCapability("agent_entry")' in script.text
    assert "/api/agent-entry" in script.text
    # The text reaches the block and the clipboard unedited, and no generator is
    # built here: a second one would be a second place for a client's
    # configuration to be wrong.
    assert 'byId("agent-entry").textContent = state.agentEntry' in script.text
    assert 'copyText(state.agentEntry, "Client entry copied.")' in script.text
    # It is the stdio entry, so the block says which client it is for.
    assert "for a client that cannot open a socket" in page.text.lower()
