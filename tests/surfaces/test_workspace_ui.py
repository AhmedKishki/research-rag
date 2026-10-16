"""Workspace host routes, shipped markup, and executable page interactions.

The Node harness tests requests and state transitions, not browser layout. The
browser stats tests own real-DOM rendering and responsive overflow.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_rag.runtime.app import CLIENT_NAME_ENV
from research_rag.surfaces.workspace import (
    SourceFile,
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


def _host(adapter: Any) -> TestClient:
    return TestClient(
        create_ui_app(
            adapter=adapter,
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


def client_page() -> str:
    with _host(ClientControlAdapter()) as client:
        return client.get("/").text


def client_css() -> str:
    with _host(ClientControlAdapter()) as client:
        return client.get("/assets/app.css").text


def test_shipped_navigation_resolves_to_unique_panels() -> None:
    class Elements(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: list[dict[str, str | None]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            self.tags.append(dict(attrs))

    page = Elements()
    page.feed(client_page())
    identifiers = [tag["id"] for tag in page.tags if "id" in tag]
    assert len(identifiers) == len(set(identifiers))
    views = [tag["data-view"] for tag in page.tags if "data-view" in tag]
    panels = [tag["data-panel"] for tag in page.tags if "data-panel" in tag]
    assert views and len(views) == len(set(views))
    assert set(views) == set(panels) - {"source"}
    assert len(panels) == len(set(panels))
    assert sum(tag.get("class") == "semantic-text-label" for tag in page.tags) == 1
    for tag in page.tags:
        if "aria-controls" in tag:
            assert set((tag["aria-controls"] or "").split()) <= set(identifiers)


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
    # The page says how a client is named, with the bridge's own variable.
    page = client_page()
    assert f"Set {CLIENT_NAME_ENV} in the client entry" in page


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
            adapter=SettingsHost(),
        ),
        base_url=LOOPBACK_BASE_URL,
        client=LOOPBACK_TEST_CLIENT,
    )


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
    gate = script.split("function syncSettingsConfirmSubmit(")[1].split("\n}\n")[0]
    assert 'byId("settings-confirm-submit").disabled' in gate
    assert "pending.word" in gate


def test_the_workspace_carries_the_chunk_exclusion_controls() -> None:
    """Every hit and every context passage offers the exclusion, and the list restores."""

    with _panel_host() as client:
        page = client.get("/")
        script = client.get("/assets/app.js").text

    assert 'id="chunk-exclusion-summary" class="panel-section"' in page.text
    assert 'id="chunk-exclusion-list"' in page.text
    assert 'id="chunk-dialog"' in page.text
    assert 'id="chunk-error"' in page.text
    assert "chunkAction(hit)," in script
    assert "/api/chunk-exclusions" in script
    assert "/api/chunk-inclusion" in script
    # A chunk the server already excluded is offered a restore instead.
    assert "state.chunkExclusions.has(chunk.chunk_id)" in script
    assert "Not in the current generation" in script
    # The whole-source exclusion is a separate control and stays as it was.
    assert 'id="exclusion-dialog"' in page.text
    assert "/api/source-inclusion" in script


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
    """Both schemes name the same tokens, so dark mode stays legible.

    Contrast of every text pair is checked once in
    `test_every_pair_that_carries_text_meets_wcag_aa`; this checks that the dark
    scheme actually re-declares the surfaces and text colours rather than
    inheriting the light ones.
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
    # and an unwritable one stays disabled; the field kinds are covered once, in
    # `test_the_settings_panel_reads_the_shape_the_server_sends`.
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


def test_the_page_carries_no_private_project_or_source_names() -> None:
    """The placeholders in a shared page are written for any installation.

    A category or project name typed into a placeholder ships one reader's corpus
    to every other reader of the same package.
    """

    with _host(ClientControlAdapter()) as client:
        page = client.get("/").text

    for name in ("ai-and-fetishism", "fetishism", "Crawford", "Gidwani", "Atlas of AI"):
        assert name not in page


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
    assert "if (!excluded) {" in script
    assert "item.append(chunkExcludeButton(passage));" in script
    # A locator longer than its card wraps rather than being clipped, because a
    # clipped locator is one the reader cannot use.
    assert "flex: 0 1 auto;" in css
    assert "max-width: 100%;" in css
    assert "overflow-wrap: anywhere;" in css


def test_navigation_keyboard_focus_and_escape(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
loadWorkspace = async () => {};
initialize();
const statusItem = document.querySelectorAll(".sidebar-item").find((item) => item.dataset.view === "status");
statusItem.querySelector(".nav-item").dispatchEvent({ type: "click" });
result.focus = document.activeElement.dataset.panel;
result.tabIndex = document.activeElement.tabIndex;
result.current = document.querySelectorAll(".sidebar-item").filter((item) => item.querySelector(".nav-item").attributes["aria-current"] === "page").map((item) => item.dataset.view);
byId("nav-toggle").dispatchEvent({ type: "click" });
result.open = navIsOpen();
document.activeElement.closest = () => null;
document.dispatchEvent({ type: "keydown", key: "Escape" });
result.closed = !navIsOpen() && byId("nav-scrim").hidden;
result.expanded = byId("nav-toggle").attributes["aria-expanded"];
result.restored = document.activeElement === byId("nav-toggle");
byId("nav-toggle").dispatchEvent({ type: "click" });
byId("nav-scrim").dispatchEvent({ type: "click" });
result.scrimClosed = !navIsOpen();
let submitted = 0;
byId("search-form").requestSubmit = () => { submitted++; };
byId("query").dispatchEvent({ type: "keydown", key: "Enter", ctrlKey: true, preventDefault() {} });
byId("query").dispatchEvent({ type: "keydown", key: "Enter", metaKey: true, preventDefault() {} });
byId("query").dispatchEvent({ type: "keydown", key: "Enter", preventDefault() {} });
result.submitted = submitted;
""",
    )
    assert result == {
        "focus": "status",
        "tabIndex": -1,
        "current": ["status"],
        "open": True,
        "closed": True,
        "expanded": "false",
        "restored": True,
        "scrimClosed": True,
        "submitted": 2,
    }


def test_results_notice_and_server_status_message(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
state.profile = { capabilities: {} };
renderStatus({ ready: true, message: "Reviewed decisions apply now." });
result.status = { message: byId("status-message").textContent, hidden: byId("status-message").hidden };
renderResults({ result_count: 2, hits: [{ chunk_id: "a", title: "First source", text: "First" }, { chunk_id: "b", title: "Second source", text: "Second" }] });
result.notice = { text: byId("results-notice").textContent, hidden: byId("results-notice").hidden };
result.cards = byId("results").children.length;
result.cardNotices = byId("results").children.map(text).filter((value) => value.includes("Cleaned passage")).length;
clearResults();
result.cleared = byId("results-notice").hidden;
renderResults({ result_count: 0 });
result.empty = byId("results-notice").hidden;
""",
    )
    assert result["status"] == {
        "message": "Reviewed decisions apply now.",
        "hidden": False,
    }
    assert result["notice"] == {"text": "Cleaned passage", "hidden": False}
    assert result["cards"] == 2 and result["cardNotices"] == 0
    assert result["cleared"] and result["empty"]


def test_client_entry_copy_is_verbatim(tmp_path: Path) -> None:
    entry = '{\n  "mcp": {"reader": {"args": ["mcp", "--project-name", "essay"]}}\n}\n'
    result = _drive_page(
        tmp_path,
        """
(async () => {
loadWorkspace = async () => {};
initialize();
state.agentEntry = __ENTRY__;
byId("agent-entry-copy").dispatchEvent({ type: "click" });
await Promise.resolve();
result.clipboard = globalThis.clipboard;
result.requests = requests;
})();
""".replace("__ENTRY__", json.dumps(entry)),
    )
    assert result == {"clipboard": entry, "requests": []}


def test_declining_and_copying_an_update_send_no_request_at_all(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(async () => {
const payload = {
  installed_version: "1.1.0", update_available: true, offline: false,
  release: {
    version: "1.2.0", tag_name: "v1.2.0", body: "<img src=x onerror=alert(1)>",
    html_url: "https://github.com/example/project/releases/tag/v1.2.0",
  },
  apply_command: "research-rag update --apply", message: null,
};
fetch = async (path, options = {}) => {
  requests.push({ path, method: options.method || "GET" });
  return { ok: true, status: 200, json: async () => payload };
};
state.profile = { capabilities: { updates: false } };
await checkUpdates();
result.deniedRequests = requests.length;
state.profile.capabilities.updates = true;
await checkUpdates();
result.available = { badge: byId("update-nav-badge").hidden, open: byId("update-dialog").opened };
result.notes = [byId("update-notes").textContent, byId("update-dialog-notes").textContent];
declineUpdate();
result.declined = { version: state.updates.declinedVersion, open: byId("update-dialog").opened, prompt: shouldPromptUpdate(payload) };
result.newVersion = shouldPromptUpdate({ ...payload, release: { ...payload.release, version: "1.3.0" } });
result.missingVersion = shouldPromptUpdate({ ...payload, release: {} });
fillUpdateDialog(payload);
await copyText(byId("update-command").textContent, "copied");
result.command = byId("update-command").textContent;
result.clipboard = globalThis.clipboard;
result.beforeRecheck = requests.length;
await checkUpdates({ manual: true });
result.rechecked = { version: state.updates.declinedVersion, open: byId("update-dialog").opened };
result.requests = requests;
result.urls = [payload.release.html_url, "http://github.com/x", "https://evil.example/x", "https://github.com.evil.example/x", "javascript:alert(1)"]
  .map((html_url) => updateFacts({ ...payload, release: { ...payload.release, html_url } }, true).filter(([label]) => label === "Release page"));
result.outcomes = [
  { ...payload, update_available: false, offline: true, release: null, message: "offline reason" },
  { ...payload, update_available: false, release: null, message: "feed failed" },
  { ...payload, update_available: false },
].map((answer) => {
  renderUpdates(answer);
  return { outcome: updateOutcome(answer), message: text(byId("update-message")), detail: byId("update-detail").hidden, prompt: shouldPromptUpdate(answer), badge: byId("update-nav-badge").hidden };
});
})();
""",
    )
    assert result["deniedRequests"] == 0
    assert result["available"] == {"badge": False, "open": True}
    assert result["notes"] == ["<img src=x onerror=alert(1)>"] * 2
    assert result["declined"] == {"version": "1.2.0", "open": False, "prompt": False}
    assert result["newVersion"] is True
    assert result["missingVersion"] is False
    assert result["command"] == "research-rag update --apply"
    assert result["clipboard"] == "research-rag update --apply"
    assert result["beforeRecheck"] == 1
    assert result["rechecked"] == {"version": "", "open": True}
    assert result["requests"] == [{"path": "/api/updates", "method": "GET"}] * 2
    assert result["urls"][0] == [
        ["Release page", "https://github.com/example/project/releases/tag/v1.2.0"]
    ]
    assert result["urls"][1:] == [[], [], [], []]
    assert [row["outcome"] for row in result["outcomes"]] == [
        "offline",
        "unavailable",
        "none",
    ]
    for row, reason in zip(
        result["outcomes"][:2], ("offline reason", "feed failed"), strict=True
    ):
        assert reason in row["message"]
        assert row["detail"] and row["badge"] and not row["prompt"]
    assert "no newer release" in result["outcomes"][2]["message"]


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
  const listeners = new Map();
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
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatchEvent(event) { return listeners.get(event.type)?.(event); },
    removeEventListener() {},
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
    focus() { document.activeElement = this; },
    set innerHTML(_value) { throw new Error("Unexpected HTML insertion"); },
    insertAdjacentHTML() { throw new Error("Unexpected HTML insertion"); },
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
// The sidebar's views, so navigation and routes have items to mark.
const navItems = ["search", "sources", "status", "config", "mcp", "updates"].map((view) => {
  const item = element("li");
  item.dataset.view = view;
  return item;
});
navItems[0].classList.add("is-active");
const panels = navItems.map((item) => {
  const panel = element("section");
  panel.dataset.panel = item.dataset.view;
  return panel;
});
const documentListeners = new Map();
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
    if (selector === ".sidebar-item") return navItems;
    if (selector === ".view-panel") return panels;
    return [];
  },
  querySelector() { return null; },
  addEventListener(name, callback) { documentListeners.set(name, callback); },
  dispatchEvent(event) { return documentListeners.get(event.type)?.(event); },
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
  navigator: { clipboard: { writeText: async (value) => { context.clipboard = value; } } },
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
    addEventListener() {},
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
    assert result["samples"] == [None, None, None, "1–2 min"]
    assert result["stalled"] == "1–2 min"
    assert result["batch"] is None
    assert result["resets"] == [None] * 6
    assert result["invalidHidden"] and result["missingHidden"]
    assert result["fallback"] == {"hidden": False, "value": 2, "max": 10}
    assert result["summary"] == "extraction · Book.pdf · 8 / 20 sources · ETA 1–2 min"


def test_source_eta_stability_and_independent_stall(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(() => {
  const estimate = phaseEtaEstimator();
  const stalled = ingestionStallDetector();
  const status = { build_id: 'a', phase: 'chunking', source: 'First.pdf',
    overall_progress: { completed: 0, total: 100, unit: 'sources' },
    progress: { completed: 0, total: 10, unit: 'pages' } };
  estimate(status, 0); stalled(status, 0);
  const earlyStall = stalled(status, 60000);
  const earlySummary = liveIngestionSummary(status, estimate(status, 60000), earlyStall);
  status.overall_progress.completed = 3;
  const first = estimate(status, 180000);
  const resumed = stalled(status, 180000);
  status.source = 'Second.pdf'; status.progress.completed = 5;
  status.overall_progress.completed = 50;
  const held = estimate(status, 210000);
  stalled(status, 210000);
  const before = stalled(status, 269999);
  const after = stalled(status, 270000);
  const revised = estimate(status, 270000);
  const materialHeld = estimate(status, 330000);
  result = { earlyStall, earlySummary, first, resumed, held, before, after,
    revised, materialHeld };
})();
""",
    )
    assert result["earlyStall"] and "No progress for 60s" in result["earlySummary"]
    assert "ETA" not in result["earlySummary"]
    assert result["first"] == result["held"] == "65–130 min"
    assert not result["resumed"] and not result["before"] and result["after"]
    assert result["revised"] == result["materialHeld"] == "2–6 min"


def test_ingestion_stall_tracks_source_batches_not_only_finished_sources(
    tmp_path: Path,
) -> None:
    result = _drive_page(
        tmp_path,
        r"""
(() => {
  const stalled = ingestionStallDetector();
  const status = { build_id: 'a', phase: 'extraction', source: 'Book.pdf',
    overall_progress: { completed: 2, total: 10, unit: 'sources' },
    source_progress: { completed: 0, total: 20, unit: 'extraction_units' },
    progress: { completed: 10, total: 100, unit: 'extraction_units' } };
  stalled(status, 0);
  status.source_progress.completed = 1;
  const advancing = stalled(status, 59000);
  const notStalled = stalled(status, 60000);
  const stalledLater = stalled(status, 119000);
  result = { advancing, notStalled, stalledLater };
})();
""",
    )
    assert not result["advancing"] and not result["notStalled"]
    assert result["stalledLater"]


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
    assert result["stalled"] == 14
    assert result["resets"] == [None] * 6
    assert "ETA 15 sec" in result["summary"]
    assert "ETA" not in result["estimating"]
    assert "ETA 17 min" in result["backend"]
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
    assert result["rounded"] == ["5 sec", "55 sec", "1 min", "1 min", "1 min"]


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
    assert result["fallback"] == "embedding · 3 / 10 chunks"


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
  compatible: {
    ...base,
    generation_diagnostics: { policy_differences: ["layout_extraction", "semantic_cleaning", "retrieval_policy"] },
  },
  upgrade: {
    ...base,
    generation_upgrade_required: true,
    upgrade_reasons: ["embedding_model"],
    message: "BM25 remains available. Restore the recorded embedding model or explicitly build matching vectors.",
  },
  foreign: {
    ...base, ready: false, generation_id: "foreign", generation_upgrade_required: true,
    upgrade_reasons: ["project_identity"],
    message: "Select a generation belonging to this project; build one only if none is retained.",
  },
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
    assert plans["compatible"]["hidden"] is True
    assert plans["compatible"]["noticeHidden"] is True
    assert plans["foreign"]["hidden"] is True
    assert plans["foreign"]["noticeHidden"] is False
    assert "Select a generation" in plans["foreign"]["notice"]
    assert "Restore the recorded embedding model" in plans["upgrade"]["notice"]
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
state.status = { searchable_source_count: 106 };
const group = (id) => document.getElementById(id).closest(".pick-group");
renderInventory("project-chips", [{ project: "all", searchable_source_count: 106 }], "project", "project-filter");
result.everySource = group("project-chips").hidden;
renderInventory("partition-chips", [{ category: "some", searchable_source_count: 40 }], "category", "partition-filter");
result.someSources = group("partition-chips").hidden;
renderInventory("partition-chips", [], "category", "partition-filter");
result.none = group("partition-chips").hidden;
""",
    )

    assert result == {"everySource": True, "someSources": False, "none": True}


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
    assert '"action-button action-button-danger")' in script
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


def test_startup_refresh_and_navigation_stay_in_the_serving_workspace(
    tmp_path: Path,
) -> None:
    """Ordinary browser use never discovers or opens another project's workspace."""

    with _panel_host() as client:
        page = client.get("/").text
    assert 'id="project-selector"' not in page
    assert 'id="project-select"' not in page
    assert 'id="project-open"' not in page

    result = _drive_page(
        tmp_path,
        """
(async () => {
const opened = [];
window.open = (...args) => opened.push(args);
byId("connection-state").lastElementChild = node("span");
const payloads = {
  "/api/ui": { capabilities: { clients: true, agent_entry: true, updates: false } },
  "/api/status": { ready: true, hybrid_ready: true, project_name: "here", project_root: "/here" },
  "/api/sources": { ready: true, sources: [], excluded_sources: [] },
  "/api/clients": { clients: [] },
  "/api/agent-entry": { entry: "current project entry" },
  "/api/settings": { sections: [] },
  "/api/chunk-exclusions": { exclusions: [] },
};
fetch = async (path, options = {}) => {
  requests.push({ path, method: options.method || "GET" });
  return { ok: true, status: 200, json: async () => payloads[path] || {} };
};
const loads = [];
const originalLoad = loadWorkspace;
loadWorkspace = (...args) => {
  const pending = originalLoad(...args);
  loads.push(pending);
  return pending;
};
initialize();
await loads[0];
await byId("refresh-button").dispatchEvent({ type: "click" });
for (const view of ["sources", "status", "search"]) {
  window.location.hash = "#/" + view;
  await applyRoute();
}
result = { requests, opened, view: activeView(), project: state.status.project_name,
  entry: state.agentEntry, loads: loads.length, routes: window.history.entries };
})();
""",
    )

    assert result["loads"] == 2
    assert result["project"] == "here"
    assert result["entry"] == "current project entry"
    assert result["view"] == "search"
    assert result["opened"] == []
    paths = [request["path"] for request in result["requests"]]
    assert paths.count("/api/status") == 2
    assert paths.count("/api/sources") == 2
    assert "/api/projects" not in paths
    assert all(path.startswith("/api/") for path in paths)
    assert all(route.startswith("#/") for _, route in result["routes"])


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


def test_each_view_is_a_page_that_back_forward_and_reload_return_to(
    tmp_path: Path,
) -> None:
    """A reload lost the search, and Back left the workspace.

    The view and what it shows now live in the fragment: a submitted search and
    a page of sources each add a history entry, a reload draws what the
    fragment names, and returning to a search already on screen does not run
    it again.
    """

    filter_text = {
        "keywords": "labour,  automation",
        "categories_any": "marxism, political economy",
        "projects_any": "essay",
        "authors_any": "Marx,  Engels",
        "titles_any": "Capital",
        "languages_any": "en, de",
        "source_ids": "src_one, src_two",
        "exclude_source_ids": "src_other",
    }
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

const filterText = __FILTER_TEXT__;
window.location.hash = `#/search?${new URLSearchParams({
  q: "filtered", ...filterText, categories: "ignored legacy filter",
})}`;
await applyRoute();
result.filters = {
  text: {
    keywords: byId("keyword-filter").value,
    categories_any: byId("category-any-filter").value,
    projects_any: byId("project-any-filter").value,
    authors_any: byId("author-filter").value,
    titles_any: byId("title-filter").value,
    languages_any: byId("language-filter").value,
    source_ids: byId("include-source-filter").value,
    exclude_source_ids: byId("exclude-source-filter").value,
  },
  body: JSON.parse(searches().at(-1).body),
  route: Object.fromEntries(parseRoute(window.location.hash).params),
  count: byId("filter-count").textContent,
};
window.location.hash = "#/search?q=unfiltered";
await applyRoute();
result.unfiltered = JSON.parse(searches().at(-1).body);

window.location.hash = "#/<script>";
await applyRoute();
result.unknown = window.location.hash;
result.history = window.history.entries;
})()
""".replace("__FILTER_TEXT__", json.dumps(filter_text)),
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
    # Routes retain the text, while requests trim comma lists and omit empty ones.
    assert result["filters"] == {
        "text": filter_text,
        "body": {
            "query": "filtered",
            "top_k": 10,
            **{
                name: [part.strip() for part in text.split(",")]
                for name, text in filter_text.items()
            },
        },
        "route": {"q": "filtered", **filter_text},
        "count": "8",
    }
    assert result["unfiltered"] == {"query": "unfiltered", "top_k": 10}
    # A fragment that names no view is replaced by the view drawn.
    assert result["unknown"] == "#/search"


def test_a_load_draws_nothing_the_profile_has_not_allowed() -> None:
    """A LAN viewer never sees a local panel flash before the profile arrives.

    Each capability-gated element starts hidden, and only the profile shows it.
    """

    with _panel_host() as client:
        page = client.get("/").text

    gated = re.findall(r"<[a-z]+\b[^>]*\bdata-capability=\"[^\"]+\"[^>]*>", page)
    assert {re.search(r'data-capability="([^"]+)"', tag).group(1) for tag in gated} == {
        "clients",
        "agent_entry",
        "updates",
    }
    for tag in gated:
        assert re.search(r"\shidden(?:\s|>)", tag), tag
    assert '<div id="application-name" class="brand-name">Research RAG</div>' in page


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
    """Equal scopes share an answer; changing one card does not change another."""

    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { capabilities: { stats: true, sources: true, passage_context: true } };
const stats = {
  searches: { search_count: 1, searches_by_day: [] }, sources: [], passages: [],
  corpus: { source_count: 0, passage_count: 0, largest_sources: [] },
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
result.historyLinks = collect(body("history"));
result.firstRequests = [...asked].sort();

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

    # A kept search is run again from its own address, filters and size included.
    assert result["historyLinks"] == [
        [
            "labour and automation",
            "#/search?q=labour+and+automation&k=12&categories_any=marxism",
        ]
    ]
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


def test_remote_sources_open_on_the_visiting_device(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
state.profile = { source_open_mode: "browser", capabilities: { source_files: true } };
result.opened = [];
window.open = (url) => result.opened.push(url);
fetch = async () => { throw new Error("Desktop viewer must not be requested"); };
await openSource("books/a b.pdf");
})()
""",
    )
    assert result["opened"] == ["/api/source-file?path=books%2Fa%20b.pdf"]


def test_http_copy_offers_selectable_text_without_clipboard(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
navigator.clipboard = undefined;
document.body = { append(field) { field.remove = () => {}; } };
result.prompts = [];
window.prompt = (message, text) => result.prompts.push([message, text]);
await copyText("A passage to copy", "Copied");
})()
""",
    )
    assert result["prompts"] == [["Select and copy this text:", "A passage to copy"]]


def test_http_copy_can_use_a_user_gesture_fallback(tmp_path: Path) -> None:
    result = _drive_page(
        tmp_path,
        """
(async () => {
navigator.clipboard = undefined;
document.body = { append(field) {
  field.select = () => {};
  field.setSelectionRange = () => {};
  field.remove = () => { result.removed = true; };
} };
document.execCommand = (command) => { result.command = command; return true; };
window.prompt = () => { throw new Error("Copy succeeded; no prompt needed"); };
await copyText("A passage to copy", "Copied passage");
result.toast = text(document.getElementById("toast-region"));
})()
""",
    )
    assert result["command"] == "copy"
    assert result["removed"] is True
    assert "Copied passage" in result["toast"]


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
result.unlinked = collect(resultCard({
  rank: 1, title: "Atlas of AI", locator: { page: 3 }, text: "The passage.",
}));
""",
    )

    assert result["hit"] == [
        ["entity-link", "Atlas of AI", "#/source?id=src_1"],
        ["entity-link locator-badge", "Page 3", "#/passage?id=chk_1"],
    ]
    assert result["source"] == [["entity-link", "Atlas of AI", "#/source?id=src_1"]]
    # A hit with no source or passage id draws plain text, not a link to nowhere.
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
