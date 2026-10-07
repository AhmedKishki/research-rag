"""The stats board laid out in a real browser at three window widths.

The Node page harness runs the shipped script against a stub document, so a value
it reads back is a value that stub kept. This file instead builds a page from the
shipped `index.html`, `app.css`, and `app.js`, serves it with a fake `fetch`, and
lays it out in headless Chrome. It answers questions the stub cannot: whether the
cards stay inside the page, whether a table that is wider than its column turns
the whole page sideways, whether the usage card draws its SVG with legible labels,
and whether content a function built as a `DocumentFragment` reached the real DOM.

The Largest card carries a contract of its own. Its rows name each source, count
its passages, state its text size, and state its page count where the format has
one, and its By and Top controls must re-ask the server for the ranking and the
list size they name. Those controls are driven here by real `change` events, so a
card that draws nothing, drops a label, or asks the wrong address fails.

The page starts without `DOMContentLoaded` so `initialize` never runs and no app,
socket, or project state is touched. The board is drawn by calling `loadStats`
directly, with the profile set in this page alone.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

STATIC = Path(__file__).parents[2] / "src/research_rag/surfaces/workspace/static"
WIDTHS = (360, 768, 1440)

# The stats payload the fake server answers with. It is wide enough to draw every
# card, several table columns, a passage locator, and a nine-day usage series.
# `largest_sources` arrives in passage order; the fake server re-ranks it when a
# request names `largest_by=size`.
_STATS: dict[str, Any] = {
    "searches": {
        "search_count": 17,
        "zero_result_count": 3,
        "mean_result_count": 7.5,
        "median_elapsed_ms": 840,
        "p95_elapsed_ms": 2400,
        "searches_by_day": [
            {"day": "2026-10-06", "count": 9},
            {"day": "2026-10-05", "count": 3},
            {"day": "2026-10-02", "count": 4},
            {"day": "2026-09-28", "count": 1},
        ],
    },
    "sources": [
        {
            "source_id": "src_1",
            "title": "Atlas of AI",
            "top_five": 9,
            "rank_one": 4,
            "in_corpus": True,
        },
        {
            "source_id": "src_2",
            "title": "Capital",
            "top_five": 3,
            "rank_one": 0,
            "in_corpus": True,
        },
    ],
    "passages": [
        {
            "chunk_id": "chk_1",
            "source_id": "src_1",
            "title": "Atlas of AI",
            "top_five": 5,
            "rank_one": 3,
            "in_current_generation": True,
            "in_corpus": True,
            "locator": {"page": 7},
        }
    ],
    "unreached_source_count": 1,
    "unreached_sources": [{"source_id": "src_9", "title": "Unread Book"}],
    "corpus": {
        "source_count": 3,
        "passage_count": 400,
        "pdf_page_count": 900,
        "passages_per_source": {"minimum": 20, "median": 100, "maximum": 280},
        "largest_by": "passages",
        "largest_sources": [
            {
                "source_id": "src_2",
                "title": "Capital",
                "passage_count": 280,
                "text_bytes": 910000,
                "physical_pages": 410,
                "in_corpus": True,
            },
            {
                "source_id": "src_1",
                "title": "Atlas of AI",
                "passage_count": 90,
                "text_bytes": 2400000,
                "physical_pages": None,
                "in_corpus": True,
            },
            {
                "source_id": "src_3",
                "title": "Nunes",
                "passage_count": 70,
                "text_bytes": 1500000,
                "physical_pages": None,
                "in_corpus": True,
            },
        ],
        "formats": [{"value": "pdf", "count": 3}],
        "languages": [{"value": "en", "count": 3}],
        "decades": [{"value": "2020s", "count": 2}, {"value": "2010s", "count": 1}],
        "categories": [
            {"value": "marxism", "count": 2},
            {"value": "labour", "count": 1},
        ],
        "authors": [{"value": "Crawford", "count": 1}, {"value": "Marx", "count": 1}],
        "missing_metadata": {"authors": 0, "year": 1, "categories": 2},
    },
    "last_build": {
        "created_at": "2026-10-05T08:36:50Z",
        "seconds": 1545,
        "phase_seconds": {"embedding": 1150, "extraction": 340},
        "reused_vector_count": 0,
        "created_vector_count": 26532,
    },
    "generations": {"count": 2, "bytes": 1048576},
}
_HISTORY: dict[str, Any] = {
    "recording": True,
    "count": 1,
    "searches": [
        {
            "searched_at": "2026-10-06T09:00:00Z",
            "query": "labour and automation",
            "filters": {"categories_any": ["marxism"]},
            "requested_top_k": 12,
            "result_count": 12,
            "elapsed_ms": 900,
            "caller": "agent",
        }
    ],
}

# The page script without its startup. The board is drawn on demand below, so the
# real `initialize` never binds the header, dialogs, or routes.
_DOM_CONTENT_LOADED = 'document.addEventListener("DOMContentLoaded", initialize);'

# Runs after the shipped script and after the markup is complete. It sets the
# profile, answers `fetch` from the fixture data, shows the stats panel, awaits
# `loadStats`, drives the Largest card's controls with real `change` events, and
# writes what it measured into a JSON script element.
_HARNESS = r"""
(async () => {
  state.profile = { capabilities: { stats: true, sources: true, passage_context: true } };
  const STATS = __STATS__;
  const HISTORY = __HISTORY__;
  const requests = [];
  const passageRows = STATS.corpus.largest_sources;
  const sizeRows = [...passageRows].sort(
    (left, right) => (right.text_bytes || 0) - (left.text_bytes || 0),
  );
  // The fake server ranks by what the request names, so a card that asks the
  // wrong scope draws the wrong order rather than a stale one.
  const statsFor = (query) => {
    const by = query.get("largest_by") === "size" ? "size" : "passages";
    const top = Number(query.get("top") || 10);
    const ordered = by === "size" ? sizeRows : passageRows;
    return {
      ...STATS,
      corpus: { ...STATS.corpus, largest_by: by, largest_sources: ordered.slice(0, top) },
    };
  };
  window.fetch = async (path) => {
    requests.push(path);
    if (path.startsWith("/api/stats/history/clear")) {
      return { ok: true, status: 200, json: async () => ({ cleared: 1, message: "Forgot 1." }) };
    }
    if (path.startsWith("/api/stats/history")) {
      return { ok: true, status: 200, json: async () => HISTORY };
    }
    const question = path.includes("?") ? path.slice(path.indexOf("?") + 1) : "";
    return { ok: true, status: 200, json: async () => statsFor(new URLSearchParams(question)) };
  };
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== "stats";
    panel.classList.toggle("is-active", panel.dataset.panel === "stats");
  });
  let error = null;
  try {
    await loadStats();
  } catch (caught) {
    error = String((caught && caught.message) || caught);
  }
  const settle = async () => {
    for (let tick = 0; tick < 5; tick += 1) await new Promise((resolve) => setTimeout(resolve, 20));
  };
  await settle();

  const root = document.documentElement;
  const viewport = root.clientWidth;
  const measure = (element) => element
    ? {
        width: Math.round(element.getBoundingClientRect().width),
        scrollWidth: element.scrollWidth,
        clientWidth: element.clientWidth,
      }
    : null;
  const panels = [...document.querySelectorAll(".stat-panel")];
  const card = (id) => document.querySelector('[data-card="' + id + '"]');
  const text = (id) => card(id) ? card(id).textContent.replace(/\s+/g, " ").trim() : "";
  const links = (id) => card(id)
    ? [...card(id).querySelectorAll("a")].map((anchor) => [anchor.textContent.trim(), anchor.getAttribute("href")])
    : [];
  const snapshot = (id) => {
    const panel = card(id);
    const count = panel.querySelector(".count-badge");
    return {
      text: text(id),
      headers: [...panel.querySelectorAll("thead th")].map((cell) => cell.textContent.trim()),
      rows: [...panel.querySelectorAll("tbody tr")].map((row) =>
        [...row.children].map((cell) => cell.textContent.trim()),
      ),
      titles: [...panel.querySelectorAll(".stat-title")].map((node) => node.textContent.trim()),
      links: [...panel.querySelectorAll("a")].map((anchor) => [anchor.textContent.trim(), anchor.getAttribute("href")]),
      count: count ? count.textContent : "",
    };
  };
  const cardOverflow = panels
    .filter((panel) => panel.scrollWidth > panel.clientWidth + 1)
    .map((panel) => ({
      card: panel.dataset.card,
      scrollWidth: panel.scrollWidth,
      clientWidth: panel.clientWidth,
    }));
  root.scrollLeft = 100000;
  const horizontalScroll = root.scrollLeft;
  root.scrollLeft = 0;
  const svg = card("usage") && card("usage").querySelector("svg");
  const probe = {
    error,
    viewport,
    documentScrollWidth: root.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    horizontalScroll,
    cardIds: panels.map((panel) => panel.dataset.card),
    cardOverflow,
    wide: Object.fromEntries(panels.map((panel) => [
      panel.dataset.card,
      panel.classList.contains("stat-panel-wide"),
    ])),
    panels: Object.fromEntries(panels.map((panel) => [panel.dataset.card, measure(panel)])),
    lists: measure(document.querySelector(".stats-lists")),
    svg: svg
      ? {
          width: Math.round(svg.getBoundingClientRect().width),
          height: Math.round(svg.getBoundingClientRect().height),
          viewBox: svg.getAttribute("viewBox"),
          role: svg.getAttribute("role"),
          bars: svg.querySelectorAll("rect.usage-bar").length,
          labelFont: getComputedStyle(svg.querySelector("text")).fontFamily,
          labelStroke: getComputedStyle(svg.querySelector("text")).stroke,
          labelWeight: getComputedStyle(svg.querySelector("text")).fontWeight,
        }
      : null,
    historyText: text("history"),
    usageText: text("usage"),
    historyLinks: links("history"),
    sourceLinks: links("sources"),
    passageLinks: links("passages"),
    sourceTitles: card("sources")
      ? [...card("sources").querySelectorAll(".stat-title")].map((node) => node.textContent.trim())
      : [],
    tables: document.querySelectorAll(".stats-board table").length,
  };

  // `initialize` is removed, so the board's own change listener is bound here
  // before the real events are dispatched.
  probe.largestInitial = snapshot("largest");
  const board = document.getElementById("stats-board");
  board.addEventListener("change", changeScope);
  const bySelect = card("largest").querySelector('.stat-scope-select[data-scope="by"]');
  const topSelect = card("largest").querySelector('.stat-scope-select[data-scope="top"]');
  probe.selects = {
    by: bySelect ? [...bySelect.options].map((option) => option.value) : [],
    top: topSelect ? [...topSelect.options].map((option) => option.value) : [],
    byInitial: bySelect ? bySelect.value : "",
    topInitial: topSelect ? topSelect.value : "",
  };

  bySelect.value = "size";
  bySelect.dispatchEvent(new Event("change", { bubbles: true }));
  await settle();
  probe.largestBySize = snapshot("largest");
  probe.requestsAfterBy = [...requests];

  topSelect.value = "10";
  topSelect.dispatchEvent(new Event("change", { bubbles: true }));
  await settle();
  probe.largestTopTen = snapshot("largest");
  probe.requestsAfterTop = [...requests];

  bySelect.value = "passages";
  bySelect.dispatchEvent(new Event("change", { bubbles: true }));
  await settle();
  probe.largestBack = snapshot("largest");
  probe.requestsAfterBack = [...requests];
  probe.requests = requests;

  const sink = document.createElement("script");
  sink.type = "application/json";
  sink.id = "browser-probe";
  sink.textContent = JSON.stringify(probe);
  document.body.appendChild(sink);
})();
"""

# An outer page that lays the built page out at an exact width inside an iframe.
# Chrome refuses a window narrower than 500px, so the iframe is how a 360px
# viewport is reached. It polls the inner document and copies its probe out.
_RUNNER = """<!doctype html>
<html><head><meta charset="utf-8"></head><body>
<iframe id="frame" src="page.html" style="border:0" width="__WIDTH__" height="900"></iframe>
<div id="probe-out">pending</div>
<script>
(async function () {
  const frame = document.getElementById("frame");
  for (let attempt = 0; attempt < 400; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 20));
    try {
      const inner = frame.contentDocument;
      const probe = inner && inner.getElementById("browser-probe");
      if (probe) {
        document.getElementById("probe-out").textContent = probe.textContent;
        return;
      }
    } catch (error) {
      document.getElementById("probe-out").textContent = JSON.stringify({ error: "iframe-unreadable" });
      return;
    }
  }
  document.getElementById("probe-out").textContent = JSON.stringify({ error: "probe-timeout" });
})();
</script></body></html>"""


def _chrome_binary() -> str | None:
    """The Chrome binary, or `None` when the machine has none."""

    override = os.environ.get("RESEARCH_RAG_CHROME")
    if override:
        return override if os.access(override, os.X_OK) else None
    return shutil.which("google-chrome") or shutil.which("chromium")


def _build_page(directory: Path) -> Path:
    """Write the shipped page with its assets inlined and its startup removed."""

    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    link = '<link rel="stylesheet" href="/assets/app.css">'
    source = '<script src="/assets/app.js" defer></script>'
    assert link in markup, "the shipped page no longer links /assets/app.css"
    assert source in markup, "the shipped page no longer defers /assets/app.js"
    assert _DOM_CONTENT_LOADED in script, (
        "the page script no longer starts on DOMContentLoaded"
    )

    script = script.replace(_DOM_CONTENT_LOADED, "")
    markup = markup.replace(link, f"<style>\n{stylesheet}\n</style>")
    markup = markup.replace(source, f"<script>\n{script}\n</script>")
    harness = _HARNESS.replace("__STATS__", json.dumps(_STATS)).replace(
        "__HISTORY__", json.dumps(_HISTORY)
    )
    markup = markup.replace("</body>", f"<script>{harness}</script>\n</body>")

    page = directory / "page.html"
    page.write_text(markup, encoding="utf-8")
    return page


def _run_probe(directory: Path, width: int) -> dict[str, Any]:
    """Lay the built page out at one width and return what the page measured."""

    binary = _chrome_binary()
    if binary is None:  # pragma: no cover - guarded by the fixture
        pytest.skip(
            "google-chrome is not installed, so the stats page cannot be laid out here."
        )

    runner = directory / f"runner-{width}.html"
    runner.write_text(_RUNNER.replace("__WIDTH__", str(width)), encoding="utf-8")
    completed = subprocess.run(
        [
            binary,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-first-run",
            f"--user-data-dir={directory / f'chrome-profile-{width}'}",
            "--allow-file-access-from-files",
            "--window-size=1500,1400",
            "--virtual-time-budget=20000",
            "--dump-dom",
            str(runner),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"google-chrome failed: {completed.stderr.strip()[:600]}")
    match = re.search(r'<div id="probe-out">(.*?)</div>', completed.stdout, re.DOTALL)
    if match is None:
        pytest.fail(
            "the stats page produced no probe; the built page did not finish drawing."
        )
    try:
        return json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as error:
        pytest.fail(f"the stats probe was not JSON: {error}")


@pytest.fixture(scope="module")
def stats_probes(tmp_path_factory: pytest.TempPathFactory) -> dict[int, dict[str, Any]]:
    """One laid-out stats page per width, built and measured once."""

    if _chrome_binary() is None:
        pytest.skip(
            "google-chrome is not installed, so the stats page cannot be laid out here."
        )
    directory = tmp_path_factory.mktemp("stats-browser")
    _build_page(directory)
    return {width: _run_probe(directory, width) for width in WIDTHS}


@pytest.mark.parametrize("width", WIDTHS)
def test_the_stats_page_does_not_scroll_sideways(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """A window below the widest table must scroll the table, not the page."""

    probe = stats_probes[width]
    assert probe["error"] is None, probe["error"]
    assert probe["horizontalScroll"] == 0, (
        f"the stats page scrolls sideways at {width}px: "
        f"document scrollWidth {probe['documentScrollWidth']} "
        f"over viewport {probe['viewport']}"
    )
    assert probe["documentScrollWidth"] <= probe["viewport"] + 1
    assert probe["bodyScrollWidth"] <= probe["viewport"] + 1


@pytest.mark.parametrize("width", WIDTHS)
def test_no_stats_card_overflows_its_own_box(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    probe = stats_probes[width]
    assert probe["error"] is None, probe["error"]
    assert probe["cardOverflow"] == []


@pytest.mark.parametrize("width", WIDTHS)
def test_history_and_usage_take_the_full_board_width(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """History and usage each hold a table or a chart wider than half a row."""

    probe = stats_probes[width]
    board = probe["lists"]
    assert board is not None and board["width"] > 0
    for card in ("history", "usage"):
        assert probe["wide"][card] is True
        panel = probe["panels"][card]
        assert panel is not None
        assert panel["width"] >= board["width"] - 1, (
            f"{card} is {panel['width']}px in a {board['width']}px board at {width}px"
        )


@pytest.mark.parametrize("width", WIDTHS)
def test_the_usage_card_draws_an_svg_chart(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    probe = stats_probes[width]
    svg = probe["svg"]
    assert svg is not None
    assert svg["viewBox"] == "0 0 800 260"
    assert svg["role"] == "img"
    assert svg["bars"] == 9
    assert svg["width"] > 0 and svg["height"] > 0
    assert svg["labelStroke"] == "none"
    assert svg["labelWeight"] == "400"
    assert "Inter" in svg["labelFont"] and "sans-serif" in svg["labelFont"]


@pytest.mark.parametrize("width", WIDTHS)
def test_the_board_renders_real_content(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """Cells a `DocumentFragment` built must reach the real DOM as elements."""

    probe = stats_probes[width]
    assert probe["error"] is None, probe["error"]
    assert probe["cardIds"] == [
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
    assert "labour and automation" in probe["historyText"]
    assert "search invocations" in probe["usageText"]
    # sourceCell and passageSourceCell return a DocumentFragment; its anchor must
    # be a real element in the table, not lost on append.
    assert ["Atlas of AI", "#/source?id=src_1"] in probe["sourceLinks"]
    assert "Atlas of AI" in probe["sourceTitles"]
    assert ["Atlas of AI", "#/source?id=src_1"] in probe["passageLinks"]
    assert ["Page 7", "#/passage?id=chk_1"] in probe["passageLinks"]
    assert probe["tables"] >= 4


@pytest.mark.parametrize("width", WIDTHS)
def test_the_largest_card_names_its_sources_and_measures_them(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """Every row names its source and states its passages, text, and pages.

    The card once drew an empty text column and a dash for every page because the
    served script expected fields an older payload did not carry. The fixture
    carries them, so a regression that drops one shows here.
    """

    probe = stats_probes[width]
    assert probe["error"] is None, probe["error"]
    largest = probe["largestInitial"]
    assert largest["headers"][:4] == ["Source", "Passages", "Text", "Pages"]
    assert largest["titles"] == ["Capital", "Atlas of AI", "Nunes"]
    assert largest["rows"] == [
        ["Capital", "280", "888.7 KB", "410", ""],
        ["Atlas of AI", "90", "2.3 MB", "—", ""],
        ["Nunes", "70", "1.4 MB", "—", ""],
    ]
    assert ["Capital", "#/source?id=src_2"] in largest["links"]
    assert ["Atlas of AI", "#/source?id=src_1"] in largest["links"]
    assert largest["count"] == "3"


@pytest.mark.parametrize("width", WIDTHS)
def test_the_by_control_asks_for_the_size_ranking_and_swaps_the_rows(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """Changing By to text size re-asks the server and reorders the rows."""

    probe = stats_probes[width]
    assert probe["selects"]["by"] == ["passages", "size"]
    assert probe["selects"]["byInitial"] == "passages"
    assert probe["requestsAfterBy"][-1] == "/api/stats?top=5&largest_by=size"
    swapped = probe["largestBySize"]
    assert swapped["titles"] == ["Atlas of AI", "Nunes", "Capital"]
    assert swapped["rows"][0][:4] == ["Atlas of AI", "90", "2.3 MB", "—"]
    assert swapped["rows"][2][:4] == ["Capital", "280", "888.7 KB", "410"]
    # The labels survive the swap.
    assert swapped["headers"][:4] == ["Source", "Passages", "Text", "Pages"]
    assert swapped["count"] == "3"


@pytest.mark.parametrize("width", WIDTHS)
def test_the_list_size_control_asks_for_its_size_and_keeps_the_labels(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    """Changing Top re-asks the server for that many rows, with labels intact."""

    probe = stats_probes[width]
    assert probe["selects"]["top"] == ["5", "10", "20", "50"]
    assert probe["selects"]["topInitial"] == "5"
    assert probe["requestsAfterTop"][-1] == "/api/stats?top=10&largest_by=size"
    resized = probe["largestTopTen"]
    assert resized["titles"] == ["Atlas of AI", "Nunes", "Capital"]
    assert resized["headers"][:4] == ["Source", "Passages", "Text", "Pages"]
    assert "2.3 MB" in resized["text"]
    assert "888.7 KB" in resized["text"]


@pytest.mark.parametrize("width", WIDTHS)
def test_returning_to_passages_restores_the_first_ranking(
    stats_probes: dict[int, dict[str, Any]], width: int
) -> None:
    probe = stats_probes[width]
    back = probe["largestBack"]
    assert back["titles"] == ["Capital", "Atlas of AI", "Nunes"]
    assert back["rows"][0][:4] == ["Capital", "280", "888.7 KB", "410"]
    # The passages answer was already drawn once, so the scope is served from it
    # rather than asked for again.
    assert probe["requestsAfterBack"] == probe["requestsAfterTop"]
