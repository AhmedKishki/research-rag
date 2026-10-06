# Surface tests

What each surface sends and must never send.

- `test_ui.py`: the UI app and adapter against a recording service. Search is always hybrid and reranked, an unserved argument is dropped, an unsafe write or a source path outside the source root is refused, a service failure becomes a safe message, and the generation and clients panels follow the same write rules.
- `test_bridge.py`: the stdio transport. A registered name resolves to its project, a machine with no such project answers `status` and nothing else, a bridge started before the app holds every tool and works once the app is up, a read dropped mid-call is repeated and a write is not, a project initialised later is found, and a path in a generated `mcp` entry is refused.
- `test_mcp_tools.py`: tool schemas, descriptions, instructions, response-size budgets, mutation annotations, follow-up IDs, and the bounded app-state projection, through a real MCP client.
- `test_workspace_app.py`: `surfaces/workspace/app.py` against a fake adapter. Every capability is declared in the markup and enforced at the route, the quotation rule is in the page, and an unactionable statement, write, or removal is refused or answered 501.
- `test_workspace_ui.py`: the shipped pages and stylesheet.
  - The clients view, SQL console, settings panel, chunk-exclusion controls, and the sidebar as the whole navigation.
  - The spacing and type scale, light and dark palettes, and WCAG AA contrast of every text pair.
  - Node-driven page-script tests: the header's build offer per corpus state, the rebuild that alone sends `force_recompute`, health conditions and their copied remedies, filter lists drawn only when they narrow a search, source lists in pages of ten, the project picker, passage wording, the generations table, and the routes Back, Forward, and reload return to.
- `test_workspace_stats_browser.py`: statistics cards, history width, usage graphs, and horizontal overflow at three viewport widths in headless Chrome. It skips when Chrome is unavailable and uses fixture data without starting an app.

- The PDF and EPUB writers come from `tests/conftest.py`. `test_bridge.py` uses the `project` fixture there and is marked `anyio`.
- No gateway or dense backend starts. The UI and workspace tests use a fake adapter, and `test_bridge.py` proxies to an in-process app it switches up and down.
- A workspace change is tested here, because the workspace ships with this app.
