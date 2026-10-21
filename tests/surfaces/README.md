# Surface tests

What each surface sends and must never send.

- `test_ui.py`: the UI app and adapter against a recording service. Search is always hybrid and reranked, an unserved argument is dropped, an unsafe write or a source path outside the source root is refused, a service failure becomes a safe message, and the generation and clients panels follow the same write rules.
- `test_bridge.py`: the stdio transport. A registered name resolves to its project, a machine with no such project answers `status` and nothing else, a bridge started before the app holds every tool and works once the app is up, a read dropped mid-call is repeated and a write is not, a project initialised later is found, and a path in a generated `mcp` entry is refused.
- `test_mcp_tools.py`: one advertised-contract handshake checks schemas, instructions, response-size budgets, and mutation annotations. Operation tests check follow-up IDs and bounded app-state projections through a real MCP client.
- `test_workspace_app.py`: `surfaces/workspace/app.py` against a fake adapter. Only the local panels wait for the profile, the quotation rule is in the page, an unactionable write or removal is refused, and a LAN viewer may synthesise but never reads or sets the DeepSeek key.
- `test_workspace_ui.py`: page wiring and executable browser-script behavior.
  - Actual navigation targets, keyboard focus and Escape, the cleaned-text notice, the server's status sentence, and verbatim client-entry copying.
  - WCAG AA contrast of every text pair, and the light and dark schemes declaring the same tokens.
  - Node-driven page-script tests: the header's build offer per corpus state, the rebuild that alone sends `force_recompute`, health conditions and their copied remedies, filter lists drawn only when they narrow a search, source pagination, navigation confined to the serving project, the generations table, and route restoration.
  - A synthesis's citations link to their passages, its gaps are marked, and its provenance and reasoning start folded; the key field sends the key once and empties itself.
  - Update outcomes, declined-version tracking, safe release links, literal release-note rendering, and request-free decline/copy actions share the page harness.
- `test_workspace_stats_browser.py`: statistics cards, history width, usage graphs, and horizontal overflow. Layout and overflow run at three viewport widths; the data the page drew runs at one, because it does not change with width. It skips when Chrome is unavailable and uses fixture data without starting an app.

- The PDF and EPUB writers come from `tests/conftest.py`. `test_bridge.py` uses the `project` fixture there and is marked `anyio`.
- No gateway or dense backend starts. The UI and workspace tests use a fake adapter, and `test_bridge.py` proxies to an in-process app it switches up and down.
- A workspace change is tested here, because the workspace ships with this app.
