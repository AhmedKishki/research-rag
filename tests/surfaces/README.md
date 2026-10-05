# The three front ends to one service

These tests hold what each surface sends and what it must never send: an argument the service does not accept never reaches it, a write is same-origin, JSON-only, and loopback-only, a capability the adapter does not declare hides its route and its control, and a stdio client entry names a project and never a directory.

- `test_ui.py` — the shared UI app and its adapter against a recording service: search is always hybrid and always reranked, an unserved argument is dropped rather than forwarded, an unsafe write and a source path outside the source root are refused, a service failure becomes a safe message, and the generation and clients panels follow the same rules as every other write.
- `test_bridge.py` — the stdio transport: a registered name resolves to its own project, a machine holding no such project answers `status` and nothing else rather than refusing the session, a missing app is answered and nothing is started, and a path in a generated `mcp` entry is refused.
- `test_mcp_tools.py` — serialized tool schemas, descriptions and instructions, response-size budgets, mutation annotations, follow-up IDs, and the bounded app-state projection through a real MCP client.
- `test_workspace_app.py` — `surfaces/workspace/app.py` against a fake adapter: every capability is declared in the markup and enforced at the route, the quotation rule is present in the page rather than assumed, and a statement, a write, or a removal this host cannot act on is refused or answered 501 rather than raising.
- `test_workspace_ui.py` — the shipped pages and stylesheet: the clients view, the SQL console, the settings panel, the chunk-exclusion controls, the sidebar that is the whole navigation, the spacing and type scale, the light and dark palettes, and the WCAG AA contrast of every pair that carries text.

The PDF and EPUB writers come from `tests/conftest.py`; `test_bridge.py` uses the `project` fixture there and marks itself `anyio`. Neither gateway nor dense backend is started: `test_ui.py` and the two workspace files stand in a fake adapter, and `test_bridge.py` proxies to an app a fixture has already served.

## Running these

```bash
.venv/bin/python -m pytest tests/surfaces -q
```

A change to what a surface sends or serves belongs here, and a workspace change is tested here, because the workspace ships with this app and a change to it is an app change rather than a presentation change.

## What it mirrors

`src/research_rag/surfaces/`, including `workspace/`.
