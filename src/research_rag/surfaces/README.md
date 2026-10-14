# `surfaces`

The three front ends, and the transport that reaches the app.

| Module | Holds |
|---|---|
| `cli/` | The command line: `__init__.py` runs a command and prints its answer; `help.py` and `parser.py` say what commands exist; `projects.py` resolves the project; `corpus.py` forwards an operation to the app or the local service; `serving.py` starts and stops the app; `installation.py` installs and updates |
| `mcp.py` | The agent surface: the tools and the resource, declared once |
| `ui.py` | This app's adapter over the workspace: the shared operations, and the few only the workspace serves |
| `workspace/` | The browser workspace itself: the Starlette host, the contracts it and an adapter share, and its assets |
| `bridge.py` | `research-rag mcp`: resolve a project by name on every call, proxy stdio to the app when it answers, and answer with the start command when it does not |

## Rules

- A surface answers from the service and holds no research state. A workspace
  route that read a file would make the browser a second implementation.
- The tools are declared once, in `mcp.py`. The bridge proxies them rather than
  re-declaring them, so an agent and a browser cannot disagree about one call.
- A blocked surface declares `status` and nothing else, and the remedy it gives is
  a command a reader runs in a terminal.
- `cli/` imports retrieval only when a command needs it.
  `tests/gates/test_lightweight_imports.py` covers the commands that must run
  without it.
- `workspace/` owns no project state and imports nothing outside itself. A host
  that stores a document in a shared package is a host that cannot say where its
  state is.
