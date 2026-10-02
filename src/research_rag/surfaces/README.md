# What this project is

The three front ends, and the transport that reaches the app.

| Module | Holds |
|---|---|
| `cli.py` | The command line: what commands exist, what each does, and how an answer is printed |
| `mcp.py` | The agent surface: the tools and the resource, declared once |
| `ui.py` | This app's adapter over the workspace, and the capability profile that says what a browser may do |
| `workspace/` | The browser workspace itself: the Starlette host, the contracts it and an adapter share, and its assets |
| `bridge.py` | `research-rag mcp`: resolve a project by name, then proxy stdio to the app or answer why it cannot |

## Rules

- A surface answers from the service and holds no research state. A workspace
  route that read a file would make the browser a second implementation.
- The tools are declared once, in `mcp.py`. The bridge proxies them rather than
  re-declaring them, so an agent and a browser cannot disagree about one call.
- A blocked surface declares `status` and nothing else, and the remedy it gives is
  a command a reader runs in a terminal.
- `cli.py` is the largest file in this project and holds the parser, the command
  bodies, and the output shaping. `TODO.md` records the cut between those three
  as work to do.
- `workspace/` owns no project state and imports nothing outside itself. A host
  that stores a document in a shared package is a host that cannot say where its
  state is.