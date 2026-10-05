# `gateway`

The stdio MCP gateway this app spawns, and the UltraRAG runtime it proxies.

| Module | Holds |
|---|---|
| `server.py` | The MCP server: one tool per UltraRAG stage, over stdio |
| `runtime.py` | The immutable UltraRAG snapshot, installed into a runtime cache and validated against a tree manifest |
| `config.py` | What the gateway resolves from its arguments |
| `manifest.py` | The pinned upstream commit, version, and the servers each snapshot exposes |
| `tree_manifest.py` | The files a complete snapshot contains |
| `instructions.py` | What an agent is told before it calls the gateway |
| `__main__.py` | `python -m research_rag.gateway` |

## Rules

- The gateway is an implementation dependency below the app, not a surface. It is
  reached over MCP and nothing else.
- The runtime cache directory, its marker file, and the environment variable that
  relocates it are found by name, so a rename re-downloads a runtime for every
  machine that already has one.
- `runtime.py` imports the standard library and `platformdirs`. It installs a
  clone-free snapshot at run time, so the gateway has no import-time dependency on
  UltraRAG itself.
- `SERVER_NAME` and the runtime's user agent name the upstream gateway, and they
  say so in a log line a reader has to interpret alongside an installed copy.
