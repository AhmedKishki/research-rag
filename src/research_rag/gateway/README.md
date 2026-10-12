# `gateway`

Migration guidance for retired gateway/runtime command names. No component here
serves MCP tools or acquires an external source runtime.

| Module | Holds |
|---|---|
| `server.py` | Parses historical gateway flags and refuses with project-bridge migration guidance |
| `runtime.py` | Parses historical runtime flags and reports that no external runtime is required |
| `__main__.py` | `python -m research_rag.gateway` |

## Rules

- Both shims import only the standard library.
- Historical flags are accepted only to explain migration. They select no runtime,
  output path, namespace, or process.
- The shims start no server or child process, perform no network request, and
  never inspect, write, validate, quarantine, or delete caches.
- The gateway command exits nonzero rather than pretending to offer retired tools.
  Its help succeeds. The runtime command reports a no-op and exits successfully.
