# Retired-command tests

- `test_gateway_runtime.py` covers historical flag parsing, migration guidance,
  gateway refusal, successful help, and the runtime command's informational no-op.
- Imports and subprocess probes establish that shims need no MCP/UltraRAG stack,
  acquire no source runtime, start no process, and leave supplied paths untouched.

- Tests use synthetic paths and require no model or upstream source cache.
