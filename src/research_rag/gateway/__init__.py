"""Retired UltraRAG gateway command names, kept as migration shims.

`server.py` and `runtime.py` keep the historical console names so an old
wrapper or client fails with guidance instead of a missing executable. The
stdio proxy gateway and its source-snapshot installer are removed: the app
serves its own nine agent tools through `research-rag mcp` and
needs no UltraRAG runtime.
"""
