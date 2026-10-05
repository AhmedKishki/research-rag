# The cross-cutting gates

These tests hold the boundaries a reader cannot see from one module: which package may import the web stack, which answer projection the two bounded readers share, and whether the manual still describes the commands, the files, and the credit that are actually shipped.

- `test_architecture.py` — the split at `surfaces/`, read with `ast` rather than executed so an import that succeeds in one environment cannot fool it: only an allowlisted module imports the web stack, no engine module imports a surface, no surface imports another, the agent surface is declared once, the two bounded readers share one projection, the settings module defines no layer machinery, and the entry point runs the command line.
- `test_documentation.py` — the documents against the code: every registered command appears in `README.md` and in the `help` menu exactly once, the storage contract names every project file the app writes, no document names a surface that moved, the UltraRAG credit survives in `README.md` and `NOTICE`, and a shipped client template names a project through the command rather than by hand.

`test_documentation.py` runs `uv run research-rag --help` at import time, so this folder needs the console command installed. Neither file opens a project, a gateway, or a model.

## Running these

```bash
.venv/bin/python -m pytest tests/gates -q
```

A change to an import edge or to a document belongs here, as does any change at all before it is pushed, because both files fail on a rule that no other folder asserts.

## What it mirrors

No single module. `test_architecture.py` reads every folder under `src/research_rag/`, and `test_documentation.py` reads every document at the repository root beside it.