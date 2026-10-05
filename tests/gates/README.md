# Cross-cutting gates

Boundaries a reader cannot see from one module.

- `test_architecture.py`: reads imports with `ast`. Only an allowlisted module imports the web stack, no engine module imports a surface, no surface imports another, the agent surface is declared once, the two bounded readers share one projection, the settings module defines no layer machinery, and the entry point runs the command line.
- `test_documentation.py`: documents against code. Every command appears in `README.md` and the `help` menu exactly once, the storage contract names every project file, no document names a moved surface, the UltraRAG credit survives, and a client template names a project through the command.

- `test_documentation.py` runs `uv run research-rag --help` at import, so the console command must be installed.
- Neither file opens a project, a gateway, or a model.
- Run this folder before any push.
