# Project configuration, the account register, and the shared locations

These tests hold the settings registry, the packaged defaults, the layer stack that resolves them, the project document a browser write rewrites, the account's project register, and every on-disk name this app and no other product may claim. A settings file under a directory nothing reads is ignored, a model cache under a new path downloads both models again, and a pid file two products share lets one stop the other's process.

- `test_settings.py` — the registry and the packaged `default.toml`, layer precedence, refusal of an unknown or out-of-bounds value, the corpus-language rules, and the published retrieval-policy fingerprint a query-time knob must not move.
- `test_settings_layers.py` — the generic layer stack, exercised against a registry of the test's own vocabulary: a layer names only the keys it changes, an undeclared key is an error in every layer, and an absent file is not a layer.
- `test_settings_layers_architecture.py` — what `settings_layers` is allowed to import, read with `ast` rather than executed, so the package cannot acquire a domain type or a retrieval import one release at a time.
- `test_settings_document.py` — the document a write rewrites: a merge keeps every key it did not change, a write against a revision the reader never held is refused, a value outside its bound is refused, and the cost of a change is computed from what a generation records rather than from the registry's own label.
- `test_registry.py` — the account's `projects.json`: resolution by name or id, a name two projects share refused, a prefix refused as a guess, a damaged record reported rather than guessed, and an entry that is not a pointer skipped so one unreadable project cannot hide the rest.
- `test_data_roots.py` — `USER_CONFIG_DIRECTORY`, `SETTINGS_ENVIRONMENT_PREFIX`, the model-cache default, the relocated-runtime record, and the pid, port, and tty file names, each asserted against the name of another product.

The `project` fixture and the throwaway account directory come from `tests/conftest.py`, and the two fixtures there are autouse for every test in this folder. `test_settings_document.py` drives `ResearchService.settings_read` and `settings_write` from `src/research_rag/core/`, so it holds the project contract as the core service answers it rather than as the document module renders it.

## Running these

```bash
.venv/bin/python -m pytest tests/project -q
```

This is the right scope when a change moves a setting, a layer, or a shared name, because each of those is read by a command, a surface, or a file outside this folder and no other test would notice it.

## What it mirrors

`src/research_rag/project/`.