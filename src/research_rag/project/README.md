# `project`

What one project is, how it is configured, and the names its state is found by.

| Module | Holds |
|---|---|
| `config.py` | `ResearchConfig`: every path a project resolves, resolved once |
| `settings.py` | This app's registry of settings, its bounds, and the packaged `default.toml` |
| `settings_document.py` | The settings a reader is shown, with their layers and costs |
| `settings_layers/` | The merge that wins per key, the coercion every layer shares, and the path helpers |
| `registry.py` | The account's record of which projects exist |
| `state_files.py` | The on-disk names every module agrees on, and the readers for a pid and a running build |
| `normalized_paths.py` | The one rule a path stored in a file must satisfy |
| `instructions.py` | What an agent is told before it calls a tool |
| `policy.py` | Error type, retrieval-method names, policy fingerprints, and UTC time helpers without retrieval imports |
| `support.py` | Shared corpus and generation helpers; still coupled to the retrieval stack |

## Rules

- A name in `state_files.py` is found by other modules, so a rename strands every
  project on disk. Nothing else may spell a state name as a literal.
- `settings.py` declares this app's keys; `settings_layers/` holds no key, no
  default file, and no directory name of its own. `tests/project/` fails a change
  that moves either.
- `normalized_paths.py` owns stored-path validation; import it rather than copy it.
- Import `ResearchError` and policy primitives from `policy.py`, not `support.py`.
  - `support.py` still imports retrieval dependencies; `TODO.md` tracks its remaining split.
  - `tests/gates/test_lightweight_imports.py` guards commands that must run without the stack.

## What lives elsewhere

The durable writes these records are made through are in `../storage/`. The
process that resolves a config and holds it is in `../runtime/app.py`.
