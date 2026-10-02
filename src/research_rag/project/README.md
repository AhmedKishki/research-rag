# What this project is

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
| `support.py` | The vocabulary every layer shares, and still the one module that does not import what it names |

## Rules

- A name in `state_files.py` is found by other modules, so a rename strands every
  project on disk. Nothing else may spell a state name as a literal.
- `settings.py` declares this app's keys; `settings_layers/` holds no key, no
  default file, and no directory name of its own. `tests/project/` fails a change
  that moves either.
- `normalized_paths.py` exists because five readers had drifted copies of one
  predicate. A new copy is the defect this module was written to remove.
- `support.py` imports `dense`, so a module that wants only `ResearchError` still
  pulls the retrieval stack in with it. `TODO.md` records the split that fixes it,
  and `runtime/update.py` is the test of it.

## What lives elsewhere now

The durable writes these records are made through are in `../storage/`. The
process that resolves a config and holds it is in `../runtime/app.py`.