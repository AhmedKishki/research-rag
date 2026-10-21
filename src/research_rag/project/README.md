# `project`

What one project is, how it is configured, and the names its state is found by.

| Module | Holds |
|---|---|
| `config.py` | `ResearchConfig`: every path a project resolves, resolved once |
| `settings.py` | This app's registry of settings, its bounds, and the packaged `default.toml` |
| `settings_document.py` | The settings a reader is shown, with their layers and costs |
| `settings_layers/` | The merge that wins per key, the coercion every layer shares, and the path helpers |
| `registry.py` | The account's record of which projects exist |
| `credentials.py` | The account's DeepSeek key: `credentials.json`, owner-only, with the environment overriding it |
| `embeddings.py` | The embedding model catalogue: each model's pinned revision, dimension, token limit, and languages |
| `rerankers.py` | The reranker catalogue and each model's pinned revision |
| `model_cache.py` | Where a pinned model snapshot lives in the cache, and whether it is complete |
| `state_files.py` | The on-disk names every module agrees on, and the readers for a pid and a running build |
| `normalized_paths.py` | The one rule a path stored in a file must satisfy |
| `instructions.py` | What an agent is told before it calls a tool |
| `policy.py` | Error type, retrieval-method names, schema and policy versions, policy fingerprints, and UTC time helpers without retrieval imports |

## Rules

- A name in `state_files.py` is found by other modules, so a rename strands every
  project on disk. Nothing else may spell a state name as a literal.
- `Setting.doc` is the one description of a key. `default.toml` holds values only,
  and `research-rag config` and the workspace print the description.
- `settings.py` declares this app's keys; `settings_layers/` holds no key, no
  default file, and no directory name of its own. `tests/project/` fails a change
  that moves either.
- `normalized_paths.py` owns stored-path validation; import it rather than copy it.
- `policy.py` imports no other folder, so a command that runs without the retrieval stack can reach the failure type and the policy versions.
  `tests/gates/test_lightweight_imports.py` guards those commands.
- The DeepSeek key is a credential, not a setting: it never enters `SETTINGS`,
  a `config.toml`, or a message. `credentials.py` names the file and the field
  it refuses, never the value.
