# Project tests

Settings, the account register, and the names only this app may claim.

- `test_settings.py`: the registry and packaged `default.toml`, layer precedence, refusal of an unknown or out-of-bounds value, corpus-language rules, and the retrieval-policy fingerprint a query-time setting must not move.
- `test_settings_layers.py`: the generic layer stack against a test registry. A layer names only the keys it changes, an undeclared key is an error in every layer, and an absent file is not a layer.
- `test_settings_layers_architecture.py`: what `settings_layers` may import, read with `ast`.
- `test_settings_document.py`: the document a write rewrites.
  - A merge keeps unchanged keys.
  - A write against an unseen revision is refused.
  - An out-of-bounds value is refused.
  - Change cost comes from what a generation records, not the registry's label.
  - It drives `ResearchService.settings_read` and `settings_write`.
- `test_registry.py`: `projects.json`. Resolution by name or id, a shared name refused, a prefix refused as a guess, a damaged record reported, and a non-pointer entry skipped.
- `test_data_roots.py`: `USER_CONFIG_DIRECTORY`, `SETTINGS_ENVIRONMENT_PREFIX`, the model-cache default, the relocated-runtime record, and the pid, port, and tty file names, each against another product's name.

- The `project` fixture and the throwaway account directory come from `tests/conftest.py`. Both are autouse here.
