# `storage`

Durable writes, and the record schemas they write.

| Module | Holds |
|---|---|
| `durable_io.py` | Atomic writes with an fsync of the file and its parent directory, and the reads that go with them |
| `records.py` | The five hand-editable record schemas and the current-generation pointer |

## Rules

- `durable_io.py` knows nothing about this app: no record, no path layout, no
  setting. A write here is atomic and persisted, and cost follows the number of
  durability operations rather than the size of a payload.
- `records.py` validates and never does I/O of its own. Every read refuses a file
  written for a later version and names the field it could not read.
- The durable primitives are re-exported by `records.py`. That re-export is
  transitional: a module that imports `records` for `atomic_write_json` alone
  should import `durable_io` instead, and no document records that sweep.

## What lives elsewhere

The rule a stored path must satisfy is in `../project/normalized_paths.py`.