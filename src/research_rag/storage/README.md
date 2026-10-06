# `storage`

Durable writes, and the record schemas they write.

| Module | Holds |
|---|---|
| `durable_io.py` | Atomic writes with an fsync of the file and its parent directory, and the reads that go with them |
| `records.py` | The five hand-editable record schemas and the current-generation pointer |
| `search_stats.py` | The machine-local search counts: one row per search and one per tracked rank, no query text |

## Rules

- `durable_io.py` knows nothing about this app: no record, no path layout, no
  setting. A write here is atomic and persisted, and cost follows the number of
  durability operations rather than the size of a payload.
- `records.py` validates and never does I/O of its own. Every read refuses a file
  written for a later version and names the field it could not read.
- `records.py` re-exports the durable primitives transitionally. A module importing `records` for `atomic_write_json` alone should import `durable_io`.
