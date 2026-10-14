# `storage`

Durable writes, and the machine-local search counts.

| Module | Holds |
|---|---|
| `durable_io.py` | `atomic_file` and the writes built on it, with an fsync of the file and its parent directory, and the reads that go with them |
| `search_stats.py` | The machine-local search counts: one row per search and one per tracked rank, no query text |

## Rules

- `durable_io.py` knows nothing about this app: no record, no path layout, no
  setting. A write here is atomic and persisted, and cost follows the number of
  durability operations rather than the size of a payload.
- An atomic write anywhere in the app goes through `durable_io.atomic_file` or `replace_durably`; no module renames over a file by hand.
