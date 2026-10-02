# What this project is

Building a generation, and what is inside the one in use.

| Module | Holds |
|---|---|
| `ingestion.py` | The resumable build: scan, extract, chunk, embed, index, and the activation that makes a new generation current |
| `generation.py` | One generation's artifacts on disk, and whether they are a complete build |
| `generation_inventory.py` | What the retained generations cost, read by a status reader |

## Rules

- `ingestion.py` is the largest file in this project and holds the phase loop as
  one closure sequence. `TODO.md` records the split into per-phase handlers, and
  the complexity threshold that would fail the build, as work to do.
- A build is staged and moved. Nothing writes into the generation a search
  reads until the complete build succeeds.
- The checkpoint in `staging/<id>/checkpoint.json` is read by
  `../project/state_files.py` to say a build is still running. That record is a
  contract between this folder and the commands that manage a project: the format
  is read by two modules and written by one.
- `generation_inventory.py` walks the filesystem and knows nothing about a status
  payload, so a reader of the inventory cannot grow a dependency on one.

## What lives elsewhere now

What a chunk and a document are is decided by `../corpus/` and indexed by
`../retrieval/`; this folder is the order those happen in.