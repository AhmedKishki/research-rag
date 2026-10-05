# The absorbed stdio gateway and its managed runtime

These tests hold the two things below the app that were once another product's: the managed runtime tree the app installs and validates offline, and the one end-to-end path that launches the real gateway, builds both indexes, retrieves the passage the fixture wrote, and proves a neighbouring file stayed out of them.

- `test_gateway_runtime.py` — the tree hash and the pinned baseline commit, a polluted or changed tree named file by file, a missing marker reported without a difference, the verified tree left read-only, and a non-POSIX platform reporting that the mode was skipped.
- `test_integration.py` — the real vanilla research flow, marked `integration`: the gateway is launched, BM25 and dense are built and the manifest records the backend it used, hybrid and dense search retrieve the passage the fixture wrote, a neighbouring markdown file is not indexed, the app restarts offline and repeats a reranked hybrid search from the caches, and a gateway that cannot start is reported by the operation that needed it.

The PDF writer comes from `tests/conftest.py`. The integration file needs the real managed runtime and the embedding models already in the account's model cache, which is the one directory `tests/conftest.py` deliberately leaves pointing at the real cache rather than a throwaway one. The unit file builds a synthetic tree and repins the module constants, so it needs neither.

## Running these

```bash
.venv/bin/python -m pytest tests/gateway -q
```

A change to the gateway, the managed runtime, or the end-to-end path belongs here. This is also the only folder that launches the real gateway, so it is the one to run before a retrieval-quality measurement.

## What it mirrors

`src/research_rag/gateway/`.