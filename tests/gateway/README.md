# Gateway tests

- `test_gateway_runtime.py`: the tree hash and pinned baseline commit, a polluted or changed tree named file by file, a missing marker, the verified tree left read-only, and a non-POSIX platform reporting the skipped mode.
- `test_integration.py` (marked `integration`): the real research flow.
  - The gateway launches, BM25 and dense indexes build, and the manifest records the backend.
  - Hybrid and dense search retrieve the fixture's passage, and a neighbouring markdown file is not indexed.
  - The app restarts offline and repeats a reranked hybrid search from caches.
  - A gateway that cannot start is reported by the operation that needed it.

- The PDF writer comes from `tests/conftest.py`.
- The integration file needs the real managed runtime and the embedding models in the account's model cache. `tests/conftest.py` leaves that one directory pointing at the real cache.
- The unit file builds a synthetic tree and needs neither.
- This is the only folder that launches the real gateway. Run it before a retrieval-quality measurement.
