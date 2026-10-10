# Gateway tests

- `test_gateway_runtime.py`: the tree hash and pinned baseline commit, a polluted or changed tree named file by file, a missing marker, the verified tree left read-only, and a non-POSIX platform reporting the skipped mode.
- `test_integration.py` (marked `integration`): the real research flow.
  - The direct backend builds BM25 and dense indexes, and the manifest records the backend.
  - Hybrid and dense search retrieve the fixture's passage, and a neighbouring markdown file is not indexed.
  - The workspace adapter returns the same search payload as the service.
  - The app restarts offline and repeats a reranked hybrid search from caches.
  - A gateway that cannot start is reported by the operation that needed it.

- The PDF writer comes from `tests/conftest.py`.
- The real research flow needs cached embedding/reranker models and the GPT-2 tokenizer, not the optional legacy runtime. `tests/conftest.py` leaves the model directory pointing at the account's real cache. The legacy failure test uses a synthetic failing executable.
- The unit file builds a synthetic tree and needs neither.
- Run the real research flow before a retrieval-quality measurement.
