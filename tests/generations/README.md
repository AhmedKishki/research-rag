# Generation reuse tests

- `test_generation.py`: `source_set_matches`, the predicate a no-op `ingest` uses to decide whether the selected generation describes the sources as they are.
  - A reviewed-metadata revision does not break the match, because that overlay applies at read time.
  - A changed exclusion revision, retrieval-policy fingerprint, or source digest, a missing index directory, or the legacy metadata storage policy refuses it.
- It builds a `ReuseSnapshot` over a temporary directory and needs no project, gateway, or model.
- `tests/core/test_service.py` exercises the rest of `src/research_rag/generations/` by running a build.
