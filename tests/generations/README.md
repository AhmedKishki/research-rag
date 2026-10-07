# Generation tests

- `test_generation.py`: `source_set_matches`, the predicate a no-op `ingest` uses to decide whether the selected generation describes the sources as they are.
  - A reviewed-metadata revision does not break the match, because that overlay applies at read time.
  - A changed exclusion revision, retrieval-policy fingerprint, or source digest, a missing index directory, or the legacy metadata storage policy refuses it.
- Reuse tests build a `ReuseSnapshot` over a temporary directory and need no project, gateway, or model.
- Inventory tests expose recorded configuration without changing manifests or inventing missing historical settings.
- `tests/core/test_service.py` exercises the rest of `src/research_rag/generations/` by running a build.
