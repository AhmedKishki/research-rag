# Generation tests

- `test_generation.py`: the reuse gates in `generation.py`.
  - `source_set_matches` is the predicate a no-op `ingest` uses to decide whether the selected generation describes the sources as they are.
    - A reviewed-metadata revision does not break the match, because that overlay applies at read time.
    - A ranking-policy edit does not break the match either, because ranking is a runtime decision that changes no stored artifact.
    - A changed exclusion revision or source digest, a missing index directory, or the legacy metadata storage policy refuses it.
  - `extraction_is_reusable` and `vectors_are_reusable` are independent: a model change keeps extraction and chunks reusable, and a processing-policy change keeps text-identical vectors reusable.
  - Reuse tests build a `ReuseSnapshot` or real artifact files under a temporary directory and need no project, gateway, or model.
- `test_reuse_axes.py`: the same gates driven through the real `ingest` state machine with the shared retrieval fakes.
  - A model-only change reuses extraction and chunks while recomputing every vector.
  - `force_recompute` disables every reuse axis.
- `test_docling_staged_ingestion.py`: the staged Docling build end to end with a recorded worker.
  - A backend or adapter drift alone is not a reason to rebuild an unchanged corpus.
  - An unchanged source is passed through whole; a legacy PDF keeps a per-document
    provenance of the retired reader and only a changed PDF is converted again.
  - A no-op returns `unchanged` before any Docling environment is resolved.
  - The checkpoint identity is stable across a first-use install; a resume under a
    changed actual environment discards the unfinished source's pages instead of
    mixing provenance.
- `test_pdf_backend_reuse.py`: the PDF backend is provenance, never a generation-wide reuse gate.
- Inventory tests expose recorded configuration without changing manifests or inventing missing historical settings.
- `tests/core/test_service.py` exercises the rest of `src/research_rag/generations/` by running a build.
