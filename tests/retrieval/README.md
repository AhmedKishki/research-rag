# Retrieval tests

The two retrieval paths, the fusion, the models, chunk identifiers, filters, and exclusions. Most tests stand in for the embedding model, so no download is needed.

- `test_dense.py`: the exact cosine and Qdrant backends, resumable batching, document filters and `top_k`, a rejected or misplaced index, embedding batch size and thread count, the token audit, and reranker defaults.
- `test_model_runtime.py`: exact repository commits reach the downloader and FastEmbed's snapshot-path interface; incomplete or mismatched snapshots cannot satisfy a pin or trigger a fallback.
- `test_artifact_lookup.py`: UTF-8 offsets, duplicate documents, vectors keyed by contents, the stored verdict against a query-time scan, and rebuilding a missing, corrupt, or stale lookup.
- `test_search_filters.py`: source selection by stable id, the six reviewed filter layers and their combinations, the candidate ceiling a filter may not widen, and the window an empty answer names.
- `test_chunk_exclusions.py`: a passage decision enforced by both halves, reversible, named in the answer, and reported when the generation does not hold the passage.
- `test_search_evaluation_trace.py`: the evaluation trace.
  - The eight ranking stages with their counts, bounded identifier lists with a cut flag, and no passage text.
  - The cosine gate's denominator and conservation.
  - The candidate-depth and rerank-window formulas.
  - A BM25 payload reporting no dense counts, not zeros.
  - An unavailable reranker traced as not applied.
  - A traced search answering identically to an untraced one.
- `test_direct.py`: self-contained historical chunk normalization/window checks,
  saved-index loading and non-ASCII stopword round-tripping through bm25s, offline
  tokenizer caches, and cancellation/concurrency boundaries. No upstream source
  checkout or framework library is required.
- `test_integration.py` exercises the real direct backend, both indexes, hybrid
  and dense search, a known passage and excluded neighbour, workspace forwarding,
  and an offline restart with a reranked search from caches. It needs the cached
  embedding/reranker models and GPT-2 tokenizer, not an upstream runtime.

- `test_search_filters.py`, `test_chunk_exclusions.py`, and `test_search_evaluation_trace.py` import `FakeDenseBackend`, `FakeUltraRAG`, `ProgressivelyFilteredUltraRAG`, and `UnavailableRerankerDenseBackend` from `tests/core/test_service.py`. The last two take the PDF writer from `tests/conftest.py`. Do not move them without updating those imports.
