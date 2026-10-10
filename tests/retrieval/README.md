# Retrieval tests

The two retrieval paths, the fusion, the models, chunk identifiers, filters, and exclusions. Most tests stand in for the embedding model, so no download is needed.

- `test_dense.py`: the exact cosine and Qdrant backends, resumable batching, document filters and `top_k`, a rejected or misplaced index, embedding batch size and thread count, the token audit, and reranker defaults.
- `test_embeddings.py`: every embedding model carries a download repository, pinned revision, dimension, token limit, and licence, and none claims a language it does not serve.
- `test_model_runtime.py`: exact repository commits reach the downloader and FastEmbed's snapshot-path interface; incomplete or mismatched snapshots cannot satisfy a pin or trigger a fallback.
- `test_rerankers.py`: every reranker is pinned to a revision, and an unknown name is refused with the choices.
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
- `test_ultrarag.py`: a failed start names the reason and both logs, a mid-call exit becomes a tool error naming its log, and the handshake has its own timeout.
- `test_direct.py`: pinned upstream chunk/BM25 differential checks, exact artifact bytes, bidirectional historical index loading, offline tokenizer caches, and cancellation/concurrency boundaries. Upstream reference checks use a cached source snapshot and skip when it is unavailable; normal backend tests need no legacy library.

- `test_search_filters.py`, `test_chunk_exclusions.py`, and `test_search_evaluation_trace.py` import `FakeDenseBackend`, `FakeUltraRAG`, `ProgressivelyFilteredUltraRAG`, and `UnavailableRerankerDenseBackend` from `tests/core/test_service.py`. The last two take the PDF writer from `tests/conftest.py`. Do not move them without updating those imports.
