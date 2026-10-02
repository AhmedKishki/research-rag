# The retrieval halves and the filters over them

These tests hold the two retrieval paths behind one fusion, the models that decide their vectors, the identifiers and the SQLite lookup that address a chunk without holding its text, the bibliographic filters, the passage exclusions, and the typed boundary over the gateway whose failures name the log the transport already wrote. The embedding model is stood in for where the vector arithmetic is not the subject, so most of the folder needs no model download.

- `test_dense.py` — the exact cosine backend and the Qdrant backend, resumable batching, document filters and `top_k`, a rejected or misplaced index, the embedding batch size and thread count, the token audit, and the reranker defaults.
- `test_embeddings.py` — every supported embedding model carries a pinned revision, a dimension, a token limit, and a licence, and a model may not claim a language it does not serve.
- `test_rerankers.py` — every supported reranker is pinned to a revision and an unknown name is refused with the choices.
- `test_artifact_lookup.py` — the UTF-8 offsets, the duplicate documents, the vectors keyed by contents, the stored chunk verdict matched against a query-time scan, and the rebuild of a missing, corrupt, or stale lookup.
- `test_search_filters.py` — the search argument contract: source selection by stable id, the six reviewed filter layers, their combinations, the candidate ceiling a filter may not widen, and the window an empty answer names. It imports `FakeDenseBackend`, `FakeUltraRAG`, and `ProgressivelyFilteredUltraRAG` from `tests/core/test_service.py`.
- `test_chunk_exclusions.py` — a passage decision enforced by both retrieval halves, reversible, named in the answer, and reported when the generation on screen does not hold the passage. It imports the same two fakes from `tests/core/test_service.py` and the PDF writer from `tests/conftest.py`.
- `test_ultrarag.py` — the transport's failures: a start that fails names the reason and both logs, a gateway that exits mid-call becomes a tool error naming its log, and the handshake is bounded by its own timeout.

## Running these

```bash
.venv/bin/python -m pytest tests/retrieval -q
```

This is the right scope for a change to ranking, a model pin, a filter, or an identifier, because every surface reads through these and none of them re-implements any of it.

## What it mirrors

`src/research_rag/retrieval/`, including `dense_backends/`.