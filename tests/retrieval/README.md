# The retrieval halves and the filters over them

These tests hold the two retrieval paths behind one fusion, the models that decide their vectors, the identifiers and the SQLite lookup that address a chunk without holding its text, the bibliographic filters, the passage exclusions, and the typed boundary over the gateway whose failures name the log the transport already wrote. The embedding model is stood in for where the vector arithmetic is not the subject, so most of the folder needs no model download.

- `test_dense.py` — the exact cosine backend and the Qdrant backend, resumable batching, document filters and `top_k`, a rejected or misplaced index, the embedding batch size and thread count, the token audit, and the reranker defaults.
- `test_embeddings.py` — every supported embedding model carries a pinned revision, a dimension, a token limit, and a licence, and a model may not claim a language it does not serve.
- `test_rerankers.py` — every supported reranker is pinned to a revision and an unknown name is refused with the choices.
- `test_artifact_lookup.py` — the UTF-8 offsets, the duplicate documents, the vectors keyed by contents, the stored chunk verdict matched against a query-time scan, and the rebuild of a missing, corrupt, or stale lookup.
- `test_search_filters.py` — the search argument contract: source selection by stable id, the six reviewed filter layers, their combinations, the candidate ceiling a filter may not widen, and the window an empty answer names.
- `test_chunk_exclusions.py` — a passage decision enforced by both retrieval halves, reversible, named in the answer, and reported when the generation on screen does not hold the passage.
- `test_search_evaluation_trace.py` — the evaluation trace a search can be asked for: the eight named ranking stages with their own counts, the bounded identifier lists and the flag that says one was cut, the absence of passage text, the cosine gate's denominator and its conservation, the candidate-depth and rerank-window formulas, a BM25 payload reporting no dense counts rather than zeros, an unavailable reranker traced as not applied, and a traced search returning an answer otherwise identical to an untraced one.
- `test_ultrarag.py` — the transport's failures: a start that fails names the reason and both logs, a gateway that exits mid-call becomes a tool error naming its log, and the handshake is bounded by its own timeout.

`test_search_filters.py`, `test_chunk_exclusions.py`, and `test_search_evaluation_trace.py` import `FakeDenseBackend`, `FakeUltraRAG`, `ProgressivelyFilteredUltraRAG`, and `UnavailableRerankerDenseBackend` from `tests/core/test_service.py`, and the last two take the PDF writer from `tests/conftest.py`, so those three files cannot be moved out of this folder without a change to those imports.

## Running these

```bash
.venv/bin/python -m pytest tests/retrieval -q
```

A change to ranking, a model pin, a filter, or an identifier belongs here, because every surface reads through these and none of them re-implements any of it.

## What it mirrors

`src/research_rag/retrieval/`, including `dense_backends/`.