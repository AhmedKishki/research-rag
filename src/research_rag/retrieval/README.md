# `retrieval`

Answering a query against a built generation.

| Module | Holds |
|---|---|
| `search.py` | `SearchWorkflow`: the filter layers, the BM25 and dense candidates, the fusion, and the reranking |
| `dense.py` | The `DenseBackend` protocol and the façade both backends are reached through |
| `dense_backends/` | `LocalQdrantDenseBackend`, `LocalVectorDenseBackend`, and the one portable-vector validation both call |
| `model_runtime.py` | The embedder, the cross-encoder, and the audit tokenizer, loaded once per process |
| `embeddings.py` | The model catalogue and the dimensions each entry declares |
| `rerankers.py` | The reranker catalogue |
| `artifact_lookup.py` | Reading a chunk or a document out of a generation without loading its vectors |
| `ultrarag.py` | The typed MCP client to the gateway in `../gateway/`, and the lazy container that defers its spawn |

## Rules

- `dense.py` imports no numpy, fastembed, qdrant_client, or tokenizers. Those
  belong to `model_runtime.py` and to the two backends, so a caller that only
  needs the protocol does not install a search stack to get it.
- `model_runtime.py` is the single copy of the model loaders and of the six methods
  both backends need. A second copy is a second answer to the same call.
- An exact index build's rows travel in an `ExactIndexBuild` or are held against
  the index path being written. One backend instance serves one process, so a
  second build in the same instance must not inherit the first one's rows.
- `ultrarag.py` speaks MCP to a child process. It does not know what a chunk is.
