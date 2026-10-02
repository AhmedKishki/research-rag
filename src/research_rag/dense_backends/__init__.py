"""The dense backends and the names they exchange.

`qdrant.py` is the Qdrant local-mode backend, `exact.py` is the portable-vector
backend, and `portable_vectors.py` is the file check they share. The model
loaders and the embedding, token-count, and rerank operations moved to
`model_runtime.py`, which both backends reach through a `ModelRuntime` rather than
through their own copies. Nothing in this package may re-open a model or reach the
service: a backend indexes and searches, and `dense.py` is the façade the app
imports.

The names below are declared before the backends are imported because the backends
read them from this module, and `qdrant_client` types stay inside `qdrant.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model_runtime import (
    DenseTokenAuditUnavailable,
    RerankerUnavailable,
)

# The collection and index filenames each backend writes, and the name each one is
# dispatched by.
COLLECTION_NAME = "research_chunks"
QDRANT_BACKEND_NAME = "embedded-qdrant"
EXACT_BACKEND_NAME = "portable-exact-vectors"
EXACT_INDEX_FILENAME = "index.json"
EXACT_DOCUMENTS_FILENAME = "documents.json"

# The generation-relative directory each dense backend writes its index into.
# The service and its ingestion workflow both read this mapping, so it lives with
# the backends it names, not in either caller.
DENSE_INDEX_PATHS = {
    QDRANT_BACKEND_NAME: "indexes/qdrant",
    EXACT_BACKEND_NAME: "indexes/vectors",
}


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    chunk_id: str
    score: float


from .exact import LocalVectorDenseBackend
from .qdrant import LocalQdrantDenseBackend

__all__ = [
    "COLLECTION_NAME",
    "DENSE_INDEX_PATHS",
    "EXACT_BACKEND_NAME",
    "EXACT_DOCUMENTS_FILENAME",
    "EXACT_INDEX_FILENAME",
    "QDRANT_BACKEND_NAME",
    "DenseSearchHit",
    "DenseTokenAuditUnavailable",
    "LocalQdrantDenseBackend",
    "LocalVectorDenseBackend",
    "RerankerUnavailable",
]
