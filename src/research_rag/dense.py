"""The dense boundary the rest of the app imports.

It is a deliberate façade: `model_runtime.py` owns the models, `dense_backends/`
owns the two backends and the names they exchange, and this module keeps the one
import path every caller already reads, so a name moving between those two does not
ripple through the service, the command line, or the tests. The protocol below is
the contract that callers and their deterministic doubles are written against.

It may never import numpy, fastembed, qdrant_client, or tokenizers, because it holds
no behaviour any of them would serve: the numpy annotations the protocol names are
read by a type checker alone. The two model loaders are re-exported because
`doctor.prefetch_models` reaches the models through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .dense_backends import (
    COLLECTION_NAME,
    DENSE_INDEX_PATHS,
    EXACT_BACKEND_NAME,
    EXACT_DOCUMENTS_FILENAME,
    EXACT_INDEX_FILENAME,
    QDRANT_BACKEND_NAME,
    DenseSearchHit,
    DenseTokenAuditUnavailable,
    LocalQdrantDenseBackend,
    LocalVectorDenseBackend,
    RerankerUnavailable,
)
from .model_runtime import (  # noqa: F401
    _load_cross_encoder,
    _load_embedder,
)

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "COLLECTION_NAME",
    "DENSE_INDEX_PATHS",
    "EXACT_BACKEND_NAME",
    "EXACT_DOCUMENTS_FILENAME",
    "EXACT_INDEX_FILENAME",
    "QDRANT_BACKEND_NAME",
    "DenseBackend",
    "DenseSearchHit",
    "DenseTokenAuditUnavailable",
    "LocalQdrantDenseBackend",
    "LocalVectorDenseBackend",
    "RerankerUnavailable",
]


class DenseBackend(Protocol):
    """Boundary used by the service and deterministic test doubles."""

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]: ...

    def embed_texts(
        self, texts: list[str]
    ) -> np.ndarray[Any, np.dtype[np.float32]]: ...

    def embedding_token_counts(self, texts: list[str]) -> list[int]: ...

    def initialize_index(self, index_path: Path, dimension: int) -> None: ...

    def upload_index_batch(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
        *,
        offset: int,
    ) -> None: ...

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, Any]: ...

    def validate_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> None: ...

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        *,
        document_ids: list[str] | None = None,
        excluded_document_ids: list[str] | None = None,
    ) -> list[DenseSearchHit]: ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]: ...
