"""Dense search over a Qdrant local-mode collection inside the project.

The Qdrant client is reached only here: its filter types, its payload shape, and
its own verification are not part of any other backend's problem. The model
loaders and the embedding, token-count, and rerank operations moved to
`model_runtime.py`, and the portable-vector file check moved to
`portable_vectors.py`; this module may never re-implement either, because two
copies of a check are two places for it to be wrong.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModel
from ..model_runtime import ModelRuntime
from ..rerankers import DEFAULT_RERANKER_MODEL
from . import COLLECTION_NAME, QDRANT_BACKEND_NAME, DenseSearchHit
from .portable_vectors import load_portable_vectors


class LocalQdrantDenseBackend:
    def __init__(
        self,
        model_cache_root: Path,
        *,
        offline: bool = False,
        embedding_threads: int | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embedding_inference_batch_size: int = 1,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        runtime: ModelRuntime | None = None,
    ) -> None:
        self.runtime = (
            runtime
            if runtime is not None
            else ModelRuntime(
                model_cache_root,
                offline=offline,
                embedding_threads=embedding_threads,
                reranker_model=reranker_model,
                embedding_inference_batch_size=embedding_inference_batch_size,
                embedding_model=embedding_model,
            )
        )
        # The index and every manifest number are written against the runtime's
        # model facts, so an injected runtime is the only statement of them.
        self.embedding_facts: EmbeddingModel = self.runtime.embedding_facts
        self._path_locks: dict[Path, threading.Lock] = {}
        self._path_locks_guard = threading.Lock()

    @contextmanager
    def _opened(self, index_path: Path) -> Iterator[QdrantClient]:
        """One local-mode client on a path at a time, closed after its use.

        Local mode locks its storage directory per client, so a second client on
        the same path fails rather than waits. Reads run beside a build and
        beside one another, so the opens of one path are taken in turn.
        """

        with self._path_locks_guard:
            lock = self._path_locks.setdefault(index_path, threading.Lock())
        with lock:
            client = QdrantClient(path=str(index_path))
            try:
                yield client
            finally:
                client.close()

    def embedding_token_counts(self, texts: list[str]) -> list[int]:
        return self.runtime.embedding_token_counts(texts)

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self.runtime.embed_texts(texts)

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]:
        vectors = load_portable_vectors(
            vectors_path,
            self.embedding_facts.dimension,
            count=len(chunks),
        )
        return self._build_index(chunks, index_path, vectors)

    def _build_index(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("Cannot build a dense index without chunks")
        if index_path.exists():
            raise ValueError(f"Dense index path already exists: {index_path}")
        dimension = int(vectors.shape[1])
        self.initialize_index(index_path, dimension)
        for offset in range(0, len(chunks), 64):
            self.upload_index_batch(
                chunks[offset : offset + 64],
                index_path,
                vectors[offset : offset + 64],
                offset=offset,
            )
        return self.finalize_index(
            index_path,
            expected_count=len(chunks),
            dimension=dimension,
        )

    def initialize_index(self, index_path: Path, dimension: int) -> None:
        if index_path.exists():
            raise ValueError(f"Dense index path already exists: {index_path}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._opened(index_path) as client:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def upload_index_batch(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
        *,
        offset: int,
    ) -> None:
        if vectors.shape != (len(chunks), self.embedding_facts.dimension):
            raise ValueError("Dense index batch has an invalid vector shape")
        with self._opened(index_path) as client:
            points = (
                models.PointStruct(
                    id=offset + index,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "source_id": chunk.get("source_id"),
                    },
                )
                for index, (chunk, vector) in enumerate(
                    zip(chunks, vectors, strict=True)
                )
            )
            client.upload_points(
                collection_name=COLLECTION_NAME,
                points=points,
                batch_size=64,
                wait=True,
            )

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, Any]:
        self.validate_index(
            index_path,
            expected_count=expected_count,
            dimension=dimension,
        )
        return {
            "backend": "Qdrant local mode",
            "dense_backend": QDRANT_BACKEND_NAME,
            "collection": COLLECTION_NAME,
            "distance": "cosine",
            "embedding_runtime": "FastEmbed ONNX Runtime (CPU)",
            "embedding_model": self.embedding_facts.name,
            "embedding_model_repository": self.embedding_facts.repository,
            "embedding_model_revision": self.embedding_facts.revision,
            "embedding_dimension": dimension,
            "point_count": expected_count,
        }

    def validate_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> None:
        if not index_path.is_dir() or index_path.is_symlink():
            raise ValueError(f"Dense index is missing or unsafe: {index_path}")
        with self._opened(index_path) as client:
            if not client.collection_exists(COLLECTION_NAME):
                raise RuntimeError(f"Qdrant collection is missing: {COLLECTION_NAME}")
            collection = client.get_collection(COLLECTION_NAME)
            point_count = int(collection.points_count or 0)
            if point_count != expected_count:
                raise RuntimeError(
                    "Qdrant verification failed: "
                    f"expected {expected_count} points, found {point_count}"
                )
            vectors_config = collection.config.params.vectors
            actual_dimension = getattr(vectors_config, "size", None)
            if actual_dimension != dimension:
                raise RuntimeError(
                    "Qdrant verification failed: "
                    f"expected dimension {dimension}, found {actual_dimension}"
                )

    @staticmethod
    def _filter(
        *,
        document_ids: list[str] | None,
        excluded_document_ids: list[str] | None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=document_ids),
                )
            )
        excluded_conditions: list[models.FieldCondition] = []
        if excluded_document_ids:
            excluded_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=excluded_document_ids),
                )
            )
        return (
            models.Filter(must=conditions, must_not=excluded_conditions)
            if conditions or excluded_conditions
            else None
        )

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        *,
        document_ids: list[str] | None = None,
        excluded_document_ids: list[str] | None = None,
    ) -> list[DenseSearchHit]:
        if not index_path.is_dir():
            raise ValueError(f"Dense index is missing: {index_path}")
        query_vectors = list(self.runtime._embedder().query_embed(query))
        if len(query_vectors) != 1:
            raise RuntimeError("FastEmbed did not return exactly one query vector")

        with self._opened(index_path) as client:
            if not client.collection_exists(COLLECTION_NAME):
                raise RuntimeError(f"Qdrant collection is missing: {COLLECTION_NAME}")
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vectors[0].tolist(),
                query_filter=self._filter(
                    document_ids=document_ids,
                    excluded_document_ids=excluded_document_ids,
                ),
                limit=top_k,
                with_payload=["chunk_id"],
                with_vectors=False,
            )

        hits: list[DenseSearchHit] = []
        for point in response.points:
            chunk_id = (point.payload or {}).get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise RuntimeError("Qdrant returned a point without a chunk_id")
            hits.append(DenseSearchHit(chunk_id=chunk_id, score=float(point.score)))
        return hits

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]:
        return self.runtime.rerank(query, documents, model=model)
