"""Project-local dense retrieval and optional CPU reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModel,
    resolve_embedding_model,
)
from .rerankers import DEFAULT_RERANKER_MODEL, resolve_reranker_model
from .storage import StorageError, atomic_write_json, read_json

# Sequences are padded to the longest member of their inference batch, so a
# large batch spends most of its compute on padding: on the reference corpus
# (mean 152 tokens, max 846) one sequence per inference measured 23.7
# chunks/s against 6.2 at a batch of 64, and a batch of 1 returns exactly
# the same floats as a batch of 64. See MEASUREMENTS.md.
COLLECTION_NAME = "research_chunks"
QDRANT_BACKEND_NAME = "embedded-qdrant"
EXACT_BACKEND_NAME = "portable-exact-vectors"
EXACT_INDEX_FILENAME = "index.json"
EXACT_DOCUMENTS_FILENAME = "documents.json"
_EXACT_VECTORS_RELATIVE = Path("portable") / "embeddings.npy"

# The generation-relative directory each dense backend writes its index into.
# The service and its ingestion workflow both read this mapping, so it lives with
# the backends it names rather than in either caller.
DENSE_INDEX_PATHS = {
    QDRANT_BACKEND_NAME: "indexes/qdrant",
    EXACT_BACKEND_NAME: "indexes/vectors",
}


class DenseTokenAuditUnavailable(RuntimeError):
    """Raised when the embedding tokenizer cannot be inspected safely."""


class RerankerUnavailable(RuntimeError):
    """Raised when the optional reranker model cannot be loaded.

    The service treats this as a recoverable condition: reranking is skipped,
    the unranked candidate order is returned, and the response discloses the
    fallback instead of failing the search.
    """


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    """One scored Qdrant result, identified by the canonical chunk ID."""

    chunk_id: str
    score: float


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


def _load_embedder(
    cache_root: Path,
    *,
    offline: bool,
    threads: int | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> TextEmbedding:
    """Load the pinned CPU embedding model from the shared model cache.

    `threads` sets the ONNX Runtime intra-op and inter-op thread count. The
    default is left to the runtime; MEASUREMENTS.md records the measured effect
    of setting it on one machine.
    """

    facts = resolve_embedding_model(model)
    cache_root.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=facts.name,
        cache_dir=str(cache_root),
        cuda=False,
        local_files_only=offline,
        revision=facts.revision,
        threads=threads,
    )


def _load_cross_encoder(
    cache_root: Path,
    *,
    offline: bool,
    model: str = DEFAULT_RERANKER_MODEL,
) -> TextCrossEncoder:
    """Load one pinned CPU cross-encoder from the shared model cache.

    Any load failure means the reranker cannot run — most often an offline call
    whose model is not cached yet — so it is reported as ``RerankerUnavailable``
    for the caller to degrade from rather than as an opaque model error.
    """

    name, revision = resolve_reranker_model(model)
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        return TextCrossEncoder(
            model_name=name,
            cache_dir=str(cache_root),
            cuda=False,
            local_files_only=offline,
            revision=revision,
        )
    except Exception as exc:
        hint = (
            " offline mode requires it to be cached already"
            if offline
            else " it is downloaded on first use"
        )
        raise RerankerUnavailable(
            f"The reranker model {name} cannot be loaded:{hint}. {exc}"
        ) from exc


def _load_audit_tokenizer(embedder: TextEmbedding) -> Tokenizer:
    """Return a non-truncating tokenizer matching the embedding model.

    The embedder's own tokenizer truncates at the model limit, which would hide
    the overflow the ingestion audit exists to measure, so the pinned tokenizer
    file is loaded separately and truncation is disabled on that copy.
    """

    model_dir = getattr(embedder.model, "_model_dir", None)
    tokenizer_path = Path(str(model_dir)) / "tokenizer.json" if model_dir else None
    if tokenizer_path is None or not tokenizer_path.is_file():
        raise DenseTokenAuditUnavailable(
            "The embedding tokenizer is missing from the model cache: "
            + str(tokenizer_path)
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    return tokenizer


class LocalQdrantDenseBackend:
    """FastEmbed CPU vectors stored in an embedded, project-local Qdrant DB."""

    def __init__(
        self,
        model_cache_root: Path,
        *,
        offline: bool = False,
        embedding_threads: int | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embedding_inference_batch_size: int = 1,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_cache_root = model_cache_root
        self.offline = offline
        self.embedding_threads = embedding_threads
        self.reranker_model = reranker_model
        # The model's own facts — name, revision, dimension, token limit, and any
        # required prefix — come from the pinned table rather than the caller.
        self.embedding_facts: EmbeddingModel = resolve_embedding_model(embedding_model)
        # Throughput only: a batch of 1 returns exactly the same floats as a
        # batch of 64 (MEASUREMENTS.md).
        self.embedding_inference_batch_size = embedding_inference_batch_size
        self._embedding_model: TextEmbedding | None = None
        self._rerankers: dict[str, TextCrossEncoder] = {}
        self._audit_tokenizer: Tokenizer | None = None

    def _embedder(self) -> TextEmbedding:
        if self._embedding_model is None:
            self._embedding_model = _load_embedder(
                self.model_cache_root,
                offline=self.offline,
                threads=self.embedding_threads,
                model=self.embedding_facts.name,
            )
        return self._embedding_model

    def _cross_encoder(self, model: str | None = None) -> TextCrossEncoder:
        """Return the pinned cross-encoder for one model, loading it on demand.

        A run that compares rerankers needs more than one model, so each model is
        loaded at most once and kept for the life of the backend.
        """

        name = model or self.reranker_model
        encoder = self._rerankers.get(name)
        if encoder is None:
            encoder = _load_cross_encoder(
                self.model_cache_root,
                offline=self.offline,
                model=name,
            )
            self._rerankers[name] = encoder
        return encoder

    @staticmethod
    def _client(index_path: Path) -> QdrantClient:
        return QdrantClient(path=str(index_path))

    def _audit_tokenizer_for_ingestion(self) -> Tokenizer:
        if self._audit_tokenizer is None:
            try:
                self._audit_tokenizer = _load_audit_tokenizer(self._embedder())
            except DenseTokenAuditUnavailable:
                raise
            except Exception as exc:
                raise DenseTokenAuditUnavailable(
                    "The embedding model is unavailable for the token audit: "
                    + str(exc)
                ) from exc
        return self._audit_tokenizer

    def embedding_token_counts(self, texts: list[str]) -> list[int]:
        """Return the embedding tokenizer length of each text, untruncated."""

        if not texts:
            return []
        tokenizer = self._audit_tokenizer_for_ingestion()
        try:
            return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
        except Exception as exc:
            raise DenseTokenAuditUnavailable(str(exc)) from exc

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not texts:
            return np.empty((0, self.embedding_facts.dimension), dtype=np.float32)
        embedded = list(
            self._embedder().passage_embed(
                texts, batch_size=self.embedding_inference_batch_size
            )
        )
        if len(embedded) != len(texts):
            raise RuntimeError(
                "FastEmbed returned a different number of vectors than passages"
            )
        vectors = np.asarray(embedded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (
            len(texts),
            self.embedding_facts.dimension,
        ):
            raise RuntimeError(
                f"Unexpected {self.embedding_facts.name} vector shape: {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("FastEmbed returned non-finite embedding values")
        return vectors

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]:
        try:
            vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot load portable embeddings: {vectors_path}"
            ) from exc
        if vectors.dtype != np.float32:
            raise ValueError("Portable embeddings must use float32 values")
        if vectors.ndim != 2 or vectors.shape != (
            len(chunks),
            self.embedding_facts.dimension,
        ):
            raise ValueError(
                "Portable embedding dimensions do not match the chunk collection"
            )
        for offset in range(0, len(vectors), 4096):
            if not np.isfinite(vectors[offset : offset + 4096]).all():
                raise ValueError("Portable embeddings contain non-finite values")
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
        client = self._client(index_path)
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        finally:
            client.close()

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
        client = self._client(index_path)
        try:
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
        finally:
            client.close()

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
        client = self._client(index_path)
        try:
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
        finally:
            client.close()

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
        query_vectors = list(self._embedder().query_embed(query))
        if len(query_vectors) != 1:
            raise RuntimeError("FastEmbed did not return exactly one query vector")

        client = self._client(index_path)
        try:
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
        finally:
            client.close()

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
        if not documents:
            return []
        scores = list(
            self._cross_encoder(model).rerank(
                query,
                documents,
                batch_size=32,
            )
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                "FastEmbed reranker returned a different number of scores than passages"
            )
        return [float(score) for score in scores]


@dataclass(frozen=True, slots=True)
class _ExactIndex:
    """A loaded exact index: memory-mapped vectors plus per-row identity."""

    vectors: np.ndarray[Any, np.dtype[np.float32]]
    chunk_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    norms: np.ndarray[Any, np.dtype[np.float32]]


def _read_index_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except StorageError as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


class LocalVectorDenseBackend:
    """Exact dense search over a generation's portable float32 vectors.

    The generation already stores one float32 row per chunk in stable chunk
    order, so a dense index needs only a small descriptor plus per-row identity
    for filtering, and search is an exact cosine scan. That removes the
    whole-corpus index build and its per-point device cost at the corpus sizes
    this server targets, and it makes dense scoring exactly reproducible.

    An index lives at `<generation>/indexes/<name>` and references the portable
    vector file relative to the generation root, so the descriptor stays valid
    when the generation directory is copied elsewhere.
    """

    def __init__(
        self,
        model_cache_root: Path,
        *,
        offline: bool = False,
        embedding_threads: int | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embedding_inference_batch_size: int = 1,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_cache_root = model_cache_root
        self.offline = offline
        self.embedding_threads = embedding_threads
        self.reranker_model = reranker_model
        # The model's own facts — name, revision, dimension, token limit, and any
        # required prefix — come from the pinned table rather than the caller.
        self.embedding_facts: EmbeddingModel = resolve_embedding_model(embedding_model)
        # Throughput only: a batch of 1 returns exactly the same floats as a
        # batch of 64 (MEASUREMENTS.md).
        self.embedding_inference_batch_size = embedding_inference_batch_size
        self._embedding_model: TextEmbedding | None = None
        self._rerankers: dict[str, TextCrossEncoder] = {}
        self._audit_tokenizer: Tokenizer | None = None
        self._loaded: dict[Path, tuple[int, _ExactIndex]] = {}
        self._pending_chunk_ids: list[str] = []
        self._pending_document_ids: list[str] = []

    def _embedder(self) -> TextEmbedding:
        if self._embedding_model is None:
            self._embedding_model = _load_embedder(
                self.model_cache_root,
                offline=self.offline,
                threads=self.embedding_threads,
                model=self.embedding_facts.name,
            )
        return self._embedding_model

    def _cross_encoder(self, model: str | None = None) -> TextCrossEncoder:
        """Return the pinned cross-encoder for one model, loading it on demand.

        A run that compares rerankers needs more than one model, so each model is
        loaded at most once and kept for the life of the backend.
        """

        name = model or self.reranker_model
        encoder = self._rerankers.get(name)
        if encoder is None:
            encoder = _load_cross_encoder(
                self.model_cache_root,
                offline=self.offline,
                model=name,
            )
            self._rerankers[name] = encoder
        return encoder

    def _audit_tokenizer_for_ingestion(self) -> Tokenizer:
        if self._audit_tokenizer is None:
            try:
                self._audit_tokenizer = _load_audit_tokenizer(self._embedder())
            except DenseTokenAuditUnavailable:
                raise
            except Exception as exc:
                raise DenseTokenAuditUnavailable(
                    "The embedding model is unavailable for the token audit: "
                    + str(exc)
                ) from exc
        return self._audit_tokenizer

    def embedding_token_counts(self, texts: list[str]) -> list[int]:
        """Return the embedding tokenizer length of each text, untruncated."""

        if not texts:
            return []
        tokenizer = self._audit_tokenizer_for_ingestion()
        try:
            return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
        except Exception as exc:
            raise DenseTokenAuditUnavailable(str(exc)) from exc

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not texts:
            return np.empty((0, self.embedding_facts.dimension), dtype=np.float32)
        embedded = list(
            self._embedder().passage_embed(
                texts, batch_size=self.embedding_inference_batch_size
            )
        )
        vectors = np.asarray(embedded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (
            len(texts),
            self.embedding_facts.dimension,
        ):
            raise RuntimeError(
                f"Unexpected {self.embedding_facts.name} vector shape: {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("FastEmbed returned non-finite embedding values")
        return vectors

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]:
        if not documents:
            return []
        scores = list(
            self._cross_encoder(model).rerank(query, documents, batch_size=32)
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                "FastEmbed reranker returned a different number of scores than passages"
            )
        return [float(score) for score in scores]

    @staticmethod
    def _generation_root(index_path: Path) -> Path:
        root = index_path.parent.parent
        if not (root / "chunks" / "chunks.jsonl").is_file():
            raise ValueError(
                "An exact dense index must live at <generation>/indexes/<name>: "
                + str(index_path)
            )
        return root

    @staticmethod
    def _load_portable_vectors(
        vectors_path: Path,
        dimension: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot load portable embeddings: {vectors_path}"
            ) from exc
        if vectors.dtype != np.float32:
            raise ValueError("Portable embeddings must use float32 values")
        if vectors.ndim != 2 or vectors.shape[1] != dimension:
            raise ValueError("Portable embeddings have an invalid shape")
        for offset in range(0, vectors.shape[0], 4096):
            if not np.isfinite(vectors[offset : offset + 4096]).all():
                raise ValueError("Portable embeddings contain non-finite values")
        return vectors

    def initialize_index(self, index_path: Path, dimension: int) -> None:
        if dimension != self.embedding_facts.dimension:
            raise ValueError(
                f"An exact dense index requires {self.embedding_facts.dimension} dimensions, "
                f"not {dimension}"
            )
        if index_path.exists():
            raise ValueError(f"Exact dense index path already exists: {index_path}")
        index_path.mkdir(parents=True)
        self._pending_chunk_ids = []
        self._pending_document_ids = []

    def upload_index_batch(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
        *,
        offset: int,
    ) -> None:
        """Record per-row identity; the vectors are already the portable file."""

        if vectors.shape != (len(chunks), self.embedding_facts.dimension):
            raise ValueError("Exact dense index batch has an invalid vector shape")
        if offset != len(self._pending_chunk_ids):
            raise ValueError("Exact dense index batches must be uploaded in order")
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            if not chunk_id or not document_id:
                raise ValueError(
                    "Exact dense index chunks need a chunk_id and a document_id"
                )
            self._pending_chunk_ids.append(chunk_id)
            self._pending_document_ids.append(document_id)

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, Any]:
        if len(self._pending_chunk_ids) != expected_count:
            raise ValueError(
                f"Exact dense index expected {expected_count} chunks but "
                f"received {len(self._pending_chunk_ids)}"
            )
        atomic_write_json(
            index_path / EXACT_DOCUMENTS_FILENAME,
            {
                "schema_version": 1,
                "chunk_ids": self._pending_chunk_ids,
                "document_ids": self._pending_document_ids,
            },
        )
        atomic_write_json(
            index_path / EXACT_INDEX_FILENAME,
            {
                "schema_version": 1,
                "backend": EXACT_BACKEND_NAME,
                "dimension": dimension,
                "count": expected_count,
                "vectors": _EXACT_VECTORS_RELATIVE.as_posix(),
                "documents": EXACT_DOCUMENTS_FILENAME,
            },
        )
        self._pending_chunk_ids = []
        self._pending_document_ids = []
        self._loaded.pop(index_path.resolve(), None)
        self.validate_index(
            index_path,
            expected_count=expected_count,
            dimension=dimension,
        )
        return {
            "backend": "portable float32 vectors (exact cosine scan)",
            "dense_backend": EXACT_BACKEND_NAME,
            "collection": None,
            "distance": "cosine",
            "embedding_runtime": "FastEmbed ONNX Runtime (CPU)",
            "embedding_model": self.embedding_facts.name,
            "embedding_model_revision": self.embedding_facts.revision,
            "embedding_dimension": dimension,
            "point_count": expected_count,
        }

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]:
        expected_vectors = self._generation_root(index_path) / _EXACT_VECTORS_RELATIVE
        if vectors_path.resolve() != expected_vectors.resolve():
            raise ValueError(
                "An exact dense index expects its portable vectors at "
                + str(expected_vectors)
            )
        vectors = self._load_portable_vectors(
            vectors_path,
            self.embedding_facts.dimension,
        )
        if vectors.shape != (len(chunks), self.embedding_facts.dimension):
            raise ValueError(
                "Portable embedding dimensions do not match the chunk collection"
            )
        self.initialize_index(index_path, self.embedding_facts.dimension)
        self.upload_index_batch(chunks, index_path, vectors, offset=0)
        return self.finalize_index(
            index_path,
            expected_count=len(chunks),
            dimension=self.embedding_facts.dimension,
        )

    def _load_index(
        self,
        index_path: Path,
        *,
        expected_count: int | None = None,
        dimension: int | None = None,
    ) -> _ExactIndex:
        root = self._generation_root(index_path)
        descriptor = _read_index_object(
            index_path / EXACT_INDEX_FILENAME,
            "exact dense index descriptor",
        )
        if descriptor.get("schema_version") != 1:
            raise ValueError("Unsupported exact dense index schema")
        if descriptor.get("backend") != EXACT_BACKEND_NAME:
            raise ValueError("Exact dense index descriptor names another backend")
        index_dimension = descriptor.get("dimension")
        if index_dimension != self.embedding_facts.dimension:
            raise ValueError(
                f"An exact dense index requires {self.embedding_facts.dimension} dimensions"
            )
        if dimension is not None and index_dimension != dimension:
            raise ValueError(
                f"Exact dense index dimension {index_dimension} does not match "
                f"{dimension}"
            )
        documents_path = index_path / str(descriptor.get("documents") or "")
        documents = _read_index_object(
            documents_path,
            "exact dense index identity",
        )
        raw_chunk_ids = documents.get("chunk_ids")
        raw_document_ids = documents.get("document_ids")
        if not isinstance(raw_chunk_ids, list) or not isinstance(
            raw_document_ids, list
        ):
            raise TypeError("Exact dense index identity lists are missing")
        chunk_ids = tuple(str(item) for item in raw_chunk_ids)
        document_ids = tuple(str(item) for item in raw_document_ids)
        if not chunk_ids or len(chunk_ids) != len(document_ids):
            raise ValueError("Exact dense index identity lists are inconsistent")
        if expected_count is not None and len(chunk_ids) != expected_count:
            raise RuntimeError(
                f"Exact dense index expected {expected_count} rows, "
                f"found {len(chunk_ids)}"
            )
        vectors_path = root / str(descriptor.get("vectors") or "")
        if not vectors_path.is_file() or vectors_path.is_symlink():
            raise ValueError(f"Exact dense index vectors are missing: {vectors_path}")
        key = index_path.resolve()
        modified_ns = vectors_path.stat().st_mtime_ns
        cached = self._loaded.get(key)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]
        vectors = self._load_portable_vectors(
            vectors_path,
            self.embedding_facts.dimension,
        )
        if vectors.shape != (len(chunk_ids), self.embedding_facts.dimension):
            raise ValueError("Exact dense index vectors do not match its identity")
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        # A zero vector scores zero instead of dividing by zero.
        norms[norms == 0.0] = 1.0
        index = _ExactIndex(
            vectors=vectors,
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            norms=norms,
        )
        self._loaded[key] = (modified_ns, index)
        return index

    def validate_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> None:
        if not index_path.is_dir() or index_path.is_symlink():
            raise ValueError(f"Dense index is missing or unsafe: {index_path}")
        self._load_index(
            index_path,
            expected_count=expected_count,
            dimension=dimension,
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
        index = self._load_index(index_path)
        query_vectors = list(self._embedder().query_embed(query))
        if len(query_vectors) != 1:
            raise RuntimeError("FastEmbed did not return exactly one query vector")
        query_vector = np.asarray(query_vectors[0], dtype=np.float32)
        if query_vector.shape != (self.embedding_facts.dimension,):
            raise RuntimeError(
                f"Unexpected {self.embedding_facts.name} query vector shape: {query_vector.shape}"
            )
        query_norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_norm) or query_norm == 0.0:
            raise RuntimeError("FastEmbed returned a degenerate query vector")

        allowed = np.ones(len(index.chunk_ids), dtype=bool)
        if document_ids:
            wanted = set(document_ids)
            allowed &= np.fromiter(
                (item in wanted for item in index.document_ids),
                dtype=bool,
                count=len(index.document_ids),
            )
        if excluded_document_ids:
            blocked = set(excluded_document_ids)
            allowed &= np.fromiter(
                (item not in blocked for item in index.document_ids),
                dtype=bool,
                count=len(index.document_ids),
            )
        rows = np.flatnonzero(allowed)
        if rows.size == 0 or top_k <= 0:
            return []

        scores = (index.vectors @ query_vector) / (index.norms * query_norm)
        candidate_scores = scores[rows]
        # A stable sort keeps lower row order for equal scores.
        order = np.argsort(-candidate_scores, kind="stable")
        hits: list[DenseSearchHit] = []
        for position in order[:top_k]:
            score = float(candidate_scores[position])
            if not np.isfinite(score):
                raise RuntimeError("Exact dense search produced a non-finite score")
            row = int(rows[position])
            hits.append(DenseSearchHit(chunk_id=index.chunk_ids[row], score=score))
        return hits
