"""Exact dense search over a generation's portable float32 vectors.

The generation already stores one float32 row per chunk in stable chunk order, so
the index needs only a descriptor plus per-row identity, and search is an exact
cosine scan. That removes the whole-corpus index build and its per-point device
cost, and makes dense scoring exactly reproducible.

The index lives at `<generation>/indexes/<name>` and references the portable vector
file relative to the generation root, so a copied generation keeps a valid
descriptor.

The model loaders and the embedding, token-count, and rerank operations moved to
`model_runtime.py`, and the portable-vector file check moved to
`portable_vectors.py`; this module may never re-implement either, because two copies
of a check are two places for it to be wrong. It may also never hold one build's
rows on the instance, because the service keeps one instance per backend name for
the life of the process and a second build would interleave rows into the first.
Each uploaded batch's row identity is written beside the index under its offset,
so a build resumed by another process, or retried after a failure, finds them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...storage.records import StorageError, atomic_write_json, read_json
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModel
from ..model_runtime import ModelRuntime
from ..rerankers import DEFAULT_RERANKER_MODEL
from . import (
    EXACT_BACKEND_NAME,
    EXACT_DOCUMENTS_FILENAME,
    EXACT_INDEX_FILENAME,
    DenseSearchHit,
)
from .portable_vectors import load_portable_vectors

_EXACT_VECTORS_RELATIVE = Path("portable") / "embeddings.npy"
# Where a build in progress keeps each batch's row identity, named by its offset.
_ROWS_DIRECTORY = ".rows"


@dataclass(frozen=True, slots=True)
class _ExactIndex:
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
    """Exact dense search over a generation's portable float32 vectors."""

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
        self._loaded: dict[Path, tuple[int, _ExactIndex]] = {}

    def embedding_token_counts(self, texts: list[str]) -> list[int]:
        return self.runtime.embedding_token_counts(texts)

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self.runtime.embed_texts(texts)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]:
        return self.runtime.rerank(query, documents, model=model)

    @staticmethod
    def _generation_root(index_path: Path) -> Path:
        root = index_path.parent.parent
        if not (root / "chunks" / "chunks.jsonl").is_file():
            raise ValueError(
                "An exact dense index must live at <generation>/indexes/<name>: "
                + str(index_path)
            )
        return root

    def initialize_index(self, index_path: Path, dimension: int) -> None:
        if dimension != self.embedding_facts.dimension:
            raise ValueError(
                f"An exact dense index requires {self.embedding_facts.dimension} dimensions, "
                f"not {dimension}"
            )
        if index_path.exists():
            raise ValueError(f"Exact dense index path already exists: {index_path}")
        (index_path / _ROWS_DIRECTORY).mkdir(parents=True)

    def upload_index_batch(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
        *,
        offset: int,
    ) -> None:
        """Record one batch's row identity; the vectors are already the portable file.

        A batch is written under its offset, so uploading the same batch again
        replaces it rather than appending a second copy.
        """

        if vectors.shape != (len(chunks), self.embedding_facts.dimension):
            raise ValueError("Exact dense index batch has an invalid vector shape")
        rows: dict[str, list[str]] = {"chunk_ids": [], "document_ids": []}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            if not chunk_id or not document_id:
                raise ValueError(
                    "Exact dense index chunks need a chunk_id and a document_id"
                )
            rows["chunk_ids"].append(chunk_id)
            rows["document_ids"].append(document_id)
        atomic_write_json(index_path / _ROWS_DIRECTORY / f"{offset:012d}.json", rows)

    def _recorded_rows(self, index_path: Path) -> tuple[list[str], list[str]]:
        """Every uploaded batch's rows, in offset order, refusing a gap."""

        chunk_ids: list[str] = []
        document_ids: list[str] = []
        for batch in sorted((index_path / _ROWS_DIRECTORY).glob("*.json")):
            if int(batch.stem) != len(chunk_ids):
                raise ValueError(
                    f"Exact dense index batch {batch.name} does not start at row "
                    f"{len(chunk_ids)}"
                )
            rows = _read_index_object(batch, "exact dense index batch")
            chunk_ids.extend(str(item) for item in rows["chunk_ids"])
            document_ids.extend(str(item) for item in rows["document_ids"])
        return chunk_ids, document_ids

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, Any]:
        # A finalize that stopped after writing the identity resumes from it.
        if not (index_path / EXACT_DOCUMENTS_FILENAME).is_file():
            chunk_ids, document_ids = self._recorded_rows(index_path)
            if len(chunk_ids) != expected_count:
                raise ValueError(
                    f"Exact dense index expected {expected_count} chunks but "
                    f"received {len(chunk_ids)}"
                )
            atomic_write_json(
                index_path / EXACT_DOCUMENTS_FILENAME,
                {
                    "schema_version": 1,
                    "chunk_ids": chunk_ids,
                    "document_ids": document_ids,
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
        shutil.rmtree(index_path / _ROWS_DIRECTORY, ignore_errors=True)
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
            "embedding_model_repository": self.embedding_facts.repository,
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
        vectors = load_portable_vectors(
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
        vectors = load_portable_vectors(
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
        query_vectors = list(self.runtime._embedder().query_embed(query))
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
