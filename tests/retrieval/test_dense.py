from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import research_rag.retrieval.model_runtime as model_runtime_module
from research_rag.retrieval.dense import (
    DenseTokenAuditUnavailable,
    LocalQdrantDenseBackend,
    LocalVectorDenseBackend,
)
from research_rag.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    resolve_embedding_model,
)
from research_rag.retrieval.model_runtime import DEFAULT_RERANKER_MODEL

EMBEDDING_DIMENSION = resolve_embedding_model(DEFAULT_EMBEDDING_MODEL).dimension


def _chunks(count: int) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": f"chunk-{index}",
            "document_id": "document-1",
            "source_path": "sources/evidence.pdf",
            "categories": [],
            "keywords": [],
        }
        for index in range(count)
    ]


def test_embed_texts_uses_the_measured_inference_batch_size(tmp_path: Path) -> None:
    recorded: list[int] = []

    class StubEmbedder:
        def passage_embed(self, texts: list[str], batch_size: int) -> list[np.ndarray]:
            recorded.append(batch_size)
            return [np.zeros(EMBEDDING_DIMENSION, dtype=np.float32) for _ in texts]

    backend = LocalVectorDenseBackend(tmp_path / "models", offline=True)
    backend.runtime._embedding_model = StubEmbedder()  # type: ignore[assignment]

    vectors = backend.embed_texts(["alpha", "beta"])

    # Sequences are padded to the longest member of their inference batch, so the batch
    # size is a throughput decision, not a memory setting.
    assert recorded == [1]
    assert vectors.shape == (2, EMBEDDING_DIMENSION)

    configured = LocalVectorDenseBackend(
        tmp_path / "models",
        offline=True,
        embedding_inference_batch_size=4,
    )
    configured.runtime._embedding_model = StubEmbedder()  # type: ignore[assignment]
    configured.embed_texts(["alpha", "beta"])
    assert recorded == [1, 4]


def test_embedding_threads_reach_the_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_loader(
        cache_root: Path,
        *,
        offline: bool,
        threads: int | None = None,
        model: str = "",
    ) -> object:
        seen["cache_root"] = cache_root
        seen["offline"] = offline
        seen["model"] = model
        seen["threads"] = threads
        return object()

    # The loaders moved to `model_runtime`, and a backend reaches them through the
    # runtime it holds rather than through module-level names of its own.
    monkeypatch.setattr(model_runtime_module, "_load_embedder", fake_loader)

    configured = LocalVectorDenseBackend(
        tmp_path / "models",
        offline=True,
        embedding_threads=8,
    )
    configured.runtime._embedder()
    assert seen["threads"] == 8

    default = LocalVectorDenseBackend(tmp_path / "models", offline=True)
    default.runtime._embedder()
    assert seen["threads"] is None


def test_embedding_token_audit_reports_a_missing_model_cache(
    tmp_path: Path,
) -> None:
    # The audit must fail loudly rather than break ingestion, so a missing or offline model
    # cache raises a dedicated error the service degrades from.
    backend = LocalQdrantDenseBackend(tmp_path / "models", offline=True)

    with pytest.raises(DenseTokenAuditUnavailable):
        backend.embedding_token_counts(["some evidence text"])


def test_embedding_token_audit_accepts_an_empty_batch(tmp_path: Path) -> None:
    backend = LocalQdrantDenseBackend(tmp_path / "models", offline=True)

    assert backend.embedding_token_counts([]) == []


class _StubQueryEmbedder:
    """Stands in for FastEmbed so exact-search tests need no model cache."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def query_embed(self, query: str) -> list[np.ndarray]:
        return [self.vector]


def _unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[index] = 1.0
    return vector


def _exact_generation(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    root = tmp_path / "generation"
    (root / "chunks").mkdir(parents=True)
    (root / "chunks" / "chunks.jsonl").write_text("", encoding="utf-8")
    (root / "portable").mkdir(parents=True)
    chunks: list[dict[str, object]] = [
        {"chunk_id": "c0", "document_id": "doc-a"},
        {"chunk_id": "c1", "document_id": "doc-b"},
        {"chunk_id": "c2", "document_id": "doc-b"},
        {"chunk_id": "c3", "document_id": "doc-b"},
    ]
    diagonal = 0.70710678
    vectors = np.asarray(
        [
            _unit(0),
            _unit(1),
            [diagonal, diagonal] + [0.0] * (EMBEDDING_DIMENSION - 2),
            [-1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
        ],
        dtype=np.float32,
    )
    np.save(root / "portable" / "embeddings.npy", vectors)
    return root, root / "indexes" / "dense-exact", chunks


def test_exact_backend_builds_and_searches_by_cosine(tmp_path: Path) -> None:
    _root, index_path, chunks = _exact_generation(tmp_path)
    backend = LocalVectorDenseBackend(tmp_path / "models", offline=True)

    metadata = backend.build_from_vectors(
        chunks,
        index_path,
        index_path.parent.parent / "portable" / "embeddings.npy",
    )

    assert metadata["backend"] == "portable float32 vectors (exact cosine scan)"
    assert metadata["point_count"] == 4
    assert metadata["distance"] == "cosine"
    backend.validate_index(index_path, expected_count=4, dimension=384)

    backend.runtime._embedder = lambda: _StubQueryEmbedder(_unit(0))
    hits = backend.search(index_path, "anything", 4)

    assert [hit.chunk_id for hit in hits] == ["c0", "c2", "c1", "c3"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score == pytest.approx(0.70710678, abs=1e-6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)
    assert hits[3].score == pytest.approx(-1.0, abs=1e-6)


def test_two_builds_in_one_backend_do_not_append_to_each_other(tmp_path: Path) -> None:
    """One backend instance serves one process, and a build is per index path.

    The ingestion workflow holds one backend per name for the whole process and
    resumes a build from a checkpoint between phases, so the rows one build has
    recorded must never become another build's rows. Before the split those
    pending rows were instance attributes and a second build inherited them.
    """

    _root, first_path, first_chunks = _exact_generation(tmp_path)
    _root, second_path, second_chunks = _exact_generation(tmp_path / "second")
    second_chunks = [
        {**chunk, "chunk_id": f"x{chunk['chunk_id'][1:]}", "document_id": "doc-x"}
        for chunk in second_chunks
    ]
    backend = LocalVectorDenseBackend(tmp_path / "models", offline=True)
    vectors = np.zeros((4, EMBEDDING_DIMENSION), dtype=np.float32)

    backend.initialize_index(first_path, EMBEDDING_DIMENSION)
    backend.initialize_index(second_path, EMBEDDING_DIMENSION)
    # Two builds in flight at once, each uploading its rows out of step.
    backend.upload_index_batch(first_chunks[:2], first_path, vectors[:2], offset=0)
    backend.upload_index_batch(second_chunks[2:], second_path, vectors[2:], offset=0)
    backend.upload_index_batch(second_chunks[:2], second_path, vectors[:2], offset=2)
    backend.upload_index_batch(first_chunks[2:], first_path, vectors[2:], offset=2)
    backend.finalize_index(first_path, expected_count=4, dimension=EMBEDDING_DIMENSION)
    backend.finalize_index(second_path, expected_count=4, dimension=EMBEDDING_DIMENSION)

    first = json.loads((first_path / "documents.json").read_text(encoding="utf-8"))
    second = json.loads((second_path / "documents.json").read_text(encoding="utf-8"))
    assert first["chunk_ids"] == ["c0", "c1", "c2", "c3"]
    assert second["chunk_ids"] == ["x2", "x3", "x0", "x1"]


def test_exact_backend_applies_document_filters_and_top_k(tmp_path: Path) -> None:
    _root, index_path, chunks = _exact_generation(tmp_path)
    backend = LocalVectorDenseBackend(tmp_path / "models", offline=True)
    backend.build_from_vectors(
        chunks,
        index_path,
        index_path.parent.parent / "portable" / "embeddings.npy",
    )
    backend.runtime._embedder = lambda: _StubQueryEmbedder(_unit(0))

    restricted = backend.search(index_path, "q", 4, document_ids=["doc-a"])
    assert [hit.chunk_id for hit in restricted] == ["c0"]

    excluded = backend.search(index_path, "q", 4, excluded_document_ids=["doc-a"])
    assert [hit.chunk_id for hit in excluded] == ["c2", "c1", "c3"]

    # An empty filter list means "no filter", matching the Qdrant backend.
    # An empty filter list means "no filter", matching the Qdrant backend.
    unfiltered = backend.search(index_path, "q", 4, document_ids=[])
    assert len(unfiltered) == 4

    unknown = backend.search(index_path, "q", 4, document_ids=["doc-missing"])
    assert unknown == []

    assert [hit.chunk_id for hit in backend.search(index_path, "q", 2)] == ["c0", "c2"]


def test_exact_backend_rejects_inconsistent_or_misplaced_indexes(
    tmp_path: Path,
) -> None:
    root, index_path, chunks = _exact_generation(tmp_path)
    backend = LocalVectorDenseBackend(tmp_path / "models", offline=True)
    vectors_path = root / "portable" / "embeddings.npy"
    backend.build_from_vectors(chunks, index_path, vectors_path)

    with pytest.raises(RuntimeError, match="expected 5 rows"):
        backend.validate_index(index_path, expected_count=5, dimension=384)
    with pytest.raises(ValueError, match="missing or unsafe"):
        backend.validate_index(
            tmp_path / "absent",
            expected_count=4,
            dimension=384,
        )

    descriptor_path = index_path / "index.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["vectors"] = "portable/absent.npy"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="vectors are missing"):
        backend.validate_index(index_path, expected_count=4, dimension=384)

    with pytest.raises(ValueError, match="must live at"):
        backend.build_from_vectors(chunks, tmp_path / "elsewhere", vectors_path)


def test_local_qdrant_batches_resume_without_skips_or_duplicates(
    tmp_path: Path,
) -> None:
    chunks = _chunks(130)
    vectors = np.ones((130, 384), dtype=np.float32)
    index_path = tmp_path / "qdrant"
    first_process = LocalQdrantDenseBackend(tmp_path / "models", offline=True)
    first_process.initialize_index(index_path, 384)
    first_process.upload_index_batch(
        chunks[:64],
        index_path,
        vectors[:64],
        offset=0,
    )

    resumed_process = LocalQdrantDenseBackend(tmp_path / "models", offline=True)
    resumed_process.upload_index_batch(
        chunks[64:128],
        index_path,
        vectors[64:128],
        offset=64,
    )
    resumed_process.upload_index_batch(
        chunks[128:],
        index_path,
        vectors[128:],
        offset=128,
    )
    resumed_process.upload_index_batch(
        chunks[64:128],
        index_path,
        vectors[64:128],
        offset=64,
    )

    metadata = resumed_process.finalize_index(
        index_path,
        expected_count=130,
        dimension=384,
    )
    assert metadata["point_count"] == 130


def test_rerank_uses_the_configured_model_and_switches_only_on_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    class StubEncoder:
        def rerank(
            self,
            query: str,
            documents: list[str],
            batch_size: int,
        ) -> list[float]:
            return [float(len(document)) for document in documents]

    def fake_loader(cache_root: Path, *, offline: bool, model: str = "") -> object:
        loaded.append(model)
        return StubEncoder()

    # The loader moved to `model_runtime`, so the stub replaces the name there.
    monkeypatch.setattr(model_runtime_module, "_load_cross_encoder", fake_loader)
    backend = LocalVectorDenseBackend(
        tmp_path / "models",
        offline=True,
        reranker_model="jinaai/jina-reranker-v1-turbo-en",
    )

    assert backend.rerank("cobalt", ["alpha", "beta"]) == [5.0, 4.0]
    assert backend.rerank("cobalt", ["alpha"], model="BAAI/bge-reranker-base") == [5.0]
    assert backend.rerank("cobalt", ["alpha"]) == [5.0]
    assert loaded == ["jinaai/jina-reranker-v1-turbo-en", "BAAI/bge-reranker-base"]


def test_cross_encoder_defaults_to_the_pinned_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_loader(cache_root: Path, *, offline: bool, model: str = "") -> object:
        seen["model"] = model
        seen["cache_root"] = cache_root
        return object()

    monkeypatch.setattr(model_runtime_module, "_load_cross_encoder", fake_loader)
    backend = LocalQdrantDenseBackend(tmp_path / "models", offline=True)

    backend.runtime._cross_encoder()

    assert seen["model"] == DEFAULT_RERANKER_MODEL
    assert seen["cache_root"] == tmp_path / "models"
