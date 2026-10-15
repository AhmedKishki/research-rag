"""Inspectable retrieval fakes and explicit service construction for offline tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from research_rag.core.service import ResearchService
from research_rag.project.config import ResearchConfig, resolve_config
from research_rag.project.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    resolve_embedding_model,
)
from research_rag.retrieval.dense import (
    DenseSearchHit,
    DenseTokenAuditUnavailable,
    RerankerUnavailable,
)
from research_rag.storage.durable_io import read_jsonl, write_jsonl

DEFAULT_EMBEDDING_FACTS = resolve_embedding_model(DEFAULT_EMBEDDING_MODEL)


class FakeUltraRAG:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.initialized: tuple[Path, Path] | None = None
        self.chunk_calls = 0
        self.bm25_build_calls = 0
        self.bm25_language = ""

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        self.chunk_calls += 1
        units = read_jsonl(input_path)
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": item["id"],
                    "title": item["title"],
                    "contents": item["contents"],
                }
                for index, item in enumerate(units)
            ),
        )

    async def build_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        self.bm25_language = language
        self.bm25_build_calls += 1
        index_path.mkdir(parents=True)
        (index_path / "fake-index.json").write_text("{}\n", encoding="utf-8")
        self.passages = [item["contents"] for item in read_jsonl(chunks_path)]
        self.initialized = (chunks_path, index_path)

    async def initialize_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        self.bm25_language = language
        index_file = index_path / "fake-index.json"
        if (
            not index_file.is_file()
            or json.loads(index_file.read_text(encoding="utf-8")) != {}
        ):
            raise RuntimeError("fake BM25 index is missing")
        self.passages = [
            str(item.get("contents") or item.get("text") or "")
            for item in read_jsonl(chunks_path)
        ]
        self.initialized = (chunks_path, index_path)

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        terms = query.casefold().split()

        def score(passage: str) -> int:
            value = passage.casefold()
            return sum(value.count(term) for term in terms)

        return sorted(self.passages, key=score, reverse=True)[:top_k]


class FakeDenseBackend:
    def __init__(self, dense_backend: str = "portable-exact-vectors") -> None:
        self.chunks: list[dict[str, object]] = []
        # Mirrors what `auto` selects for a small corpus so manifest assertions
        # match the real default. Pass a name to emulate the other backend.
        self.dense_backend = dense_backend
        self.fail_build = False
        self.build_calls = 0
        self.embed_calls = 0
        self.embedded_text_count = 0
        # The exact strings a vector was created from, so a test can assert what
        # the dense half was given rather than only how much of it there was.
        self.embedded_texts: list[str] = []
        self.upload_batches: list[tuple[int, int]] = []
        self.audit_unavailable = False
        self.audited_text_count = 0
        self.token_count_override: int | None = None
        # One entry per rerank call: the model the caller named, or None when it
        # left the choice to the engine.
        self.rerank_models: list[str | None] = []

    def embedding_token_counts(self, texts: list[str]) -> list[int]:
        if self.audit_unavailable:
            raise DenseTokenAuditUnavailable("simulated missing tokenizer")
        self.audited_text_count += len(texts)
        if self.token_count_override is not None:
            return [self.token_count_override for _ in texts]
        # Deterministic stand-in for the real tokenizer: one token per word.
        return [len(text.split()) for text in texts]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        self.embedded_text_count += len(texts)
        self.embedded_texts.extend(texts)
        return np.ones((len(texts), 384), dtype=np.float32)

    def _build_index(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors: np.ndarray,
    ) -> dict[str, object]:
        self.build_calls += 1
        if self.fail_build:
            raise RuntimeError("simulated dense-index failure")
        assert vectors.shape == (len(chunks), 384)
        index_path.mkdir(parents=True)
        (index_path / "fake-qdrant.json").write_text(
            json.dumps({"point_count": len(chunks), "dimension": 384}) + "\n",
            encoding="utf-8",
        )
        self.chunks = chunks
        return {
            "backend": "fake Qdrant",
            "embedding_model": DEFAULT_EMBEDDING_FACTS.name,
            "embedding_model_revision": DEFAULT_EMBEDDING_FACTS.revision,
            "embedding_dimension": 384,
            "point_count": len(chunks),
        }

    def initialize_index(self, index_path: Path, dimension: int) -> None:
        self.build_calls += 1
        if self.fail_build:
            raise RuntimeError("simulated dense-index failure")
        assert dimension == 384
        index_path.mkdir(parents=True)
        self.chunks = []

    def upload_index_batch(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors: np.ndarray,
        *,
        offset: int,
    ) -> None:
        assert index_path.is_dir()
        assert vectors.shape == (len(chunks), 384)
        assert offset == len(self.chunks)
        self.upload_batches.append((offset, len(chunks)))
        self.chunks.extend(chunks)

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, object]:
        assert index_path.is_dir()
        assert dimension == 384
        assert len(self.chunks) == expected_count
        (index_path / "fake-qdrant.json").write_text(
            json.dumps({"point_count": expected_count, "dimension": dimension}) + "\n",
            encoding="utf-8",
        )
        return {
            "backend": "fake Qdrant",
            "dense_backend": self.dense_backend,
            "embedding_model": DEFAULT_EMBEDDING_FACTS.name,
            "embedding_model_revision": DEFAULT_EMBEDDING_FACTS.revision,
            "embedding_dimension": 384,
            "point_count": expected_count,
        }

    def validate_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> None:
        metadata = json.loads(
            (index_path / "fake-qdrant.json").read_text(encoding="utf-8")
        )
        if metadata != {"point_count": expected_count, "dimension": dimension}:
            raise RuntimeError("fake dense index validation failed")

    def build_from_vectors(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, object]:
        vectors = np.load(vectors_path, allow_pickle=False)
        assert vectors.shape == (len(chunks), 384)
        return self._build_index(chunks, index_path, vectors)

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        *,
        document_ids: list[str] | None = None,
        excluded_document_ids: list[str] | None = None,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        document_filter = set(document_ids or [])
        excluded_document_filter = set(excluded_document_ids or [])
        terms = query.casefold().split()
        scored: list[tuple[float, str]] = []
        for chunk in self.chunks:
            if document_filter and chunk["document_id"] not in document_filter:
                continue
            if chunk["document_id"] in excluded_document_filter:
                continue
            text = str(chunk["contents"]).casefold()
            score = float(sum(text.count(term) for term in terms))
            scored.append((score, str(chunk["chunk_id"])))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            DenseSearchHit(chunk_id=chunk_id, score=score)
            for score, chunk_id in scored[:top_k]
        ]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]:
        self.rerank_models.append(model)
        return [
            float(document.casefold().count(query.split()[0].casefold()))
            for document in documents
        ]


class ProgressivelyFilteredUltraRAG(FakeUltraRAG):
    def __init__(self) -> None:
        super().__init__()
        self.search_depths: list[int] = []

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        self.chunk_calls += 1
        unit = read_jsonl(input_path)[0]
        title = str(unit["title"])
        count = 30 if title == "Distractor" else 5
        emphasis = "cobalt cobalt" if title == "Distractor" else "cobalt"
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": unit["id"],
                    "title": title,
                    "contents": f"{emphasis} {title} evidence {index}.",
                }
                for index in range(count)
            ),
        )

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        self.search_depths.append(top_k)
        return await super().search_bm25(query, top_k)


class UnavailableRerankerDenseBackend(FakeDenseBackend):
    """A dense backend whose reranker model cannot be loaded."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
    ) -> list[float]:
        raise RerankerUnavailable(
            "The reranker model is not present in the shared model cache."
        )


def make_service(
    project: Path | ResearchConfig,
    *,
    lexical: FakeUltraRAG | None = None,
    dense: FakeDenseBackend | None = None,
) -> ResearchService:
    """Construct fresh fakes without writing sources or starting ingestion.

    Pass a config when its settings or reuse across restarts are under test.
    Pass backend instances when a test inspects them or exercises a variant.
    """

    config = (
        project
        if isinstance(project, ResearchConfig)
        else resolve_config(project, vanilla_executable=sys.executable)
    )
    return ResearchService(
        config,
        lexical if lexical is not None else FakeUltraRAG(),  # type: ignore[arg-type]
        dense=dense if dense is not None else FakeDenseBackend(),
    )
