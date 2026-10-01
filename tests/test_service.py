from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import write_epub, write_pdf, write_reviewed_metadata
from filelock import AsyncFileLock

import research_rag.artifact_lookup as artifact_lookup_module
import research_rag.ingestion as ingestion_module
import research_rag.service as service_module
import research_rag.status as status_module
import research_rag.support as support_module
from research_rag.config import (
    ConfigurationError,
    ResearchConfig,
    resolve_config,
)
from research_rag.dense import (
    DenseSearchHit,
    DenseTokenAuditUnavailable,
    RerankerUnavailable,
)
from research_rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    resolve_embedding_model,
)
from research_rag.extraction import ExtractionError
from research_rag.rerankers import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODELS,
)
from research_rag.service import (
    ResearchError,
    ResearchService,
    _citation,
    _effective_document_metadata,
    _enrich_chunks,
    _public_document,
)
from research_rag.settings import resolve_gate_stopwords
from research_rag.storage import atomic_write_json, read_json, read_jsonl, write_jsonl

# The default model's own facts, resolved rather than hard-coded.
DEFAULT_EMBEDDING_FACTS = resolve_embedding_model(DEFAULT_EMBEDDING_MODEL)

# The function words the English lexical gate ignores, resolved the way a
# search resolves them, so the helper tests exercise the real rule.
ENGLISH_GATE_STOPWORDS = resolve_gate_stopwords(("en",))

CORRUPT_TEXT = (
    "��ѪҶޜഝǄ䘉Ӌਁ ⧠ᢃ⹤Ҷἅ ൠ؞༽൷㜭ᡀ࣏ "
    "䖜රѪտᆵǃ୶ъㅹ儈ԧ٬ъᘱⲴ⡷䶒ਉһǄ൘ᡰᴹᵳ╄ਈᯩ "
    "䶒ˈབྷཊᮠ൪ൠ൘䗷 ৫ഋॱᒤѝ࿻㓸؍ᤱ⿱ᴹᡆޜᴹ኎ᙗǄ "
    "ᵜ᮷䘈ᇎ䇱ਁ ⧠ˈ䜘࠶൪ൠᡰᴹᵳ⭡⊑ḃර⿱㩕Աъࡂ䖜㠣᭯ "
    "ᓌˈ⭡᭯ ᓌ᢯ᣵޘ䜘؞༽䍴䠁ˈᴰ㓸䙐ᡀ⊑ḃ⋫⨶ᡀᵜ⽮Պॆ "
    "ˈᒦᕅਁ ⧟ຳнޜǄᴹ∂ᓏ⢙൪ൠ 䳮�"
)


class FakeUltraRAG:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.initialized: tuple[Path, Path] | None = None
        self.chunk_calls = 0
        self.bm25_build_calls = 0

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
        # The language is recorded so a test can prove the setting reaches the
        # gateway rather than being dropped in transit.
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
        # The language is recorded so a test can prove the setting reaches the
        # gateway rather than being dropped in transit, on the load path too.
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


class SimulatedProcessExit(BaseException):
    """Models a hard process exit that no handler in the service can catch."""


class CrashDuringChunkUltraRAG(FakeUltraRAG):
    """Records the units each chunk call was given and can fail one call."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.requested_unit_ids: list[list[str]] = []

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.requested_unit_ids.append(
            [str(unit["id"]) for unit in read_jsonl(input_path)]
        )
        if self.fail_on_call is not None and len(self.requested_unit_ids) == (
            self.fail_on_call
        ):
            raise SimulatedProcessExit
        await super().chunk(
            input_path,
            output_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


class SymbolChunkUltraRAG(FakeUltraRAG):
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
        write_jsonl(
            output_path,
            [
                {
                    "id": 0,
                    "doc_id": unit["id"],
                    "title": unit["title"],
                    "contents": unit["contents"],
                },
                {
                    "id": 1,
                    "doc_id": unit["id"],
                    "title": unit["title"],
                    "contents": "— • ∎",
                },
            ],
        )


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


class ScoredDenseBackend(FakeDenseBackend):
    def __init__(self, score: float) -> None:
        super().__init__()
        self.score = score

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        **_filters: object,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        return [
            DenseSearchHit(chunk_id=str(chunk["chunk_id"]), score=self.score)
            for chunk in self.chunks[:top_k]
        ]


class BandedDenseBackend(FakeDenseBackend):
    """Scores the first chunk above the floor and the rest inside a band below it.

    A one-word query behaves this way: the whole candidate list sits a few
    hundredths under the floor, so an absolute gate slices it arbitrarily.
    """

    def __init__(self, strong: float, band: float) -> None:
        super().__init__()
        self.strong = strong
        self.band = band

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        **_filters: object,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        return [
            DenseSearchHit(
                chunk_id=str(chunk["chunk_id"]),
                score=self.strong if position == 0 else self.band,
            )
            for position, chunk in enumerate(self.chunks[:top_k])
        ]


class CancelOnceUltraRAG(FakeUltraRAG):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_next_chunk = True

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if self.cancel_next_chunk:
            self.cancel_next_chunk = False
            raise asyncio.CancelledError
        await super().chunk(
            input_path,
            output_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


class ManyChunkUltraRAG(FakeUltraRAG):
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
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": unit["id"],
                    "title": unit["title"],
                    "contents": f"Research evidence passage number {index}.",
                }
                for index in range(130)
            ),
        )


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


def test_unsearchable_upstream_chunks_are_counted_without_losing_a_unit() -> None:
    document = {
        "document_id": "doc_test",
        "source_id": "src_test",
        "source_path": "sources/test.pdf",
        "title": "Test",
        "authors": [],
        "year": None,
        "doi": "",
        "categories": [],
        "keywords": [],
    }
    unit = {
        "id": "doc_test:pdf-page:000001",
        "document_id": "doc_test",
        "source_id": "src_test",
        "locator": {"type": "pdf_page", "page": 1, "page_label": "1"},
    }
    chunks, discarded_empty, discarded_symbol_only, discarded_corrupt = _enrich_chunks(
        [
            {"doc_id": unit["id"], "contents": "Evidence remains searchable."},
            {"doc_id": unit["id"], "contents": ""},
            {"doc_id": unit["id"], "contents": "— • ∎"},
            {"doc_id": unit["id"], "contents": "W–(M–C–M′)–W′"},
            {"doc_id": unit["id"], "contents": "2026 ± 4"},
            {"doc_id": unit["id"], "contents": CORRUPT_TEXT},
        ],
        [unit],
        [document],
    )
    assert [chunk["contents"] for chunk in chunks] == [
        "Evidence remains searchable.",
        "W–(M–C–M′)–W′",
        "2026 ± 4",
    ]
    assert discarded_empty == 1
    assert discarded_symbol_only == 1
    assert discarded_corrupt == 1

    second_unit = {
        "id": "doc_test:pdf-page:000002",
        "document_id": "doc_test",
        "source_id": "src_test",
        "locator": {"type": "pdf_page", "page": 2, "page_label": "2"},
    }
    mixed_chunks, _, mixed_symbol_only, _ = _enrich_chunks(
        [
            {"doc_id": unit["id"], "contents": "— • ∎"},
            {"doc_id": second_unit["id"], "contents": "Other page evidence."},
        ],
        [unit, second_unit],
        [document],
    )
    assert [chunk["contents"] for chunk in mixed_chunks] == ["Other page evidence."]
    assert mixed_symbol_only == 1

    with pytest.raises(ResearchError, match="No searchable chunks were produced"):
        _enrich_chunks(
            [{"doc_id": unit["id"], "contents": "— • ∎"}],
            [unit],
            [document],
        )


def test_legacy_automatic_metadata_uses_runtime_corruption_guard() -> None:
    legacy = {
        "source_path": "sources/safe-name.pdf",
        "title": CORRUPT_TEXT,
        "authors": [CORRUPT_TEXT, "Ada Example"],
        "categories": [],
        "keywords": [],
        "doi": "",
        "metadata_provenance": {
            "title": "pdf_metadata",
            "authors": "pdf_metadata",
        },
        "metadata_warnings": [],
    }

    public = _public_document(legacy)

    assert public["title"] == "safe-name"
    assert public["authors"] == ["Ada Example"]
    assert "corrupt_extracted_title" in public["metadata_warnings"]
    assert "corrupt_extracted_authors" in public["metadata_warnings"]

    legacy["title"] = "— • ∎"
    legacy["authors"] = ["— • ∎", "Ada Example"]
    symbol_only = _public_document(legacy)
    assert symbol_only["title"] == "safe-name"
    assert symbol_only["authors"] == ["Ada Example"]

    legacy["metadata_provenance"] = {
        "title": "reviewed_override",
        "authors": "reviewed_override",
    }
    reviewed = _public_document(legacy)
    assert reviewed["title"] == "— • ∎"
    assert reviewed["authors"][0] == "— • ∎"


def test_epub_citation_prefers_exact_existing_fragment() -> None:
    document = {
        "title": "Anchored Book",
        "authors": ["Ada Example"],
        "year": 2026,
    }
    locator = {
        "type": "epub_section",
        "section_index": 4,
        "section_title": "Long Chapter",
        "href": "chapter-4.xhtml",
        "href_with_fragment": "chapter-4.xhtml#paragraph-12",
    }

    assert _citation(document, locator).endswith("section chapter-4.xhtml#paragraph-12")


async def _assert_research_generation_and_structured_search(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        [
            "Cobalt evidence about labour and artificial intelligence.",
            "Quartz material unrelated to the primary question.",
        ],
        title="Research Article",
    )
    (project / "sources" / "notes.md").write_text(
        "cobalt cobalt should never be indexed",
        encoding="utf-8",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(config, fake, dense=dense)  # type: ignore[arg-type]

    write_reviewed_metadata(
        config,
        "article.pdf",
        {
            "authors": ["Researcher One"],
            "year": 2026,
            "categories": ["political economy"],
            "keywords": ["labour", "AI"],
        },
    )

    result = await service.ingest(chunk_size=100, chunk_overlap=10)
    assert result["document_count"] == 1
    assert result["pdf_count"] == 1
    assert result["epub_count"] == 0
    assert result["ignored_extensions"] == {".md": 1}
    assert result["discarded_empty_chunk_count"] == 0
    assert result["default_retrieval_method"] == "hybrid"
    assert set(result["available_retrieval_methods"]) == {"bm25", "dense", "hybrid"}

    status = await service.status()
    assert status["ready"] is True
    assert status["stale"] is False
    assert status["hybrid_ready"] is True
    assert status["hybrid_upgrade_required"] is False
    assert status["generation_upgrade_required"] is False
    assert status["model_cache_root"] == str(config.model_cache_root)
    assert status["last_build_metrics"]["created_vector_count"] == result["chunk_count"]

    sources = await service.list_sources()
    assert sources["source_count"] == 1
    assert sources["sources"][0]["source_path"] == "sources/article.pdf"
    assert sources["sources"][0]["categories"] == ["political economy"]

    generation_root = Path(result["generation_root"])
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert "raw_extraction" not in manifest["files"]
    assert "raw_ultrarag_chunks" not in manifest["files"]
    assert not (generation_root / "work").exists()
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    stored_chunks = read_jsonl(chunks_path)
    stored_chunks[0]["text"] = (
        "Cobalt evidence about\nlabour and artificial\nintelligence."
    )
    write_jsonl(chunks_path, stored_chunks)

    search = await service.search(
        "cobalt labour",
        top_k=1,
        categories_any=["political economy"],
    )
    hit = search["hits"][0]
    assert hit["source_path"] == "sources/article.pdf"
    assert hit["locator"]["type"] == "pdf_page"
    assert hit["locator"]["page"] == 1
    assert "Cobalt evidence" in hit["text"]
    assert hit["text"] == ("Cobalt evidence about labour and artificial intelligence.")
    assert "p. 1" in hit["citation"]
    assert search["retrieval_method"] == "hybrid"
    assert hit["component_ranks"]["bm25"] == 1
    assert hit["component_ranks"]["dense"] == 1
    assert hit["fusion_score"] is not None
    assert hit["direct_quote_safe"] is False
    assert hit["text_fidelity"] == "cleaned_semantic_text"
    assert hit["content_kind"] == "prose"
    assert hit["metadata_provenance"]["authors"] == "reviewed_override"
    assert search["requested_top_k"] == 1
    assert search["relevance_limited"] is False
    assert search["relevance_policy"]["dense_minimum_cosine_similarity"] == 0.72
    # No legacy "notes" field survives; text and script state use explicit names.
    assert "notes" not in search
    assert "notes" not in hit
    assert hit["text_notes"] == []

    dense_search = await service.search(
        "cobalt labour",
        top_k=1,
        categories_any=["political economy"],
        retrieval_method="dense",
        rerank=True,
    )
    assert dense_search["retrieval_method"] == "dense"
    assert dense_search["reranked"] is True
    assert dense_search["hits"][0]["rerank_score"] is not None

    passage = await service.get_passage(hit["chunk_id"], context_chunks=1)
    assert passage["requested_chunk_id"] == hit["chunk_id"]
    assert len(passage["context"]) == 2
    assert passage["context"][0]["text"] == hit["text"]

    manifest_path = generation_root / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["retrieval"]["relevance_gates"][
        "dense_minimum_cosine_similarity"
    ] = 0.71
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    policy_status = await service.status()
    assert policy_status["generation_upgrade_required"] is True
    assert "retrieval_policy" in policy_status["upgrade_reasons"]

    legacy_manifest["schema_version"] = 1
    legacy_manifest["retrieval"] = {
        "backend": "UltraRAG BM25",
        "language": "en",
        "tokenizer": "default",
    }
    legacy_manifest["files"].pop("dense_index")
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_status = await service.status()
    assert legacy_status["hybrid_upgrade_required"] is True
    assert legacy_status["generation_upgrade_required"] is True
    assert "generation_schema" in legacy_status["upgrade_reasons"]
    with pytest.raises(ResearchError, match="does not support 'hybrid'"):
        await service.search("cobalt", top_k=1)
    legacy_search = await service.search(
        "cobalt",
        top_k=1,
        retrieval_method="bm25",
    )
    assert legacy_search["hits"][0]["retrieval_method"] == "bm25"

    write_pdf(project / "sources" / "new.pdf", ["A newly added source."])
    stale = await service.status()
    assert stale["stale"] is True
    assert stale["changes"]["added"] == ["new.pdf"]


def test_research_generation_and_structured_search(project: Path) -> None:
    asyncio.run(_assert_research_generation_and_structured_search(project))


async def _assert_search_can_skip_the_staleness_walk(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Cobalt evidence about labour."], title="Research Article")
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=100, chunk_overlap=10)

    checked = await service.search("cobalt labour", top_k=1)
    assert checked["staleness_checked"] is True
    assert checked["stale"] is False

    write_pdf(project / "sources" / "added.pdf", ["Another source."])
    stale = await service.search("cobalt labour", top_k=1)
    assert stale["stale"] is True
    unchecked = await service.search("cobalt labour", top_k=1, include_staleness=False)
    assert unchecked["staleness_checked"] is False
    assert unchecked["stale"] is None
    assert unchecked["hits"]
    assert unchecked["generation_id"] == stale["generation_id"]

    def unexpected_walk(_config: ResearchConfig) -> None:
        raise AssertionError("search walked the source tree")

    monkeypatch.setattr(status_module, "scan_sources", unexpected_walk)
    with pytest.raises(AssertionError):
        await service.search("cobalt labour", top_k=1)
    no_walk = await service.search("cobalt labour", top_k=1, include_staleness=False)
    assert no_walk["hits"]
    assert no_walk["stale"] is None

    # The policy fingerprint is an instance attribute now, computed from the
    # settings this service resolved; patching it stands in for a generation that
    # recorded a different policy.
    monkeypatch.setattr(
        service,
        "retrieval_policy_fingerprint",
        "patched-retrieval-policy",
    )
    upgraded = await service.search("cobalt labour", top_k=1, include_staleness=False)
    assert upgraded["generation_upgrade_required"] is True
    assert upgraded["staleness_checked"] is False
    assert upgraded["stale"] is None


def test_search_can_skip_the_staleness_walk(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_assert_search_can_skip_the_staleness_walk(project, monkeypatch))


async def _assert_reviewed_metadata_is_a_runtime_overlay(project: Path) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(
        source,
        ["Cobalt evidence remains searchable after metadata correction."],
        title="Automatic Source Title",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )

    write_reviewed_metadata(
        config,
        "article.pdf",
        {
            "title": "Incorrect Reviewed Title",
            "authors": ["Incorrect Author"],
            "year": 1999,
            "doi": "10.1000/incorrect",
            "categories": ["old category"],
            "keywords": ["old keyword"],
        },
    )
    pending_status = await service.status()
    assert pending_status["metadata_overlay_active"] is False
    assert pending_status["generation_metadata_snapshot_outdated"] is False

    ingested = await service.ingest(chunk_size=100, chunk_overlap=10)
    generation_root = Path(ingested["generation_root"])
    current_before = config.current_path.read_bytes()
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    chunks_before = chunks_path.read_bytes()
    generation_before = {
        path.relative_to(generation_root).as_posix(): path.read_bytes()
        for path in generation_root.rglob("*")
        if path.is_file()
    }

    write_reviewed_metadata(
        config,
        "article.pdf",
        {
            "title": "Corrected\nReviewed Title",
            "authors": ["Correct\nAuthor"],
            "year": 2026,
            "doi": "10.1000/corrected",
            "categories": ["new\ncategory"],
            "keywords": ["new\tkeyword"],
            "project": ["new\nproject"],
        },
    )
    assert config.current_path.read_bytes() == current_before
    assert chunks_path.read_bytes() == chunks_before
    assert {
        path.relative_to(generation_root).as_posix(): path.read_bytes()
        for path in generation_root.rglob("*")
        if path.is_file()
    } == generation_before

    status = await service.status()
    assert status["stale"] is False
    assert status["metadata_overlay_active"] is True
    assert status["changes"]["metadata_changed"] is True
    assert status["generation_id"] == ingested["generation_id"]

    sources = await service.list_sources()
    assert sources["source_count"] == 1
    listed = sources["sources"][0]
    assert listed["title"] == "Corrected Reviewed Title"
    assert listed["authors"] == ["Correct Author"]
    assert listed["year"] == 2026
    assert listed["doi"] == "10.1000/corrected"
    assert listed["categories"] == ["new category"]
    assert listed["keywords"] == ["new keyword"]

    old_dense = await service.search(
        "cobalt evidence",
        categories_any=["old category"],
        keywords=["old keyword"],
        retrieval_method="dense",
    )
    assert old_dense["hits"] == []
    search = await service.search(
        "cobalt evidence",
        categories_any=["new category"],
        keywords=["new keyword"],
        retrieval_method="dense",
    )
    hit = search["hits"][0]
    assert hit["title"] == "Corrected Reviewed Title"
    assert hit["authors"] == ["Correct Author"]
    assert hit["year"] == 2026
    assert hit["doi"] == "10.1000/corrected"
    assert hit["categories"] == ["new category"]
    assert hit["keywords"] == ["new keyword"]
    assert "Correct Author, Corrected Reviewed Title (2026)" in hit["citation"]
    assert "doi:10.1000/corrected" in hit["citation"]

    passage = await service.get_passage(hit["chunk_id"])
    context = passage["context"][0]
    assert context["title"] == "Corrected Reviewed Title"
    assert context["authors"] == ["Correct Author"]
    assert context["categories"] == ["new category"]
    assert context["keywords"] == ["new keyword"]
    assert "doi:10.1000/corrected" in context["citation"]
    assert config.current_path.read_bytes() == current_before
    assert chunks_path.read_bytes() == chunks_before

    # Explicit empty values authoritatively clear wrong automatic fields. An
    # empty object removes the complete override and restores automatic values.
    write_reviewed_metadata(
        config,
        "article.pdf",
        {
            "title": "",
            "authors": [],
            "year": None,
            "doi": "",
            "categories": [],
            "keywords": [],
        },
    )
    explicitly_cleared = (await service.list_sources())["sources"][0]
    assert explicitly_cleared["title"] == ""
    assert explicitly_cleared["authors"] == []
    assert explicitly_cleared["year"] is None
    assert explicitly_cleared["doi"] == ""
    assert (await service.status())["metadata_overlay_active"] is True

    write_reviewed_metadata(config, "article.pdf", {})
    fallback = (await service.list_sources())["sources"][0]
    assert fallback["title"] == "Automatic Source Title"
    assert fallback["authors"] == ["Test Author"]
    assert fallback["year"] is None
    assert fallback["doi"] == ""
    assert fallback["categories"] == []
    assert fallback["keywords"] == []
    assert (
        "automatic_metadata_unavailable_after_override_removal"
        not in fallback["metadata_warnings"]
    )
    cleared_status = await service.status()
    assert cleared_status["metadata_overlay_active"] is False
    assert cleared_status["generation_metadata_snapshot_outdated"] is True
    assert json.loads(config.metadata_path.read_text(encoding="utf-8"))["sources"] == {}
    assert config.current_path.read_bytes() == current_before
    assert chunks_path.read_bytes() == chunks_before
    assert {
        path.relative_to(generation_root).as_posix(): path.read_bytes()
        for path in generation_root.rglob("*")
        if path.is_file()
    } == generation_before


def test_reviewed_metadata_is_a_runtime_overlay(project: Path) -> None:
    asyncio.run(_assert_reviewed_metadata_is_a_runtime_overlay(project))


async def _assert_source_ids_address_reviewed_sources(project: Path) -> None:
    source_path = project / "sources" / "article.pdf"
    write_pdf(source_path, ["Stable cobalt evidence for source identity."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    discovered = await service.list_sources()
    assert discovered["ready"] is False
    source_id = discovered["discovered_sources"][0]["source_id"]
    assert source_id.startswith("src_")
    assert discovered["known_sources"] == [
        {
            "source_id": source_id,
            "source_relative_path": "article.pdf",
            "exists": True,
            "included": True,
            "indexed_in_current_generation": False,
            "has_reviewed_metadata": False,
        }
    ]
    write_reviewed_metadata(config, "article.pdf", {"title": "Reviewed by ID"})

    with pytest.raises(ResearchError, match="exactly one"):
        await service.set_source_inclusion(included=True)
    with pytest.raises(ResearchError, match="exactly one"):
        await service.set_source_inclusion(
            included=True,
            source_id=source_id,
            source_path="article.pdf",
        )

    await service.ingest(chunk_size=50, chunk_overlap=10)
    listed = await service.list_sources()
    assert listed["sources"][0]["source_id"] == source_id
    assert listed["sources"][0]["title"] == "Reviewed by ID"
    assert listed["discovered_sources"][0]["source_id"] == source_id

    source_path.unlink()
    write_reviewed_metadata(
        config,
        "article.pdf",
        {"title": "Corrected after source removal"},
    )
    listed_after_removal = await service.list_sources()
    assert listed_after_removal["discovered_sources"] == []
    assert listed_after_removal["sources"][0]["source_id"] == source_id
    assert listed_after_removal["sources"][0]["title"] == (
        "Corrected after source removal"
    )


def test_source_ids_address_reviewed_sources(project: Path) -> None:
    asyncio.run(_assert_source_ids_address_reviewed_sources(project))


async def _assert_find_source_answers_for_one_source(project: Path) -> None:
    first = project / "sources" / "crawford-atlas.pdf"
    second = project / "sources" / "crawford-notes.pdf"
    write_pdf(first, ["Cobalt evidence about archives and borders."])
    write_pdf(second, ["Cobalt evidence about a different archive."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    # Before an ingestion the files are known and neither is searchable, so the
    # answer says the call that changes that.
    before = await service.find_source("crawford")
    assert before["ready"] is False
    assert before["match_count"] == 2
    assert before["truncated"] is False
    assert before["message"] == (
        "2 sources match 'crawford' and none is searchable: each needs "
        "ingesting, or a reviewed exclusion is in force."
    )
    assert all(record["searchable"] is False for record in before["matches"])
    assert {record["source_relative_path"] for record in before["matches"]} == {
        "crawford-atlas.pdf",
        "crawford-notes.pdf",
    }

    await service.ingest(chunk_size=50, chunk_overlap=10)

    by_path = await service.find_source("crawford-atlas")
    assert by_path["match_count"] == 1
    assert by_path["matches"][0]["source_relative_path"] == "crawford-atlas.pdf"
    assert by_path["matches"][0]["indexed_in_current_generation"] is True
    assert by_path["matches"][0]["searchable"] is True
    assert by_path["matches"][0]["source_id"].startswith("src_")

    # The title and the authors of a reviewed override are matchable names too.
    write_reviewed_metadata(
        config,
        "crawford-notes.pdf",
        {"title": "Reviewed Atlas Notes", "authors": ["R. Researcher"]},
    )
    by_title = await service.find_source("atlas notes")
    assert by_title["matches"][0]["source_relative_path"] == "crawford-notes.pdf"
    assert by_title["matches"][0]["authors"] == ["R. Researcher"]
    by_author = await service.find_source("RESEARCHER")
    assert by_author["match_count"] == 1

    # A reviewed exclusion is reported as one, because it withholds retrieval
    # even though the source is indexed.
    await service.set_source_inclusion(
        source_path="crawford-atlas.pdf",
        included=False,
        reason="Reviewed duplicate.",
    )
    excluded = await service.find_source("crawford-atlas")
    assert excluded["matches"][0]["included"] is False
    assert excluded["matches"][0]["indexed_in_current_generation"] is True
    assert excluded["matches"][0]["searchable"] is False
    assert excluded["searchable_match_count"] == 0

    # A source whose file is gone stays answerable, and says so.
    second.unlink()
    gone = await service.find_source("crawford-notes")
    assert gone["match_count"] == 1
    assert gone["matches"][0]["exists"] is False
    assert gone["matches"][0]["indexed_in_current_generation"] is True

    missing = await service.find_source("nobody at all")
    assert missing["match_count"] == 0
    assert missing["matches"] == []
    assert missing["message"].startswith("No source matches 'nobody at all'.")

    # A cap that hid matches is disclosed rather than read as the whole answer.
    for index in range(4):
        write_pdf(
            project / "sources" / f"crawford-note-{index}.pdf",
            [f"Cobalt evidence number {index}."],
        )
    capped = await service.find_source("crawford", limit=2)
    assert len(capped["matches"]) == 2
    assert capped["truncated"] is True
    assert capped["match_count"] == 6
    # One indexed source is excluded and four are new files, so only the ingested,
    # included one is searchable.
    assert capped["message"] == (
        "1 of the 6 sources matching 'crawford' are searchable now."
    )
    one = await service.find_source("crawford", limit=1)
    assert one["matches"] == capped["matches"][:1]
    assert one["match_count"] == 6

    with pytest.raises(ResearchError, match="look up"):
        await service.find_source("   ")


def test_find_source_answers_for_one_source(project: Path) -> None:
    asyncio.run(_assert_find_source_answers_for_one_source(project))


async def _assert_reviewed_metadata_survives_source_removal(
    project: Path,
) -> None:
    source = project / "sources" / "temporary.pdf"
    write_pdf(source, ["Temporary evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    source_id = (await service.list_sources())["discovered_sources"][0]["source_id"]
    write_reviewed_metadata(
        config,
        "temporary.pdf",
        {"title": "Initial reviewed title"},
    )
    source.unlink()

    listed = await service.list_sources()
    assert listed["reviewed_metadata_source_count"] == 1
    assert listed["reviewed_metadata_sources"] == [
        {
            "source_id": source_id,
            "source_relative_path": "temporary.pdf",
            "source_path": "sources/temporary.pdf",
            "metadata": {"title": "Initial reviewed title"},
            "indexed_in_current_generation": False,
        }
    ]
    write_reviewed_metadata(
        config,
        "temporary.pdf",
        {"title": "Corrected after deletion"},
    )
    corrected = await service.list_sources()
    assert corrected["reviewed_metadata_sources"][0]["metadata"] == {
        "title": "Corrected after deletion"
    }
    write_reviewed_metadata(config, "temporary.pdf", {})
    cleared = await service.list_sources()
    assert cleared["reviewed_metadata_sources"] == []
    assert json.loads(config.metadata_path.read_text(encoding="utf-8"))["sources"] == {}
    restarted = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    write_reviewed_metadata(
        restarted.config,
        "temporary.pdf",
        {"title": "Addressable after clearing"},
    )
    addressable = await restarted.list_sources()
    assert addressable["reviewed_metadata_sources"] == [
        {
            "source_id": source_id,
            "source_relative_path": "temporary.pdf",
            "source_path": "sources/temporary.pdf",
            "metadata": {"title": "Addressable after clearing"},
            "indexed_in_current_generation": False,
        }
    ]


def test_reviewed_metadata_survives_source_removal(project: Path) -> None:
    asyncio.run(_assert_reviewed_metadata_survives_source_removal(project))


async def _assert_catalog_retains_unindexed_deleted_and_renamed_sources(
    project: Path,
) -> None:
    original = project / "sources" / "temporary.pdf"
    write_pdf(original, ["Temporary evidence registered before ingestion."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    listed = await service.list_sources()
    original_id = listed["discovered_sources"][0]["source_id"]

    renamed = original.with_name("renamed.pdf")
    original.rename(renamed)
    after_rename = await service.list_sources()
    renamed_id = after_rename["discovered_sources"][0]["source_id"]
    assert renamed_id != original_id
    assert {
        (item["source_id"], item["source_relative_path"], item["exists"])
        for item in after_rename["known_sources"]
    } == {
        (original_id, "temporary.pdf", False),
        (renamed_id, "renamed.pdf", True),
    }

    renamed.unlink()
    restarted = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    write_reviewed_metadata(
        restarted.config,
        "temporary.pdf",
        {"title": "Reviewed after deletion"},
    )
    addressable = await restarted.list_sources()
    assert addressable["reviewed_metadata_sources"] == [
        {
            "source_id": original_id,
            "source_relative_path": "temporary.pdf",
            "source_path": "sources/temporary.pdf",
            "metadata": {"title": "Reviewed after deletion"},
            "indexed_in_current_generation": False,
        }
    ]


def test_catalog_retains_unindexed_deleted_and_renamed_sources(
    project: Path,
) -> None:
    asyncio.run(_assert_catalog_retains_unindexed_deleted_and_renamed_sources(project))


def test_removed_legacy_reviewed_metadata_uses_safe_fallback() -> None:
    legacy = {
        "document_id": "doc_legacy",
        "source_path": "sources/legacy.pdf",
        "source_relative_path": "legacy.pdf",
        "title": "Wrong Reviewed Title",
        "authors": ["Wrong Author"],
        "year": 1900,
        "doi": "10.1000/wrong",
        "categories": ["wrong category"],
        "keywords": ["wrong keyword"],
        "metadata_override_revision": service_module.value_fingerprint(
            {
                "title": "Wrong Reviewed Title",
                "authors": ["Wrong Author"],
                "year": 1900,
                "doi": "10.1000/wrong",
                "categories": ["wrong category"],
                "keywords": ["wrong keyword"],
            }
        ),
        "metadata_provenance": {
            field: "reviewed_override"
            for field in (
                "title",
                "authors",
                "year",
                "doi",
                "categories",
                "keywords",
            )
        },
        "metadata_confidence": {},
        "metadata_warnings": [],
    }

    effective = _effective_document_metadata(legacy, {})

    assert effective["title"] == "legacy"
    assert effective["authors"] == []
    assert effective["year"] is None
    assert effective["doi"] == ""
    assert effective["categories"] == []
    assert effective["keywords"] == []
    assert "metadata_confidence" not in effective
    assert (
        "automatic_metadata_unavailable_after_override_removal"
        in effective["metadata_warnings"]
    )


def test_provenance_free_legacy_metadata_falls_back_only_after_snapshot_change() -> (
    None
):
    legacy_document = {
        "document_id": "doc_legacy",
        "source_path": "sources/legacy.pdf",
        "source_relative_path": "legacy.pdf",
        "title": "Valid Automatic Title",
        "authors": ["Automatic Author"],
        "year": 2020,
        "doi": "10.1000/automatic",
        "categories": [],
        "keywords": [],
    }
    empty_revision = service_module.value_fingerprint({})
    matching_manifest = {
        "metadata_revision": empty_revision,
        "documents": [legacy_document],
    }

    matching = service_module._effective_documents(matching_manifest, {})["doc_legacy"]
    assert matching["title"] == "Valid Automatic Title"
    assert matching["authors"] == ["Automatic Author"]
    assert matching["year"] == 2020
    assert matching["doi"] == "10.1000/automatic"

    changed = service_module._effective_documents(
        matching_manifest,
        {"legacy.pdf": {"categories": ["corrected"]}},
    )["doc_legacy"]
    assert changed["title"] == "legacy"
    assert changed["authors"] == []
    assert changed["year"] is None
    assert changed["doi"] == ""
    assert changed["categories"] == ["corrected"]
    assert (
        "automatic_metadata_unavailable_after_override_removal"
        in changed["metadata_warnings"]
    )


async def _assert_symbol_only_chunks_do_not_reach_indexes(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = SymbolChunkUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(config, ultrarag, dense=dense)  # type: ignore[arg-type]

    result = await service.ingest(chunk_size=50, chunk_overlap=10)

    assert result["discarded_symbol_only_chunk_count"] == 1
    generation_root = Path(result["generation_root"])
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["discarded_symbol_only_chunk_count"] == 1
    assert manifest["build_metrics"]["discarded_symbol_only_chunk_count"] == 1
    chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
    assert [chunk["contents"] for chunk in chunks] == ["Stable research evidence."]
    assert [chunk["contents"] for chunk in dense.chunks] == [
        "Stable research evidence."
    ]
    assert set(chunks[0]) == {
        "annotations",
        "chunk_id",
        "content_kind",
        "contents",
        "dense_truncated",
        "document_chunk_index",
        "document_id",
        "embedding_token_count",
        "id",
        "locator",
        "quality_flags",
        "source_id",
        "unit_id",
    }
    # The ingestion audit records a real count and an unset truncation flag.
    assert chunks[0]["embedding_token_count"] == len(
        ["Stable", "research", "evidence."]
    )
    assert chunks[0]["dense_truncated"] is False
    assert manifest["build_metrics"]["dense_token_audit"] == "counted"
    assert manifest["build_metrics"]["dense_truncated_chunk_count"] == 0
    assert ultrarag.passages == ["Stable research evidence."]


def test_symbol_only_chunks_do_not_reach_indexes(project: Path) -> None:
    asyncio.run(_assert_symbol_only_chunks_do_not_reach_indexes(project))


def test_status_reports_the_relocated_runtime_root(project: Path) -> None:
    async def exercise() -> None:
        runtime_root = project.parent / "relocated-runtime"
        config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root=runtime_root,
        )
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )

        status = await service.status()
        assert status["runtime_root"] == str(runtime_root.resolve())
        assert status["state_root"] == str(runtime_root.resolve())
        assert status["project_root"] == str(project.resolve())

        default_service = ResearchService(  # type: ignore[arg-type]
            resolve_config(project, vanilla_executable=sys.executable),
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        assert (await default_service.status())["runtime_root"] is None

    asyncio.run(exercise())


def test_dense_backend_selection_honors_the_threshold_and_config(
    project: Path,
) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    limit = config.settings.exact_backend_chunk_limit

    assert service._select_build_dense_backend(0) == "portable-exact-vectors"
    assert service._select_build_dense_backend(limit) == "portable-exact-vectors"
    assert service._select_build_dense_backend(limit + 1) == "embedded-qdrant"

    forced_qdrant = ResearchService(  # type: ignore[arg-type]
        replace(
            config,
            settings=replace(config.settings, dense_backend="qdrant"),
        ),
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    assert forced_qdrant._select_build_dense_backend(1) == "embedded-qdrant"

    forced_exact = ResearchService(  # type: ignore[arg-type]
        replace(
            config,
            settings=replace(config.settings, dense_backend="exact"),
        ),
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    assert forced_exact._select_build_dense_backend(limit + 1) == (
        "portable-exact-vectors"
    )

    with pytest.raises(ConfigurationError, match="dense.backend"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            dense_backend="lancedb",
        )

    # Every reranker is pinned to a revision, so an unknown name is refused
    # instead of resolving to whatever the model hub serves that day.
    with pytest.raises(ConfigurationError, match="dense.reranker_model"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            reranker_model="some-org/some-reranker",
        )

    configured = resolve_config(
        project,
        vanilla_executable=sys.executable,
        reranker_model="jinaai/jina-reranker-v1-turbo-en",
    )
    assert configured.reranker_model == "jinaai/jina-reranker-v1-turbo-en"


def test_generation_records_its_dense_backend_and_dispatch_follows_it(
    project: Path,
) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    qdrant_double = FakeDenseBackend()
    exact_double = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=qdrant_double,
    )
    # Both recorded kinds resolve to the single injected double by default, so
    # replace the mapping to observe dispatch itself.
    service._dense_backends = {
        "embedded-qdrant": qdrant_double,
        "portable-exact-vectors": exact_double,
    }

    # A generation with no recorded backend predates the two-backend contract.
    assert service._dense_for(None) is qdrant_double
    assert service._dense_for({"retrieval": {"dense": {}}}) is qdrant_double
    assert (
        service._dense_for(
            {"retrieval": {"dense": {"dense_backend": "embedded-qdrant"}}}
        )
        is qdrant_double
    )
    assert (
        service._dense_for(
            {"retrieval": {"dense": {"dense_backend": "portable-exact-vectors"}}}
        )
        is exact_double
    )
    # An unrecognized or malformed record falls back to the legacy backend.
    assert (
        service._dense_for({"retrieval": {"dense": {"dense_backend": "faiss"}}})
        is qdrant_double
    )
    assert (
        service._dense_for({"retrieval": {"dense": {"dense_backend": 7}}})
        is qdrant_double
    )


def test_ingest_records_the_exact_backend_and_its_index_path(project: Path) -> None:
    async def exercise() -> None:
        write_pdf(project / "sources" / "article.pdf", ["Stable cobalt evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )

        result = await service.ingest(chunk_size=50, chunk_overlap=10)

        manifest = json.loads(
            (Path(result["generation_root"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["retrieval"]["dense"]["dense_backend"] == (
            "portable-exact-vectors"
        )
        assert manifest["files"]["dense_index"] == "indexes/vectors"
        status = await service.status()
        assert status["retrieval"]["dense"]["dense_backend"] == (
            "portable-exact-vectors"
        )

    asyncio.run(exercise())


async def _assert_dense_token_audit_flags_truncation(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    dense.token_count_override = 600
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )

    result = await service.ingest(chunk_size=50, chunk_overlap=10)

    assert result["dense_token_audit"] == "counted"
    assert result["embedding_maximum_tokens"] == 512
    assert result["dense_truncated_chunk_count"] == result["chunk_count"]
    assert result["maximum_embedding_token_count"] == 600
    generation_root = Path(result["generation_root"])
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["build_metrics"]["dense_token_audit"] == "counted"
    chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
    assert all(chunk["dense_truncated"] is True for chunk in chunks)

    search = await service.search("evidence", top_k=4, retrieval_method="bm25")
    assert search["hits"]
    assert search["dense_fidelity"]["embedding_maximum_tokens"] == 512
    assert search["dense_fidelity"]["truncated_passages_returned"] == len(
        search["hits"]
    )
    assert all(hit["embedding_token_count"] == 600 for hit in search["hits"])


def test_dense_token_audit_flags_truncation(project: Path) -> None:
    asyncio.run(_assert_dense_token_audit_flags_truncation(project))


async def _assert_dense_token_audit_degrades_without_failing_ingestion(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    dense.audit_unavailable = True
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )

    result = await service.ingest(chunk_size=50, chunk_overlap=10)

    assert result["status"] == "ready"
    assert result["dense_token_audit"] == "unavailable"
    assert result["dense_audited_chunk_count"] == 0
    assert result["dense_truncated_chunk_count"] == 0
    generation_root = Path(result["generation_root"])
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
    assert "embedding_token_count" not in chunks[0]
    assert "dense_truncated" not in chunks[0]

    search = await service.search("evidence", top_k=4, retrieval_method="bm25")
    assert search["dense_fidelity"]["audited_passages_returned"] == 0
    assert search["dense_fidelity"]["truncated_passages_returned"] == 0
    assert all(hit["embedding_token_count"] is None for hit in search["hits"])


def test_dense_token_audit_degrades_without_failing_ingestion(project: Path) -> None:
    asyncio.run(_assert_dense_token_audit_degrades_without_failing_ingestion(project))


async def _assert_agent_reviewed_source_exclusion(project: Path) -> None:
    preferred = project / "sources" / "preferred.pdf"
    duplicate = project / "sources" / "duplicate.pdf"
    write_pdf(preferred, ["Shared cobalt evidence for duplicate review."])
    write_pdf(duplicate, ["Shared cobalt evidence for duplicate review."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert first["source_file_count"] == 2
    assert first["document_count"] == 2

    initial = await service.search("shared cobalt evidence", top_k=10)
    duplicate_hit = next(
        hit for hit in initial["hits"] if hit["source_path"] == "sources/duplicate.pdf"
    )

    with pytest.raises(ResearchError, match="reason is required"):
        await service.set_source_inclusion("duplicate.pdf", included=False)
    with pytest.raises(ResearchError, match="escapes"):
        await service.set_source_inclusion(
            "../outside.pdf",
            included=False,
            reason="Invalid path must not be accepted.",
        )

    excluded = await service.set_source_inclusion(
        "duplicate.pdf",
        included=False,
        reason="Agent-reviewed duplicate; preferred.pdf is the retained copy.",
    )
    assert excluded["effective_immediately"] is True
    assert excluded["source_file_changed"] is False
    assert duplicate.is_file()

    status = await service.status()
    assert status["stale"] is True
    assert status["excluded_source_count"] == 1
    assert status["searchable_source_count"] == 1
    assert status["changes"]["source_exclusions_changed"] is True
    assert status["excluded_sources"][0]["indexed_in_current_generation"] is True

    search = await service.search("shared cobalt evidence", top_k=10)
    assert search["excluded_source_count"] == 1
    assert {hit["source_path"] for hit in search["hits"]} == {"sources/preferred.pdf"}
    sources = await service.list_sources()
    assert sources["source_count"] == 1
    assert sources["sources"][0]["source_path"] == "sources/preferred.pdf"
    with pytest.raises(ResearchError, match="currently excluded"):
        await service.get_passage(duplicate_hit["chunk_id"])

    rebuilt = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert rebuilt["source_file_count"] == 2
    assert rebuilt["excluded_source_count"] == 1
    assert rebuilt["document_count"] == 1
    assert rebuilt["reused_document_count"] == 1
    assert rebuilt["rebuilt_document_count"] == 0
    rebuilt_status = await service.status()
    assert rebuilt_status["stale"] is False
    assert (
        rebuilt_status["excluded_sources"][0]["indexed_in_current_generation"] is False
    )

    included = await service.set_source_inclusion(
        "duplicate.pdf",
        included=True,
    )
    assert included["effective_immediately"] is False
    assert included["generation_rebuild_recommended"] is True
    assert duplicate.is_file()
    assert (await service.status())["stale"] is True

    restored = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert restored["document_count"] == 2
    assert restored["excluded_source_count"] == 0
    assert restored["reused_document_count"] == 1
    assert restored["rebuilt_document_count"] == 1


def test_agent_reviewed_source_exclusion_is_immediate_and_reversible(
    project: Path,
) -> None:
    asyncio.run(_assert_agent_reviewed_source_exclusion(project))


async def _assert_deleted_excluded_source_is_restorable_by_id(project: Path) -> None:
    source = project / "sources" / "temporary.pdf"
    write_pdf(source, ["Temporary source that is reviewed before ingestion."])
    service = ResearchService(  # type: ignore[arg-type]
        resolve_config(project, vanilla_executable=sys.executable),
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    discovered = await service.list_sources()
    source_id = discovered["discovered_sources"][0]["source_id"]
    await service.set_source_inclusion(
        source_id=source_id,
        included=False,
        reason="Source was removed after review.",
    )
    source.unlink()

    excluded = await service.list_sources()
    assert excluded["excluded_sources"][0]["source_id"] == source_id
    write_reviewed_metadata(
        service.config,
        "temporary.pdf",
        {"title": "Reviewed while excluded and missing"},
    )
    write_reviewed_metadata(service.config, "temporary.pdf", {})
    restored = await service.set_source_inclusion(
        source_id=source_id,
        included=True,
    )
    assert restored["status"] == "changed"
    assert restored["source_id"] == source_id
    assert (await service.list_sources())["excluded_sources"] == []


def test_deleted_excluded_source_is_restorable_by_advertised_id(project: Path) -> None:
    asyncio.run(_assert_deleted_excluded_source_is_restorable_by_id(project))


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    ordered, scores = ResearchService._fuse_rankings(
        ["bm25-only", "shared"],
        ["shared", "dense-only"],
        rrf_k=60,
        bm25_weight=1.25,
        dense_weight=1.0,
        maximum_candidates=200,
    )
    assert ordered[0] == "shared"
    assert scores["shared"] > scores["bm25-only"]
    assert scores["bm25-only"] > scores["dense-only"]


def _source_level_metrics(
    result: dict[str, object],
    grades: dict[str, int],
    *,
    cutoff: int,
) -> dict[str, float]:
    ranked_sources = list(
        dict.fromkeys(
            str(hit["source_path"])
            for hit in result["hits"]  # type: ignore[index]
        )
    )
    relevant = {source for source, grade in grades.items() if grade > 0}
    returned_relevant = relevant.intersection(ranked_sources)
    precision = len(returned_relevant) / len(ranked_sources) if ranked_sources else 0.0
    recall = len(returned_relevant) / len(relevant) if relevant else 0.0

    def discounted_gain(values: list[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(values, 1)
        )

    actual = [grades[source] for source in ranked_sources[:cutoff]]
    ideal = sorted(grades.values(), reverse=True)[:cutoff]
    ideal_gain = discounted_gain(ideal)
    return {
        "precision": precision,
        "recall": recall,
        "ndcg": discounted_gain(actual) / ideal_gain if ideal_gain else 0.0,
    }


async def _assert_relevance_gates_allow_abstention(project: Path) -> None:
    write_pdf(
        project / "sources" / "brands.pdf",
        ["Lenovo ZTE Sony (Motorola) Microsoft Samsung Vodafone"],
        title="Brand Table",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    unrelated = await service.search("hello", top_k=8)
    assert unrelated["result_count"] == 0
    assert unrelated["relevance_limited"] is True
    assert unrelated["rejected_candidates"] == {
        "bm25_no_query_token_overlap": 1,
        "bm25_extraction_artifact": 0,
        "bm25_corrupt_text": 0,
        "bm25_too_short": 0,
        "dense_below_threshold": 1,
        "dense_extraction_artifact": 0,
        "dense_corrupt_text": 0,
        "dense_too_short": 0,
    }

    relevant = await service.search("Lenovo", top_k=8)
    assert relevant["result_count"] == 1
    assert relevant["hits"][0]["match_kind"] == "hybrid"


def test_relevance_gates_allow_zero_results(project: Path) -> None:
    asyncio.run(_assert_relevance_gates_allow_abstention(project))


async def _assert_contentless_queries_abstain(project: Path) -> None:
    """A question word carries no topic, so the lexical half must reject it."""

    write_pdf(
        project / "sources" / "questions.pdf",
        ["How does the analysis work, and what does it show?"],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    for query in ("how?", "what?", "why not?", "what does it do?"):
        answer = await service.search(
            query,
            top_k=5,
            retrieval_method="bm25",
            include_staleness=False,
        )
        assert answer["result_count"] == 0, query
        assert answer["relevance_limited"] is True, query
        # The gateway still returned the chunk; the gate is what dropped it.
        assert answer["rejected_candidates"]["bm25_no_query_token_overlap"] == 1, query

    anchored = await service.search(
        "analysis",
        top_k=5,
        retrieval_method="bm25",
        include_staleness=False,
    )
    assert anchored["result_count"] == 1


def test_contentless_queries_abstain(project: Path) -> None:
    asyncio.run(_assert_contentless_queries_abstain(project))


async def _assert_dense_threshold_boundary(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Semantic research material."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = ScoredDenseBackend(0.7199)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    rejected = await service.search("unrelated", retrieval_method="dense", top_k=1)
    assert rejected["result_count"] == 0
    assert rejected["rejected_candidates"]["dense_below_threshold"] == 1

    dense.score = 0.72
    accepted = await service.search("unrelated", retrieval_method="dense", top_k=1)
    assert accepted["result_count"] == 1
    assert accepted["hits"][0]["match_kind"] == "semantic"


def test_dense_threshold_boundary(project: Path) -> None:
    asyncio.run(_assert_dense_threshold_boundary(project))


async def _assert_a_dense_band_below_the_floor_is_not_arbitrary(project: Path) -> None:
    """A query whose whole candidate list sits under the floor still finds them.

    A one-word query's best passage can score below the cosine floor while its
    neighbours sit a few hundredths behind it, and the floor then keeps a handful
    of candidates and drops passages that match the query about as closely. The
    margin admits that band, names what it still dropped, and never invents
    support for a query nothing cleared the floor for.
    """

    for name, text in (
        ("a.pdf", "Semantic research material about waste and disposal."),
        ("b.pdf", "Another passage about waste streams and repair."),
        ("c.pdf", "A third passage about waste and its afterlives."),
    ):
        write_pdf(project / "sources" / name, [text], title=name)

    async def search_with(margin: float, strong: float, band: float) -> dict:
        dense = BandedDenseBackend(strong=strong, band=band)
        service = ResearchService(  # type: ignore[arg-type]
            resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=[
                    f"retrieval.dense_relative_similarity_margin={margin}"
                ],
            ),
            FakeUltraRAG(),
            dense=dense,
        )
        await service.ingest(chunk_size=50, chunk_overlap=10, force_recompute=True)
        return await service.search("waste", retrieval_method="dense", top_k=5)

    # The floor alone keeps only the candidate that cleared it, and the answer
    # names the sources it removed rather than only counting them.
    floor_only = await search_with(margin=0.0, strong=0.75, band=0.65)
    assert floor_only["result_count"] == 1
    assert floor_only["rejected_candidates"]["dense_below_threshold"] == 2
    assert floor_only["dense_gate"]["admitted_below_floor"] == 0
    assert floor_only["dense_gate"]["best_cosine_similarity"] == 0.75
    examples = floor_only["rejected_candidate_examples"]["reasons"][
        "dense_below_threshold"
    ]
    assert {example["source_relative_path"] for example in examples} == {
        "b.pdf",
        "c.pdf",
    }
    assert {example["cosine_similarity"] for example in examples} == {0.65}

    # Within the margin of the best score the band is admitted, and the answer
    # spans the sources the floor had thinned.
    rescued = await search_with(margin=0.10, strong=0.75, band=0.65)
    assert rescued["result_count"] == 3
    assert rescued["distinct_reference_count"] == 3
    assert rescued["dense_gate"]["admitted_below_floor"] == 2
    assert rescued["rejected_candidates"]["dense_below_threshold"] == 0

    # With nothing above the floor the margin must not manufacture support.
    unsupported = await search_with(margin=0.10, strong=0.65, band=0.65)
    assert unsupported["result_count"] == 0
    assert unsupported["relevance_limited"] is True
    assert unsupported["dense_gate"]["admitted_below_floor"] == 0


def test_a_dense_band_below_the_floor_is_not_arbitrary(project: Path) -> None:
    asyncio.run(_assert_a_dense_band_below_the_floor_is_not_arbitrary(project))


async def _assert_a_short_fragment_is_not_evidence(project: Path) -> None:
    """An index line matches a query about its own words, so length decides.

    Chunks never span extraction units, so one short unit becomes one short chunk
    that can outrank prose: a five-word index entry contains the query's words and
    nothing to cite. The floor admits it, the diversity rule cannot see it, and
    the reranker scores it as relevant — only a length rule separates it from a
    passage. The rule ships off, and this is what turning it on does.
    """

    write_pdf(project / "sources" / "fragment.pdf", ["Waste, 12-14"], title="Index")
    write_pdf(
        project / "sources" / "prose.pdf",
        [
            (
                "Waste from data infrastructure accumulates in landfills and in the "
                "water table around them, and the extraction feeding it leaves "
                "trails of its own across the regions that host the servers."
            )
        ],
        title="Prose",
    )

    async def search_with(minimum: int, fraction: float = 0.0) -> dict:
        service = ResearchService(  # type: ignore[arg-type]
            resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=[
                    f"retrieval.minimum_passage_words={minimum}",
                    f"retrieval.minimum_passage_token_fraction={fraction}",
                ],
            ),
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        await service.ingest(chunk_size=50, chunk_overlap=10, force_recompute=True)
        return await service.search("waste", top_k=5)

    everything = await search_with(minimum=0)
    assert everything["result_count"] == 2
    assert everything["rejected_candidates"]["bm25_too_short"] == 0
    assert everything["rejected_candidates"]["dense_too_short"] == 0
    assert everything["passage_length_policy"]["minimum_words"] == 0
    assert everything["passage_length_policy"]["minimum_tokens"] == 0
    assert everything["passage_length_policy"]["chunk_size"] is None

    evidence_only = await search_with(minimum=12)
    assert [hit["source_relative_path"] for hit in evidence_only["hits"]] == [
        "prose.pdf"
    ]
    assert evidence_only["rejected_candidates"]["bm25_too_short"] == 1
    assert evidence_only["rejected_candidates"]["dense_too_short"] == 1
    assert evidence_only["passage_length_policy"]["minimum_words"] == 12
    examples = evidence_only["rejected_candidate_examples"]["reasons"]
    assert {
        example["source_relative_path"]
        for example in examples["dense_too_short"] + examples["bm25_too_short"]
    } == {"fragment.pdf"}

    # The same floor in the unit the chunker counts: half of a 50-token chunk is
    # 25 tokens, which the index line cannot reach while the prose sentence can.
    tokens_only = await search_with(minimum=0, fraction=0.5)
    assert [hit["source_relative_path"] for hit in tokens_only["hits"]] == ["prose.pdf"]
    assert tokens_only["rejected_candidates"]["bm25_too_short"] == 1
    assert tokens_only["rejected_candidates"]["dense_too_short"] == 1
    policy = tokens_only["passage_length_policy"]
    assert policy["minimum_words"] == 0
    assert policy["minimum_tokens"] == 25
    assert policy["chunk_size"] == 50
    assert policy["tokenizer"] == "gpt2"
    assert policy["token_fraction"] == 0.5


def test_a_short_fragment_is_not_evidence(project: Path) -> None:
    asyncio.run(_assert_a_short_fragment_is_not_evidence(project))


def test_the_length_floor_is_counted_in_the_chunkers_tokens() -> None:
    """A fraction of a chunk size is only a length rule if the unit matches.

    The chunker counts a chunk in GPT-2 tokens and the generation records that
    name, so the floor is counted in the same tokens over the returned text.
    """

    index_line = "value production chain, 70-71"
    prose = (
        "Waste from data infrastructure accumulates in landfills and in the water "
        "table around them, and the extraction feeding it leaves trails of its own "
        "across the regions that host the servers."
    )
    assert support_module.passage_token_count(index_line, "gpt2") == 7
    assert support_module.passage_token_count(prose, "gpt2") > 30
    assert support_module.passage_token_count("", "gpt2") == 0
    assert support_module.TOKENIZER_REPOSITORIES["gpt2"] == "openai-community/gpt2"
    with pytest.raises(support_module.ResearchError, match="unknown chunker tokenizer"):
        support_module.passage_token_count(prose, "not-a-tokenizer")


async def _assert_no_change_ingest_is_noop_and_force_rebuilds(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(config, ultrarag, dense=dense)  # type: ignore[arg-type]

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    second = await service.ingest(chunk_size=50, chunk_overlap=10)
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
    )

    assert first["generation_id"] == second["generation_id"]
    assert second["generation_changed"] is False
    assert second["rebuilt_document_count"] == 0
    assert second["created_vector_count"] == 0
    assert forced["generation_id"] != first["generation_id"]
    assert forced["generation_changed"] is True
    assert forced["reused_document_count"] == 0
    assert forced["created_vector_count"] == forced["chunk_count"]
    assert Path(first["generation_root"]).is_dir()
    assert Path(forced["generation_root"]).is_dir()
    assert ultrarag.chunk_calls == 2
    assert ultrarag.bm25_build_calls == 2
    assert dense.build_calls == 2
    current = json.loads(config.current_path.read_text(encoding="utf-8"))
    assert current["generation_id"] == forced["generation_id"]


def test_no_change_ingest_is_noop_and_force_rebuilds(project: Path) -> None:
    asyncio.run(_assert_no_change_ingest_is_noop_and_force_rebuilds(project))


@pytest.mark.parametrize(
    "artifact",
    ["chunks", "vectors", "bm25", "dense"],
)
def test_matching_sources_rebuild_a_corrupt_selected_generation(
    project: Path,
    artifact: str,
) -> None:
    async def exercise() -> None:
        write_pdf(project / "sources" / "article.pdf", ["Stable evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        first = await service.ingest(chunk_size=50, chunk_overlap=10)
        generation_root = Path(first["generation_root"])
        manifest = json.loads(
            (generation_root / "manifest.json").read_text(encoding="utf-8")
        )
        targets = {
            "chunks": generation_root / manifest["files"]["chunks"],
            "vectors": generation_root / manifest["files"]["portable_embeddings"],
            "bm25": generation_root
            / manifest["files"]["bm25_index"]
            / "fake-index.json",
            "dense": generation_root
            / manifest["files"]["dense_index"]
            / "fake-qdrant.json",
        }
        targets[artifact].write_bytes(b"corrupt\n")

        rebuilt = await service.ingest(chunk_size=50, chunk_overlap=10)

        assert rebuilt["status"] == "ready"
        assert rebuilt["generation_changed"] is True
        assert rebuilt["generation_id"] != first["generation_id"]
        assert rebuilt["rebuilt_document_count"] == 1

    asyncio.run(exercise())


async def _assert_selective_reuse_tracks_every_input_change(project: Path) -> None:
    first_path = project / "sources" / "first.pdf"
    second_path = project / "sources" / "second.pdf"
    write_pdf(first_path, ["Stable cobalt evidence."], title="First")
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    ultrarag = FakeUltraRAG()
    service = ResearchService(config, ultrarag, dense=dense)  # type: ignore[arg-type]

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(second_path, ["Amber evidence in a second source."], title="Second")
    added = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert added["generation_changed"] is True
    assert added["reused_document_count"] == 1
    assert added["rebuilt_document_count"] == 1
    assert added["reused_chunk_count"] == 1
    assert added["created_vector_count"] == 1

    previous_stat = second_path.stat()
    original_bytes = second_path.read_bytes()
    # Swap one same-length character inside the PDF's title entry so the bytes
    # change while the size and mtime stay identical — the case under test. The
    # page text stream is deflated, but the Info dictionary is plain bytes, and
    # regenerating the file instead would depend on pymupdf's varying embedded
    # timestamp, which shifts the total by a few bytes and made this case
    # intermittently untestable.
    assert b"/Title(Second)" in original_bytes
    second_path.write_bytes(
        original_bytes.replace(b"/Title(Second)", b"/Title(Secand)", 1)
    )
    assert second_path.stat().st_size == previous_stat.st_size
    os.utime(
        second_path,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns),
    )
    changed_bytes = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert changed_bytes["generation_changed"] is True
    assert changed_bytes["reused_document_count"] == 1
    assert changed_bytes["rebuilt_document_count"] == 1

    write_reviewed_metadata(config, "first.pdf", {"categories": ["theory"]})
    metadata_changed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert metadata_changed["status"] == "unchanged"
    assert metadata_changed["generation_id"] == changed_bytes["generation_id"]
    assert metadata_changed["rebuilt_document_count"] == 0
    assert metadata_changed["reused_document_count"] == 2
    assert metadata_changed["reused_vector_count"] == metadata_changed["chunk_count"]
    assert metadata_changed["created_vector_count"] == 0

    reuse_search = await service.search("evidence", top_k=8)
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
    )
    full_search = await service.search("evidence", top_k=8)
    assert forced["created_vector_count"] == forced["chunk_count"]
    assert [item["chunk_id"] for item in reuse_search["hits"]] == [
        item["chunk_id"] for item in full_search["hits"]
    ]
    assert [item["rank"] for item in reuse_search["hits"]] == [
        item["rank"] for item in full_search["hits"]
    ]

    second_path.unlink()
    removed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert removed["generation_changed"] is True
    assert removed["document_count"] == 1
    assert removed["reused_document_count"] == 1
    assert removed["rebuilt_document_count"] == 0
    assert removed["generation_id"] != first["generation_id"]


def test_selective_reuse_tracks_every_input_change(project: Path) -> None:
    asyncio.run(_assert_selective_reuse_tracks_every_input_change(project))


async def _assert_ordinary_ingest_migrates_legacy_metadata_storage(
    project: Path,
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Stable cobalt evidence."], title="Automatic Title")
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    write_reviewed_metadata(
        config,
        "article.pdf",
        {"title": "Reviewed Title", "authors": ["Reviewed Author"]},
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    old_manifest_path = Path(first["generation_root"]) / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_manifest.pop("metadata_storage_policy")
    old_document = old_manifest["documents"][0]
    old_document["title"] = "Reviewed Title"
    old_document["authors"] = ["Reviewed Author"]
    old_document["metadata_provenance"]["title"] = "reviewed_override"
    old_document["metadata_provenance"]["authors"] = "reviewed_override"
    old_manifest_path.write_text(
        json.dumps(old_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_status = await service.status()
    assert legacy_status["generation_upgrade_required"] is True
    assert "metadata_storage" in legacy_status["upgrade_reasons"]

    migrated = await service.ingest(chunk_size=50, chunk_overlap=10)

    assert migrated["status"] == "ready"
    assert migrated["generation_changed"] is True
    assert migrated["generation_id"] != first["generation_id"]
    assert migrated["rebuilt_document_count"] == 1
    migrated_manifest = json.loads(
        (Path(migrated["generation_root"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert migrated_manifest["metadata_storage_policy"] == (
        service_module.METADATA_STORAGE_POLICY
    )
    assert migrated_manifest["documents"][0]["title"] == "Automatic Title"

    # The reviewed overlay remains live throughout the migration, but removing
    # it now reveals the recovered automatic metadata without another build.
    listed = (await service.list_sources())["sources"][0]
    assert listed["title"] == "Reviewed Title"
    write_reviewed_metadata(config, "article.pdf", {})
    automatic = (await service.list_sources())["sources"][0]
    assert automatic["title"] == "Automatic Title"
    assert (
        "automatic_metadata_unavailable_after_override_removal"
        not in automatic["metadata_warnings"]
    )


def test_ordinary_ingest_migrates_legacy_metadata_storage(project: Path) -> None:
    asyncio.run(_assert_ordinary_ingest_migrates_legacy_metadata_storage(project))


async def _assert_pending_metadata_status_is_not_reported_as_active(
    project: Path,
) -> None:
    first_source = project / "sources" / "first.pdf"
    write_pdf(first_source, ["Stable cobalt evidence."], title="First")
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)

    # A legacy manifest with no observational revision and no current overrides
    # is consistent across both the setter and status response.
    manifest_path = Path(first["generation_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("metadata_revision")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_reviewed_metadata(config, "first.pdf", {})
    initial_status = await service.status()
    assert initial_status["generation_metadata_snapshot_outdated"] is False
    assert initial_status["metadata_overlay_active"] is False

    second_source = project / "sources" / "second.pdf"
    write_pdf(second_source, ["Unindexed amber evidence."], title="Second")
    write_reviewed_metadata(config, "second.pdf", {"title": "Pending Reviewed Title"})

    status = await service.status()
    assert status["stale"] is True
    assert status["changes"]["added"] == ["second.pdf"]
    assert status["metadata_overlay_active"] is False
    assert status["metadata_pending_source_paths"] == ["second.pdf"]
    assert status["generation_metadata_snapshot_outdated"] is True
    assert "call ingest" in status["message"]
    assert "ingestion is not required" not in status["message"]


def test_pending_metadata_status_is_not_reported_as_active(project: Path) -> None:
    asyncio.run(_assert_pending_metadata_status_is_not_reported_as_active(project))


async def _assert_failed_dense_build_does_not_replace_current(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    dense.fail_build = True

    with pytest.raises(RuntimeError, match="simulated dense-index failure"):
        await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            force_recompute=True,
        )

    current = json.loads(config.current_path.read_text(encoding="utf-8"))
    assert current["generation_id"] == first["generation_id"]
    assert [path.name for path in config.generations_root.iterdir()] == [
        first["generation_id"]
    ]
    assert not any(config.staging_root.iterdir())
    failures = list(config.failures_root.glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["error"] == "simulated dense-index failure"


def test_failed_dense_build_does_not_replace_current(project: Path) -> None:
    asyncio.run(_assert_failed_dense_build_does_not_replace_current(project))


def test_checkpoint_progress_is_reconciled_from_per_source_state(
    project: Path,
) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    root = config.staging_root / "build"
    checkpoint: dict[str, object] = {
        "phase": "extraction",
        "selected_source_paths": ["article.pdf", "book.epub"],
        "source_inventory": [
            {"source_relative_path": "article.pdf", "format": "pdf"},
            {"source_relative_path": "book.epub", "format": "epub"},
        ],
        "extracted_source_paths": ["article.pdf"],
        "extraction_work_total": 0,
        "extraction_work_completed": 0,
        "chunking_work_total": 0,
        "chunking_work_completed": 0,
    }
    pdf_root = service._source_artifact_root(root, "article.pdf")
    epub_root = service._source_artifact_root(root, "book.epub")
    pdf_root.mkdir(parents=True)
    epub_root.mkdir(parents=True)
    (pdf_root / "state.json").write_text(
        json.dumps(
            {
                "reused": False,
                "extraction_stage": "complete",
                "total": 2,
                "next_index": 2,
                "chunking_work_total": 2,
                "chunked_unit_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (epub_root / "state.json").write_text(
        json.dumps(
            {
                "reused": False,
                "extraction_stage": "complete",
                "total": 3,
                "next_index": 3,
            }
        ),
        encoding="utf-8",
    )

    service._reconcile_checkpoint_progress(root, checkpoint)  # type: ignore[arg-type]

    assert checkpoint["extraction_work_total"] == 11
    # The EPUB final state is committed, but its global completion checkpoint is
    # not; the normal resume path will account for that last atomic unit once.
    assert checkpoint["extraction_work_completed"] == 10
    assert checkpoint["chunking_work_total"] == 2
    assert checkpoint["chunking_work_completed"] == 1


async def _assert_pdf_ingestion_checkpoints_fixed_page_batches(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(
        project / "sources" / "batched.pdf",
        [
            f"{topic} research develops a distinct account of material evidence."
            for topic in (
                "Cobalt",
                "Amber",
                "Copper",
                "Lithium",
                "Silicon",
                "Nickel",
                "Graphite",
                "Quartz",
                "Manganese",
            )
        ],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    first = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    initialized = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    scanned = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )

    assert first["phase"] == "source_hashing"
    assert initialized["phase"] == "extraction"
    assert initialized["progress"] == {
        "completed": 0,
        "total": 6,
        "unit": "pdf_page_batches_or_epub_sections",
    }
    assert scanned["progress"] == {
        "completed": 1,
        "total": 6,
        "unit": "pdf_page_batches_or_epub_sections",
    }
    staging_root = next(config.staging_root.iterdir())
    state_path = next((staging_root / "work" / "sources").glob("*/state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["page_batch_size"] == config.settings.pdf_page_batch_size == 8
    assert state["next_index"] == 8

    ready = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=300,
    )
    assert ready["status"] == "ready"
    units = read_jsonl(Path(ready["generation_root"]) / "corpus/extracted-units.jsonl")
    assert [unit["locator"]["page"] for unit in units] == list(range(1, 10))


def test_pdf_ingestion_checkpoints_fixed_page_batches(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_pdf_ingestion_checkpoints_fixed_page_batches(project, monkeypatch)
    )


async def _assert_partial_pdf_batch_is_replayed_after_hard_crash(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(
        project / "sources" / "batched.pdf",
        [
            f"{topic} research preserves a distinct account of durable evidence."
            for topic in (
                "Cobalt",
                "Amber",
                "Copper",
                "Lithium",
                "Silicon",
                "Nickel",
                "Graphite",
                "Quartz",
                "Manganese",
            )
        ],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    class SimulatedProcessExit(BaseException):
        pass

    original_atomic_write = ingestion_module.atomic_write_json
    crashed = False

    def fail_during_first_scan_batch(
        path: Path, value: object, **kwargs: object
    ) -> None:
        nonlocal crashed
        if (
            not crashed
            and path.parent.name == "page-scans"
            and path.name == "00000001.json"
        ):
            crashed = True
            raise SimulatedProcessExit
        original_atomic_write(path, value, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            ingestion_module,
            "atomic_write_json",
            fail_during_first_scan_batch,
        )
        with pytest.raises(SimulatedProcessExit):
            await service.ingest(chunk_size=50, chunk_overlap=10)

    staging_root = next(config.staging_root.iterdir())
    state_path = next((staging_root / "work" / "sources").glob("*/state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["extraction_stage"] == "pdf_scan"
    assert state["next_index"] == 0
    assert (state_path.parent / "page-scans/00000000.json").is_file()
    assert not (state_path.parent / "page-scans/00000001.json").exists()

    resumed = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    ready = await resumed.ingest(chunk_size=50, chunk_overlap=10)
    assert ready["status"] == "ready"
    assert ready["resumed"] is True
    units = read_jsonl(Path(ready["generation_root"]) / "corpus/extracted-units.jsonl")
    assert [unit["locator"]["page"] for unit in units] == list(range(1, 10))


def test_hard_crash_mid_chunking_redoes_at_most_one_batch(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        write_pdf(
            project / "sources" / "evidence.pdf",
            [
                "Alpha cobalt evidence.",
                "Beta amber evidence.",
                "Gamma copper evidence.",
            ],
        )
        # Two units share a call, so the first batch commits and the second one
        # dies mid-flight.
        config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["chunking.batch_units=2"],
        )
        crashing = CrashDuringChunkUltraRAG(fail_on_call=2)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            crashing,
            dense=FakeDenseBackend(),
        )

        with pytest.raises(SimulatedProcessExit):
            await service.ingest(chunk_size=50, chunk_overlap=10)

        # The first batch of two units is durable; the failing batch left no
        # durable state behind.
        staging_root = next(config.staging_root.iterdir())
        state = json.loads(
            next((staging_root / "work" / "sources").glob("*/state.json")).read_text(
                encoding="utf-8"
            )
        )
        assert state["chunked_unit_count"] == 2
        assert len(crashing.requested_unit_ids) == 2
        committed_ids = set(crashing.requested_unit_ids[0])
        failed_ids = set(crashing.requested_unit_ids[1])
        assert len(committed_ids) == 2
        assert committed_ids.isdisjoint(failed_ids)

        resumed_ultrarag = CrashDuringChunkUltraRAG()
        resumed = ResearchService(  # type: ignore[arg-type]
            config,
            resumed_ultrarag,
            dense=FakeDenseBackend(),
        )
        ready = await resumed.ingest(chunk_size=50, chunk_overlap=10)

        assert ready["status"] == "ready"
        assert ready["resumed"] is True
        requested = {
            unit_id for call in resumed_ultrarag.requested_unit_ids for unit_id in call
        }
        # Only the unfinished units are re-chunked: a crash mid-chunking redoes
        # at most the batch that was in flight, never a committed one.
        assert requested == failed_ids
        assert ready["chunk_count"] == 3
        units = read_jsonl(
            Path(ready["generation_root"]) / "corpus/extracted-units.jsonl"
        )
        assert len(units) == 3

    asyncio.run(exercise())


def test_chunker_batching_does_not_change_chunk_identity(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        # Distinct wording per page keeps the extraction from treating repeated
        # text as a removable margin block.
        write_pdf(
            project / "sources" / "evidence.pdf",
            [
                f"Page {index} discusses {word} evidence and value."
                for index, word in enumerate(
                    ("cobalt", "amber", "copper", "lithium", "silicon")
                )
            ],
        )

        async def build(units_per_call: int) -> Path:
            batch_config = resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=[f"chunking.batch_units={units_per_call}"],
            )
            service = ResearchService(  # type: ignore[arg-type]
                batch_config,
                FakeUltraRAG(),
                dense=FakeDenseBackend(),
            )
            result = await service.ingest(
                chunk_size=50,
                chunk_overlap=10,
                force_recompute=True,
            )
            assert result["status"] == "ready"
            return Path(result["generation_root"])

        batched = read_jsonl((await build(16)) / "chunks/chunks.jsonl")
        per_unit = read_jsonl((await build(1)) / "chunks/chunks.jsonl")

        assert batched and len(batched) == len(per_unit)
        for field in ("chunk_id", "contents", "document_chunk_index", "unit_id"):
            assert [chunk[field] for chunk in per_unit] == [
                chunk[field] for chunk in batched
            ]

    asyncio.run(exercise())


def test_query_gate_uses_the_stored_chunk_verdict(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        write_pdf(
            project / "sources" / "evidence.pdf",
            ["Cobalt heron evidence from the amber marsh."],
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        ready = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert ready["status"] == "ready"
        # The first search builds or reuses the verdict-carrying lookup.
        assert (await service.search("cobalt heron", top_k=1))["hits"]

        def explode(_text: str, *, quality_flags: object = None) -> int:
            raise AssertionError("the query rescanned chunk text for health")

        # Only the retrieval fallback is intercepted; the document-title health
        # check reaches text_corruption_reasons through a different caller.
        monkeypatch.setattr(support_module, "chunk_health_flags", explode)
        result = await service.search("cobalt heron", top_k=1)

        assert result["hits"]

    asyncio.run(exercise())


def test_query_gate_falls_back_without_a_stored_verdict(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        write_pdf(
            project / "sources" / "evidence.pdf",
            ["Cobalt heron evidence from the amber marsh."],
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        first = await service.search("cobalt heron", top_k=1)

        calls: list[str] = []
        real_flags = support_module.chunk_health_flags

        def counting_flags(_text: str, *, quality_flags: object = None) -> int:
            calls.append(_text)
            return real_flags(_text, quality_flags=quality_flags)

        monkeypatch.setattr(support_module, "chunk_health_flags", counting_flags)
        # Model a lookup that predates the stored verdict: the query path has to
        # recompute the same flags from the text, with identical results.
        monkeypatch.setattr(support_module, "LOOKUP_HEALTH_FLAGS_KEY", "_absent_key")
        second = await service.search("cobalt heron", top_k=1)

        assert calls, "the guard must recompute the verdict when none is stored"
        assert [hit["chunk_id"] for hit in second["hits"]] == [
            hit["chunk_id"] for hit in first["hits"]
        ]

    asyncio.run(exercise())


def test_partial_pdf_batch_is_replayed_after_hard_crash(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_partial_pdf_batch_is_replayed_after_hard_crash(project, monkeypatch)
    )


def test_vector_outputs_are_fsynced_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_syncs = 0
    directory_syncs: list[Path] = []
    real_fsync = os.fsync

    def tracking_fsync(descriptor: int) -> None:
        nonlocal file_syncs
        file_syncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(
        ingestion_module,
        "fsync_directory",
        lambda path: directory_syncs.append(path),
    )
    vectors = np.ones(
        (2, DEFAULT_EMBEDDING_FACTS.dimension),
        dtype=np.float32,
    )
    batches_root = tmp_path / "batches"
    batch_path = batches_root / "000000000000.npy"
    ResearchService._save_vector_batch(batch_path, vectors)
    assembled_path = tmp_path / "portable" / "embeddings.npy"
    ResearchService._assemble_vector_batches(
        assembled_path,
        batches_root,
        len(vectors),
        2,
        DEFAULT_EMBEDDING_FACTS.dimension,
    )

    assert file_syncs == 2
    assert directory_syncs == [batch_path.parent, assembled_path.parent]
    np.testing.assert_array_equal(
        np.load(assembled_path, allow_pickle=False),
        vectors,
    )


async def _assert_checkpointless_staging_is_cleaned_before_ingest(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    orphan = config.staging_root / "interrupted-before-checkpoint"
    orphan.mkdir()
    (orphan / ".checkpoint.tmp").write_text("partial", encoding="utf-8")
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    result = await service.ingest(chunk_size=50, chunk_overlap=10)

    assert result["status"] == "ready"
    assert not orphan.exists()
    diagnostic = json.loads(
        (config.failures_root / "interrupted-before-checkpoint-orphan.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["resumable"] is False
    assert "checkpointless" in diagnostic["error"]


def test_checkpointless_staging_is_cleaned_before_ingest(project: Path) -> None:
    asyncio.run(_assert_checkpointless_staging_is_cleaned_before_ingest(project))


async def _assert_bounded_ingestion_resumes_after_service_restart(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()

    first_service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    result = await first_service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert result["status"] == "in_progress"
    assert result["phase"] == "source_hashing"
    assert not config.current_path.exists()

    status = await first_service.status()
    assert status["ingestion_progress"]["build_id"] == result["build_id"]

    resumed_service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    for _ in range(20):
        result = await resumed_service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if result["status"] != "in_progress":
            break
        assert not config.current_path.exists()
    else:  # pragma: no cover - protects the bounded state machine from stalling.
        raise AssertionError("bounded ingestion did not finish")

    assert result["status"] == "ready"
    assert result["resumed"] is True
    assert config.current_path.exists()
    assert not any(config.staging_root.iterdir())
    assert (await resumed_service.status())["ingestion_progress"] is None


def test_bounded_ingestion_resumes_after_service_restart(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_bounded_ingestion_resumes_after_service_restart(project, monkeypatch)
    )


async def _assert_metadata_edit_preserves_ingestion_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Stable cobalt evidence."],
        title="Automatic Title",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    pending = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert pending["status"] == "in_progress"
    write_reviewed_metadata(
        config,
        "article.pdf",
        {"title": "Reviewed During Build", "categories": ["theory"]},
    )

    for _ in range(30):
        result = await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if result["status"] == "ready":
            break
        assert result["build_id"] == pending["build_id"]
    else:  # pragma: no cover
        raise AssertionError("metadata-independent checkpoint did not finish")

    assert result["generation_id"] == pending["build_id"]
    manifest = json.loads(
        (Path(result["generation_root"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata_storage_policy"] == (
        service_module.METADATA_STORAGE_POLICY
    )
    assert manifest["documents"][0]["title"] == "Automatic Title"
    listed = await service.list_sources()
    assert listed["sources"][0]["title"] == "Reviewed During Build"
    assert not any(config.failures_root.iterdir())


def test_metadata_edit_preserves_ingestion_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_metadata_edit_preserves_ingestion_checkpoint(project, monkeypatch)
    )


async def _assert_selected_generation_remains_searchable_during_build(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable cobalt evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )
    selected = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(project / "sources" / "new.pdf", ["New amber evidence."])

    pending = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    search = await service.search("stable cobalt", top_k=1)
    status = await service.status()

    assert pending["status"] == "in_progress"
    assert search["generation_id"] == selected["generation_id"]
    assert search["hits"][0]["chunk_id"]
    assert status["generation_id"] == selected["generation_id"]
    assert status["ingestion_progress"]["build_id"] == pending["build_id"]


def test_selected_generation_remains_searchable_during_build(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_selected_generation_remains_searchable_during_build(
            project,
            monkeypatch,
        )
    )


async def _assert_cancelled_ingestion_preserves_its_checkpoint(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = CancelOnceUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.ingest(chunk_size=50, chunk_overlap=10)

    staging_roots = list(config.staging_root.iterdir())
    assert len(staging_roots) == 1
    checkpoint = json.loads(
        (staging_roots[0] / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "chunking"
    assert checkpoint["last_interruption"] == "cancelled"
    assert not config.current_path.exists()

    resumed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert resumed["status"] == "ready"
    assert resumed["resumed"] is True


def test_cancelled_ingestion_preserves_its_checkpoint(project: Path) -> None:
    asyncio.run(_assert_cancelled_ingestion_preserves_its_checkpoint(project))


def test_a_build_replaced_by_a_source_change_says_so(project: Path) -> None:
    """Progress that restarts has to say why.

    A checkpoint cannot be resumed once the corpus changed, so the build behind it
    is discarded. Unreported, a caller looping on ingest reads that as its own
    mistake and repeats the same call, which is exactly what an agent did for ten
    hours while sources were being added.
    """

    async def exercise() -> None:
        write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            CancelOnceUltraRAG(),
            dense=FakeDenseBackend(),
        )
        with pytest.raises(asyncio.CancelledError):
            await service.ingest(chunk_size=50, chunk_overlap=10)
        staged = json.loads(
            (next(config.staging_root.iterdir()) / "checkpoint.json").read_text(
                encoding="utf-8"
            )
        )

        write_pdf(project / "sources" / "added.pdf", ["Additional evidence."])
        result = await service.ingest(chunk_size=50, chunk_overlap=10)

        assert result["superseded_build"]["build_id"] == staged["build_id"]
        assert result["superseded_build"]["phase"] == staged["phase"]
        assert "discarded" in str(result["message"])

    asyncio.run(exercise())


@pytest.mark.parametrize("reuse_existing", [False, True])
def test_restart_reconciles_completed_extraction_before_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing: bool,
) -> None:
    async def exercise() -> None:
        write_pdf(project / "sources" / "article.pdf", ["Stable evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        ultrarag = FakeUltraRAG()
        dense = FakeDenseBackend()
        if reuse_existing:
            initial_service = ResearchService(  # type: ignore[arg-type]
                config,
                ultrarag,
                dense=dense,
            )
            await initial_service.ingest(chunk_size=50, chunk_overlap=10)
            write_pdf(project / "sources" / "new.pdf", ["New evidence."])

        service = ResearchService(  # type: ignore[arg-type]
            config,
            ultrarag,
            dense=dense,
        )
        original_write = service._write_checkpoint
        crashed = False

        class SimulatedProcessExit(BaseException):
            pass

        def fail_after_completed_state(
            root: Path,
            checkpoint: dict[str, object],
        ) -> None:
            nonlocal crashed
            if (
                not crashed
                and checkpoint.get("phase") == "extraction"
                and checkpoint.get("extracted_source_paths")
                and any(
                    json.loads(path.read_text(encoding="utf-8")).get("extraction_stage")
                    == "complete"
                    for path in (root / "work" / "sources").glob("*/state.json")
                )
            ):
                crashed = True
                raise SimulatedProcessExit
            original_write(root, checkpoint)

        with monkeypatch.context() as patcher:
            patcher.setattr(service, "_write_checkpoint", fail_after_completed_state)
            with pytest.raises(SimulatedProcessExit):
                await service.ingest(chunk_size=50, chunk_overlap=10)

        staging_root = next(config.staging_root.iterdir())
        stored_checkpoint = json.loads(
            (staging_root / "checkpoint.json").read_text(encoding="utf-8")
        )
        assert stored_checkpoint["extracted_source_paths"] == []

        resumed_service = ResearchService(  # type: ignore[arg-type]
            config,
            ultrarag,
            dense=dense,
        )
        resumed = await resumed_service.ingest(chunk_size=50, chunk_overlap=10)
        assert resumed["status"] == "ready"
        assert resumed["resumed"] is True
        assert resumed["reused_document_count"] == int(reuse_existing)
        assert resumed["rebuilt_document_count"] == 1

    asyncio.run(exercise())


async def _assert_incompatible_force_mode_supersedes_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    normal = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
        work_budget_seconds=0,
    )
    forced_resume = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
        work_budget_seconds=0,
    )

    assert normal["build_id"] != forced["build_id"]
    assert forced_resume["build_id"] == forced["build_id"]
    failure = json.loads(
        (config.failures_root / f"{normal['build_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "could not be resumed" in failure["error"]


def test_incompatible_force_mode_supersedes_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_incompatible_force_mode_supersedes_checkpoint(project, monkeypatch)
    )


async def _assert_final_revalidation_catches_same_stat_mutation(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Stable research evidence."])
    original_stat = source.stat()
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    for _ in range(30):
        result = await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if (
            result["phase"] == "dense_indexing"
            and result["progress"]["completed"] == result["progress"]["total"]
        ):
            break
    else:  # pragma: no cover
        raise AssertionError("ingestion did not reach final source revalidation")

    original_bytes = source.read_bytes()
    mutated = bytearray(original_bytes)
    mutated[-1] = 32 if mutated[-1] != 32 else 10
    source.write_bytes(mutated)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns

    revalidated = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert revalidated["build_id"] == result["build_id"]
    assert revalidated["phase"] == "source_revalidation"
    restarted = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert restarted["status"] == "in_progress"
    assert restarted["build_id"] != result["build_id"]
    assert "Source bytes changed" in restarted["message"]
    assert not config.current_path.exists()


def test_final_revalidation_catches_same_stat_mutation(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_final_revalidation_catches_same_stat_mutation(project, monkeypatch)
    )


async def _assert_bounded_and_single_call_builds_are_equivalent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_project = project / "single"
    bounded_project = project / "bounded"
    (single_project / "sources").mkdir(parents=True)
    (bounded_project / "sources").mkdir(parents=True)
    source = single_project / "sources" / "evidence.pdf"
    write_pdf(
        source,
        ["First page cobalt evidence.", "Second page amber evidence."],
    )
    (bounded_project / "sources" / "evidence.pdf").write_bytes(source.read_bytes())

    single_config = resolve_config(single_project, vanilla_executable=sys.executable)
    bounded_config = resolve_config(bounded_project, vanilla_executable=sys.executable)
    single = ResearchService(  # type: ignore[arg-type]
        single_config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    bounded = ResearchService(  # type: ignore[arg-type]
        bounded_config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    single_result = await single.ingest(chunk_size=50, chunk_overlap=10)
    for _ in range(40):
        bounded_result = await bounded.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if bounded_result["status"] == "ready":
            break
    else:  # pragma: no cover
        raise AssertionError("bounded equivalent build did not finish")

    single_root = Path(single_result["generation_root"])
    bounded_root = Path(bounded_result["generation_root"])

    def without_project_scoped_id(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {key: value for key, value in item.items() if key != "source_id"}
            for item in records
        ]

    assert without_project_scoped_id(
        read_jsonl(single_root / "corpus" / "extracted-units.jsonl")
    ) == without_project_scoped_id(
        read_jsonl(bounded_root / "corpus" / "extracted-units.jsonl")
    )
    assert without_project_scoped_id(
        read_jsonl(single_root / "chunks" / "chunks.jsonl")
    ) == without_project_scoped_id(read_jsonl(bounded_root / "chunks" / "chunks.jsonl"))
    assert np.array_equal(
        np.load(single_root / "portable" / "embeddings.npy", allow_pickle=False),
        np.load(bounded_root / "portable" / "embeddings.npy", allow_pickle=False),
    )


def test_bounded_and_single_call_builds_are_equivalent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_bounded_and_single_call_builds_are_equivalent(project, monkeypatch)
    )


async def _assert_invalid_activation_journals_self_heal(project: Path) -> None:
    write_pdf(project / "sources" / "evidence.pdf", ["Recoverable evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    journal_path = config.state_root / "pending-activation.json"

    journal_path.write_text("{not-json", encoding="utf-8")
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert first["status"] == "ready"
    assert not journal_path.exists()

    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "project_id": config.project_id,
                "build_id": "../../unsafe",
            }
        ),
        encoding="utf-8",
    )
    second = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert second["status"] == "unchanged"
    assert not journal_path.exists()
    diagnostics = list(config.failures_root.glob("*-invalid-activation-journal.json"))
    assert len(diagnostics) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["resumable"] is False
        for path in diagnostics
    )


def test_invalid_activation_journals_do_not_block_ingestion(project: Path) -> None:
    asyncio.run(_assert_invalid_activation_journals_self_heal(project))


@pytest.mark.parametrize(
    "failure_point",
    ["before_journal", "before_move", "before_pointer"],
)
def test_pending_activation_recovers_crash_window(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    async def exercise() -> None:
        write_pdf(project / "sources" / "evidence.pdf", ["Recoverable evidence."])
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )

        if failure_point == "before_journal":

            class SimulatedProcessExit(BaseException):
                pass

            original_atomic_write = ingestion_module.atomic_write_json

            def fail_journal(path: Path, value: object, **kwargs: object) -> None:
                if path == config.state_root / "pending-activation.json":
                    raise SimulatedProcessExit
                original_atomic_write(path, value, **kwargs)

            with monkeypatch.context() as patcher:
                patcher.setattr(ingestion_module, "atomic_write_json", fail_journal)
                with pytest.raises(SimulatedProcessExit):
                    await service.ingest(chunk_size=50, chunk_overlap=10)
            staging_root = next(config.staging_root.iterdir())
            assert (staging_root / "work").is_dir()
        elif failure_point == "before_move":
            original_replace = os.replace

            def fail_generation_move(source: object, destination: object) -> None:
                if Path(destination).parent == config.generations_root:
                    raise OSError("simulated crash before generation move")
                original_replace(source, destination)

            with monkeypatch.context() as patcher:
                patcher.setattr(os, "replace", fail_generation_move)
                with pytest.raises(OSError, match="before generation move"):
                    await service.ingest(chunk_size=50, chunk_overlap=10)
        else:
            original_atomic_write = ingestion_module.atomic_write_json

            def fail_pointer(path: Path, value: object, **kwargs: object) -> None:
                if path == config.current_path:
                    raise OSError("simulated crash before pointer write")
                original_atomic_write(path, value, **kwargs)

            with monkeypatch.context() as patcher:
                patcher.setattr(ingestion_module, "atomic_write_json", fail_pointer)
                with pytest.raises(OSError, match="before pointer write"):
                    await service.ingest(chunk_size=50, chunk_overlap=10)

        assert (config.state_root / "pending-activation.json").is_file() is (
            failure_point != "before_journal"
        )
        recovered_service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        recovered = await recovered_service.ingest(
            chunk_size=50,
            chunk_overlap=10,
        )
        assert recovered["status"] == "ready"
        assert recovered.get("activation_recovered", False) is (
            failure_point != "before_journal"
        )
        assert not (config.state_root / "pending-activation.json").exists()
        generation_root = Path(recovered["generation_root"])
        assert not (generation_root / "checkpoint.json").exists()
        assert (
            json.loads(config.current_path.read_text(encoding="utf-8"))["generation_id"]
            == recovered["generation_id"]
        )
        search = await recovered_service.search(
            "recoverable evidence",
            retrieval_method="bm25",
        )
        assert search["result_count"] == 1

    asyncio.run(exercise())


async def _assert_resumed_dense_batches_are_exactly_once(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = ManyChunkUltraRAG()
    dense = FakeDenseBackend()

    for _ in range(60):
        service = ResearchService(  # type: ignore[arg-type]
            config,
            ultrarag,
            dense=dense,
        )
        result = await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if result["status"] == "ready":
            break
    else:  # pragma: no cover
        raise AssertionError("multi-batch ingestion did not finish")

    assert result["chunk_count"] == 130
    assert dense.embed_calls == 3
    assert dense.embedded_text_count == 130
    assert dense.upload_batches == [(0, 64), (64, 64), (128, 2)]
    assert len(dense.chunks) == 130
    assert len({str(chunk["chunk_id"]) for chunk in dense.chunks}) == 130


def test_resumed_dense_batches_are_exactly_once(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_assert_resumed_dense_batches_are_exactly_once(project, monkeypatch))


async def _assert_legacy_corrupt_chunks_are_rejected_immediately(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    result = await service.ingest(chunk_size=50, chunk_overlap=10)
    generation_root = Path(result["generation_root"])
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    chunk_id = str(chunks[0]["chunk_id"])
    clean_chunk = dict(chunks[0])
    clean_chunk["chunk_id"] = "clean-neighbor"
    clean_chunk["id"] = "clean-neighbor"
    clean_chunk["document_chunk_index"] = 0
    corrupt_chunk = dict(chunks[0])
    corrupt_chunk["document_chunk_index"] = 1
    corrupt_chunk["contents"] = CORRUPT_TEXT
    symbol_chunk = dict(chunks[0])
    symbol_chunk["chunk_id"] = "symbol-only"
    symbol_chunk["id"] = "symbol-only"
    symbol_chunk["document_chunk_index"] = 2
    symbol_chunk["contents"] = "— • ∎"
    chunks = [clean_chunk, corrupt_chunk, symbol_chunk]
    write_jsonl(chunks_path, chunks)
    ultrarag.passages = [
        CORRUPT_TEXT,
        str(symbol_chunk["contents"]),
        str(clean_chunk["contents"]),
    ]
    dense.chunks = [corrupt_chunk, symbol_chunk]

    for retrieval_method in ("bm25", "dense", "hybrid"):
        search = await service.search(
            "evidence",
            top_k=8,
            retrieval_method=retrieval_method,
            rerank=True,
        )
        assert all(
            hit["chunk_id"] not in {chunk_id, "symbol-only"} for hit in search["hits"]
        )
        if retrieval_method in {"bm25", "hybrid"}:
            assert search["rejected_candidates"]["bm25_corrupt_text"] == 1
            assert search["rejected_candidates"]["bm25_extraction_artifact"] == 1
        if retrieval_method in {"dense", "hybrid"}:
            assert search["rejected_candidates"]["dense_corrupt_text"] == 1
            assert search["rejected_candidates"]["dense_extraction_artifact"] == 1
    with pytest.raises(ResearchError, match="corrupt extracted text"):
        await service.get_passage(chunk_id)
    with pytest.raises(ResearchError, match="extraction artifact"):
        await service.get_passage("symbol-only")
    passage = await service.get_passage("clean-neighbor", context_chunks=1)
    assert [item["chunk_id"] for item in passage["context"]] == ["clean-neighbor"]


def test_legacy_corrupt_chunks_are_rejected_immediately(project: Path) -> None:
    asyncio.run(_assert_legacy_corrupt_chunks_are_rejected_immediately(project))


async def _assert_foreign_script_chunks_stay_retrievable(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    result = await service.ingest(chunk_size=50, chunk_overlap=10)
    generation_root = Path(result["generation_root"])
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    base = read_jsonl(chunks_path)[0]

    quotation = dict(base)
    quotation["chunk_id"] = "greek-quotation"
    quotation["id"] = "greek-quotation"
    quotation["document_chunk_index"] = 0
    quotation["contents"] = "ὁ ἄργυρος κακὸν νόμισμ᾽ ἔβλαστε καὶ πόλεις πορθεῖ �"

    corrupt_chunk = dict(base)
    corrupt_chunk["document_chunk_index"] = 1
    corrupt_chunk["contents"] = CORRUPT_TEXT

    write_jsonl(chunks_path, [quotation, corrupt_chunk])
    ultrarag.passages = [str(quotation["contents"]), CORRUPT_TEXT]

    search = await service.search("ἄργυρος", top_k=8, retrieval_method="bm25")
    hit = next(item for item in search["hits"] if item["chunk_id"] == "greek-quotation")
    assert hit["text_notes"] == ["non_latin_dominant"]
    assert search["withheld_candidates"]["policy"] == "corruption_evidence_only"
    assert search["withheld_candidates"]["flagged_passages_returned"] == 1
    assert search["withheld_candidates"]["total"] == 2
    reasons = search["withheld_candidates"]["reasons"]
    assert reasons["replacement_characters"]["count"] == 1
    assert reasons["private_or_unassigned_characters"]["count"] == 1
    assert reasons["replacement_characters"]["example_chunk_ids"]

    passage = await service.get_passage("greek-quotation")
    assert [item["chunk_id"] for item in passage["context"]] == ["greek-quotation"]


def test_foreign_script_chunks_stay_retrievable(project: Path) -> None:
    asyncio.run(_assert_foreign_script_chunks_stay_retrievable(project))


async def _assert_legacy_text_chunks_remain_searchable(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Legacy cobalt evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )
    result = await service.ingest(chunk_size=50, chunk_overlap=10)
    generation_root = Path(result["generation_root"])
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    legacy_text = chunks[0].pop("contents")
    chunks[0]["text"] = legacy_text
    write_jsonl(chunks_path, chunks)
    ultrarag.passages = [str(legacy_text)]

    search = await service.search("legacy cobalt", retrieval_method="bm25")
    assert search["hits"][0]["text"] == legacy_text
    passage = await service.get_passage(str(chunks[0]["chunk_id"]), context_chunks=0)
    assert passage["context"][0]["text"] == legacy_text


def test_legacy_text_chunks_remain_searchable(project: Path) -> None:
    asyncio.run(_assert_legacy_text_chunks_remain_searchable(project))


async def _assert_unreadable_source_fails_without_activation(project: Path) -> None:
    write_epub(project / "sources" / "broken.epub", CORRUPT_TEXT)
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    with pytest.raises(ExtractionError, match="no readable English-oriented text"):
        await service.ingest(chunk_size=50, chunk_overlap=10)

    assert not config.current_path.exists()
    assert not any(config.staging_root.iterdir())
    failures = list(config.failures_root.glob("*.json"))
    assert len(failures) == 1
    assert CORRUPT_TEXT not in failures[0].read_text(encoding="utf-8")


def test_unreadable_source_fails_without_activation(project: Path) -> None:
    asyncio.run(_assert_unreadable_source_fails_without_activation(project))


async def _assert_query_paths_use_generation_lookup(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        [
            "Cobalt evidence in the first passage.",
            "Cobalt context in the neighboring passage.",
        ],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    result = await service.ingest(chunk_size=100, chunk_overlap=10)
    generation_root = Path(result["generation_root"])
    lookup_path = generation_root / "indexes" / "artifact-lookup.sqlite3"
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    assert lookup_path.is_file()
    assert chunks_path.is_file()

    # A generation created before the sidecar existed is upgraded lazily: the
    # first query after the sidecar is gone rescans the artifact files once and
    # writes it.
    lookup_path.unlink()
    search = await service.search(
        "cobalt",
        top_k=1,
        retrieval_method="dense",
    )
    assert search["hits"]
    assert lookup_path.is_file()

    # Every later query seeks the records it returns instead of scanning the
    # store again. The property is enforced where the scan actually happens: the
    # query path reads byte ranges through the lookup, so a rescanned chunk store
    # can only mean `build_artifact_lookup` ran for a current generation.
    def refuse_rescan(*_: Any, **__: Any) -> Any:
        raise AssertionError("a query path rescanned the complete chunk store")

    monkeypatch.setattr(artifact_lookup_module, "_index_chunks", refuse_rescan)
    monkeypatch.setattr(artifact_lookup_module, "_index_units", refuse_rescan)
    steady = await service.search(
        "cobalt",
        top_k=1,
        retrieval_method="dense",
    )
    assert steady["hits"]
    passage = await service.get_passage(
        steady["hits"][0]["chunk_id"],
        context_chunks=1,
    )
    assert passage["context"]

    # The guard proves itself: with the sidecar gone, the same patch has to fire,
    # which is what makes the steady-state silence above mean anything.
    lookup_path.unlink()
    with pytest.raises(AssertionError, match="rescanned the complete chunk store"):
        await service.search(
            "cobalt",
            top_k=1,
            retrieval_method="dense",
        )


def test_query_paths_use_generation_lookup(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_assert_query_paths_use_generation_lookup(project, monkeypatch))


async def _assert_filtered_bm25_deepens_without_loading_the_corpus(
    project: Path,
) -> None:
    write_pdf(
        project / "sources" / "a-distractor.pdf",
        ["Distractor source."],
        title="Distractor",
    )
    write_pdf(
        project / "sources" / "b-target.pdf",
        ["Target source."],
        title="Target",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = ProgressivelyFilteredUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )
    write_reviewed_metadata(config, "b-target.pdf", {"categories": ["selected"]})
    await service.ingest(chunk_size=100, chunk_overlap=10)

    ultrarag.search_depths.clear()
    search = await service.search(
        "cobalt",
        top_k=5,
        categories_any=["selected"],
        retrieval_method="bm25",
    )

    assert len(search["hits"]) == 5
    assert {hit["title"] for hit in search["hits"]} == {"Target"}
    assert ultrarag.search_depths == [20, 35]


def test_filtered_bm25_deepens_without_loading_the_corpus(project: Path) -> None:
    asyncio.run(_assert_filtered_bm25_deepens_without_loading_the_corpus(project))


async def _assert_status_lists_retained_generations(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence about labour."],
        title="Article",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(
        project / "sources" / "second.pdf",
        ["Amber evidence in a second source."],
        title="Second",
    )
    second = await service.ingest(chunk_size=50, chunk_overlap=10)

    status = await service.status()

    records = status["generations"]
    assert records[0]["generation_id"] == second["generation_id"]
    assert {item["generation_id"] for item in records} == {
        first["generation_id"],
        second["generation_id"],
    }
    assert status["retained_generation_count"] == 2
    assert status["retained_generation_bytes"] == sum(
        item["size_bytes"] for item in records
    )
    assert all(item["file_count"] > 0 and item["size_bytes"] > 0 for item in records)
    by_id = {item["generation_id"]: item for item in records}
    assert by_id[first["generation_id"]]["chunk_count"] == 1
    assert by_id[first["generation_id"]]["document_count"] == 1
    assert by_id[second["generation_id"]]["chunk_count"] == 2
    assert by_id[second["generation_id"]]["document_count"] == 2
    assert (
        by_id[first["generation_id"]]["created_at"]
        <= by_id[second["generation_id"]]["created_at"]
    )
    current = [item for item in records if item["is_current"]]
    assert len(current) == 1
    assert current[0]["generation_id"] == second["generation_id"]
    assert current[0]["schema_version"] is not None

    # A damaged or orphaned directory is reported rather than breaking status:
    # the point of the inventory is to show what occupies disk.
    damaged = config.generations_root / "20260101T000000Z-orphan"
    damaged.mkdir(parents=True)
    (damaged / "manifest.json").write_text("{ not json", encoding="utf-8")
    (damaged / "stray.bin").write_bytes(b"orphaned staging debris")

    after = await service.status()

    assert after["retained_generation_count"] == 3
    orphan = next(
        item for item in after["generations"] if item["generation_id"] == damaged.name
    )
    assert orphan["is_current"] is False
    assert "manifest_error" in orphan
    assert orphan["size_bytes"] > 0
    assert after["retained_generation_bytes"] == sum(
        item["size_bytes"] for item in after["generations"]
    )
    # The damaged directory must not make the selected generation unusable.
    assert after["ready"] is True
    assert after["generation_id"] == second["generation_id"]


def test_status_lists_retained_generations(project: Path) -> None:
    asyncio.run(_assert_status_lists_retained_generations(project))


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


async def _assert_search_falls_back_when_reranking_is_unavailable(
    project: Path,
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        [
            "Cobalt evidence about labour and artificial intelligence.",
            "Quartz material unrelated to the primary question.",
        ],
        title="Research Article",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=UnavailableRerankerDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    unranked = await service.search("cobalt labour", top_k=3)
    fell_back = await service.search("cobalt labour", top_k=3, rerank=True)

    # The requested reranking could not run, so the search still succeeds with
    # the plain candidate order and says exactly why.
    assert fell_back["rerank_requested"] is True
    assert fell_back["reranked"] is False
    assert fell_back["rerank_fallback"]["reason"] == "reranker_model_unavailable"
    assert fell_back["rerank_fallback"]["effect"] == "unranked_candidate_order_returned"
    assert (
        "not present in the shared model cache"
        in fell_back["rerank_fallback"]["message"]
    )
    assert fell_back["reranker_model"] is None
    assert fell_back["reranker_model_revision"] is None
    assert [hit["chunk_id"] for hit in fell_back["hits"]] == [
        hit["chunk_id"] for hit in unranked["hits"]
    ]
    assert all(hit["rerank_score"] is None for hit in fell_back["hits"])

    # A search that does not ask for reranking neither loads the model nor
    # reports a fallback.
    assert unranked["rerank_requested"] is False
    assert unranked["reranked"] is False
    assert unranked["rerank_fallback"] is None


def test_search_falls_back_when_reranking_is_unavailable(project: Path) -> None:
    asyncio.run(_assert_search_falls_back_when_reranking_is_unavailable(project))


async def _assert_search_reranks_with_a_named_model(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        [
            "Cobalt evidence about labour and artificial intelligence.",
            "Quartz material unrelated to the primary question.",
        ],
        title="Research Article",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    default = await service.search("cobalt labour", top_k=3, rerank=True)
    named = await service.search(
        "cobalt labour",
        top_k=3,
        rerank=True,
        rerank_model="jinaai/jina-reranker-v1-turbo-en",
    )

    # A tool call leaves the model to the engine, and the answer names the model
    # and the revision that actually ran.
    assert default["reranker_model"] == DEFAULT_RERANKER_MODEL
    assert default["reranker_model_revision"] == RERANKER_MODELS[DEFAULT_RERANKER_MODEL]
    assert named["reranker_model"] == "jinaai/jina-reranker-v1-turbo-en"
    assert (
        named["reranker_model_revision"]
        == RERANKER_MODELS["jinaai/jina-reranker-v1-turbo-en"]
    )
    # Only the call that named a model switched it: a comparison run does not
    # change what the next tool call uses.
    assert dense.rerank_models == [None, "jinaai/jina-reranker-v1-turbo-en"]


def test_search_reranks_with_a_named_model(project: Path) -> None:
    asyncio.run(_assert_search_reranks_with_a_named_model(project))


async def _assert_search_rejects_an_unsupported_reranker_model(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence about labour and artificial intelligence."],
        title="Research Article",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    with pytest.raises(ResearchError, match="Unsupported reranker model"):
        await service.search(
            "cobalt labour",
            top_k=3,
            rerank=True,
            rerank_model="some-org/some-reranker",
        )
    with pytest.raises(ResearchError, match="rerank_model requires rerank=True"):
        await service.search(
            "cobalt labour",
            top_k=3,
            rerank_model="jinaai/jina-reranker-v1-turbo-en",
        )


def test_search_rejects_an_unsupported_reranker_model(project: Path) -> None:
    asyncio.run(_assert_search_rejects_an_unsupported_reranker_model(project))


async def _assert_the_bm25_language_reaches_the_gateway(project: Path) -> None:
    write_pdf(
        project / "sources" / "artikel.pdf",
        ["Ein deutscher Absatz ueber Arbeit und Technik."],
        title="Aufsatz",
    )
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["language.corpus=de"],
    )
    gateway = FakeUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        gateway,
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    # The corpus language is not decoration: the lexical index is built with it.
    assert gateway.bm25_language == "de"
    assert config.settings.language_corpus == "de"


def test_the_bm25_language_reaches_the_gateway(project: Path) -> None:
    asyncio.run(_assert_the_bm25_language_reaches_the_gateway(project))


def test_dense_backends_embed_with_the_configured_model(tmp_path: Path) -> None:
    """A non-default embedding model has to reach the backends, not just the record.

    A generation records the dimension of the model it was built with, so a
    backend still embedding with the shipped default fills an index of the wrong
    width: the German model returns 768 values where the default returns 384, and
    the width check is all that stands between that and a mismatched index.
    """
    project = tmp_path / "project"
    overlay = project / ".research-rag"
    overlay.mkdir(parents=True)
    (overlay / "config.toml").write_text(
        '[language]\ncorpus = "de"\n\n[dense]\n'
        'embedding_model = "jinaai/jina-embeddings-v2-base-de"\n',
        encoding="utf-8",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    assert config.settings.embedding_dimension == 768

    service = ResearchService(config, FakeUltraRAG())  # type: ignore[arg-type]
    assert service._dense_backends
    for name, backend in service._dense_backends.items():
        dimension = backend.embedding_facts.dimension  # type: ignore[attr-defined]
        assert dimension == config.settings.embedding_dimension, name


def test_the_reranked_window_follows_its_settings(project: Path) -> None:
    """The window is a setting, which is what makes a deeper one measurable."""

    paragraphs = [
        (
            "Commodity fetishism describes how relations between people appear as "
            "relations between things, so that value seems to belong to the object "
            "rather than to the labour that produced it, and the analysis has to "
            "begin from that appearance."
        ),
        (
            "The mysterious character of the commodity arises from the form of "
            "value rather than from its use, and the inversion is practical rather "
            "than merely mistaken, which is why criticism starts at the surface."
        ),
        (
            "Later writers extend the argument to attention, data, and platforms, "
            "where the same inversion reappears between users and the systems that "
            "measure them and turn their activity into a resource."
        ),
        (
            "The study of labour under digital conditions asks how work is made "
            "visible to capital, and which forms of effort remain unmeasured and "
            "therefore unpaid in the accounts that firms keep."
        ),
        (
            "Colonial histories of extraction connect the factory to the mine, and "
            "the mine to the plantation, across a single circuit of accumulation "
            "that reorganised labour on several continents."
        ),
        (
            "Waste studies follow what is discarded, showing that the remainder of "
            "production is not outside the economy but one of its conditions, and "
            "a source of value in its own right."
        ),
        (
            "Fetishism is not only an error of perception: it is a form of social "
            "organisation that the analysis has to describe from the inside, in "
            "the language its participants already use."
        ),
        (
            "An account of appearance begins with the surface, because the surface "
            "is where the inversion is lived, and where any criticism that wants "
            "to be more than moralising has to start."
        ),
    ]

    async def exercise() -> None:
        source = project / "sources" / "article.pdf"
        write_pdf(source, paragraphs, title="Window Article")

        base = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            base, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        built = await service.ingest(chunk_size=50, chunk_overlap=5)
        count = built["chunk_count"]
        assert count >= 6, count

        async def window(overrides: list[str], *, top_k: int) -> int:
            config = resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=overrides,
            )
            scoped = ResearchService(  # type: ignore[arg-type]
                config, FakeUltraRAG(), dense=FakeDenseBackend()
            )
            result = await scoped.search(
                "commodity fetishism labour", top_k=top_k, rerank=True
            )
            return result["rerank_window"]

        # The corpus and the relevance gates cap the window, so the observed
        # default is the budget these settings are tested against. Each one is
        # exercised below that default, which is what proves it is read rather
        # than assumed: with the setting ignored every value here would stay at
        # the default.
        default = await window([], top_k=1)
        assert default >= 3, default
        assert (
            await window(
                [
                    "retrieval.rerank_window_multiple=1",
                    "retrieval.rerank_window_floor=1",
                ],
                top_k=1,
            )
            == 1
        )
        assert await window(["retrieval.rerank_window_floor=3"], top_k=1) == 3
        assert await window(["retrieval.rerank_max_candidates=2"], top_k=1) == 2
        # top_k=3 with a multiple of 1 gives 3, where the shipped default gives
        # the floor of 10 and would therefore be capped by the corpus instead.
        assert (
            await window(
                [
                    "retrieval.rerank_window_multiple=1",
                    "retrieval.rerank_window_floor=1",
                ],
                top_k=3,
            )
            == 3
        )

    asyncio.run(exercise())


def test_a_ranking_change_reuses_chunks_and_vectors(project: Path) -> None:
    """A ranking value decides how a generation is searched, not what it holds."""

    async def exercise() -> None:
        source = project / "sources" / "article.pdf"
        write_pdf(
            source,
            [
                (
                    "Commodity fetishism describes how relations between people "
                    "appear as relations between things, so value seems to belong "
                    "to the object rather than to the labour that produced it."
                ),
                (
                    "The form of value, not its use, produces the mysterious "
                    "character of the commodity, and the inversion is practical "
                    "rather than merely mistaken."
                ),
            ],
            title="Reuse Article",
        )
        base = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            base, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        first = await service.ingest(chunk_size=50, chunk_overlap=10)
        assert first["chunk_count"] >= 2, first["chunk_count"]

        # The same corpus under a different ranking policy: rrf_k is a ranking
        # value, so it must not invalidate a single chunk or vector.
        changed_config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["retrieval.rrf_k=30"],
        )
        changed = ResearchService(  # type: ignore[arg-type]
            changed_config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        second = await changed.ingest(chunk_size=50, chunk_overlap=10)

        assert second["generation_changed"] is True
        assert second["rebuilt_document_count"] == 0
        assert second["reused_chunk_count"] == second["chunk_count"]
        assert second["rebuilt_chunk_count"] == 0
        assert second["created_vector_count"] == 0

    asyncio.run(exercise())


def test_pseudo_relevance_feedback_expands_the_lexical_query(project: Path) -> None:
    """Off by default; on, it reports the terms it took from the leaders."""

    async def exercise() -> None:
        source = project / "sources" / "article.pdf"
        write_pdf(
            source,
            [
                (
                    "Commodity fetishism describes how relations between people "
                    "appear as relations between things, so that value seems to "
                    "belong to the object rather than to the labour which produced "
                    "it, and the analysis begins from that appearance."
                ),
                (
                    "The mysterious character of the commodity arises from the "
                    "form of value rather than from its use, and the inversion is "
                    "practical rather than merely mistaken, which is why criticism "
                    "starts at the surface it describes."
                ),
            ],
            title="Feedback Article",
        )
        base = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            base, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        await service.ingest(chunk_size=50, chunk_overlap=5)

        off = await service.search(
            "commodity fetishism", top_k=3, retrieval_method="bm25"
        )
        assert off["prf_requested"] is False
        assert off["prf_terms"] == []

        config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["retrieval.prf=true", "retrieval.prf_terms=3"],
        )
        scoped = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        on = await scoped.search(
            "commodity fetishism", top_k=3, retrieval_method="bm25"
        )

        assert on["prf_requested"] is True
        assert 1 <= len(on["prf_terms"]) <= 3
        # Terms the query already contains are never re-added: they would change
        # nothing and would hide whether the expansion did anything.
        assert not {"commodity", "fetishism"} & set(on["prf_terms"])

    asyncio.run(exercise())


def test_document_frequencies_count_a_term_once_per_text() -> None:
    """A passage repeating a word must not make it look common."""

    frequencies = support_module.document_frequencies(
        ["alpha alpha alpha beta", "beta gamma"],
        ENGLISH_GATE_STOPWORDS,
    )

    assert frequencies == {"alpha": 1, "beta": 2, "gamma": 1}


def test_feedback_ranks_a_rare_term_above_one_every_passage_uses() -> None:
    """Leader support alone mines the words the leaders share, not the ones that matter."""

    ranked = support_module._pseudo_relevance_terms(
        query="fetishism",
        texts=[
            "commodity fetishism and the data centre",
            "fetishism, data, and the digital object",
            "digital data and the object of fetishism",
        ],
        maximum_terms=3,
        stopwords=ENGLISH_GATE_STOPWORDS,
        document_frequencies={
            # Corpus-wide counts, so never below the leader support above.
            "commodity": 6,
            "data": 40,
            "digital": 12,
            "fetishism": 9,
            "object": 2,
            "centre": 3,
        },
        corpus_size=40,
    )

    # "data" leads on support in all three leaders and appears in every chunk.
    assert "data" not in ranked
    assert ranked == ["object", "digital", "centre"]


def test_feedback_without_a_table_ranks_by_leader_support() -> None:
    ranked = support_module._pseudo_relevance_terms(
        query="question",
        texts=["alpha beta beta", "beta gamma"],
        maximum_terms=2,
        stopwords=ENGLISH_GATE_STOPWORDS,
    )

    assert ranked == ["beta", "alpha"]


def test_feedback_is_reproducible_for_the_same_input() -> None:
    arguments: dict[str, object] = {
        "query": "commodity fetishism",
        "texts": ["fetishism and the digital object", "the object of fetishism"],
        "maximum_terms": 3,
        "stopwords": ENGLISH_GATE_STOPWORDS,
        "document_frequencies": {"digital": 9, "object": 2},
        "corpus_size": 40,
    }

    assert support_module._pseudo_relevance_terms(
        **arguments
    ) == support_module._pseudo_relevance_terms(**arguments)  # type: ignore[arg-type]


def test_diversity_relevance_normalizes_the_scored_band() -> None:
    """The charge is a share of this ranking's own confidence, not of the pool."""

    relevance = support_module._selection_relevance(
        ["a1", "a2", "tail"],
        {"a1": 4.0, "a2": 2.0},
    )

    # A candidate the order carries without a score is last already, so it sits
    # at 0.0 next to the worst score rather than anywhere the band could raise.
    assert relevance == {"a1": 1.0, "a2": 0.0, "tail": 0.0}


def test_diversity_relevance_treats_a_flat_band_as_equal() -> None:
    """Equal scores say nothing about relative order, so the order must stand."""

    relevance = support_module._selection_relevance(
        ["a1", "b1"], {"a1": 0.5, "b1": 0.5}
    )

    assert relevance == {"a1": 1.0, "b1": 1.0}


def test_diversity_relevance_without_a_score_is_zero() -> None:
    relevance = support_module._selection_relevance(["a1", "b1"], {})

    assert relevance == {"a1": 0.0, "b1": 0.0}


def test_an_unscored_ranking_keeps_its_own_order() -> None:
    """BM25 or dense without reranking has no relevance to charge a repeat against.

    Charging anyway would leave the penalty as the only signal in the pool and
    replace that mode's ranking with a round-robin over sources, which is a
    different search rather than a reordering of this one.
    """

    selected = support_module._source_diverse_selection(
        ["a1", "a2", "b1"],
        source_id_by_chunk={"a1": "a", "a2": "a", "b1": "b"},
        scores={},
        top_k=2,
        penalty=1.0,
    )

    assert selected == ["a1", "a2"]


def test_source_diversity_brings_another_source_forward() -> None:
    """A second passage from one source loses to a first from another."""

    selected = support_module._source_diverse_selection(
        ["a1", "a2", "b1", "b2"],
        source_id_by_chunk={"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
        scores={"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0},
        top_k=3,
        penalty=0.5,
    )

    # Equal scores leave ties to the fused position, so b1 beats a2 once a2 is
    # charged, and the earlier-positioned a2 then beats b2 for the third slot.
    assert selected == ["a1", "b1", "a2"]


def test_zero_diversity_penalty_keeps_the_ranked_order() -> None:
    arguments: dict[str, object] = {
        "ordered_ids": ["a1", "a2", "b1", "b2"],
        "source_id_by_chunk": {"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
        "scores": {"a1": 3.0, "a2": 2.0, "b1": 1.0, "b2": 0.0},
        "top_k": 3,
    }

    for penalty in (0.0, -0.5):
        assert support_module._source_diverse_selection(
            **arguments,
            penalty=penalty,  # type: ignore[arg-type]
        ) == ["a1", "a2", "b1"]


def test_a_pool_no_deeper_than_the_request_is_returned_as_ranked() -> None:
    """Nothing can be swapped for anything, so the ranked order is the answer."""

    selected = support_module._source_diverse_selection(
        ["a1", "a2"],
        source_id_by_chunk={"a1": "a", "a2": "a"},
        scores={"a1": 1.0, "a2": 0.0},
        top_k=2,
        penalty=1.0,
    )

    assert selected == ["a1", "a2"]


def test_a_single_source_pool_still_fills_the_answer() -> None:
    """A project whose relevant material is one source is not answered around."""

    selected = support_module._source_diverse_selection(
        ["a1", "a2", "a3"],
        source_id_by_chunk={"a1": "a", "a2": "a", "a3": "a"},
        scores={"a1": 3.0, "a2": 2.0, "a3": 1.0},
        top_k=2,
        penalty=1.0,
    )

    assert selected == ["a1", "a2"]


def test_an_unranked_tail_yields_to_a_charged_repeat() -> None:
    """The appended tail is 0.0, so a full charge is what pushes a repeat below it."""

    selected = support_module._source_diverse_selection(
        ["a1", "a2", "tail"],
        source_id_by_chunk={"a1": "a", "a2": "a", "tail": "b"},
        scores={"a1": 2.0, "a2": 0.0},
        top_k=2,
        penalty=1.0,
    )

    assert selected == ["a1", "tail"]


def test_diversity_selection_is_reproducible_for_the_same_input() -> None:
    arguments: dict[str, object] = {
        "ordered_ids": ["a1", "b1", "a2", "b2", "c1"],
        "source_id_by_chunk": {
            "a1": "a",
            "a2": "a",
            "b1": "b",
            "b2": "b",
            "c1": "c",
        },
        "scores": {"a1": 5.0, "b1": 4.0, "a2": 3.0, "b2": 2.0, "c1": 1.0},
        "top_k": 4,
        "penalty": 0.25,
    }

    first = support_module._source_diverse_selection(**arguments)  # type: ignore[arg-type]
    second = support_module._source_diverse_selection(**arguments)  # type: ignore[arg-type]

    assert first == second


def test_a_diversity_penalty_is_reported_but_rebuilds_nothing(
    project: Path,
) -> None:
    """A reordering of ranked candidates is not a reason to rebuild a generation.

    The penalty applies to what a generation already ranked, so it changes none
    of its files and is no part of what a generation is: one build answers two
    penalties, each answer names the one it used, and the manifest is untouched.
    """

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Cobalt labour in the mine.", "Wages and the working day."],
        )
        write_pdf(
            project / "sources" / "second.pdf",
            ["Cobalt, labour, and the machinery of extraction."],
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        ingested = await service.ingest(chunk_size=50, chunk_overlap=10)
        manifest = json.loads(
            (Path(ingested["generation_root"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert "selection" not in manifest["retrieval"]

        answer = await service.search("cobalt labour", top_k=2, include_staleness=False)
        assert answer["generation_upgrade_required"] is False
        assert answer["selection_policy"] == {
            "method": "greedy_source_diversity",
            "source_diversity_penalty": 0.25,
        }

        spread = ResearchService(  # type: ignore[arg-type]
            resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=["retrieval.source_diversity_penalty=0.6"],
            ),
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        respread = await spread.search(
            "cobalt labour", top_k=2, include_staleness=False
        )
        assert respread["generation_id"] == answer["generation_id"]
        assert respread["generation_upgrade_required"] is False
        assert respread["selection_policy"] == {
            "method": "greedy_source_diversity",
            "source_diversity_penalty": 0.6,
        }

    asyncio.run(exercise())


def test_contextual_headers_reach_the_embedding_and_never_the_passage(
    project: Path,
) -> None:
    """A header is context for the dense half, not text a search returns."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            [
                (
                    "Commodity fetishism describes how relations between people "
                    "appear as relations between things, so that value seems to "
                    "belong to the object rather than to the labour which produced "
                    "it, and the analysis begins from that appearance."
                ),
                (
                    "The mysterious character of the commodity arises from the "
                    "form of value rather than from its use, and the inversion is "
                    "practical rather than merely mistaken, which is why criticism "
                    "starts at the surface it describes."
                ),
            ],
            title="Headed Article",
        )

        async def ingest_with(
            overrides: list[str],
        ) -> tuple[dict[str, object], FakeDenseBackend]:
            config = resolve_config(
                project,
                vanilla_executable=sys.executable,
                settings_overrides=overrides,
            )
            backend = FakeDenseBackend()
            service = ResearchService(  # type: ignore[arg-type]
                config, FakeUltraRAG(), dense=backend
            )
            return await service.ingest(chunk_size=50, chunk_overlap=10), backend

        plain, plain_backend = await ingest_with([])
        headed, headed_backend = await ingest_with(["chunking.headers=true"])

        # Off, every vector covers the passage and nothing else.
        assert plain_backend.embedded_texts
        assert all(
            "Headed Article" not in text for text in plain_backend.embedded_texts
        )

        # On, every vector covers a header naming the source first. The chunk set
        # is the same build, because a header changes what is embedded rather than
        # what is chunked.
        assert headed["chunk_count"] == plain["chunk_count"]
        assert headed_backend.embedded_texts
        assert all(
            text.startswith("Headed Article\n\n")
            for text in headed_backend.embedded_texts
        )

        # The vectors cover different text under the other policy, so nothing is
        # reusable across the two.
        assert headed["reused_chunk_count"] == 0
        assert headed["created_vector_count"] == headed["chunk_count"]

        # Whatever went into the vector, the returned passage is the passage.
        config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["chunking.headers=true"],
        )
        service = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        result = await service.search("commodity fetishism labour", top_k=1)
        text = str(result["hits"][0]["text"])
        assert text
        assert "Headed Article" not in text

    asyncio.run(exercise())


def test_reviewed_language_filters_a_search(project: Path) -> None:
    """A language review binds retrieval at once, without a rebuild."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "english.pdf",
            [
                (
                    "Commodity fetishism describes how relations between people "
                    "appear as relations between things, so that value seems to "
                    "belong to the object rather than to the labour which produced "
                    "it."
                )
            ],
            title="English Article",
        )
        write_pdf(
            project / "sources" / "german.pdf",
            [
                (
                    "Die Ware erscheint als Verhaeltnis zwischen Dingen, und der "
                    "Wert scheint den Dingen anzugehoeren statt der Arbeit, die "
                    "sie hervorgebracht hat."
                )
            ],
            title="German Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)

        # Reviews, not detection: this is what makes the field authoritative
        # without re-ingesting.
        await service.set_source_metadata(
            source_path="english.pdf", metadata={"language": ["en"]}
        )
        await service.set_source_metadata(
            source_path="german.pdf", metadata={"language": ["de"]}
        )

        german = await service.search(
            "Ware Wert Arbeit", top_k=10, languages_any=["de"]
        )
        assert german["filters"]["languages_any"] == ["de"]
        assert {hit["title"] for hit in german["hits"]} == {"German Article"}

        english = await service.search(
            "commodity fetishism labour", top_k=10, languages_any=["en"]
        )
        assert {hit["title"] for hit in english["hits"]} == {"English Article"}

    asyncio.run(exercise())


def test_a_busy_project_is_reported_rather_than_waited_for(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller must not sit in silence behind another process's build.

    The project lock serialises builds, which is right, but waiting one out
    outlasts the client that asked: a long wait turns a busy project into a client
    timeout while the work the caller was queued behind carries on unseen.
    """

    async def exercise() -> None:
        config = resolve_config(project, vanilla_executable=sys.executable)
        # The wait is a property of the service, so shorten it before construction.
        monkeypatch.setattr(service_module, "PROJECT_LOCK_TIMEOUT_SECONDS", 0.2)
        service = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        holder = AsyncFileLock(config.state_root / "project.lock", timeout=1)
        started = time.perf_counter()
        async with holder:
            with pytest.raises(ResearchError, match="Another research process"):
                await service.ingest(chunk_size=50, chunk_overlap=10)
        assert time.perf_counter() - started < 5

    asyncio.run(exercise())


def _aliased_dense_backend() -> FakeDenseBackend:
    """A dense backend whose vectors follow the words, so tests can aim them.

    The shared fake gives every passage the same vector, which is enough to say
    "this matches" and not enough to say "these are nearly the same" — and a
    near-duplicate threshold is about the second. This one hashes each word onto
    an axis, so two passages of the same essay land close together and two
    passages on one theme land further apart.
    """

    class AliasedDenseBackend(FakeDenseBackend):
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            rows = []
            for text in texts:
                vector = np.zeros(384, dtype=np.float32)
                for word in re.findall(r"[^\W_]+", str(text).casefold()):
                    vector[zlib.crc32(word.encode("utf-8")) % 384] += 1.0
                rows.append(vector)
            return np.vstack(rows) if rows else np.zeros((0, 384), dtype=np.float32)

    return AliasedDenseBackend()


def test_a_repeated_passage_is_collapsed_from_the_answer(project: Path) -> None:
    """Two copies of one passage in two files: the answer shows one, and says which.

    The corpus keeps both — a person who wants the essay from the second file can
    still ask for it by name, and excluding a source is not what a repetition
    deserves — so the overlap is settled where it is seen, which is the answer.
    """

    from research_rag.config import resolve_config

    async def exercise() -> tuple[dict[str, Any], list[str]]:
        write_pdf(
            project / "sources" / "essay.pdf",
            ["Labour evidence from the cobalt mine in the north of the country."],
        )
        # The same essay from another file, with the punctuation and casing a
        # second extractor produces, which is what two copies of one passage
        # actually differ by.
        write_pdf(
            project / "sources" / "book.pdf",
            ["LABOUR evidence, from the cobalt mine; in the north of the country."],
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(
            config, FakeUltraRAG(), dense=_aliased_dense_backend()
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        answer = await service.search("cobalt mine", top_k=10, rerank=False)
        listed = await service.list_sources()
        return answer, [item["source_relative_path"] for item in listed["sources"]]

    answer, sources = asyncio.run(exercise())

    # Both passages are in the corpus, and both files are still listed.
    assert set(sources) == {"essay.pdf", "book.pdf"}
    # One of them is in the answer, and the other is reported as the same thing.
    assert len(answer["hits"]) == 1
    disclosure = answer["collapsed_repetitions"]
    assert disclosure["repetitions_collapsed"] == 1
    assert disclosure["collapsed_by_same_words"] == 1
    pair = disclosure["pairs"][0]
    # The pair names the file each copy came from, because a person who sees one
    # essay arriving from two files can retire a source.
    assert {pair["source_relative_path"], pair["repeated_source_relative_path"]} == {
        "essay.pdf",
        "book.pdf",
    }
    assert pair["repeated_chunk_id"] != pair["chunk_id"]


def test_the_repetition_threshold_is_a_setting_and_the_words_are_not(
    project: Path,
) -> None:
    """What counts as a repetition is a number in a config file, and it is read.

    Three pairs at three thresholds. Words that are the same words are one
    passage whatever the number, because no number is a claim about words; the
    number decides the part cosine decides, and both directions of it are read
    from the file. A number no cosine can reach decides that nothing is close
    enough rather than switching the check off.
    """

    from research_rag.config import resolve_config

    async def search_with(overrides: list[str]) -> tuple[int, list[str]]:
        # Each case its own project, so one case's generation is never the
        # corpus the next case is asked about.
        case = project / Path(overrides[0].split("=", 1)[1])
        (case / "sources").mkdir(parents=True, exist_ok=True)
        write_pdf(
            case / "sources" / "one.pdf",
            ["The committee reviewed the ledger and approved the second draft."],
        )
        write_pdf(
            case / "sources" / "two.pdf",
            ["The committee approved the second draft after reviewing the ledger."],
        )
        config = resolve_config(
            case, vanilla_executable=sys.executable, settings_overrides=overrides
        )
        service = ResearchService(
            config, FakeUltraRAG(), dense=_aliased_dense_backend()
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        answer = await service.search("committee ledger", top_k=10, rerank=False)
        return len(answer["hits"]), [
            str(item["collapsed_by"])
            for item in answer["collapsed_repetitions"]["pairs"]
        ]

    # The same two sentences, two orders: alike enough to be one passage at a
    # threshold below how alike they are, and two passages at one above it.
    assert asyncio.run(search_with(["retrieval.duplicate_cosine=0.8"])) == (
        1,
        ["same_meaning"],
    )
    assert asyncio.run(search_with(["retrieval.duplicate_cosine=0.99"])) == (2, [])
    # A number no cosine can reach collapses nothing, and says so by not claiming
    # to: the check ran and found nothing close enough.
    assert asyncio.run(search_with(["retrieval.duplicate_cosine=2.0"])) == (2, [])


def test_two_passages_on_one_theme_are_both_shown(project: Path) -> None:
    """The rule is for a repeated passage, not for two passages on a subject."""

    from research_rag.config import resolve_config

    async def exercise() -> int:
        write_pdf(
            project / "sources" / "mining.pdf",
            ["The cobalt mine employs four hundred people in the northern town."],
        )
        write_pdf(
            project / "sources" / "ledger.pdf",
            ["The quarterly ledger records the cobalt price the northern town paid."],
        )
        config = resolve_config(
            project,
            vanilla_executable=sys.executable,
            settings_overrides=["retrieval.duplicate_cosine=0.99"],
        )
        service = ResearchService(
            config, FakeUltraRAG(), dense=_aliased_dense_backend()
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        answer = await service.search("cobalt northern", top_k=10, rerank=False)
        return len(answer["hits"])

    assert asyncio.run(exercise()) == 2


async def _two_generation_service(project: Path) -> tuple[Any, Any, str, str]:
    """A service over a project holding exactly two retained generations."""

    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence about labour."],
        title="Article",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(
        project / "sources" / "second.pdf",
        ["Amber evidence in a second source."],
        title="Second",
    )
    second = await service.ingest(chunk_size=50, chunk_overlap=10)
    return service, config, first["generation_id"], second["generation_id"]


async def _assert_use_generation_rolls_the_pointer_back(project: Path) -> None:
    """A rollback moves the pointer only after the target validated."""

    service, config, first, second = await _two_generation_service(project)

    assert second != first
    before = await service.status()
    assert before["generation_id"] == second

    switched = await service.use_generation(first)

    assert switched["status"] == "changed"
    assert switched["generation_id"] == first
    assert switched["previous_generation_id"] == second
    assert switched["effective_immediately"] is True

    status = await service.status()
    assert status["generation_id"] == first
    assert status["ready"] is True
    current = [item for item in status["generations"] if item["is_current"]]
    assert [item["generation_id"] for item in current] == [first]
    # The pointer is the only thing that moved; both generations are still on
    # disk, which is what makes the switch reversible.
    assert status["retained_generation_count"] == 2
    assert read_json(config.current_path)["generation_id"] == first


async def _assert_use_generation_reports_an_already_selected_generation(
    project: Path,
) -> None:
    """Asking for the generation in use is not an error and changes nothing."""

    service, _config, _first, second = await _two_generation_service(project)

    again = await service.use_generation(second)

    assert again["status"] == "unchanged"
    assert again["generation_id"] == second
    assert (await service.status())["generation_id"] == second


async def _assert_use_generation_refuses_a_damaged_generation(project: Path) -> None:
    """A rollback that lands on broken artifacts fails and leaves the pointer."""

    service, config, first, second = await _two_generation_service(project)
    target = config.generations_root / first
    # Break the index the activation validator probes, so the target cannot
    # pass without the guard that is the point of this test.
    vectors = target / "portable" / "embeddings.npy"
    vectors.write_bytes(b"not a vector matrix")

    with pytest.raises(ResearchError) as failure:
        await service.use_generation(first)

    assert "failed validation" in str(failure.value)
    assert read_json(config.current_path)["generation_id"] == second
    assert (await service.status())["generation_id"] == second
    assert (await service.status())["ready"] is True


async def _assert_use_generation_refuses_a_name_that_is_not_a_generation_id(
    project: Path,
) -> None:
    """A caller-supplied name never becomes a path."""

    service, _config, _first, second = await _two_generation_service(project)

    for name in ("../staging", "..", "/etc", "", "not-an-id"):
        with pytest.raises(ResearchError) as failure:
            await service.use_generation(name)
        assert "is not a generation id" in str(failure.value)
    assert (await service.status())["generation_id"] == second


async def _assert_use_generation_refuses_while_a_pending_activation_is_outstanding(
    project: Path,
) -> None:
    """A waiting activation is about to move the pointer, so a rollback must not."""

    service, config, first, second = await _two_generation_service(project)
    pending = "20260101T000000Z-aabbccdd"
    atomic_write_json(
        config.state_root / "pending-activation.json",
        {"schema_version": 1, "project_id": config.project_id, "build_id": pending},
    )

    with pytest.raises(ResearchError) as failure:
        await service.use_generation(first)

    assert "pending activation" in str(failure.value)
    assert read_json(config.current_path)["generation_id"] == second


async def _assert_remove_generation_removes_the_one_search_does_not_read(
    project: Path,
) -> None:
    """Removal frees a generation the app is not serving."""

    service, config, first, second = await _two_generation_service(project)
    doomed = config.generations_root / first
    assert doomed.is_dir()

    removed = await service.remove_generation(first, confirm=first)

    assert removed["status"] == "removed"
    assert removed["generation_id"] == first
    assert removed["freed_file_count"] > 0
    assert removed["freed_bytes"] > 0
    assert removed["recoverable_by"] == "ingest"
    assert not doomed.exists()
    status = await service.status()
    assert status["retained_generation_count"] == 1
    assert [item["generation_id"] for item in status["generations"]] == [second]
    assert status["generation_id"] == second
    assert status["ready"] is True


async def _assert_remove_generation_refuses_the_generation_in_use(
    project: Path,
) -> None:
    """Deleting the generation search reads would break every read surface."""

    service, config, _first, second = await _two_generation_service(project)

    with pytest.raises(ResearchError) as failure:
        await service.remove_generation(second, confirm=second)

    assert "the one search reads" in str(failure.value)
    assert (config.generations_root / second).is_dir()
    assert (await service.status())["ready"] is True


async def _assert_remove_generation_will_not_run_on_a_mistyped_id(
    project: Path,
) -> None:
    """The id is repeated because a listing is not a decision."""

    service, config, first, _second = await _two_generation_service(project)

    with pytest.raises(ResearchError) as failure:
        await service.remove_generation(first, confirm="y")
    assert "Pass --confirm" in str(failure.value)

    with pytest.raises(ResearchError) as failure:
        await service.remove_generation(first, confirm="")
    assert "Pass --confirm" in str(failure.value)

    assert (config.generations_root / first).is_dir()
    assert (await service.status())["retained_generation_count"] == 2


async def _assert_remove_generation_refuses_a_pending_activation_target(
    project: Path,
) -> None:
    """A generation an activation is about to move into place is not free."""

    service, config, first, _second = await _two_generation_service(project)
    pending = "20260101T000000Z-aabbccdd"
    target = config.generations_root / pending
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}", encoding="utf-8")
    atomic_write_json(
        config.state_root / "pending-activation.json",
        {"schema_version": 1, "project_id": config.project_id, "build_id": pending},
    )

    with pytest.raises(ResearchError) as failure:
        await service.remove_generation(pending, confirm=pending)

    assert "waiting to become the one search reads" in str(failure.value)
    assert target.is_dir()
    assert (config.generations_root / first).is_dir()


async def _assert_a_removed_generation_can_be_rebuilt_and_selected_again(
    project: Path,
) -> None:
    """Removal is permanent, and the stated recovery is one ingestion."""

    service, config, first, second = await _two_generation_service(project)
    await service.remove_generation(first, confirm=first)
    assert not (config.generations_root / first).exists()

    rebuilt = await service.ingest(chunk_size=50, chunk_overlap=10)

    # The build is refused as unchanged while the current generation still
    # matches the corpus, so the recovered generation is a rollback target
    # rather than a second copy of what is already in use.
    assert rebuilt["status"] in {"unchanged", "complete"}
    status = await service.status()
    assert status["generation_id"] == second
    assert status["ready"] is True


def test_use_generation_rolls_the_pointer_back(project: Path) -> None:
    asyncio.run(_assert_use_generation_rolls_the_pointer_back(project))


def test_use_generation_reports_an_already_selected_generation(project: Path) -> None:
    asyncio.run(_assert_use_generation_reports_an_already_selected_generation(project))


def test_use_generation_refuses_a_damaged_generation(project: Path) -> None:
    asyncio.run(_assert_use_generation_refuses_a_damaged_generation(project))


def test_use_generation_refuses_a_name_that_is_not_a_generation_id(
    project: Path,
) -> None:
    asyncio.run(
        _assert_use_generation_refuses_a_name_that_is_not_a_generation_id(project)
    )


def test_use_generation_refuses_while_a_pending_activation_is_outstanding(
    project: Path,
) -> None:
    asyncio.run(
        _assert_use_generation_refuses_while_a_pending_activation_is_outstanding(
            project
        )
    )


def test_remove_generation_removes_the_one_search_does_not_read(project: Path) -> None:
    asyncio.run(_assert_remove_generation_removes_the_one_search_does_not_read(project))


def test_remove_generation_refuses_the_generation_in_use(project: Path) -> None:
    asyncio.run(_assert_remove_generation_refuses_the_generation_in_use(project))


def test_remove_generation_will_not_run_on_a_mistyped_id(project: Path) -> None:
    asyncio.run(_assert_remove_generation_will_not_run_on_a_mistyped_id(project))


def test_remove_generation_refuses_a_pending_activation_target(project: Path) -> None:
    asyncio.run(_assert_remove_generation_refuses_a_pending_activation_target(project))


def test_a_removed_generation_can_be_rebuilt_and_selected_again(project: Path) -> None:
    asyncio.run(_assert_a_removed_generation_can_be_rebuilt_and_selected_again(project))
