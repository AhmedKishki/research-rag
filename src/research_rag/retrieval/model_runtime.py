"""The models both dense backends run against.

It owns the three model loaders and the six operations the backends each carried a
copy of, so an embedding is produced, counted, and reranked one way per process.
The backends, the Qdrant filter type, and the portable-vector checks moved to
`dense_backends/`; this module may never import any of those, so a runtime stays
usable by a backend that stores nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer

from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModel, resolve_embedding_model
from .rerankers import DEFAULT_RERANKER_MODEL, resolve_reranker_model


class DenseTokenAuditUnavailable(RuntimeError):
    pass


class RerankerUnavailable(RuntimeError):
    """Raised when the optional reranker model cannot be loaded.

    The service treats this as recoverable: reranking is skipped and the response
    discloses the fallback.
    """


def _load_embedder(
    cache_root: Path,
    *,
    offline: bool,
    threads: int | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> TextEmbedding:
    """The pinned CPU embedding model from the shared model cache.

    `threads` sets the ONNX Runtime thread count, left to the runtime by default;
    nothing is auto-detected, because the measured optimum is machine-specific.
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
    """One pinned CPU cross-encoder from the shared model cache.

    Any load failure means the reranker cannot run, most often an offline call whose
    model is not cached, so it is reported as ``RerankerUnavailable``.
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
    """A non-truncating tokenizer matching the embedding model.

    The embedder's own tokenizer truncates at the model limit, which would hide the
    overflow the ingestion audit measures, so the pinned file is loaded separately.
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


class ModelRuntime:
    """One set of model arguments, and every model loaded for it.

    A runtime belongs to the process that built it and is not shared with a second
    one, because each holds the models it has loaded so far.
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
        # The model's facts — name, revision, dimension, token limit, and any
        # required prefix — come from the pinned table rather than the caller.
        self.embedding_facts: EmbeddingModel = resolve_embedding_model(embedding_model)
        # Sequences are padded to the longest member of their inference batch, so a
        # large batch spends most of its compute on padding, while one sequence per
        # inference returns exactly the same floats as a large batch does.
        # MEASUREMENTS.md states why that is a throughput setting only.
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
        """The pinned cross-encoder for one model, loaded on demand.

        A run comparing rerankers needs more than one, so each is loaded at most once
        and kept for the runtime's life.
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
        """The embedding tokenizer length of each text, untruncated."""

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
        """One float32 vector per text, each checked against the pinned dimension.

        The two backends carried this method separately, and only the Qdrant copy
        refused a vector count that disagreed with the passage count. The check is
        kept for both because it is the one ``DenseBackend.embed_texts`` documents: a
        short count would pair every vector with the wrong chunk, and no later stage
        could tell.
        """

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
