from __future__ import annotations

import pytest

from research_rag.rerankers import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_CHOICES,
    RERANKER_MODELS,
    resolve_reranker_model,
)


def test_every_supported_reranker_is_pinned_to_a_revision() -> None:
    # An unpinned reranker would change retrieval quality with no change to any
    # recorded input, so a name alone cannot identify the weights.
    assert RERANKER_MODELS
    for model, revision in RERANKER_MODELS.items():
        assert len(revision) == 40 and revision.isalnum(), model
    assert DEFAULT_RERANKER_MODEL in RERANKER_MODELS
    assert tuple(RERANKER_MODELS) == RERANKER_MODEL_CHOICES


def test_resolve_returns_the_named_model_with_its_pinned_revision() -> None:
    name, revision = resolve_reranker_model("jinaai/jina-reranker-v1-turbo-en")

    assert name == "jinaai/jina-reranker-v1-turbo-en"
    assert revision == RERANKER_MODELS["jinaai/jina-reranker-v1-turbo-en"]


def test_resolve_rejects_unknown_names_and_lists_what_is_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported reranker model"):
        resolve_reranker_model("some-org/some-reranker")

    # A name differing only by whitespace is still unsupported, and the refusal
    # names the choices.
    with pytest.raises(ValueError, match="jinaai/jina-reranker-v1-turbo-en"):
        resolve_reranker_model(f"{DEFAULT_RERANKER_MODEL} ")
