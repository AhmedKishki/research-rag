from __future__ import annotations

import pytest

from research_rag.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_CHOICES,
    EMBEDDING_MODELS,
    models_covering,
    resolve_embedding_model,
)


def test_every_supported_model_is_pinned_and_described() -> None:
    # An unpinned model would change what the index contains with nothing in a
    # generation recording the change.
    assert EMBEDDING_MODELS
    for model in EMBEDDING_MODELS:
        assert len(model.revision) == 40 and model.revision.isalnum(), model.name
        assert model.dimension > 0, model.name
        assert model.maximum_tokens > 0, model.name
        assert model.license, model.name
    assert DEFAULT_EMBEDDING_MODEL in EMBEDDING_MODEL_CHOICES


def test_the_german_model_is_offered_with_its_own_dimension() -> None:
    model = resolve_embedding_model("jinaai/jina-embeddings-v2-base-de")

    assert model.dimension == 768
    assert model.maximum_tokens == 8192
    assert model.covers("de")
    assert not model.covers("en")


def test_a_language_less_entry_covers_every_language() -> None:
    multilingual = resolve_embedding_model("intfloat/multilingual-e5-large")

    assert multilingual.languages == ()
    assert multilingual.covers("de")
    assert multilingual.covers("en")
    assert multilingual.covers("ja")
    # The E5 family loses quality without its prefixes, so they ship with the model
    # rather than being a caller's detail.
    assert multilingual.query_prefix == "query: "
    assert multilingual.passage_prefix == "passage: "


def test_an_english_model_does_not_claim_a_german_corpus() -> None:
    model = resolve_embedding_model(DEFAULT_EMBEDDING_MODEL)

    assert model.covers("en")
    assert not model.covers("de")


def test_resolve_rejects_an_unknown_model_and_lists_the_choices() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        resolve_embedding_model("some-org/some-embedder")

    with pytest.raises(ValueError, match="jinaai/jina-embeddings-v2-base-de"):
        resolve_embedding_model(f"{DEFAULT_EMBEDDING_MODEL} ")


def test_models_covering_lists_the_cheapest_model_that_serves_a_corpus() -> None:
    """A corpus in several languages needs a model that covers all of them."""
    assert [model.name for model in models_covering(("de",))] == [
        "jinaai/jina-embeddings-v2-base-de",
        "intfloat/multilingual-e5-large",
    ]
    mixed = models_covering(("de", "en"))
    assert [model.name for model in mixed] == ["intfloat/multilingual-e5-large"]
    # The English corpus is served by the smallest English model first.
    assert models_covering(("en",))[0].name == DEFAULT_EMBEDDING_MODEL
    assert models_covering(("ar",))[0].name == "intfloat/multilingual-e5-large"
