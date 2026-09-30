"""The embedding models this server can drive, each pinned to a revision.

The embedding model decides what the dense half of retrieval can match, so it is
a setting rather than a constant — and it is a setting whose value is checked:
every entry declares the languages it covers, and a corpus in a language the
model was not trained for is reported instead of being silently mis-embedded.

FastEmbed resolves a name to whatever the hub serves that day, so each entry pins
the revision its weights were resolved to, exactly as the reranker table does. A
model that is not in this table is refused.

Two facts travel with a model and are easy to get wrong by hand: its vector
dimension, which the index and every stored vector depend on, and any prefix its
training requires on a query or a passage. FastEmbed does not apply those
prefixes itself, so a model that needs them declares them here and the dense
backends add them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """One supported embedding model and the facts that must not drift."""

    name: str
    revision: str
    dimension: int
    maximum_tokens: int
    languages: tuple[str, ...]
    license: str
    size_gb: float
    query_prefix: str = ""
    passage_prefix: str = ""

    def covers(self, language: str) -> bool:
        """Whether this model was trained for that language.

        An empty language tuple describes a multilingual model, which covers
        any language this server can be pointed at.
        """

        if not self.languages:
            return True
        return language.strip().casefold() in self.languages


EMBEDDING_MODELS: tuple[EmbeddingModel, ...] = (
    EmbeddingModel(
        name="BAAI/bge-small-en-v1.5",
        revision="52398278842ec682c6f32300af41344b1c0b0bb2",
        dimension=384,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=0.07,
    ),
    EmbeddingModel(
        name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        dimension=768,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=0.21,
    ),
    EmbeddingModel(
        name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        dimension=1024,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=1.20,
    ),
    EmbeddingModel(
        name="jinaai/jina-embeddings-v2-base-de",
        revision="3f9eede875721714945b6a99a3198299243cf2be",
        dimension=768,
        maximum_tokens=8192,
        languages=("de",),
        license="Apache-2.0",
        size_gb=0.32,
    ),
    EmbeddingModel(
        name="mixedbread-ai/mxbai-embed-large-v1",
        revision="b33106f585b9ce46904ad7443a3b52b7a63e231c",
        dimension=1024,
        maximum_tokens=512,
        languages=("en",),
        license="Apache-2.0",
        size_gb=0.64,
    ),
    EmbeddingModel(
        name="intfloat/multilingual-e5-large",
        revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        dimension=1024,
        maximum_tokens=512,
        languages=(),
        license="MIT",
        size_gb=2.24,
        # The E5 family is trained with these prefixes and loses quality without
        # them, so they are part of the model choice rather than a caller detail.
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
)


def models_covering(languages: Sequence[str]) -> tuple[EmbeddingModel, ...]:
    """Return the pinned models that cover every one of these languages.

    Ordered smallest first, because the useful suggestion is the cheapest model
    that can serve the corpus.
    """

    wanted = tuple(code.strip().casefold() for code in languages if code.strip())
    return tuple(
        model
        for model in sorted(
            EMBEDDING_MODELS, key=lambda item: (item.size_gb, item.name)
        )
        if all(model.covers(code) for code in wanted)
    )


EMBEDDING_MODELS_BY_NAME = {model.name: model for model in EMBEDDING_MODELS}
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_CHOICES = tuple(EMBEDDING_MODELS_BY_NAME)


def resolve_embedding_model(name: str) -> EmbeddingModel:
    """Return the pinned description of one supported embedding model."""

    try:
        return EMBEDDING_MODELS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unsupported embedding model: {name!r}; expected one of: "
            + ", ".join(EMBEDDING_MODEL_CHOICES)
        ) from None
