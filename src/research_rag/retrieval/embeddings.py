"""The embedding models this server can drive.

The embedding model decides what the dense half of retrieval can match, so it is
a checked setting: every entry declares the languages it covers, and a corpus in
a language the model was not trained for is reported instead of being silently
mis-embedded.

Each entry pins the download repository and revision of its ONNX weights.
The model runtime loads that exact snapshot rather than FastEmbed's moving head.
A model outside this table is refused.

Two facts travel with a model and are easy to get wrong by hand: its vector
dimension, which the index and every stored vector depend on, and any prefix its
training requires on a query or a passage. FastEmbed does not apply those
prefixes, so a model that needs them declares them here and the dense backends
add them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model_cache import TOKENIZER_FILES


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """One supported embedding model. Every field here is load-bearing."""

    name: str
    repository: str
    revision: str
    model_file: str
    dimension: int
    maximum_tokens: int
    languages: tuple[str, ...]
    license: str
    size_gb: float
    query_prefix: str = ""
    passage_prefix: str = ""
    additional_files: tuple[str, ...] = ()

    @property
    def required_files(self) -> tuple[str, ...]:
        return (*TOKENIZER_FILES, self.model_file, *self.additional_files)

    def matches_dense_metadata(self, dense: Mapping[str, Any]) -> bool:
        """Match recorded vector identity, including repositories when recorded.

        Older manifests lack a repository. Matching name, commit and dimension
        remain sufficient for the unchanged direct-repository pins.
        """
        return (
            dense.get("embedding_model") == self.name
            and dense.get("embedding_model_revision") == self.revision
            and dense.get("embedding_dimension") == self.dimension
            and dense.get("embedding_model_repository", self.repository)
            == self.repository
        )

    def covers(self, language: str) -> bool:
        """Whether this model was trained for that language.

        An empty language tuple describes a multilingual model, which covers any
        language this server can be pointed at.
        """

        if not self.languages:
            return True
        return language.strip().casefold() in self.languages


EMBEDDING_MODELS: tuple[EmbeddingModel, ...] = (
    EmbeddingModel(
        name="BAAI/bge-small-en-v1.5",
        repository="qdrant/bge-small-en-v1.5-onnx-q",
        revision="aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        model_file="model_optimized.onnx",
        dimension=384,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=0.07,
    ),
    EmbeddingModel(
        name="BAAI/bge-base-en-v1.5",
        repository="qdrant/bge-base-en-v1.5-onnx-q",
        revision="199291fdd1aa89faf9c20b722dc72ad5e17aa0d0",
        model_file="model_optimized.onnx",
        dimension=768,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=0.21,
    ),
    EmbeddingModel(
        name="BAAI/bge-large-en-v1.5",
        repository="qdrant/bge-large-en-v1.5-onnx",
        revision="e93b9305e013fdf771b471426516285da0cca188",
        model_file="model.onnx",
        dimension=1024,
        maximum_tokens=512,
        languages=("en",),
        license="MIT",
        size_gb=1.20,
    ),
    EmbeddingModel(
        name="jinaai/jina-embeddings-v2-base-de",
        repository="jinaai/jina-embeddings-v2-base-de",
        revision="3f9eede875721714945b6a99a3198299243cf2be",
        model_file="onnx/model_fp16.onnx",
        dimension=768,
        maximum_tokens=8192,
        languages=("de",),
        license="Apache-2.0",
        size_gb=0.32,
    ),
    EmbeddingModel(
        name="mixedbread-ai/mxbai-embed-large-v1",
        repository="mixedbread-ai/mxbai-embed-large-v1",
        revision="b33106f585b9ce46904ad7443a3b52b7a63e231c",
        model_file="onnx/model.onnx",
        dimension=1024,
        maximum_tokens=512,
        languages=("en",),
        license="Apache-2.0",
        size_gb=0.64,
    ),
    EmbeddingModel(
        name="intfloat/multilingual-e5-large",
        repository="qdrant/multilingual-e5-large-onnx",
        revision="ac6781cd1cf88b8306a536d7c9d18a5bd57cc14b",
        model_file="model.onnx",
        additional_files=("model.onnx_data",),
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

    Ordered smallest first: the useful suggestion is the cheapest model that can
    serve the corpus.
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
    try:
        return EMBEDDING_MODELS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unsupported embedding model: {name!r}; expected one of: "
            + ", ".join(EMBEDDING_MODEL_CHOICES)
        ) from None
