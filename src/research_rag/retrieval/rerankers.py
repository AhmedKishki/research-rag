"""The reranker models this server can drive, each pinned to a revision.

Which model reranks is decided here, never per call, which lets the harness
measure one model against another over the same judged queries while an agent
sees one fixed behavior.

Every name is also a FastEmbed cross-encoder registry name, which makes a choice
portable: the name resolves to the same weights inside the shared model cache,
and FastEmbed rejects anything it does not recognize.
"""

from __future__ import annotations

from .model_cache import TOKENIZER_FILES

RERANKER_REQUIRED_FILES = (*TOKENIZER_FILES, "onnx/model.onnx")

# Model name -> the revision its weights were resolved to. An unpinned reranker
# would change retrieval quality without changing any recorded input, so a model
# outside this table is not offered.
RERANKER_MODELS: dict[str, str] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
    "Xenova/ms-marco-MiniLM-L-12-v2": "42a4a787e30451cf9dbd09080c2a5b8dde332c1e",
    "jinaai/jina-reranker-v1-tiny-en": "aca45de6945b5dc6399abcd2a9c55ded5dc9111f",
    "jinaai/jina-reranker-v1-turbo-en": "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "jinaai/jina-reranker-v2-base-multilingual": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
}
# The model every published decision rests on, and the cheaper of the two the
# judged set compared.
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_CHOICES = tuple(RERANKER_MODELS)


def resolve_reranker_model(model: str) -> tuple[str, str]:
    try:
        return model, RERANKER_MODELS[model]
    except KeyError:
        raise ValueError(
            f"Unsupported reranker model: {model!r}; expected one of: "
            + ", ".join(RERANKER_MODEL_CHOICES)
        ) from None
