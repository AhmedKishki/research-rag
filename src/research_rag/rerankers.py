"""The reranker models this server can drive, each pinned to a revision.

The reranker is an engine setting, not a tool parameter: every search the server
answers is reranked, and *which* model reranks it is decided here. That is what
lets the harness measure one model against another over the same judged queries
while an agent sees one fixed behavior.

Every name is also a FastEmbed cross-encoder registry name, which is what makes
a choice portable: the name resolves to the same weights inside the shared model
cache, and FastEmbed rejects anything it does not recognize.
"""

from __future__ import annotations

# Model name -> the revision its weights were resolved to. An unpinned reranker
# would change retrieval quality without changing any recorded input, so a model
# that is not in this table is not offered. Both verification surfaces and the
# evaluation harness state the resolved pair in their output.
RERANKER_MODELS: dict[str, str] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
    "Xenova/ms-marco-MiniLM-L-12-v2": "42a4a787e30451cf9dbd09080c2a5b8dde332c1e",
    "jinaai/jina-reranker-v1-tiny-en": "aca45de6945b5dc6399abcd2a9c55ded5dc9111f",
    "jinaai/jina-reranker-v1-turbo-en": "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "jinaai/jina-reranker-v2-base-multilingual": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
}
# What a server serves unless its operator says otherwise: the model every
# published number was taken with, and the faster of the two measured so far.
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_CHOICES = tuple(RERANKER_MODELS)


def resolve_reranker_model(model: str) -> tuple[str, str]:
    """Return the pinned ``(name, revision)`` pair for one supported reranker."""

    try:
        return model, RERANKER_MODELS[model]
    except KeyError:
        raise ValueError(
            f"Unsupported reranker model: {model!r}; expected one of: "
            + ", ".join(RERANKER_MODEL_CHOICES)
        ) from None
