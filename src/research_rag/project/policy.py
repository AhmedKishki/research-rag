"""The vocabulary a command needs before any stack is installed.

A reader who asks what is installed, what a project holds, or what a setting costs
must be answered on a machine where the retrieval stack, the models, and the
managed runtime are all absent. `AGENTS.md` states that as a rule, and this module
is what makes it hold: the user-facing failure type, the identity of the ranking
policy, the canonical digest both are written with, and the UTC stamp every record
carries. Nothing here reads a file, resolves a setting, or imports another folder,
so importing this module cannot pull in fastembed, qdrant_client, pymupdf,
tokenizers, or numpy.

`support.py` imports the failure type and the digest from here rather than defining
them, because it is above this module and a second definition is a second answer.

A decision that no longer binds is deleted rather than filed: `TODO.md` records the
rest of `support.py`'s split, and this module is the part a lightweight command
needs from it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import EffectiveSettings

#: The retrieval methods this app serves, and the one a search defaults to. They
#: are part of what a generation is, so they stay in code rather than in a settings
#: file a project could edit.
DEFAULT_RETRIEVAL_METHOD = "hybrid"
RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})


class ResearchError(RuntimeError):
    """User-facing research workflow failure."""


def _utc_now() -> str:
    """The stamp every durable record carries: UTC, second-resolution, `Z`-suffixed."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def value_fingerprint(value: Any) -> str:
    """Return the canonical digest of a JSON-serializable value.

    Keys are sorted and the separators are fixed, so two equal values digest the
    same however they were written.
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def retrieval_policy_fingerprint(settings: EffectiveSettings) -> str:
    """Return the identity of the ranking policy these settings describe.

    The fusion constants and gates decide what a search returns, so they are part
    of what a generation *is*, and a generation recording a different policy is
    not reusable.

    A value stays out when it only reorders what those already return: the
    source-diversity penalty is taken at the final top_k pick and changes no stored
    artifact, so fingerprinting it would report a byte-identical rebuild.
    """

    return value_fingerprint(
        {
            "default_method": DEFAULT_RETRIEVAL_METHOD,
            "available_methods": sorted(RETRIEVAL_METHODS),
            "bm25": {
                "language": settings.bm25_stopwords_language,
                "tokenizer": "default",
            },
            "fusion": {
                "method": "weighted_reciprocal_rank_fusion",
                "rrf_k": settings.rrf_k,
                "bm25_weight": settings.bm25_weight,
                "dense_weight": settings.dense_weight,
                "minimum_candidates": settings.minimum_candidates,
                "maximum_candidates": settings.maximum_candidates,
            },
            "relevance_gates": {
                "bm25_requires_query_token_overlap": True,
                "dense_minimum_cosine_similarity": (
                    settings.dense_minimum_cosine_similarity
                ),
            },
        }
    )


__all__ = [
    "DEFAULT_RETRIEVAL_METHOD",
    "RETRIEVAL_METHODS",
    "ResearchError",
    "retrieval_policy_fingerprint",
    "value_fingerprint",
]
