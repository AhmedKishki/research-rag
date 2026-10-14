"""The vocabulary a command needs before any stack is installed.

A reader who asks what is installed, what a project holds, or what a setting costs
must be answered on a machine where the retrieval stack, the models, and the
managed runtime are all absent. `AGENTS.md` states that as a rule, and this module
is what makes it hold: the user-facing failure type, the identity of the ranking
policy, the canonical digest both are written with, the UTC stamp every record
carries, and the schema and policy versions a generation is identified by. Nothing
here reads a file, resolves a setting, or imports another folder, so importing this
module cannot pull in fastembed, qdrant_client, pymupdf, tokenizers, or numpy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import EffectiveSettings

#: The retrieval methods this app serves, and the one a search defaults to. They
#: are part of what a generation is, so they stay in code rather than in a settings
#: file a project could edit.
DEFAULT_RETRIEVAL_METHOD = "hybrid"
RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})

#: The PDF-extraction backend a generation with no recorded backend carries. It is
#: the legacy default, so recording a backend does not by itself invalidate an
#: existing corpus. It lives here, in a module that pulls no retrieval stack, so
#: lightweight commands can compare it without importing one.
DEFAULT_PDF_BACKEND_FINGERPRINT = "pdf-backend:custom:v1"


SCHEMA_VERSION = 5


# Extraction decides which blocks and units exist; semantic cleaning decides the
# text of a unit that survived. Bumping extraction alone would report a generation
# as needing only `layout_extraction` when its unit text also changed, so both move.
# Version 11 recovers documented symbol-font glyphs and stops a formatting glyph
# or a punctuation-only fragment from withholding or corrupting a passage.
# Version 12 bounds native recovery resources and rejects incomplete predictor
# rows, so extraction generated before those safeguards is not reused.
# Version 13 records a PDF's lost substantive text against the substantive
# characters it retained, so a generation that carries no per-document loss
# denominator is not reused.
EXTRACTION_POLICY_VERSION = 14


# EPUB and MOBI share one set of HTML reading rules. Their version is recorded on
# each EPUB or MOBI document rather than on the generation, so a change to those
# rules re-reads those sources on an ordinary ingest and passes every unchanged PDF
# through untouched. A document without the field was read under version 1.
# Version 2 reads `role="paragraph"` blocks and drops note references.
REFLOWABLE_EXTRACTION_POLICY_VERSION = 2


REFLOWABLE_FORMATS = frozenset({"epub", "mobi"})


def reflowable_document_is_current(document: Mapping[str, Any]) -> bool:
    """Whether a document was read under the current EPUB/MOBI rules; a PDF is."""

    if document.get("format") not in REFLOWABLE_FORMATS:
        return True
    recorded = document.get("reflowable_extraction_policy_version", 1)
    return recorded == REFLOWABLE_EXTRACTION_POLICY_VERSION


# Version 9 records partially cleaned passages, corrupt span counts, and lost
# characters separately from whole omissions, so a generation built under the
# earlier cleaning counters is not reused.
CLEANING_POLICY_VERSION = 10


ARTIFACT_POLICY_VERSION = 3


# The ranking policy is not part of this identity: it never affected artifacts,
# and fingerprinting it made a ranking edit discard a build in progress.
INGESTION_IDENTITY_POLICY_VERSION = 3


METADATA_STORAGE_POLICY = "automatic_only_runtime_overlay_v1"


# The shape the builder writes for a build identifier, and the shape nothing
# else may supply. A generation directory is named by a caller in `generations`,
# `use_generation`, and `remove_generation`, so those three check the name against
# this rather than joining it to a path: a name that does not look like a build
# identifier never reaches the filesystem as one.
GENERATION_ID_PATTERN = r"\d{8}T\d{6}Z-[0-9a-f]{8}"


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
    "ARTIFACT_POLICY_VERSION",
    "CLEANING_POLICY_VERSION",
    "DEFAULT_PDF_BACKEND_FINGERPRINT",
    "DEFAULT_RETRIEVAL_METHOD",
    "EXTRACTION_POLICY_VERSION",
    "GENERATION_ID_PATTERN",
    "INGESTION_IDENTITY_POLICY_VERSION",
    "METADATA_STORAGE_POLICY",
    "REFLOWABLE_EXTRACTION_POLICY_VERSION",
    "REFLOWABLE_FORMATS",
    "RETRIEVAL_METHODS",
    "SCHEMA_VERSION",
    "ResearchError",
    "reflowable_document_is_current",
    "retrieval_policy_fingerprint",
    "value_fingerprint",
]
