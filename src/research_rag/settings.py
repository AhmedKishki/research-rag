"""Every tunable this app reads, as one registry over a shared layer stack.

`default.toml` in this package holds the defaults; each layer above names only what
it changes, and later layers win **per key**:

    default.toml  <  user config  <  project config  <  environment  <  command line

The layer machinery lives in the pinned `config-ultra-rag-mcp` library. `AGENTS.md`
states which tunables a generation records and which are runtime only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from config_ultra_rag_mcp import (
    Setting,
    SettingsError,
    SettingsSources,
    default_config_path,
    project_config_path,
    user_config_path,
)

from .embeddings import (
    EMBEDDING_MODEL_CHOICES,
    EmbeddingModel,
    models_covering,
    resolve_embedding_model,
)
from .rerankers import RERANKER_MODELS

Layer = Literal["identity", "engine", "runtime"]

# The names this app resolves its own settings layers by. `PROJECT_CONFIG_RELATIVE`
# is name-free because it lives inside the project. The other two are the MCP
# server's, retained deliberately: `USER_CONFIG_DIRECTORY` is
# `research-ultra-rag-mcp` and `SETTINGS_ENVIRONMENT_PREFIX` is
# `RESEARCH_ULTRARAG_`, so this app reads the same `~/.config` directory and the
# same environment variables the MCP server wrote. Renaming either splits one
# user's settings across two products: the MCP server's file is silently ignored,
# and the environment a user exported for it stops reaching this app.
# `tests/test_data_roots.py` fails if either changes.
USER_CONFIG_DIRECTORY = "research-ultra-rag-mcp"
PROJECT_CONFIG_RELATIVE = Path(".research-rag") / "config.toml"
SETTINGS_ENVIRONMENT_PREFIX = "RESEARCH_ULTRARAG_"


# The answer detail an agent's tool call receives. The browser workspace and the
# command line always want the complete payload; the projection exists for the
# one reader whose context is bounded, and `MEASUREMENTS.md` carries both sizes.
LEAN_TOOL_DETAIL = "lean"
FULL_TOOL_DETAIL = "full"
TOOL_DETAIL_MODES = (LEAN_TOOL_DETAIL, FULL_TOOL_DETAIL)
LOG_LEVELS = ("debug", "info", "warn", "error")
DENSE_BACKENDS = ("auto", "exact", "qdrant")
DEFAULT_LANGUAGE = "en"
LANGUAGE_PATTERN = r"^[a-z]{2,3}$"
# The longest one ingest call may be told to run. A caller driving a build in one
# call sets the budget below its own client's request timeout; a caller that wraps
# this app in another process has to allow longer, or the wrapper's timeout fires
# first and the work continues unseen.
MAXIMUM_WORK_BUDGET_SECONDS = 3600

# bm25s ships a stopword list for exactly these languages and rejects every other
# name, so a corpus language outside this set has to fail here: the BM25 index is
# built after extraction and embedding, and a build that dies there has already
# spent an hour on work it cannot keep.
BM25_STOPWORD_LANGUAGES = frozenset(
    {"en", "de", "nl", "fr", "es", "pt", "it", "ru", "sv", "no", "zh", "tr", "ko"}
)


def bm25_stopwords(language: str) -> frozenset[str] | None:
    """One language's BM25 stopword list, or None when there is none.

    The list bm25s itself filters with, so a detector built on it scores the same
    function words the lexical half ignores. None means bm25s is absent or unknown for
    that language, never a wrong answer.
    """

    code = str(language).strip().casefold()
    if code not in BM25_STOPWORD_LANGUAGES:
        return None
    try:
        from bm25s.tokenization import _infer_stopwords
    except ImportError:
        return None
    try:
        return frozenset(str(item).casefold() for item in _infer_stopwords(code))
    except ValueError:
        return None


# The gate's own floor: the function words this project stops when bm25s
# supplies no list for a corpus language. It holds the question words and
# do-support forms no query is anchored by, so a contentless query abstains in
# any corpus.
FALLBACK_GATE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _english_plus_stopwords() -> frozenset[str]:
    """bm25s's fuller English list, or nothing when bm25s is absent.

    bm25s filters with the shorter list for ``en``, which keeps question words and
    do-support forms. The gate needs the fuller list: those anchor no query.
    """
    try:
        from bm25s.stopwords import STOPWORDS_EN_PLUS
    except ImportError:
        return frozenset()
    return frozenset(str(item).casefold() for item in STOPWORDS_EN_PLUS)


def resolve_gate_stopwords(languages: Sequence[str]) -> frozenset[str]:
    """The function words the lexical abstention gate ignores.

    At least what the index stops, so no query is admitted on a token BM25 never
    scored. English takes bm25s's fuller list, every other language its own.
    """

    words: set[str] = set(FALLBACK_GATE_STOPWORDS)
    for item in dict.fromkeys(languages):
        code = str(item).strip().casefold()
        words.update(bm25_stopwords(code) or ())
        if code == "en":
            words.update(_english_plus_stopwords())
    return frozenset(words)


def normalize_corpus_languages(value: Any) -> tuple[str, ...]:
    """Parse `language.corpus` into the languages a corpus is written in.

    Order is kept as written rather than sorted, because the value is part of the
    ranking policy: `de,en` and `en,de` are the same corpus only if nothing reads the
    order.
    """

    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = str(value).split(",")
    languages: list[str] = []
    for item in raw:
        code = item.strip().casefold()
        if not code:
            raise SettingsError(f"language.corpus has an empty language: {value!r}")
        if not re.fullmatch(LANGUAGE_PATTERN, code):
            raise SettingsError(
                "language.corpus must be one or more two- or three-letter "
                f"ISO 639-1 codes separated by commas: {value!r}"
            )
        if code not in BM25_STOPWORD_LANGUAGES:
            raise SettingsError(
                f"language.corpus has no BM25 stopword list: {code!r}. "
                "Supported: "
                + ", ".join(sorted(BM25_STOPWORD_LANGUAGES))
                + ". BM25 needs one of those, or it fails after the corpus has "
                "already been extracted and embedded."
            )
        if code not in languages:
            languages.append(code)
    return tuple(languages)


def _bm25_stopword_roundtrip_error(language: str) -> str | None:
    """Why a stopword list cannot survive bm25s's own save/load, if any.

    bm25s 0.3.10 wrote its stopwords file with a non-JSON escape for non-ASCII
    characters, so a German corpus failed in the BM25 index phase after extraction and
    embedding had finished. Exercising the *installed* serializer makes that a
    settings-time error, and stops mattering once bm25s is fixed.
    """

    try:
        from bm25s.tokenization import _infer_stopwords
        from bm25s.utils import json_functions
    except ImportError:
        return None  # bm25s is not installed here; the build would fail anyway.

    try:
        stopwords = _infer_stopwords(language)
    except ValueError:
        return None  # already rejected as an unknown language earlier.

    try:
        json_functions.loads(json_functions.dumps(list(stopwords)))
    except Exception as exc:  # noqa: BLE001 - any parse error is the same failure.
        return (
            f"language.bm25_stopwords {language!r} has a stopword list the "
            f"installed bm25s cannot re-read after writing it ({exc}). This "
            "would fail the BM25 index after extraction and embedding. Pin a "
            "fixed bm25s, or choose a language whose list round-trips (English "
            "does)."
        )
    return None


# The registry, in the order `--print-config` prints it. Every value in
# `default.toml` is validated against this table, and a `--set` or environment
# name absent here is refused.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="language.corpus",
        field="language_corpus",
        kind=str,
        layer="identity",
        doc=(
            "The languages the corpus is written in, as ISO 639-1 codes: one "
            "code, or several separated by commas. A language BM25 cannot "
            "tokenize is refused, because the stopword list comes from it, and "
            "every language named must be covered by the embedding model."
        ),
        env="RESEARCH_ULTRARAG_LANGUAGE_CORPUS",
    ),
    Setting(
        key="language.bm25_stopwords",
        field="bm25_stopwords",
        kind=str,
        layer="identity",
        doc=(
            "Which language's stopword list BM25 filters with. Empty means the "
            "first language in language.corpus, because BM25 takes a single list."
        ),
        env="RESEARCH_ULTRARAG_LANGUAGE_BM25_STOPWORDS",
    ),
    Setting(
        key="runtime.offline",
        field="offline",
        kind=bool,
        layer="runtime",
        doc=(
            "Require an installed vanilla runtime and already-cached models "
            "instead of downloading anything."
        ),
        env="RESEARCH_ULTRARAG_OFFLINE",
    ),
    Setting(
        key="runtime.log_level",
        field="log_level",
        kind=str,
        layer="runtime",
        normalize_case=True,
        doc="Verbosity of this process's own logging.",
        choices=LOG_LEVELS,
        env="RESEARCH_ULTRARAG_LOG_LEVEL",
    ),
    Setting(
        key="runtime.tool_detail",
        field="tool_detail",
        kind=str,
        layer="runtime",
        normalize_case=True,
        doc=(
            "Agent answer detail: 'lean' returns the fields an agent acts on, "
            "'full' returns the complete payload. The workspace and the command "
            "line always use 'full'."
        ),
        choices=TOOL_DETAIL_MODES,
        env="RESEARCH_ULTRARAG_TOOL_DETAIL",
    ),
    Setting(
        key="runtime.embedding_threads",
        field="embedding_threads",
        kind=int,
        layer="runtime",
        doc=(
            "ONNX Runtime threads for the embedding model; 0 leaves the choice "
            "to the runtime."
        ),
        minimum=0,
        maximum=1024,
        env="RESEARCH_ULTRARAG_EMBEDDING_THREADS",
    ),
    Setting(
        key="runtime.nice",
        field="nice",
        kind=int,
        layer="runtime",
        doc=(
            "CPU niceness for this process and every child it starts; 0 leaves "
            "priority unchanged, and a higher value yields to whatever else is "
            "running during a long build."
        ),
        minimum=0,
        maximum=19,
        env="RESEARCH_ULTRARAG_NICE",
    ),
    Setting(
        key="runtime.model_cache_root",
        field="model_cache_root",
        kind=str,
        layer="runtime",
        doc=(
            "Shared FastEmbed model cache; empty means the per-user cache "
            "directory for this application."
        ),
        env="RESEARCH_ULTRARAG_MODEL_CACHE_ROOT",
    ),
    # --- Retrieval: what the fused ranking is, and what it will not accept. ---
    Setting(
        key="retrieval.rrf_k",
        field="rrf_k",
        kind=int,
        layer="identity",
        doc="Reciprocal-rank-fusion constant: higher flattens the rank curve.",
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RRF_K",
    ),
    Setting(
        key="retrieval.bm25_weight",
        field="bm25_weight",
        kind=float,
        layer="identity",
        doc="Weight of the BM25 rank in the fusion.",
        minimum=0.0,
        maximum=10.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_BM25_WEIGHT",
    ),
    Setting(
        key="retrieval.dense_weight",
        field="dense_weight",
        kind=float,
        layer="identity",
        doc="Weight of the dense rank in the fusion.",
        minimum=0.0,
        maximum=10.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_WEIGHT",
    ),
    Setting(
        key="retrieval.minimum_candidates",
        field="minimum_candidates",
        kind=int,
        layer="identity",
        doc=(
            "Fewest fused candidates a search considers, before relevance gates "
            "and reranking."
        ),
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_CANDIDATES",
    ),
    Setting(
        key="retrieval.maximum_candidates",
        field="maximum_candidates",
        kind=int,
        layer="identity",
        doc="Most fused candidates a search considers.",
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MAXIMUM_CANDIDATES",
    ),
    Setting(
        key="retrieval.dense_minimum_cosine_similarity",
        field="dense_minimum_cosine_similarity",
        kind=float,
        layer="identity",
        doc=(
            "Dense relevance gate: a candidate below this cosine similarity is "
            "withheld rather than ranked. Measured: a fused ranking is "
            "insensitive below this default and loses answers above it, while a "
            "dense-only ranking wants a much lower value."
        ),
        minimum=-1.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_MINIMUM_COSINE_SIMILARITY",
    ),
    Setting(
        key="retrieval.rerank_max_candidates",
        field="rerank_max_candidates",
        kind=int,
        layer="identity",
        doc=(
            "Most candidates the cross-encoder reorders by score; the unranked "
            "tail is appended after them, so a reference group is reachable."
        ),
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_MAX_CANDIDATES",
    ),
    Setting(
        key="retrieval.rerank_window_multiple",
        field="rerank_window_multiple",
        kind=int,
        layer="identity",
        doc=(
            "Depth of the reranked window as a multiple of the requested top_k: "
            "max(top_k * this, rerank_window_floor), capped by "
            "rerank_max_candidates and the fused candidate count. Measured: 20 "
            "is the shallowest window that reaches the plateau, and each ten "
            "more candidates cost about 0.7 s per query."
        ),
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_WINDOW_MULTIPLE",
    ),
    Setting(
        key="retrieval.rerank_window_floor",
        field="rerank_window_floor",
        kind=int,
        layer="identity",
        doc=(
            "Fewest candidates the cross-encoder reorders, whatever top_k asks "
            "for, so a shallow request still ranks a useful group."
        ),
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_WINDOW_FLOOR",
    ),
    Setting(
        key="retrieval.prf",
        field="prf",
        kind=bool,
        layer="identity",
        doc=(
            "Pseudo-relevance feedback: mine terms from the lexical leaders and "
            "search again with them, so a question that does not use the "
            "author's words still reaches the passages that do. Off by default "
            "until it is measured."
        ),
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF",
    ),
    Setting(
        key="retrieval.prf_documents",
        field="prf_documents",
        kind=int,
        layer="identity",
        doc="How many of the lexical leaders the feedback terms are mined from.",
        minimum=1,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF_DOCUMENTS",
    ),
    Setting(
        key="retrieval.prf_terms",
        field="prf_terms",
        kind=int,
        layer="identity",
        doc="Most feedback terms added to one query.",
        minimum=1,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF_TERMS",
    ),
    Setting(
        key="retrieval.maximum_withheld_examples",
        field="maximum_withheld_examples",
        kind=int,
        layer="identity",
        doc="Withheld candidates quoted per gate reason in a search answer.",
        minimum=0,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MAXIMUM_WITHHELD_EXAMPLES",
    ),
    Setting(
        key="retrieval.duplicate_cosine",
        field="duplicate_cosine",
        kind=float,
        layer="runtime",
        doc=(
            "How alike two passages may be before one search answer shows only "
            "the better-ranked of them. Two sources can hold the same text, and a "
            "search that answered the same question twice with two copies of one "
            "passage is not an answer. The words are compared first, which needs "
            "no vector; this number decides the part cosine decides, the "
            "statement that says the same thing in other words. A number no "
            "cosine can reach decides that nothing is close enough rather than "
            "turning the check off, because the words are compared whatever it "
            "says."
        ),
        minimum=-1.0,
        maximum=2.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DUPLICATE_COSINE",
    ),
    Setting(
        key="retrieval.source_diversity_penalty",
        field="source_diversity_penalty",
        kind=float,
        layer="runtime",
        doc=(
            "Share of a candidate's normalized relevance charged for each "
            "candidate already selected from the same source, so one prolific "
            "source cannot fill the answer. Applied to the final top_k pick over "
            "candidates that were already ranked: it reorders what the fusion "
            "and the reranker returned, and can neither add nor remove a "
            "candidate. Zero returns the ranked order unchanged, and so does an "
            "unreranked BM25 or dense ranking, which has no score to charge a "
            "repeat against."
        ),
        minimum=0.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_SOURCE_DIVERSITY_PENALTY",
    ),
    Setting(
        key="retrieval.dense_relative_similarity_margin",
        field="dense_relative_similarity_margin",
        kind=float,
        layer="runtime",
        doc=(
            "How far below the query's own best dense similarity a candidate may "
            "score and still be admitted when it misses the cosine floor, so a "
            "short or abstract query whose candidate list sits in a narrow band "
            "is not left with a handful of passages. The rescue applies only when "
            "at least one candidate cleared the floor, so a query the corpus "
            "cannot support still abstains. 0 applies the floor to every "
            "candidate. Runtime: it re-ranks a query and changes no artifact."
        ),
        minimum=0.0,
        maximum=0.5,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_RELATIVE_SIMILARITY_MARGIN",
    ),
    Setting(
        key="retrieval.minimum_passage_words",
        field="minimum_passage_words",
        kind=int,
        layer="runtime",
        doc=(
            "Words a candidate's cleaned text must have before it can be "
            "evidence. Chunks never span extraction units, so a short unit — an "
            "index line, a heading, a copyright line, a caption — becomes a short "
            "chunk that matches a query about its own words while carrying no "
            "prose to cite. 0 admits every candidate. Runtime: it filters a query "
            "and changes no artifact."
        ),
        minimum=0,
        maximum=400,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_PASSAGE_WORDS",
    ),
    Setting(
        key="retrieval.minimum_passage_token_fraction",
        field="minimum_passage_token_fraction",
        kind=float,
        layer="runtime",
        doc=(
            "Smallest candidate a query may return, as a fraction of the "
            "generation's recorded `chunking.size`. A chunk holding a small "
            "fraction of that is a fragment — an index line, a heading, a "
            "caption — whatever its word count. Counted in the tokens the "
            "generation was chunked with, over the returned text, so a "
            "contextual header cannot make a fragment look long. 0 admits every "
            "candidate. Runtime: it filters a query and changes no artifact."
        ),
        minimum=0.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_PASSAGE_TOKEN_FRACTION",
    ),
    # --- Chunking: what a chunk is. Recorded per generation. ---
    Setting(
        key="chunking.size",
        field="chunk_size",
        kind=int,
        layer="identity",
        doc="Target chunk length in GPT-2 tokens.",
        minimum=50,
        maximum=384,
        env="RESEARCH_ULTRARAG_CHUNKING_SIZE",
    ),
    Setting(
        key="chunking.overlap",
        field="chunk_overlap",
        kind=int,
        layer="identity",
        doc="Tokens consecutive chunks share; must be below the chunk size.",
        minimum=0,
        maximum=383,
        env="RESEARCH_ULTRARAG_CHUNKING_OVERLAP",
    ),
    Setting(
        key="chunking.headers",
        field="chunk_headers",
        kind=bool,
        layer="identity",
        doc=(
            "Prepend the source title and the section to the text a chunk is "
            "embedded from, never to the text a search returns, so returned "
            "text stays quote-clean. A re-ingest with it on recomputes every "
            "vector, because vector reuse is keyed on the passage. Measured "
            "neutral on the reference corpus, so the default is off."
        ),
        env="RESEARCH_ULTRARAG_CHUNKING_HEADERS",
    ),
    Setting(
        key="chunking.batch_units",
        field="chunk_batch_units",
        kind=int,
        layer="runtime",
        doc=(
            "Extraction units sent to one chunker call. Throughput only: each "
            "unit keeps its own durable output and redo boundary."
        ),
        minimum=1,
        maximum=256,
        env="RESEARCH_ULTRARAG_CHUNKING_BATCH_UNITS",
    ),
    # --- Ingestion: how much work one call does, and in what batches. ---
    Setting(
        key="ingestion.work_budget_seconds",
        field="work_budget_seconds",
        kind=int,
        layer="runtime",
        doc=(
            "Soft time budget for one ingest call; exhausting it returns a "
            "checkpointed in_progress result instead of losing work. A build "
            "larger than one budget needs one call per slice, which an agent "
            "whose client stops repeating identical calls cannot finish: set "
            "this below that client's own request timeout."
        ),
        minimum=10,
        maximum=MAXIMUM_WORK_BUDGET_SECONDS,
        env="RESEARCH_ULTRARAG_INGESTION_WORK_BUDGET_SECONDS",
    ),
    Setting(
        key="ingestion.embedding_batch_size",
        field="embedding_batch_size",
        kind=int,
        layer="runtime",
        doc="Chunks embedded per gateway call during ingestion.",
        minimum=1,
        maximum=1024,
        env="RESEARCH_ULTRARAG_INGESTION_EMBEDDING_BATCH_SIZE",
    ),
    Setting(
        key="ingestion.pdf_page_batch_size",
        field="pdf_page_batch_size",
        kind=int,
        layer="runtime",
        doc="PDF pages extracted per gateway call.",
        minimum=1,
        maximum=64,
        env="RESEARCH_ULTRARAG_INGESTION_PDF_PAGE_BATCH_SIZE",
    ),
    Setting(
        key="dense.backend",
        field="dense_backend",
        kind=str,
        layer="engine",
        normalize_case=True,
        doc=(
            "Dense index backend for new generations: 'auto' scans the portable "
            "vectors below the documented corpus threshold."
        ),
        choices=DENSE_BACKENDS,
        env="RESEARCH_ULTRARAG_DENSE_BACKEND",
    ),
    Setting(
        key="dense.embedding_model",
        field="embedding_model",
        kind=str,
        layer="engine",
        doc=(
            "Embedding model for the dense half of retrieval. Every supported "
            "name is pinned to a revision in embeddings.py, and each declares "
            "the languages it covers."
        ),
        choices=EMBEDDING_MODEL_CHOICES,
        env="RESEARCH_ULTRARAG_EMBEDDING_MODEL",
    ),
    Setting(
        key="dense.reranker_model",
        field="reranker_model",
        kind=str,
        layer="engine",
        doc=(
            "CPU cross-encoder that reranks every search. Every supported name "
            "is pinned to a revision in rerankers.py."
        ),
        choices=tuple(RERANKER_MODELS),
        env="RESEARCH_ULTRARAG_RERANKER_MODEL",
    ),
    Setting(
        key="dense.embedding_inference_batch_size",
        field="embedding_inference_batch_size",
        kind=int,
        layer="runtime",
        doc=(
            "Sequences per embedding inference. Throughput only: a batch of 1 "
            "returns exactly the same floats as a batch of 64 (MEASUREMENTS.md)."
        ),
        minimum=1,
        maximum=1024,
        env="RESEARCH_ULTRARAG_EMBEDDING_INFERENCE_BATCH_SIZE",
    ),
    Setting(
        key="dense.exact_backend_chunk_limit",
        field="exact_backend_chunk_limit",
        kind=int,
        layer="engine",
        doc=(
            "Corpus size above which 'auto' selects the embedded ANN index "
            "instead of the exact scan; recorded in each generation."
        ),
        minimum=1,
        maximum=100_000_000,
        env="RESEARCH_ULTRARAG_EXACT_BACKEND_CHUNK_LIMIT",
    ),
)


SETTINGS_BY_KEY: dict[str, Setting] = {}
for _setting in SETTINGS:
    if _setting.key in SETTINGS_BY_KEY:
        raise SettingsError(f"Duplicate setting key: {_setting.key}")
    SETTINGS_BY_KEY[_setting.key] = _setting
SETTINGS_BY_FIELD = {setting.field: setting for setting in SETTINGS}
SETTINGS_SECTIONS = tuple(
    dict.fromkeys(setting.key.split(".")[0] for setting in SETTINGS)
)


def sources_for(project_root: str | Path) -> SettingsSources:
    project = Path(project_root)
    return SettingsSources(
        default_file=default_config_path(__file__),
        user_config=user_config_path(USER_CONFIG_DIRECTORY),
        project_root=project,
        project_config=project_config_path(project, PROJECT_CONFIG_RELATIVE),
    )


@dataclass(frozen=True, slots=True)
class EffectiveSettings:
    offline: bool
    log_level: str
    tool_detail: str
    embedding_threads: int | None
    nice: int
    model_cache_root: Path | None
    rrf_k: int
    bm25_weight: float
    dense_weight: float
    minimum_candidates: int
    maximum_candidates: int
    dense_minimum_cosine_similarity: float
    rerank_max_candidates: int
    rerank_window_multiple: int
    rerank_window_floor: int
    prf: bool
    prf_documents: int
    prf_terms: int
    maximum_withheld_examples: int
    source_diversity_penalty: float
    dense_relative_similarity_margin: float
    minimum_passage_words: int
    minimum_passage_token_fraction: float
    duplicate_cosine: float
    chunk_size: int
    chunk_overlap: int
    chunk_headers: bool
    chunk_batch_units: int
    work_budget_seconds: int
    embedding_batch_size: int
    pdf_page_batch_size: int
    dense_backend: str
    embedding_model: str
    reranker_model: str
    language_corpus: str
    bm25_stopwords: str
    embedding_inference_batch_size: int
    exact_backend_chunk_limit: int

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> EffectiveSettings:
        known = {item.name for item in fields(cls)}
        missing = sorted(known - set(values))
        if missing:
            raise SettingsError(
                "The default config is incomplete; missing: " + ", ".join(missing)
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise SettingsError("Unknown settings: " + ", ".join(unknown))

        languages = normalize_corpus_languages(values["language_corpus"])
        language = ",".join(languages)
        stopwords = str(values["bm25_stopwords"]).strip().casefold()
        if stopwords and stopwords not in BM25_STOPWORD_LANGUAGES:
            raise SettingsError(
                "language.bm25_stopwords has no BM25 stopword list: "
                f"{stopwords!r}. Supported: "
                + ", ".join(sorted(BM25_STOPWORD_LANGUAGES))
            )

        roundtrip_error = _bm25_stopword_roundtrip_error(stopwords or languages[0])
        if roundtrip_error:
            raise SettingsError(roundtrip_error)

        threads = values["embedding_threads"]
        cache_root = values["model_cache_root"]
        settings = cls(
            **{
                **values,
                "language_corpus": language,
                "bm25_stopwords": stopwords,
                "embedding_threads": None if not threads else int(threads),
                "model_cache_root": (
                    None if not cache_root else Path(str(cache_root)).expanduser()
                ),
            }
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise SettingsError(
                "chunking.overlap must be below chunking.size: "
                f"{settings.chunk_overlap} >= {settings.chunk_size}"
            )
        if settings.minimum_candidates > settings.maximum_candidates:
            raise SettingsError(
                "retrieval.minimum_candidates must not exceed "
                "retrieval.maximum_candidates: "
                f"{settings.minimum_candidates} > {settings.maximum_candidates}"
            )
        return settings

    # The embedding model carries facts that must not be configured twice: its
    # vector dimension, token limit, revision, and covered languages.

    @property
    def embedding_facts(self) -> EmbeddingModel:
        return resolve_embedding_model(self.embedding_model)

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_facts.dimension

    @property
    def embedding_model_revision(self) -> str:
        return self.embedding_facts.revision

    @property
    def embedding_maximum_tokens(self) -> int:
        return self.embedding_facts.maximum_tokens

    @property
    def corpus_languages(self) -> tuple[str, ...]:
        return tuple(self.language_corpus.split(","))

    @property
    def bm25_stopwords_language(self) -> str:
        """The one language whose stopword list BM25 filters with.

        A corpus in several languages filters the first language it names unless
        another is chosen.
        """

        return self.bm25_stopwords or self.corpus_languages[0]

    @property
    def gate_stopwords(self) -> frozenset[str]:
        """The function words the lexical abstention gate ignores.

        Every corpus language plus the one BM25 filters with, so a contentless query
        abstains in any language of the corpus. English uses bm25s's fuller list; the
        index's own keeps question words.
        """
        return resolve_gate_stopwords(
            (*self.corpus_languages, self.bm25_stopwords_language)
        )

    @property
    def embedding_language_warning(self) -> str | None:
        facts = self.embedding_facts
        missing = [code for code in self.corpus_languages if not facts.covers(code)]
        if not missing:
            return None
        covered = ", ".join(facts.languages) if facts.languages else "any language"
        message = (
            f"The embedding model {facts.name} covers {covered}, not "
            + " or ".join(f"'{code}'" for code in missing)
            + ": the dense half of retrieval will be weak for this corpus."
        )
        alternatives = models_covering(self.corpus_languages)
        if alternatives:
            return (
                message
                + " Set dense.embedding_model to one that covers it: "
                + ", ".join(
                    f"{model.name} ({model.size_gb:g} GB)" for model in alternatives
                )
                + "."
            )
        return (
            message
            + " No model in the pinned table covers "
            + " and ".join(f"'{code}'" for code in missing)
            + ", so this corpus needs a model added to embeddings.py."
        )

    def as_values(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def value(self, key: str) -> Any:
        setting = SETTINGS_BY_KEY.get(key)
        if setting is None:
            raise SettingsError(f"Unknown setting: {key}")
        return getattr(self, setting.field)
