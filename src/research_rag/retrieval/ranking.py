"""The rules a search ranks and selects passages by.

Lexical tokens and their document frequencies, the feedback terms mined from
the lexical leaders, the token floor counted in the chunker's own tokens, the
key two copies of a passage share, and the source-diverse selection.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ..project.policy import ResearchError

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _selection_relevance(
    ordered_ids: Sequence[str],
    scores: Mapping[str, float],
) -> dict[str, float]:
    """Normalize the score an order was built from onto ``0.0..1.0``.

    The best-scored candidate is 1.0 and the worst is 0.0, so the diversity
    penalty is a share of this ranking's confidence rather than of the pool's
    depth. An unscored candidate -- the unranked tail after a reranked window --
    is 0.0, and a flat score band leaves every scored candidate at 1.0.
    """

    relevance = dict.fromkeys(ordered_ids, 0.0)
    scored = [scores[chunk_id] for chunk_id in ordered_ids if chunk_id in scores]
    if not scored:
        return relevance
    low = min(scored)
    high = max(scored)
    if high <= low:
        # A flat score band says nothing about relative order, so every scored
        # candidate counts as equally relevant and the order itself stands.
        for chunk_id in ordered_ids:
            if chunk_id in scores:
                relevance[chunk_id] = 1.0
        return relevance
    span = high - low
    for chunk_id in ordered_ids:
        if chunk_id in scores:
            relevance[chunk_id] = (scores[chunk_id] - low) / span
    return relevance


def _passage_equality_key(text: str) -> str:
    """Return the words of a passage, so two copies of it read the same.

    Two copies differ only in case, punctuation, and spacing, which changes no
    meaning.
    """

    return " ".join(_WORD.findall(str(text).casefold()))


def _source_diverse_selection(
    ordered_ids: Sequence[str],
    *,
    source_id_by_chunk: Mapping[str, str],
    scores: Mapping[str, float],
    top_k: int,
    penalty: float,
) -> list[str]:
    """Pick ``top_k`` candidates, charging a source for each of its repeats.

    Greedy maximal-marginal-relevance over a fused order: a candidate's adjusted
    score is its relevance minus ``penalty`` per candidate already taken from the
    same source, with fused position then chunk ID settling ties.

    A penalty at or below zero, a pool no deeper than the request, or a pool with
    no scored candidate returns the plain slice, because an unreranked ranking
    offers no relevance to charge against, so the penalty alone would make it a
    round-robin over sources.
    """

    if penalty <= 0.0 or len(ordered_ids) <= top_k or not scores:
        return list(ordered_ids[:top_k])
    relevance = _selection_relevance(ordered_ids, scores)
    base_rank = {chunk_id: index for index, chunk_id in enumerate(ordered_ids)}
    remaining = list(ordered_ids)
    selected: list[str] = []
    repeats: Counter[str] = Counter()

    def ordering_key(chunk_id: str) -> tuple[float, int, str]:
        source = source_id_by_chunk.get(chunk_id, chunk_id)
        adjusted = relevance.get(chunk_id, 0.0) - penalty * repeats[source]
        return (-adjusted, base_rank[chunk_id], chunk_id)

    while remaining and len(selected) < top_k:
        chosen = min(remaining, key=ordering_key)
        remaining.remove(chosen)
        selected.append(chosen)
        repeats[source_id_by_chunk.get(chosen, chosen)] += 1
    return selected


def _content_tokens(value: str, stopwords: frozenset[str]) -> set[str]:
    """Return meaningful Unicode word tokens used for lexical abstention.

    ``stopwords`` is the gate's resolved function-word set, which stops at least
    what the index stops.
    """

    return {
        token
        for match in _WORD.finditer(value.casefold())
        if (token := match.group(0)) not in stopwords
    }


def document_frequencies(
    texts: Iterable[str],
    stopwords: frozenset[str],
) -> Counter[str]:
    """Count, for each content token, how many of these texts contain it.

    Distinct tokens per text, so a repeated word does not look common. A feedback
    rule weights against this table and it costs a full corpus pass, so build it
    once.
    """

    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(_content_tokens(text, stopwords))
    return frequencies


def _inverse_document_frequency(
    frequencies: Mapping[str, int] | None,
    corpus_size: int,
    term: str,
) -> float:
    """Return a term's rarity weight, smoothed so a ubiquitous term stays positive.

    Without a table the weight is 1, so a caller that has none gets the
    unweighted ranking.
    """

    if not frequencies or corpus_size <= 0:
        return 1.0
    return math.log((corpus_size + 1) / (frequencies.get(term, 0) + 1)) + 1.0


def _pseudo_relevance_terms(
    *,
    query: str,
    texts: list[str],
    maximum_terms: int,
    stopwords: frozenset[str],
    document_frequencies: Mapping[str, int] | None = None,
    corpus_size: int = 0,
) -> list[str]:
    """Mine expansion terms from the first-pass lexical leaders.

    A candidate is scored by leader support times rarity, because support alone
    mines the words that appear everywhere and discriminate nothing.

    Ordered by score then term, so one query expands the same way on every run.
    Terms the query already holds are skipped: they would change nothing and hide
    whether the expansion did anything.
    """

    if maximum_terms <= 0:
        return []
    query_tokens = _content_tokens(query, stopwords)
    support: Counter[str] = Counter()
    for text in texts:
        support.update(_content_tokens(text, stopwords) - query_tokens)

    def rank(item: tuple[str, int]) -> tuple[float, int, str]:
        term, count = item
        weight = _inverse_document_frequency(document_frequencies, corpus_size, term)
        return (-count * weight, -count, term)

    return [term for term, _count in sorted(support.items(), key=rank)[:maximum_terms]]


# The chunker counts a chunk in GPT-2 tokens and the generation records that
# name, so a length rule stated as a fraction of the chunk size and a token count
# are one unit. The counter is imported and loaded on first use, so a query
# that applies no token floor never pays for the dependency.
TOKENIZER_REPOSITORIES = {"gpt2": "openai-community/gpt2"}


def passage_token_count(
    text: str, tokenizer: str, *, offline: bool = False, cache_root: Path | None = None
) -> int:
    """Count a passage in the tokens the generation was chunked by."""

    if tokenizer not in TOKENIZER_REPOSITORIES:
        raise ResearchError(
            "The current generation records an unknown chunker tokenizer: "
            f"{tokenizer!r}. Known names: "
            f"{', '.join(sorted(TOKENIZER_REPOSITORIES))}."
        )
    from .direct import count_gpt2_tokens

    try:
        return count_gpt2_tokens(text, offline=offline, cache_root=cache_root)
    except Exception as exc:
        raise ResearchError(
            f"The {tokenizer} tokenizer could not be loaded: {exc}"
        ) from exc
