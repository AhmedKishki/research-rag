# Tests for the retrieval evaluation harness

## What is checked

- `tests/evaluation/test_evaluation.py` — the metrics, and the payload readers
  behind them. Each measure is checked against what it says it compares: an exact
  equality that collapses whitespace and nothing else, keeping letter case,
  word order, a repeated word, a sign, a decimal, an operator, a closing mark,
  and a script it cannot segment; a lexical containment that refuses a pair whose
  figures or negations disagree and that never counts one passage twice; missing
  passages reported as coverage rather than as duplicates; the cosine gate read
  only when the method ranked the dense half, and its counts checked against its
  own denominator; repetition rates split into all-queries and
  cross-question-family; latency as percentiles with the slowest reported beside
  them. The reranked-row policy is checked through `_run_one` against the payload
  a search returns, including a real service whose cross-encoder cannot load.
- The judged set beside this file resolves: `ai-and-fetishism-queries.json` is
  checked for one target per query, and `no_answer_support` is checked to report
  that abstention is unmeasured rather than to invent a label.
- `tests/retrieval/test_search_evaluation_trace.py` holds the engine-side payload
  contract, because it is the engine's payload: the eight named stages, the
  bounded identifier lists, the gate's denominator and conservation, the
  candidate-depth and rerank-window formulas, a BM25 payload reporting no dense
  counts, an unavailable reranker traced as not applied, and the proof that a
  traced search returns an answer otherwise identical to an untraced one.

## Running them

```bash
uv run pytest tests/evaluation -q
```

The suite needs no project, no gateway, and no model: the two deterministic
fakes and the PDF writer come from `tests/core/test_service.py` and
`tests/conftest.py`, so these tests read the payload a real service emits rather
than a shape the engine cannot produce. The harness is loaded by path, because
`scripts/` is not a package.