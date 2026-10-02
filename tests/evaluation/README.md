# The retrieval-evaluation harness

These tests hold the measurement rather than a retrieval claim: the metrics the harness computes, the way it loads and refuses a judged set, the way a target is re-resolved after a rebuild or a rename, and the shape of the mode table its published numbers were taken from.

- `test_evaluation.py` — `success@k`, reciprocal rank, and `nDCG@k`; lexical overlap that ignores stopwords and short tokens; `load_judgments` refusing an unknown class and a missing target; target resolution by snippet, by `document_id`, and by a skip list, with ambiguity refused; the per-mode and per-class summary; the shipped judged set being structurally valid; and the report version 2 result-list measures — exact and near duplicate slots, same-source pairs, the repeated-slot rate, and latency percentiles — including that an absent measure is reported as `None` rather than as zero.

The harness is a measurement script rather than a shipped module, so it is loaded by path from `scripts/evaluate_retrieval.py` and no test imports it as `research_rag`. It reads `evaluation/ai-and-fetishism-queries.json` and reaches `ResearchService` through the same call the script makes. No test here runs an ingestion, so a retrieval number is produced by the script and not by this folder.

## Running these

```bash
.venv/bin/python -m pytest tests/evaluation -q
```

This is the right scope for a change to the harness or to the judged set, and it is the folder to run before a documented retrieval default changes, because a protocol change has to keep the metrics it published comparable.

## What it mirrors

`evaluation/` and `scripts/evaluate_retrieval.py`, which exercise the app end to end rather than one package folder.