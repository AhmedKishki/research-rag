# Evaluation harness tests

These tests need no project, gateway, or model. The fakes and the PDF writer come from `tests/core/test_service.py` and `tests/conftest.py`. The harness is loaded by path, because `scripts/` is not a package.

- `test_evaluation.py`: the metrics and payload readers.
  - Each measure against what it says it compares: exact equality, lexical containment, missing-passage coverage, the cosine gate against its own denominator, repetition rates, and latency percentiles.
  - The raw `evaluation_trace` surviving into the report with its truncation flags, and the scored-then-collapsed count complete against a cut list.
  - The reranked-row policy through `_run_one`, including a real service whose cross-encoder cannot load.
- Target resolution collects failures instead of short-circuiting. A skipped target neither fails nor excuses another.
- The shipped judged set resolves to one target per query, and `no_answer_support` reports abstention as unmeasured.
- `tests/retrieval/test_search_evaluation_trace.py` holds the engine-side payload contract.
