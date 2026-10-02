# The generation reuse decision

These tests hold the question a no-op `ingest` answers: whether the selected generation already describes the sources as they are now. A reviewed-metadata revision is not part of the answer, because that overlay is applied at read time and a hand edit must not force a rebuild. A changed exclusion revision, a changed retrieval-policy fingerprint, a changed source digest, a missing index directory, or the legacy metadata storage policy all refuse the match.

- `test_generation.py` — `source_set_matches` and the four conditions that end it.

The file builds a `ReuseSnapshot` over a temporary generation directory and reads no index, so it runs without a project, a gateway, or a model. The rest of `src/research_rag/generations/` is exercised by `tests/core/test_service.py`, where a build is run rather than described.

## Running these

```bash
.venv/bin/python -m pytest tests/generations -q
```

This is the right scope when the reuse rule itself changes, because the rule is one predicate and this folder tests it directly rather than through an ingestion that would also assert half a dozen unrelated answers.

## What it mirrors

`src/research_rag/generations/`.