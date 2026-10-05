# The one service behind the three surfaces

These tests hold the operations every surface calls: ingestion and its reuse decisions, checkpointing and resume, generation activation, rollback, and removal, the search path and its gates, reviewed metadata and exclusions as read-time overlays, and the bounded answer an agent reads. The gateway and the dense backend are faked here, so the folder needs no model download and no managed runtime.

- `test_service.py` — ingestion, selective reuse, bounded checkpointing, crash windows, activation, retrieval, filtering, reranking, pseudo-relevance feedback, source diversity, and generation selection, plus the `FakeUltraRAG` and `FakeDenseBackend` that two other folders import.
- `test_review_state_edits.py` — the hand-edited review files, written to disk the way a person writes them: a hand edit applies at the next read with no re-ingestion, a service write leaves an entry edited by hand intact, and a wrong field, type, or path is refused by name rather than ignored.
- `test_tool_views.py` — bounded agent replies: follow-up identifiers, quote safeguards, actionable conditions, blocked answers and remedies, and the absence of irrelevant diagnostics.

`test_review_state_edits.py` imports the two fakes from `test_service.py` and the PDF writer from `tests/conftest.py`. `tests/retrieval/test_chunk_exclusions.py` and `tests/retrieval/test_search_filters.py` import the same fakes from this folder, so those two files cannot be moved out of it without a change to those imports.

## Running these

```bash
.venv/bin/python -m pytest tests/core -q
```

A change to an operation, an answer shape, or a review file belongs here, because the workspace, the command line, and the agent surface all reach those through this service and none of them owns a copy.

## What it mirrors

`src/research_rag/core/`.
