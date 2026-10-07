# Service tests

The operations every surface calls. The gateway and dense backend are faked, so no model download or managed runtime is needed.

- `test_service.py`: ingestion, selective reuse, checkpointing, crash windows, activation, retrieval, filtering, reranking, pseudo-relevance feedback, source diversity, and generation selection. It also defines `FakeUltraRAG` and `FakeDenseBackend`.
  - Corpus-derived and explicit BM25 languages reach the gateway and match the manifest and status.
- `test_review_state_edits.py`: hand-edited review files. A hand edit applies at the next read without re-ingestion, a service write keeps an entry edited by hand, and a wrong field, type, or path is refused by name.
- `test_admission.py`: callers served in rounds, slots running together up to their number, a bounded wait that names the queue it stood in, a cancelled wait that frees its place, and a search refused behind a full queue.
- `test_search_stats.py`: a search counts its first five ranks and never its query, a measurement counts nothing, and the stats answer reads corpus and build facts from the generation.
- `test_tool_views.py`: bounded agent replies, quote safeguards, actionable conditions, blocked answers and remedies, and the absence of irrelevant diagnostics.

- `test_review_state_edits.py` imports the fakes from `test_service.py` and the PDF writer from `tests/conftest.py`.
- `tests/retrieval/test_chunk_exclusions.py` and `tests/retrieval/test_search_filters.py` import the same fakes. Do not move these files without updating those imports.
- A change to an operation, an answer shape, or a review file belongs here, because every surface reaches it through this service.
