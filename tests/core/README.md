# Service tests

The operations every surface calls. Retrieval and dense backends are faked, so no model download or managed runtime is needed.

- `test_service.py`: ingestion, selective reuse, checkpointing, crash windows, activation, retrieval, filtering, reranking, pseudo-relevance feedback, source diversity, and generation selection.
  - Corpus-derived and explicit BM25 languages reach the backend and match the manifest and status.
- `test_review_state_edits.py`: hand-edited review files. A hand edit applies at the next read without re-ingestion, a service write keeps an entry edited by hand, and a wrong field, type, or path is refused by name.
- `test_mobi_service.py`: staged MOBI ingestion, hybrid retrieval, source lookup, no-op and extraction reuse, reviewed metadata and exclusions, and a DRM omission retaining a partial generation without changing the selected one.
- `test_operations.py`: the operation table's defaults, a query string read as each argument's kind, diagnostic arguments only from the command line in process, and every refusal named by its argument.
- `test_admission.py`: callers served in rounds, slots running together up to their number, a bounded wait that names the queue it stood in, a cancelled wait that frees its place, and a search refused behind a full queue.
- `test_search_stats.py`: a search counts its first five ranks and never its query, a measurement counts nothing, and the stats answer reads corpus and build facts from the generation.
- `test_tool_views.py`: bounded agent replies, the absence of per-passage quotation flags, actionable conditions, blocked answers and remedies, and the absence of irrelevant diagnostics.
- `test_health.py`: one check per condition in `ok`, `warn`, `blocked`, or `unknown`.
  - Direct dependency pins, missing model and GPT-2 caches, a held project lock, and a stale lock from a dead process.
  - Free space against the build it must fit, and an app older than the installed code.
  - A check that did not run never reads as healthy.
- `test_version.py`: running and installed versions, the `--version` answer, and the declared version matching the installed distribution.
- `test_records.py`: the source catalog's round trip and its refusal of a non-normalized path.
- `test_generation_compatibility.py`: diagnostic-only processing differences, recorded/current policy provenance, DEBUG compatibility decisions, and refusal of foreign project identities.

- Shared fake backends and service construction come from `tests/service_helpers.py`. Source writers and the project fixture come from `tests/conftest.py`.
- A change to an operation, an answer shape, or a review file belongs here, because every surface reaches it through this service.
