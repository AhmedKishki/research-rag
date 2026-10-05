# Storage tests

- `test_storage.py` covers both modules of `src/research_rag/storage/`.
  - A JSONL stream that validates lines lazily without materializing the file.
  - A JSON reader that reports a non-UTF-8 state file as invalid.
  - The source catalog's round trip and its refusal of a non-normalized path.
  - Both atomic writers held to one durability order: fsync the file and its parent directory once each, and leave no temporary file.
- No test needs a project, gateway, or model. This is the cheapest folder in the suite.
