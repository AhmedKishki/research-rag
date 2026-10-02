# Durable writes and the portable record schemas

These tests hold the primitives every artifact is written through and the record layer above them: a JSONL stream that validates a line lazily without materializing the file, a JSON reader that reports a non-UTF-8 state file as invalid rather than reading it on a guess, the source catalog's round trip and its refusal of a path that is not normalized, and an atomic writer that fsyncs the file and its parent directory once each and leaves no temporary file behind.

- `test_storage.py` — every case above, with the two atomic writers held to one durability order.

The folder holds one file, and it imports both modules of `src/research_rag/storage/` to cover them together: the durability rule and the record rules are asserted against the same temporary path. No test here needs a project, a gateway, or a model.

## Running these

```bash
.venv/bin/python -m pytest tests/storage -q
```

This is the right scope for a change to a write path or a record schema, and it is the cheapest folder in the suite, so it is the one to run while the rest of the suite is still in use.

## What it mirrors

`src/research_rag/storage/`.