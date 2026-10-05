# Source selection and text extraction

These tests hold what enters an index: the selection of regular PDF and EPUB files beneath the configured source directory, the rejection of a symlink and of a directory that escapes the project, the locator each unit carries, the folding of formula letters and ligatures, the layout repair, the removal of running furniture, sidebars, footnotes, EPUB furniture markup, reference lists, and note apparatus, the corruption policy that omits a unit with a diagnostic instead of repairing it, and the source health gate that refuses a file its own units agree has nothing to retrieve from. They also hold the source-directory and runtime-root resolution in `src/research_rag/project/config.py`, which decides what is selected and where derived state lives.

- `test_sources_and_extraction.py` — source selection and stable ids, PDF page and EPUB section locators, batched page extraction sharing one handle, EPUB anchor stability, text normalization, every corruption and script signal with its inspectable reason, front-matter metadata, layout restoration, language detection, and the relocated runtime root's claim marker.
- `test_extraction_health_gate.py` — the pre-chunk cleanup and the source health gate: each removal rule against the text it must keep, a section continued across unit boundaries beside one that must stop at the next argument heading, the sidebar, footnote, and EPUB furniture geometry beside the page cases that keep every block, every source verdict including a cleanup that removed more than it kept, the refusal message naming the file and its remedy, the counters each removal category writes, and that the staged build path reaches the same verdict as the direct one while a refusal preserves the current generation and leaves no partial unit behind.

The PDF and EPUB writers come from `tests/conftest.py`. `test_sources_and_extraction.py` builds its own `pymupdf` and `ebooklib` documents where a case needs a malformed one, so it reaches those libraries directly as well as through the shared writers. `test_extraction_health_gate.py` drives `extraction.screen_source_units` through both a real staged ingestion and a direct extraction, so it needs the shared writers and the fakes from `tests/core/test_service.py`.

## Running these

```bash
.venv/bin/python -m pytest tests/corpus -q
```

A change to what is selected, how a unit is read, or how text is cleaned belongs here, as does any change before a retrieval run, because a change here changes what a generation holds.

## What it mirrors

`src/research_rag/corpus/`.