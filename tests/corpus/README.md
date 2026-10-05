# Source selection and text extraction

These tests hold what enters an index: the selection of regular PDF and EPUB files beneath the configured source directory, the rejection of a symlink and of a directory that escapes the project, the locator each unit carries, the folding of formula letters and ligatures, the layout repair, and the corruption policy that omits a unit with a diagnostic instead of repairing it. They also hold the source-directory and runtime-root resolution in `src/research_rag/project/config.py`, which decides what is selected and where derived state lives.

- `test_sources_and_extraction.py` — source selection and stable ids, PDF page and EPUB section locators, batched page extraction sharing one handle, EPUB anchor stability, text normalization, every corruption and script signal with its inspectable reason, front-matter metadata, layout restoration, language detection, and the relocated runtime root's claim marker.

The PDF and EPUB writers come from `tests/conftest.py`. `test_sources_and_extraction.py` builds its own `pymupdf` and `ebooklib` documents where a case needs a malformed one, so it reaches those libraries directly as well as through the shared writers.

## Running these

```bash
.venv/bin/python -m pytest tests/corpus -q
```

A change to what is selected, how a unit is read, or how text is cleaned belongs here, as does any change before a retrieval run, because a change here changes what a generation holds.

## What it mirrors

`src/research_rag/corpus/`.