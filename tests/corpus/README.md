# Corpus tests

What enters an index: source selection, locators, text cleanup, and the health gate. They also cover source-directory and runtime-root resolution in `src/research_rag/project/config.py`.

- `test_sources_and_extraction.py`: source selection and stable ids, symlink and escaping-directory rejection, Docling PDF and EPUB section locators, EPUB anchor stability, text normalization, corruption and script signals, embedded metadata, language detection, and the relocated runtime root's claim marker.
- `test_docling_adapter.py` and `test_docling_routing.py`: retained structural labels and provenance, furniture/index exclusion, protected captions/formulas/notes, legacy setting aliases, and the shared gate's configured budget.
- `test_mobi.py`: real-parser uncompressed, PalmDOC, HUFF/CDIC, KF8, and dual-rendition containers; metadata and logical-part anchors; DRM, malformed-container, resource-limit, and symlink refusals. Synthetic containers come from `tests/mobi_fixtures.py`, not private books.
- `test_ocr.py`: OCR is gone: the parser, help menu, module, and package extra offer none, and no shipped document names the removed command.
- `test_apparatus.py`: the apparatus verdict, each kind it names by structure, and the argument, mixed passage, table, and quoted-language prose it keeps.
- `test_extraction_health_gate.py`: pre-chunk cleanup and the source health gate, including the share of unreadable text a source may lose before it is refused.
  - Each removal rule against the text it must keep.
  - A section continued across unit boundaries, and one that stops at the next argument heading.
  - Docling-declared furniture, protected explanatory notes, and EPUB furniture cases.
  - Every source verdict, including a cleanup that removed more than it kept.
  - The refusal message and its remedy, and the counters each removal writes.
  - A staged build reaching the same verdict as a direct one, with a refusal preserving the current generation.

- The PDF and EPUB writers come from `tests/conftest.py`. `test_sources_and_extraction.py` also builds malformed `pymupdf` and `ebooklib` documents directly.
- `test_extraction_health_gate.py` also needs the fakes from `tests/core/test_service.py`.
- Fast tests use the offline synthetic Docling worker in `tests/conftest.py`; recorded payloads exercise structural roles. `real_docling` opts out of the stub for a worker-boundary test.
- Run this folder before any retrieval run, because a change here changes what a generation holds.
