# `corpus`

What is on disk, how it is read, and what a file's text says about itself.

| Module | Holds |
|---|---|
| `sources.py` | The scan of `sources/`, the policy that decides what is a source, and the stable source ID |
| `extraction.py` | PDF page extraction, EPUB traversal, the units they yield, the stripping of non-argument sections across a source, and the gate both build paths apply |
| `text_normalization.py` | Reading text as it is read: compatibility characters, whitespace, sentence ends |
| `ocr.py` | `research-rag ocr` alone: a copy of one PDF whose scanned pages carry a recognised text layer |
| `text_quality.py` | Whether a string is usable: corruption, scripts, the flags a chunk carries, which paragraphs belong to a non-argument section, and whether a whole source is readable |

## Rules

- `text_normalization.py` and `text_quality.py` import only the standard library and each other. A module asking whether text is readable imports `text_quality` and no PDF or EPUB reader.
- `extraction.py` may not decide what a source is. That is `sources.py`'s.
- `pymupdf.no_recommend_layout()` runs at import of `extraction.py`, so importing that module has a cost.
- The stored flags a chunk carries (`text_quality.CHUNK_FLAG_*`) and the token
  `extraction.py` writes into `quality_flags` are one contract:
  `EXTRACTION_ARTIFACT_TOKEN` is its name and lives in `text_quality.py`.
- `extraction.screen_source_units` is the one gate. It strips non-argument
  sections, withholds the units a source cannot use, and refuses the source if
  none survive, in that order and over the whole source's units. `extract_sources`
  calls it, and a staged build calls the same function at finalize, because a
  staged build writes a page batch to disk before it knows the whole file. A
  staged build cannot reach a generation a direct build would have refused.
- Because the gate is the only caller of the stripping, `text_quality` decides
  which paragraphs belong to a non-argument section and `extraction` removes
  them. Neither module may decide this alone.
- A refusal raises `ExtractionError`, which aborts the new generation and leaves
  the current one searchable. The message names the reasons, where the first
  excluded unit sits, and the two remedies, and quotes no unit text.

## What the cleanup removes, and what it does not

- Running furniture, a sidebar, and a footnote block are prose-shaped, so no
  corruption signal can tell them from evidence. They are recognised by page
  geometry in `extraction._pdf_page_units`, which asks `_repeated_margin_signatures`,
  `_sidebar_blocks`, and `_footnote_blocks` in that order, so a page's own running
  head cannot be read as a narrow column beside its text and a caption stays
  reachable as a figure's label. A page that cannot establish a body block or a
  body size yields no sidebar and no footnote, which is the direction that keeps a
  two-column page and a page of tables whole.
- `_footnote_blocks` requires a leading note marker, smaller type, and a bottom-page
  position outside detected table regions. Unmarked small-font prose stays.
  `_body_font_size` uses the character-weighted modal size rather than the median.
- An EPUB has no page geometry, so `extraction._is_epub_furniture` names it
  instead of measuring it, from the element's own `role`, `class`, and `id`. Only
  the markup is read, so a paragraph that mentions footnotes is untouched.
- A reference list or note apparatus can span pages or EPUB elements.
  `extraction.strip_non_argument_units` reads the whole source in reading order.
  `text_quality.non_argument_removal_flags` decides each paragraph inside that
  sequence. A unit left with no paragraph contributes no unit at all.
- A section is removed only while the paragraphs after its heading keep that
  section's own shape, and the first paragraph that does not ends it. That includes
  the next argument heading, a new chapter, and the end of the source, which is
  what stops a removal running past the argument it was meant to stop at. A heading
  is removed only once a paragraph of the section's own shape has confirmed it, so
  a "References" line above ordinary prose removes nothing. `text_quality` owns
  every one of those tests.
- Unicode text is normalized, not stripped. `text_normalization` composes to NFC,
  removes control characters, and folds only the two compatibility blocks English
  scholarship produces; combining marks, accents, superscripts, and subscripts are
  preserved, so a script or a diacritic that a query cannot type is not made
  unreadable by normalization.
- Cleanup counts are recorded per source in `REMOVAL_FIELDS`.
  `removed_non_argument_characters` includes geometric furniture removals and
  reference/note section removals. A source that lost
  some units says so on `excluded_corrupt_unit_count` with each unit's locator and
  reason. A source with image-only pages records the count and adds the
  `image_only_pages_present` warning even when it is indexed, because a mixed file
  is a fact about it rather than a refusal.
- A source is refused on evidence about the whole file: nothing readable left, a
  text layer that is absent, no letters in retained text, almost nothing retained,
  text dominated by symbols,
  or a cleaning that removed more than it kept. The last of those is
  `unsafe_to_clean`, and it exists because a rule that is wrong about a document is
  worse than no rule. `text_quality.source_health_reasons` owns the verdicts.
  Mixed image-only and readable pages produce a warning rather than a scan-only refusal.
- Extraction performs no OCR, and `ocr.py` is not reached from it. `ocr.py` serves `research-rag ocr` alone and writes a copy of one PDF. A page is judged a scan by
  `extraction._image_only_page`, which asks whether the page has no text and an
  image covering most of it; a share of digits or symbols is not a scan.
  An existing OCR text layer must pass the same health checks as other text.
  A PDF with no readable text is refused with the remedy to exclude it or open it.
- No judged precision or recall exists for these cleanup rules. Garbled text without detectable corruption or structure can pass the gate.
