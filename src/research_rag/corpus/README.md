# `corpus`

What is on disk, how it is read, and what a file's text says about itself.

| Module | Holds |
|---|---|
| `sources.py` | The scan of `sources/`, the policy that decides what is a source, and the stable source ID |
| `extraction.py` | Docling PDF page-unit adaptation, EPUB traversal, MOBI text-part adaptation, source-wide apparatus cleanup, and the shared quality/loss gate |
| `mobi_reader.py` | Offline, resource-bounded MOBI parsing, in-memory KF8 reconstruction, and bounded serialized-unit handoff; calls shared extraction helpers inside the worker without conversion output |
| `text_normalization.py` | Reading text as it is read: compatibility characters, whitespace, sentence ends, and the formatting glyphs a symbol face exposed as private use |
| `text_quality.py` | Whether a string is usable: corruption, scripts, the flags a chunk carries, which paragraphs belong to a non-argument section, and whether a whole source is readable |
| `pdf_backend.py` | The sole operational Docling PDF backend, its resource boundary, readiness, and active versus historical provenance |
| `docling_env.py` | The managed, versioned Docling environment: lazy first-use install under a bounded lock and cgroup, spec/ABI completion manifest, atomic publish, and the isolated worker bootstrap |
| `docling_worker.py` | The bounded subprocess entry point; verifies its cgroup before importing Docling and never runs in the parent |
| `docling_adapter.py` | Pure conversion of a Docling export into units: physical page provenance, retained structural labels, figure captions and specialised children, and furniture/index exclusion |

## Rules

- `text_normalization.py` and `text_quality.py` import only the standard library and each other. A module asking whether text is readable imports `text_quality` and no document reader.
- `extraction.extract_mobi` requests worker-built units; the worker calls `extraction._mobi_units_from_text` for HTML parsing, metadata inference, and unit construction under the parser's resource boundary. MOBI text parts share EPUB HTML normalization, structural and furniture cleanup, and quality rules. `FEATURES.md` owns parser variants and safety limits; `STORAGE.md` owns logical-part locators.
- `extraction.py` may not decide what a source is. That is `sources.py`'s.
- `docling_adapter.py` imports neither Docling nor a PDF library; it reads a plain mapping. `docling_worker.py` imports Docling only inside its `run`, after the resource-boundary check, and the parent process imports neither.
- An offline Docling extraction never fetches and fails closed on a missing model; an online extraction may fetch one lazily on its first call, inside the verified worker. An online first use also builds the managed environment through `docling_env.py`; an offline run with no ready environment fails closed. `doctor --prefetch-models` provisions both ahead of an offline run.
- The managed environment is separate from the app's own: its interpreter is run isolated (`-I`) with only the shipped worker pointed at from the app source, so the app's site-packages never reach it and Docling never enters the app's environment.
- PyMuPDF supplies PDF page-count probing in the parent; it is not a text-extraction fallback. The Docling worker probes metadata and printed page labels inside its managed runtime.
- The stored flags a chunk carries (`text_quality.CHUNK_FLAG_*`) and the token
  `extraction.py` writes into `quality_flags` are one contract:
  `EXTRACTION_ARTIFACT_TOKEN` is its name and lives in `text_quality.py`.
- `extraction.screen_source_units` is the one gate. It strips non-argument
  sections, screens text, and refuses the source if
  none survive, in that order and over the whole source's units. A build calls
  it at finalize, because it writes a page batch to disk before it knows the
  whole file.
- Because the gate is the only caller of the stripping, `text_quality` decides
  which paragraphs belong to a non-argument section and `extraction` removes
  them. Neither module may decide this alone.
- A refusal raises `ExtractionError`, which aborts the new generation and leaves
  the current one searchable. The message names the reasons, where the first
  excluded unit sits, and the two remedies, and quotes no unit text.

## What the cleanup removes, and what it does not

- Docling separates page furniture and document indexes from body text. Its
  adapter excludes those declared items before chunking and retains labels,
  content layers, item references, and every provenance span on surviving units.
- Only declared PDF headings open an apparatus section. Confirmed bibliographic
  entries can be removed across pages; a reference label alone does not remove
  prose. Captions, formulas, tables, code, explanatory footnotes, and unknown
  labels end the section without losing their text. A marked note needs
  bibliographic content, not merely a number.
- An EPUB has no page geometry, so `extraction._is_epub_furniture` names it
  instead of measuring it, from the element's own `role`, `class`, and `id`. Only
  the markup is read, so a paragraph that mentions footnotes is untouched.
- An element with `role="paragraph"` is read as a paragraph, because some journal
  EPUBs print no `<p>` at all.
- A note reference that follows text in its block is removed with the empty
  anchor printed before it, so the sentence around it stays one unit and no
  marker leaks into the text. A marker that opens its block is a note's own number
  and stays. `extraction._is_epub_noteref` decides, from markup only, and the
  removal counts on `removed_epub_furniture_elements`.
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
- Recognised formatting glyphs are not evidence of corrupt prose.
  `text_quality.is_known_formatting_glyph` keeps those glyphs out of the
  corruption count. Docling supplies the PDF text; no custom font decoder or
  alternate content-stream reader repairs it.
- Cleanup counts are recorded per source in `REMOVAL_FIELDS`.
  `removed_non_argument_characters` includes geometric furniture removals and
  reference/note section removals. A source that lost
  whole units says so on `excluded_corrupt_unit_count` with each unit's locator and
  reason. A source with image-only pages records the count and adds the
  `image_only_pages_present` warning even when it is indexed, because a mixed file
  is a fact about it rather than a refusal.
- `screen_source_units` evaluates each PDF paragraph before chunking. Clean paragraphs
  bypass repair. `clean_unclean_passage` reverses recognizable UTF-8 mojibake without
  inventing missing glyphs. Paragraphs still unreadable are withheld individually.
- `excluded_corrupt_passages` records unit IDs, paragraph offsets, locators, reasons,
  span offsets, and lost characters without repeating damaged text.
  `cleaned_passage_count` records lossless repairs and local partial cleanups;
  `cleaned_corrupt_span_count` and `partially_cleaned_passage_count` separate those
  from whole omissions. `substantive_character_count` counts retained and discarded
  substantive characters after known furniture and symbol-only non-evidence are removed;
  `discarded_corrupt_character_count` counts what it lost, and
  `unclean_character_rate` is that loss as a share of the substantive text. The rate
  is rounded for disclosure; acceptance compares the exact character counts.
  A unit is withheld whole only when it has no
  readable paragraph.
- A paragraph with no alphanumeric content and no corruption evidence is
  non-evidence, not corruption: `excluded_symbol_only_passages` records its unit,
  offset, and locator, and `excluded_symbol_only_units` does the same for a unit
  left with only such paragraphs. Neither enters `excluded_corrupt_*`. EPUB keeps
  its existing unit and source quality rules, so its symbol-only units stay on the
  corrupt counters.
- For PDFs, `ingestion.maximum_unclean_percent` caps the substantive text one
  document may lose to unrecoverable corruption: `discarded_corrupt_character_count`
  divided by `substantive_character_count`. A document over the cap is omitted from
  the generation, and the collection still builds from the sources that remain.
  A value of 100 accepts any PDF that keeps readable text, never one with none. The
  generation and checkpoint record the threshold so changing it rebuilds text.
- A PDF is omitted when it loses more than the cap, or when no readable text
  survives. No passage-level loss or cleaning ratio vetoes a readable passage.
  Mixed image-only and readable pages produce a warning rather than a scan-only
  omission.
- EPUB keeps its existing majority-readable unit and aggregate source checks.
- Extraction performs no OCR. It reads existing PDF, EPUB, or MOBI text only, so a scanned
  source needs OCR performed outside this app before it is added. The Docling
  worker reports image-only pages from its source probe; digits or symbols do not
  make a page a scan.
  A PDF with no readable text is refused with the remedy to exclude it or open it.
- No judged precision or recall exists for these cleanup rules. Garbled text without detectable corruption or structure can pass the gate.
