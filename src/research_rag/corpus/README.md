# What this project is

The corpus: what is on disk, how it is read, and what a file's text says about
itself.

| Module | Holds |
|---|---|
| `sources.py` | The scan of `sources/`, the policy that decides what is a source, and the stable source ID |
| `extraction.py` | PDF page extraction, EPUB traversal, and the units they yield |
| `text_normalization.py` | Reading text as it is read: compatibility characters, whitespace, sentence ends |
| `text_quality.py` | Whether a string is usable: corruption, scripts, and the flags a chunk carries |

The split exists so that a module asking "is this text readable" imports
`text_quality` and nothing else, and does not pull PDF and EPUB readers in to ask.

## Rules

- `text_normalization.py` and `text_quality.py` import the standard library and
  each other. They never import `extraction`, `sources`, or anything from this
  package beyond those two.
- `extraction.py` may not decide what a source is. That is `sources.py`'s.
- `pymupdf.no_recommend_layout()` runs at import of `extraction.py`. This is
  recorded in `TODO.md` as work to move, and it is why importing this folder's
  extraction module has a cost.
- The stored flags a chunk carries (`text_quality.CHUNK_FLAG_*`) and the token
  `extraction.py` writes into `quality_flags` are one contract:
  `EXTRACTION_ARTIFACT_TOKEN` is its name and lives in `text_quality.py`.

## What lives elsewhere now

The code that decides whether a source is reviewed, included, or metadata-
overridden moved to `../core/`, because it is a service operation and not part of
reading a corpus. What builds a generation from what this folder reads lives in
`../generations/`.