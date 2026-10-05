---
name: STORAGE.md
description: The state format: every file, every field, portability, and hand edits.
---

# Storage contract

- This document names every file the app writes, what each one means, and which ones are worth backing up.
- Read this document before editing a project by hand.

## The tree

```text
my-research-project/
├── sources/                              untouched PDF/EPUB originals
└── .research-rag/                        all project state
    ├── project.json                      stable id, name, source directory
    ├── source-catalog.json               durable source-id-to-path registry
    ├── source-metadata.json              authoritative reviewed metadata overlay
    ├── source-exclusions.json            reviewed decisions, when present
    ├── chunk-exclusions.json             reviewed decisions about single passages, when present
    ├── config.toml                     this project's settings layer
    └── runtime/                       disposable derived state
        ├── current.json                  selected generation pointer
        ├── project.lock
        ├── research-rag-ui.port          port a running app serves, while it runs
        ├── research-rag-ui.pid           its process, while it runs
        ├── research-rag-ui.tty           the terminal it is attached to; absent when it has none
        ├── logs/
        ├── failures/                     small failed-build records
        ├── staging/<build-id>/           resumable incomplete build plus checkpoint
        ├── ultrarag-runtime/
        └── generations/<generation-id>/
            ├── manifest.json
            ├── corpus/extracted-units.jsonl
            ├── chunks/chunks.jsonl
            ├── portable/embeddings.npy
            └── indexes/
                ├── artifact-lookup.sqlite3
                ├── bm25/
                └── qdrant/

~/.cache/research-rag/models/  shared model binaries only
```

- `~/.config/research-rag/config.toml` is the user settings file.
- `~/.config/research-rag/projects.json` is the account's project register, beside those settings.
- `projects.lock` beside that register serializes read-modify-write operations across processes.
  - Readers use atomic snapshots without taking the lock.
- `~/.cache/research-rag/models/` holds the embedding and reranker binaries, about 150 MB in total, and is the only cross-project shared state, because those binaries are immutable once downloaded.
- `tests/project/test_data_roots.py` asserts both roots and states why each is named for this app.

## Portable state

- These six files are your decisions about the project: the part worth backing up, and the part that survives a rebuild.

| File | What it holds | Keyed by |
|---|---|---|
| `project.json` | project name, stable id, sources directory | — |
| `config.toml` | this project's settings layer, one key per setting | setting key |
| `source-catalog.json` | the durable source-id registry | normalized source-relative path |
| `source-metadata.json` | the reviewed metadata overlay | normalized source-relative path |
| `source-exclusions.json` | each exclusion decision and its reason | normalized source-relative path |
| `chunk-exclusions.json` | each passage exclusion, its reason, and the generation it was recorded in | `chunk_id` |

- No hand edit is kept here: `src/research_rag/core/source_inventory.py` rewrites `source-catalog.json` from the current scan whenever it is stale, which is the price of remembering a source the project has seen and no longer holds.

### project.json

```json
{
  "schema_version": 1,
  "project_id": "c0ffee...",
  "name": "My research project",
  "source_directory": "sources"
}
```

- `project_id` is authoritative and never rewritten.
- `source_directory` is relative to the project root.
- Every command reuses `source_directory` when `--source-directory` is omitted.
- A `--source-directory` value that differs from `source_directory` is refused rather than quietly accepted.

### config.toml

- `config.toml` is the project's own settings layer, and it is written by hand or by the workspace in the browser.
- Every key is a `Setting` declared in the registry in `src/research_rag/project/settings.py`, with its default written in `src/research_rag/project/default.toml` under a table named by its section, so `[retrieval]` holds `retrieval.rrf_k`.
- The browser rewrites the whole file from the values it read.
- A key the browser does not know about is preserved by that rewrite.
- A comment in `config.toml` is not preserved by that rewrite.
- A key the environment, the command line, or `--config` supplied cannot be written here, because those layers outrank this file.
- `research-rag config` prints every value with the layer it came from and what a change to it costs.

## Derived state

- Everything under `.research-rag/runtime/` is rebuilt from `sources/` plus the portable state.
- Deleting `.research-rag/runtime/` costs one ingestion.
- `current.json` names the one generation search uses.
- Earlier successful generations stay on disk and are not searched, and nothing prunes them automatically.
- `research-rag status` reports `retained_generation_count` and `retained_generation_bytes`.
- `research-rag generations` lists every generation with its creation time, chunk and document counts, file count, and size.
- A generation directory whose manifest is missing or unreadable is reported with `manifest_error` rather than failing the call.
- Two commands move a generation, and both refuse the generation search reads.
- `research-rag generations --use GENERATION_ID` points the project at a retained generation.
  - It validates the generation's artifacts and both indexes exactly as a build's activation does.
  - It rewrites `current.json` only after that validation succeeds, so a rollback that lands on a damaged generation fails instead of bricking every read surface.
  - It rebuilds no originals, so the corpus the pointer now describes is whatever that generation indexed.
- `research-rag remove-generation GENERATION_ID --confirm GENERATION_ID` deletes one generation permanently.
  - The repeated id is the check, because a generation named by a listing and removed by a copy of that listing cannot be walked back.
  - It refuses a generation a pending activation has named, because that activation is about to move it into place.
  - It returns the space only through a rebuild, which costs one ingestion.
  - It does not touch the original files.
- A running app can still hold a generation: the retrieval gateway keeps a live index against whichever generation it last loaded, and the exact dense backend memory-maps a generation's vector file for the process's lifetime.
- Nothing refuses a delete loudly: an unlinked mapping keeps reading the old bytes until its last reference drops, and a later open returns a missing-file error rather than a failure at the delete. Both generation commands therefore take the project lock and refuse the current generation rather than merely warning about it.

### Generation layout

```text
generations/<generation-id>/
  manifest.json                 what this generation contains and the policy that built it
  corpus/extracted-units.jsonl  cleaned extraction units, one JSON object per line
  chunks/chunks.jsonl           the final chunks, one JSON object per line
  portable/embeddings.npy       float32 vectors, row i belongs to chunk i
  indexes/artifact-lookup.sqlite3
  indexes/bm25/                 the pinned UltraRAG BM25 index
  indexes/qdrant/               the dense index, when the manifest names it
```

- `manifest.json` decides what a generation is.
  - It carries the schema version, the extraction policy, the chunking configuration, the retrieval-policy fingerprint, the model and revision that produced the vectors, the dense backend, and the file map.
  - `retrieval.bm25.language` records the stopword list the BM25 index was built with, which is `language.bm25_stopwords` and may differ from `language.corpus` on a mixed corpus.
- Two generations are interchangeable only when their manifests agree, and that is what makes reuse safe.
- A generation whose policy fingerprint does not match the current settings is reported as requiring a new ingestion.
- The directory names under `indexes/` vary by backend, and the manifest records which one a generation uses.
- No reader may assume a fixed directory name.

### The artifact lookup

- `artifact-lookup.sqlite3` holds identifiers, ordinals, content hashes, byte offsets into the canonical JSONL files, and one integer retrieval verdict per chunk.
  - It never stores passage text.
  - It is rebuilt from the canonical files when it is missing.
- The stored verdict is a bitmask over properties of the chunk alone — corrupt text, extraction artifact — and it mirrors the query-time check exactly.
- `AGENTS.md` carries the rules that govern that verdict.

## Generations

- `ingest` hashes every source.
- An exact input match returns the selected generation unchanged.
- Otherwise `ingest` reuses compatible unchanged documents, chunks, and exact-text vectors, and builds complete new BM25 and dense indexes.
- Work is checkpointed between bounded units, so an interrupted build resumes and `status.ingestion_progress` reports one.
- `current.json` changes only after both indexes succeed, so a build that fails or is cancelled leaves the selected generation in place.
- A cancelled or timed-out build keeps its checkpoint; a build whose inputs no longer match supersedes it with a small diagnostic; a build that cannot resume leaves only a small record under `failures/` and removes its heavy staging data.
- Raw coordinate extraction is not generated.
- UltraRAG raw chunks are staging data only and do not survive activation.

## Sources

- Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed.
- Markdown, symlinks, and everything outside that directory are ignored.
- Nothing in this app ever edits an original.
- A `source_id` combines the project id with the normalized source-relative path, so replacing a file's bytes preserves it and renaming or moving the file changes it.
- A `document_id` identifies one path-and-content version.
  - It changes between generations and is never reported in an answer.
- A `chunk_id` belongs to the generation that returned it and may change after a rebuild.
- A source's locator is the position alone: a page, carrying `page_label` only where the printed label differs from the physical page, or a section for an EPUB.
- Cleaned semantic text is what a search returns and what is indexed.
  - It is never a transcript, so `text` is not quotable.
  - `direct_quote_safe` is `false` on every passage the app builds.
  - In passage context, `excluded_from_search: true` marks a reviewed excluded neighbour; it is context, not an eligible search hit.
- The untouched original at `source_relative_path` and `locator` is the quote authority.

## Review state

### source-metadata.json

- A metadata entry is the complete override for that source.

```json
{
  "schema_version": 1,
  "sources": {
    "Harvey, The Fetish of Technology - Causes and Consequences.pdf": {
      "title": "The Fetish of Technology: Causes and Consequences",
      "authors": ["David Harvey"],
      "year": 2003,
      "doi": "",
      "categories": ["Commodity fetishism", "marxism", "media theory"],
      "keywords": ["fetishism of technology", "technology", "ideology", "marxism"],
      "project": ["ai-and-fetishism"]
    }
  }
}
```

- A hand edit applies at the next read, covering the inventory, the filters, the search results, the citations, and the neighbouring passages, with no ingestion.
- `status.metadata_revision` changes after a hand edit.
- `generation_metadata_snapshot_outdated` may become true after a hand edit, and it says only that the generation predates the review.
- Omitting a field stops overriding it, so the extracted or automatic value returns.
- An explicitly empty list or string clears the field it stands for.
- An entry of `{}`, or the whole entry deleted, clears every reviewed field for that source.
- `research-rag metadata` and the workspace dialog rewrite only the named source's entry, so every other hand edit in the file survives.
- An unknown field name is rejected with `Unsupported metadata fields: …`.
- A wrong type, such as `"year": "2003"`, is rejected with that field's rule.
- A source path that is absolute, that contains `..` or a backslash, or that is otherwise not normalized is rejected.
- Any `schema_version` other than 1 is rejected.
- Do not edit this file while a workspace is running for the project, because the read path honours a hand edit and a write in flight rewrites the same file.
- `metadata_provenance` in the source inventory reports which values are reviewed and which are still automatic.

### source-exclusions.json

```json
{
  "schema_version": 1,
  "sources": {
    "evidence.pdf": {"included": false, "reason": "Reviewed duplicate"}
  }
}
```

- An exclusion is explicit, reversible, and enforced immediately by every retrieval surface.
- An exclusion never touches the source file.
- Re-ingesting omits the excluded source from new indexes.
- Nothing duplicates a file automatically.

### chunk-exclusions.json

- A chunk exclusion is the same decision about one passage instead of a whole file.

```json
{
  "schema_version": 1,
  "chunks": {
    "chk_3f1c0a9b2e7d8456": {
      "reason": "Table header repeated as a passage",
      "excluded_at": "2026-09-30T19:12:35.104Z",
      "generation_id": "20260930T191235Z-45608dc5"
    }
  }
}
```

| Field | Meaning |
|---|---|
| `chunks` | one entry per excluded passage, keyed by the `chunk_id` a search returned |
| `reason` | why the decision was made; a non-empty string, required |
| `excluded_at` | when it was recorded, as an ISO 8601 timestamp |
| `generation_id` | the generation the decision was recorded in |

- A hand edit applies at the next read, covering searches, the passage request, and the inventory, with no ingestion and no rebuild.
- The filter reads the file, so a decision holds for the generation on screen and for every one built after it.
- An ingestion leaves the excluded passage in the indexes with the exclusion still over it.
- `research-rag include --chunk CHUNK_ID` removes an entry, and deleting the entry by hand does the same.
- A decision is never inferred from the corpus, because nothing is excluded for looking like something else.
- A `chunk_id` is derived from content, so unchanged text keeps it across a rebuild, and any other text changes it.
  - `generation_id` is what makes an entry this generation cannot match identifiable by hand.
  - The identifier alone cannot tell those two cases apart.
- `research-rag status` counts the entries recorded against another generation, and the workspace's chunk panel reports each entry as withheld or not.
- An entry whose passage this generation does not hold is neither an error nor a stale corpus, because no ingestion brings a removed chunk back.
  - Remove such an entry when you no longer want it.
- An unknown field name is rejected with `Unsupported chunk exclusion fields: …`.
- A missing or empty `reason`, `excluded_at`, or `generation_id` is rejected by name.
- A `chunk_id` that is not a non-empty string is rejected.
- Any `schema_version` other than 1 is rejected.
- This file is separate from `source-exclusions.json` because it decides one passage rather than a whole source.
  - Excluding a whole source is recorded in `source-exclusions.json`.

### Filter layers

- Reviewed metadata carries six independent layers, and all six are authoritative at read time:

| Field | Meaning | Match |
|---|---|---|
| `project` | which project a source was gathered for | any of |
| `categories` | the branch or branches it belongs to | any of |
| `keywords` | terms that identify it or that it leans on | all of |
| `language` | what it is written in | any of |
| `title` | the work's title | case-insensitive substring |
| `authors` | who wrote it, one string per name | case-insensitive substring |

- `title` is a scalar string where the other fields are lists, so it is normalized on its own.
- Authors and titles are matched as substrings because a name is a phrase rather than a controlled tag.
- Language is the only field extraction detects.
- Extraction records its language decision in `metadata_provenance.language`, as `pdf_catalog`, `epub_opf`, `text_sample`, or `missing`.
- Filtering resolves the current overlay to document IDs at query time.
- Metadata is never baked into an index.

## Extraction decisions worth knowing

- A corrupt extraction unit is omitted whole, with a locator and a reason.
- Rejected wording is never reconstructed, repaired, or invented.
- Script mixing and non-Latin dominance never withhold a passage.
- A foreign-language quotation inside an English source is evidence.
- Formula-font letters (`𝑀` to `M`) and the `ﬁ`, `ﬂ`, `ﬀ` ligatures are folded to plain spellings during cleaning, so a typed query matches the printed text.
- Accents, superscripts, and subscripts are unchanged by that cleaning.
- A chunk with at least one Unicode letter or digit is not symbol-only, and that holds for an alphanumeric formula or numeric content.
- `dense_truncated_chunk_count` in an ingestion answer means some chunks run past the embedding model's token limit, so their dense vector covers a prefix while BM25 still matches the whole text.
  - Read a truncated passage by its locator.
- Layout wrapping is normalized before chunking, controls and soft hyphens are removed, and alphabetic line-end hyphen splits are joined.
- Every other word and punctuation mark is preserved.

## Relocated derived state

- A project on slow storage can point its derived state at a fast device:

```bash
research-rag \
  --project-root /mnt/data/projects/ai-and-fetishism \
  --runtime-root /ssd/research-runtime/ai-and-fetishism
```

- `RESEARCH_RAG_RUNTIME_ROOT` is the equivalent variable.
- The first run claims an empty directory by writing `.research-rag-runtime.json`, naming this project's `project_id`.
- A root is refused when it is not absolute, when its marker names a different project, when it is non-empty with no marker, or when it is a file.
- Only derived state moves, and your portable review state stays in `<project>/.research-rag`.
- `status.runtime_root` reports the effective location, and `null` when the default is in use.
- Drop the option and relocate the directory to move back.

## Outside the project

- `research-rag install` puts its launcher in `~/.local/bin`.
  - `--desktop` writes a menu entry in `~/.local/share/applications` and its icon in `~/.local/share/icons/hicolor/scalable/apps`.
- One account's project register lives beside its settings, at `~/.config/research-rag/projects.json`.
- That register holds one entry per project, and each entry holds its `project_id`, `project_name`, `project_root`, and `registered_at`.
- The register sits outside the project directory on purpose, because a register that travelled inside a project could not list the projects that had none.
- The register holds no corpus, no index, no review, and no derived state.
- A deleted project costs only its name in that file, and deleting the register is always safe: `research-rag init` restores the names.
- A damaged register is reported rather than guessed at.
- An entry that is not a pointer is skipped, so one unreadable project cannot hide the rest.

## Versioned state

- `schema_version` appears in every portable JSON file and is `1`.
- Any other value is refused, so an unknown version stops the app rather than being read on a guess.
- Schema versions, policy versions, the retrieval-method set, and the state-root boundary names are defined in code rather than in a settings file.
- Those four decide what a generation is or where the app may write, and a settings file must not be able to forge either.

## Moving a project

- Copy the project directory: `sources/` plus `.research-rag/` is a complete project on another disk or another machine.
- A relocated runtime root stays claimed by its owning project marker, so never point a second project at one.
