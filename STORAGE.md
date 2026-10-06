# Storage contract

Read this before editing a project by hand.

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
    ├── config.toml                       this project's settings layer
    └── runtime/                          disposable derived state
        ├── current.json                  selected generation pointer
        ├── project.lock
        ├── research-rag-ui.port          port a running app serves, while it runs
        ├── research-rag-ui.pid           its process, while it runs
        ├── research-rag-ui.tty           the terminal it is attached to; absent when it has none
        ├── search-stats.sqlite3          what searches returned at their first five ranks, and the questions kept while search history is on
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
- `~/.config/research-rag/projects.json` is the account's project register.
- `projects.lock` beside the register serializes read-modify-write operations across processes. Readers use atomic snapshots without the lock.
- `~/.cache/research-rag/models/` holds the embedding and reranker binaries, about 150 MB in total. It is the only cross-project shared state, because the binaries are immutable once downloaded.
- `tests/project/test_data_roots.py` asserts both roots and why each is named for this app.

## Portable state

- These six files are the project decisions worth backing up. They survive a rebuild.

| File | Holds | Keyed by |
|---|---|---|
| `project.json` | project name, stable id, sources directory | — |
| `config.toml` | this project's settings layer, one key per setting | setting key |
| `source-catalog.json` | the durable source-id registry | normalized source-relative path |
| `source-metadata.json` | the reviewed metadata overlay | normalized source-relative path |
| `source-exclusions.json` | each exclusion decision and its reason | normalized source-relative path |
| `chunk-exclusions.json` | each passage exclusion, its reason, and the generation it was recorded in | `chunk_id` |

- Do not hand-edit `source-catalog.json`. `src/research_rag/core/source_inventory.py` rewrites it from the current scan whenever it is stale. That is the cost of remembering a source the project has seen and no longer holds.
- Every portable JSON file carries `schema_version`, currently `1`. The app refuses any other value rather than reading it on a guess.
- Schema versions, policy versions, the retrieval-method set, and the state-root boundary names live in code, not in a settings file. A settings file must not be able to forge what a generation is or where the app may write.

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
- Every command reuses `source_directory` when `--source-directory` is omitted. A differing `--source-directory` is refused.

### config.toml

- The project's own settings layer, written by hand or by the workspace.
- Every key is a `Setting` declared in `src/research_rag/project/settings.py`. Its default is in `src/research_rag/project/default.toml` under a table named for its section, so `[retrieval]` holds `retrieval.rrf_k`.
- A browser save rewrites the whole file from the values it read.
  - It preserves keys it does not know.
  - It drops comments.
- A key supplied by the environment, the command line, or `--config` cannot be written here, because those layers outrank this file.
- `research-rag config` prints every value with its layer and change cost.

## Derived state

- Everything under `.research-rag/runtime/` is rebuilt from `sources/` plus the portable state. Deleting it costs one ingestion.
  - `search-stats.sqlite3` is the exception: it is this machine's search counts and kept questions, and no ingestion rebuilds it. Deleting it resets the counts and the history and nothing else.
- `current.json` names the one generation search uses.
- Earlier successful generations stay on disk, are not searched, and are not pruned automatically.
- `research-rag status` reports `retained_generation_count` and `retained_generation_bytes`.
- `research-rag generations` lists every generation with creation time, chunk and document counts, file count, and size. A generation whose manifest is missing or unreadable shows `manifest_error`.
- Both commands that move a generation take the project lock and refuse the generation search reads.
- Reads take no project lock. A removal waits for a read of the generation it removes, and refuses after the lock wait.
- `research-rag generations --use GENERATION_ID` points the project at a retained generation.
  - It validates the artifacts and both indexes as a build's activation does.
  - It rewrites `current.json` only after validation succeeds, so a rollback onto a damaged generation fails instead of breaking every read surface.
  - The corpus the pointer now describes is whatever that generation indexed.
- `research-rag remove-generation GENERATION_ID --confirm GENERATION_ID` deletes one generation permanently.
  - The repeated id is the check, because a generation named by a listing and removed from a copy of that listing cannot be recovered.
  - It refuses a generation a pending activation names.
  - Space returns only through a rebuild, which costs one ingestion.
  - It never touches original files.
- A running app can hold a generation. The gateway keeps a live index against the generation it last loaded, and the exact dense backend memory-maps the vector file for the process's lifetime. An unlinked mapping keeps reading the old bytes, and a later open fails with a missing-file error. That is why both commands refuse the current generation.

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

- `manifest.json` decides what a generation is. It carries the schema version, extraction policy, chunking configuration, retrieval-policy fingerprint, the model and revision that produced the vectors, the dense backend, and the file map.
- `retrieval.bm25.language` records the stopword list the BM25 index was built with. It is `language.bm25_stopwords` and can differ from `language.corpus`.
- Two generations are interchangeable only when their manifests agree, which makes reuse safe.
- A generation whose policy fingerprint does not match current settings requires a new ingestion.
- Directory names under `indexes/` vary by backend. No reader may assume a fixed name.

### The artifact lookup

- `artifact-lookup.sqlite3` holds identifiers, ordinals, content hashes, byte offsets into the canonical JSONL files, and one integer retrieval verdict per chunk.
  - It never stores passage text.
  - It is rebuilt from the canonical files when missing.
- The verdict is a bitmask over properties of the chunk alone, such as corrupt text and extraction artifact. It mirrors the query-time check exactly.

## Ingestion

- `ingest` hashes every source.
- An exact input match returns the selected generation unchanged.
- Otherwise `ingest` reuses compatible unchanged documents, chunks, and exact-text vectors, and builds complete new BM25 and dense indexes.
- Work is checkpointed between bounded units. An interrupted build resumes, and `status.ingestion_progress` reports it.
- `current.json` changes only after both indexes succeed. A failed or cancelled build leaves the selected generation in place.
- A cancelled or timed-out build keeps its checkpoint.
- A build whose inputs no longer match supersedes the checkpoint with a small diagnostic.
- A build that cannot resume leaves a small record under `failures/` and removes its heavy staging data.
- UltraRAG raw chunks are staging data and do not survive activation. Raw coordinate extraction is not generated.

## Sources

- Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown, symlinks, and everything outside it are ignored.
- The app never edits an original.
- A `source_id` combines the project id with the normalized source-relative path. Replacing a file's bytes preserves it. Renaming or moving the file changes it.
- A `document_id` identifies one path-and-content version. It changes between generations and is never reported in an answer.
- A `chunk_id` belongs to the generation that returned it and can change after a rebuild.
- A locator is a position only: a page, or a section for an EPUB. It carries `page_label` only where the printed label differs from the physical page.
- Search returns, and the index holds, cleaned semantic text. It is not a transcript, so `text` is not quotable. `direct_quote_safe` is `false` on every passage the app builds.
- In passage context, `excluded_from_search: true` marks a reviewed excluded neighbour. It is context, not an eligible hit.
- The untouched original at `source_relative_path` and `locator` is the quote authority.

## Review state

### source-metadata.json

- An entry is the complete override for that source.

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

- A hand edit applies at the next read to the inventory, filters, results, citations, and neighbouring passages. No ingestion is needed.
- `status.metadata_revision` changes after a hand edit.
- `generation_metadata_snapshot_outdated` can become true after a hand edit. It says only that the generation predates the review.
- Omitting a field stops overriding it, and the extracted or automatic value returns.
- An explicitly empty list or string clears that field.
- An entry of `{}`, or a deleted entry, clears every reviewed field for the source.
- `research-rag metadata` and the workspace dialog rewrite only the named source's entry, so other hand edits survive.
- Do not edit this file while a workspace runs for the project. A write in flight rewrites the same file.
- `metadata_provenance` in the source inventory reports which values are reviewed and which are automatic.
- The app rejects:
  - An unknown field, with `Unsupported metadata fields: …`.
  - A wrong type, such as `"year": "2003"`, with that field's rule.
  - A source path that is absolute, contains `..` or a backslash, or is otherwise not normalized.
  - Any `schema_version` other than 1.

### source-exclusions.json

```json
{
  "schema_version": 1,
  "sources": {
    "evidence.pdf": {"included": false, "reason": "Reviewed duplicate"}
  }
}
```

- An exclusion is explicit, reversible, and enforced at once by every retrieval surface.
- It never touches the source file.
- Re-ingesting omits the excluded source from new indexes.
- Nothing duplicates or excludes a file automatically.

### chunk-exclusions.json

- A chunk exclusion is the same decision about one passage.

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
| `reason` | why the decision was made; a required non-empty string |
| `excluded_at` | when it was recorded, as an ISO 8601 timestamp |
| `generation_id` | the generation the decision was recorded in |

- A hand edit applies at the next read to searches, the passage request, and the inventory. No ingestion or rebuild is needed.
- The filter reads the file, so a decision holds for the generation on screen and every later one.
- An ingestion leaves the excluded passage in the indexes with the exclusion still over it.
- `research-rag include --chunk CHUNK_ID` removes an entry. Deleting the entry by hand does the same.
- A `chunk_id` derives from content. Unchanged text keeps it across a rebuild, and any other text changes it. `generation_id` lets a reader tell the two cases apart for an entry this generation cannot match.
- `research-rag status` counts entries recorded against another generation. The workspace's chunk panel reports each entry as withheld or not.
- An entry whose passage this generation does not hold is neither an error nor a stale corpus, because no ingestion brings a removed chunk back. Remove it when no longer wanted.
- This file is separate from `source-exclusions.json` because it decides one passage, not a whole source.
- The app rejects:
  - An unknown field, with `Unsupported chunk exclusion fields: …`.
  - A missing or empty `reason`, `excluded_at`, or `generation_id`, by name.
  - A `chunk_id` that is not a non-empty string.
  - Any `schema_version` other than 1.

### Filter layers

- Reviewed metadata carries six independent layers, all authoritative at read time.

| Field | Meaning | Match |
|---|---|---|
| `project` | which project a source was gathered for | any of |
| `categories` | the branch or branches it belongs to | any of |
| `keywords` | terms that identify it or that it leans on | all of |
| `language` | what it is written in | any of |
| `title` | the work's title | case-insensitive substring |
| `authors` | who wrote it, one string per name | case-insensitive substring |

- `title` is a scalar string where the other fields are lists, so it is normalized on its own.
- Authors and titles match as substrings because a name is a phrase, not a controlled tag.
- Language is the only field extraction detects. `metadata_provenance.language` records the decision as `pdf_catalog`, `epub_opf`, `text_sample`, or `missing`.
- Filtering resolves the current overlay to document IDs at query time. Metadata is never baked into an index.

## Extraction behaviour

- A corrupt extraction unit is omitted whole, with a locator and a reason.
- Rejected wording is never reconstructed, repaired, or invented.
- Script mixing and non-Latin dominance never withhold a passage. A foreign-language quotation inside an English source is evidence.
- A chunk with at least one Unicode letter or digit is not symbol-only, including alphanumeric formulas and numeric content.
- Layout wrapping is normalized before chunking, controls and soft hyphens are removed, and alphabetic line-end hyphen splits are joined. Every other word and punctuation mark is preserved.
- `dense_truncated_chunk_count` in an ingestion answer means some chunks run past the embedding model's token limit. Their dense vector covers a prefix, and BM25 still matches the whole text. Read a truncated passage by its locator.

## Relocated derived state

- A project on slow storage can put its derived state on a fast device:

```bash
research-rag \
  --project-root /mnt/data/projects/ai-and-fetishism \
  --runtime-root /ssd/research-runtime/ai-and-fetishism
```

- `RESEARCH_RAG_RUNTIME_ROOT` is the equivalent variable.
- The first run claims an empty directory by writing `.research-rag-runtime.json`, which names this project's `project_id`.
- The app refuses a root that is not absolute, whose marker names a different project, that is non-empty with no marker, or that is a file.
- Only derived state moves. Portable review state stays in `<project>/.research-rag`.
- `status.runtime_root` reports the effective location, or `null` for the default.
- To move back, drop the option and relocate the directory.

## Outside the project

- `research-rag install` puts its launcher in `~/.local/bin`.
  - `--desktop` writes a menu entry in `~/.local/share/applications` and its icon in `~/.local/share/icons/hicolor/scalable/apps`.
- The account's project register is `~/.config/research-rag/projects.json`. It sits outside the project directory because a register inside a project could not list projects that had none.
- It holds one entry per project: `project_id`, `project_name`, `project_root`, and `registered_at`. It holds no corpus, index, review, or derived state.
- Deleting the register is always safe, because `research-rag init` restores the names. A deleted project costs only its name in the file.
- A damaged register is reported, not guessed at. An entry that is not a pointer is skipped, so one unreadable project cannot hide the rest.

## Moving a project

- Copy the project directory. `sources/` plus `.research-rag/` is a complete project on another disk or machine.
- A relocated runtime root stays claimed by its owning project's marker. Never point a second project at it.
