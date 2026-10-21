# Storage contract

Read this before editing a project by hand.

## The tree

```text
my-research-project/
├── sources/                              untouched PDF/EPUB/MOBI originals
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
        ├── ultrarag-runtime/              retained inert compatibility path
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
~/.cache/research-rag/backend-envs/<spec>/  the managed Docling environment
```

- `~/.config/research-rag/config.toml` is the user settings file.
- `~/.config/research-rag/projects.json` is the account's project register.
- `projects.lock` beside the register serializes read-modify-write operations across processes. Readers use atomic snapshots without the lock.
- `~/.cache/research-rag/models/` holds the embedding and reranker binaries, about 150 MB in total. It is the only cross-project shared state, because the binaries are immutable once downloaded.
- `~/.cache/research-rag/backend-envs/<spec>/` is Docling's own virtual environment, named by a spec fingerprint of its pinned packages and the interpreter ABI. It is built on the first PDF conversion, or by `doctor --prefetch-models`, under a bounded installer, and published only after its pinned versions are read back. An incomplete install leaves no `ready.json` and is never used; an environment that was already ready is never touched. The app's own environment never carries Docling.
- Model snapshots use `models--<repository>/snapshots/<commit>`, with `/` in the repository replaced by `--`. Health checks require the pinned repository, commit, ONNX files, and tokenizer files. They do not hash the cached binaries.
- `tests/project/test_data_roots.py` asserts both roots and why each is named for this app.
- The `ultrarag-runtime` path and legacy `vanilla_executable` configuration argument remain passive compatibility names. Neither requires a gateway binary or an external runtime. Historical source-snapshot caches are not read, migrated, or deleted by the retirement shims.

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

- The project's own settings layer, written by hand, by the workspace, or by `research-rag config --write`.
- Every key is a `Setting` declared in `src/research_rag/project/settings.py`. Its default is in `src/research_rag/project/default.toml` under a table named for its section, so `[retrieval]` holds `retrieval.rrf_k`.
- A save from the workspace or `config --write` rewrites the whole file from the values it read.
- A running app takes a save at once and a hand edit to any settings file at its next call, reading every layer again as at start.
  - An edit that does not resolve keeps the values in force. The health report names it until the file changes again.
  - A setting the app reads once at start, such as `runtime.nice`, waits for a restart, and the health report names it until then.
  - It preserves keys it does not know.
  - It drops comments.
- A key supplied by the environment, the command line, or `--config` cannot be written here, because those layers outrank this file.
- `research-rag config` prints every value with its layer and change cost. `--write KEY=VALUE` changes this file, and `--confirm` accepts a change that alters what the next build records.
- A legacy `ingestion.pdf_backend="custom"` value resolves to Docling without rewriting the settings file. It never enables a custom extractor.

## Derived state

- Everything under `.research-rag/runtime/` is rebuilt from `sources/` plus the portable state. Deleting it costs one ingestion.
  - `search-stats.sqlite3` is the exception: it is this machine's search counts and kept questions, and no ingestion rebuilds it. Deleting it resets the counts and the history and nothing else.
- `current.json` names the one generation search uses.
- Earlier successful generations stay on disk, are not searched, and are not pruned automatically.
- `research-rag status` reports `retained_generation_count` and `retained_generation_bytes`.
- `research-rag generations` lists every generation with creation time, chunk and document counts, file count, and size.
  - Each record exposes the manifest's `chunking`, `retrieval`, policy versions, and `retrieval_policy_fingerprint`.
  - These are recorded build facts, including embedding and reranker models and revisions, not current settings.
  - A missing or malformed configuration block is `null`; today's defaults never fill historical gaps.
  - A generation whose manifest is missing or unreadable shows `manifest_error`.
- Both commands that move a generation take the project lock and refuse the generation search reads.
- Reads take no project lock. A removal waits for a read of the generation it removes, and refuses after the lock wait.
- `research-rag generations --use GENERATION_ID` points the project at a retained generation.
  - The workspace's Generations panel exposes the same operation through **Load**.
  - It validates the artifacts and both indexes as a build's activation does.
  - It rewrites `current.json` only after validation succeeds, so a rollback onto a damaged generation fails instead of breaking every read surface.
  - The corpus the pointer now describes is whatever that generation indexed.
  - Loading does not restore project settings or change generation artifacts.
  - Current settings still govern model compatibility; incompatible embedding identities restrict search to BM25, and incompatible dimensions fail activation validation.
- `research-rag remove-generation GENERATION_ID --confirm GENERATION_ID` deletes one generation permanently.
  - The repeated id is the check, because a generation named by a listing and removed from a copy of that listing cannot be recovered.
  - It refuses a generation a pending activation names.
  - Space returns only through a rebuild, which costs one ingestion.
  - It never touches original files.
- A running app can hold a generation. The direct backend keeps a live BM25 index against the generation it last loaded, and the exact dense backend memory-maps the vector file for the process's lifetime. An unlinked mapping keeps reading the old bytes, and a later open fails with a missing-file error. That is why both commands refuse the current generation.

### Generation layout

```text
generations/<generation-id>/
  manifest.json                 what this generation contains and the policy that built it
  corpus/extracted-units.jsonl  cleaned extraction units, one JSON object per line
  chunks/chunks.jsonl           the final chunks, one JSON object per line
  portable/embeddings.npy       float32 vectors, row i belongs to chunk i
  indexes/artifact-lookup.sqlite3
  indexes/bm25/                 compatible bm25s index, historical layout unchanged
  indexes/qdrant/               the dense index, when the manifest names it
```

- `manifest.json` decides what a generation is. It carries the schema version, extraction policy, chunking configuration, retrieval-policy fingerprint, the model, download repository and revision that produced the vectors, the dense backend, and the file map.
- `retrieval.dense.embedding_model_repository` records the download repository, which can differ from the configured FastEmbed model name. Older manifests without this field remain compatible only when their model, revision, and dimension match.
- Correcting a mirror's revision requires re-embedding. Existing generations remain unchanged and searchable through BM25 until ingestion builds a compatible generation.
- `retrieval.bm25.language` records the stopword list the BM25 index was built with. It is `language.bm25_stopwords` and can differ from `language.corpus`.
- `passage_cleaning.maximum_unclean_percent` records the per-document substantive-loss cap. The name is preserved; its meaning is the share of a PDF's substantive text it may lose to unrecoverable corruption. Extraction reuse requires the same threshold.
- Recorded policies describe the generation's build; a different current policy does not rewrite its manifest.
- Historical UltraRAG-compatible backend labels remain readable identifiers. Dependency retirement changes no chunk boundaries, index layout, source/chunk IDs, or policy fingerprints and does not trigger ingestion.
- `source_files` records each source the generation holds, with the bytes its document was built from, and each excluded file. A source carried from bytes that have changed since keeps its old record, so `status` reports it modified; a source neither held nor excluded is absent, so `status` reports it added.
- `build_metrics.source_selection` names the sources a build read or took out when it named them; a build of every source has none.
- `build_metrics.phase_timings_seconds` gives each phase's wall time, checkpoint saves and reuse lookups included, through `finalizing`, which ends as the manifest is written. The ingest answer adds `activation`: validation and the pointer switch that follow it.
- A PDF document's `extraction_backend` identifies the extractor that produced its retained text. A generation can contain unchanged legacy PDF artifacts beside new Docling artifacts; its configured backend describes new extraction, not a conversion of every document.
- An EPUB or MOBI document's `reflowable_extraction_policy_version` records the HTML reading rules that produced its units. A record without the field was read under version 1.
- New Docling units retain `docling_label`, `docling_content_layer`, and `docling_ref` beside their provenance. These guide pre-chunk filtering and are not quotation offsets.
- A query-time retrieval-policy fingerprint change does not invalidate stored evidence or require ingestion.
- Directory names under `indexes/` vary by backend. No reader may assume a fixed name.

### The artifact lookup

- `artifact-lookup.sqlite3` holds identifiers, ordinals, content hashes, byte offsets into the canonical JSONL files, retained structural context, and one integer retrieval verdict per chunk.
  - It never stores passage text.
  - It is rebuilt from the canonical files when missing or incompatible with the current lookup schema.
- The verdict is a bitmask over the chunk's text, quality flags, and declared structure: corrupt text, extraction artifact, and scholarly apparatus. It mirrors the query-time check exactly.
  - Search always rejects the first two. It rejects apparatus only while `retrieval.exclude_apparatus` is on, and a passage window shows an apparatus neighbour as context.
  - Query-only verdicts and context never enter canonical records carried into another generation.

## Ingestion

- `ingest` hashes every source.
- An exact artifact-affecting input match returns the selected generation unchanged, even when query-time ranking settings differ.
- A Docling backend or adapter change alone does not reprocess unchanged PDFs. Normal ingestion carries their existing artifacts and provenance forward; `force_recompute` explicitly reprocesses them. Existing generation files are never relabelled in place.
- An EPUB or MOBI document read under earlier HTML rules is read again by ordinary ingestion. Unchanged PDFs still pass through.
- Otherwise `ingest` checks text and vector reuse independently and builds complete new BM25 and dense indexes. Vector reuse requires matching model identity and exact embedding input; a text-processing mismatch does not alone discard reusable vectors.
- Work is checkpointed between bounded units. An interrupted build resumes, and `status.ingestion_progress` reports it.
- `current.json` changes only after both indexes succeed. A failed or cancelled build leaves the selected generation in place.
- A cancelled or timed-out build keeps its checkpoint.
- A failed build keeps its checkpoint and records `last_failure`, unless its staged state is unusable or the same failure repeats three times in one phase.
- A build whose inputs no longer match supersedes the checkpoint with a small diagnostic.
- A build that cannot resume leaves a small record under `failures/` and removes its heavy staging data.
- Raw chunker output is staging data and does not survive activation. Raw coordinate extraction is not generated.
- Historical manifests keep their UltraRAG provenance. The direct backend reads the same saved BM25 layout without rewriting it. New manifests name Chonkie and bm25s as compatible direct implementations; their labels do not invalidate historical chunk and vector reuse.

## Sources

- Only regular `.pdf`, `.epub`, and `.mobi` files beneath the configured source directory are indexed. Markdown and everything outside it are ignored; source symlinks are rejected.
- The app never edits an original.
- A `source_id` combines the project id with the normalized source-relative path. Replacing a file's bytes preserves it. Renaming or moving the file changes it.
- A `document_id` identifies one path-and-content version. It changes between generations and is never reported in an answer.
- A `chunk_id` belongs to the generation that returned it and can change after a rebuild.
- A locator is a position only: a PDF page, an EPUB section, or a logical MOBI part. It carries `page_label` only where the printed label differs from the physical page.
- A MOBI locator has type `mobi_section`: a logical rendition part index, the current heading, and element-path, block-index, or existing `id`/`name` anchors. Its synthetic `part-NNNNNN.html` reference identifies reconstructed text, not a printed page or an EPUB archive path.
- Search returns, and the index holds, cleaned semantic text. It is not a transcript, so `text` is not quotable.
- Document records keep `cleaned_passage_count`, `partially_cleaned_passage_count`, `excluded_corrupt_passage_count`, `substantive_character_count`, `discarded_corrupt_character_count`, `cleaned_corrupt_span_count`, `unclean_character_rate`, and the corresponding `cleaned_passages` and `excluded_corrupt_passages` decisions. Each decision carries a unit ID, a zero-based paragraph index, a locator, its reasons, the span offsets it covers, and the characters it lost, never damaged passage text.
- Historical `pdf_text_recovery_pages` and `recovered_pdf_blocks` record native-reader retries and accepted replacements. They remain readable; new Docling extraction writes zero for both compatibility counters.
- In passage context, `excluded_from_search: true` marks a reviewed excluded neighbour. It is context, not an eligible hit.
- A result names its source by `source_id` and `title`. A path handle (`source_relative_path`) is returned by `find_source` and the workspace source list, not repeated on every passage.
- The untouched original, found through `find_source`, and the `locator` are the quote authority.

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
- An excluded source is never ingested, and an exclusion never makes the generation stale.
- Data a generation already holds for an excluded source stays in later generations, filtered from every search, until `uningest` takes it out.
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
- Language is the only field extraction detects. `metadata_provenance.language` records the decision as `pdf_catalog`, `epub_opf`, `mobi_header`, `text_sample`, or `missing`.
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
- The account's DeepSeek key is `~/.config/research-rag/credentials.json`, written with mode 0600.
  - It holds `{"schema_version": 1, "deepseek": {"api_key": "…"}}`. A cleared key leaves `"deepseek": {}`.
  - It is account state, not project state. Moving a project does not move it, and no project file names it.
  - `RESEARCH_RAG_DEEPSEEK_API_KEY` overrides it.
  - A hand edit is read at the next answer. A later `schema_version`, an unknown provider or field, or a value that is not a string is refused by name, without the key.
  - `doctor` warns when other accounts can read it.

## Moving a project

- Copy the project directory. `sources/` plus `.research-rag/` is a complete project on another disk or machine.
- A relocated runtime root stays claimed by its owning project's marker. Never point a second project at it.
