# Storage contract

Every file this app writes, what it means, and which is worth backing up. Read this before editing a project by hand.

## The tree

```text
my-research-project/
├── sources/                              untouched PDF/EPUB originals
├── open-research-rag-ui.sh               symlink to .research-rag/bin/open-research-rag-ui.sh
└── .research-rag/                        all project state
    ├── bin/open-research-rag-ui.sh       generated machine-local workspace launcher
    ├── project.json                      stable id, name, source directory
    ├── source-catalog.json               durable source-id-to-path registry
    ├── source-metadata.json              authoritative reviewed metadata overlay
    ├── source-exclusions.json            reviewed decisions, when present
    ├── chunk-exclusions.json             reviewed decisions about single passages, when present
    ├── config.toml                    this project's settings layer
    └── runtime/                       disposable derived state
        ├── current.json                  selected generation pointer
        ├── project.lock
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

~/.cache/research-ultra-rag-mcp/models/  shared model binaries only
```

Two roots carry the name of the MCP server this app was seeded from rather than the app's own, so both products read the same machine during the migration:

- `~/.config/research-ultra-rag-mcp/config.toml` is the user settings file.
- `~/.cache/research-ultra-rag-mcp/models/` holds the embedding and reranker binaries, about 150 MB in total. It is the only cross-project shared state, and only because those files are immutable once downloaded.

`tests/test_data_roots.py` asserts both, with the reason each is retained. Renaming either is a migration, not a refactor.

Everything else inside `.research-rag/` is byte-compatible with `research-ultra-rag-mcp`, which is frozen and still installed on the machines that carry it. The app names no product in the on-disk state, so both read and write the same project. A field or schema version changed here is read by that product too, so the format is frozen while it is installed.

## Portable state

These six files are your decisions about the project, the part worth backing up, and the part that survives a rebuild.

| File | What it holds | Keyed by |
|---|---|---|
| `project.json` | project name, stable id, sources directory | — |
| `config.toml` | this project's settings layer, one key per setting | setting key |
| `source-catalog.json` | the durable source-id registry | normalized source-relative path |
| `source-metadata.json` | the reviewed metadata overlay | normalized source-relative path |
| `source-exclusions.json` | each exclusion decision and its reason | normalized source-relative path |
| `chunk-exclusions.json` | each passage exclusion, its reason, and the generation it was recorded in | `chunk_id` |

`source-catalog.json` is written by `research-rag sources`, not by hand. Leave it alone.

### project.json

```json
{
  "schema_version": 1,
  "project_id": "c0ffee...",
  "name": "My research project",
  "source_directory": "sources"
}
```

`project_id` is authoritative and never rewritten. `source_directory` is relative to the project root, and every command reuses it when `--source-directory` is omitted; a differing explicit value is refused rather than quietly accepted.

### config.toml

The project's own settings layer, written by hand or by the workspace in the browser. Each key is a `SETTING` in `src/research_rag/default.toml` under a table named by its section, so `[retrieval]` holds `retrieval.rrf_k`. The browser rewrites the whole file from the values it read, so keys it does not know about are preserved and comments are not.

A key the environment, the command line, or `--config` supplied cannot be written here, because those layers outrank this file. `research-rag config` prints every value with the layer it came from and what a change to it costs.

## Derived state

Everything under `.research-rag/runtime/` is rebuilt from `sources/` plus the portable state. Deleting it costs one ingestion.

`current.json` names the one generation search uses. Earlier successful generations stay on disk and are not searched; nothing prunes them automatically. `research-rag status` reports `retained_generation_count` and `retained_generation_bytes`, and `generations` lists each with its creation time, chunk and document counts, file count, and size. A directory whose manifest is missing or unreadable is reported with `manifest_error` rather than failing the call.

Two commands move a generation, and both refuse the one search reads:

- `research-rag generations --use GENERATION_ID` points the project at a retained generation. It validates that generation's artifacts and both indexes exactly as a build's activation does, and only then rewrites `current.json`, so a rollback that lands on a damaged generation fails instead of bricking every read surface. The originals are not rebuilt, so the corpus the pointer now describes is whatever that generation indexed.
- `research-rag remove-generation GENERATION_ID --confirm GENERATION_ID` deletes one permanently. The repeat is the check: a generation named by a listing and removed by a copy of that listing is a mistake that cannot be walked back. It also refuses a generation a pending activation has named, because that activation is about to move it into place. The space comes back only from a rebuild, which costs one ingestion; the original files are not touched.

A running app may still hold a generation: the retrieval gateway keeps a live index against whichever generation it last loaded, and the exact dense backend memory-maps a generation's vector file for the process's lifetime. Neither refuses a delete — an unlinked mapping keeps reading the old bytes until its last reference drops, and a later open returns a missing-file error rather than a failure at the delete. That is why both commands take the project lock and refuse the current generation rather than merely warning about it.

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

`manifest.json` decides what a generation is. It carries the schema version, the extraction policy, the chunking configuration, the retrieval-policy fingerprint, the model and revision that produced the vectors, the dense backend, and the file map. Two generations are interchangeable only when their manifests agree, which makes reuse safe; a generation whose policy fingerprint does not match the current settings is reported as requiring a new ingestion.

`indexes/` names vary by backend; the manifest records which one a generation uses. Never assume a fixed directory name.

### The artifact lookup

`artifact-lookup.sqlite3` holds only identifiers, ordinals, content hashes, byte offsets into the canonical JSONL files, and one integer retrieval verdict per chunk. It never stores passage text, and it may be rebuilt from the canonical files when it is missing.

The stored verdict is a bitmask over properties of the chunk alone — corrupt text, extraction artifact — and it must mirror the query-time check exactly. `AGENTS.md` has the rules that govern it.

## Generations

`ingest` hashes every source. An exact input match returns the selected generation unchanged; otherwise it reuses compatible unchanged documents, chunks, and exact-text vectors and builds complete new BM25 and dense indexes. Work is checkpointed between bounded units, so an interrupted build resumes. `status.ingestion_progress` reports an unfinished build.

`current.json` changes only after both indexes succeed. A cancelled or timed-out build keeps its checkpoint; a build whose inputs no longer match supersedes the checkpoint with a small diagnostic; a build that cannot resume leaves only a small record under `failures/` and removes its heavy staging data.

Raw coordinate extraction is not generated. UltraRAG raw chunks are staging data only and do not survive activation.

## Sources

Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown, symlinks, and everything outside the directory are ignored, and nothing in this app ever edits an original.

- A `source_id` combines the project id with the normalized source-relative path, so replacing a file's bytes preserves it and renaming or moving the file changes it.
- A `document_id` identifies one path-and-content version. It changes between generations and is not reported in an answer.
- A `chunk_id` belongs to the generation that returned it and may change after a rebuild.
- A source's locator is the position alone: a page, carrying `page_label` only where the printed label differs from the physical page, or a section for an EPUB.

Cleaned semantic text is what a search returns and what is indexed. It is never a transcript, so `text` is not quotable and `direct_quote_safe` is `false` on every passage the app builds. The untouched original at `source_relative_path` and `locator` is the quote authority.

## Review state

### source-metadata.json

A metadata entry is the **complete** override for that source.

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

How a hand edit behaves:

- It applies at the next read — inventory, filters, search results, citations, and neighbouring passages — with no ingestion. `status.metadata_revision` changes, and `generation_metadata_snapshot_outdated` may become true, which only says the generation predates this review.
- Omitting a field stops overriding it, so the extracted or automatic value returns. An explicitly empty list or string clears it.
- An entry of `{}` clears every reviewed field for that source; so does deleting the whole entry.
- `research-rag metadata` and the workspace dialog rewrite only the named source's entry, so every other hand edit in the file survives.

Mistakes fail loudly:

- an unknown field name is rejected with `Unsupported metadata fields: …`;
- a wrong type, such as `"year": "2003"`, is rejected with that field's rule;
- a source path that is absolute, contains `..` or a backslash, or is otherwise not normalized is rejected;
- any `schema_version` other than 1 is rejected.

Do not edit this file while a workspace is running for the project: the read path honours a hand edit, but a write in flight rewrites the same file.

`metadata_provenance` in the source inventory reports which values are reviewed and which are still automatic.

### source-exclusions.json

```json
{
  "schema_version": 1,
  "sources": {
    "evidence.pdf": {"included": false, "reason": "Reviewed duplicate"}
  }
}
```

Exclusion is explicit, reversible, and enforced immediately by every retrieval surface, without touching the source file. Re-ingesting omits the source from new indexes. There is no automatic duplicate guessing.

### chunk-exclusions.json

The same decision about one passage instead of a whole file, for a reader who objects to a passage rather than to the work it came from.

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

How a hand edit behaves:

- It applies at the next read — searches, the passage request, and the inventory — with no ingestion and no rebuild. The filter reads the file, so the decision holds for the generation on screen and for every one built after it, and an ingestion leaves the excluded passage in the indexes with the exclusion still over it.
- An entry is removed by running `research-rag include --chunk CHUNK_ID`, or by deleting it here. A decision is never inferred from the corpus: nothing is excluded because it looks like something else.
- `generation_id` is what makes an entry this generation cannot match identifiable by hand. A `chunk_id` is derived from content, so unchanged text keeps it across a rebuild and anything else changes it, and the two cannot be told apart from the identifier alone. `research-rag status` counts the entries recorded against another generation, and the workspace's chunk panel reports each entry as withheld or not.
- An entry whose passage this generation does not hold is not an error and not a stale corpus: nothing is withheld from the passages you can reach, and no ingestion brings a removed chunk back. Remove the entry when you no longer want it.

Mistakes fail loudly, with the same rules the reviewed-metadata file uses:

- an unknown field name is rejected with `Unsupported chunk exclusion fields: …`;
- a missing or empty `reason`, `excluded_at`, or `generation_id` is rejected by name;
- a `chunk_id` that is not a non-empty string is rejected;
- any `schema_version` other than 1 is rejected.

`research-ultra-rag-mcp` is frozen and still reads the projects this app serves. It resolves `project.json`, `source-metadata.json`, `source-exclusions.json`, and `source-catalog.json` by name, so this file is invisible to it, and that is accepted: while both products are installed, a passage excluded here is still served by the frozen product. Excluding a whole source is the decision both products honour, and the file above is the one that decides it.

### Filter layers

Reviewed metadata carries six independent layers, all authoritative at read time:

| Field | Meaning | Match |
|---|---|---|
| `project` | which project a source was gathered for | any of |
| `categories` | the branch or branches it belongs to | any of |
| `keywords` | terms that identify it or that it leans on | all of |
| `language` | what it is written in | any of |
| `title` | the work's title | case-insensitive substring |
| `authors` | who wrote it, one string per name | case-insensitive substring |

`title` is a scalar string where the others are lists, so it is normalized on its own. Authors and titles are matched as substrings because a name is a phrase rather than a controlled tag. Language is the only field extraction detects, and it records its decision in `metadata_provenance.language` (`pdf_catalog`, `epub_opf`, `text_sample`, or `missing`).

Filtering resolves the current overlay to document IDs at query time. Metadata is never baked into an index.

## Extraction decisions worth knowing

- Corrupt extraction units are omitted whole, with a locator and a reason. Never reconstruct, repair, or invent rejected wording.
- Script mixing and non-Latin dominance are never a reason to withhold. A foreign-language quotation inside an English source is evidence.
- Formula-font letters (`𝑀` to `M`) and the `ﬁ`, `ﬂ`, `ﬀ` ligatures are folded to plain spellings during cleaning, so a typed query matches the printed text. Accents, superscripts, and subscripts are unchanged.
- A chunk with at least one Unicode letter or digit is not symbol-only, including an alphanumeric formula or numeric content.
- `dense_truncated_chunk_count` in an ingestion answer means some chunks run past the embedding model's token limit, so their dense vector covers a prefix while BM25 still matches the whole text. Read those passages by their locator.
- Layout wrapping is normalized before chunking, controls and soft hyphens are removed, and alphabetic line-end hyphen splits are joined. Every other word and punctuation mark is preserved.

## Relocated derived state

If a project lives on slow storage, point its derived state at a fast device:

```bash
research-rag \
  --project-root /mnt/data/projects/ai-and-fetishism \
  --runtime-root /ssd/research-runtime/ai-and-fetishism
```

`RESEARCH_ULTRARAG_RUNTIME_ROOT` is the equivalent variable.

- The path must be absolute.
- The first run claims an empty directory by writing `.research-ultra-rag-runtime.json` naming this project's `project_id`. A root whose marker names a different project, a non-empty root with no marker, and a path that is a file are all refused.
- Only derived state moves. Your portable review state stays in `<project>/.research-rag`.
- `status.runtime_root` reports the effective location, and `null` when the default is in use. Drop the option and relocate the directory to move back.

## Outside the project

One account's project register lives beside its settings, at `~/.config/research-ultra-rag-mcp/projects.json`: one entry per project holding its `project_id`, `project_name`, `project_root`, and `registered_at`. It is deliberately outside the project directory, because a register that travelled inside a project could not list the projects that had none. It holds no corpus, no index, no review, and no derived state, so a deleted project costs only its name in that file, and a deleted register costs only the names, which `research-rag init` restores. Deleting the file is always safe; a damaged one is reported rather than guessed at, and an entry that is not a pointer is skipped so one unreadable project cannot hide the rest.

## Versioned state

`schema_version` appears in every portable JSON file and is `1`. Any other value is refused, so an unknown version stops the app rather than being read on a guess.

Schema versions, policy versions, the retrieval-method set, and the state-root boundary names are defined in code, not in a settings file: they decide what a generation is or where the app may write, and a settings file must not be able to forge either.

## Moving a project

Copy the project directory. `sources/` plus `.research-rag/` is a complete project on another disk or machine; `runtime/` is disposable and rebuilds on the next ingestion. A relocated runtime root stays claimed by its owning project marker, so never point a second project at one.
