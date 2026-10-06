# Features

- **UltraRAG** marks a capability from the pinned upstream framework, used as it is.
- **Added here** marks the research layer around UltraRAG.

## What UltraRAG provides

- The UltraRAG gateway exposes about 78 tools and 26 prompts across corpus, retrieval, reranking, prompt, generation, routing, memory, benchmark, and evaluation components. It also ships several vector-database backends and a web interface.
- This app uses three upstream capabilities and owns everything else.
  - The chunker and BM25 are upstream. The chunk size, overlap, which units are chunked, result filtering, and the disclosure of rejected text are this app's.
  - The app needs no upstream service, credential, GPU, or vector database.

| UltraRAG capability | Used | What this app does |
|---|---|---|
| GPT-2 token chunking (`corpus_chunk_documents`) | Yes | Splits extraction units into token chunks of a configured size and overlap. Several units share one call; each keeps its own durable output. |
| BM25 index and search (`retriever_retriever_init`, `retriever_bm25_index`, `retriever_bm25_search`) | Yes | Supplies the lexical half of retrieval, in English, on CPU. |
| FAISS, Qdrant, Milvus backends (`retriever_init(index_backend=...)`) | No | Replaced by FastEmbed CPU embeddings with an exact scan of portable vectors, or an embedded Qdrant collection. |
| Reranking (`reranker_init`, `reranker_rerank`) | No | Replaced by a FastEmbed CPU cross-encoder. |
| Prompt assembly and answer generation | No | Left out. The reader writes the answer. |
| Routing, memory, benchmark, evaluation | No | Left out. They serve multi-corpus pipelines, conversational memory, and boxed-answer scoring. Retrieval quality is measured offline against a judged set. |
| Web-search retrieval | No | Left out. It would send the query to a third party and mix outside text into project evidence. |
| Web interface | No | Replaced by the bundled workspace. |

## Reuse decisions

- An upstream component is reused when it meets the research contract without a new dependency or operational requirement.
- The research contract needs:
  - Stable `chunk_id`, `document_id`, and `source_id` for every hit.
  - A locator into the original file.
  - Query-time metadata filtering resolved against the reviewed overlay.
  - A visible ranking signal.
  - CPU-only, offline, credential-free operation with no external service.
- Any decision below is revisited when the upstream component also:
  - Returns identifiers and scores, not anonymous passage text.
  - Filters by metadata at query time, using values checkable against canonical records.
  - Runs on CPU, offline, pinned by revision, with no server, GPU, or credential.
  - Validates as one index of an immutable generation, so a failed index can never be selected.

| Component | Why not | What replaced it |
|---|---|---|
| FAISS | `search()` returns passage strings from `self.contents[doc_id]`, so identity and scores are lost and no metadata filter exists. The index is built at `retriever_init` and rebuilt over the whole corpus. | An exact cosine scan of the generation's float32 vectors, with rows excluded by document ID during the scan. An exclusion or metadata change applies without a rebuild. |
| Qdrant | `search()` returns only the payload text field. It assigns `uuid5` point IDs instead of canonical chunk IDs and creates its collection at init. | This app's own thin backend over the same `qdrant-client`. It stores integer point IDs with `chunk_id`, `document_id`, and `source_id` as payload, filters with `MatchAny` on document IDs the service resolved, and returns `point.score`. |
| Milvus | Needs a running Milvus service or Milvus Lite. That conflicts with project-local derived state and the no-external-service rule. `search()` returns strings only. Upstream warns against non-demo use, and its demo mode forces OpenAI embeddings. | Nothing. At this app's sizes the exact scan answers from vectors the generation already stores. |
| Reranking components | The `sentence_transformers` backend pulls in PyTorch. `infinity` is a model-serving engine. `openai` needs a credential and sends queries to a third party. Shipped parameters target `device: cuda`. The result is `rerank_psg`, reordered strings with scores discarded. | A FastEmbed cross-encoder on the pinned ONNX dependency and shared model cache. |

- The exact scan is the default. Above 200,000 chunks the embedded Qdrant backend is selected per generation and recorded in its manifest. It returns scored hits under a payload filter, which upstream FAISS cannot.
- The reranker orders at most 50 candidates by score, then appends the unreranked tail, so a source whose best passage fell outside the window can still be reached. A string-only reranker cannot express that.
- The vanilla gateway can start a `reranker` namespace, but this app requests `corpus` and `retriever` only. Adding it would add a second model-serving surface.
- Six FastEmbed cross-encoders are supported, each pinned to a revision in `rerankers.py`. The default is `Xenova/ms-marco-MiniLM-L-6-v2`. A name outside the table is refused.
- Reranking is fixed behaviour. `rerank=false` is reachable only from the engine. When the model cannot load, search falls back to the unranked order and discloses it.

## The workspace

- The upstream interface is bound to upstream pipeline and session state, and it exposes operations this app does not.
- The bundled workspace in `surfaces/workspace/` is dependency-free and owned by this app, with its adapter, source authorization, and host.
- It binds to loopback and serves only an allowlisted original PDF or EPUB.

## Features added on top of UltraRAG

### Ingestion and text quality

| Feature | What it does |
|---|---|
| PDF and EPUB ingestion | Walks the source directory recursively and accepts regular `.pdf` and `.epub` files only. A stray `notes.md` never enters the knowledge base. |
| Layout-aware extraction | Reads PDFs page by page and EPUBs by spine section. Keeps structure and drops repeated page furniture. |
| Pre-chunk cleanup, reported by count | Removes confidently identified page furniture, sidebars, marked small-font footnotes, EPUB furniture markup, and confirmed reference and note sections. Section cleanup follows entries across pages and EPUB elements and stops at ordinary prose. Removal counts are recorded per source. |
| Unicode normalization | Composes to NFC, removes control characters, and folds formula-font letters (`𝑀` to `M`) and the ligatures `ﬁ`, `ﬂ`, `ﬀ`. Combining marks, accents, superscripts, and subscripts are preserved. |
| Source health gate | Refuses sources with no readable units, no text layer, no letters, almost all units withheld, symbol-dominated text, or excessive cleanup. Names the reason and remedy, with a locator when a rejected unit provides one. A failed build preserves the current generation. Mixed readable and image-only pages produce a warning. |
| Rebuild-required policies | Each generation records its extraction and cleaning policy. A change to either reports `generation_upgrade_required`, and the next ingestion re-extracts every source. |
| Original-file locators | Every unit and chunk points to a PDF page (with the printed label when available) or an EPUB section. Locators are navigation aids, not quote offsets. |
| Bibliographic metadata with provenance | Resolves title, authors, year, DOI, and language per document, and records each value's origin and warnings. Language detection compares the source's function words with the BM25 stopword lists and answers nothing when the sample is too short or no list is covered. An answer carries only the authors it resolved. The full-detail payload adds the citation, per-field provenance, and warnings. |
| Reviewed metadata overlay | Corrections saved outside the generation apply at the next read to listings, filters, citations, results, and neighbouring passages. The file is plain JSON. `research-rag metadata` and the workspace dialog write it one source at a time, so a hand edit and a save cannot clobber each other. |
| Corrupt-text rejection | Rejects a page or section only on strong evidence of a broken character map. Reports reason codes, counts, and example chunk IDs. |
| Script mixing never withholds | A Greek, Cyrillic, or other-script quotation stays retrievable and unchanged. An advisory script note appears in the full-detail payload only. |
| Symbol-only chunk exclusion | Drops nonempty chunks with no letters or digits. Numbers and formulas are unaffected. |
| Embedding-token audit | Every chunk records its embedding token count and whether its vector covers only a prefix. Truncation is visible per hit and in aggregate. |
| Contextual chunk headers | `chunking.headers` prepends the source title and section to the text a chunk is embedded from, never to the text a search returns. Off by default. It is an identity setting, so changing it rebuilds. |
| Compatible-material reuse | Unchanged documents keep their units and chunks. Identical chunk text keeps its vector. |
| Resumable ingestion | Builds are checkpointed. A call past its soft time budget returns `in_progress`, and the reader repeats it. Cancellation, timeout, and restart lose at most one bounded batch. |
| Explicit regeneration | `force_recompute` bypasses all reuse and still resumes its own checkpoint. |

### Retrieval

| Feature | What it does |
|---|---|
| BM25 lexical search (UltraRAG) | Matches the typed words. Serves exact names, terms, and phrases. |
| Dense semantic search | FastEmbed `bge-small-en-v1.5`, 384 dimensions, CPU, revision-pinned. Finds passages phrased differently from the question. |
| Hybrid search | Weighted reciprocal-rank fusion of the two rankings, with an opt-out to inspect either alone. |
| Exact dense scan | Scans the generation's portable float32 vectors directly. Above a documented corpus size an embedded index is used, and the manifest records which backend built the generation. |
| Reranking, always on | A CPU cross-encoder reorders up to 50 fused candidates. When the pinned model cannot load, search returns the unranked order and reports `rerank_fallback`. |
| Metadata filters | Narrow by project, category, keyword, or document. `status` shows the project and category inventories. |
| Author and title filters | `authors_any` and `titles_any` match case-insensitive substrings of the reviewed value. `applied_filters` reports a filter that matched nothing. |
| Source selection | `source_ids` includes and `exclude_source_ids` removes named sources by stable ID. Both default to empty. Filters apply before ranking, so `top_k` is a budget inside the selection. |
| Category partitions | `categories_any` requires at least one listed branch, resolved from reviewed metadata at query time. `status.categories` reports each branch with its searchable source count. |
| Project layer | `project` records which project a source was gathered for. `projects_any` requires one of the listed tags. `status.projects` reports each tag with its count. |
| Per-source language | Every source carries the detected language and any reviewed language. `languages_any` keeps results in any listed ISO 639 code. `status.languages` reports counts. |
| Source diversity | `retrieval.source_diversity_penalty` charges a candidate a share of its normalized relevance for each candidate already taken from the same source, at the final `top_k` pick. It only reorders. A bare BM25 or dense ranking has no score to charge and keeps its order. |
| Pseudo-relevance feedback | `retrieval.prf` mines terms from the lexical leaders of a first pass and searches again with them, weighted by rarity across the generation. Off by default. |
| Relevance gates with abstention | Weak candidates are dropped, so a search can return fewer than `top_k` results, including none. A dense candidate below the cosine floor is still admitted when the query's best candidate cleared the floor and this one is within `retrieval.dense_relative_similarity_margin` of it. The full-detail payload reports each gate's rejected count, the best similarity, and the sources the floor dropped. |
| Minimum passage length | `retrieval.minimum_passage_words` and `retrieval.minimum_passage_token_fraction` drop short candidates before fusion in each half. Both are off by default. The full-detail payload reports the rules, counts, and dropped sources. |
| Precomputed usability verdict | Whether a candidate is structurally unusable is computed once when the lookup is built and stored per chunk. |
| Evaluation harnesses | A judged query set and a harness that measures each retrieval mode through the engine the tools call. A second harness drives index builds, durability writes, embedding, and the query gate against the real gateway, models, and service. |

### Project model, review, and durability

| Feature | What it does |
|---|---|
| One project per process | A `--project-root` defines the boundary. All state lives under `<project>/.research-rag`. |
| Many projects per install | `init` records each project's id, name, and root in an account-wide register. `projects` lists them with app state. `--project <name-or-id>` selects one in place of a path. |
| Portable and derived state | Reviewed decisions are portable. Generations, staging, logs, and locks are derived and rebuildable. |
| Source inclusion and exclusion | A source can be excluded as a duplicate and restored. The original is never deleted or modified. |
| Passage exclusions | One passage is excluded by the `chunk_id` a search returned, and restored later. Both halves of a search apply it. The decision holds at once, with no ingestion. |
| Immutable generations | Each build produces a new generation. The active pointer moves only after both indexes validate. |
| Relocatable derived state | `--runtime-root` puts indexes, staging, and logs on a chosen disk, claimed by a marker naming the owning project. A copy cannot silently share it. |
| Copy-based portability | A project moves, is shared, or is archived with ordinary file tools. There is no export format. |
| Editable project settings | The workspace and `research-rag config` show every effective value with its description, source layer, and change cost. A browser save writes the project's `config.toml`. It refuses a value a higher layer supplied, names the keys that force a rebuild before writing, and preserves keys it did not change. |

### Installation and updates

| Feature | What it does |
|---|---|
| Command on `PATH` | `research-rag install` links the running interpreter's console script into the account's binary directory, reports what it wrote, and is a no-op on a second run. `--uninstall` removes only that link. It needs no project. |
| Desktop menu entry | `install --desktop` writes one freedesktop entry per project that starts the project's launcher with `--open`. `--force` is required to replace an entry this app did not write. `doctor` reports an entry whose project root is gone. |
| Version reporting | `research-rag --version` prints the version the process started with, the version installed now, and `restart_required`. `status.version` carries the same three, and the workspace header shows them. |
| One-command update | `research-rag update` compares the installation with the latest release, read from the remote's tags with `git ls-remote`, so it needs no token. `--apply` refuses a held project lock and a tree with uncommitted work, stops every app the installation serves, re-syncs the environment, and prints the start command for each app. |

- `update` reads tags with `git ls-remote --tags`.
  - `DIST_TAG` (`latest`) names the current release rather than a version. A remote that publishes it wins over the highest version tag.
  - A tag with a pre-release or build-metadata suffix is not a release. Two tags claiming one version are refused.
  - A checkout's position is one of `at_release`, `behind_release`, `ahead_of_release`, `no_release`, and `unreadable_release`.
  - Only `behind_release` is an update to apply. `ahead_of_release` is unreleased work and changes nothing.

### Interfaces and operational transparency

| Feature | What it does |
|---|---|
| One running app | `research-rag start` brings up one process per project. It owns the project lock and the UltraRAG gateway and serves the workspace, the agent surface at `/mcp`, and the control API on one loopback port. The gateway opens on the first call that needs it. |
| Workspace attached to the terminal | A bare `research-rag` serves from its terminal and asks which project when several are registered. `--start-ui` opens a browser. Ctrl-C or closing the terminal stops the app and gateway, and an app whose terminal disappears without a signal stops too. A process with no terminal is refused rather than served. A running app is reported, not duplicated. The pid, port, and terminal records let `projects` identify the owner. |
| Reads during a build | `status`, `sources`, search, passages, and the stats answer while a build runs. Only a build's BM25 phase holds a search, and one held past the lock wait is refused with the build named. |
| Client registry and disconnect | The app records every MCP client by declared name and by the program, directory, and process the stdio bridge inherited. One agent is one row however many sessions it opened. An unnamed client shows as its connection kind. A client silent for 15 minutes is forgotten. `research-rag disconnect <session>` refuses that client before its requests reach the tools. |
| Stdio bridge | `research-rag mcp --project-name NAME` proxies a running app and never starts one. It takes no project path. It decides nothing at launch: every call asks whether the app answers, so an agent that connected before the app started, or while it restarted, has all eight tools and finds them working once the app is up. While it is down each call returns the start command, and `status` returns it as a structured answer. A read dropped mid-call is repeated once; a write is never repeated and its error says to check `status`. An unregistered project exposes only `status` with an `init` remedy, and finds the project once `init` has run. |
| Fair queue for several agents | Writes take turns one at a time and searches `runtime.search_concurrency` at a time, default 2. Waiting callers are served in rounds, one agent after another, so an agent asking twenty times waits behind the others' first questions. A wait is bounded, 45 s for a search and 20 s for a write, and a refusal says how many were ahead and to ask again. A second `ingest` during a build is refused with the build's phase, not queued. |
| Core command line | One command per job. A command that touches the corpus reaches the running app over the control API. With no app up, it is answered in process. |
| Dependency report and doctor | `status` reports `blocked_by` and `degraded`, each a list of `{check, reason, remedy}`. `research-rag doctor` prints the same checks one per line and reads without writing. On request it repairs a mismatched UltraRAG runtime or fetches the pinned models. |
| Browser workspace | A loopback-only workspace over the command line's operations, with a tab per job. A search asks for any number of passages from 1 to 50, ten by default. It covers per-query source include and exclude, category filters, a partition list with counts, project settings, health checks with remedies, and a rebuild that reuses nothing, offered apart from ingestion. Each view, search, and page of sources is addressed in the URL fragment, so Back, Forward, and reload return to it. Sources show five to a page, each card's actions sit behind one menu, and a list of more than five starts folded. |
| Project selector and client entry | The workspace names the project on screen and lists the account's others with whether an app is up. The MCP tab shows the agent endpoint and a copyable client entry, and folds attached clients behind one click with the count on the summary. |
| CPU niceness | `runtime.nice` raises the process's niceness once at start, and every child inherits it. Default `10`, which keeps a desktop responsive during a build. |
| Generation reporting | `status` reports whether the selected generation is stale and why, whether a policy upgrade is required, and the count and size of retained generations. `generations` lists each one, including any with an unreadable manifest. `generations --use` validates one and moves the pointer. `remove-generation` deletes one that search does not read, after the id is repeated. |
| Search counts and corpus stats | Every search counts the source and passage at each of its first five ranks, with its time, result count, and kind of caller, in `runtime/search-stats.sqlite3`. Measurement scripts do not count. `research-rag stats` and the workspace's Stats view report rank-one and top-five counts per source and passage, searches with no results, search time, how many searchable sources no search has reached (`stats` names them), corpus composition and missing metadata, and the last build's phase times. Each Stats card has its own time scope, and the ranked lists their own size. The board is a few full cards: the searches' headline figures, sources and passages by appearances, history, searches by day, the largest sources ranked by passages or by extracted text size, categories and authors, the corpus's size, missing metadata, and its formats, languages, and decades, and the last build. A passage table has the source and the place in it as separate columns. |
| Search history | While `runtime.search_history` is on, each search keeps its question and filters beside its counts. The Stats view and `research-rag history` list recent searches, and a link runs one again with its filters. `history --clear` forgets every question and leaves the counts. Turn the setting off to keep none. |
| Source passages and links | A source name opens that source's page: its facts, and its passages in reading order, twenty to a page, each with its size, whether it is excluded, and how often searches returned it. A passage's place, in a result or a table, opens the text around it. `research-rag chunks SOURCE` prints the same page. Every such view has an address, so Back, Forward, and reload return to it. |
| Open a source in the desktop viewer | The workspace's Open original hands the file to the desktop's own program, so nothing downloads. A machine with no desktop session shows the PDF in the browser instead. |
| OCR on request | `research-rag ocr SOURCE` writes a copy of a scanned PDF with the recognised text laid invisibly over each scanned page. It is separate from everything else: ingestion never runs it, and no agent, control route, or workspace action reaches it. It needs the optional `ocr` extra, writes outside the sources directory, never edits the original, and leaves moving the copy in to the reader. |
| Refuse a source that is not clean | When a source is extracted, more than `ingestion.maximum_unclean_percent` of its text, in characters, being unreadable refuses it before chunking. The default is 1. The message says a loss is expected and names `research-rag ocr`, which is never run for the reader. 100 accepts every source. |
| Build metrics and failure records | Phase timings, reuse counts, rejection counts, and truncation totals are recorded per generation. Non-resumable failures leave a small record. |

## What this app does not do

- **No answer generation.** It returns evidence with provenance. `ROADMAP.md` covers the deferred stage.
- **No quote-safe transcripts.** Text is cleaned for retrieval, so a quotation comes from the original.
- **No judgement of its own.** It never decides what is true and never excludes a source by itself. Duplicate and metadata decisions are reviewed and reversible.
- **No automatic OCR.** Ingestion never recognises text. An existing OCR text layer must pass the same health checks as other text, and PDFs with no readable text and password-protected PDFs are rejected. `research-rag ocr` makes a copy a reader chooses to bring in.
- **Conservative cleanup.** A PDF footnote needs a note marker, smaller type, and a bottom-page position outside detected tables. Unmarked notes, full-width sidebars, reference lists under unrecognised headings, bibliographies without strong author, locator, and citation signals, and garbled text without detectable corruption can remain. Synthetic tests and read-only collection checks do not establish removal precision or recall.
- **No external service.** No hosted embedding API, no vector database server, no credentials. Models download once and cache locally.
- **No cached freshness.** A directory signature cannot see a source replaced in place, so a cached verdict could call a changed corpus current. `MEASUREMENTS.md` describes the one switch that skips the check, which no reader-facing surface may set.
- **English by default.**
  - Extraction is language-neutral. The two language-dependent stages are settings.
  - `language.corpus` names the corpus languages, and `language.bm25_stopwords` chooses the single list BM25 filters.
  - `dense.embedding_model` chooses the model. The pinned table has a German-native model and a multilingual one.
  - Retrieval quality is not measured on another language. `status` reports a corpus language the chosen model does not cover.
