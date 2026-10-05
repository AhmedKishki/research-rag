---
name: FEATURES.md
description: Capability inventory: what comes from UltraRAG, what this app adds, what is excluded, what is planned.
---

# Features

- This document is the capability inventory: what the app can do and where each capability comes from.
- **UltraRAG** means the capability comes from the upstream framework this project pins, and this app uses it as it is.
- **Added here** means the research layer around UltraRAG.
- **Planned** means the capability is described in `ROADMAP.md` or `TODO.md` and is not built yet.
- `MEASUREMENTS.md` carries the protocol behind the performance-related features and the decisions that protocol once justified.
- `README.md` carries how to use them.
- Where UltraRAG offers a capability this app does not use, **Reuse decisions** records the reason.

## What UltraRAG provides, and what this app uses

- The UltraRAG gateway exposes around 78 tools and 26 prompts across corpus, retrieval, reranking, prompt, generation, routing, memory, benchmark, and evaluation components, and ships several vector-database backends and an upstream web interface.
- This app uses three upstream capabilities and owns everything else itself:

| UltraRAG capability | Used here? | What this app does with it |
|---|---|---|
| GPT-2 token chunking (`corpus_chunk_documents`) | Yes | Splits extraction units into token chunks with a configured size and overlap. Several units share one call for speed; each still gets its own durable output. |
| BM25 lexical index and search (`retriever_retriever_init`, `retriever_bm25_index`, `retriever_bm25_search`) | Yes | Supplies the lexical half of retrieval, in English, on CPU. |
| FAISS, Qdrant and Milvus dense index backends (`retriever_init(index_backend=...)`) | No | The dense path is built here instead: FastEmbed CPU embeddings plus either an exact scan of the generation's portable vectors or an embedded Qdrant collection created with the same pinned `qdrant-client` library, chosen per generation and recorded in its manifest. See **Reuse decisions**. |
| Reranking components (`reranker_init`, `reranker_rerank`) | No | A FastEmbed CPU cross-encoder reorders at most 50 candidates and keeps its scores. See **Reuse decisions**. |
| Prompt assembly and answer generation | No | The reader writes the answer. The app returns evidence, not prose. See **Capabilities left out instead of replaced**, and `ROADMAP.md` under **Answers** for the deferred stage. |
| Routing, memory, benchmark, evaluation components | No | Not part of this server's contract. Retrieval-quality evaluation is planned as its own work. See **Planned additions** and **Capabilities left out instead of replaced**. |
| Web-search retrieval | No | Out of scope: this server answers from a project's own documents only. See **Capabilities left out instead of replaced**. |
| Upstream web interface | No | This app ships a browser workspace for its own fourteen operations. See **The upstream web interface**. |

- The chunker and BM25 are upstream; the parameters and the surrounding policy are not: chunk size, overlap, which units are chunked, how results are filtered, and what is disclosed about rejected text.
- An UltraRAG upgrade is evaluated against three well-understood call sites instead of dozens.
- This app never depends on an upstream service, credential, GPU, or vector database being available.
- Every capability marked No there is either replaced by a component whose output the research contract can use, or left out deliberately.

### Reuse decisions

- An upstream component is reused whenever it can satisfy the research contract without adding a dependency or an operational requirement this project refuses to take.
- The research contract needs:
  - a stable `chunk_id`, `document_id`, and `source_id` for every hit.
  - a locator into the original file.
  - query-time metadata filtering resolved against the reviewed overlay.
  - a visible ranking signal.
  - CPU-only, offline-capable, credential-free operation with no external service required.
- A component is replaced only when at least one of those cannot be met.
- Any decision below would be revisited if the upstream component gained all five of the following.
  - It returns identifiers and a score rather than only anonymous passage text.
  - It returns a locator into the original file.
  - It filters by metadata at query time, using values the server can verify against canonical records rather than copies baked into an index.
  - It runs on CPU, offline, pinned by revision, with no server, GPU, or API credential.
  - It can be validated as one index of an immutable generation, so a failed index can never be selected.

**FAISS (`retriever_init(index_backend="faiss")`, `servers/retriever/src/index_backends/faiss_backend.py`).**

- **Benefit if reused:** a mature, in-process CPU index with no server, less custom code to maintain, and alignment with the backend upstream's own Vanilla RAG pipeline uses.
- **Why not:**
  - its `search()` returns `List[List[str]]` built from `self.contents[doc_id]`, so chunk identity and scores are discarded and no metadata filter exists.
  - reviewed category or keyword filters could only be applied by copying mutable metadata into the index or by dropping hits after the fact.
  - the index is built from the corpus during `retriever_init` and rebuilt over the whole corpus, at a per-point cost the exact scan this server uses instead does not pay.
- **What replaced it:**
  - an exact cosine scan of the generation's portable float32 vectors, which ranks a query the way a brute-force scan of those vectors does.
  - rows are excluded by document ID during the scan, so an exclusion or a reviewed metadata change takes effect without rebuilding anything.
- **Threshold:** the exact scan is the default, and above 200,000 chunks the embedded Qdrant backend is preferred, because it returns scored hits under a payload filter, which the upstream FAISS component cannot do.

**Qdrant (`retriever_init(index_backend="qdrant")`, `servers/retriever/src/index_backends/qdrant_backend.py`).**

- **Benefit if reused:** the payload filtering and scored results this server wants, delivered by the same `qdrant-client` library it already depends on.
- **Why not:**
  - the upstream component's `search()` returns only the payload text field (`str((hit.payload or {}).get(self.text_field, ""))`) and discards ids and scores.
  - it assigns `uuid5` point IDs derived from values instead of canonical chunk IDs.
  - it creates its collection at init.
- **Benefit of the reuse that does happen:**
  - this server's own thin backend stores integer point IDs with `chunk_id`, `document_id`, and `source_id` as payload.
  - that backend applies Qdrant's `MatchAny` payload filter to the document IDs the service already resolved from the reviewed overlay.
  - that backend returns `point.score` as a ranking signal.
- **Threshold:** the backend is selected per generation only above 200,000 chunks and recorded in that generation's manifest; below that, the exact scan is both simpler and faster.

**Milvus (`retriever_init(index_backend="milvus")`, `servers/retriever/src/index_backends/milvus_backend.py`).**

- **Benefit if reused:** a server-grade, horizontally scalable vector database with multi-client access, which is useful for a shared collection far larger than this project targets.
- **Why not:**
  - it needs a running Milvus service, or Milvus Lite, alongside the process.
  - that requirement conflicts with portable project-local derived state, the no-external-service rule, and the design envelope `MEASUREMENTS.md` states.
  - its `search()` also returns passage strings only.
  - upstream logs a warning that using Milvus outside demo mode is not recommended in its simplified architecture, and its demo mode forces OpenAI embeddings with Milvus, which adds a network credential on top.
- **Why the workload does not need it:** at the sizes this app is built for the exact scan answers a query from vectors the generation already stores, so a vector database would add an operational dependency without a capability.

**Reranking components (`reranker_init`, `reranker_rerank`, `servers/reranker`).**

- **Benefit if reused:** upstream's reranker model catalogue, including `openbmb/MiniCPM-Reranker-Light` as its shipped default, and consistency with the upstream pipeline.
- **Why not:**
  - the `sentence_transformers` backend pulls PyTorch into a package that otherwise needs only ONNX Runtime.
  - the `infinity` backend is a model-serving engine.
  - the `openai` backend needs a network credential and would send research queries to a third party.
  - upstream's shipped parameters target `device: cuda`.
  - the upstream result is `rerank_psg`, which is reordered passage strings with the scores discarded.
- **Why scores matter here:** the server reorders at most 50 candidates by score and then appends the unreranked candidate tail, so a reference group can still reach a source whose best passage fell outside the reranked prefix, and a string-only reranker cannot express that ordering.
- **Why the gateway cannot be asked for it:** `reranker` is one of the stateful namespaces the vanilla gateway can start, but the research transport requests `corpus` and `retriever` only, so no reranker child process runs and adding one would add a second model-serving surface.
- **What replaced it:**
  - a FastEmbed cross-encoder through the already-pinned FastEmbed ONNX dependency and the same shared model cache, lazily loaded, applied to at most 50 candidates, and chosen by the engine rather than by the caller.
  - six of FastEmbed's registered cross-encoders are supported, each pinned to a revision in `rerankers.py`, with `Xenova/ms-marco-MiniLM-L-6-v2` as the default.
  - a name outside that table is refused instead of being resolved to whatever the model hub serves that day.
- **Why the gate, and its default:** `MEASUREMENTS.md` records that reranked hybrid is the largest quality gain the judged set showed, and holds the command that compares the supported models over the same judged queries. Reranking is the server's fixed behavior, with `rerank=false` reachable only from the engine and a disclosed fallback to the unranked order when its model cannot be loaded.

### Capabilities left out instead of replaced

- These upstream components have no custom replacement: adopting them would change what the app is rather than how it is built.
- **Prompt assembly and answer generation.**
  - Adopting upstream's prompt and generation components would add a model-serving or API dependency and move prose generation into the app.
  - The contract instead has the reader write the answer and cite returned evidence, so `search` returns structured passages with provenance rather than an answer string.
  - Adding generation later, through a local model or a hosted API, is recorded in `ROADMAP.md` under **Answers**.
- **Routing, memory, benchmark, and evaluation.**
  - These serve multi-corpus pipelines, conversational memory, and benchmark scoring with boxed answers.
  - This app is one project, one corpus, and one immutable generation, and retrieval quality is measured offline against a judged query set rather than by an upstream evaluation component.
- **Web-search retrieval.**
  - It would send a researcher's query to a third-party provider and mix outside text into evidence that must be traceable to a project source, so retrieval stays inside the project's own documents.

### The upstream web interface

- The upstream interface is bound to upstream pipeline and session state, and it exposes operations this app deliberately does not.
- The bundled workspace is a dependency-free browser interface in `surfaces/workspace/`, owned by this app.
- This app owns its adapter, its source authorization, and the process that hosts it.
- The workspace is bound to loopback and may serve only an allowlisted original PDF or EPUB.
- Reusing it would mean exposing the upstream surface through this app or maintaining a second, divergent view of the same project.

## Features added on top of UltraRAG

### Ingestion and text quality

| Feature | What it does | Why it matters |
|---|---|---|
| PDF and EPUB ingestion | Walks the source directory recursively and accepts regular `.pdf` and `.epub` files only. | Other formats are ignored on purpose, so a stray `notes.md` never enters the knowledge base. |
| Layout-aware extraction | Reads PDFs page by page and EPUBs by spine section, keeping structure and discarding repeated page furniture. | Reading order and paragraphs survive; running headers do not become fake content. |
| Original-file locators | Every unit and chunk points back to a PDF page (with the printed page label when available) or an EPUB section. | You can open the source and check the passage. Locators are navigation aids, not quote offsets. |
| Bibliographic metadata with provenance | Resolves title, authors, year, DOI, and language per document, recording where each value came from and any warnings. Language detection reads the source's own function words against the stopword lists BM25 knows, and answers nothing when a sample is too short or no list is covered. | Automatic metadata is provisional: an answer carries only the authors it resolved, while the full-detail payload carries the citation that uses the metadata, the per-field provenance, and the warnings that flag a value needing review. |
| Reviewed metadata overlay | Corrections saved outside the generation apply at the next read to listings, filters, citations, search results, and neighbouring passages. The file is plain JSON and hand-editable at any time; `research-rag metadata`, and the workspace dialog built on it, write the same file one source at a time, so a hand edit and a save cannot clobber each other. | Fixing a wrong author needs no re-ingest, rewrites no immutable data, and is one command or one text edit. |
| Corrupt-text rejection with disclosure | Rejects a page or section only when its text shows strong evidence of a broken character map, then reports reason codes, counts, and example chunk IDs. | Bad text leaves the index, and you are told what was removed rather than silently losing material. |
| Script mixing never withholds | A quotation in Greek, Cyrillic, or any other script stays retrievable and its text is returned unchanged; the advisory script note stays in the full-detail payload. | — |
| Targeted normalisation folding | Formula-font letters (`𝑀` becomes `M`) and the presentation ligatures `ﬁ`, `ﬂ`, and `ﬀ` fold to plain spellings. Accents, superscripts, subscripts, and symbols are left alone. | A typed query matches printed text, without destroying notation that carries meaning. |
| Symbol-only chunk exclusion | Nonempty chunks containing no letters or digits are dropped. | Numbers and formulas are unaffected. |
| Embedding-token audit | Every chunk records its embedding token count and whether its vector covers only a prefix. | Silent truncation becomes visible per hit and in aggregate instead of quietly degrading dense search. |
| Contextual chunk headers | `chunking.headers` prepends the source's title and section to the text a chunk is embedded from, never to the text a search returns. Off by default, and a rebuild rather than a live switch, because it is an identity setting. | A passage cannot state which work and which section it came from, and that is exactly what a question about a work or a chapter asks for; the returned text stays quotable as it stands. |
| Compatible-material reuse | Unchanged documents keep their extracted units and chunks; identical chunk text keeps its vector. | Adding one source costs a fraction of a rebuild. |
| Resumable ingestion | Long builds are checkpointed; a call that exceeds its soft time budget returns `in_progress` and you repeat it. Cancellation, timeout, and restart lose at most one bounded batch. | A rebuild of a large collection need not succeed in one sitting. |
| Explicit regeneration | `force_recompute` bypasses all reuse while still resuming its own checkpoint. | — |

### Retrieval

| Feature | What it does | Why it matters |
|---|---|---|
| BM25 lexical search (UltraRAG) | Matches the words you typed. | Exact names, terms, and phrases. Also the honest baseline when a semantic result surprises you. |
| Dense semantic search (added here) | FastEmbed `bge-small-en-v1.5` embeddings, 384 dimensions, CPU, revision-pinned. | Finds passages phrased differently from your question. |
| Hybrid search (UltraRAG plus added here) | Weighted reciprocal-rank fusion of the two rankings, with an opt-out to inspect either signal alone. | The default that behaves well on real questions: `MEASUREMENTS.md` records it ordering better than BM25 alone at the same reach, so choosing BM25 for its speed gives up ordering rather than coverage. |
| Exact dense scan by default (added here) | Dense search scans the generation's portable float32 vectors directly; an embedded index is used above a documented corpus size. The manifest records which backend built the generation. | Nothing to build or keep in sync at normal sizes, and results are exactly reproducible. |
| Metadata filters (added here) | Narrow by project, category, keyword, or document, and see the project and category inventories in `status`. | — |
| Bibliographic filters by author and title (added here) | `authors_any` and `titles_any` keep results whose source carries one of the listed names, matched as case-insensitive substrings of the reviewed value, and `applied_filters` reports a filter that matched nothing. | Reaches the passages of one known work or one author inside a large collection: `"crawford"` finds `Kate Crawford` and `"atlas of ai"` finds the title without its subtitle, so a surname or a remembered fragment is enough. |
| Source selection (added here) | `source_ids` includes and `exclude_source_ids` removes named sources by their stable ID; both default to empty, which means include everything and exclude nothing. Filters apply before ranking, so `top_k` is a budget inside the selection. | A query can be confined to, or kept away from, specific works without a second corpus or a new generation. |
| Corpus partitions by branch (added here) | `categories_any` requires at least one of the listed branches, resolved from reviewed metadata at query time; `status.categories` reports each branch with its searchable source count. | One corpus can be searched as several parts — branches, strands, sub-projects — and the partitioning is reviewable metadata, not an index-time decision. |
| Project layer (added here) | `project` records which project a source was gathered for; `projects_any` requires at least one of the listed tags, and `status.projects` reports each tag with its searchable source count. | A corpus stays self-describing when it is shared between projects, and a search can be confined to the sources gathered for one project. |
| Per-source language, filterable (added here) | Every source carries the language extraction detected and the language a review set instead, and `languages_any` keeps results written in any of the listed ISO 639 codes. `status.languages` inventories them with each language's searchable source count. | A corpus in more than one language can be partitioned and searched per language, without pretending BM25 can filter two stopword lists at once. |
| Reranking, always on (added here) | A CPU cross-encoder reorders up to 50 fused candidates; when its pinned model cannot be loaded the search returns the unranked order and reports `rerank_fallback`. | The largest quality gain the judged set showed, and a missing model degrades instead of failing. `MEASUREMENTS.md` records the decision and what it costs per query. |
| Source diversity in the final selection (added here) | `retrieval.source_diversity_penalty` charges a candidate a share of its normalized relevance for every candidate already taken from the same source, applied at the final `top_k` pick over candidates fusion and reranking already ranked. It only reorders, never adds or drops a candidate; a bare BM25 or dense ranking has no score to charge against and keeps its own order. | A one-source answer is the case the charge exists for, and the cost is measured in rank: `MEASUREMENTS.md` records the mechanism and the trade. |
| Retrieval quality harness and its findings | A judged query set and a harness that measures each retrieval mode through the engine the tool calls, beside a second harness that drives index builds, durability writes, embedding, and the query gate against the real gateway, models, and service. `MEASUREMENTS.md` records both commands and what the runs decided. | "Is hybrid better than BM25 here?" is answered by measurement rather than by assertion: reranking gains most, dense alone is weakest, and paraphrase queries defeat every mode. |
| Honest evaluation limits (added here) | The judged set is known-item and single-annotator, so a passage that makes the same point is scored as a miss. | — |
| Pseudo-relevance feedback (added here) | `retrieval.prf` mines terms from the lexical leaders of a first pass and searches again with them, weighting each term by how rare it is across the generation so a question asked in other words can reach the passages that use the author's. Off by default. | It is the one lever aimed at the paraphrase gap rather than at candidate depth, and the judged set cannot measure it, so it stays off until the judgments are pooled. |
| Relevance gates with abstention (added here) | Weak candidates are dropped, so a search can legitimately return fewer results than `top_k`, including none. A dense candidate below the cosine floor is still admitted when the query's best candidate cleared the floor and this one is within `retrieval.dense_relative_similarity_margin` of it, which is how a one-word query keeps its band. The full-detail payload reports each gate's rejected count, the query's best similarity, and the sources the floor still dropped. | An honest empty answer beats a confident irrelevant one, and a candidate band a few hundredths under the floor is not evidence of irrelevance. |
| Minimum passage length (added here) | Two floors, both applied before fusion in each retrieval half: `retrieval.minimum_passage_words` counts words, and `retrieval.minimum_passage_token_fraction` counts tokens as a fraction of the generation's recorded `chunking.size`, using the tokenizer the generation was chunked with. Chunks never span extraction units, so a short unit becomes a short chunk that matches a query about its own words: an index line, a heading, a caption, a copyright line. Off by default. The full-detail payload reports both rules, the counts per rule, and the sources dropped. | A book's index is a list of the corpus's own words and is not evidence; `MEASUREMENTS.md` records why both floors ship off and what a corpus of table rows would lose. |
| Precomputed usability verdict (added here) | The decision "is this candidate structurally unusable?" is computed once when the lookup is built and stored per chunk. | The verdict is stored rather than recomputed so a rejection decision and the counter that reports it cannot drift; `MEASUREMENTS.md` states the mechanism. |

### Project model, review, and durability

| Feature | What it does | Why it matters |
|---|---|---|
| One project per process | A `--project-root` defines the boundary; all state lives under `<project>/.research-rag`. | A workspace can never serve a project it was not started for. |
| Many projects under one install (added here) | `init` records each project in an account-wide register of ids, names, and roots; `projects` lists them with the app state of each, and `--project <name-or-id>` selects one in place of a path. | One install serves a body of work spread over several directories, without each project needing its own configuration, and without one project's state ever reaching another. |
| Portable versus derived state | Reviewed decisions are portable; generations, staging, logs, and locks are derived and rebuildable. `STORAGE.md` has the layout. | You can back up what a human decided. |
| Reviewed inclusions and exclusions | A source can be excluded as a duplicate and later restored. The original file is never deleted or modified. | — |
| Passage exclusions (added here) | One passage can be excluded by the `chunk_id` a search returned and later restored, beside the whole-source decision, and the same filter applies it to both halves of a search. The decision holds at once, no ingestion is needed, and the original file is never changed. | A reader who objects to one passage rather than to the work it came from does not have to lose the rest of the argument, and the decision stays reversible and hand-editable. |
| Immutable generations | Each build produces a new generation. The active pointer moves only after both indexes validate. | — |
| Relocatable derived state | `--runtime-root` puts indexes, staging, and logs on a chosen disk, claimed by a marker naming its owning project. | A project on a slow disk keeps its working files on a fast one, with no OS-level bind mount. |
| Copy-based project portability | A relocated runtime root is claimed by a marker naming its owning project, so a copy cannot silently share it. | A project can be moved, shared, or archived with ordinary file tools and no export format to version. |
| Editable project settings (added here) | The workspace and `research-rag config` read every effective value with what the key does, the layer it came from, and what a change to it costs, and a save in the browser writes this project's own `config.toml`, refusing a value a higher layer supplied and naming which keys force a rebuild before anything is written. Keys the writer did not change are preserved. | Tuning needs no editor, and a change that invalidates a generation says so before it is saved rather than in the next answer. |

### Installation and updates

| Feature | What it does | Why it matters |
|---|---|---|
| Command on the account's `PATH` (added here) | `research-rag install` links the running interpreter's console script into the account's own binary directory, reports what it wrote, is a no-op on a second run, and `--uninstall` removes only that link. It needs no project and writes no project state. | Without it every other command needs a path or a wrapper to be usable. |
| Desktop menu entry per project (added here) | `research-rag install --desktop` writes one freedesktop entry per project that starts that project's own launcher with `--open`. `--uninstall` removes it, `--force` is required to replace an entry this app did not write, and `doctor` reports an entry whose project root no longer exists. | A project is opened by clicking its name rather than by remembering a command, the launcher keeps the port, pid, and group stop, and the entry needs nothing on a desktop session's `PATH`. |

### Interfaces and operational transparency

| Feature | What it does | Why it matters |
|---|---|---|
| One running app (added here) | `research-rag start` brings up one process per project that owns the project lock and the UltraRAG gateway and serves, on one loopback port, the browser workspace, the agent surface at `/mcp`, and the control API the command line speaks. The gateway opens on the first call that needs it, so an app asked only whether it is healthy never starts one. | One project has one index, one lock, and one set of reviews, so an agent, a browser, and a script cannot see two different states. |
| Workspace attached to the terminal (added here) | A bare `research-rag` serves from its terminal and asks which project when several are registered. `--start-ui` opens a browser. Ctrl-C or closing the terminal stops the app and gateway; an already running app is reported rather than duplicated. The pid, port, and terminal records let `projects` identify its owner. | Serving has an explicit terminal owner. |
| Client registry and forced disconnect (added here) | The app records every MCP client that connects, by the name it declares and by the program, directory, and process the stdio bridge inherited; one agent is one row however many sessions it opened, a client that named nothing is shown as the kind of connection it opened rather than as a session id, a client silent for a quarter of an hour is forgotten rather than listed, and `research-rag disconnect <session>` refuses that client before its requests reach the tools, which ends it. | An agent attached to a research app can be seen, identified, and ended rather than being an invisible process holding a project open. |
| Stdio bridge (added here) | `research-rag mcp --project-name NAME` proxies an already running app; it never starts one. It takes no project path. An unregistered project exposes only `status`, with an `init` remedy. A registered but stopped app reports the start command. | Stdio clients use the same agent surface without starting another server. |
| Core command line (added here) | One command per job over one process: a command that touches the corpus reaches the running app over the control API, and a project with no app up is answered in process. | A knowledge base can be created, built, searched, reviewed, browsed, and stopped from a script. |
| Dependency report and doctor (added here) | `status` reports `blocked_by` and `degraded`, each a list of `{check, reason, remedy}`, and `research-rag doctor` prints the same checks one per line, reads without writing anything, and can repair a mismatched UltraRAG runtime or fetch the pinned models on request. | A failure is named with the command that fixes it instead of arriving later as a connection error or an empty search, and a reader can ask what is wrong without opening a log. |
| Browser workspace | A loopback-only workspace over the same fourteen operations and the same project state, with a tab per job, including per-query source include/exclude, category-partition filters, a partition list with searchable counts, and this project's settings. | — |
| Projects selector and client entry (added here) | The workspace names the project it is serving and lists the account's others with whether an app is up; the MCP tab shows the agent endpoint and a copyable client entry, and folds the attached clients behind one click with the count on the summary. | One installation serving several projects says which one is on screen, and a reader can connect an agent without finding the entry by hand. |
| Generated per-project workspace launcher (added here) | Initialising a project writes a launcher and links it into the project root, so the workspace starts without a command, and `--stop` ends the whole process group. An existing file or symlink is never overwritten, and `status.ui_launcher` reports the state. `README.md` has the commands. | Starting the workspace for a project is one command with no flags to remember, and the launcher names carry this app's so two products serving one project cannot stop each other's process. |
| Version reporting (added here) | `research-rag --version` prints the version the running process started with, the version installed in the environment now, and a `restart_required` flag; `status.version` carries the same three numbers, and the workspace shows them in its header. The version is read from the installed distribution's metadata, which is what `pyproject.toml` becomes, so the number the app prints and the number an update compares are the same number. | — |
| One-command update (added here) | `research-rag update` compares this installation with the latest published release, read from the remote's tags with `git ls-remote` so no token and no API key are needed, and says whether a checkout is at that release, behind it, or ahead of it. `--apply` refuses a held project lock and a tree with uncommitted work, naming the files, then stops every app the installation serves, re-syncs the environment, and prints the command that starts each app again. | Updating compares against something a reader can install rather than against a branch head, and a repository with no release answers honestly instead of implying one exists. |
| CPU niceness inherited by children (added here) | `runtime.nice` raises this process's CPU niceness once at start, and every child inherits it, so the vanilla gateway and every model thread below it yield together. `0` by default, so nothing changes until it is asked for. | A long build saturates the machine you are working on, and priority costs no throughput. |
| Staleness, upgrade, and retained-generation reporting | `status` says whether the selected generation is stale, and why, whether a policy upgrade is required, and what the retained generations occupy on disk as a count and a total; `generations` lists each one, including any whose manifest is unreadable, and `--use` validates one and moves the pointer to it; `remove-generation` deletes one that search does not read, after the id is repeated. | You know when a search is answering from older material, and you can see what the retained generations cost without a shell command, roll back to a previous build, and reclaim a build's space without a shell command either, while a routine status answer stays a statement about the generation you are searching. |
| Build metrics and failure records | Phase timings, reuse counts, rejection counts, and truncation totals are recorded per generation; non-resumable failures leave a small record. | — |

## What this app deliberately does not do

- These are choices rather than missing pieces, and each one would change what the app is.
- **No answer generation.** It returns evidence candidates with provenance, and the reader interprets them. A generation stage through a local model or a hosted API is deferred work, recorded in `ROADMAP.md` under **Answers**.
- **No quote-safe transcripts.** Text comes back cleaned for retrieval, so a quotation is taken from the original.
- **No judgement of its own.** It does not decide what is true, and it never deletes or excludes a source on its own: duplicate and metadata decisions are reviewed and reversible.
- **No OCR.** Scanned PDFs need OCR first, and password-protected PDFs are rejected.
- **No external service.** No hosted embedding API, no vector database server, no credentials. Models download once and cache locally.
- **No cached freshness.** A directory signature cannot see a source replaced in place, so a cached verdict could call a changed corpus current. `MEASUREMENTS.md` carries the mechanism and the one switch that skips the check, which no reader-facing surface may set.
- **English by default rather than by design.**
  - Extraction is language-neutral, and the two stages that depend on language are settings.
  - `language.corpus` names the corpus languages, one or several for a mixed corpus, with `language.bm25_stopwords` choosing the single list BM25 filters.
  - `dense.embedding_model` chooses the embedding model, with a German-native model and a multilingual one that covers a mixed corpus in the pinned table.
  - The default stays English, retrieval quality has not been measured on another language yet, and a corpus language the chosen model does not cover is reported by `status`.

## Planned additions

- `TODO.md` lists the open work inside this repository's scope.
- `ROADMAP.md` lists product ideas that are not in scope yet.
- Nothing in either file is claimed as present.

## Comparison

- A comparison with a general-purpose RAG server, and the guidance for choosing between them, is in this collection's `README.md`.
