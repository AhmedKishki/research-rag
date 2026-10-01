# Features

What the app can do, and where each capability comes from. Some of it is UltraRAG, some is built on top of UltraRAG, and some is only planned. It also compares this app with a different, more general RAG server, because "which one should I use?" depends on the job.

Three labels are used throughout:

- **UltraRAG** — the capability comes from the upstream framework this project pins, and this app uses it as it is.
- **Added here** — the research layer around UltraRAG. This is where most of the product lives.
- **Planned** — described in `ROADMAP.md` or `TODO.md`, not built yet. Nothing in this document claims otherwise.

For measurements behind the performance-related features, see `MEASUREMENTS.md`. For how to use them, see `README.md`. Where UltraRAG offers a capability this app does not use, section 1.1 records the reason and what the upstream component would have added: an unused upstream feature is a documented decision, not an oversight.

## 1. What UltraRAG provides, and what this app uses

UltraRAG is much larger than the part this app needs. Its gateway exposes around 78 tools and 26 prompts across corpus, retrieval, reranking, prompt, generation, routing, memory, benchmark, and evaluation components, with several vector-database backends and an upstream web interface.

This app uses three upstream capabilities and owns everything else itself:

| UltraRAG capability | Used here? | What this app does with it |
|---|---|---|
| GPT-2 token chunking (`corpus_chunk_documents`) | Yes | Splits extraction units into token chunks with a configured size and overlap. Several units share one call for speed; each still gets its own durable output. |
| BM25 lexical index and search (`retriever_retriever_init`, `retriever_bm25_index`, `retriever_bm25_search`) | Yes | Supplies the lexical half of retrieval, in English, on CPU. |
| FAISS, Qdrant and Milvus dense index backends (`retriever_init(index_backend=...)`) | No | The dense path is built here instead: FastEmbed CPU embeddings plus either an exact scan of the generation's portable vectors or an embedded Qdrant collection created with the same pinned `qdrant-client` library, chosen per generation and recorded in its manifest. See section 1.1. |
| Reranking components (`reranker_init`, `reranker_rerank`) | No | A FastEmbed CPU cross-encoder reorders at most 50 candidates and keeps its scores. See section 1.1. |
| Prompt assembly and answer generation | No | The reader writes the answer. The app returns evidence, not prose. See section 1.2, and `ROADMAP.md` under **Answers** for the deferred stage. |
| Routing, memory, benchmark, evaluation components | No | Not part of this server's contract. Retrieval-quality evaluation is planned as its own work (see section 4 and section 1.2). |
| Web-search retrieval | No | Out of scope: this server answers from a project's own documents only. See section 1.2. |
| Upstream web interface | No | This app ships a browser workspace for its own seven operations. See section 1.3. |

Three consequences follow.

First, **the chunker and BM25 are upstream; the parameters and the surrounding policy are not**: chunk size, overlap, which units are chunked, how results are filtered, and what is disclosed about rejected text are all decided here.

Second, **pinning a small upstream surface is a feature, not a gap**. An UltraRAG upgrade can then be evaluated against three well-understood call sites instead of dozens, and this app never depends on an upstream service, credential, GPU, or vector database being available.

Third, **every capability marked No in the table above has a recorded reason**. Each is either replaced by a component whose output the research contract can use (section 1.1) or left out deliberately (sections 1.2 and 1.3).

### 1.1 Reuse decisions: when this app builds a component instead of reusing one

The rule is to reuse an upstream component whenever it can satisfy the research contract without adding a dependency or an operational requirement this project refuses to take. That contract needs, for every hit, a stable `chunk_id`, `document_id`, and `source_id`; a locator into the original file; query-time metadata filtering resolved against the reviewed overlay; a visible ranking signal; and CPU-only, offline-capable, credential-free operation with no external service required. A component is replaced only when at least one of those cannot be met.

Any decision below would be revisited if the upstream component gained all four of the following.

- It returns identifiers and a score, not only anonymous passage text.
- It filters by metadata at query time, using values the server can verify against canonical records rather than copies baked into an index.
- It runs on CPU, offline, pinned by revision, with no server, GPU, or API credential.
- It can be validated as one index of an immutable generation, so a failed index can never be selected.

**FAISS (`retriever_init(index_backend="faiss")`, `servers/retriever/src/index_backends/faiss_backend.py`).**

- **Benefit if reused:** a mature, in-process CPU index with no server, less custom code to maintain, and alignment with the backend upstream's own Vanilla RAG pipeline uses.
- **Why not:** its `search()` returns `List[List[str]]` built from `self.contents[doc_id]`, so chunk identity and scores are discarded and no metadata filter exists, so reviewed category or keyword filters could only be applied by copying mutable metadata into the index or by dropping hits after the fact; the index is also built from the corpus during `retriever_init` and rebuilt over the whole corpus, which costs 3,040.73 s for the reference corpus on the project's HDD and 75.5× less on NVMe (`MEASUREMENTS.md`), while the exact scan this server uses instead costs 0.07 s of index work.
- **What replaced it:** an exact cosine scan of the generation's portable float32 vectors, whose descriptor builds in 0.03 s and 0.58 MB on the reference corpus and returns the same top 20 as a brute-force ranking; rows are excluded by document ID during the scan, so an exclusion or a reviewed metadata change takes effect without rebuilding anything.
- **Threshold:** the exact scan is the default; an ANN backend earns its build cost only above 200,000 chunks, and the embedded Qdrant backend is preferred there because it returns scored hits under a payload filter, which the upstream FAISS component cannot do.

**Qdrant (`retriever_init(index_backend="qdrant")`, `servers/retriever/src/index_backends/qdrant_backend.py`).**

- **Benefit if reused:** the payload filtering and scored results this server wants, delivered by the same `qdrant-client` library it already depends on.
- **Why not:** the upstream component's `search()` returns only the payload text field (`str((hit.payload or {}).get(self.text_field, ""))`) and discards ids and scores, assigns `uuid5` point IDs derived from values instead of canonical chunk IDs, and creates its collection at init. The library is reused; the component is not.
- **Benefit of the reuse that does happen:** this server's own thin backend stores integer point IDs with `chunk_id`, `document_id`, and `source_id` as payload, applies Qdrant's `MatchAny` payload filter to the document IDs the service already resolved from the reviewed overlay, and returns `point.score` as a ranking signal.
- **Threshold:** selected per generation only above 200,000 chunks and recorded in that generation's manifest; below that, the exact scan is both simpler and faster.

**Milvus (`retriever_init(index_backend="milvus")`, `servers/retriever/src/index_backends/milvus_backend.py`).**

- **Benefit if reused:** a server-grade, horizontally scalable vector database with multi-client access, which is genuinely useful for a shared collection far larger than this project targets.
- **Why not:** it needs a running Milvus service, or Milvus Lite, alongside the process, which conflicts with portable project-local derived state, the no-external-service rule, and the 5,000–50,000 chunk design envelope; its `search()` also returns passage strings only. Upstream logs a warning that using Milvus outside demo mode is not recommended in its simplified architecture, and its demo mode forces OpenAI embeddings with Milvus, which adds a network credential on top.
- **Why the workload does not need it:** at the reference corpus size the exact scan answers a query in tens of milliseconds from vectors the generation already stores, so a vector database would add an operational dependency without a capability.

**Reranking components (`reranker_init`, `reranker_rerank`, `servers/reranker`).**

- **Benefit if reused:** upstream's reranker model catalogue, including `openbmb/MiniCPM-Reranker-Light` as its shipped default, and consistency with the upstream pipeline.
- **Why not:** every upstream backend adds something this project refuses. The `sentence_transformers` backend pulls PyTorch into a package that otherwise needs only ONNX Runtime; the `infinity` backend is a model-serving engine; the `openai` backend needs a network credential and would send research queries to a third party; and upstream's shipped parameters target `device: cuda`. Its result is also `rerank_psg` — reordered passage strings with the scores discarded.
- **Why scores matter here:** the server reorders at most 50 candidates by score and then appends the unreranked candidate tail, so a reference group can still reach a source whose best passage fell outside the reranked prefix; a string-only reranker cannot express that ordering.
- **Why the gateway cannot be asked for it:** `reranker` is one of the stateful namespaces the vanilla gateway can start, but the research transport requests `corpus` and `retriever` only, so no reranker child process runs today and adding one would add a second model-serving surface.
- **What replaced it:** a FastEmbed cross-encoder through the already-pinned FastEmbed ONNX dependency and the same shared model cache, lazily loaded, applied to at most 50 candidates, and chosen by the engine rather than by the caller. Six of FastEmbed's registered cross-encoders are supported, each pinned to a revision in `rerankers.py`, with `Xenova/ms-marco-MiniLM-L-6-v2` as the default; a name outside that table is refused instead of being resolved to whatever the model hub serves that day.
- **Measured benefit of the custom component, and its default:** on the judged set, reranked hybrid reaches 81.2% first-position success against 65.6% unreranked and 59.4% for BM25, at 2.30 s per query against 0.17 s. It is the server's fixed behavior, with `rerank=false` reachable only from the engine (the harness and the tests) and a disclosed fallback to the unranked order when its model cannot be loaded. `MEASUREMENTS.md` records how the supported models compare on the same judged set.

The net position is narrower than it may look. These are not weaker technologies, and none of the decisions is permanent: the Qdrant library is already reused, the reranker is a pinned model whose revision is a one-line change, and an upstream component that returned identity and scores under a CPU-friendly backend would be adopted rather than rebuilt. What this server will not do is trade chunk identity, ranking signals, query-time filtering, or local-only operation for the sake of delegating to an upstream class.

### 1.2 Capabilities that are left out instead of replaced

These upstream components have no custom replacement, because adopting them would change what the server is rather than how it is built.

- **Prompt assembly and answer generation.** Adopting upstream's prompt and generation components would add a model-serving or API dependency and move prose generation into the app. The contract instead has the reader write the answer and cite returned evidence, which is why `search` returns structured passages with provenance rather than an answer string. Adding generation later, through a local model or a hosted API, is recorded in `ROADMAP.md` under **Answers**, with the citation contract and the disclosure it would need.
- **Routing, memory, benchmark, and evaluation.** These serve multi-corpus pipelines, conversational memory, and benchmark scoring with boxed answers. This app is one project, one corpus, and one immutable generation, and retrieval quality is measured offline against a judged query set rather than by an upstream evaluation component.
- **Web-search retrieval.** It would send a researcher's query to a third-party provider and mix outside text into evidence that must be traceable to a project source, so retrieval stays inside the project's own documents.

### 1.3 The upstream web interface

The upstream interface is bound to upstream pipeline and session state, and it exposes operations this app deliberately does not. The bundled workspace is a pinned, dependency-free browser interface owned by the separate `ui-ultra-rag-mcp` package; this app owns its adapter, its source authorization, and the process that hosts it. It is bound to loopback and may serve only an allowlisted original PDF or EPUB. Reusing the upstream interface would mean either exposing the upstream surface through this app or maintaining a second, divergent view of the same project.

## 2. Features added on top of UltraRAG

### Ingestion and text quality

| Feature | What it does | Why it matters |
|---|---|---|
| PDF and EPUB ingestion | Walks the source directory recursively and accepts regular `.pdf` and `.epub` files only. | Other formats are ignored on purpose, so a stray `notes.md` never enters the knowledge base. |
| Layout-aware extraction | Reads PDFs page by page and EPUBs by spine section, keeping structure and discarding repeated page furniture. | Reading order and paragraphs survive; running headers do not become fake content. |
| Original-file locators | Every unit and chunk points back to a PDF page (with the printed page label when available) or an EPUB section. | You can open the source and check the passage. Locators are navigation aids, not quote offsets. |
| Bibliographic metadata with provenance | Resolves title, authors, year, DOI, and language per document, recording where each value came from and any warnings. Language detection reads the source's own function words against the stopword lists BM25 knows, and answers nothing when a sample is too short or no list is covered. | Automatic metadata is provisional: an answer carries only the authors it resolved, while the full-detail payload carries the citation that uses the metadata, the per-field provenance, and the warnings that flag a value needing review. |
| Reviewed metadata overlay | Corrections saved outside the generation apply at the next read to listings, filters, citations, search results, and neighbouring passages. The file is plain JSON and hand-editable at any time; `research-rag metadata`, and the workspace dialog built on it, write the same file one source at a time, so a hand edit and a save cannot clobber each other. | Fixing a wrong author needs no re-ingest, rewrites no immutable data, and is one command or one text edit. |
| Corrupt-text rejection with disclosure | Rejects a page or section only when its text shows strong evidence of a broken character map, then reports reason codes, counts, and example chunk IDs. | Bad text leaves the index, and you are told what was removed rather than silently losing material. |
| Script mixing never withholds | A quotation in Greek, Cyrillic, or any other script stays retrievable and its text is returned unchanged; the advisory script note stays in the full-detail payload. | English-language scholarship quotes other languages; withholding those passages was a real defect that this fixes. |
| Targeted normalisation folding | Formula-font letters (`𝑀` becomes `M`) and the presentation ligatures `ﬁ`, `ﬂ`, and `ﬀ` fold to plain spellings. Accents, superscripts, subscripts, and symbols are left alone. | A typed query matches printed text, without destroying notation that carries meaning. |
| Symbol-only chunk exclusion | Nonempty chunks containing no letters or digits are dropped. | Extraction artefacts do not pollute results; numbers and formulas are unaffected. |
| Embedding-token audit | Every chunk records its embedding token count and whether its vector covers only a prefix. | Silent truncation becomes visible per hit and in aggregate instead of quietly degrading dense search. |
| Contextual chunk headers | `chunking.headers` prepends the source's title and section to the text a chunk is embedded from, never to the text a search returns. Off by default: a rebuild of the PDF reference corpus measured no change, so a project opts in. It is an identity setting, so it is a rebuild rather than a live switch. | A passage cannot state which work and which section it came from, and that is exactly what a question about a work or a chapter asks for; the returned text stays quotable as it stands. |
| Compatible-material reuse | Unchanged documents keep their extracted units and chunks; identical chunk text keeps its vector. | Adding one source costs a fraction of a rebuild. |
| Resumable ingestion | Long builds are checkpointed; a call that exceeds its soft time budget returns `in_progress` and you repeat it. Cancellation, timeout, and restart lose at most one bounded batch. | A rebuild of a large collection need not succeed in one sitting. |
| Explicit regeneration | `force_recompute` bypasses all reuse while still resuming its own checkpoint. | A rebuild from scratch is deliberate rather than accidental. |

### Retrieval

| Feature | What it does | Why it matters |
|---|---|---|
| BM25 lexical search (UltraRAG) | Matches the words you typed. | Exact names, terms, and phrases. Also the honest baseline when a semantic result surprises you. |
| Dense semantic search (added here) | FastEmbed `bge-small-en-v1.5` embeddings, 384 dimensions, CPU, revision-pinned. | Finds passages phrased differently from your question. |
| Hybrid search (UltraRAG plus added here) | Weighted reciprocal-rank fusion of the two rankings, with an opt-out to inspect either signal alone. | The default that behaves well on real questions: 65.6% first-position success on the judged set, against 59.4% for BM25 alone. |
| Exact dense scan by default (added here) | Dense search scans the generation's portable float32 vectors directly; an embedded index is used above a documented corpus size. The manifest records which backend built the generation. | Nothing to build or keep in sync at normal sizes, and results are exactly reproducible. |
| Metadata filters (added here) | Narrow by project, category, keyword, or document, and see the project and category inventories in `status`. | Keeps a search inside the part of the collection you care about. |
| Bibliographic filters by author and title (added here) | `authors_any` and `titles_any` keep results whose source carries one of the listed names, matched as case-insensitive substrings of the reviewed value, and `applied_filters` reports a filter that matched nothing. | Reaches the passages of one known work or one author inside a large collection: `"crawford"` finds `Kate Crawford` and `"atlas of ai"` finds the title without its subtitle, so a surname or a remembered fragment is enough. |
| Source selection (added here) | `source_ids` includes and `exclude_source_ids` removes named sources by their stable ID; both default to empty, which means include everything and exclude nothing. Filters apply before ranking, so `top_k` is a budget inside the selection. | A query can be confined to, or kept away from, specific works without a second corpus or a new generation. |
| Corpus partitions by branch (added here) | `categories_any` requires at least one of the listed branches, resolved from reviewed metadata at query time; `status.categories` reports each branch with its searchable source count. | One corpus can be searched as several parts — branches, strands, sub-projects — and the partitioning is reviewable metadata, not an index-time decision. |
| Project layer (added here) | `project` records which project a source was gathered for; `projects_any` requires at least one of the listed tags, and `status.projects` reports each tag with its searchable source count. | A corpus stays self-describing when it is exported as a bundle, imported elsewhere, or shared between projects, and a search can be confined to the sources gathered for one project. |
| Per-source language, filterable (added here) | Every source carries the language extraction detected and the language a review set instead, and `languages_any` keeps results written in any of the listed ISO 639 codes. `status.languages` inventories them with each language's searchable source count. | A corpus in more than one language can be partitioned and searched per language, without pretending BM25 can filter two stopword lists at once. |
| Reranking, always on (added here) | A CPU cross-encoder reorders up to 50 fused candidates; when its pinned model cannot be loaded the search returns the unranked order and reports `rerank_fallback`. | The largest measured quality gain: first-position success rises from 65.6% to 81.2% and document-level success from 90.6% to 93.8%, for about thirteen times the query latency of unreranked hybrid — and a missing model degrades instead of failing. |
| Source diversity in the final selection (added here) | `retrieval.source_diversity_penalty` charges a candidate a share of its normalized relevance for every candidate already taken from the same source, applied at the final `top_k` pick over candidates fusion and reranking already ranked. It only reorders, never adds or drops a candidate; a bare BM25 or dense ranking has no score to charge against and keeps its own order. | Four of thirty judged top-10 answers came from a single source with the charge off. The default 0.25 leaves no one-source answer and lifts the mean distinct sources from 3.9 to 7.4, while every success column, MRR, nDCG, and document success stay within one query's rank. |
| Measured retrieval quality (added here) | A 32-query judged set over 19 passages of a real corpus and a harness that measures each retrieval mode through the engine the tool calls, reported per mode and per query class. | "Is hybrid better than BM25 here?" is answered by measurement: reranking gains most, dense alone is weakest, and paraphrase queries defeat every mode. |
| Honest evaluation limits (added here) | The judged set is known-item and single-annotator, so a passage that makes the same point is scored as a miss and true recall is not claimed. | The numbers say what they do not cover instead of implying benchmark-grade precision. |
| Pseudo-relevance feedback (added here) | `retrieval.prf` mines terms from the lexical leaders of a first pass and searches again with them, weighting each term by how rare it is across the generation so a question asked in other words can reach the passages that use the author's. Off by default. | It is the one lever aimed at the paraphrase gap rather than at candidate depth, and the known-item judged set cannot measure it, so it stays off until the judgments are pooled. |
| Relevance gates with abstention (added here) | Weak candidates are dropped, so a search can legitimately return fewer results than `top_k`, including none. A dense candidate below the cosine floor is still admitted when the query's best candidate cleared the floor and this one is within `retrieval.dense_relative_similarity_margin` of it, which is how a one-word query keeps its band. The full-detail payload reports each gate's rejected count, the query's best similarity, and the sources the floor still dropped. | An honest empty answer beats a confident irrelevant one, and a candidate band a few hundredths under the floor is not evidence of irrelevance. |
| Minimum passage length (added here) | Two floors, both applied before fusion in each retrieval half: `retrieval.minimum_passage_words` counts words, and `retrieval.minimum_passage_token_fraction` counts tokens as a fraction of the generation's recorded `chunking.size`, using the tokenizer the generation was chunked with. Chunks never span extraction units, so a short unit becomes a short chunk that matches a query about its own words: an index line, a heading, a caption, a copyright line. The token floor says what a fragment is in the corpus's own unit, and a contextual header cannot enlarge it because only the returned text is counted. Off by default; the reference project sets 12 words and a tenth of the chunk size. The full-detail payload reports both rules, the counts per rule, and the sources dropped. | A book's index is a list of the corpus's own words and is not evidence: a fifth of the reference corpus's chunks are short fragments — 23.6% of them fall below a tenth of the chunk size — while holding 1.9% of its text, 16.6% are under 12 words, and 2.7% of the passages its searches returned were. Dropping them removed the five-word index line `value production chain, 70–71` that a one-word query had ranked first, and put nine sources in an answer that had shown eight including it. |
| Precomputed usability verdict (added here) | The decision "is this candidate structurally unusable?" is computed once when the lookup is built and stored per chunk. | The per-query gate measured 10.3× cheaper without changing a rejection decision. |

### Project model, review, and durability

| Feature | What it does | Why it matters |
|---|---|---|
| One project per process | A `--project-root` defines the boundary; all state lives under `<project>/.research-rag`. | Two projects cannot read each other's documents or indexes, and a workspace can never serve a project it was not started for. |
| Many projects under one install (added here) | `init` records each project in an account-wide register of ids, names, and roots; `projects` lists them with the app state of each, and `--project <name-or-id>` selects one in place of a path. | One install serves a body of work spread over several directories, without each project needing its own configuration, and without one project's state ever reaching another. |
| Portable versus derived state | Reviewed decisions are portable; generations, staging, logs, and locks are derived and rebuildable. `STORAGE.md` has the layout. | You can back up what a human decided, and regenerate the rest. |
| Reviewed inclusions and exclusions | A source can be excluded as a duplicate and later restored. The original file is never deleted or modified. | Review stays reversible, and the server never destroys evidence. |
| Immutable generations | Each build produces a new generation. The active pointer moves only after both indexes validate. | A failed or interrupted build leaves the previous generation searchable. |
| Relocatable derived state | `--runtime-root` puts indexes, staging, and logs on a chosen disk, claimed by a marker naming its owning project. | A project on a slow disk keeps its working files on a fast one, with no OS-level bind mount. |
| Copy-based project portability | A project is self-contained: copying `sources/` plus `.research-rag/` moves it, and `runtime/` rebuilds on the next ingestion. A relocated runtime root is claimed by a marker naming its owning project, so a copy cannot silently share it. | A project can be moved, shared, or archived with ordinary file tools and no export format to version. |

### Interfaces and operational transparency

| Feature | What it does | Why it matters |
|---|---|---|
| One running app (added here) | `research-rag start` brings up one process per project that owns the project lock and the UltraRAG gateway and serves, on one loopback port, the browser workspace, the agent surface at `/mcp`, and the control API the command line speaks. The gateway opens on the first call that needs it, so an app asked only whether it is healthy never starts one. | One project has one index, one lock, and one set of reviews, so an agent, a browser, and a script cannot see two different states, and whether it is up is the only question to ask. |
| Client registry and forced disconnect (added here) | The app records every MCP client that connects, by the name it declares, and `research-rag clients` lists them; `research-rag disconnect <session>` refuses that session before it reaches the tools, which ends it. | An agent attached to a research app can be seen and ended rather than being an invisible process holding a project open. |
| Stdio bridge (added here) | `research-rag mcp` makes sure the app is up and then proxies stdio to its agent endpoint, so a client that cannot open a socket gets the same seven operations. It re-declares nothing. | A stdio-only MCP client keeps working, and there is still only one server. |
| Core command line (added here) | `research-rag` with `init`, `status`, `ingest`, `search`, `sources`, `passage`, `include`, `exclude`, `metadata`, `config`, `doctor`, `start`, `ui`, `clients`, `disconnect`, `mcp`, `serve`, and `stop`. A command that touches the corpus reaches the running app over the control API; a project with no app up is answered in process. | A knowledge base can be created, built, searched, reviewed, browsed, and stopped from a script, and the terminal, the workspace, and the agent cannot disagree because they are the same service. |
| Dependency report and doctor (added here) | `status` reports `blocked_by` and `degraded`, each a list of `{check, reason, remedy}`, and `research-rag doctor` prints the same checks one per line, reads without writing anything, and can repair a mismatched UltraRAG runtime or fetch the pinned models on request. | A failure is named with the command that fixes it instead of arriving later as a connection error or an empty search, and a reader can ask what is wrong without opening a log. |
| Browser workspace | A loopback-only workspace over the same seven operations and the same project state, including per-query source include/exclude, category-partition filters, and a partition list with searchable counts. | You can inspect the knowledge base and narrow a search to the works or strands you are working on. |
| Generated per-project workspace launcher (added here) | Initialising a project writes a launcher and links it into the project root, so the workspace starts without a command. `README.md` has the commands. It starts the workspace with this project's own root, runtime root, and port, chooses that port while holding a lock, retries the next one when a start fails, records the pid and port only once its own process serves that port, and `--stop` ends the whole process group and sweeps any process of this app a long build left behind. An existing file or symlink is never overwritten, and `status.ui_launcher` reports the state. | Starting the workspace for a project is one command with no flags to remember, and the launcher names carry this app's so two products serving one project cannot stop each other's process. |
| Version reporting (added here) | `status.version` gives the version the running process started with, the version installed in the environment now, the pinned shared-workspace version, the checkout it came from, and a `restart_required` flag; the workspace shows both versions in its header. | Whether the process answering you is the code you just installed is otherwise invisible, which is exactly how an update appears not to have applied. |
| One-command update (added here) | `scripts/update.sh` pulls, syncs the environment, optionally restarts a named project's workspace, and reports the processes still running older code. | Updating stops being a remembered sequence, and the script says plainly that a running workspace must be restarted rather than pretending it can hot-reload. |
| Considerate scheduling (added here) | `runtime.nice` raises this process's CPU niceness once at start, and every child inherits it, so the vanilla gateway and every model thread below it yield together. `0` by default, so nothing changes until it is asked for. | A build that saturates eight cores for a quarter of an hour makes the machine you are working on unusable, and priority costs no throughput. |

| Staleness, upgrade, and retained-generation reporting | `status` says whether the selected generation is stale, and why, whether a policy upgrade is required, and what the retained generations occupy on disk as a count and a total; `generations` lists each one, including any whose manifest is unreadable, and `--use` validates one and moves the pointer to it; `remove-generation` deletes one that search does not read, after the id is repeated. | You know when a search is answering from older material, and you can see what the retained generations cost without a shell command, roll back to a previous build, and reclaim a build's space without a shell command either, while a routine status answer stays a statement about the generation you are searching. |
| Build metrics and failure records | Phase timings, reuse counts, rejection counts, and truncation totals are recorded per generation; non-resumable failures leave a small record. | Slow or surprising builds can be diagnosed without guessing. |
| Documented performance characteristics | Measured, reproducible numbers for index builds, durability writes, embedding, and the query gate, plus retrieval quality against a judged query set, and a benchmark script to reproduce them on your hardware. | Claims can be checked instead of trusted. |

## 3. What this app deliberately does not do

These are choices, not missing pieces. Each one would change what the app is:

- **It does not generate answers.** It returns evidence candidates with provenance; the reader interprets them. A generation stage through a local model or a hosted API is deferred work, recorded in `ROADMAP.md` under **Answers**.
- **It does not provide quote-safe transcripts.** Text comes back cleaned for retrieval, so a quotation is taken from the original.
- **It does not decide what is true, and never deletes or excludes sources on its own.** Duplicate and metadata decisions are reviewed and reversible.
- **It does not run OCR.** Scanned PDFs need OCR first; password-protected PDFs are rejected.
- **It is English by default, not by design.** Extraction is language-neutral, and the two stages that depend on language are settings: `language.corpus` names the corpus languages — one, or several for a mixed corpus, with `language.bm25_stopwords` choosing the single list BM25 filters — and `dense.embedding_model` chooses the embedding model, with a German-native model and a multilingual one that covers a mixed corpus in the pinned table. The default stays English, retrieval quality has not been measured on another language yet, and a corpus language the chosen model does not cover is reported by `status`.
- **It does not require an external service.** No hosted embedding API, no vector database server, no credentials. Models download once and cache locally.
- **It does not cache freshness.** A directory signature cannot see a source replaced in place, so a cached verdict could call a changed corpus current. The engine can skip the check for measurement runs; no reader-facing surface can, because a search that cannot say whether its corpus moved is answering about something unknown.

## 4. Planned additions

Deferred work is tracked in two places and not repeated here: `TODO.md` lists the open work inside this repository's scope, and `ROADMAP.md` lists product ideas that are not in scope yet. Nothing in either is claimed as present.

## 5. Comparison with a general-purpose RAG server

The comparison below is with [`mcp-rag-server`](https://github.com/kwanLeeFrmVi/mcp-rag-server), a general-purpose RAG server written in TypeScript. It is a different tool for a different job, and it is not part of this collection.

Two profiles rather than a scoreboard. It reflects that project's README at the time of writing; check its repository for the current state.

| Aspect | This app | `mcp-rag-server` |
|---|---|---|
| What it is for | Research evidence over scholarly PDF and EPUB collections, with provenance you can check | General RAG context for an LLM, over plain text files |
| Language and install | Python, `uv sync --frozen` from this repository | Node.js, `npm install -g mcp-rag-server` or `npx` |
| Document formats | Regular `.pdf` and `.epub` only | `.txt`, `.md`, `.json`, `.jsonl`, `.csv` |
| Chunking | UltraRAG GPT-2 token chunking, configurable size and overlap | Character count (`CHUNK_SIZE`, default 500) |
| Embeddings | FastEmbed CPU model pinned by revision, downloaded once, no service needed | An external embedding API (OpenAI, Ollama, Granite, or Nomic) over HTTP, Ollama by default |
| Retrieval | Hybrid with reranking, with metadata filters and relevance gates that can abstain | Dense nearest-chunk retrieval, top `k` default 15 |
| Dense store | The generation's portable vectors scanned exactly, or an embedded index above a size threshold | Local SQLite vector store (LangChain `LibSQLVectorStore`) |
| Metadata, citations, locators | Resolved title, authors, year, DOI with per-field provenance and warnings in the full-detail payload, plus original-file locators and a reviewed-metadata overlay | Not part of the documented feature set |
| Index lifecycle | Immutable generations, validated before activation, resumable builds, reusable per-document and per-chunk work | Sequential indexing with progress reporting, plus per-document and whole-index removal |
| Cancellation and restart behaviour | Checkpointed: a build resumes where it stopped | Progress is reported; resumability is not documented |
| Interface | A browser workspace, seven MCP tools and one resource, and one command line, all over one running app on one port, and one install that serves as many projects as you register | Five MCP tools and four MCP resources (`rag://documents`, `rag://document/{path}`, `rag://query-document/{chunks}/{query}`, `rag://embedding/status`), over one store per connection |
| Human review | Reviewed metadata corrections and reversible exclusions in the UI | Not part of the documented feature set |
| Retrieval evaluation | Measured: 32 known-item judged queries on one reference corpus, reported per mode and per query class, with pooled recall still pending | Not part of the documented feature set |
| Licence | Apache-2.0 for this repository's own code, which is recorded in `NOTICE` | MIT |

Which to choose:

- **Choose `mcp-rag-server`** if you want the shortest path from "I have notes in Markdown or CSV" to "my LLM can see them" — a single `npx` command, an existing Ollama or hosted embedding endpoint, and no concept of projects, generations, or citations to learn.
- **Choose this server** if your material is PDFs and EPUBs, if you need to cite and verify what you found, if you want the knowledge base to survive interruption and updates without being rebuilt by hand, and if you would rather not depend on an external embedding service.
- **Use both** if that is what the work needs. They share no state and can run side by side: one for quick text context, one for a citable document collection.
