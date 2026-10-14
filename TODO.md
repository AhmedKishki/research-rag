# TODO

An item is done when the harness has produced its measurement and the validation in `AGENTS.md` is clean.

## future

* [arxiv.org/pdf/2602.07297](https://arxiv.org/pdf/2602.07297)
* Hierarchical Navigable Small World and KNN

## generating

* [ ] White box LLM with API generation. The generate request goes to an LLM under the hood, and the LLM is responsible for assembling the evidence and formulating the reply. The LLM can have a prompt ready to help it know how to produce the output, how to best search, and so on. Additionally we can add a white box option, where the LLM must utilise the outputs from the RAG for the answer, and never invent a connection. Invented wording must enter via [AI authored: here is the ai authored content].
* [ ] Use the white-box-synthesis: [github.com/AhmedKishki/draft-writing-skill/blob/main/references/tools/white-box-synthesis.md](https://github.com/AhmedKishki/draft-writing-skill/blob/main/references/tools/white-box-synthesis.md)
* [ ] For white box synthesis, the user question, the RAG search results enter as synthesis materials. Note that the RAG may be queried as many times as desired to fomulate a solution. Also note that synthesis can not be mindless but must preserve causality, and the answer must follow a clear logical chain.
* [ ] Alternatively a default black box synthesis can just give the LLM full authority
* [ ] For all generation modes, citation to the sources is a must, and the citation must be inferred from the RAG outputs. Apply citations only when a source carries a strong claim.

## method

* [ ] **Scalability of embeddings is questioned.** if most results are concentrated between 0.7 and 0.9 - there is not so much possibility to differentiate between good and better matches.

## cli

- [ ] **Let a command start the app when the corpus is not ready.** Decision. A command that touches the corpus goes to the running app. With no app up, it is answered in process, which opens a second service nobody serves. Starting an app instead makes `status` leave a process behind. App should be started explicitly for any process.
- [ ] **Decide whether `search --method` and `--no-rerank` stay.** Decision. They reproduce a `MEASUREMENTS.md` row on demand, and no reader-facing surface offers either. Removing them is a deliberate simplification.
  - The running app's control route ignores both flags and returns hybrid, reranked results. Forward the diagnostic choices if retained, or reject unsupported flags explicitly. Test local and running-app behavior.
- [ ] **Share workflow decisions without coupling the surfaces.** Refactor. Audit duplicated argument validation and workflow decisions in the CLI, control API, and workspace adapter. Move shared rules below the surfaces; keep terminal formatting and browser interaction separate. Preserve the no-cross-surface-import gate and test equivalent calls against the same service.
- [ ] **Split `surfaces/cli.py`.** Refactor. One 3,352-line module holds the help text, the parser, the local and remote operations, attached serving, and `update`. Give each its own module behind the same `main`.

## agents

- [ ] **Add answer generation as its own operation with an API.** Feature. It changes the contract and needs a choice before code. `ROADMAP.md` holds the design and the open questions.
  - It is a separate endpoint, not an argument on `search`.
  - The eight agent tools and the one resource do not grow for it.
  - The evidence-only contract in `AGENTS.md` holds until the choice is made.
- [ ] **Measure the payload difference the app makes.** Nothing states the lean and full search sizes. Measure them against the running app, including the workspace, agent, and control surfaces' own overhead.

## app

- [ ] **Choose the project in the workspace, not the terminal.** Feature. A bare `research-rag` asks which project when the installation holds several, before the app can answer anything.

  - The app binds no project, serves every known project, and re-binds in place to the one chosen.
  - One terminal then serves every project, and closing it ends all of them.
  - `App` is built around a fixed `ResearchConfig`: the gateway, service, project lock, and pid and port files all derive from it. Binding later tears those down and rebuilds them.
  - The port, client registry, and process outlive every project, and `Surfaces` reads its two surfaces per request.
  - Binding a different project detaches every client on the previous one.
  - The workspace's project selector only links to a running app or hands out a start command. Re-binding needs an action in `surfaces/workspace/`.
- [ ] **Serve detached.** Feature. A serving process with no controlling terminal is refused, not adopted. Detaching suits a workspace that stays up between sessions or a machine serving a phone.

  - Add a named flag with its own state, log, and `stop`.
  - Detached server must be stoppable by a new server attach request, or ownership transfered to terminal.
  - Record the terminal as deliberately none, because an absent terminal file cannot tell the two cases apart.
  - `stop` needs the same ownership proof plus the flag, and a plain `stop` must never end a detached app.
  - `doctor` and the project selector must learn that a detached app outlives its terminal.
- [ ] **Authenticated, encrypted remote serving.** Feature. Home-LAN mode has no login and uses unencrypted HTTP; it is not a public or multi-user deployment.

  - Design authentication, session expiry, TLS trust, and permissions before supporting untrusted networks or public exposure.
  - Choose authenticated remote access or withholding LAN exposure before changing the supported security boundary. Protect search, passages, originals, and enabled writes; a write-only token leaves the corpus readable.
  - Treat Host, private-peer, and Origin checks as boundary and browser-write protections, not authentication. A bearer secret over HTTP does not protect against interception.
  - Preserve local-only CLI control, MCP, and installation management.
  - Keep loopback the default. Extend the explicit host, peer, and browser-origin policies rather than disabling them.
- [ ] **A connection an unnamed client opened cannot be dropped.** Limit. A client that declared nothing is one client per MCP session, so a hand-written HTTP client has only its sessions refused. `RESEARCH_RAG_CLIENT_NAME` fixes it, and the stdio bridge always sets it.
- [ ] **Diagnose intermittent MCP initialization HTTP 500.** Test. Reproduce `/mcp` termination before `http.response.start` with synthetic projects and a minimized execution-order or load trigger. Capture session, lifespan, and cancellation events; compare the trigger against `c95e435` with identical dependencies. Correct the demonstrated cause with a deterministic regression test, not retries, skips, longer timeouts, or weaker security. Validate app/LAN tests and a serialized non-integration suite.
  - Test `BaseHTTPMiddleware` first. `runtime/app.py` wraps the streaming MCP app in it, and Starlette's base middleware raises `No response returned` when the wrapped app ends before `http.response.start`. Pure ASGI middleware avoids it.
- [ ] **Bound each client's session list.** Bug. The stdio bridge opens an upstream session per call, so an active client's `session_ids` grows for the life of the app and `clients` reports every one.
- [ ] **Reuse the bridge's upstream session.** Feature. Each bridged tool call costs a health request, a control request, and a new MCP handshake.

## reporting

- [ ] **Show ingestion and generation activity in the workspace.** Feature. Show the active phase, completed and total work where available, elapsed time, and the last progress update so a user can distinguish a working server from a stalled operation. Show an activity indicator during phases without measurable progress. Distinguish waiting, running, failed, and completed states; do not imply that animation alone proves progress. Test long phases and stale or disconnected status updates.
- [ ] **Give `stale` one meaning.** Feature. With no generation it doubles as "ready to build", so a caller cannot tell a changed corpus from a never-built one.
- [ ] **Report activation failures as structured values.** Feature. A failed activation does not say which step failed or what it left on disk.
- [ ] **Decide how the source inventory exposes the keyword vocabulary.** Feature. The keyword layer is all-of, and the vocabulary exists only across sources, so a reader cannot discover which keywords exist. Report counts in `status` or reduce the list to handles.

## ingestion

- [ ] **Consider docling for ingestion.** [github.com/docling-project/docling](https://github.com/docling-project/docling)
- [ ] **Let one `ingest` call name its budget.** Feature. The budget is a runtime setting, so a reader cannot raise it for one call. An argument would let the caller say how long to wait.
- [ ] **Refuse to start a build that cannot fit.** Feature. `status` and `doctor` report free space against existing generation sizes, but `ingest` ignores the verdict. A build that runs out of room leaves a staging directory and no generation.
- [ ] **Add an `ingest` dry run.** Feature. Report what would change and what would be reused, and write nothing.
- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `generations/ingestion.py::_advance_ingestion` holds the whole loop. Three phase blocks call closures defined inside it, and eight read loop-local state, so the split needs an ingestion state object that handlers take and return. Acceptance: a green suite, a re-ingest that reuses every chunk and vector, and `C901` enabled at the level the rest of the package meets.
- [ ] **Open an EPUB once per build call.** Feature. `extract_epub_spine_item` re-reads the whole book for every spine item.
- [ ] **Validate an unchanged generation without holding it in memory.** Feature. A no-op ingest parses every artifact and hashes both JSONL files, and `validate_artifact_lookup` loads every chunk record at once.
- [ ] **Survive a touched source mid-build.** Feature. The checkpoint identity and the final inventory check include `mtime_ns`, so touching a file between ingest calls discards a build whose bytes did not change.

## text extraction

- [ ] **Refuse a garbage passage, not only a garbage unit.** Feature. `corpus/text_quality.py` judges a unit and extraction withholds one whose text carries a reason. Extend the verdicts to what a reader retrieves — running heads, page numbers, a table of contents, an index, a reference list, boilerplate — and refuse the passage at build time. Measure what it removes, and what it takes with it, against the judged set.
- [ ] **Measure the EPUB note-reference rule.** Test. `extraction._is_epub_noteref` treats a linked number as a note reference when the link carries an `id`, so a cross-reference number with an `id` loses its text. Count what the rule removes on the judged corpus.
- [ ] **Track open EPUB blocks without descendant sets.** Feature. `_epub_text_segments` builds a set of every descendant for each content element it enters.

## chunking

* [ ] **Context-aware chunking.** [github.com/coleam00/ottomator-agents/blob/main/all-rag-strategies/docs/07-context-aware-chunking.md](https://github.com/coleam00/ottomator-agents/blob/main/all-rag-strategies/docs/07-context-aware-chunking.md)

- [ ] **Enforce the embedding model's token limit when chunking.** Feature. GPT-2-sized chunks can exceed the embedding model's WordPiece limit, especially on table-of-contents dot leaders. Count the actual embedding input, including headers and special tokens, and split oversized chunks without dropping text. Test punctuation-heavy input and ordinary prose. Measure retrieval quality and rebuild cost before changing generation artifacts.
  - Investigate the observed overflow before choosing the fix. A live build produced 36,358 chunks; 15 exceeded the embedding model's 512-token limit, and the maximum actual input was 3,417 tokens. Their canonical text remains stored, but their vectors represent truncated input.
  - Identify every affected chunk through the generation's dense-token audit. Record its source locator, content kind, canonical text length, chunker token count, embedding token count, and exact embedding input.
  - Determine whether each overflow comes from tokenizer mismatch, punctuation or dot leaders, lost word boundaries, preprocessing, headers or special tokens, or a chunk-boundary defect. Do not assume one cause covers all 15.
  - Reproduce the largest case and each distinct cause. Acceptance: every actual embedding input fits the model limit without losing source text, with preserved locators and explicit regeneration cost.
- [ ] **Reuse vectors across a contextual-header change.** Feature. Vector reuse keys on canonical passage text, so turning `chunking.headers` on recomputes every vector.
  - Fix: two hash columns, one canonical and one embedded, plus a lookup-schema bump. The bump rebuilds the sidecar from canonical artifacts, not the corpus.
  - `generation_is_reusable` validates the sidecar before anything ensures it, so a version bump denies reuse to the first ingest after it.
- [ ] **Chunk by paragraph, not by a fixed token window.** Feature. `chunking.size` and `chunking.overlap` cut fixed GPT-2 token windows, so a passage can begin and end mid-sentence and neighbours repeat the overlap. A paragraph boundary, with a size ceiling and an overlap only where one paragraph exceeds it, is what a reader quotes. Needs a judged-set measurement before it changes the default; both settings are identity, so a change rebuilds the generation.
- [ ] **Measure headers on a corpus with sections.** Test. A PDF locator carries a page, so the header is the title alone. An EPUB locator carries a section, which the passage does not state.

## embedding

- [ ] **Measure the registry's embedding models.** Test: `BAAI/bge-base-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1` for English, `jinaai/jina-embeddings-v2-base-de` for German, `intfloat/multilingual-e5-large` across languages.

## indexing

- [ ] **Build the exact index whenever `dense.backend` is `auto`.** Bug. Above `dense.exact_backend_chunk_limit` `auto` selects Qdrant local mode, which loads every point on each client open and scores by brute force. `qdrant.py` opens a client per search and per 64-chunk upload, so a build is quadratic. Existing Qdrant generations stay readable.
- [ ] **Bound the exact backend's index cache.** Bug. `LocalVectorDenseBackend._loaded` never evicts. Each build leaves a staging-path entry, and a removed generation's vectors stay mapped, so `remove_generation` frees no disk space until the app restarts.
- [ ] **Measure the operating envelope.** Test. No figure covers search latency, index size, resident memory, or build time at the corpus size this app claims to serve. Nothing states where the exact scan stops being usable. Record the envelope in `MEASUREMENTS.md`, and let it bound the source and passage counts the app supports.
- [ ] **Choose whether canonical retrieval text moves into SQLite.** Decision. `retrieval/artifact_lookup.py` keeps a SQLite sidecar of identifiers, ordinals, digests, and byte offsets, while passage and extraction text stay in JSONL and are read on demand; a large corpus pays a file read per hit. Options: keep the split and measure it against the recorded envelope, or make the sidecar the canonical store for retrieval text. `sources/` remains the authority for a quotation, and the artifacts stay portable, either way.

## retrieval

- [ ] **Report weak relevance separately from candidate availability.** Feature. An unrelated query can return a full result set with uniformly poor reranker scores while `relevance_limited` remains false. Measure a low-relevance warning or abstention policy against judged relevant and unrelated queries. Do not assume raw reranker scores share a universal cutoff. Preserve the evidence-only contract and expose the condition consistently across surfaces.
- [ ] **Choose a duplicate policy that preserves distinct claims.** Decision. `MEASUREMENTS.md` records cosine-collapse counterexamples and a sign contrast lost by `same_words`. Compare the current policy, no semantic suppression, explicit project opt-in, and strict canonical-text grouping with each member's source and locator preserved. Approve and evaluate a policy before changing retrieval defaults.
  - Test negation, reversed actors, changed quantities, and meaningful punctuation. Same-source membership or a higher similarity threshold does not prove claim equality.
  - Define an explicit disabled state. Zero is a cosine threshold in the current implementation, not an off switch.
  - Cover one essay alone and inside a book, a byte-identical source under two names, and repeated passages without merging provenance or inferring independent evidence.
  - Keep suppression at search time, never at chunk- or record-time, so a duplicate keeps the rank the next query could use.
- [ ] **Return scored lexical identities and filter before top-k.** Refactor. `DirectRetrieval.search_bm25` still returns strings and discards scores. Choose a typed row/chunk-ID boundary before implementation. Preserve occurrence identity for identical text at different locators. Apply reviewed metadata, document selection, and source/passage exclusions to eligible rows before top-k while retaining corpus-wide BM25 statistics. Remove content remapping and filter-window widening only after compatibility tests cover ties, selective filters, old indexes, concurrent reads/builds, and offline restart.
- [ ] **Split the search pipeline into explicit stages.** Refactor. `retrieval/search.py::SearchWorkflow.search` owns admission, filtering, fusion, relevance gates, reranking, repetition handling, selection, and disclosure. Give stages explicit input/output state and preserve ordered hit IDs and diagnostic counts in differential tests before removing its complexity exemption.
- [ ] **Improve paraphrase matching, and price each lever.** Feature. Measure the levers the app already reaches — the pseudo-relevance expansion, the CPU reranker, and a larger embedding model from the pinned registry — against the pooled judgments once they exist. Keep only a lever whose pooled result improves and whose cost is stated.
- [ ] **Merge the stopword lists of a mixed corpus.** Feature. `language.corpus` can name several languages while BM25 filters the one list in `language.bm25_stopwords`. bm25s accepts a list. Check that the pinned runtime passes a list through `bm25.lang`, then measure the union against one list.
- [ ] **Add an Arabic stopword source.** Feature. bm25s ships no Arabic list, so `language.corpus = "ar"` is refused although a source can declare Arabic. Add an explicit list option or an empty list, plus an Arabic-capable model in the pinned table.

## storage

- [ ] **Narrow the two broad `except Exception` handlers.** Feature. Both sit at durability boundaries, so a storage fault is swallowed, not raised.
- [ ] **Measure retained-generation growth before changing storage.** Decision. Compare the current self-contained generations, explicit retention controls, filesystem-assisted sharing, and content-addressed blobs. A passage occurrence ID is not a text-only blob key. Any shared store needs durable references, crash-safe garbage collection, safe interrupted-build cleanup, portability, and preservation of current, leased, and pending-activation generations. Get a choice before changing layouts or deleting retained evidence.
- [ ] **Bound search-history retention.** Decision. `search-stats.sqlite3` gains one search row and up to five rank rows per search with no retention limit, and history is on by default.
- [ ] **Close artifact-lookup connections explicitly.** Bug. `retrieval/artifact_lookup.py` opens SQLite with `with sqlite3.connect(...)`, which commits but does not close.

## security

- [ ] **Gate LAN routes with an allowlist.** Feature. `runtime/network.py::LOCAL_ONLY_PATHS` is a denylist, so a new route is reachable from the LAN until someone lists it.
- [ ] **Decide which writes a LAN peer may make.** Decision. A LAN peer can remove and select generations, write settings, start ingest, and read or clear search history.
- [ ] **Authenticate local callers.** Decision. `/control` and `/mcp` accept any local process, so another user on the machine can enable LAN access, write settings, or remove a generation. A per-run token in a 0600 runtime file is one option; it adds a runtime file.
- [ ] **Validate `Origin` on `/mcp`.** Feature. The MCP transport requires the check. The Host check already blocks DNS rebinding.
- [ ] **Bound EPUB parsing.** Feature. EPUB is parsed in the serving process without limits, while MOBI and Docling run in bounded children. A zip bomb or an oversized XHTML item can exhaust the app.
- [ ] **Lock the Docling environment's transitive dependencies.** Feature. `corpus/docling_env.py` pins three packages. torch and the Docling components resolve at install time without hashes, so two machines can extract the same PDF differently.

## release

- [ ] **Install the release `update` approved.** Bug. The distribution path asks the package index for a package published only on GitHub, and `uv tool upgrade` re-resolves the git specifier to the default branch head rather than the tag whose changelog was shown. README installs from the branch head as well.
- [ ] **Run `uv lock --check` in CI.** Bug. `AGENTS.md` requires it before a commit, and `.github/workflows/test.yml` omits it.
- [ ] **Point the `default.toml` package data at its file.** Bug. `pyproject.toml` names `research_rag/default.toml`; the file is `research_rag/project/default.toml` and ships only through setuptools' default discovery.

## evaluation

- [ ] **Refuse to measure while a build runs.** Feature. `scripts/evaluate_retrieval.py` started alongside an ingestion starves instead of failing: it sat at zero CPU time for ten minutes while the build held eleven cores. Detect a staging build under the runtime root and stop with a message.
- [ ] **Pool relevance judgments.** Test. The set is known-item, so a passage making the same point scores as a miss. Collect every candidate from every mode and judge the pool. Pooling also lets the pseudo-relevance expansion be measured, which a known-item set cannot see.
- [ ] **Grow the judged set from real questions.** Test. Needs a settled privacy position on query logs.
- [ ] **Keep the measurement current and wider.** Test. Re-run `scripts/evaluate_retrieval.py` when the corpus, extraction policy, or a retrieval default changes. Add a second corpus and filtered queries.

## project

- [ ] **Split the remaining concerns in `project/support.py`.** Metadata overlays, chunk records, IR statistics, citations, source identity, and tokenizer caching need separate owners. `project/policy.py` owns lightweight errors and policy primitives. Acceptance: shared definitions have one owner, and `tests/gates/test_lightweight_imports.py` keeps diagnostic commands from loading retrieval dependencies.
- [ ] **Remove private-helper compatibility re-exports.** Refactor. Update tests and harnesses to import private helpers from their owning modules rather than `core/service.py`. Remove unused re-exports without changing supported service operations, and keep architecture and lightweight-import gates green.
- [ ] **Assess settings layers before removing supported overrides.** Decision. Compare retaining the current stack with a smaller stack and an explicit deprecation path. Measure actual use and ambiguity; retain validation, effective-value provenance, rebuild-cost calculation, and protection against ineffective writes. A single user can still need several projects and machine-specific overrides.
- [ ] **Type-check in CI.** Refactor. `ResearchService` combines six workflow mixins that share undeclared attributes, and no checker reads the type hints. Declare the shared state as a `Protocol` and add pyright or mypy to CI.
- [ ] **Remove internal compatibility names and dead code.** Refactor. `ResearchService(ultrarag=...)`, `ResearchService.ultrarag`, `App.gateway`, the module-level `runtime/app.py::_security_headers`, `runtime/write_guard.py`, and `retrieval/direct.py::_sync_directory` serve tests only. The search payload's `conserved` check cannot fail, and `direct.py` overwrites the numpy fallback with `backend = "numba"`.
- [ ] **Remove workspace routes this app never enables.** Decision. The memory, SQL, and bundle routes and their `LOCAL_ONLY_PATHS` entries serve no capability here. The workspace copy shared with memory-rag is deliberate, so removal is a choice.
- [ ] **Correct the `runtime.offline` comment.** Bug. `default.toml` still says offline mode requires an installed vanilla runtime.
