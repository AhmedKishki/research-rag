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
  - Preserve local-only CLI control, MCP, and installation management.
  - Keep loopback the default. Extend the explicit host, peer, and browser-origin policies rather than disabling them.
- [ ] **A connection an unnamed client opened cannot be dropped.** Limit. A client that declared nothing is one client per MCP session, so a hand-written HTTP client has only its sessions refused. `RESEARCH_RAG_CLIENT_NAME` fixes it, and the stdio bridge always sets it.

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

## text extraction

- [ ] **Refuse a garbage passage, not only a garbage unit.** Feature. `corpus/text_quality.py` judges a unit and extraction withholds one whose text carries a reason. Extend the verdicts to what a reader retrieves — running heads, page numbers, a table of contents, an index, a reference list, boilerplate — and refuse the passage at build time. Measure what it removes, and what it takes with it, against the judged set.

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

- [ ] **Measure the operating envelope.** Test. No figure covers search latency, index size, resident memory, or build time at the corpus size this app claims to serve. `retrieval.exact_backend_chunk_limit = 200000` is where `auto` selects the ANN backend, but nothing states where the exact scan stops being usable. Record the envelope in `MEASUREMENTS.md`, and let it bound the source and passage counts the app supports.
- [ ] **Choose whether canonical retrieval text moves into SQLite.** Decision. `retrieval/artifact_lookup.py` keeps a SQLite sidecar of identifiers, ordinals, digests, and byte offsets, while passage and extraction text stay in JSONL and are read on demand; a large corpus pays a file read per hit. Options: keep the split and measure it against the recorded envelope, or make the sidecar the canonical store for retrieval text. `sources/` remains the authority for a quotation, and the artifacts stay portable, either way.

## retrieval

- [ ] **Report weak relevance separately from candidate availability.** Feature. An unrelated query can return a full result set with uniformly poor reranker scores while `relevance_limited` remains false. Measure a low-relevance warning or abstention policy against judged relevant and unrelated queries. Do not assume raw reranker scores share a universal cutoff. Preserve the evidence-only contract and expose the condition consistently across surfaces.
- [ ] **Make duplicate handling a stated, measured policy.** Feature. `retrieval.duplicate_cosine` already suppresses a passage a search has shown, at search time. Name every case it must cover — one essay alone and inside a book, a byte-identical source under two names, a passage repeated across a generation — and measure what still appears twice on a large corpus.
  - Keep suppression at search time, never at chunk- or record-time, so a duplicate keeps the rank the next query could use.
- [ ] **Improve paraphrase matching, and price each lever.** Feature. Measure the levers the app already reaches — the pseudo-relevance expansion, the CPU reranker, and a larger embedding model from the pinned registry — against the pooled judgments once they exist. Keep only a lever whose pooled result improves and whose cost is stated.
- [ ] **Merge the stopword lists of a mixed corpus.** Feature. `language.corpus` can name several languages while BM25 filters the one list in `language.bm25_stopwords`. bm25s accepts a list. Check that the pinned runtime passes a list through `bm25.lang`, then measure the union against one list.
- [ ] **Add an Arabic stopword source.** Feature. bm25s ships no Arabic list, so `language.corpus = "ar"` is refused although a source can declare Arabic. Add an explicit list option or an empty list, plus an Arabic-capable model in the pinned table.

## storage

- [ ] **Narrow the two broad `except Exception` handlers.** Feature. Both sit at durability boundaries, so a storage fault is swallowed, not raised.

## evaluation

- [ ] **Refuse to measure while a build runs.** Feature. `scripts/evaluate_retrieval.py` started alongside an ingestion starves instead of failing: it sat at zero CPU time for ten minutes while the build held eleven cores. Detect a staging build under the runtime root and stop with a message.
- [ ] **Pool relevance judgments.** Test. The set is known-item, so a passage making the same point scores as a miss. Collect every candidate from every mode and judge the pool. Pooling also lets the pseudo-relevance expansion be measured, which a known-item set cannot see.
- [ ] **Grow the judged set from real questions.** Test. Needs a settled privacy position on query logs.
- [ ] **Keep the measurement current and wider.** Test. Re-run `scripts/evaluate_retrieval.py` when the corpus, extraction policy, or a retrieval default changes. Add a second corpus and filtered queries.

## project

- [ ] **Split the remaining concerns in `project/support.py`.** Metadata overlays, chunk records, IR statistics, citations, source identity, and tokenizer caching need separate owners. `project/policy.py` owns lightweight errors and policy primitives. Acceptance: shared definitions have one owner, and `tests/gates/test_lightweight_imports.py` keeps diagnostic commands from loading retrieval dependencies.
