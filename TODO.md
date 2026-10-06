# TODO

An item is done when the harness has produced its measurement and the validation in `AGENTS.md` is clean.

Entries are grouped by the module each one touches.

## cli

- [ ] **Let a command start the app when the corpus is not ready.** Decision. A command that touches the corpus goes to the running app. With no app up, it is answered in process, which opens a second service nobody serves. Starting an app instead makes `status` leave a process behind. App should be started explicitly for any process.
- [ ] **Decide whether `search --method` and `--no-rerank` stay.** Decision. They reproduce a `MEASUREMENTS.md` row on demand, and no reader-facing surface offers either. Removing them is a deliberate simplification.

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
- [ ] **Serve beyond loopback with an explicit security design.** Feature. The workspace binds one loopback port and refuses every other peer, `Host`, and `Origin`; it has no authentication, so `AGENTS.md` requires a design before remote exposure.

  - Choose the shape: a first-party remote mode, or an authenticated proxy that terminates on loopback and is documented as the supported way to reach a phone or another machine.
  - A first-party mode needs a configured bind address and allowed `Host` and `Origin`, an access token, TLS or a trusted TLS terminator, and CSRF handling.
  - State a threat model for every write surface it would expose: settings, ingestion, source and passage exclusions, generation removal, the SQL console, and memory writes.
  - Keep loopback the default. `doctor` and `README.md` state when remote serving is on and what protects it.
- [ ] **A connection an unnamed client opened cannot be dropped.** Limit. A client that declared nothing is one client per MCP session, so a hand-written HTTP client has only its sessions refused. `RESEARCH_RAG_CLIENT_NAME` fixes it, and the stdio bridge always sets it.

## reporting

- [ ] **Give `stale` one meaning.** Feature. With no generation it doubles as "ready to build", so a caller cannot tell a changed corpus from a never-built one.
- [ ] **Report activation failures as structured values.** Feature. A failed activation does not say which step failed or what it left on disk.
- [ ] **Decide how the source inventory exposes the keyword vocabulary.** Feature. The keyword layer is all-of, and the vocabulary exists only across sources, so a reader cannot discover which keywords exist. Report counts in `status` or reduce the list to handles.

## ingestion

- [ ] **Let one `ingest` call name its budget.** Feature. The budget is a runtime setting, so a reader cannot raise it for one call. An argument would let the caller say how long to wait.
- [ ] **Refuse to start a build that cannot fit.** Feature. `status` and `doctor` report free space against existing generation sizes, but `ingest` ignores the verdict. A build that runs out of room leaves a staging directory and no generation.
- [ ] **Add an `ingest` dry run.** Feature. Report what would change and what would be reused, and write nothing.
- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `generations/ingestion.py::_advance_ingestion` holds the whole loop. Three phase blocks call closures defined inside it, and eight read loop-local state, so the split needs an ingestion state object that handlers take and return. Acceptance: a green suite, a re-ingest that reuses every chunk and vector, and `C901` enabled at the level the rest of the package meets.

## text extraction

- [ ] **Refuse a garbage passage, not only a garbage unit.** Feature. `corpus/text_quality.py` judges a unit and extraction withholds one whose text carries a reason. Extend the verdicts to what a reader retrieves — running heads, page numbers, a table of contents, an index, a reference list, boilerplate — and refuse the passage at build time. Measure what it removes, and what it takes with it, against the judged set.

## chunking

- [ ] **Reuse vectors across a contextual-header change.** Feature. Vector reuse keys on canonical passage text, so turning `chunking.headers` on recomputes every vector.
  - Fix: two hash columns, one canonical and one embedded, plus a lookup-schema bump. The bump rebuilds the sidecar from canonical artifacts, not the corpus.
  - `generation_is_reusable` validates the sidecar before anything ensures it, so a version bump denies reuse to the first ingest after it.
- [ ] **Chunk by paragraph, not by a fixed token window.** Feature. `chunking.size` and `chunking.overlap` cut fixed GPT-2 token windows, so a passage can begin and end mid-sentence and neighbours repeat the overlap. A paragraph boundary, with a size ceiling and an overlap only where one paragraph exceeds it, is what a reader quotes. Needs a judged-set measurement before it changes the default; both settings are identity, so a change rebuilds the generation.
- [ ] **Measure headers on a corpus with sections.** Test. A PDF locator carries a page, so the header is the title alone. An EPUB locator carries a section, which the passage does not state.

## embedding

- [ ] **Check the embedding model against the weights the dense backend loads.** Problem. `doctor` warns that `BAAI/bge-small-en-v1.5` is not cached, and `doctor --prefetch-models` does not clear it.
  - FastEmbed resolves that name to the mirror `Qdrant/bge-small-en-v1.5-onnx-Q`. The cache holds `models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/aa8f8b060edb00e03bfdd08813a2949946c8ba55`.
  - `runtime/health.py` globs `models--*/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2`, the `BAAI` revision from `retrieval/embeddings.py`. The two revisions belong to different repositories, so the check cannot pass.
  - Four of the six embedding models load from a mirror: `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, and `intfloat/multilingual-e5-large`. `jinaai/jina-embeddings-v2-base-de` and `mixedbread-ai/mxbai-embed-large-v1` load from their own repositories, and their pins match. Rerankers pass because their registry names are their repositories.
  - The `revision` passed to `TextEmbedding` and `TextCrossEncoder` is discarded: `TextEmbeddingBase.__init__` drops the `**kwargs`, and `OnnxTextEmbedding.__init__` calls `download_model` without them. Weights come from the mirror's head, so the `embedding_model_revision` that `generations/generation.py` and `core/status.py` compare names a commit that was never loaded.
  - Both `healthy` fixtures cache `qdrant/bge-small-en-v1.5-onnx-q` under the `BAAI` revision, a directory `snapshot_download` cannot produce. The check passes only because it uses `repository=None` and globs every repository.
  - Fix: record each model's download repository beside its revision, pass it to the check, and re-pin the mirrors to their own revisions.
    - Re-pinning changes `embedding_model_revision`, so every generation built with an affected model reports `embedding_model` as stale and needs a rebuild. That changes portable state and needs a choice first.
  - Decide whether the pin is enforced or only recorded.
    - Enforcing needs a `snapshot_download` revision or a verified snapshot hash.
    - Recording needs `FEATURES.md` and `STORAGE.md` to describe what the code does.
- [ ] **Measure the registry's embedding models.** Test: `BAAI/bge-base-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1` for English, `jinaai/jina-embeddings-v2-base-de` for German, `intfloat/multilingual-e5-large` across languages.

## indexing

- [ ] **Measure the operating envelope.** Test. No figure covers search latency, index size, resident memory, or build time at the corpus size this app claims to serve. `retrieval.exact_backend_chunk_limit = 200000` is where `auto` selects the ANN backend, but nothing states where the exact scan stops being usable. Record the envelope in `MEASUREMENTS.md`, and let it bound the source and passage counts the app supports.
- [ ] **Choose whether canonical retrieval text moves into SQLite.** Decision. `retrieval/artifact_lookup.py` keeps a SQLite sidecar of identifiers, ordinals, digests, and byte offsets, while passage and extraction text stay in JSONL and are read on demand; a large corpus pays a file read per hit. Options: keep the split and measure it against the recorded envelope, or make the sidecar the canonical store for retrieval text. `sources/` remains the authority for a quotation, and the artifacts stay portable, either way.

## retrieval

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

## tests

- [ ] **Decide which tests earn their place.** Problem. The suite runs in minutes, and five tests account for most of that: two start the real vanilla gateway, two hold the project lock against a real second process, and one runs the update script.
  - The `integration` marker is registered and nothing selects on it. `-m "not integration"` saves little, because two of the five carry no marker.
  - Nobody has read the suite for duplication. Start with `test_service.py`. Delete what a stronger test covers, or mark what is slow, so an ordinary change can run the fast subset.
