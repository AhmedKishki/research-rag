# TODO

- Open work, grouped by the problem each item solves. Behaviour is in `README.md`, capabilities in `FEATURES.md`, the measurement protocol and the decisions it justified in `MEASUREMENTS.md`, deferred ideas in `ROADMAP.md`, the state format in `STORAGE.md`.

## The app and its front ends

- [ ] **Choose the project in the command centre rather than in the terminal.** Feature. A bare `research-rag` asks which project when this installation holds more than one, and it asks before there is anything to answer in. The app should bind no project, serve every project this installation knows, and re-bind in place to whichever one is chosen there, so one terminal serves every project and closing it ends all of them.
  - `App` is built around a fixed `ResearchConfig`: the gateway, the service, the project lock, and the per-project pid and port files are all built from it, so binding later means tearing those down and building them again. The port, the client registry, and the process outlive every project, and `Surfaces` reads its two surfaces per request, so the swap itself is small.
  - Binding a different project detaches every client on the previous one, because a session answered about one corpus cannot answer about another.
  - The project selector the workspace already renders is navigational: it links to an app that is running and otherwise hands out a command to start one. A host that re-binds in place needs the workspace to offer the choice as an action, and that capability belongs in `ui-ultra-rag-mcp` before the pin here moves.

- [ ] **Let a command start the app when the corpus is not ready.** Decision. A command that touches the corpus goes to the running app, and a project with no app up is answered in process, which opens a second service for a project nobody is serving. Starting the app instead makes `status` leave a process behind. The answer may be a flag, or a short-lived app for read-only commands.
- [ ] **A session an unnamed client opened cannot have its stream dropped.** Limit. A disconnect folds the client's sightings onto the session by the name the client declared, and a client that declared nothing has only its session refused. `RESEARCH_RAG_CLIENT_NAME` is the fix, and the stdio bridge always sets it, so this only reaches a hand-written HTTP client.
- [ ] **Measure the payload difference the app makes.** Missing number. Nothing states the lean and full search sizes since the repackaging retired them. Measure them against the running app, and add the workspace, agent, and control surfaces' own overhead, because one process now serves all three and the claim that they cannot disagree is only as good as the evidence that they are one service.

## A report that is true

- [ ] **Give `stale` one meaning.** Feature. In the no-generation branch it doubles as "ready to build", so a caller cannot tell a corpus that changed from one that was never built.
- [ ] **Report activation failures as structured values**, not one all-or-nothing message. Feature. A failed activation says that it failed rather than which step failed and what it left on disk.
- [ ] **Decide whether `search --method` and `--no-rerank` stay.** Decision. They exist so a row of `MEASUREMENTS.md` can be reproduced on demand, and no reader-facing surface offers either. `AGENTS.md` records the exception; removing them is a deliberate simplification, not a cleanup.
- [ ] **Decide how the source inventory exposes the keyword vocabulary.** Feature. The keyword layer is all-of and its vocabulary only exists across sources, so nothing lets a reader discover which keywords exist; report counts in `status` or reduce the list to handles.
- [ ] **Reads should not queue behind a build.** Problem. Every operation takes the project lock, so `status`, `sources`, `search`, and `passage` are refused while a build runs. A build never mutates the selected generation in place: activation swaps `current.json` atomically and leaves the old generation root intact, so reads should resolve the selected generation without the lock and answer during the build.
- [ ] **Let one `ingest` call name its own budget.** Feature. The budget is a runtime setting, so a reader cannot raise it for a build that needs longer: the call still needs repeating, and a reader who stops repeating identical calls still cannot finish it. An argument would let the caller say how long it is willing to wait.
- [ ] **Refuse to measure while a build is running.** Feature. `scripts/evaluate_retrieval.py` started alongside an ingestion does not fail, it starves: it sat with 66 ONNX threads at zero CPU time for ten minutes while the build held eleven cores. Notice a staging build under the project's runtime root and stop with a message.

## The corpus and the machine holding it

- Work that protects the data, or that stops a build from costing more than it should.
  - [ ] **Narrow the two broad `except Exception` handlers.** Feature. At durability boundaries, so a storage fault cannot be swallowed.
  - [ ] **Refuse to start a build that cannot fit.** Feature. `status` and `doctor` report free space against the size of the generations already on disk, and `ingest` does not read that verdict: a build that runs out of room partway leaves a staging directory and no generation, on the machine least able to afford the retry.
  - [ ] **An `ingest` dry run** that reports what would change and what would be reused, and writes nothing. Feature.
  - [ ] **A CPU reserve, so a build leaves cores free.** Feature.
    - `runtime.embedding_threads` is a thread count, not a promise about the machine, and cutting threads costs build throughput, where `runtime.nice` costs none.
    - A reserve expresses the intent directly, as physical cores minus the reserve, and needs a measurement to price it.
  - [ ] **Reuse vectors across a contextual-header change.** Feature. Vector reuse is keyed on canonical passage text, because the same hash is what resolves a BM25 passage back to its chunk, so turning `chunking.headers` on recomputes every vector. The fix is two hash columns — one canonical, one embedded — and a lookup-schema bump, which rebuilds the sidecar from canonical artifacts rather than the corpus. Check the ordering too: `generation_is_reusable` validates the sidecar before anything ensures it, so a version bump denies reuse to the first ingest that follows it.

## Closing the paraphrase gap

- Judged paraphrase queries miss the designated passage within the top ten while nothing is withheld: the passages are there and the ranking cannot find them. `MEASUREMENTS.md` records what the judged set showed. These are the levers.
  - [ ] **Pooled relevance judgments.** Test. The set is known-item — one designated passage per query, one annotator — so a passage that makes the same point scores as a miss, and a change that ranks an equally good passage above the designated one reads as a regression. Collect every candidate from every mode and judge the pool. This is also what would let the pseudo-relevance expansion be measured: with rarity-weighted terms it mines the corpus's own vocabulary, and a known-item set cannot see that.
  - [ ] **Grow the judged set from real questions**, if the privacy of a query log can be settled. Test.
  - [ ] **Measure the headers on a corpus with sections.** Test. A PDF locator carries a page rather than a section, so the header is the title alone and the first chunk of a paper already repeats it; an EPUB corpus is where the locator carries a section and where the header says something the passage does not.
  - [ ] **Measure the embedding models the registry offers.** Test: `BAAI/bge-base-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1` for English, `jinaai/jina-embeddings-v2-base-de` for German, `intfloat/multilingual-e5-large` across languages. The German model is the first candidate for a corpus in that language, where the English default cannot help.
  - [ ] **Merge the stopword lists of a mixed corpus.** Feature. `language.corpus` can name several languages while BM25 filters the one list in `language.bm25_stopwords`, and per-source `language` metadata reports which sources are which. bm25s accepts a list, so the union is expressible: check that the pinned runtime passes a list through `bm25.lang`, then measure the union against one list.
  - [ ] **An Arabic stopword source.** Feature. bm25s ships no Arabic list, so `language.corpus = "ar"` is refused while settings are read even though a source can declare Arabic. Add an explicit list option, or an empty list that filters nothing, plus an Arabic-capable model in the pinned table.
  - [ ] **Keep the measurement current and wider.** Test. Re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and add a second corpus and filtered queries.

## Keeping the code changeable

- [ ] **Decide which of the suite's tests earn their place.** Problem. 636 tests run in 2m50s, and the five slowest account for 156s of that: two start the real vanilla gateway, two hold the project lock against a real second process, and one runs the update script. The remaining 631 cost about 14s together, so the suite is not slow by volume.
  - The `integration` marker is registered and nothing selects on it, and `-m "not integration"` saves 54s rather than the 120s the slowest five suggest, because two of them carry no marker.
  - Nothing has read the suite for duplication. Start with `test_service.py`, where thin repetition is most likely, and either delete what a stronger test already covers or mark what is genuinely slow so an ordinary change can run the fast subset.

- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `_advance_ingestion` is 1,341 lines at complexity 107 against 35 for the next worst function in the package; three of its phase blocks call closures defined inside it and eight read loop-local state, so the split is an ingestion state object that handlers take and return. Accept on a green suite and a re-ingest that reuses every chunk and vector.

- An item is done when the harness has produced its measurement, and the validation in `AGENTS.md` is clean.