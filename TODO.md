---
name: TODO.md
description: Open work, grouped by the problem each item solves.
---

# TODO

- Open work, grouped by the problem each item solves. Behaviour is in `README.md`, capabilities in `FEATURES.md`, the measurement protocol and the decisions it justified in `MEASUREMENTS.md`, deferred ideas in `ROADMAP.md`, and the state format in `STORAGE.md`.

## The app and its front ends

- [ ] **Choose the project in the command centre rather than in the terminal.** Feature. A bare `research-rag` asks which project when this installation holds more than one, and it asks before there is anything to answer in. The app should bind no project, serve every project this installation knows, and re-bind in place to whichever one is chosen there, so one terminal serves every project and closing it ends all of them.
  - `App` is built around a fixed `ResearchConfig`: the gateway, the service, the project lock, and the per-project pid and port files are all built from it, so binding later means tearing those down and building them again. The port, the client registry, and the process outlive every project, and `Surfaces` reads its two surfaces per request, so the swap itself is small.
  - Binding a different project detaches every client on the previous one, because a session answered about one corpus cannot answer about another.
  - The project selector the workspace already renders is navigational: it links to an app that is running and otherwise hands out a command to start one. A host that re-binds in place needs the workspace to offer the choice as an action, and that capability belongs in `surfaces/workspace/`.

- [ ] **Add answer generation, as its own operation with an API.** Feature. The app returns evidence and stops there, so this changes the contract rather than completing one, and it needs a choice before any code is written. `ROADMAP.md` owns the design, the citation contract, and the questions to settle first.
  - It is a separate operation and endpoint, not an argument on `search`, so retrieval stays what a caller gets by default and a caller that wants an answer asks for one.
  - The provider is a setting, so a local model and a hosted API are both reachable, and a hosted provider states what leaves the machine: a query and the passages it retrieved.
  - The answer carries the passages it rests on, each with its locator and `direct_quote_safe: false`, the answer is labelled as generated, and a request for an answer with no evidence is refused rather than answered from the model's own memory.
  - Decide what an answer looks like when the reranker found nothing, because a model asked to write from a thin result set writes from its own knowledge, which is the failure the retrieval contract exists to prevent.
  - The eight agent tools and the one resource do not grow for it; an answer is a caller decision the agent surfaces may expose separately, and `AGENTS.md` records that the evidence-only contract holds until the choice is made.

- [ ] **Let a command start the app when the corpus is not ready.** Decision. A command that touches the corpus goes to the running app, and a project with no app up is answered in process, which opens a second service for a project nobody is serving. Starting the app instead makes `status` leave a process behind. The answer may be a flag, or a short-lived app for read-only commands.
- [ ] **Serve detached, as a feature rather than an absence.** Feature. An app is served from the terminal that started it, and a serving process with no controlling terminal is refused rather than adopted; closing a terminal is how an app ends, and that is the whole argument. Detaching is a legitimate want — a workspace that stays up between sessions, a machine serving a phone — and it belongs in the command line as a named flag with its own state, its own log, and its own `stop`, rather than as a process someone started from a script and walked away from.
  - The ownership proof is already two arguments in the command line plus the project; a detached app needs the terminal recorded as deliberately none rather than absent, because an absent terminal file cannot tell the two apart.
  - `stop` refuses to kill what it cannot prove it owns, so a detached app needs the same proof plus the flag, and a plain `stop` must never end one by accident.
  - Anything that reads the pid file has to learn that a detached app outlives its terminal, including `doctor` and the workspace's project selector.
- [ ] **A connection an unnamed client opened cannot be dropped.** Limit. A client that declared nothing is one client per MCP session, and a disconnect folds its connections onto the session by the name it declared, so a hand-written HTTP client has only its sessions refused. `RESEARCH_RAG_CLIENT_NAME` is the fix, and the stdio bridge always sets it.
- [ ] **Measure the payload difference the app makes.** Missing number. Nothing states the lean and full search sizes. Measure them against the running app, and add the workspace, agent, and control surfaces' own overhead, because one process now serves all three and the claim that they cannot disagree is only as good as the evidence that they are one service.

## A report that is true

- [ ] **Give `stale` one meaning.** Feature. In the no-generation branch it doubles as "ready to build", so a caller cannot tell a corpus that changed from one that was never built.
- [ ] **Report activation failures as structured values**, not one all-or-nothing message. Feature. A failed activation says that it failed rather than which step failed and what it left on disk.
- [ ] **Decide whether `search --method` and `--no-rerank` stay.** Decision. They exist so a row of `MEASUREMENTS.md` can be reproduced on demand, and no reader-facing surface offers either. `AGENTS.md` records the exception; removing them is a deliberate simplification, not a cleanup.
- [ ] **Decide how the source inventory exposes the keyword vocabulary.** Feature. The keyword layer is all-of and its vocabulary only exists across sources, so nothing lets a reader discover which keywords exist; report counts in `status` or reduce the list to handles.
- [ ] **Reads should not queue behind a build.** Problem. Every operation takes the project lock, so `status`, `sources`, `search`, and `passage` are refused while a build runs. A build never mutates the selected generation in place, because activation swaps `current.json` atomically and leaves the old generation root intact, so reads should resolve the selected generation without the lock.
- [ ] **Let one `ingest` call name its own budget.** Feature. The budget is a runtime setting, so a reader cannot raise it for a build that needs longer: the call still needs repeating, and a reader who stops repeating identical calls still cannot finish it. An argument would let the caller say how long it is willing to wait.
- [ ] **Refuse to measure while a build is running.** Feature. `scripts/evaluate_retrieval.py` started alongside an ingestion does not fail, it starves: it sat with 66 ONNX threads at zero CPU time for ten minutes while the build held eleven cores. Notice a staging build under the project's runtime root and stop with a message.
- [ ] **Check the embedding model against the weights the dense backend loads.** Problem. `doctor` warns that `BAAI/bge-small-en-v1.5` is not cached on a machine running dense retrieval, and `doctor --prefetch-models` does not clear the warning. FastEmbed resolves that registry name to the mirror `Qdrant/bge-small-en-v1.5-onnx-Q`, so the cache holds `models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/aa8f8b060edb00e03bfdd08813a2949946c8ba55`, while `runtime/health.py` globs `models--*/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2` from the `BAAI/bge-small-en-v1.5` revision in `retrieval/embeddings.py`. Those revisions belong to different repositories, so the check cannot pass.
  - Four of the six embedding models are downloaded from a mirror and carry a revision of a repository FastEmbed never fetches: `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, and `intfloat/multilingual-e5-large`. `jinaai/jina-embeddings-v2-base-de` and `mixedbread-ai/mxbai-embed-large-v1` are downloaded from their own repositories, and their pins match. A reranker passes because its registry name is the repository it is downloaded from.
  - The `revision` passed to `TextEmbedding` and `TextCrossEncoder` is discarded: `TextEmbeddingBase.__init__` pops `local_files_only` and drops the rest of `**kwargs`, and `OnnxTextEmbedding.__init__` calls `download_model` without forwarding them. The weights come from the mirror's head, so the pin never selected them, and the `embedding_model_revision` that `generations/generation.py` and `core/status.py` compare describes a commit that was never loaded.
  - Both `healthy` fixtures cache `qdrant/bge-small-en-v1.5-onnx-q` under the `BAAI` revision, a directory `snapshot_download` cannot produce. `runtime/health.py` matches it only because the embedding check passes `repository=None` and globs every cached repository, so the test cannot see the defect.
  - Record the repository each model is downloaded from beside its revision, pass that repository to the check, and re-pin the mirrors to their own revisions. Re-pinning changes `embedding_model_revision`, so every generation built with an affected model reports `embedding_model` as stale and needs a rebuild. That changes portable state, so it needs a choice first.
  - Decide whether the pin is enforced or only recorded. Enforcing it needs a `snapshot_download` revision or a verified snapshot hash; recording it honestly needs the revision claims in `FEATURES.md` and `STORAGE.md` to describe what the code does.

## The corpus and the machine holding it

- Work that protects the data, or that stops a build from costing more than it should.
  - [ ] **Narrow the two broad `except Exception` handlers.** Feature. Both are at durability boundaries, so a storage fault is swallowed rather than raised.
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

- [ ] **Decide which of the suite's tests earn their place.** Problem. The suite runs in minutes and the five slowest tests account for most of that: two start the real vanilla gateway, two hold the project lock against a real second process, and one runs the update script. The rest cost seconds together, so the suite is not slow by volume.
  - The `integration` marker is registered and nothing selects on it, and `-m "not integration"` saves far less than the slowest five suggest, because two of them carry no marker.
  - Nothing has read the suite for duplication. Start with `test_service.py`, where thin repetition is most likely, and either delete what a stronger test already covers or mark what is genuinely slow, so an ordinary change can run the fast subset.

- [ ] **Split the remaining concerns in `project/support.py`.** Metadata overlays, chunk records, IR statistics, citations, source identity, and tokenizer caching need separate owners. `project/policy.py` owns lightweight errors and policy primitives; do not route those through `support.py`. Acceptance: shared definitions have one owner and `tests/gates/test_lightweight_imports.py` prevents diagnostic commands from loading retrieval dependencies.

- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `generations/ingestion.py::_advance_ingestion` holds the whole resumable loop, and it is the worst offender in the package for cyclomatic complexity: three of its phase blocks call closures defined inside it and eight read loop-local state, so the split is an ingestion state object that handlers take and return. Accept on a green suite, a re-ingest that reuses every chunk and vector, and `C901` enabled at the level the rest of the package already meets.

- An item is done when the harness has produced its measurement and the validation in `AGENTS.md` is clean.
