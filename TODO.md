# TODO

Open work, grouped by the problem each item solves. A finished item leaves this file: git history is the archive. Behaviour is in `README.md`, capabilities in `FEATURES.md`, numbers and limits in `MEASUREMENTS.md`, deferred ideas in `ROADMAP.md`, the state format in `STORAGE.md`.

## The app and its front ends

- [ ] **Let a command start the app when the corpus is not ready.** Decision. A command that touches the corpus goes to the running app, and a project with no app up is answered in process, which opens a second service for a project nobody is serving. Starting the app instead would make `status` leave a process behind, so neither is right; the answer may be a flag, or a short-lived app for read-only commands.
- [ ] **A session an unnamed client opened cannot have its stream dropped.** Limit. A disconnect folds the client's sightings onto the session by the name the client declared, and a client that declared nothing has only its session refused. `RESEARCH_ULTRARAG_CLIENT_NAME` is the fix, and the stdio bridge always sets it, so this only reaches a hand-written HTTP client.
- [ ] **Measure the payload difference the app makes.** Missing number. `MEASUREMENTS.md` carries the lean and full search sizes from before the repackaging. Re-measure them against the running app, and add the workspace, agent, and control surfaces' own overhead, because one process now serves all three and the claim that they cannot disagree is only as good as the evidence that they are one service.
- [ ] **Decide what a second app for a second project costs.** Problem. One app owns one project, so two projects need two apps and two ports. That is correct and unsurprising, but nothing states it where a reader meets it, and the launcher was built for the case where only one project was ever open.
- [ ] **Record the skew between this app and the frozen MCP server.** Gap. `code_currency` catches two checkouts of one product, not this app answering a project the frozen product also serves. The on-disk contract is frozen and stated in this app's `AGENTS.md`, but nothing reports the two disagreeing about a field set. A number in `MEASUREMENTS.md` measured by one and read in the other is not comparable until something says they are the same tree.

## Settings a person sets

- [ ] **Let a command write a project's settings.** Gap. `research-rag config` prints every effective value and the layer it came from, and `--set key=value` overrides a value for one command and forgets it, so a project is configured by hand-editing `.research-rag/config.toml`. The writer needs an atomic replace that preserves the comments and the keys it does not own, a rule for which layer a command writes to, and a statement of what a change costs: most keys decide what a generation contains, so editing one is a rebuild, and the running app resolves its settings at start, so it does not see the edit until it restarts. Start with the command line; the workspace view is the same reader in a browser and can follow.
- [ ] **Say what a settings change does to the current generation.** Problem. A value that decides what a generation contains enters the retrieval-policy fingerprint, so editing it makes the selected generation stale, and nothing in a settings answer says which keys those are. A reader who changes one and runs a search gets `stale` and has to work out why.


## A report that is true

What a reader is told has to match what the app holds.

- [ ] **Surface the retained-generation inventory in the workspace.** Feature. `status` reports `generations` and `retained_generation_bytes`; the pinned status view shows none of them, so a browser cannot see what a prune would consider.
- [ ] **Give `stale` one meaning.** Feature. In the no-generation branch it doubles as "ready to build", so a caller cannot tell a corpus that changed from one that was never built.
- [ ] **Report activation failures as structured values** instead of one all-or-nothing message. Feature. A failed activation says that it failed rather than which step failed and what it left on disk.
- [ ] **Decide whether `search --method` and `--no-rerank` stay.** Decision. They exist so a row of `MEASUREMENTS.md` can be reproduced on demand, and no reader-facing surface offers either. `AGENTS.md` records the exception; removing them is a deliberate simplification, not a cleanup.
- [ ] **Decide how the source inventory exposes the keyword vocabulary.** Feature. The keyword layer is all-of and its vocabulary only exists across sources, so nothing lets a reader discover which keywords exist; report counts in `status` or reduce the list to handles.
- [ ] **Reads should not queue behind a build.** Problem. Every operation takes the project lock, so `status`, `sources`, `search`, and `passage` are refused while a build runs. A build never mutates the selected generation in place: activation swaps `current.json` atomically and leaves the old generation root intact, so reads should resolve the selected generation without the lock and answer while the build proceeds.
- [ ] **Let one `ingest` call name its own budget.** Feature. The budget is a runtime setting, so a reader cannot raise it for a build that needs longer: the call still needs repeating, and a reader who stops repeating identical calls still cannot finish it. An argument would let the caller say how long it is willing to wait.
- [ ] **Refuse to measure while a build is running.** Feature. `scripts/evaluate_retrieval.py` started alongside an ingestion does not fail, it starves: it sat with 66 ONNX threads at zero CPU time for ten minutes while the build held eleven cores. Notice a staging build under the project's runtime root and stop with a message instead.

## The corpus and the machine holding it

Work that protects the data, or that stops a build from costing more than it should.

- [ ] **Narrow the two broad `except Exception` handlers** at durability boundaries, so a storage fault cannot be swallowed. Feature.
- [ ] **Refuse to start a build that cannot fit.** Feature. `status` and `doctor` report free space against the size of the generations already on disk, and `ingest` does not read that verdict: a build that runs out of room partway leaves a staging directory and no generation, on the machine least able to afford the retry.
- [ ] **An `ingest` dry run** that reports what would change and what would be reused, and writes nothing. Feature.
- [ ] **A CPU reserve, so a build leaves cores free.** Feature. `runtime.embedding_threads` is a thread count, not a promise about the machine, and cutting threads costs build throughput — 8 threads measured 31.66 chunks/s against the default's 23.65 — where `runtime.nice` costs none. A reserve expresses the intent directly, as physical cores minus the reserve, and needs a measurement to price it.
- [ ] **Reuse vectors across a contextual-header change.** Feature. Vector reuse is keyed on canonical passage text, because the same hash is what resolves a BM25 passage back to its chunk, so turning `chunking.headers` on recomputes every vector. The fix is two hash columns — one canonical, one embedded — and a lookup-schema bump, which rebuilds the sidecar from canonical artifacts rather than the corpus. Check the ordering too: `generation_is_reusable` validates the sidecar before anything ensures it, so a version bump denies reuse to the first ingest that follows it.

## Closing the paraphrase gap

Three of ten judged paraphrase queries miss the designated passage within the top ten while nothing is withheld: the passages are there and the ranking cannot find them. These are the levers.

- [ ] **Pooled relevance judgments.** Test. The set is known-item — one designated passage per query, one annotator — so a passage that makes the same point scores as a miss, and a change that ranks an equally good passage above the designated one reads as a regression. Collect every candidate from every mode and judge the pool. This is also what would let the pseudo-relevance expansion be measured: with rarity-weighted terms it mines the corpus's own vocabulary, and a known-item set cannot see that.
- [ ] **Grow the judged set from real questions**, if the privacy of a query log can be settled. Test.
- [ ] **Measure the headers on a corpus with sections.** Test. On a PDF corpus the header is the title alone and the first chunk of a paper already repeats it; an EPUB corpus is where the locator carries a section and where the header says something the passage does not.
- [ ] **Measure the embedding models the registry offers.** Test: `BAAI/bge-base-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1` for English, `jinaai/jina-embeddings-v2-base-de` for German, `intfloat/multilingual-e5-large` across languages. The German model is the first candidate for a corpus in that language, where the English default cannot help.
- [ ] **Merge the stopword lists of a mixed corpus.** Feature. `language.corpus` can name several languages while BM25 filters the one list in `language.bm25_stopwords`, and per-source `language` metadata reports which sources are which. bm25s accepts a list, so the union is expressible: check that the pinned runtime passes a list through `bm25.lang`, then measure the union against a single list.
- [ ] **An Arabic stopword source.** Feature. bm25s ships no Arabic list, so `language.corpus = "ar"` is refused while settings are read even though a source can declare Arabic. Add an explicit list option, or an empty list that filters nothing, plus an Arabic-capable model in the pinned table.
- [ ] **Keep the measurement current and wider.** Test. Re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and add a second corpus and filtered queries.

## Keeping the code changeable

- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `_advance_ingestion` is 1,341 lines at complexity 107 against 35 for the next worst function in the package; three of its phase blocks call closures defined inside it and eight read loop-local state, so the split is an ingestion state object that handlers take and return. Accept on a green suite and a re-ingest that reuses every chunk and vector. The map and the transformation rules are in git history.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run research-rag --project-root /path/to/project status
uv run research-rag --project-root /path/to/project doctor
uv run research-rag --project-root /path/to/project search "your question"
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
```

An item is done when the change is committed, its measurement is in `MEASUREMENTS.md` if it moved a number, and the commands above are clean.
