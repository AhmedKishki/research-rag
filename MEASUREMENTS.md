# Measurements

The protocol a measurement is taken by, and the decisions measurements once justified.

This file states no figure. A retrieval default changes only on a fresh run of a harness below, and the run's own output is the record.

Two harnesses produce everything these decisions rest on, each driving the real gateway, the real models, and the real service:

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
```

## Workload and design envelope

Built for 25–250 English-primary born-digital PDF/EPUB sources, 5,000–50,000 chunks, on CPU only. Out of scope, and not claimed: non-English-primary corpora, multilingual retrieval, OCR or scanned material, handwriting, and formula-heavy corpora.

A corpus above that range is outside what this app is built for rather than something it was tested at. A cost stated for one is arithmetic extrapolated from the exact scan, not a measurement.

## Why the dense index is an exact scan

Three things decide this. An embedded index is dominated by per-point device cost, and every query reopens it. It returns neither chunk identity nor scores through the upstream API, which the research contract needs. The exact scan reads the portable float32 vectors the generation already stores, so it needs no separate index and ranks a query the way a brute-force scan of those vectors does.

Above `dense.exact_backend_chunk_limit` the embedded backend is chosen instead, and the choice is recorded in the manifest so a generation is never rebuilt with the other one.

## Why writes are grouped

Cost follows the number of durability operations rather than the size of a payload, so a unit's artifacts are written together and committed with one directory fsync before the checkpoint that claims the unit complete.

A file handed to a peer process and rewritten before every use may skip the write barrier, because no later read can observe an earlier one.

## Chunking in bounded batches

Several extraction units share one chunker call, and each unit still gets its own durable output and its own index. Chunk identities are identical either way, and a crash redoes at most one bounded batch rather than a whole phase.

## Embedding: the inference batch is a padding decision

FastEmbed pads every sequence to the longest member of its batch and the runtime still computes the padded positions, so a large batch spends its compute on padding rather than on text. One sequence per inference returns the same vectors a large batch returns, so `dense.embedding_inference_batch_size` ships at one and is a throughput setting only.

## The candidate gate is stored, not recomputed

A chunk's usability verdict is a property of the chunk, not of the query, so it is computed once when a generation's artifact lookup is built and stored as a bitmask per chunk. One shared function computes the bitmask for both the build and the query-time fallback, so a rejection decision and the counter that reports it cannot drift. Reason codes are recomputed only for a chunk the verdict flags as corrupt, because the response discloses them.

## What a reader waits for

The first search pays the cold cost of the engines this app runs: a query embedding model and a cross-encoder load from disk on first use. Nothing downloads, because the models are cached.

A gateway that cannot start is reported by the operation that needed it, with the tail of the logs the transport wrote and the path of each one.

## What is per query and what is not

The source-tree walk behind the staleness answer is the only per-query cost that grows with the number of source files, which is why the engine takes a staleness switch at all. Only a measurement run sets it: no reader-facing operation can skip it, because a search that cannot say whether its corpus moved is answering about something unknown.

Nothing else per query is cached, deliberately. Caching the document map adds cross-request state that must be invalidated correctly, and reusing one SQLite connection requires that connection to be reachable from whichever thread serves the next call. At a scale where either would matter, parsing the manifest dominates both, and the answer there is a persistent document-metadata index rather than a per-process cache.

The dependency report is the one term that reads outside this project, because it validates the managed UltraRAG runtime against the installed snapshot. It is cached per process and re-read only when the marker identifying that snapshot changes, which is what a reinstall or a repair rewrites.

## Retrieval quality

The judged set and its protocol are in `evaluation/`; `scripts/evaluate_retrieval.py` runs it through `ResearchService.search`, the engine the workspace and the command line both call. The harness measures every mode explicitly, so its rows do not depend on the app's default.

Four findings hold as decisions:

- **Reranking is the largest single quality gain**, so it is on by default. It is also the only configuration that returns the judged passage first for every quote query.
- **Hybrid orders better than BM25 alone at the same reach**, so choosing BM25 for its speed gives up ordering quality rather than coverage.
- **Dense alone is the weakest mode**, and returns fewer passages because the cosine gate rejects most of its candidates. Its worst class is entity queries: a proper noun needs its words matched, which is what BM25 is for.
- **Paraphrase is the hardest class for every mode.** The gap is part depth and part ordering, not the relevance gates.
- **No returned passage duplicated another**, at any gate or diversity setting measured: `mean_exact_duplicate_slots` and `mean_near_duplicate_slots` were `0.0` throughout. The corpus holds reprinted material in its chunks; a result list does not.
- **No passage dominated the results.** `repeated_slot_rate` was 0.119 on the held-out half, meaning one passage in eight occupied a slot it also held for another query, and the most frequent passage appeared in three of sixteen queries.

## What a judged set cannot establish

- The judgments are known-item and single-annotator. A mode that returns a different passage making the same point is scored as a miss, and no pooled judgment exists, so no true recall figure is claimed.
- They come from one English-primary corpus and one generation, not from a benchmark suite.
- Fusion weights are deliberately untouched: a judged set this size cannot separate a real weight effect from noise.
- Thirty queries over nineteen passages register a ranking change only when the designated passage crosses the depth. On a sweep of the source-diversity penalty, which reordered every query's results and cut same-source pairs from 86 slots to 12, the quality columns moved by one query in fourteen. A metric that moves at that scale cannot score a change that improves evidence coverage without moving the designated passage, so a result-list change is read against the redundancy counts beside it rather than against `succ@k`.
- No held-out split is recorded here. A development and a held-out half split by target, so that a quote and its own paraphrase cannot appear on both sides, is produced by a script outside this repository and belongs with the judged set rather than in a measured figure.

## Which product produced a figure

A number is comparable only with another measured by the same engine, because a process answering from a different checkout reads a different reviewed-metadata field set and can refuse the project's own state. `doctor` reports the two directories when they differ, `research-rag update` names the processes still running the pre-update code, and the reviewed-metadata file records the fields its writer understood, so an older process refuses the file by cause instead of naming a field the caller never typed.

A figure is comparable with another measured by the same engine, because both call the same unchanged retrieval code. A figure measured by a different checkout is not comparable until `doctor` reports the two trees as the same, and `update` names the processes still running the pre-update code while it applies.

## Which version an update compares against

`update` reads the remote's release tags with `git ls-remote --tags`, so a check needs no token and no API key. `DIST_TAG` is the tag name that names the current release rather than a version of its own (`latest`); a remote that publishes it wins over the highest version tag, because a maintainer moves it when a release is superseded. A tag carrying a pre-release or build-metadata suffix is not a release, and two tags claiming one version is a refusal rather than a coin toss. A checkout's position against that release is one of `at_release`, `behind_release`, `ahead_of_release`, `no_release`, and `unreadable_release`, and only `behind_release` is an update to apply: `ahead_of_release` is unreleased work, which is a fact to report and never a reason to change anything.

## The reranker model is an engine setting

The model is not a search option, so the harness can measure a second one over the same judged queries in one run:

```bash
uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

- **The shipped default stays.** The comparison reached the same depth as the alternative and ordered quote queries slightly better, at a lower cost per query, so it is neither the better nor the cheaper model here.
- **A set this size cannot rank two models.** Treat a comparison this narrow as directional, and re-measure before trusting either on a different corpus.
- **Changing the model is an operator decision, not a per-search one.** `--reranker-model` (or `RESEARCH_RAG_RERANKER_MODEL`) changes every search this app answers, and `search(rerank_model=...)` changes it for one engine call.
- **Which model is smaller is not a parameter to tune.** Both run through the same FastEmbed cross-encoder class with the runtime's default thread count, so the difference is an ONNX-export property.

## Reply depth is the depth the ranking reaches

The reranker reorders a window of `max(top_k * retrieval.rerank_window_multiple, retrieval.rerank_window_floor)` passages, capped by `retrieval.rerank_max_candidates` and the fused candidate count. The depth a reader asks for is therefore the depth the ranking reaches, which makes `top_k` the cheapest quality lever this app has.

`rerank_max_candidates` and the window floor are not levers at a normal corpus size: the window formula never reaches them. They stay settings because a larger corpus could reach them.

`maximum_candidates` bounds the widening a filter triggers, and that is where it acts. A filter drops passages after the corpus is ranked, so a narrow one cannot fill `top_k` and the lexical window doubles until it can; without a bound that doubling walks the whole corpus, ranking it at every step. The response reports the window it reached, so a filter that matched nothing inside it cannot read as an empty corpus. The bound costs reach: a document whose passages all rank below the ceiling is not found by a filter that names it.

## What a ranking change costs the corpus

Nothing, for the artifacts. `generation_is_reusable` compares the schema, extraction, cleaning and artifact policy versions, the project id, the chunk size and overlap, the chunk-headers switch, and the embedding model — and nothing else — so weights, gates, caps and the reranked window cannot affect what a corpus holds. A retrieval experiment that reaches a new baseline costs a re-ingest and no re-extraction.

The staging checkpoint identity does not include the retrieval-policy fingerprint, so editing a ranking value resumes an interrupted build rather than discarding it, and a build in progress survives the edit. Only disposable staging is identified this way: a published generation records its policy in its manifest, and a test pins that a ranking change reuses every chunk and vector.

## The cosine gate

`retrieval.dense_minimum_cosine_similarity` withholds a dense candidate that scores below it, before the fusion sees it, so a search can legitimately return fewer results than `top_k` and an empty answer is a real answer rather than a failure.

The gate is calibrated for a fused ranking. A dense-only one wants a much lower value, because the gate is what thins its candidate list; the agent-facing search tool is hybrid-only, and a dense-only method exposed later would have to not apply it.

Measured across `retrieval.dense_minimum_cosine_similarity` 0.0, 0.60, 0.66, the shipped 0.72, 0.78, and 0.85 at a fixed candidate window and rerank budget, success at each depth, MRR, and nDCG were identical from 0.0 through 0.72 on both halves of the judged set, and document success fell one query in fourteen at 0.78 and again at 0.85. The gate reordered results well before it cost anything: at 0.72 five of sixteen held-out rankings differed from the ungated run, and at 0.85 fifteen of sixteen did. The judged set cannot say whether those reorderings are better, so the floor is neither justified nor refuted by this measurement and the shipped value is unchanged on grounds of a fused ranking rather than of a measured effect.

## The dense floor's relative rescue

`retrieval.dense_relative_similarity_margin` answers the case the floor alone gets wrong: a query whose whole candidate list sits in a band just under it. A floor decides on score alone, so a one-word query — whose best passage can score below the floor while its neighbours sit a few hundredths behind — loses most of its semantic candidates before fusion sees them. The margin admits a below-floor candidate when the query's best candidate cleared the floor and this one is within the margin of it, and admits nothing when nothing cleared the floor, which keeps abstention intact.

The margin is a runtime setting, like the source-diversity penalty: it enters neither the retrieval-policy fingerprint nor the generation manifest, so generations built before it keep validating and a project can set it, try it, and drop it without rebuilding.

Two disclosures accompany it, both full-detail only. `dense_gate` reports the floor, the margin, the query's best cosine similarity, and how many candidates were admitted or rejected below the floor. `rejected_candidate_examples` names up to `retrieval.maximum_withheld_examples` sources per reason, so a thin answer reads as thinned rather than silent.

## The minimum passage length

Chunks never span extraction units, so one short unit becomes one short chunk. A book's back matter is made of short units, and an index line is a list of the corpus's own words, so a query can match a fragment that carries nothing to cite.

`retrieval.minimum_passage_words` drops a candidate whose cleaned text is shorter than the setting, before fusion, in both halves. It ships off, because a corpus of deliberately short passages is a legitimate corpus.

`retrieval.minimum_passage_token_fraction` states the same floor as a fraction of the generation's recorded `chunking.size`, counted with the tokenizer that generation was chunked by, because "short" is a unit question: a chunk is built to hold `chunking.size` tokens, so a candidate holding a tenth of that is a fragment whatever its word count. The count is taken over the returned text, which is cleaned after chunking. Like the margin, the setting is runtime and applies at the next search with no rebuild.

Neither floor is a back-matter filter, because a short unit is short whatever it holds: an index line is exactly as short as a one-line paragraph, and only the query can tell them apart. A corpus whose answers are table rows, catalogue entries, or bibliography lines would lose evidence a judged set could never detect, which is why the packaged default is off and the value belongs in a project's own configuration.

## The lexical abstention gate

The gate drops a lexical candidate that shares no query token outside the corpus language's function-word set, so `how?`, `why not?`, and `what does it do?` abstain instead of matching whatever passage contains the word they asked with. English takes bm25s's fuller list, which also stops question and do-support words. A corpus in several languages unions every named language's list, and the fallback words stay stopped even when bm25s is unavailable.

The judged set holds no contentless query, so it cannot measure this behaviour; `tests/test_service.py::test_contentless_queries_abstain` and the settings tests pin it.

## Chunk size and overlap

`chunking.size` is bounded by the embedding model's input limit, because a chunk has to fit inside that with its contextual header, so a chunk at the model's own limit is not expressible and the sweep runs down from the shipped size with the overlap tested separately.

**A fragment problem is a length-floor problem, not a chunk-size problem.** A smaller chunk splits long units into more chunks, almost all of them above the floor, so the fragment *share* falls while the fragments themselves stay. Neither the chunk size nor the overlap touches a unit that was already short. That is why `retrieval.minimum_passage_token_fraction` is stated in tokens rather than as a hint to re-chunk.

## The reranked window

The window is shallow enough that a target the fusion ranks outside it is never seen by the cross-encoder, so a shallower window does not merely lose precision: it loses that band entirely. The shipped window is the shallowest one that reaches the quality plateau, and its cost per query grows linearly with the candidates it holds.

The window is `min(candidates ranked, retrieval.rerank_max_candidates, max(top_k * retrieval.rerank_window_multiple, retrieval.rerank_window_floor))`, so `rerank_max_candidates` binds only below `top_k * multiple`. At the shipped `top_k` of 10 and multiple of 2 that floor is 20, and budgets of 20, 30, and 50 produced identical rerank windows and byte-identical rankings across 96 of 96 queries. A budget sweep is therefore a measurement of the window multiple or of `top_k` until the budget is set below that floor, and a run reporting an unchanged window across different budgets measured one configuration.

## Pseudo-relevance feedback

`retrieval.prf` searches the lexical half once, mines terms from the leading passages, searches again with the terms added, and ranks with the second result. Selection weights each candidate by how many leaders use it times how rare it is across the generation, from a document-frequency table built on the first query that needs one and kept while that generation stays loaded, so the cost stays off the query path and a process with the feature off never reads the corpus for it.

The rarity weight is the point: ranking by leader support alone mines the words every passage shares instead. The judged set cannot see the difference, because a known-item set registers a change only when the exact designated passage moves. Widening the judgments is what would measure this feature, and the default stays off until then.

## Contextual chunk headers

`chunking.headers` prepends the source title and, where a locator carries one, the section to the text a chunk is *embedded* from. The passage itself is untouched, so returned text and citations stay as they were, and the header lives in a separate `embedding_text` on the chunk.

A PDF locator carries a page rather than a section, so a PDF chunk is headed by its title alone, and the first chunk of a paper usually repeats that title in its own first line. For the rest of a paper the header adds the work's name where the passage had none, which is the case the feature exists for.

The default stays off, and it ships off for a second reason beyond the measurement: it is an identity setting, so the next ingestion of a project that turns it on rebuilds, and a re-ingest without it would silently drop the headers. That is why the choice belongs in the project's config rather than on one command line.

## The source-diversity penalty

`retrieval.source_diversity_penalty` reorders the final `top_k` pick. A candidate's adjusted score is its normalized relevance in the ranking that fusion and the reranker produced — 1.0 at the top, 0.0 at the bottom and for the unreranked tail — charged once for every candidate already taken from the same source, and the best adjusted score wins.

It only reorders candidates that were already ranked, so it adds and removes nothing, and a ranking with no score to charge against keeps its own order. Its cost is measured in rank, not in evidence: a stronger charge displaces more of a source's own relevant passages, which a known-item judged set cannot score, because it registers that the designated passage moved and never that a run of adjacent passages became less useful.

Measured across `source_diversity_penalty` 0.0, 0.25, 0.5, and 1.0 at a fixed candidate window and rerank budget, on a development and a held-out half of the judged set split by target: mean distinct sources per query rose from 3.9 to 9.1 on the development half and 4.6 to 9.5 on the held-out half, and held-out success at depth was unchanged at every value. On the development half success at depth held through 0.25 and fell one query in fourteen at 0.5. The shipped 0.25 is therefore the largest charge the judged set scores as free, not a midpoint.

That reading rests on a set of thirty queries and the free claim rests on one query of movement in the other direction, so it bounds the setting rather than establishing an optimum. The redundancy measures are what a change in this setting should be read against: at every value tested, `mean_exact_duplicate_slots` and `mean_near_duplicate_slots` were `0.0`, so the charge was reordering sources and never suppressing reprinted text.

## What a running project costs the machine

`runtime.nice` reaches the whole tree because children inherit it. Processes started before the setting existed keep their own value until they are restarted.

Two facts bound what a long build can do to a machine. An `ingest` call yields between atomic units and returns a checkpointed `in_progress` result when the configured `ingestion.work_budget_seconds` runs out, so cancelling between calls loses nothing. Builds are serialized by the project lock, so two ingests cannot compound.

## Current limits

- **Retrieval quality is judged, not settled.** Pooled judgments, a second annotator, and a second corpus are open work in `TODO.md`.
- **A cost above the design envelope is arithmetic.** The switch to the embedded backend comes from the exact scan's cost extrapolated, and that backend is unoptimised for it: one connection per phase and time-boxed upload batches are unbuilt.
- **Over-limit chunks are flagged, not split.** Splitting would change chunk identities and reuse behaviour without a long-input strategy, so the build counts and flags them.
- **Older generations are read as they are.** A schema-1 generation stays BM25-only until it is re-ingested, and a lookup written before the retrieval-verdict column is recomputed on first use rather than regenerated silently.
- **The embedding thread count is left to the runtime.** The optimum is the physical core count, which is machine-specific, and nothing is auto-detected.
- **A changed default needs a fresh run.** A corpus, an extraction policy, or a retrieval default changes on the harness's own output and nothing else.