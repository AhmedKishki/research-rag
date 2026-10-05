---
name: MEASUREMENTS.md
description: The measurement protocol, the envelope this app is built for, and the mechanism behind each tunable.
---

# Measurements

The measurement protocols, workload envelope, and limits on interpreting retrieval experiments.

- A retrieval default changes only after a reviewed evaluation.
- Retained run records own the exact engine revision, settings, inputs, outputs, and timings.
- Lexical diagnostics do not establish semantic relevance, usability, independence, or contradiction.

Two harnesses produce everything these decisions rest on, each driving the gateway, the models, and the service:

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
```

## Workload and design envelope

Built for 25–250 English-primary born-digital PDF/EPUB sources, 5,000–50,000 chunks, on CPU only. Out of scope, and not claimed: non-English-primary corpora, multilingual retrieval, OCR or scanned material, handwriting, and formula-heavy corpora.

A corpus above that range is outside what this app is built for rather than something it was tested at. A cost stated for one is arithmetic extrapolated from the exact scan, not a measurement.

## Why the dense index is an exact scan

Three things decide this:

- An embedded index is dominated by per-point device cost, and every query reopens it.
- It returns neither chunk identity nor scores through the upstream API, which the research contract needs.
- The exact scan reads the portable float32 vectors the generation already stores, so it needs no separate index and ranks a query the way a brute-force scan of those vectors does.

Above `dense.exact_backend_chunk_limit` the embedded backend is chosen instead, and the choice is recorded in the manifest so a generation is never rebuilt with the other one.

## Why writes are grouped

Cost follows the number of durability operations rather than the size of a payload, so a unit's artifacts are written together and committed with one directory fsync before the checkpoint that claims the unit complete.

A file handed to a peer process and rewritten before every use may skip the write barrier, because no later read can observe an earlier one.

## Chunking in bounded batches

Several extraction units share one chunker call, and each unit still gets its own durable output and its own index. Chunk identities are identical either way, and a crash redoes at most one bounded batch rather than a whole phase.

## Embedding: the inference batch is a padding decision

FastEmbed pads every sequence to the longest member of its batch and the runtime still computes the padded positions, so a large batch spends its compute on padding rather than on text. One sequence per inference returns the same vectors a large batch returns, so `dense.embedding_inference_batch_size` ships at one and is a throughput setting only.

## The candidate gate is stored, not recomputed

A chunk's mechanical eligibility verdict is cached as a bitmask when the generation's artifact lookup is built. One shared function computes it for the build and query-time fallback. Reason codes are computed for flagged candidates. These rules are not author judgments of relevance or usability.

## What a reader waits for

The first search pays the cold cost of the engines this app runs: a query embedding model and a cross-encoder load from disk on first use. Nothing downloads, because the models are cached.

A gateway that cannot start is reported by the operation that needed it, with the tail of the logs the transport wrote and the path of each one.

## What is per query and what is not

The source-tree walk behind the staleness answer is the only per-query cost that grows with the number of source files, which is why the engine takes a staleness switch at all. Only a measurement run sets it, and no reader-facing operation can, because a search that cannot say whether its corpus moved is answering about something unknown.

Nothing else per query is cached, deliberately. Caching the document map adds cross-request state that must be invalidated correctly, and reusing one SQLite connection requires that connection to be reachable from whichever thread serves the next call. At a scale where either would matter, parsing the manifest dominates both, and the answer there is a persistent document-metadata index rather than a per-process cache.

The dependency report is the one term that reads outside this project, because it validates the managed UltraRAG runtime against the installed snapshot. It is cached per process and re-read only when the marker identifying that snapshot changes, which is what a reinstall or a repair rewrites.

## Retrieval quality

The judged set and its protocol are in `evaluation/`; `scripts/evaluate_retrieval.py` runs it through `ResearchService.search`, the engine the workspace and the command line both call. The harness measures every mode explicitly, so its rows do not depend on the app's default.

- Known-item success, reciprocal rank, and nDCG measure where one designated passage appears.
- A relevant alternative passage receives no relevance credit without a pooled judgment.
- Final-list duplication is measured after the engine's repetition collapse.
- Low final duplication does not establish that the reranker scored no copies or that collapse preserved all distinct claims.
- Repetition across quote and paraphrase queries for one target is expected; it is not evidence of a generic leader.
- Requested candidate caps and rerank caps are not observed depths or windows.
- A reranked row requires evidence that the reranker ran; an unavailable-model fallback is a degraded result, not a successful reranking experiment.
- Current shipped retrieval settings remain unchanged; these diagnostics do not establish an optimum.

## What a judged set cannot establish

- The judgments are known-item and single-annotator. A mode that returns a different passage making the same point is scored as a miss, and no pooled judgment exists, so no true recall figure is claimed.
- They come from one English-primary corpus and one generation, not from a benchmark suite.
- The shipped set has 32 queries over 19 targets.
- Excluding an unavailable target with an explicit reason leaves 30 queries over 18 targets.
- Multiple queries about one target are correlated; query counts, result-slot counts, and repeated runs are not independent sample sizes.
- Partitions of an inspected benchmark are exploratory even when target families are disjoint.
- One-query changes do not establish a policy improvement or statistical equivalence.
- Pooled graded relevance, usable-passage precision, counterevidence, boundary correctness, and no-answer behavior need author judgments.
- An untouched confirmation set and paired target-family uncertainty are required before changing defaults.

## Which product produced a figure

- Record the engine revision, source content hash, generation identity, model configuration, and observed budgets.
- A code-arm comparison deliberately changes the engine while keeping scoring definitions and judged inputs fixed.
- Do not compute deltas across incompatible scorer definitions or silently substituted rerank implementations.
- `doctor` and `update` identify running processes whose loaded code differs from the checkout.
- A running app is not restarted by an evaluation run.

## Which version an update compares against

- `update` reads the remote's release tags with `git ls-remote --tags`, so a check needs no token and no API key.
- `DIST_TAG` is the tag name that names the current release rather than a version of its own (`latest`), and a remote that publishes it wins over the highest version tag, because a maintainer moves it when a release is superseded.
- A tag carrying a pre-release or build-metadata suffix is not a release, and two tags claiming one version is a refusal rather than a coin toss.
- A checkout's position against that release is one of `at_release`, `behind_release`, `ahead_of_release`, `no_release`, and `unreadable_release`.
- Only `behind_release` is an update to apply: `ahead_of_release` is unreleased work, which is a fact to report and never a reason to change anything.

## The reranker model is an engine setting

The model is not a search option, so the harness can measure a second one over the same judged queries in one run:

```bash
uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

- **The shipped model stays.** A model decision requires comparable observed windows, applied-reranking checks, repeated timings, and adjudicated relevance.
- **The current benchmark is exploratory.** Model superiority is not established by this known-item set alone.
- **Changing the model is an operator decision, not a per-search one.** `--reranker-model` (or `RESEARCH_RAG_RERANKER_MODEL`) changes every search this app answers, and `search(rerank_model=...)` changes it for one engine call.
- **Which model is smaller is not a parameter to tune.** Both run through the same FastEmbed cross-encoder class with the runtime's default thread count, so the difference is an ONNX-export property.

## Reply depth is the depth the ranking reaches

The requested reply depth influences candidate retrieval and the rerank window. Keep reply depth fixed in a policy comparison, and record the observed scored window rather than assuming the cap binds.

- The requested branch depth is `min(active_chunk_count, maximum_candidates, max(minimum_candidates, top_k * 4))`.
- Pin both candidate settings to the desired depth to distinguish a branch-depth sweep from a cap-only sweep.
- Filtering can widen the lexical index window; record that window separately from the requested branch depth.
- Rerank caps bind only when the candidate pool and window multiple/floor permit them to bind.
- Record observed windows and candidate shortfalls for every query.

## What a ranking change costs the corpus

Query-time ranking changes do not rewrite generation artifacts. `generation_is_reusable` compares processing and artifact identity rather than query-time weights, gates, caps, or rerank windows. Record the generation's policy and the experiment's effective policy separately.

The staging checkpoint identity does not include the retrieval-policy fingerprint, so editing a ranking value resumes an interrupted build rather than discarding it, and a build in progress survives the edit. Only disposable staging is identified this way: a published generation records its policy in its manifest, and a test pins that a ranking change reuses every chunk and vector.

## The cosine gate

`retrieval.dense_minimum_cosine_similarity` withholds a dense candidate that scores below it, before the fusion sees it, so a search can legitimately return fewer results than `top_k` and an empty answer is a real answer rather than a failure.

- Gate counts establish activity, not whether rejected candidates were relevant or irrelevant.
- Count dense quality exclusions separately from score rejections.
- Eligible candidates equal admitted-above-floor candidates plus margin rescues plus score rejections.
- A BM25-only query does not run the dense gate; report its dense diagnostics as not applicable.
- `withheld_candidates` and dense score rejection are different counters.
- Target presence before and after admission identifies designated-target losses; it does not measure exhaustive candidate recall.
- An unchanged known-item score does not establish that a gate is harmless, necessary, or calibrated.
- Final admission and abstention calibration require relevant, hard-negative, and no-answer judgments.

## The dense floor's relative rescue

- The relative margin admits a below-floor candidate only when the query's best eligible candidate clears the floor and the candidate lies within the margin.
- This score rule alone does not establish valid abstention for no-answer questions.

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

- Smaller chunks do not join extraction units that are already short.
- A length floor rejects fragments; it does not repair paragraph boundaries.
- Legitimate short prose, table rows, captions, and bibliography need judgments before evaluating a floor or a re-chunking policy.

## The reranked window

- Only candidates inside the observed rerank window receive cross-encoder scores.
- The window is `min(candidates ranked, retrieval.rerank_max_candidates, max(top_k * retrieval.rerank_window_multiple, retrieval.rerank_window_floor))`.
- Raising a cap above a lower multiple/floor bound does not widen the window.
- Equal observed windows under different caps do not establish that wider reranking is ineffective.
- A quality plateau requires a correctly configured sweep and relevance judgments, not unchanged cap labels.

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

- More sources, fewer same-source selections, and lower repetition are structural changes, not relevance judgments.
- Preserving success at one depth does not establish that a penalty is free; reciprocal rank and nDCG may still change.
- A discrete sweep cannot establish an optimum or the largest safe penalty between tested values.
- Separate within-target-family repetition from repetition across distinct target families.
- Retain distinct authorship, conflicting claims, and provenance when assessing any source-diversity replacement.

## What a running project costs the machine

`runtime.nice` reaches the whole tree because children inherit it. Processes started before the setting existed keep their own value until they are restarted.

Two facts bound what a long build can do to a machine. An `ingest` call yields between atomic units and returns a checkpointed `in_progress` result when the configured `ingestion.work_budget_seconds` runs out, so cancelling between calls loses nothing. Builds are serialized by the project lock, so two ingests cannot compound.

## Current limits

- **Retrieval quality is judged, not settled.** Pooled judgments, a second annotator, and a second corpus are open work in `TODO.md`.
- **A cost above the design envelope is arithmetic.** The switch to the embedded backend comes from the exact scan's cost extrapolated, and that backend is unoptimised for it: one connection per phase and time-boxed upload batches are unbuilt.
- **Over-limit chunks are flagged, not split.** Splitting would change chunk identities and reuse behaviour without a long-input strategy, so the build counts and flags them.
- **Older generations are read as they are.** A schema-1 generation stays BM25-only until it is re-ingested, and a lookup written before the retrieval-verdict column is recomputed on first use rather than regenerated silently.
- **The embedding thread count is left to the runtime.** The optimum is the physical core count, which is machine-specific, and nothing is auto-detected.

## Local exploratory measurements

- The generation is `20261002T080853Z-f1ec4db7`; no new generation is built or activated.
- Its manifest records extraction policy 7, which the shipped policy 8 now reports as needing an upgrade, so none of these figures measures the extraction the app builds today.
- Inputs contain 30 known-item queries over 18 targets, partitioned into 14 and 16 queries without shared target families.
- Target `t12` is explicitly excluded because its source is absent from this generation.
- The inputs are inspected and exploratory; neither partition is a confirmation set.
- Primary depth is ten; the deep pass is disabled.
- Records retain scorer hashes and definitions, applied-reranking flags, frozen inputs, report hashes, and closed source/registry/engine guards.
- Report-schema-3 batches use revisions `38f5303` and `7be7709`; deltas are computed only within a batch with one scorer identity.

| Observation | Scope and interpretation |
|---|---|
| 120 paired baseline ranking/stage comparisons agree | Two independent project copies, four modes, 30 queries; timing and run paths are excluded from equality. |
| Branch depths 40/80/160 and rerank caps 20/30/50 produce distinct observed work | At branch depth 40, some cap-50 windows contain only 40 or 41 candidates; at depths 80 and 160 the scored windows reach 50. |
| Default hybrid-reranked success is 26/30; depth-80/cap-50 success is 27/30 | One extra designated passage appears on partition B; partition A stays 12/14. This is exploratory, not a demonstrated policy improvement. |
| The gate admits 837/1,200 eligible candidates at the shipped floor | 425 are above the floor, 412 are margin rescues, and 363 are rejected; the counts conserve the eligible pool. |
| Five of 30 queries lose their designated target from the dense branch at that floor | BM25 recovers them in the fused pool on these queries; this does not establish safe admission or abstention on other questions. |
| Default scoring spends 0/600 scores on candidates later collapsed | Six candidates are collapsed from the unscored tail; final-list duplicate counts alone do not expose this distinction. |
| Depth-80/cap-50 spends 3/1,500 scores on candidates later collapsed | Three queries account for those scores: one engine `same_words` collapse and two cosine-based `same_meaning` collapses. Those labels are mechanisms, not human judgments. |
| Increasing the source penalty from zero to the shipped value does not reduce cross-family repetition | Partition A stays 2/140 repeated slots; partition B rises from 2/160 to 4/160. Fewer repeated results within one question family do not establish fewer generic leaders. |

### Retained local evidence

- The experiment toolkit keeps generated artifacts outside tracked source under its `runs/` directory.

| Batch | Record directory | Query-mode observations |
|---|---|---:|
| Paired baseline | `pinned-baseline-repeat-20261002T225230.896936` | 240 |
| Corrected budget grid | `corrected-budget-grid-20261002T225521.887240` | 270 |
| Gate-stage comparison | `gate-stage-audit-20261002T230706.629280` | 90 |
| Source-charge comparison | `source-charge-audit-20261002T231034.507688` | 60 |
| Scored-collapse tracing | `scored-collapse-audit-20261002T232931.884177` | 60 |

- These 720 observations repeatedly evaluate the same 30 queries; they are not 720 independent questions.
- A refused real preflight also retains its error and an unchanged source guard, without measured arms.
- Process files, locks, logs, and gateway scratch state are declared guard exclusions for the serving app.
- Single-pass latency observations do not establish a stable p95 or a performance winner.
- The next quality test needs blinded author judgments of relevance, usability, independent evidence, contradictions, and no-answer cases.
