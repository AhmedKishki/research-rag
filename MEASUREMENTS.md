# Measurements

- A retrieval default changes only after a reviewed evaluation.
- Retained run records own the exact engine revision, settings, inputs, outputs, and timings.
- Lexical diagnostics do not establish semantic relevance, usability, independence, or contradiction.

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
```

## Workload envelope

- The app is built for 25–250 English-primary born-digital PDF/EPUB sources, 5,000–50,000 chunks, on CPU only.
- Not claimed: non-English-primary corpora, multilingual retrieval, OCR or scanned material, handwriting, and formula-heavy corpora.
- A corpus above that range was not tested. A cost stated for one is arithmetic extrapolated from the exact scan.

## Why the dense index is an exact scan

- An embedded index is dominated by per-point device cost, and every query reopens it.
- The upstream API returns neither chunk identity nor scores, which the research contract needs.
- The exact scan reads the portable float32 vectors the generation already stores. It needs no separate index and ranks like a brute-force scan of those vectors.
- Above `dense.exact_backend_chunk_limit` the embedded backend is chosen. The manifest records the choice, so a generation is never rebuilt with the other backend.

## Build mechanisms

- **Grouped writes.** Cost follows the number of durability operations, not payload size.
  - A unit's artifacts are written together and committed with one directory fsync before the checkpoint that marks the unit complete.
  - A file handed to a peer process and rewritten before every use may skip the write barrier, because no later read can observe an earlier one.
- **Bounded chunking batches.** Several extraction units share one chunker call, and each keeps its own durable output and index. Chunk identities are identical either way. A crash redoes at most one batch.
- **Embedding batch size.** FastEmbed pads every sequence to the longest in its batch, and the runtime computes the padded positions.
  - One sequence per inference returns the same vectors as a large batch.
  - `dense.embedding_inference_batch_size` ships at one and is a throughput setting only.
- **Stored candidate gate.** A chunk's mechanical eligibility verdict is cached as a bitmask when the artifact lookup is built. One shared function computes it for the build and the query-time fallback. Reason codes are computed for flagged candidates only. The rules are not judgments of relevance or usability.
- **Chunk size and overlap.** `chunking.size` is bounded by the embedding model's input limit, because a chunk must fit that limit with its contextual header. The sweep runs down from the shipped size, with overlap tested separately.
  - Smaller chunks do not join extraction units that are already short.
  - A length floor rejects fragments and does not repair paragraph boundaries.
  - Short prose, table rows, captions, and bibliography need judgments before a floor or re-chunking policy is evaluated.

## What a reader waits for

- The first search loads the query embedding model and the cross-encoder from disk. Nothing downloads, because the models are cached.
- A gateway that cannot start is reported by the operation that needed it, with the tail of the transport's logs and each log's path.
- The source-tree walk behind the staleness answer is the only per-query cost that grows with the number of source files.
  - The engine therefore takes a staleness switch. Only a measurement run sets it. A search that cannot say whether its corpus moved answers about something unknown.
- Nothing else per query is cached.
  - A cached document map adds cross-request state that needs correct invalidation.
  - A shared SQLite connection must be reachable from whichever thread serves the next call.
  - At a scale where either matters, manifest parsing dominates, and the answer is a persistent document-metadata index.
- The dependency report reads outside the project, because it validates the managed UltraRAG runtime against the installed snapshot. It is cached per process and re-read when the snapshot marker changes, which a reinstall or repair rewrites.
- `runtime.nice` reaches the whole process tree, because children inherit it. Processes started before the setting existed keep their own value until restarted.
- An `ingest` call yields between atomic units and returns a checkpointed `in_progress` result when `ingestion.work_budget_seconds` runs out. Cancelling between calls loses nothing. The project lock serializes builds.

## Retrieval quality

- The judged set and its protocol are in `evaluation/`.
- `scripts/evaluate_retrieval.py` runs the set through `ResearchService.search`, the engine the workspace and command line both call. The harness measures every mode explicitly, so rows do not depend on the app's default.
- Known-item success, reciprocal rank, and nDCG measure where one designated passage appears.
- A relevant alternative passage gets no credit without a pooled judgment.
- Final-list duplication is measured after the engine's repetition collapse.
  - Low duplication does not show that the reranker scored no copies or that collapse kept every distinct claim.
  - Repetition across quote and paraphrase queries for one target is expected and is not a generic leader.
- Requested candidate caps and rerank caps are not observed depths or windows.
- A reranked row needs evidence that the reranker ran. A fallback to the unranked order is a degraded result, not a reranking experiment.
- These diagnostics do not establish an optimum. Shipped settings stay unchanged.

### What the judged set cannot establish

- Judgments are known-item and single-annotator. A mode that returns a different passage making the same point scores as a miss. No true recall figure is claimed.
- They come from one English-primary corpus and one generation.
- The shipped set has 32 queries over 19 targets. Excluding an unavailable target with a stated reason leaves 30 queries over 18 targets.
- Queries about one target are correlated. Query counts, result slots, and repeated runs are not independent samples.
- Partitions of an inspected benchmark are exploratory even when target families are disjoint.
- A one-query change shows neither a policy improvement nor statistical equivalence.
- Pooled graded relevance, usable-passage precision, counterevidence, boundary correctness, and no-answer behaviour need author judgments.
- Changing a default needs an untouched confirmation set and paired target-family uncertainty.

### Which product produced a figure

- Record the engine revision, source content hash, generation identity, model configuration, and observed budgets.
- A code-arm comparison changes the engine and keeps scoring definitions and judged inputs fixed.
- Do not compute deltas across incompatible scorer definitions or silently substituted rerank implementations.
- `doctor` and `update` identify running processes whose loaded code differs from the checkout.
- An evaluation run does not restart a running app.

### The reranker model

- The model is an engine setting, not a search option, so one run can measure a second model over the same queries:

```bash
uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

- The shipped model stays. A model decision needs comparable observed windows, applied-reranking checks, repeated timings, and adjudicated relevance. This known-item set alone does not establish superiority.
- `--reranker-model` or `RESEARCH_RAG_RERANKER_MODEL` changes every search the app answers. `search(rerank_model=...)` changes one engine call.
- Both models run through the same FastEmbed cross-encoder class and thread count, so a speed difference is a property of the ONNX export.

### Reply depth and the rerank window

- The requested reply depth influences candidate retrieval and the rerank window. Keep it fixed in a policy comparison.
- Record the observed scored window, because a cap does not always bind.
- The requested branch depth is `min(active_chunk_count, maximum_candidates, max(minimum_candidates, top_k * 4))`.
  - Pin both candidate settings to the target depth to separate a branch-depth sweep from a cap-only sweep.
  - Filtering can widen the lexical index window. Record it separately.
- The rerank window is `min(candidates ranked, retrieval.rerank_max_candidates, max(top_k * retrieval.rerank_window_multiple, retrieval.rerank_window_floor))`.
  - Only candidates inside it get cross-encoder scores.
  - Raising a cap above a lower multiple or floor bound does not widen it.
  - Equal windows under different caps do not show that wider reranking is ineffective.
  - A quality plateau needs a correctly configured sweep and relevance judgments.
- Record observed windows and candidate shortfalls for every query.

### What a ranking change costs the corpus

- Query-time ranking changes do not rewrite generation artifacts. `generation_is_reusable` compares processing and artifact identity, not query-time weights, gates, caps, or windows.
- Record the generation's policy and the experiment's effective policy separately.
- The staging checkpoint identity excludes the retrieval-policy fingerprint, so editing a ranking value resumes an interrupted build. Only disposable staging works this way. A published generation records its policy in its manifest, and a test pins that a ranking change reuses every chunk and vector.

## Tunable mechanisms

### The cosine gate

- `retrieval.dense_minimum_cosine_similarity` withholds a dense candidate below it before fusion. A search can return fewer than `top_k` results, and an empty answer is a real answer.
- Gate counts show activity, not whether rejected candidates were relevant.
- Count dense quality exclusions separately from score rejections.
- Eligible candidates equal admitted-above-floor candidates plus margin rescues plus score rejections.
- A BM25-only query does not run the dense gate. Report its dense diagnostics as not applicable.
- `withheld_candidates` and dense score rejection are different counters.
- Target presence before and after admission identifies designated-target losses. It does not measure exhaustive candidate recall.
- An unchanged known-item score does not show that a gate is harmless, necessary, or calibrated.
- Final admission and abstention calibration need relevant, hard-negative, and no-answer judgments.

### The dense floor's relative rescue

- The margin admits a below-floor candidate only when the query's best eligible candidate clears the floor and the candidate lies within the margin.
- This score rule alone does not establish valid abstention for no-answer questions.
- The margin is a runtime setting, like the source-diversity penalty. It enters neither the retrieval-policy fingerprint nor the manifest, so a project can set, try, and drop it without a rebuild.
- Two full-detail disclosures accompany it:
  - `dense_gate` reports the floor, the margin, the query's best cosine similarity, and the counts admitted or rejected below the floor.
  - `rejected_candidate_examples` names up to `retrieval.maximum_withheld_examples` sources per reason.

### The minimum passage length

- Chunks never span extraction units, so one short unit becomes one short chunk. Back matter is made of short units, and an index line is a list of the corpus's own words, so a query can match a fragment with nothing to cite.
- `retrieval.minimum_passage_words` drops a candidate whose cleaned text is shorter than the setting, before fusion, in both halves.
- `retrieval.minimum_passage_token_fraction` states the floor as a fraction of the generation's recorded `chunking.size`, counted with the tokenizer that generation was chunked with. A chunk holds `chunking.size` tokens, so a tenth of that is a fragment whatever its word count. The count covers the returned text, which is cleaned after chunking.
- Both are runtime settings and apply at the next search.
- Neither is a back-matter filter. An index line is as short as a one-line paragraph, and only the query tells them apart. A corpus whose answers are table rows, catalogue entries, or bibliography lines would lose evidence a judged set cannot detect. Both ship off, and the value belongs in a project's own configuration.

### The lexical abstention gate

- The gate drops a lexical candidate that shares no query token outside the corpus language's function-word set. `how?`, `why not?`, and `what does it do?` abstain instead of matching a passage that contains the word they asked with.
- English uses bm25s's fuller list, which also stops question and do-support words. A multilingual corpus unions every named language's list. The fallback words stay stopped when bm25s is unavailable.
- The judged set has no contentless query, so it cannot measure this. `tests/test_service.py::test_contentless_queries_abstain` and the settings tests pin it.

### Pseudo-relevance feedback

- `retrieval.prf` searches the lexical half once, mines terms from the leading passages, searches again with the terms added, and ranks with the second result.
- Selection weights each candidate by how many leaders use it times how rare it is across the generation. The document-frequency table is built on the first query that needs one and kept while the generation stays loaded. A process with the feature off never reads the corpus for it.
- Ranking by leader support alone mines the words every passage shares. The rarity weight prevents that.
- A known-item set registers a change only when the exact designated passage moves, so it cannot see the difference. The default stays off until the judgments are widened.

### Contextual chunk headers

- `chunking.headers` prepends the source title and, where the locator carries one, the section to the text a chunk is embedded from. The passage is untouched. The header lives in a separate `embedding_text` field.
- A PDF locator carries a page, not a section, so a PDF chunk is headed by its title alone. The first chunk of a paper usually repeats the title in its first line. Later chunks gain the work's name.
- The default stays off. The setting is an identity setting: the next ingestion of a project that enables it rebuilds, and a re-ingest without it drops the headers silently. The choice belongs in the project's config, not on one command line.

### The source-diversity penalty

- `retrieval.source_diversity_penalty` reorders the final `top_k` pick.
  - A candidate's adjusted score is its normalized relevance in the fusion and reranker ranking: 1.0 at the top, 0.0 at the bottom and for the unreranked tail.
  - The score is charged once for each candidate already taken from the same source. The best adjusted score wins.
- It only reorders candidates already ranked. A ranking with no score to charge keeps its order.
- Its cost is measured in rank, not evidence. A stronger charge displaces more of a source's relevant passages, which a known-item set cannot score.
- More sources, fewer same-source selections, and lower repetition are structural changes, not relevance judgments.
- Preserving success at one depth does not show a penalty is free. Reciprocal rank and nDCG may still change.
- A discrete sweep cannot establish an optimum or the largest safe penalty between tested values.
- Separate within-target-family repetition from repetition across target families.
- Keep distinct authorship, conflicting claims, and provenance when assessing a source-diversity replacement.

## Current limits

- **Retrieval quality is judged, not settled.** Pooled judgments, a second annotator, and a second corpus are open in `TODO.md`.
- **A cost above the envelope is arithmetic.** The switch to the embedded backend comes from the exact scan's extrapolated cost. That backend is unoptimised for it: one connection per phase and time-boxed upload batches are unbuilt.
- **Over-limit chunks are flagged, not split.** Splitting would change chunk identities and reuse behaviour without a long-input strategy.
- **Older generations are read as they are.** A schema-1 generation stays BM25-only until re-ingested. A lookup written before the retrieval-verdict column is recomputed on first use.
- **The embedding thread count is left to the runtime.** The optimum is the physical core count, which is machine-specific and not auto-detected.

## Local exploratory measurements

- Generation `20261002T080853Z-f1ec4db7`. No new generation was built or activated.
- Its manifest records extraction policy 7. The shipped policy 8 reports it as needing an upgrade, so these figures do not measure today's extraction.
- Inputs: 30 known-item queries over 18 targets, split into partitions of 14 and 16 queries without shared target families.
  - Target `t12` is excluded because its source is absent from this generation.
  - The inputs are inspected and exploratory. Neither partition is a confirmation set.
- Primary depth is ten. The deep pass is disabled.
- Report-schema-3 batches used revisions `38f5303` and `7be7709`. Deltas are computed only within a batch with one scorer identity.
- The 720 query-mode observations repeat the same 30 queries. They are not 720 independent questions.
- Single-pass latency does not establish a stable p95 or a performance winner.

| Observation | Scope and interpretation |
|---|---|
| 120 paired baseline ranking and stage comparisons agree | Two independent project copies, four modes, 30 queries. Timing and run paths are excluded from equality. |
| Branch depths 40/80/160 and rerank caps 20/30/50 produce distinct observed work | At depth 40, some cap-50 windows hold only 40 or 41 candidates. At depths 80 and 160 the scored windows reach 50. |
| Default hybrid-reranked success is 26/30. Depth-80/cap-50 success is 27/30 | One extra designated passage appears on partition B. Partition A stays 12/14. This is not a demonstrated policy improvement. |
| The gate admits 837/1,200 eligible candidates at the shipped floor | 425 are above the floor, 412 are margin rescues, and 363 are rejected. The counts conserve the eligible pool. |
| Five of 30 queries lose the designated target from the dense branch at that floor | BM25 recovers them in the fused pool. This does not show safe admission or abstention on other questions. |
| Default scoring spends 0/600 scores on candidates later collapsed | Six candidates are collapsed from the unscored tail. Final-list duplicate counts alone do not show this. |
| Depth-80/cap-50 spends 3/1,500 scores on candidates later collapsed | Three queries account for them: one engine `same_words` collapse and two cosine-based `same_meaning` collapses. The labels are mechanisms, not human judgments. |
| Raising the source penalty from zero to the shipped value does not reduce cross-family repetition | Partition A stays at 2/140 repeated slots. Partition B rises from 2/160 to 4/160. Fewer repeats within one question family do not mean fewer generic leaders. |

- The next quality test needs blinded author judgments of relevance, usability, independent evidence, contradictions, and no-answer cases.

## Cosine-validity probes

`scripts/audit_cosine_validity.py` measures constructed semantic contrasts with the
pinned embedding model. Its optional project input supplies the existing judged
target texts. It embeds those texts afresh in memory and never mixes new query
vectors with stored generation vectors.

- Pair suppression calls the engine's collapse function.
- Admission is a score-only projection of the configured rule, not a production search.
- Constructed answer-selection expectations are not corpus relevance judgments.
- Other known-item targets remain unjudged competitors, not assumed negatives.
- The report records model identities, input token counts, scores, and scope limits.
- Engine fingerprints cover the executed collapse and equality-key modules.
- Canonical project file hashes must agree before and after the run.
- The diagnostic initializes no project, downloads no model, and records no searches.

The exploratory run used generation `20261007T061545Z-6bbae6e2` and the current
pinned BGE-small ONNX snapshot. Its older stored embedding revision prevents a
current dense production search. No generation was rebuilt or activated.

| Observation | Scope and interpretation |
|---|---|
| Target resolution scans 32,368 chunks and retains 36 records | The 18 targets use streamed snippet matches and source witnesses, not a full in-memory chunk list. |
| Reversed actors have passage cosine 0.994545 | “The managers monitor the workers” and “The workers monitor the managers” collapse as `same_meaning` at the configured threshold. |
| A negated claim in shared background has passage cosine 0.996666 | The engine suppresses one of two constructed passages that disagree about whether automation increased wages. |
| A changed quantity in shared background has passage cosine 0.994618 | The engine suppresses one of two constructed passages reporting 10 versus 90 percent. |
| The sign contrast collapses as `same_words` | Dropping `+` and `-` from the equality key loses the distinction. This is not a cosine-based collapse. |
| Cosine selects the constructed answer in eight of nine contrast-ranking cases | The sign case fails. These cases are inspected probes, not independent held-out judgments. |
| Cosine ranks the designated target first in 29 of 30 queries | The pool holds only 18 freshly embedded judged passages. This is not full-corpus recall. The same-pool cross-encoder comparison also ranks 29 targets first. |
| The admission projection rejects eight designated targets | The simulated best score comes from the bounded pool. A full-corpus candidate set can change relative rescue, so this is not a measured production loss rate. |
| 24 of 30 designated-target scores lie between 0.7 and 0.9 | Only 25 of all 540 pool comparisons lie in that band. Selecting top results changes the apparent distribution. The comparisons are not independent samples. |

None of the constructed or judged passage inputs exceeded the model's token limit.
These observations disprove semantic-equivalence guarantees for this cosine
threshold. They do not establish corpus-wide false-collapse incidence, safe
admission thresholds, or superiority of a replacement model or scoring method.
